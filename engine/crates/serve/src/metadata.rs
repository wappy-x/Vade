//! Serving receipts. A cold checkpoint hashes in the background, once per load.
use crate::protocol::{quote, DraftMode};
use engine_model::flash::{manifest::{Receipt, ReceiptPlan}, WeightFormat};
use std::{ffi::CStr, path::Path, sync::{Arc, Mutex, OnceLock}, time::{Instant, SystemTime, UNIX_EPOCH}};

#[path = "../../../tools/export/src/sha256.rs"]
mod sha256;

#[derive(Debug)]
struct State {
    pending: String,
    mtp_field: String,
    stale: OnceLock<String>,
    complete: OnceLock<String>,
    work: Mutex<Option<Work>>,
}
#[derive(Debug)]
struct Work { plan: ReceiptPlan, fixed: String, directory: String, started: Instant }
#[derive(Clone, Debug)]
pub struct Metadata(Arc<State>);
impl Metadata {
    /// PTX excludes its C terminator, matching sha256sum of the loaded file.
    /// Files must remain immutable for the loaded model's lifetime.
    pub fn load(root: &Path, ptx: &CStr, mtp_depth: usize, world_size: usize,
        topology: &str, server_start: SystemTime) -> Result<Self, String> {
        let metadata = Self::prepare(root, ptx, mtp_depth, world_size, topology, server_start)?;
        metadata.start();
        Ok(metadata)
    }
    /// Capture file identities before loading without competing for payload I/O.
    /// Call start once target and draft weights have finished loading.
    pub fn prepare(root: &Path, ptx: &CStr, mtp_depth: usize, world_size: usize,
        topology: &str, server_start: SystemTime) -> Result<Self, String> {
        let started = Instant::now();
        let directory = root.file_name().and_then(|s| s.to_str()).ok_or("checkpoint needs a UTF-8 directory name")?;
        let plan = ReceiptPlan::open(root).map_err(|e| e.to_string())?;
        let experts = std::env::var_os("FLASH_BF16_EXPERTS");
        let sampled = std::env::var_os("FLASH_MTP_SAMPLED");
        let nvfp4 = plan.format == WeightFormat::Nvfp4;
        // VADE_W2_TP=1 (flash-serve pins it at world 2 FP4-Local unless VADE_W2_PATH=pipeline, and
        // refuses it elsewhere): the tensor split's recipe and levels, so its receipts differ from
        // the pipeline's. The engine checks each level's value at startup.
        let tensor = pipeline_path(topology, |name| std::env::var(name).ok());
        let sha = env!("VADE_BUILD_GIT_SHA");
        #[cfg(target_os = "linux")]
        let layer_sha = quote(&sha256::digest(spark_gpu_probe::layer_ptx()?.to_bytes()));
        #[cfg(not(target_os = "linux"))]
        let layer_sha = "null";
        let fixed = format!("\"build\":{{\"git_sha\":{},\"dirty\":{},\"source\":{}}},\"ptx_sha256\":\"{}\",\"layer_ptx_sha256\":{layer_sha},\"numerics\":{{\"flash_dense_arithmetic\":\"bf16_hi_lo_tensor_core\",\"flash_bf16_experts_set\":{},\"flash_bf16_experts\":{},\"flash_mtp_sampled_set\":{},\"flash_mtp_sampled\":{},\"nvfp4\":{nvfp4},\"mtp_depth\":{mtp_depth},\"world_size\":{world_size},\"topology\":{}{tensor}}},\"server_start_unix_ms\":{}",
            if sha == "unknown" { "null".into() } else { quote(sha) }, env!("VADE_BUILD_DIRTY"), quote(env!("VADE_BUILD_SOURCE")),
            sha256::digest(ptx.to_bytes()),
            experts.is_some(), experts.as_deref().is_some_and(|v| v == "1"),
            sampled.is_some(), sampled.as_deref().is_some_and(|v| v == "1"), quote(topology),
            server_start.duration_since(UNIX_EPOCH).map_err(|_| "server clock precedes Unix epoch")?.as_millis());
        let directory = quote(directory);
        let pending = render(&fixed, &directory, None,
            if plan.is_stale() { "failed-stale" } else { "pending" }, "background");
        let metadata = Self(Arc::new(State { pending, mtp_field: format!("\"mtp_depth\":{mtp_depth},"), stale: OnceLock::new(),
            complete: OnceLock::new(), work: Mutex::new(None) }));
        if let Some(receipt) = plan.cached().map_err(|e| e.to_string())? {
            log_receipt(&receipt, started, "manifest");
            let _ = metadata.0.complete.set(render(&fixed, &directory, Some(&receipt), "ready", "manifest"));
        } else {
            *metadata.0.work.lock().unwrap() = Some(Work { plan, fixed, directory, started });
        }
        eprintln!("server_metadata vade={} elapsed_ms={:.3}", metadata.json(), started.elapsed().as_secs_f64() * 1000.0);
        Ok(metadata)
    }
    /// Idempotent across metadata clones; thread startup failure only fails the
    /// receipt, so serving can still start with a truthful null checkpoint pin.
    pub fn start(&self) {
        let Some(work) = self.0.work.lock().unwrap_or_else(|e| e.into_inner()).take() else { return; };
        let Work { plan, fixed, directory, started } = work;
        let failed = render(&fixed, &directory, None, "failed", "background");
        let state = self.0.clone();
        let worker = std::thread::Builder::new().name("checkpoint-sha256".into()).spawn(move || {
            let result = std::panic::catch_unwind(|| plan.finish_with(|| {
                eprintln!("checkpoint_hash state=failed-stale rehash=true");
                let _ = state.stale.set(render(&fixed, &directory, None, "failed-stale", "background"));
            })).unwrap_or_else(|_| Err(engine_model::Error("checkpoint hash worker panicked".into())));
            let json = match result {
                Ok(receipt) => {
                    let source = if receipt.reused_files == 0 { "background" } else { "manifest+background" };
                    log_receipt(&receipt, started, source);
                    render(&fixed, &directory, Some(&receipt), "ready", source)
                }
                Err(error) => {
                    eprintln!("checkpoint_hash state=failed error={error}");
                    render(&fixed, &directory, None, "failed", "background")
                }
            };
            let _ = state.complete.set(json);
        });
        if worker.is_err() {
            eprintln!("checkpoint_hash state=failed error=worker_start_failed");
            let _ = self.0.complete.set(failed);
        }
    }
    /// Clones held by health and request echoes see the same completed receipt.
    pub fn json(&self) -> &str { self.0.complete.get().or_else(|| self.0.stale.get()).unwrap_or(&self.0.pending) }
    /// The loader may prepare receipts before the scheduler resolves its defaults.
    /// Overlay only our own encoded field; checkpoint hash updates remain shared.
    pub(crate) fn json_with_drafting(&self, draft: DraftMode, mtp_depth: usize) -> String {
        let mtp_depth = if draft == DraftMode::Plain { 0 } else { mtp_depth };
        self.json().replacen(&self.0.mtp_field,
            &format!("\"draft\":\"{}\",\"mtp_depth\":{mtp_depth},", draft.name()), 1)
    }
    pub(crate) fn fields(&self, draft: DraftMode, mtp_depth: usize) -> String {
        let json = self.json_with_drafting(draft, mtp_depth); json[1..json.len() - 1].to_owned()
    }
}

/// VADE_W2_TP_PROJ's level for the receipt: unset and "0" are 0; the engine refuses any other
/// value than 1, 2 or 3 at startup.
fn projection_level(value: Option<&str>) -> u8 {
    match value { Some("1") => 1, Some("2") => 2, Some("3") => 3, _ => 0 }
}
/// The receipt's path at a pipeline topology, read after flash-serve's pins: `"path"` names it
/// (the tensor split or the layer pipeline), and the tensor split adds its recipe with each level
/// (projection level, the exchange wire, the verify head kept whole or split by vocabulary halves).
fn pipeline_path(topology: &str, read: impl Fn(&str) -> Option<String>) -> String {
    if topology != "pipeline" { return String::new(); }
    if read("VADE_W2_TP").as_deref() != Some("1") { return ",\"path\":\"layer_pipeline\"".into(); }
    format!(",\"path\":\"tensor_split\",\"tensor_split\":{{\"experts\":\"output_columns\",\"long_prompts\":\"pipeline_tiles_then_state_handoff\",\"projection_level\":{},\"wire\":\"{}\",\"head\":\"{}\"}}",
        projection_level(read("VADE_W2_TP_PROJ").as_deref()),
        if read("VADE_W2_TP_WIRE").as_deref() == Some("write") { "write" } else { "proxy" },
        if read("VADE_W2_TP_HEAD").as_deref() == Some("1") { "vocabulary_halves" } else { "whole" })
}
fn render(fixed: &str, directory: &str, receipt: Option<&Receipt>, state: &str, source: &str) -> String {
    format!("{{{fixed},\"checkpoint\":{{\"directory\":{directory},\"sha256\":{},\"hash_state\":\"{state}\",\"hash_source\":\"{source}\"}}}}",
        receipt.map_or_else(|| "null".into(), |r| quote(&r.sha256)))
}
fn log_receipt(receipt: &Receipt, started: Instant, source: &str) {
    eprintln!("checkpoint_hash state=ready source={source} sha256={} elapsed_ms={:.3} checkpoint_bytes={} hashed_bytes={} reused_files={} hash_buffer_bytes_per_worker_max={} max_workers=8",
        receipt.sha256, started.elapsed().as_secs_f64() * 1000.0, receipt.bytes, receipt.hashed_bytes, receipt.reused_files, engine_model::flash::manifest::HASH_BUFFER_BYTES);
}

#[cfg(test)]
mod tests {
    /// Each projection level gives its own tensor split recipe in the receipt.
    #[test]
    fn receipt_projection_level_covers_every_value() {
        assert_eq!([None, Some("0"), Some("1"), Some("2"), Some("3")].map(projection_level), [0, 0, 1, 2, 3]);
    }
    /// The receipt names the pipeline topology's path and each split level; other topologies add nothing.
    #[test]
    fn receipt_names_the_path_and_the_split_levels() {
        let env = |pairs: &'static [(&'static str, &'static str)]| move |name: &str| pairs.iter().find(|(n, _)| *n == name).map(|(_, v)| v.to_string());
        let all = &[("VADE_W2_TP", "1"), ("VADE_W2_TP_WIRE", "write"), ("VADE_W2_TP_PROJ", "3"), ("VADE_W2_TP_HEAD", "1")];
        assert_eq!(pipeline_path("pipeline", env(all)), ",\"path\":\"tensor_split\",\"tensor_split\":{\"experts\":\"output_columns\",\"long_prompts\":\"pipeline_tiles_then_state_handoff\",\"projection_level\":3,\"wire\":\"write\",\"head\":\"vocabulary_halves\"}");
        assert!(pipeline_path("pipeline", env(&[("VADE_W2_TP", "1")])).ends_with("\"projection_level\":0,\"wire\":\"proxy\",\"head\":\"whole\"}"));
        assert_eq!(pipeline_path("pipeline", env(&[])), ",\"path\":\"layer_pipeline\"");
        assert_eq!(pipeline_path("pipeline", env(&[("VADE_W2_TP", "0"), ("VADE_W2_TP_PROJ", "3")])), ",\"path\":\"layer_pipeline\"");
        assert_eq!(pipeline_path("local", env(all)), "");
    }
    use super::*;
    use std::io::Write;
    #[test]
    fn goldens_pin_and_official_sized_index() {
        let mut hashes = std::collections::BTreeMap::new();
        for (i, hash) in "774f0ceeadb40d165f2b3ff397d5f3840e6ca8fcb8f3d39d8acb4fea9e52c941 6841fe21fa8a8a7a693c585efe65cd2732889095b696da88bda0cb287366910b 974a2a2ab551f8f1405a4955ab32a8721c68c73dd85b382491d9f0e6a34ee752 591f488ab5cd5f0bd4fe28266099523761b9b9339137800734c34ffd84595538 c50bf465a4a0129f1a1196c0e75d881048238f7107b356fc734007eac0d3b123 57f103367fe36c9dc355c648d521d026caabb4ce5ba51038c78bd935f6593736 cde2a12854770c706a74a5c7b85ffc140e3d4dba93dd4bee1cd7aa07cc348c90 98ae323b5a61abdcf58a0dc9db7eb1a5a748cebc1ef1932804a1411500c09d84 70f4b66bca5c4c0b441cf4284d4ab311670e182bfc5ec51c7294a4054baeb20f 6d5dfa1dfa9a6c482f100f6e75c1f5e8ac02391c62e844e53f4a53996baf2206 de085adb563e530e7ad157220cd1a219be189a1db0fcc3f2507886a2d78bd4e1 97dac7c366ba19858d78c01364054136449ccb3cf60768587461cbf0bb582c90 467be10b473d25e0581f8fb1bc986916c8b9301272f72e7a05762484ffc080a9 38def6e9bbe6f242c2f92858c3e9a2b356d74d0180e323677158016dbee32ae7 90c6c665f1f819bc6350714652c8fb77064fec57832ef51b878a0cdf9f602236 88185c2530641b37dcad80d37b505c5257035e87ad29fa86640aae4814903e4f 42e3b96359116d7471a74cb7de08f6a4e9e173830d562f1f74ff09d1d253ad38 040adc7270efca6498c6f6a07dd7da14925d20217fe6de67223d6856fb4f2ad7 b8a831992deb1a694439e37b7136ff2e181457a8683bff65f1f4b99772cabe58 9426112fbd49ef4f5e9dd38dc2f93b046602e6d92cff732d33b96413ff1b40dc fce8426665f749ef38c4fad33b612f892b728c0f74fe60cab0e43ece785b4e77 a20dd415f280774321538e6b9e78fba9f8e218a9090eae52ea7172b85b9c5a6d 4100591839471a8721d9cedb251554dcef5ebc1dfd96ea7dce8912dd1e9921b2 bbd6d7a4b54b6a6032e15b7c2fc9d6526377f89992d61da8837ad059f3210b45 f9fa7b04de155dd2be74b5693d5b36e5463dabf956b9be36f67d4aae6b8dd304 193c6b9da2c6b69c34d8a14b65234871db272761e712c2e56bac3f13c2d0cbf7 d7efa372552c07ee8783e27c4b8ebd12fc75333f92553861bec8c7c0155a03fe 5fc13d084d62c06b928dfb388a1b36f5c27cd25bd3cbb68a0b21e1c4cfc68e6e a11f5a6360b349159575871db5c553c40c37be68382b82551a2c2372f15f2811 cac780a675f87fc699435a5207cbecb30a887271b5514b60b7e45d9bebe0e1ec d390a872eb3928b9d7c0a720ced792cc111613812a163d9fd6c97fd35d633897 3b36fe4e7215a12cdc6c7499aafde51cfbce51266785184b99d3838a6b37b2c8 3fb0504f406c0c71ee40e9e158492973c5c35337889c81b974892b407620c64c 6d989a812a8d8d9330c8c4d21d7be98a0114705ca3bd7834d43bb268bad6be08 8e56ba0a714198bd07a2d8fe9d91fbf5c72601c1b98edf09b289a2538445cb75 c23fd6f6cfa9acfa961ba26e7c9c0dd7b05782c5bbf8f55e2526119b5c516fd4 b824a11bf460bd78c068e3d301e914a0bec2f243a6eea4679be7200d179248e7 716eda81a763ffdefd1ac5306b4c80eef123c4dee24ed3a565a22438497581cb b91e3bfbcd765ce704db85b78f69055da59456ae733df635e3973ff2f966a148 3311ed7821b68bddb1542f06ca349feec72e68288113fe4899b08a3b8156343f 1d79949b61730a0fa1f6cd1af149fa50fedc051566cdea48283de8c7a32d972c 3f61684338e203cb2ccff6e451eda5705e10043d386b7ed55d4732d23eb5c77b bab92cfefbb811ee63c4e32595024117c7df3537f5359606eb1de23fe4f115b2 1d05f138848edca4138c3d1c6be4a432a0817eb29fa65f12d8611ecc7685c493 4610384e28f7a91b8bf2a1527178c82cbe7d50ecd9a33b61ee878fa8e8bced85 5ee9b0e45810300cbd4fd6f2acfe85080e7032a9ce5d77038b86f357b6700c23 efa6c0cc0e527486dc5955b00fa75665151d99997b722d8e56bd32da575e414c 57b148a266c55a896f171b5c98b350092bf394c1fb836658c58c589a0a317ccf 801dd3b475b43413ade810f4e8d3ace69d6ef5ee4bac5cb98d15e369343a5569 b4d00aca34ba7a9f8427983fd0829408fb5e15b2bb7268199e4010bf0c460a04 f430125e9ac92c2de15bbb74bcd3c91bc5374eef389a1be1bc4a592f3a082f0c 06c3d64136b097919b7e5a1715a70a60b8c2302fc4e80a33235607bb99d00f35 147d38f786512f1a8f570a00100b46a3faab1bd649b3445ca0a50c00aa2e99a3 31c08218039a246c333bfe0b068b8027fd92dddd542e581997af546eb104dd4d 971cd69c8d955bef6d557ebab3f170807a599517f7c69e9d8a6ba375214137f4 a6a975ac346aa3bd1176cb816d8bbd31a4d7542aa3de9306ca5899bea44bd159 d6acae1622abfa1e0253ae548bc1bdd4bc38cd9b096082acd455b6be592f4a54 f02b6bd8f5cbd742e8bac3ba9537aa14a64c5a8f3fc75495227a14e1a5cec4ec f3e2c94b8157a021b27e1efdb659333a8d1db330da5cd72ab9012fa0e5351506 908cce7f5fe8dc34f4cca1fc8a676569ab6ce08bad536bd0091c4d75e7b65a05 718532c88f4b7a4f402a50e4f1b6efc3cf71484c9e5a083ee30e96d5094a1bb3 3c21b895a5a4f0dc03b669c730f8d8d9ce9a29cbc7628a25894a16ffda6f0a7a 91d215e2135b2f5db5bc5a9541e5f6bf761205d67c6e7b8955ac73c815198f58 03a11cad2c283fb9752c6ac01c8f1d8b3127dcda963c5a8225b78f567239d872 0f2f96331120779ce1ebd3f957882092d3c733aa9a6073ca87fc86c6eb4e9465 5dbe501e4e0cb9e3f37cc43da46673b4157fdffba09c065ca0b12d75092a79f2 3ece325d2ca53eddf7b1782d5bbc4c376bc75c6d3da67813e110fd0af67cd09b 0c5ef706afd6b3058af5bf8fdc1e0938cb6e29d0a526c771893c9dcb8e702463 84999b24e3c55d2eae3ca187219a2e1a1872436ae9bafc91cc3e301180f36cfd eac7940317ee401859caf38089b2b924274ab81a3e5fbd87c42d89923361aad8 5ac60e474ccdab21257245630ed8240d75bc7115b3adb27ca65e39eb9ea8f20e b50a05c4c426fb00635d513a733a290da1007a89b8785f1ec02aad8e4d93d955 9e0594d389debdfb26d22259725831ff34280edeef4144e683d70205050919d9 1b1a0294f8df2a431885783185cdb185106228c9567e67224fe2c3c6a2467f5a 7dba189804ef7a8f9d926b27e1ddbe73767f1213cb7bb3cc582ab15fffec1792 d95ebacb144b4220f2df60cce85a2c86e87cb8291b836d1bd04f9902fcbb9297 26046977983e025defdf0f84cf5fc5f4ddfeabbd27e7abc43ff2b413265ab062 6fd46a95777392265b2afa66bf71d488856dafb7b76ff0337a0fe4ac857e961c 77398002634ce852917c59cd642bec7dd3903381239da68912bf69743631e54e f44aad95d36c443995b965aeef844c8e3a30ed172647dc57fcc0c5476c18d024 b295a48088573654d1b4754e861ec329956504cb6687e9e2a8eb652b35eaf988 83e0e171116c22eb8fdddfa3c387bebb9017eb0586410fd704ce70e7b3d23d68 7e1021cde06c970e4b1d9eb3d02f66f29786c59a0da8a3f8879de1a4e3fbf6c0 8938fd3c7f88d61fd5b5306eb0ab253d3e42e7f64baf3c3e494d315c8ad5ab55 cd5875a29d28d52578ae699fdd8891f605a0953cd64516bd88dc45dab50c383a 885e570a662aaf266fd9af9562c65116998df5710b2a27d1bf6aa0c8708b3b74 1472604826f0a0c2bbf7cd0f4a0c5b29f3012e4b3a4424d024d1ec17a3c9d0ea e7273f5435571e761429330ece81732e8fb3f8b5f0087bf80db693630cf702a1 eb349562d23e330bbbd0a5f5ba90ac25ed0baa660dff230538a0fcaaf4c8b875 532915eebd4e231a240e185ba838a95a00ee14277e28fb04ad21508da5ff50aa 68d8f46ffcd0ff46f4e73c848525cc6fd6af15eebbb9311ce9b4eb14638f5660 91abc087cd244ae62b8469576e209c8e25bc7726bc4fc020b687378059f5b5bd dacd0f2e5215fbf18fb7c6803f61963333e6243f404de83fcd5dfd38f43fe0d1 cac515a778e89fb37e9f38fde6c8fdfa01ebf9d8944a2d6ff35c51ecf4bc875f a35975c2d21f18d58f21cb59e4eaf34a4bce11440e5112948e7fa6923658f8c4 03fd77f9d35024bba9e894c811300ef8aaf5b5ff99ebc2c89c9d9f87685adfe8 f7b81ae09d97b8e2bb96a1916ede51da92209ec8481a31c56fb446acd691b8a8 70f208f47d99c25d5413cd32e0553cce22524cd3863c7797e2cfa37c2d526319 de5c33dd717495657496067ea715442356fd9f7b64384d1a4c10c9f8fe6f9b7d 43752e81a5c2b376073d5d0d7b70da8d08c49bdbbfcfeb695ea947a2913e4bc8 c2e978f7238036063914fe909214246730640be77eca215e3cf66d65724de9f1 639efed3ea400a8dddeae25db120096898a514697dfc4cbbc0cd15f621c18421 f8da1b61296e012dc268b4f0d84c127ef0967df488fe65b037f74c8f893decd1 adff0f2aee7b7b701e39247cfdfa135c0cab8dfa4f8c7662352c0233adb78829 6643d71382f00f7ffe16b05879e6ff0f6e56986dee8ec079a84aae82912c6987 680495ea0a23e73844e91670437b7b2a5ac5d46f9582923fb7fa91fe640821df 886528d121b77433c1d66654294c13b4f61a004f1e6d1abf2d445fb0a99f7234 a966c28a6267d43052fe87cb19e2e5730bad6154b12eb806b9ce49e007bfec96 f6074252e492ee5b0933f78cc48f3b90694a8096befb7666cc379c3d47093756 e3e45e96bc13b52b748a0e2f397a624ad49958396468667d73ed4d725c8b5c1c d1d354e691eab0b77607faf7eb4adf00a75ee7f270c1cbcbad06aefc19aeac94 e61a9a6ef0e538afca09d97a3ccbf3e7e2c647f65cc8eeebebb18621d7779b02 a14b94db5326bfaab1d8029296b8938f6daf91ba2dc07a4950a38afafa84f776 ce4aeac52776038ec0c885915bb270dc65ee07d70d5f71827d5fba1045e2b43b 8185d8822428d2125876a9aa05aea11c8165713bc99e04ace29921eef0e1b683 65954dccfa3be20611728311d97679e7f5d78ad519e52312524aa7cf666a48e5 12c8027363b1a7bb782282293314fa57b07be27d17670861858c0aed1e28a3ac 2d06ec9c1726f42bfc9ce0bbb47129917d8ab373c88eed4e758fb6940c92ad4a 36008b48c4480085bfd1a81439d70d1029cfaf06cfdd037cec19b491a40659ec 49e4f90d92f60f6489bfe6d3e5250d8fe879c5995ae72ce67379cc7187fa4b0a ae8e07cceeef2f568df035a7bc0c555656acb566ba5936f61a8a019df23361ad b3b21d33b925e31d475a509757cf7459aa4d3e73cb493706f40461387541e137 f32fec30685afadd76f09bbb1dda680445ad765a97ff1febfc3122b7f73d7ec3 e857f36ddd0db135f858c2c87c9fcf39c758965c626b280e47cb0c1e6c1423f8 b9de3774218d3cd7946cdbf913dedfc82591c18212d9c3fc9b9484e7dc68e9fe 29b39319522a00b1ef4ac909c4fb306059999f0a48fd07ca73cea687647631ee bf48ccb3b4ba75c288d9db2b94817c6a3993e9f84a5b9410d9f24edcec12e177 7feaa51c906a1a735e5fb52fd5499b59c473d442d77b44cbd610c9b6a3a9e0ba a5374c8d86aea48f21a4be2ad6cda37fbfbb01eff333d1179939a2e55d8db816 204b122d02425f61115db9ab896648d95f6dadd6a716c65daf6aec297392ed56 ba6fc69ebb6a9da1938c2e041b8ef625f4550b641812da75845bc7dd7c361d80".split_whitespace().enumerate() { hashes.insert(format!("model-{:05}-of-00131.safetensors", i + 1), hash.into()); }
        hashes.insert("model.safetensors.index.json".into(), "0419e2c2dfbb925257d7409405433a793cf7ff7d96f3eba882a815ec6d9fe7a6".into());
        assert_eq!(engine_model::flash::manifest::checkpoint_sha256(&hashes), "8f69659fb459c65ab2729022c87080c3eca51643db40288f9854632c4f0d0f03");
        let root = std::env::temp_dir().join(format!("vade-metadata-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&root).unwrap();
        let mut index = String::from("{\"metadata\":{},\"weight_map\":{");
        for i in 0..152_089 { if i != 0 { index.push(','); } index.push_str(&format!("\"tensor-{i:06}\":\"model.safetensors\"")); }
        index.push_str("}}"); index.extend(std::iter::repeat_n(' ', 17_500_000 - index.len()));
        std::fs::write(root.join("model.safetensors.index.json"), index).unwrap(); std::fs::write(root.join("model.safetensors"), b"a").unwrap();
        let started = Instant::now();
        let metadata = Metadata::load(&root, c"ptx", 0, 1, "local", SystemTime::now()).unwrap(); let shared = metadata.clone();
        assert!(metadata.0.pending.contains("\"sha256\":null,\"hash_state\":\"pending\""));
        while metadata.0.complete.get().is_none() && started.elapsed().as_secs() < 30 { std::thread::yield_now(); }
        assert!(shared.json().contains("\"hash_state\":\"ready\"")); assert!(root.join("manifest.sha256.sizes").is_file());
        if let Some(receipt) = ReceiptPlan::open(&root).unwrap().cached().unwrap() { assert_eq!(receipt.hashed_bytes, 0); eprintln!("synthetic_cache hashed_bytes=0 reused_files={} checkpoint_bytes={}", receipt.reused_files, receipt.bytes); }
        std::fs::write(root.join("model.safetensors"), b"b").unwrap(); let mut stale = false; assert!(ReceiptPlan::open(&root).unwrap().finish_with(|| stale = true).is_ok()); assert!(stale);
        for case in 0..3 {
            match case { 0 => std::fs::write(root.join("model.safetensors"), b"bb").unwrap(), 1 => std::fs::OpenOptions::new().append(true).open(root.join("model.safetensors.index.json")).unwrap().write_all(b" ").unwrap(), _ => std::fs::remove_file(root.join("manifest.sha256")).unwrap() }
            let refreshed = Metadata::load(&root, c"ptx", 0, 1, "local", SystemTime::now()).unwrap(); let start = Instant::now(); assert!(refreshed.0.pending.contains("\"sha256\":null,\"hash_state\":\"failed-stale\""));
            while refreshed.0.complete.get().is_none() && start.elapsed().as_secs() < 30 { std::thread::yield_now(); }
            assert!(refreshed.json().contains("\"hash_state\":\"ready\""));
            let index = std::fs::read(root.join("model.safetensors.index.json")).unwrap(); let shard = std::fs::read(root.join("model.safetensors")).unwrap(); let manifest = format!("{}  model.safetensors.index.json\n{}  model.safetensors\n", sha256::digest(&index), sha256::digest(&shard));
            assert_eq!(std::fs::read_to_string(root.join("manifest.sha256")).unwrap(), manifest); assert_eq!(std::fs::read_to_string(root.join("manifest.sha256.sizes")).unwrap(), format!("{}\n{}  model.safetensors\n{}  model.safetensors.index.json\n", sha256::digest(manifest.as_bytes()), shard.len(), index.len()));
        }
        eprintln!("synthetic_metadata tensors=152089 index_bytes=17500000 shard_bytes=1 elapsed_ms={:.3} pin=8f69659fb459c65ab2729022c87080c3eca51643db40288f9854632c4f0d0f03", started.elapsed().as_secs_f64() * 1000.0);
        std::fs::remove_dir_all(root).unwrap();
    }
}
