//! Host unit and end-to-end HTTP cross-restart tests for the persistent disk prefix cache.
use super::*;
use crate::{
    bounded_decoder::Host,
    loop_decoder::{
        Driver,
        draft::DraftChat,
        fixture::Fixture,
        rows::Limits,
        service::{Seats, Service, limits},
    },
};
use engine_model::mtp_draft::MtpConfig;
use std::{
    net::{SocketAddr, TcpListener, TcpStream},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

struct TempDir(PathBuf);
impl TempDir {
    fn new(tag: &str) -> Self {
        let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "vade-disk-cache-test-{tag}-{}-{seq}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn path(&self) -> &Path { &self.0 }
}
impl Drop for TempDir {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

const TEST_FP: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const TEST_FP_B: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
const CONTEXT: usize = 48;
const TEST_LIMITS: Limits = Limits { step_rows: 4, key_bytes: 1 << 20 };

fn direct_decode(fixture: &Fixture, context: usize, prompt: &[u32], count: usize) -> Vec<u32> {
    let driver = Driver::new(&Host, fixture, &fixture.description, context as u64).unwrap();
    let mut state = driver.clean().unwrap();
    let mut ids = Vec::new();
    let mut top = None;
    for &token in prompt {
        top = Some(driver.step(&mut state, token).unwrap().top);
    }
    while ids.len() < count {
        let id = top.unwrap();
        ids.push(id);
        top = Some(driver.step(&mut state, id).unwrap().top);
    }
    ids
}

fn serve_once(
    fixture: &Fixture,
    context: usize,
    disk_dir: &Path,
    budget: Option<u64>,
    fp: &str,
    requests: usize,
    client: impl FnOnce(&(dyn Fn(String) -> String + Sync), SocketAddr),
) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    struct Stop<'a>(&'a AtomicBool);
    impl Drop for Stop<'_> {
        fn drop(&mut self) { self.0.store(true, Ordering::Relaxed); }
    }
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let stop = &stop;
        let _stop = Stop(stop);
        let dir = disk_dir.to_path_buf();
        let fp = fp.to_owned();
        scope.spawn(move || {
            let seats = Seats { limits: TEST_LIMITS, chat: 1 << 20, room: 64 << 20, most: 2 };
            let service = Service::new(&Host, fixture, &fixture.description, context, vec![1], seats).unwrap();
            let layouts = crate::loop_decoder::state::layouts_for(&fixture.description, context as u64, TEST_LIMITS).unwrap();
            let disk = DiskPrefixCache::open(
                dir,
                budget,
                fp,
                &layouts,
                fixture.description.head_rows as usize,
                context,
                None,
            )
            .unwrap();
            let service = service.with_disk_cache(disk);
            let lim = limits(context, 24).unwrap();
            let (cache, config) = service.schedule();
            spark_serve::scheduler::serve_requests(
                listener, &service, None::<&spark_serve::tokenizer::ModelTokenizer>, &lim, &cache, &config, stop, "fixture", requests,
            )
            .unwrap();
        });
        let send = |text: String| {
            let mut stream = TcpStream::connect(address).unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
            stream.write_all(text.as_bytes()).unwrap();
            let mut reply = String::new();
            let _ = stream.read_to_string(&mut reply);
            reply
        };
        client(&send, address);
    });
}

fn ask_ids(send: &dyn Fn(String) -> String, address: SocketAddr, prompt: &[u32], max_tokens: usize, extra: &str) -> String {
    let ids = prompt.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    let body = format!(
        r#"{{"prompt_token_ids":[{ids}],"stream":true,"max_tokens":{max_tokens},"temperature":0,"ignore_eos":true,"stream_options":{{"include_usage":true}}{extra}}}"#
    );
    send(format!(
        "POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    ))
}

fn parse_token_ids(reply: &str) -> Vec<u32> {
    reply
        .split("\"token_id\":")
        .skip(1)
        .map(|part| part.chars().take_while(char::is_ascii_digit).collect::<String>().parse::<u32>().unwrap())
        .collect()
}

fn parse_cached_tokens(reply: &str) -> usize {
    reply
        .split("\"cached_tokens\":")
        .nth(1)
        .map(|part| part.chars().take_while(char::is_ascii_digit).collect::<String>().parse::<usize>().unwrap())
        .unwrap_or(0)
}

fn vdpc_files(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "vdpc"))
        .collect();
    out.sort();
    out
}

#[test]
fn cross_restart_exact_replay_continuation_and_cross_context_match_direct_decode() {
    let fixture = Fixture::small();
    let dir = TempDir::new("cross-restart");
    let base: Vec<u32> = vec![3, 5, 2, 7, 11, 13, 4, 6, 8, 10, 12, 14, 15, 9, 17, 18];
    let mut cont = base.clone();
    cont.extend_from_slice(&[19, 2, 4, 6]);
    let want_base = direct_decode(&fixture, CONTEXT, &base, 8);
    let want_cont = direct_decode(&fixture, CONTEXT, &cont, 8);

    // Server #1: cold prefill -> saves snapshot to disk, then stops completely.
    serve_once(&fixture, CONTEXT, dir.path(), Some(16 << 20), TEST_FP, 1, |send, addr| {
        let reply = ask_ids(send, addr, &base, 8, "");
        assert!(reply.starts_with("HTTP/1.1 200 "), "{reply}");
        assert_eq!(parse_cached_tokens(&reply), 0);
        assert_eq!(parse_token_ids(&reply), want_base);
    });
    assert_eq!(vdpc_files(dir.path()).len(), 1, "snapshot saved to disk");

    // Server #2 (fresh process state, empty RAM): exact replay hits disk with 0 diffs, then RAM hits on repeat!
    serve_once(&fixture, CONTEXT, dir.path(), Some(16 << 20), TEST_FP, 2, |send, addr| {
        let reply1 = ask_ids(send, addr, &base, 8, "");
        assert!(reply1.starts_with("HTTP/1.1 200 "), "{reply1}");
        assert_eq!(parse_cached_tokens(&reply1), base.len(), "disk hit on exact replay: {reply1}");
        assert_eq!(parse_token_ids(&reply1), want_base);

        let reply2 = ask_ids(send, addr, &base, 8, "");
        assert!(reply2.starts_with("HTTP/1.1 200 "), "{reply2}");
        assert_eq!(parse_cached_tokens(&reply2), base.len(), "promoted RAM hit: {reply2}");
        assert_eq!(parse_token_ids(&reply2), want_base);
    });

    // Server #3 (fresh process state, DIFFERENT context 64 > 48): continuation hits disk with 0 diffs!
    serve_once(&fixture, 64, dir.path(), Some(16 << 20), TEST_FP, 1, |send, addr| {
        let reply = ask_ids(send, addr, &cont, 8, "");
        assert!(reply.starts_with("HTTP/1.1 200 "), "{reply}");
        assert_eq!(parse_cached_tokens(&reply), base.len(), "disk hit on continuation across context: {reply}");
        assert_eq!(parse_token_ids(&reply), want_cont);
    });
    assert_eq!(vdpc_files(dir.path()).len(), 2, "continuation snapshot also persisted");

    // Server #4: checkpoint fingerprint mismatch cleanly misses and still serves 0-diff cold prefill.
    serve_once(&fixture, CONTEXT, dir.path(), Some(16 << 20), TEST_FP_B, 1, |send, addr| {
        let reply = ask_ids(send, addr, &base, 8, "");
        assert!(reply.starts_with("HTTP/1.1 200 "), "{reply}");
        assert_eq!(parse_cached_tokens(&reply), 0, "foreign checkpoint must miss: {reply}");
        assert_eq!(parse_token_ids(&reply), want_base);
    });
}

#[test]
fn tamper_and_truncation_at_every_region_cleanly_miss_and_remove_corrupt_file() {
    let fixture = Fixture::small();
    let prompt: Vec<u32> = vec![3, 5, 2, 7, 11, 13, 4, 6, 8, 10, 12, 14];
    let want = direct_decode(&fixture, CONTEXT, &prompt, 8);

    // Prepare a valid reference snapshot once.
    let seed_dir = TempDir::new("tamper-seed");
    serve_once(&fixture, CONTEXT, seed_dir.path(), Some(16 << 20), TEST_FP, 1, |send, addr| {
        let reply = ask_ids(send, addr, &prompt, 8, "");
        assert_eq!(parse_token_ids(&reply), want);
    });
    let orig_path = vdpc_files(seed_dir.path()).pop().unwrap();
    let orig_bytes = fs::read(&orig_path).unwrap();
    let orig_name = orig_path.file_name().unwrap().to_owned();

    // Test tampering: prelude (byte 0), header (byte 80), middle of payload (50%), footer CRC, footer magic, and truncation.
    let cases: [(&str, Box<dyn Fn(&mut Vec<u8>)>); 6] = [
        ("prelude_flip", Box::new(|b| b[0] ^= 0x01)),
        ("header_flip", Box::new(|b| b[80] ^= 0x80)),
        ("payload_mid_flip", Box::new(|b| { let i = b.len() / 2; b[i] ^= 0x04; })),
        ("footer_crc_flip", Box::new(|b| { let i = b.len() - 42; b[i] ^= 0x20; })),
        ("footer_magic_flip", Box::new(|b| { let i = b.len() - 1; b[i] ^= 0xFF; })),
        ("truncated", Box::new(|b| { let n = b.len() - 17; b.truncate(n); })),
    ];

    for (label, mutate) in cases {
        let dir = TempDir::new(label);
        let mut mutated = orig_bytes.clone();
        mutate(&mut mutated);
        let bad_path = dir.path().join(&orig_name);
        fs::write(&bad_path, &mutated).unwrap();

        // Direct unit check on DiskPrefixCache::restore (no silent accept, corrupt file removed).
        let layouts = crate::loop_decoder::state::layouts_for(&fixture.description, CONTEXT as u64, TEST_LIMITS).unwrap();
        let mut disk = DiskPrefixCache::open(
            dir.path().to_path_buf(),
            Some(16 << 20),
            TEST_FP.to_owned(),
            &layouts,
            fixture.description.head_rows as usize,
            CONTEXT,
            None,
        )
        .unwrap();
        let restored = disk.restore(&Host, &layouts, None, None, &prompt, CONTEXT).unwrap();
        assert!(restored.is_none(), "{label}: corrupt file must never restore");
        assert!(!bad_path.exists(), "{label}: corrupt file must be removed");

        // End-to-end HTTP server check: cleanly falls back to cold prefill with 0 diffs!
        fs::write(&bad_path, &mutated).unwrap();
        serve_once(&fixture, CONTEXT, dir.path(), Some(16 << 20), TEST_FP, 1, |send, addr| {
            let reply = ask_ids(send, addr, &prompt, 8, "");
            assert!(reply.starts_with("HTTP/1.1 200 "), "{label}: {reply}");
            assert_eq!(parse_cached_tokens(&reply), 0, "{label}: must fall back to cold prefill");
            assert_eq!(parse_token_ids(&reply), want, "{label}: 0 token-ID diffs required");
        });
    }
}

#[test]
fn draft_chat_state_roundtrips_bit_exactly_across_restarts() {
    let fixture = Fixture::small();
    let layouts = crate::loop_decoder::state::layouts_for(&fixture.description, CONTEXT as u64, TEST_LIMITS).unwrap();
    let driver = Driver::with_limits(&Host, &fixture, &fixture.description, CONTEXT as u64, TEST_LIMITS).unwrap();
    let mut state = driver.clean().unwrap();
    let prompt: Vec<u32> = (0..18).map(|i| ((i * 5 + 3) % 20) as u32).collect();
    let mut logits = Vec::new();
    for &t in &prompt {
        let step = driver.step(&mut state, t).unwrap();
        logits = step.logits;
    }

    // Build a synthetic DraftChat with 1 local sliding layer (window 8) + 1 global paged layer.
    let cfg = MtpConfig {
        hidden_size: 16,
        intermediate_size: 16,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 4,
        sliding_window_size: 8,
        block_size: 4,
        markov_rank: 4,
        vocab_size: 20,
        unpadded_vocab_size: 20,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        final_logit_softcapping: 30.0,
        local_layer_ids: vec![0],
        target_layer_ids: vec![0, 1],
    };
    let kv_w = cfg.kv_width();
    // Fill 18 rows of local + global KV and feature so local window slides past local_base.
    let ks: Vec<f32> = (0..prompt.len() * kv_w).map(|i| 0.25 + (i as f32) * 0.01).collect();
    let vs: Vec<f32> = (0..prompt.len() * kv_w).map(|i| -0.5 - (i as f32) * 0.01).collect();
    let feat: Vec<f32> = (0..cfg.hidden_size).map(|i| (i as f32) * 0.125 - 1.0).collect();
    // Populate draft via alloc_for_restore to build a clean committed DraftChat at len=18.
    let mut draft = DraftChat::alloc_for_restore(&Host, &cfg, false, prompt.len(), CONTEXT, CONTEXT).unwrap();
    {
        let (kv_slots, f_slot) = draft.restore_targets_mut();
        let win = cfg.sliding_window_size - 1; // 7 rows
        let local_start = prompt.len() - win; // 11
        let (lk, lv) = kv_slots[0].as_mut().unwrap();
        lk[..win * kv_w].copy_from_slice(&ks[local_start * kv_w..]);
        lv[..win * kv_w].copy_from_slice(&vs[local_start * kv_w..]);
        let (gk, gv) = kv_slots[1].as_mut().unwrap();
        gk[..prompt.len() * kv_w].copy_from_slice(&ks);
        gv[..prompt.len() * kv_w].copy_from_slice(&vs);
        f_slot.unwrap().copy_from_slice(&feat);
    }

    let dir = TempDir::new("draft-roundtrip");
    {
        let mut cache1 = DiskPrefixCache::open(
            dir.path().to_path_buf(),
            Some(16 << 20),
            TEST_FP.to_owned(),
            &layouts,
            fixture.description.head_rows as usize,
            CONTEXT,
            Some(&cfg),
        )
        .unwrap();
        cache1
            .save(&Host, &layouts, &state, Some((&draft, &cfg)), Some("salt-a"), &prompt, &logits)
            .unwrap();
    }

    // Reopen fresh cache instance (simulated restart): verify scope isolation and bit-exact restore!
    let mut cache2 = DiskPrefixCache::open(
        dir.path().to_path_buf(),
        Some(16 << 20),
        TEST_FP.to_owned(),
        &layouts,
        fixture.description.head_rows as usize,
        CONTEXT,
        Some(&cfg),
    )
    .unwrap();
    assert!(cache2.restore(&Host, &layouts, Some(&cfg), None, &prompt, CONTEXT).unwrap().is_none(), "scope isolation");
    assert!(cache2.restore(&Host, &layouts, Some(&cfg), Some("salt-b"), &prompt, CONTEXT).unwrap().is_none(), "scope isolation");

    let restored = cache2
        .restore(&Host, &layouts, Some(&cfg), Some("salt-a"), &prompt, CONTEXT)
        .unwrap()
        .expect("matching scope hits disk");
    assert_eq!(restored.tokens, prompt);
    assert_eq!(
        restored.next_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    // Bit-exact target state check across every visit plane & conv ring.
    assert_eq!(restored.state.position, state.position);
    for ((a, b), l) in restored.state.visits.iter().zip(&state.visits).zip(&layouts) {
        let words = state::prefix_plane_words(l, state.position);
        assert_eq!(
            a.keys[..words].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.keys[..words].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            a.values[..words].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.values[..words].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        for r in 0..4 {
            assert_eq!(
                a.rings[r].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.rings[r].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    // Bit-exact DraftChat check across local window, global plane, and recurrent feature.
    let mut r_draft = restored.draft.expect("draft restored");
    assert_eq!(r_draft.len(), draft.len());
    let (r_kv, r_feat) = r_draft.restore_targets_mut();
    let win = cfg.sliding_window_size - 1;
    let local_start = prompt.len() - win;
    let (rlk, rlv) = r_kv[0].as_ref().unwrap();
    assert_eq!(&rlk[..win * kv_w], &ks[local_start * kv_w..]);
    assert_eq!(&rlv[..win * kv_w], &vs[local_start * kv_w..]);
    let (rgk, rgv) = r_kv[1].as_ref().unwrap();
    assert_eq!(&rgk[..prompt.len() * kv_w], &ks[..]);
    assert_eq!(&rgv[..prompt.len() * kv_w], &vs[..]);
    assert_eq!(r_feat.unwrap().as_slice(), feat.as_slice());

    // Also verify a `--mtp 0` server (`draft_config = None`) can still restore the target state cleanly!
    let mut cache_plain = DiskPrefixCache::open(
        dir.path().to_path_buf(),
        Some(16 << 20),
        TEST_FP.to_owned(),
        &layouts,
        fixture.description.head_rows as usize,
        CONTEXT,
        None,
    )
    .unwrap();
    let restored_plain = cache_plain
        .restore(&Host, &layouts, None, Some("salt-a"), &prompt, CONTEXT)
        .unwrap()
        .expect("plain server restores target from drafted disk snapshot");
    assert!(restored_plain.draft.is_none());
    assert_eq!(restored_plain.tokens, prompt);
}

#[test]
fn lru_eviction_enforces_disk_budget_and_refreshes_hit_entries() {
    let fixture = Fixture::small();
    let layouts = crate::loop_decoder::state::layouts_for(&fixture.description, CONTEXT as u64, TEST_LIMITS).unwrap();
    let driver = Driver::with_limits(&Host, &fixture, &fixture.description, CONTEXT as u64, TEST_LIMITS).unwrap();

    let make = |p: &[u32]| {
        let mut s = driver.clean().unwrap();
        let mut l = Vec::new();
        for &t in p {
            l = driver.step(&mut s, t).unwrap().logits;
        }
        (s, l)
    };
    let p1 = [1u32, 2, 3, 4, 5, 6];
    let p2 = [2u32, 3, 4, 5, 6, 7];
    let p3 = [3u32, 4, 5, 6, 7, 8];
    let (s1, l1) = make(&p1);
    let (s2, l2) = make(&p2);
    let (s3, l3) = make(&p3);

    // Measure single entry size, then set budget to hold exactly 2 entries (not 3).
    let probe = TempDir::new("lru-probe");
    let single_size = {
        let mut c = DiskPrefixCache::open(
            probe.path().to_path_buf(), Some(64 << 20), TEST_FP.to_owned(), &layouts, 20, CONTEXT, None,
        ).unwrap();
        c.save(&Host, &layouts, &s1, None, None, &p1, &l1).unwrap();
        c.total_bytes()
    };
    let budget = single_size * 2 + single_size / 2;

    let dir = TempDir::new("lru-evict");
    // Drop a stale temp file from our own PID to verify startup cleans it up.
    let stale_tmp = dir.path().join(format!("dead.vdpc.tmp.{}.999", std::process::id()));
    fs::write(&stale_tmp, b"partial").unwrap();

    let mut cache = DiskPrefixCache::open(
        dir.path().to_path_buf(), Some(budget), TEST_FP.to_owned(), &layouts, 20, CONTEXT, None,
    ).unwrap();
    assert!(!stale_tmp.exists(), "stale tmp file cleaned on startup");

    cache.save(&Host, &layouts, &s1, None, None, &p1, &l1).unwrap();
    cache.save(&Host, &layouts, &s2, None, None, &p2, &l2).unwrap();
    assert_eq!(cache.len(), 2);

    // Touch p1 via restore so p2 becomes the LRU victim.
    assert!(cache.restore(&Host, &layouts, None, None, &p1, CONTEXT).unwrap().is_some());

    // Saving p3 must evict p2 (oldest), keeping p1 and p3 within budget.
    cache.save(&Host, &layouts, &s3, None, None, &p3, &l3).unwrap();
    assert!(cache.total_bytes() <= budget, "total {} <= budget {budget}", cache.total_bytes());
    assert_eq!(cache.len(), 2);
    assert!(cache.restore(&Host, &layouts, None, None, &p1, CONTEXT).unwrap().is_some(), "p1 survived LRU");
    assert!(cache.restore(&Host, &layouts, None, None, &p3, CONTEXT).unwrap().is_some(), "p3 present");
    assert!(cache.restore(&Host, &layouts, None, None, &p2, CONTEXT).unwrap().is_none(), "p2 evicted");

    // Verify default hardware-derived budget resolves cleanly via `df -kP`.
    let default_budget = resolve_budget_bytes(dir.path(), None, 0).unwrap();
    assert!(default_budget > 0);
}

#[test]
fn bf16_draft_roundtrip_and_layout_mismatch() {
    let fixture = Fixture::small();
    let layouts = crate::loop_decoder::state::layouts_for(&fixture.description, CONTEXT as u64, TEST_LIMITS).unwrap();
    let driver = Driver::with_limits(&Host, &fixture, &fixture.description, CONTEXT as u64, TEST_LIMITS).unwrap();
    let prompt: Vec<u32> = (0..18).map(|i| (i % 19) as u32 + 1).collect();
    let mut state = driver.clean().unwrap();
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = driver.step(&mut state, t).unwrap().logits;
    }
    let cfg = MtpConfig {
        hidden_size: 16,
        intermediate_size: 16,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 4,
        sliding_window_size: 8,
        block_size: 4,
        markov_rank: 4,
        vocab_size: 20,
        unpadded_vocab_size: 20,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        final_logit_softcapping: 30.0,
        local_layer_ids: vec![0],
        target_layer_ids: vec![0, 1],
    };
    let kv_w = cfg.kv_width() / 2; // BF16 packed = half the f32 words
    let ks: Vec<f32> = (0..prompt.len() * kv_w).map(|i| 0.5 + (i as f32) * 0.02).collect();
    let vs: Vec<f32> = (0..prompt.len() * kv_w).map(|i| -0.25 - (i as f32) * 0.02).collect();
    let feat: Vec<f32> = (0..cfg.hidden_size).map(|i| (i as f32) * 0.25 - 0.5).collect();
    let mut draft = DraftChat::alloc_for_restore(&Host, &cfg, true, prompt.len(), CONTEXT, CONTEXT).unwrap();
    {
        let (kv_slots, f_slot) = draft.restore_targets_mut();
        let win = cfg.sliding_window_size - 1;
        let local_start = prompt.len() - win;
        let (lk, lv) = kv_slots[0].as_mut().unwrap();
        lk[..win * kv_w].copy_from_slice(&ks[local_start * kv_w..]);
        lv[..win * kv_w].copy_from_slice(&vs[local_start * kv_w..]);
        let (gk, gv) = kv_slots[1].as_mut().unwrap();
        gk[..prompt.len() * kv_w].copy_from_slice(&ks);
        gv[..prompt.len() * kv_w].copy_from_slice(&vs);
        f_slot.unwrap().copy_from_slice(&feat);
    }

    let dir = TempDir::new("draft-bf16-roundtrip");
    let mut cache1 = DiskPrefixCache::open_with_kv_bf16(
        dir.path().to_path_buf(),
        Some(16 << 20),
        TEST_FP.to_owned(),
        &layouts,
        fixture.description.head_rows as usize,
        CONTEXT,
        Some(&cfg),
        true,
    )
    .unwrap();
    cache1
        .save(&Host, &layouts, &state, Some((&draft, &cfg)), Some("bf16-scope"), &prompt, &logits)
        .unwrap();

    // Reopen with kv_bf16=true: must restore bit-exact with is_kv_bf16() == true.
    let mut cache2 = DiskPrefixCache::open_with_kv_bf16(
        dir.path().to_path_buf(),
        Some(16 << 20),
        TEST_FP.to_owned(),
        &layouts,
        fixture.description.head_rows as usize,
        CONTEXT,
        Some(&cfg),
        true,
    )
    .unwrap();
    let restored = cache2
        .restore(&Host, &layouts, Some(&cfg), Some("bf16-scope"), &prompt, CONTEXT)
        .unwrap()
        .expect("bf16 draft snapshot restores");
    let mut r_draft = restored.draft.expect("bf16 draft present");
    assert!(r_draft.is_kv_bf16());
    assert_eq!(r_draft.len(), draft.len());
    let (r_kv, r_feat) = r_draft.restore_targets_mut();
    let win = cfg.sliding_window_size - 1;
    let local_start = prompt.len() - win;
    let (rlk, rlv) = r_kv[0].as_ref().unwrap();
    assert_eq!(&rlk[..win * kv_w], &ks[local_start * kv_w..]);
    assert_eq!(&rlv[..win * kv_w], &vs[local_start * kv_w..]);
    let (rgk, rgv) = r_kv[1].as_ref().unwrap();
    assert_eq!(&rgk[..prompt.len() * kv_w], &ks[..]);
    assert_eq!(&rgv[..prompt.len() * kv_w], &vs[..]);
    assert_eq!(r_feat.unwrap().as_slice(), feat.as_slice());

    // Opening with kv_bf16=false must NOT load the half-width BF16 entry as FP32.
    let mut cache_fp32 = DiskPrefixCache::open_with_kv_bf16(
        dir.path().to_path_buf(),
        Some(16 << 20),
        TEST_FP.to_owned(),
        &layouts,
        fixture.description.head_rows as usize,
        CONTEXT,
        Some(&cfg),
        false,
    )
    .unwrap();
    assert!(
        cache_fp32
            .restore(&Host, &layouts, Some(&cfg), Some("bf16-scope"), &prompt, CONTEXT)
            .unwrap()
            .is_none(),
        "fp32 cache must reject bf16 draft snapshot"
    );
}
