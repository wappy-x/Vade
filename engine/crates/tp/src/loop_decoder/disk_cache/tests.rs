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
    io::{Read, Write},
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

    // Saving p3 must evict p2 (oldest), keeping p1 and p3 within budget and last_dir_mtime synchronized.
    cache.save(&Host, &layouts, &s3, None, None, &p3, &l3).unwrap();
    assert_eq!(cache.last_dir_mtime, fs::metadata(dir.path()).and_then(|m| m.modified()).ok());
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

#[test]
fn longer_disk_prefix_beats_shorter_ram_prefix_after_restart_with_zero_diffs() {
    let fixture = Fixture::small();
    let dir = TempDir::new("ram-shorter-than-disk");
    let shared: Vec<u32> = vec![3, 5, 2, 7, 11, 13, 4, 6, 8, 10];
    let mut long = shared.clone();
    long.extend_from_slice(&[12, 14, 15, 9, 17, 18, 19, 2]);
    let mut long_cont = long.clone();
    long_cont.extend_from_slice(&[4, 6, 8, 10]);

    let want_shared = direct_decode(&fixture, CONTEXT, &shared, 8);
    let want_long = direct_decode(&fixture, CONTEXT, &long, 8);
    let want_cont = direct_decode(&fixture, CONTEXT, &long_cont, 8);

    // Server #1: warm both the short shared-system prompt and the longer session onto disk.
    serve_once(&fixture, CONTEXT, dir.path(), Some(16 << 20), TEST_FP, 2, |send, addr| {
        let r1 = ask_ids(send, addr, &shared, 8, "");
        assert!(r1.starts_with("HTTP/1.1 200 "), "{r1}");
        assert_eq!(parse_token_ids(&r1), want_shared);

        let r2 = ask_ids(send, addr, &long, 8, "");
        assert!(r2.starts_with("HTTP/1.1 200 "), "{r2}");
        assert_eq!(parse_token_ids(&r2), want_long);
    });
    assert_eq!(vdpc_files(dir.path()).len(), 2, "both short and long snapshots persisted");

    // Server #2 (fresh RAM, warm disk): first promote the shorter shared prefix into RAM,
    // then send the longer session — disk MUST override the shorter RAM fork and hit 100% (18/18),
    // not shadow disk with the 10-token RAM prefix.
    serve_once(&fixture, CONTEXT, dir.path(), Some(16 << 20), TEST_FP, 3, |send, addr| {
        let r_short = ask_ids(send, addr, &shared, 8, "");
        assert!(r_short.starts_with("HTTP/1.1 200 "), "{r_short}");
        assert_eq!(parse_cached_tokens(&r_short), shared.len(), "short session hits disk and warms RAM: {r_short}");
        assert_eq!(parse_token_ids(&r_short), want_shared);

        let r_long = ask_ids(send, addr, &long, 8, "");
        assert!(r_long.starts_with("HTTP/1.1 200 "), "{r_long}");
        assert_eq!(
            parse_cached_tokens(&r_long),
            long.len(),
            "longer disk prefix must beat shorter RAM prefix: {r_long}"
        );
        assert_eq!(parse_token_ids(&r_long), want_long);

        // Next continuation matches the 18-token promoted RAM prefix (disk is not strictly longer).
        let r_cont = ask_ids(send, addr, &long_cont, 8, "");
        assert!(r_cont.starts_with("HTTP/1.1 200 "), "{r_cont}");
        assert_eq!(parse_cached_tokens(&r_cont), long.len(), "continuation hits promoted RAM prefix: {r_cont}");
        assert_eq!(parse_token_ids(&r_cont), want_cont);
    });

    // Server #3 (fresh RAM, warm disk): warm `shared` into RAM, then send a *continuation*
    // (`long_cont` > `long`) directly — disk's 18-token entry MUST still beat RAM's 10-token prefix!
    serve_once(&fixture, CONTEXT, dir.path(), Some(16 << 20), TEST_FP, 2, |send, addr| {
        let r_short = ask_ids(send, addr, &shared, 8, "");
        assert!(r_short.starts_with("HTTP/1.1 200 "), "{r_short}");
        assert_eq!(parse_cached_tokens(&r_short), shared.len());
        assert_eq!(parse_token_ids(&r_short), want_shared);

        let mut longer_tail = long_cont.clone();
        longer_tail.extend_from_slice(&[1, 3]);
        let want_tail = direct_decode(&fixture, CONTEXT, &longer_tail, 8);
        let r_tail = ask_ids(send, addr, &longer_tail, 8, "");
        assert!(r_tail.starts_with("HTTP/1.1 200 "), "{r_tail}");
        assert_eq!(
            parse_cached_tokens(&r_tail),
            long_cont.len(),
            "longer disk continuation must beat shorter RAM prefix: {r_tail}"
        );
        assert_eq!(parse_token_ids(&r_tail), want_tail);
    });
}

#[test]
fn multi_chunk_direct_stream_roundtrips_and_detects_per_chunk_tamper() {
    // Construct a synthetic state whose payload spans > 2.5 * CHUNK_BYTES (> 10 MiB, 3+ chunks)
    // so multi-worker parallel DirectReader + DirectSequentialWriter + sub-chunk upload +
    // cross-chunk tensor boundaries are all exercised end-to-end.
    let layout = Layout {
        rows: 1024,
        window: 1024,
        step_rows: 16,
        key_bytes: 64 << 20,
        ring: false,
        row_words: 1536, // 1024 * 1536 * 4 = 6 MiB per plane => K+V = 12 MiB (+ rings => 4 chunks)
        kv_channels: 16,
        kv_taps: 4,
        channels: 32,
        taps: 4,
    };
    let layouts = [layout];
    let pos = 1024u64;
    let plane_words = state::prefix_plane_words(&layout, pos);
    let keys: Vec<f32> = (0..plane_words)
        .map(|i| f32::from_bits(0x3F80_0000 ^ ((i as u32).wrapping_mul(2_654_435_761) & 0x007F_FFFF)))
        .collect();
    let values: Vec<f32> = (0..plane_words)
        .map(|i| f32::from_bits(0xBF80_0000 ^ ((i as u32).wrapping_mul(1_597_334_677) & 0x007F_FFFF)))
        .collect();
    let rings: [Vec<f32>; 4] = [
        (0..64).map(|i| i as f32 * 0.25).collect(),
        (0..64).map(|i| -(i as f32) * 0.5).collect(),
        (0..128).map(|i| i as f32 * 0.125 + 1.0).collect(),
        (0..128).map(|i| -(i as f32) * 0.125 - 1.0).collect(),
    ];
    let state = State {
        position: pos,
        visits: vec![state::VisitState {
            keys: keys.clone(),
            values: values.clone(),
            rings: rings.clone(),
            next: std::array::from_fn(|_| None),
            planes: None,
            every: std::array::from_fn(|_| None),
            written: None,
        }],
    };
    let prompt: Vec<u32> = (0..pos as u32).map(|i| (i * 7 + 3) % 1000).collect();
    let logits: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect();

    let dir = TempDir::new("multichunk-direct");
    let mut cache = DiskPrefixCache::open(
        dir.path().to_path_buf(),
        Some(128 << 20),
        TEST_FP.to_owned(),
        &layouts,
        64,
        pos as usize,
        None,
    )
    .unwrap();
    cache.save(&Host, &layouts, &state, None, Some("mc"), &prompt, &logits).unwrap();
    let saved_path = vdpc_files(dir.path()).pop().unwrap();
    let saved_bytes = fs::read(&saved_path).unwrap();
    assert!(saved_bytes.len() > 3 * CHUNK_BYTES, "test must span >= 4 chunks");

    // Clean restore must match every single float bit-for-bit across all chunks and tensors.
    let restored = cache
        .restore(&Host, &layouts, None, Some("mc"), &prompt, pos as usize)
        .unwrap()
        .expect("multi-chunk snapshot restores");
    assert_eq!(restored.tokens, prompt);
    assert_eq!(restored.state.visits[0].keys.len(), keys.len());
    assert!(restored.state.visits[0].keys.iter().zip(&keys).all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!(restored.state.visits[0].values.iter().zip(&values).all(|(a, b)| a.to_bits() == b.to_bits()));
    for (got_ring, want_ring) in restored.state.visits[0].rings.iter().zip(&rings) {
        assert!(got_ring.iter().zip(want_ring).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    // Flipping a single bit in chunk 0, chunk 1, chunk 2, or final chunk must fail CRC and delete file.
    let file_len = saved_bytes.len();
    for (label, offset) in [
        ("chunk0", 5000usize),
        ("chunk1", CHUNK_BYTES + 12345),
        ("chunk2", 2 * CHUNK_BYTES + 98765),
        ("last_chunk", file_len - 128),
    ] {
        let tdir = TempDir::new(&format!("mc-tamper-{label}"));
        let mut bad = saved_bytes.clone();
        bad[offset] ^= 0x08;
        let bad_path = tdir.path().join(saved_path.file_name().unwrap());
        fs::write(&bad_path, &bad).unwrap();
        let mut tcache = DiskPrefixCache::open(
            tdir.path().to_path_buf(),
            Some(128 << 20),
            TEST_FP.to_owned(),
            &layouts,
            64,
            pos as usize,
            None,
        )
        .unwrap();
        assert!(
            tcache.restore(&Host, &layouts, None, Some("mc"), &prompt, pos as usize).unwrap().is_none(),
            "{label}: tampered multi-chunk snapshot must never restore"
        );
        assert!(!bad_path.exists(), "{label}: tampered multi-chunk snapshot must be removed");
    }
}

#[test]
fn multi_replica_shared_dir_discovers_peer_entries_after_own_save_and_enforces_shared_budget() {
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
    let p4 = [4u32, 5, 6, 7, 8, 9];
    let (s1, l1) = make(&p1);
    let (s2, l2) = make(&p2);
    let (s3, l3) = make(&p3);
    let (s4, l4) = make(&p4);

    let probe = TempDir::new("mr-probe");
    let single_size = {
        let mut c = DiskPrefixCache::open(
            probe.path().to_path_buf(), Some(64 << 20), TEST_FP.to_owned(), &layouts, 20, CONTEXT, None,
        ).unwrap();
        c.save(&Host, &layouts, &s1, None, None, &p1, &l1).unwrap();
        c.total_bytes()
    };
    let budget = single_size * 2 + single_size / 2;

    // 1. Lost-update mtime race: Replica A saves p1, then Replica B saves p2 (bumping dir mtime).
    // Replica B MUST still discover and restore p1 from the shared directory without restarting!
    let dir = TempDir::new("mr-handoff");
    let mut rep_a = DiskPrefixCache::open(
        dir.path().to_path_buf(), Some(64 << 20), TEST_FP.to_owned(), &layouts, 20, CONTEXT, None,
    ).unwrap();
    let mut rep_b = DiskPrefixCache::open(
        dir.path().to_path_buf(), Some(64 << 20), TEST_FP.to_owned(), &layouts, 20, CONTEXT, None,
    ).unwrap();

    rep_a.save(&Host, &layouts, &s1, None, None, &p1, &l1).unwrap();
    rep_b.save(&Host, &layouts, &s2, None, None, &p2, &l2).unwrap();

    let hit_b_on_p1 = rep_b.restore(&Host, &layouts, None, None, &p1, CONTEXT).unwrap();
    assert!(
        hit_b_on_p1.is_some(),
        "Replica B must discover Replica A's snapshot even after Replica B saved its own entry"
    );
    let hit_a_on_p2 = rep_a.restore(&Host, &layouts, None, None, &p2, CONTEXT).unwrap();
    assert!(
        hit_a_on_p2.is_some(),
        "Replica A must discover Replica B's snapshot"
    );

    // 2. Shared tight LRU budget: two replicas saving to the same directory must jointly respect budget.
    let bdir = TempDir::new("mr-budget");
    let mut b_a = DiskPrefixCache::open(
        bdir.path().to_path_buf(), Some(budget), TEST_FP.to_owned(), &layouts, 20, CONTEXT, None,
    ).unwrap();
    let mut b_b = DiskPrefixCache::open(
        bdir.path().to_path_buf(), Some(budget), TEST_FP.to_owned(), &layouts, 20, CONTEXT, None,
    ).unwrap();

    b_a.save(&Host, &layouts, &s1, None, None, &p1, &l1).unwrap();
    b_a.save(&Host, &layouts, &s2, None, None, &p2, &l2).unwrap();
    b_b.save(&Host, &layouts, &s3, None, None, &p3, &l3).unwrap();
    b_b.save(&Host, &layouts, &s4, None, None, &p4, &l4).unwrap();

    let files = vdpc_files(bdir.path());
    let actual_bytes: u64 = files.iter().map(|p| fs::metadata(p).unwrap().len()).sum();
    assert!(
        actual_bytes <= budget,
        "shared dir bytes ({actual_bytes}) exceeded budget ({budget}); files={}",
        files.len()
    );
    assert_eq!(files.len(), 2, "only 2 entries fit in shared budget");
    // The newest entry saved by Replica B must survive its own eviction pass.
    assert!(b_a.restore(&Host, &layouts, None, None, &p4, CONTEXT).unwrap().is_some());

    // 3. Cross-replica LRU touch propagation: Replica A saves p1 & p2; Replica B hits p1
    // (advancing mtime); Replica A saves p3 -> Replica A must evict p2 and keep p1.
    let ldir = TempDir::new("mr-lru-touch");
    let mut l_a = DiskPrefixCache::open(
        ldir.path().to_path_buf(), Some(budget), TEST_FP.to_owned(), &layouts, 20, CONTEXT, None,
    ).unwrap();
    let mut l_b = DiskPrefixCache::open(
        ldir.path().to_path_buf(), Some(budget), TEST_FP.to_owned(), &layouts, 20, CONTEXT, None,
    ).unwrap();
    l_a.save(&Host, &layouts, &s1, None, None, &p1, &l1).unwrap();
    l_a.save(&Host, &layouts, &s2, None, None, &p2, &l2).unwrap();
    std::thread::sleep(Duration::from_millis(20));
    assert!(l_b.restore(&Host, &layouts, None, None, &p1, CONTEXT).unwrap().is_some());
    l_a.save(&Host, &layouts, &s3, None, None, &p3, &l3).unwrap();
    assert!(l_a.restore(&Host, &layouts, None, None, &p1, CONTEXT).unwrap().is_some(), "peer-touched p1 must survive LRU");
    assert!(l_a.restore(&Host, &layouts, None, None, &p3, CONTEXT).unwrap().is_some(), "newly saved p3 must survive");
    assert!(l_a.restore(&Host, &layouts, None, None, &p2, CONTEXT).unwrap().is_none(), "untouched p2 must be evicted");
}

#[test]
fn concurrent_dual_replica_http_stress_with_tight_lru_has_zero_diffs_and_zero_corrupt_files() {
    let fixture = Fixture::small();
    let layouts = crate::loop_decoder::state::layouts_for(&fixture.description, CONTEXT as u64, TEST_LIMITS).unwrap();
    let driver = Driver::with_limits(&Host, &fixture, &fixture.description, CONTEXT as u64, TEST_LIMITS).unwrap();

    // Measure single entry size to set a tight shared budget holding ~4 of 16 sessions.
    let sample_prompt: Vec<u32> = (0..12).map(|i| ((i * 3 + 2) % 18 + 2) as u32).collect();
    let single_size = {
        let mut s = driver.clean().unwrap();
        let mut l = Vec::new();
        for &t in &sample_prompt {
            l = driver.step(&mut s, t).unwrap().logits;
        }
        let probe = TempDir::new("mr-http-probe");
        let mut c = DiskPrefixCache::open(
            probe.path().to_path_buf(), Some(64 << 20), TEST_FP.to_owned(), &layouts, 20, CONTEXT, None,
        ).unwrap();
        c.save(&Host, &layouts, &s, None, None, &sample_prompt, &l).unwrap();
        c.total_bytes()
    };
    let budget = single_size * 4 + single_size / 2;

    let prompts: Vec<Vec<u32>> = (0u32..16)
        .map(|s| (0u32..12).map(|i| ((s * 5 + i * 3 + 2) % 18) + 2).collect())
        .collect();
    let expected: Vec<Vec<u32>> = prompts.iter().map(|p| direct_decode(&fixture, CONTEXT, p, 8)).collect();

    let shared_dir = TempDir::new("mr-http-stress");
    let listener_a = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr_a = listener_a.local_addr().unwrap();
    let listener_b = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr_b = listener_b.local_addr().unwrap();
    let stop = AtomicBool::new(false);

    std::thread::scope(|scope| {
        for (listener, dir) in [(listener_a, shared_dir.path().to_path_buf()), (listener_b, shared_dir.path().to_path_buf())] {
            let stop = &stop;
            let fixture = &fixture;
            let layouts = &layouts;
            scope.spawn(move || {
                let seats = Seats { limits: TEST_LIMITS, chat: 1 << 20, room: 64 << 20, most: 4 };
                let service = Service::new(&Host, fixture, &fixture.description, CONTEXT, vec![1], seats).unwrap();
                let disk = DiskPrefixCache::open(
                    dir, Some(budget), TEST_FP.to_owned(), layouts, fixture.description.head_rows as usize, CONTEXT, None,
                ).unwrap();
                let service = service.with_disk_cache(disk);
                let lim = limits(CONTEXT, 24).unwrap();
                let (cache, config) = service.schedule();
                spark_serve::scheduler::serve_requests(
                    listener, &service, None::<&spark_serve::tokenizer::ModelTokenizer>, &lim, &cache, &config, stop, "fixture", 16,
                ).unwrap();
            });
        }

        let send_one = |addr: SocketAddr, prompt: &[u32]| -> String {
            let mut stream = TcpStream::connect(addr).unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
            let ids = prompt.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
            let body = format!(
                r#"{{"prompt_token_ids":[{ids}],"stream":true,"max_tokens":8,"temperature":0,"ignore_eos":true,"stream_options":{{"include_usage":true}}}}"#
            );
            let req = format!(
                "POST /v1/completions HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(req.as_bytes()).unwrap();
            let mut reply = String::new();
            let _ = stream.read_to_string(&mut reply);
            reply
        };

        // Wave 1 (16 concurrent writes across both replicas) + Wave 2 (16 cross-replica reads/writes).
        for wave in 0..2 {
            let mut handles = Vec::new();
            for s in 0..16 {
                let addr = if (s + wave) % 2 == 0 { addr_a } else { addr_b };
                let p = prompts[s].clone();
                let want = expected[s].clone();
                handles.push(scope.spawn(move || {
                    let reply = send_one(addr, &p);
                    assert!(reply.starts_with("HTTP/1.1 200 "), "wave={wave} s={s}: {reply}");
                    assert_eq!(parse_token_ids(&reply), want, "wave={wave} s={s}: token-ID mismatch");
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        }
    });

    // Zero leftover temp files, zero corrupt entries, and total bytes within budget (allowing at most 1 in-flight concurrent overlap? Actually sequential end settles <= budget + 1 entry, verify intact envelopes).
    let mut vdpc_count = 0usize;
    let mut total_bytes = 0u64;
    for entry in fs::read_dir(shared_dir.path()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        assert!(!name.contains(".tmp."), "leftover temp file: {name}");
        if name.ends_with(".vdpc") {
            let mut file = File::open(entry.path()).unwrap();
            read_and_verify_envelope(&mut file, true).expect("every surviving .vdpc must pass full envelope verification");
            total_bytes += entry.metadata().unwrap().len();
            vdpc_count += 1;
        }
    }
    assert!(vdpc_count > 0 && vdpc_count <= 5, "expected bounded entries under tight LRU, got {vdpc_count} ({total_bytes} B vs budget {budget} B)");
}
