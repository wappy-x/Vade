use super::*;

// Host-only recurrent state. It restores exactly its retained checkpoint.
struct Aligned { position: usize }
impl crate::Session for Aligned {
    fn step(&mut self, _: u32) -> Result<Vec<f32>, String> { unreachable!() }
    fn can_fork(&self) -> bool { self.position > 0 }
    fn can_fork_prefix(&self, position: usize) -> bool { position > 0 && position == self.position }
    fn checkpoint_reusable_tokens(&self, tokens: usize) -> usize {
        if self.position <= tokens { self.position } else { 0 }
    }
    fn fork(&self) -> Result<Self, String> { Ok(Self { position: self.position }) }
    fn fork_prefix(&self, position: usize) -> Result<Self, String> {
        assert_eq!(position, self.position);
        self.fork()
    }
}
fn state(tokens: &[u32]) -> CachedSession<Aligned> {
    CachedSession { checkpoint_lease: None, session: Aligned { position: tokens.len() / 64 * 64 },
        tokens: tokens.to_vec(), next_logits: vec![1.], reserved_session_bytes: 1 }
}

#[test]
fn returned_probe_history_supplies_a_longer_automatic_prefix_than_the_named_session() {
    // Source: returned run.out and run.err from the 0e1dafc tool kit.
    // Draft mode belongs to the request, not the cache namespace. The host
    // fixture calls the real cache methods used by scheduler/dispatch.rs.
    let now = Instant::now();
    let mut cache = SessionCache::new(CacheLimits { entries: 32, retained_bytes: 1 << 20,
        tokens_per_entry: 512, logits_per_entry: 1, idle_ttl: Duration::from_secs(3600) }).unwrap();
    let mut next_key = 0;
    let mut branches = Vec::new();
    for (length, draft) in [(16, "plain"), (16, "strict"), (128, "plain"), (128, "strict"), (128, "plain"), (128, "strict")] {
        let prompt = [1, 42, 1024, 17].repeat(length / 4);
        let mut divergent = prompt.clone(); divergent[8] = 220;
        let mut parent_key = None;
        for (index, ids) in [&prompt, &prompt, &prompt, &prompt, &divergent, &divergent, &prompt].into_iter().enumerate() {
            let disabled = index == 5;
            let named = if matches!(index, 1 | 3 | 4 | 6) { parent_key } else { None };
            let explicit = named.and_then(|key| cache.fork_scoped(&key, None, ids, now).state);
            let automatic = if disabled { None } else { cache.fork_automatic(None, ids, now).state };
            let explicit_rows = explicit.as_ref().map_or(0, |s| s.tokens.len());
            let automatic_rows = automatic.as_ref().map_or(0, |s| s.tokens.len());
            let selected = if automatic_rows > explicit_rows { automatic } else { explicit };
            let rows = selected.as_ref().map_or(0, |s| s.tokens.len());
            if let Some(selected) = &selected { assert_eq!(selected.tokens, ids[..rows]); }
            if index == 4 {
                assert_eq!(shared_prefix(&prompt, ids), 8);
                assert_eq!(explicit_rows, 0);
                println!("HISTORY prompt_tokens={length} draft={draft} named_shared={} named_restored={explicit_rows} automatic_restored={automatic_rows} selected_restored={rows}", shared_prefix(&prompt, ids));
                branches.push((length, draft, rows));
            }
            if disabled { assert_eq!(rows, 0); continue; }
            next_key += 1;
            let key = SessionKey([next_key; 32]);
            cache.snapshot_scoped(key, None, &state(ids), now, true);
            if matches!(index, 0 | 2) { parent_key = Some(key); }
        }
    }
    assert_eq!(branches, vec![(16, "plain", 0), (16, "strict", 0), (128, "plain", 0),
        (128, "strict", 128), (128, "plain", 128), (128, "strict", 128)]);
}
