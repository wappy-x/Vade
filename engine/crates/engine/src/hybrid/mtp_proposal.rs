// Proposal rows are scratch beyond the committed MTP prefix.
// This body is also exercised by the host storage/failure adapter.
// Forks copy only the rows preceding the restored cursor.
impl Session<'_, '_, '_> {
    pub fn set_draft_depth(&mut self, depth: usize) -> Result<()> {
        if depth > if self.external.is_some() { 8 } else { 4 } { return Err("hybrid draft depth exceeds model capacity".into()); }
        self.draft_depth = depth;
        Ok(())
    }
    /// Produce successors while retaining the committed prefix.
    /// The target verifier owns acceptance; proposal scratch is never a prefix.
    /// Errors can leave scratch KV rows, but cannot advance committed history.
    /// Repeating a proposal uses the same root hidden state and past.
    pub fn propose_mtp_tokens(
        &mut self,
        mut token: u32,
        depth: usize,
    ) -> Result<Vec<u32>> {
        self.finish_mtp_verification()?;
        let model = self.model;
        // Rank-local timing must never change the number of TP collectives.
        let gated_depth = self.draft_policy.options.gated_depth
            && model.bound.as_ref().is_none_or(|bound| !bound.group.distributed());
        let mtp = self
            .mtp
            .as_mut()
            .ok_or("MTP weights were not loaded")?;
        if mtp.past + 1 != self.position {
            return Err("MTP cache does not match target prefix".into());
        }
        if depth == 0 || depth > 4 || self.position + depth + 1 > model.capacity {
            return Err("MTP depth must be 1..=4 within the target context".into());
        }
        let past = mtp.past;
        let root = token;
        mtp.proposal_root = None;
        let result = (|| {
            let mut tokens = Vec::with_capacity(depth);
            let mut cumulative = 1.0;
            for i in 0..depth {
                let hidden = if i == 0 {
                    mtp.last.slice(0..model.config.hidden as usize)?
                } else {
                    mtp.normalized.slice(0..model.config.hidden as usize)?
                };
                token = mtp
                    .forward(model, token, &hidden, true)?
                    .ok_or("missing MTP prediction")?;
                tokens.push(token);
                if i + 1 < depth && gated_depth {
                    cumulative *= mtp.confidence(model)?;
                    if !self.draft_policy.more(cumulative, i+1) { break; }
                }
            }
            Ok(tokens)
        })();
        // Verification owns commits; retries and forks see only the old prefix.
        // Only the root used the canonical hidden row. Later proposal rows
        // must be repaired with target hidden rows even when their IDs match.
        if result.is_ok() { mtp.proposal_root = Some((past, root)); }
        mtp.past = past;
        result
    }
}
