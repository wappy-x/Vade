//! Pipeline front: the verify groups of one tick in flight on a layer chain (next_pp_overlap.rs).
//! A group's pass is the one verify_batch_greedy runs, cut at the exchange: `verify_begin` runs
//! layers [0, split) and posts the rows; `verify_finish` takes the rows back and runs the tail
//! (draft history, head) and the result readback. Groups hold different chats, and this rank
//! runs each group's work in the same order as before (begin, then tail), so one workspace
//! serves them in turn: a tail reads only the returned rows, its sessions and scratch.
use super::*;

impl<'a, 'w> NextSession<'a, 'w> {
    /// Greedy verify of `groups` (consecutive chat ranges, mtp_batch::verify_groups). On a chain
    /// front, group k + 1 runs layers [0, split) while group k is on the later ranks, with at
    /// most IN_FLIGHT groups on the wire. Elsewhere (one Spark, the tensor split) the groups run
    /// one after the other, as verify_batch_greedy. A chain front cuts a one-group wide tick
    /// into GROUP_ROWS groups when two can be in flight.
    pub fn verify_groups_greedy(b: &Backend<'a>, sessions: &mut [&mut Self], tokens: &[&[u32]],
        groups: &[std::ops::Range<usize>]) -> Result<Vec<Vec<u32>>> {
        if sessions.len() != tokens.len() || groups.last().is_none_or(|g| g.end != tokens.len()) {
            return Err("invalid MTP cohort groups".into());
        }
        let model = sessions[0].model;
        let mut out = Vec::with_capacity(tokens.len());
        let ready = |gs: &[std::ops::Range<usize>]| {
            let widest = gs.iter().map(|g| (g.len(), tokens[g.clone()].iter().map(|t| t.len()).sum::<usize>()))
                .fold((0, 0), |a, b| (a.0.max(b.0), a.1.max(b.1)));
            gs.len() > 1 && model.pp_overlap_ready(widest.0, widest.1)
        };
        // A wide tick (one group above the verify tile) on a chain front: tile-sized groups in
        // flight. Each chat's rows are the same; only the group it shares a pass with changes.
        let tiled = if groups.len() == 1 {
            super::super::pp::split_groups(&tokens.iter().map(|t| t.len()).collect::<Vec<_>>(), super::super::pp::GROUP_ROWS)
        } else { Vec::new() };
        let groups = if ready(&tiled) { &tiled[..] } else { groups };
        if !ready(groups) {
            for g in groups {
                out.extend(Self::verify_batch_greedy(b, &mut sessions[g.clone()], &tokens[g.clone()])?);
            }
            return Ok(out);
        }
        let mut open = std::collections::VecDeque::new();
        let result = (|| -> Result<()> {
            for g in groups {
                if open.len() == super::super::pp::IN_FLIGHT {
                    let done: std::ops::Range<usize> = open.pop_front().unwrap();
                    out.extend(Self::verify_finish(b, &mut sessions[done.clone()], &tokens[done])?);
                }
                match Self::verify_begin(b, &mut sessions[g.clone()], &tokens[g.clone()])? {
                    Some(ids) => out.extend(ids),
                    None => open.push_back(g.clone()),
                }
            }
            while let Some(done) = open.pop_front() {
                out.extend(Self::verify_finish(b, &mut sessions[done.clone()], &tokens[done])?);
            }
            Ok(())
        })();
        // A reply still owed (or a pass cut in two) leaves the ranks' state unknown: stop.
        result.map_err(|e| model.pp_fail_after_final(e))?;
        Ok(out)
    }
    /// Layers [0, split) of one group and its rows posted down the chain: None. Some(ids) if
    /// the pass did not post (it ran whole, as verify_batch_greedy).
    fn verify_begin(b: &Backend<'a>, sessions: &mut [&mut Self], tokens: &[&[u32]]) -> Result<Option<Vec<Vec<u32>>>> {
        let count = Self::verify_rows_of(sessions, tokens)?;
        let model = sessions[0].model;
        model.pp_defer(true);
        let result = b.next_rows_profile(true, sessions[0].pos, count, || {
            for (s, t) in sessions.iter_mut().zip(tokens) {
                s.prepare_verify(b, t)?;
            }
            Self::run_rows_inner(b, sessions, tokens, true, true, true).map(|_| ())
        });
        model.pp_defer(false);
        let posted = model.pp_posted(true);
        result?;
        if posted { Ok(None) } else { Self::verify_ids(b, sessions, tokens).map(Some) }
    }
    /// The posted group's rows back, its tail, its positions and its ids.
    fn verify_finish(b: &Backend<'a>, sessions: &mut [&mut Self], tokens: &[&[u32]]) -> Result<Vec<Vec<u32>>> {
        let count = Self::verify_rows_of(sessions, tokens)?;
        let model = sessions[0].model;
        b.next_rows_profile(true, sessions[0].pos, count, || {
            let spans = crate::flash_next_rows::validate(&sessions.iter().map(|s| s.pos).collect::<Vec<_>>(), tokens,
                &sessions.iter().map(|s| s.capacity).collect::<Vec<_>>(), model.checkpoint.config.vocab)?;
            let owner = crate::flash_next_rows::workspace_owner(sessions.iter().map(|s| s.capacity))?;
            let mut work = model.take_work(b, sessions[owner].capacity, count)?;
            work.bound_head = Self::bound_head(model, sessions, count, true, true, work.head_result.len());
            let result = (|| -> Result<()> {
                work.status.clear(b, count, 0)?;
                work.residual.resize_within_capacity(count * W)?;
                work.mixed.resize_within_capacity(count * H)?;
                work.block.resize_within_capacity(count * H)?;
                model.pp_collect_verify(b, &mut work.residual, count)?;
                Self::execute_tail(b, &mut work, sessions, tokens, &spans, true, true, true)?;
                b.synchronize()?;
                if work.status.failed_after_readback()? { Err("Flash-Next device routing rejected a row; reset cohort".into()) } else { Ok(()) }
            })();
            model.return_work(work);
            result?;
            for (s, t) in sessions.iter_mut().zip(tokens) {
                s.pos += t.len();
                s.sync_pool_state(b)?;
                s.valid = true;
                s.logits_ready = true;
            }
            Self::verify_ids(b, sessions, tokens)
        })
    }
    /// verify_batch_kind's checks; the group's row count.
    fn verify_rows_of(sessions: &[&mut Self], tokens: &[&[u32]]) -> Result<usize> {
        if sessions.len() != tokens.len() || sessions.is_empty() || sessions.len() > 16 {
            return Err("invalid MTP cohort".into());
        }
        let count = tokens.iter().try_fold(0usize, |n, t| n.checked_add(t.len()).ok_or("MTP row count overflow"))?;
        let _ = crate::flash_next_rows::head_tiles(count)?;
        Ok(count)
    }
    /// A finished verify pass: each chat's head ids from the workspace results.
    pub(in super::super) fn verify_ids(b: &Backend<'a>, sessions: &mut [&mut Self], tokens: &[&[u32]]) -> Result<Vec<Vec<u32>>> {
        let model = sessions[0].model;
        let held = model.work.borrow();
        let work = held.as_ref().ok_or("missing verifier workspace")?;
        let result = Self::split_results(b, sessions, &work.head_result, tokens.iter().map(|t| t.len()).sum());
        drop(held);
        if result.is_err() {
            for s in sessions.iter_mut() {
                s.valid = false;
            }
        }
        let ids: Vec<_> = result?.into_iter().map(|v| v.0).collect();
        let mut at = 0;
        Ok(tokens.iter().map(|t| {
            let row = ids[at..at + t.len()].to_vec();
            at += t.len();
            row
        }).collect())
    }
}
