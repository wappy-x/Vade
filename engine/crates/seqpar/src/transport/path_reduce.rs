//! Pipelined path all-reduce that keeps the pair-tree sum (tree_adds).
use super::Ring;
#[cfg(target_os = "linux")]
use super::{parallel_copy, timed, Instant, Result, TIMEOUT};

/// Neighbour sides, as in Ring::endpoints: 0 is the higher rank, 1 the lower.
const RIGHT: usize = 0;
const LEFT: usize = 1;

/// One step on one piece: receive from a side, or post to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Step { pub side: usize, pub receive: bool }

/// Steps of `rank` on each piece, for the reduce and then the broadcast, or None
/// when neighbour chains cannot give the pair-tree sum and the gather runs.
/// Ranks below the tree's top split add rightward into its last rank, the others
/// add leftward into the split rank, which adds the two halves and broadcasts the
/// total. A chain of one or two ranks is a pair, so this holds for three and four
/// ranks: ((r0 + r1) + r2) and ((r0 + r1) + (r2 + r3)).
pub(super) fn path_tree_steps(rank: usize, count: usize) -> Option<([Option<Step>; 2], [Option<Step>; 2])> {
    let split = 1usize << count.checked_sub(1)?.checked_ilog2()?;
    if split != 2 || rank >= count { return None; }
    let (left, right) = (rank > 0, rank + 1 < count);
    let step = |side, receive| Some(Step { side, receive });
    Some(if rank < split {
        ([left.then_some(Step { side: LEFT, receive: true }), step(RIGHT, false)],
            [step(RIGHT, true), left.then_some(Step { side: LEFT, receive: false })])
    } else if rank == split {
        ([right.then_some(Step { side: RIGHT, receive: true }), step(LEFT, true)],
            [step(LEFT, false), right.then_some(Step { side: RIGHT, receive: false })])
    } else {
        ([right.then_some(Step { side: RIGHT, receive: true }), step(LEFT, false)],
            [step(LEFT, true), right.then_some(Step { side: RIGHT, receive: false })])
    })
}

/// Reduce receives add in tree_adds' operand order: the lower rank's sum first.
#[cfg(any(target_os = "linux", test))]
fn merge(piece: &mut [f32], arrived: &[f32], side: usize, broadcast: bool) {
    if broadcast { piece.copy_from_slice(arrived); }
    else if side == LEFT { for (value, prior) in piece.iter_mut().zip(arrived) { *value = *prior + *value; } }
    else { for (value, later) in piece.iter_mut().zip(arrived) { *value += *later; } }
}

impl Ring {
    /// Pipeline the pair-tree sum along the path, then broadcast it. Each link
    /// carries one piece each way, as the sequential path pipeline did; small
    /// pieces let neighbouring links progress together without copy workers.
    #[cfg(target_os = "linux")]
    pub(super) fn reduce_path_tree(&mut self, values: &mut [f32], steps: ([Option<Step>; 2], [Option<Step>; 2])) -> Result<()> {
        if self.poisoned { return Err("ring is unusable after an earlier collective failure".into()); }
        const PIECE: usize = 65536;
        let profiling = self.profile.is_some();
        let start = Instant::now();
        let wall = if self.trace_transfers { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()) } else { 0 };
        let before = self.stats;
        let result = (|| -> Result<()> {
            let base = self.collective.as_mut_ptr().cast::<u8>();
            for (endpoint, mailbox) in self.endpoints.iter_mut().zip(&self.mailbox).chain(self.second.iter_mut().zip(&self.mailbox2)) {
                if let (Some(endpoint), Some(m)) = (endpoint, mailbox) {
                    if m.slot < PIECE * 4 || m.slot % 4 != 0 { return Err("path reduction needs aligned mailbox slots of at least 256 KiB".into()); }
                    unsafe { endpoint.register_cached(base.add(m.offset), 3 * m.slot, &mut self.collective_regions)?; }
                }
            }
            let deadline = start + TIMEOUT;
            for (broadcast, phase) in [(false, steps.0), (true, steps.1)] {
                for piece in values.chunks_mut(PIECE) {
                    let bytes = std::mem::size_of_val(piece);
                    for Step { side, receive } in phase.into_iter().flatten() {
                        // The first rail carries the piece's head, the second its tail.
                        let first = Self::first_rail(bytes, self.second[side].is_some());
                        let missing = || format!("path reduction lost its rank {} link", if side == LEFT { "-1" } else { "+1" });
                        let (endpoint, m) = (self.endpoints[side].as_mut().ok_or_else(missing)?, self.mailbox[side].as_mut().ok_or_else(missing)?);
                        if receive {
                            let (head, tail) = piece.split_at_mut(first / 4);
                            let stats = &mut self.stats;
                            let take = |arrived: &[u8], part: &mut [f32], ns: &mut u128| timed(profiling, ns, || {
                                merge(part, unsafe { std::slice::from_raw_parts(arrived.as_ptr().cast::<f32>(), part.len()) }, side, broadcast); Ok(()) });
                            unsafe { m.receive(endpoint, base, first, deadline, |arrived| take(arrived, head, &mut stats.consumer_ns))?; }
                            if let (Some(endpoint), Some(m)) = (&mut self.second[side], &mut self.mailbox2[side]) {
                                unsafe { m.receive(endpoint, base, bytes - first, deadline, |arrived| take(arrived, tail, &mut stats.consumer_ns))?; }
                            }
                            stats.received += bytes as u64;
                        } else {
                            let own = piece.as_ptr().cast::<u8>();
                            unsafe { m.post(endpoint, base, first, deadline, |slot| timed(profiling, &mut self.stats.producer_ns, || { parallel_copy(std::slice::from_raw_parts(own, first), slot); Ok(()) }))?; }
                            if let (Some(endpoint), Some(m)) = (&mut self.second[side], &mut self.mailbox2[side]) {
                                unsafe { m.post(endpoint, base, bytes - first, deadline, |slot| timed(profiling, &mut self.stats.producer_ns, || { parallel_copy(std::slice::from_raw_parts(own.add(first), bytes - first), slot); Ok(()) }))?; }
                            }
                            self.stats.sent += bytes as u64;
                        }
                    }
                }
            }
            Ok(())
        })();
        if result.is_err() { self.abort(); }
        let elapsed = start.elapsed().as_nanos();
        self.stats.elapsed_ns += elapsed;
        self.stats.exchanges += 1;
        if self.trace_transfers {
            let delta = self.stats.since(before);
            println!("seqpar_transfer rank={} exchange={} rdma=true sent_bytes={} received_bytes={} control_sent_bytes=0 control_received_bytes=0 elapsed_ns={elapsed} post_ns=0 wait_ns={elapsed} start_unix_ns={wall} mailbox=true path_reduce=true ok={}",
                self.rank, self.stats.exchanges, delta.sent, delta.received, result.is_ok());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run every rank's steps over FIFO links; a full pass without progress is a deadlock.
    fn simulate(count: usize, start: &[Vec<f32>]) -> Vec<Vec<f32>> {
        use std::collections::{HashMap, VecDeque};
        let mut values = start.to_vec();
        let steps = (0..count).map(|r| path_tree_steps(r, count).unwrap()).collect::<Vec<_>>();
        for broadcast in [false, true] {
            let mut links: HashMap<(usize, usize), VecDeque<Vec<f32>>> = HashMap::new();
            let mut next = vec![0; count];
            let phase = |r: usize| { let s = if broadcast { steps[r].1 } else { steps[r].0 }; s.into_iter().flatten().collect::<Vec<_>>() };
            while (0..count).any(|r| next[r] < phase(r).len()) {
                let mut progress = false;
                for r in 0..count {
                    let Some(&Step { side, receive }) = phase(r).get(next[r]) else { continue };
                    let peer = if side == LEFT { r - 1 } else { r + 1 };
                    if receive {
                        let Some(arrived) = links.entry((peer, r)).or_default().pop_front() else { continue };
                        merge(&mut values[r], &arrived, side, broadcast);
                    } else { links.entry((r, peer)).or_default().push_back(values[r].clone()); }
                    next[r] += 1;
                    progress = true;
                }
                assert!(progress, "path schedule deadlocks at {count} ranks");
            }
            assert!(links.values().all(VecDeque::is_empty), "unconsumed piece at {count} ranks");
        }
        values
    }

    #[test]
    fn path_pipeline_gives_the_pair_tree_bits() {
        for count in 1..=8 {
            let piped = path_tree_steps(0, count).is_some();
            assert_eq!(piped, (3..=4).contains(&count), "{count} ranks");
            assert!((0..count).all(|r| path_tree_steps(r, count).is_some() == piped));
            if !piped { continue; }
            // Values whose sums round differently in other orders.
            let start = (0..count).map(|r| (0..37).map(|i| [1.0e8f32, 1.0, -1.0e8, 3.0, 0.5, -7.25, 2.0e-3, 9.0e7][(r + i) % 8] * (1.0 + i as f32 / 7.0)).collect::<Vec<_>>()).collect::<Vec<_>>();
            let mut tree = start.clone();
            let mut total = tree[0].clone();
            super::super::tree_adds(count, |into, from| {
                let addend = tree[from].clone();
                let target = if into == 0 { &mut total } else { &mut tree[into] };
                for (value, other) in target.iter_mut().zip(&addend) { *value += *other; }
            });
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            for (rank, values) in simulate(count, &start).iter().enumerate() {
                assert_eq!(bits(values), bits(&total), "{count} ranks, rank {rank}");
            }
        }
    }
}
