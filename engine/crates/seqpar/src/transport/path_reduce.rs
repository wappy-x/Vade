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

/// The order a rank runs its steps in, as (broadcast, piece, step): one pass in
/// which the reduce of piece i comes before the broadcast of piece i - lag, where
/// lag is the rank's distance from the split rank. So the split rank sends while
/// it receives, every rank keeps at most two unread pieces in each mailbox, and
/// no rank waits on a round trip of the piece it just sent. Each piece still
/// takes its reduce steps and then its broadcast steps, so the sums are unchanged.
pub(super) fn actions(rank: usize, steps: ([Option<Step>; 2], [Option<Step>; 2]), pieces: usize)
    -> impl Iterator<Item = (bool, usize, Step)> {
    let lag = rank.abs_diff(2);
    (0..pieces + lag).flat_map(move |at| {
        let reduce = steps.0.into_iter().flatten().filter(move |_| at < pieces).map(move |s| (false, at, s));
        let broadcast = steps.1.into_iter().flatten().filter(move |_| at >= lag).map(move |s| (true, at - lag, s));
        reduce.chain(broadcast)
    })
}

/// Reduce receives add in tree_adds' operand order: the lower rank's sum first.
#[cfg(any(target_os = "linux", test))]
#[allow(clippy::assign_op_pattern)] // operand order in sums stays as written.
fn merge(piece: &mut [f32], arrived: &[f32], side: usize, broadcast: bool) {
    if broadcast { piece.copy_from_slice(arrived); }
    else if side == LEFT { for (value, prior) in piece.iter_mut().zip(arrived) { *value = *prior + *value; } }
    else { for (value, later) in piece.iter_mut().zip(arrived) { *value += *later; } }
}

impl Ring {
    /// Pipeline the pair-tree sum along the path and broadcast it in the same pass
    /// (actions). Each link carries one piece each way, as the sequential path
    /// pipeline did; small pieces let neighbouring links progress together
    /// without copy workers.
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
            let (len, pieces) = (values.len(), values.len().div_ceil(PIECE));
            for (broadcast, at, Step { side, receive }) in actions(self.rank, steps, pieces) {
                let piece = &mut values[at * PIECE..((at + 1) * PIECE).min(len)];
                let bytes = std::mem::size_of_val(piece);
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

    /// Run every rank's actions over links that hold at most two unread pieces
    /// (a mailbox's two receive slots); a full pass without progress is a deadlock.
    /// `order` picks the rank polling order of each pass: forward, reverse, or
    /// one rank run as far as it can go before the others (arrival-order skew).
    fn simulate(count: usize, start: &[Vec<f32>], piece: usize) -> Vec<Vec<f32>> { simulate_order(count, start, piece, 0) }
    fn simulate_order(count: usize, start: &[Vec<f32>], piece: usize, order: usize) -> Vec<Vec<f32>> {
        use std::collections::{HashMap, VecDeque};
        let mut values = start.to_vec();
        let pieces = start[0].len().div_ceil(piece);
        let programs = (0..count).map(|r| actions(r, path_tree_steps(r, count).unwrap(), pieces).collect::<Vec<_>>()).collect::<Vec<_>>();
        let mut links: HashMap<(usize, usize), VecDeque<Vec<f32>>> = HashMap::new();
        let mut next = vec![0; count];
        while (0..count).any(|r| next[r] < programs[r].len()) {
            let mut progress = false;
            let ranks: Vec<usize> = match order {
                0 => (0..count).collect(),
                1 => (0..count).rev().collect(),
                o => { let first = o % count; std::iter::repeat_n(first, 64).chain((0..count).filter(|&r| r != first)).collect() }
            };
            for r in ranks {
                let Some(&(broadcast, at, Step { side, receive })) = programs[r].get(next[r]) else { continue };
                let peer = if side == LEFT { r - 1 } else { r + 1 };
                let range = at * piece..((at + 1) * piece).min(values[r].len());
                if receive {
                    let Some(arrived) = links.entry((peer, r)).or_default().pop_front() else { continue };
                    merge(&mut values[r][range], &arrived, side, broadcast);
                } else {
                    let link = links.entry((r, peer)).or_default();
                    if link.len() == 2 { continue; }
                    link.push_back(values[r][range].to_vec());
                    assert!(link.len() <= 2, "more than two unread pieces");
                }
                next[r] += 1;
                progress = true;
            }
            assert!(progress, "path schedule deadlocks at {count} ranks, {pieces} pieces");
        }
        assert!(links.values().all(VecDeque::is_empty), "unconsumed piece at {count} ranks");
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
            // 1 to 37 pieces: fewer than, as many as and more than the lags, with partial tails.
            for piece in [37, 19, 13, 9, 7, 5, 4, 3, 2, 1] {
                for (rank, values) in simulate(count, &start, piece).iter().enumerate() {
                    assert_eq!(bits(values), bits(&total), "{count} ranks, piece {piece}, rank {rank}");
                }
            }
        }
    }

    #[test]
    fn path_pipeline_bits_do_not_depend_on_arrival_order() {
        // Any order in which pieces arrive gives the same bits on every rank:
        // receives pop in link order and each add takes fixed operands.
        for count in [3, 4] {
            let start = (0..count).map(|r| (0..23).map(|i| [1.0e8f32, 1.0, -1.0e8, 3.0][(r + i) % 4] * (1.0 + i as f32 / 5.0)).collect::<Vec<_>>()).collect::<Vec<_>>();
            let bits = |all: Vec<Vec<f32>>| all.iter().map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>();
            let first = bits(simulate_order(count, &start, 2, 0));
            for order in 1..2 + count {
                assert_eq!(bits(simulate_order(count, &start, 2, order)), first, "{count} ranks, order {order}");
            }
            assert!(first.iter().all(|values| values == &first[0]));
        }
    }
}
