//! The ring's forms for more than two ranks on a path that ask the ring itself, not a count of ranks: the step
//! barrier on the neighbor mailboxes, and whether the ring relays its small commands.
#[cfg(target_os="linux")]
use super::{Result, Ring, command_relay_world, step_barrier_on_mailbox};

#[cfg(target_os="linux")]
impl Ring {
    /// Whether this ring sends its small commands rank to rank on RDMA mailboxes (`open_command_relay`): the world and
    /// the form that `command_relay_world` names, with RDMA links. Every rank of a ring gives the same answer.
    pub fn relays_commands(&self) -> bool { command_relay_world(self.world, self.link.is_path()) && self.uses_rdma() }
    /// The step barrier where the ring is not a pair with a mailbox (`step_barrier`). A path of ranks takes the
    /// mailboxes its neighbors already poll (`self.mailbox` is [next rank, previous rank]; a rank at an end of the
    /// path has one). Every exchange is `Mailbox::reduce`: both ranks of a link send a word and add the other's.
    /// Three steps, each only where the link exists.
    ///
    /// ARRIVE with the next rank: this rank sends at once; the next rank answers when every rank behind it has
    /// arrived. This rank's word is 2 when a rank stands before it (it will send a release on this link), else 1.
    ///
    /// ARRIVE with the previous rank: this rank sends 1. A sum of 2 says the previous rank is the first, so every
    /// rank has arrived. A sum of 3 says a release follows: one more exchange, which the previous rank answers when
    /// it knows that every rank has arrived.
    ///
    /// RELEASE with the next rank, when this rank said it would send one.
    ///
    /// So a rank proceeds only when every rank has arrived, as on the control stream, and no rank sleeps in the
    /// kernel for a receive interrupt. On the first link of the path the exchange is the pair's: one word each way,
    /// sum 2. The deadline and the error text are the pair's. A ring that is not a path, a path without mailboxes
    /// and VADE_TP_STEP_BARRIER=tcp keep `barrier`.
    pub(super) fn step_barrier_links(&mut self) -> Result<()> {
        if !self.link.is_path() || self.mailbox.iter().all(Option::is_none) || !step_barrier_on_mailbox() { return self.barrier(); }
        let result = (|| {
            let trade = |mailbox: &mut crate::rdma::Mailbox, word: f32| { let mut sum = [word]; mailbox.reduce(&mut sum, crate::rdma::Wait::Barrier).map(|()| sum[0]) };
            let [next, previous] = &mut self.mailbox;
            let release = previous.is_some();
            if let Some(next) = next.as_mut() {
                let word = [1f32, 2f32][usize::from(release)];
                if !trade(next, word)?.eq(&(word + 1.0)) { return Err("the next rank's barrier word differs".to_string()); }
            }
            if let Some(previous) = previous.as_mut() {
                let mut sum = trade(previous, 1.0)?;
                if sum.eq(&3.0) { sum = trade(previous, 1.0)?; }
                if !sum.eq(&2.0) { return Err("the previous rank's barrier word differs".to_string()); }
            }
            if let Some(next) = next.as_mut().filter(|_| release) && !trade(next, 1.0)?.eq(&2.0) { return Err("the next rank's release word differs".to_string()); }
            Ok(())
        })();
        // A failed mailbox is dropped, as after any failed collective: the service has failed.
        result.map_err(|e| { self.mailbox = [None, None]; format!("rank barrier: {e}") })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_step_barrier_of_a_path_without_mailboxes_is_the_control_stream_barrier() {
        // Loopback has no fabric mailbox: four ranks on a path fall back, on every rank, and stay in step with `barrier`.
        // The ranks take their ports from the loopback helper of the transport tests (fresh ports if one is taken).
        super::super::tests::rings(4, true, |_, mut ring| {
            for _ in 0..3 { ring.step_barrier().unwrap(); ring.barrier().unwrap(); }
            ring.step_barrier().unwrap();
        });
    }
    /// SECOND RAIL for more than two ranks: every rank trades its two offers (toward the rank after it, toward the
    /// rank before it) and its port. Every cable gets its rail, or every rank stays on one rail and says why.
    #[test]
    fn four_ranks_find_the_second_rail_of_every_cable_or_all_stay_on_one_rail() {
        use crate::second_rail::{Candidate, Offer};
        use std::net::{Ipv4Addr, SocketAddr};
        // Addresses built from their parts (a documentation range). Cable k joins rank k and the rank after it; its
        // second rail is subnet 10 + k, with host 1 at rank k and host 2 at the rank after it.
        let at = |cable: usize, host: u8, port: u16| SocketAddr::from((Ipv4Addr::new(198, 51, 10 + cable as u8, host), port));
        let offer = |cable: usize, host: u8| Offer::Open(vec![Candidate { address: Ipv4Addr::new(198, 51, 10 + cable as u8, host), prefix: 24, apart: true }]);
        let port = |slot: usize| 29960 + 10 * slot as u16;
        for (path, whole) in [(true, true), (false, true), (true, false), (false, false)] {
            // Without `whole`, the cable between rank 1 and rank 2 has no second rail at rank 1.
            let after = [offer(0, 1), if whole { offer(1, 1) } else { Offer::Open(Vec::new()) }, offer(2, 1), offer(3, 1)];
            let rails = super::super::tests::rings(4, path, |rank, mut ring| {
                let found = ring.second_rail(&[after[rank].clone(), offer((rank + 3) % 4, 2)], port(rank)).unwrap();
                // The trade leaves the ring as a start expects it: nothing counted, and the next collective in step.
                assert_eq!((found.0.is_some(), ring.exchanges), (found.1.is_empty(), 0));
                assert_eq!(ring.gather_batch_f32(&[rank as f32], 4, 1, 1).unwrap(), [0.0, 1.0, 2.0, 3.0]);
                found
            });
            if whole {
                // A rank listens on its end of the cable before it and connects to the far end of the cable after it,
                // each on the port after that rank's own. An end of a path has one cable: its two ends, its own first.
                let listen = |slot: usize| at((slot + 3) % 4, 2, port(slot) + 1);
                let next = |slot: usize| at(slot, 2, port((slot + 1) % 4) + 1);
                let expected = if path { [(at(0, 1, port(0) + 1), next(0)), (listen(1), next(1)), (listen(2), next(2)), (listen(3), at(2, 1, port(2) + 1))] }
                    else { [0, 1, 2, 3].map(|slot| (listen(slot), next(slot))) };
                assert_eq!(rails.iter().map(|found| found.0.unwrap()).collect::<Vec<_>>(), expected, "path={path}");
            } else {
                let lone = "has no second fast link that is up, has an address and runs at the speed of the first";
                let other = "another link of this world has no second rail";
                assert!(rails.iter().all(|found| found.0.is_none()), "path={path}");
                assert_eq!(rails.iter().map(|found| found.1).collect::<Vec<_>>(), [other.to_string(), format!("this rank {lone}"), format!("the other rank {lone}"), other.to_string()], "path={path}");
            }
        }
    }
}
