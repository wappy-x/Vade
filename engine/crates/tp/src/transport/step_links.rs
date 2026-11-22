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
}
