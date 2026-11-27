//! Neighbour mailbox gather and receive-slot relay pipeline.
use super::Ring;
#[cfg(target_os = "linux")]
use super::{copy_relay, parallel_copy, rdma, relay_packet, timed, Instant, Kind, Mailbox, Result, SlotPacket, TIMEOUT};

impl Ring {
    /// Bytes of a message carried by the link's first rail: everything below
    /// 64 KiB, else a 64-byte aligned half; the second rail carries the rest.
    #[cfg(any(target_os = "linux", test))]
    pub(super) fn first_rail(bytes: usize, dual: bool) -> usize {
        if !dual || bytes < 65536 { bytes } else { bytes / 2 / 64 * 64 }
    }

    /// Forward the same rank-owned ranges as the staged path. Both directions
    /// are posted before either is awaited; every FP32 reduction still happens
    /// only after the full gather, in increasing rank order.
    #[cfg(target_os = "linux")]
    pub(super) fn gather_neighbour_mailboxes(&mut self, buffer: &mut [u8], offsets: &[usize]) -> Result<()> {
        let base = buffer.as_mut_ptr();
        let own = offsets[self.rank];
        self.gather_slots_pipeline_inner(&[offsets.to_vec()],
            |_, at, slot| { unsafe { parallel_copy(std::slice::from_raw_parts(base.add(own + at), slot.len()), slot); } Ok(()) },
            |_, r, at, arrived| { unsafe { parallel_copy(arrived, std::slice::from_raw_parts_mut(base.add(offsets[r] + at), arrived.len())); } Ok(()) },
            |()| Ok(()))
    }

    /// The neighbour gather's registered slots also support owned GPU events.
    /// Forward opaque packets before releasing receive slots; prefetch the next
    /// piece before this piece's final consumers, as in the pair pipeline.
    #[cfg(target_os = "linux")]
    pub fn gather_slots_pipeline<T>(&mut self, offsets: &[Vec<usize>],
        fill: impl FnMut(usize, usize, &mut [u8]) -> Result<T>,
        receive: impl FnMut(usize, usize, usize, &[u8]) -> Result<T>,
        wait: impl FnMut(T) -> Result<()>) -> Result<()> {
        self.collective(Kind::Pipeline, |ring| ring.gather_slots_pipeline_inner(offsets, fill, receive, wait))
    }
    #[cfg(target_os = "linux")]
    fn gather_slots_pipeline_inner<T>(&mut self, offsets: &[Vec<usize>],
        mut fill: impl FnMut(usize, usize, &mut [u8]) -> Result<T>,
        mut receive: impl FnMut(usize, usize, usize, &[u8]) -> Result<T>,
        mut wait: impl FnMut(T) -> Result<()>) -> Result<()> {
        if self.poisoned { return Err("ring is unusable after an earlier collective failure".into()); }
        if self.link.duplex() || !self.mailbox.iter().any(Option::is_some) { return Err("gather pipeline needs neighbour RDMA mailboxes".into()); }
        if offsets.iter().any(|o| o.len() != self.world + 1 || o[0] != 0 || o.windows(2).any(|w| w[0] > w[1])) {
            return Err("gather pipeline needs ordered rank ranges".into());
        }
        let profiling = self.profile.is_some();
        let start = Instant::now();
        let wall = if self.trace_transfers { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()) } else { 0 };
        let before = self.stats;
        let pieces = offsets.len();
        let (mut producer_ns, mut consumer_ns) = (0, 0);
        let mut producers: [Option<T>; 4] = std::array::from_fn(|_| None);
        let mut consumers: [Option<T>; 4] = std::array::from_fn(|_| None);
        let result = (|| -> Result<()> {
            let base = self.collective.as_mut_ptr().cast::<u8>();
            let dual = self.second.each_ref().map(Option::is_some);
            let owners = (0..=self.world).collect::<Vec<_>>();
            let schedule = (0..self.rounds()).map(|s| self.ranges(s, &owners, false)).collect::<Vec<_>>();
            let mut rails = self.endpoints.iter_mut().zip(&mut self.mailbox)
                .zip(self.second.iter_mut().zip(&mut self.mailbox2))
                .flat_map(|((e, m), (e2, m2))| [(e.as_mut(), m.as_mut()), (e2.as_mut(), m2.as_mut())]).collect::<Vec<_>>();
            for (endpoint, mailbox) in &mut rails {
                if let (Some(endpoint), Some(m)) = (endpoint, mailbox) {
                    if offsets.iter().any(|o| o.windows(2).any(|w| w[1] - w[0] > m.slot)) { return Err("collective exceeds the RDMA mailbox slot".into()); }
                    unsafe { endpoint.register_cached(base.add(m.offset), 3 * m.slot, &mut self.collective_regions)?; }
                }
            }
            let mut arrived: [Option<SlotPacket>; 4] = [None; 4];
            let mut prepare = |p: usize, rails: &mut [(Option<&mut rdma::Endpoint>, Option<&mut Mailbox>)], producers: &mut [Option<T>; 4]| -> Result<u128> {
                let mut ns = 0;
                let deadline = Instant::now() + TIMEOUT;
                let bytes = offsets[p][self.rank + 1] - offsets[p][self.rank];
                for (k, (endpoint, mailbox)) in rails.iter_mut().enumerate() {
                    let first = Self::first_rail(bytes, dual[k / 2]);
                    let (at, len) = if k % 2 == 0 { (0, first) } else { (first, bytes - first) };
                    if let (Some(endpoint), Some(m)) = (endpoint, mailbox) {
                        m.wait_send(*endpoint, deadline)?;
                        if len != 0 { producers[k] = Some(timed(profiling, &mut ns, || fill(p, at, unsafe { std::slice::from_raw_parts_mut(base.add(m.offset + 2 * m.slot), len) }))?); }
                    }
                }
                Ok(ns)
            };
            if !offsets.is_empty() { producer_ns += prepare(0, &mut rails, &mut producers)?; }
            for (p, offsets) in offsets.iter().enumerate() {
                for (step, [sr, sl, rl, rr]) in schedule.iter().enumerate() {
                    let deadline = Instant::now() + TIMEOUT;
                    let mut relays = [None; 4];
                    for (k, (endpoint, mailbox)) in rails.iter_mut().enumerate() {
                        let range = [&sr, &sl][k / 2];
                        if range.is_empty() { continue; }
                        let bytes = offsets[range.start + 1] - offsets[range.start];
                        let first = Self::first_rail(bytes, dual[k / 2]);
                        let len = if k % 2 == 0 { first } else { bytes - first };
                        if let Some(event) = producers[k].take() { timed(profiling, &mut producer_ns, || wait(event))?; }
                        if let (Some(endpoint), Some(m)) = (endpoint, mailbox) {
                            unsafe {
                                if let Some(packet) = relay_packet(k, range.start, bytes, dual, &arrived) {
                                    m.relay(*endpoint, packet as *mut u8, len, deadline)?;
                                    relays[k ^ 2] = Some(k);
                                } else {
                                    m.post(endpoint, base, len, deadline, |slot| {
                                        if step != 0 { timed(profiling, &mut producer_ns, || { copy_relay(k, bytes, dual, &arrived, slot); Ok(()) })?; }
                                        Ok(())
                                    })?;
                                }
                            }
                            self.stats.sent += len as u64;
                        }
                    }
                    // Post both directions before waiting, then release only after
                    // the outgoing HCA and the GPU consumer have stopped reading.
                    for destination in relays.into_iter().flatten() {
                        if let (Some(endpoint), Some(m)) = &mut rails[destination] { m.wait_send(*endpoint, deadline)?; }
                    }
                    for (k, (endpoint, mailbox)) in rails.iter_mut().enumerate() {
                        if let Some(event) = consumers[k].take() { timed(profiling, &mut consumer_ns, || wait(event))?; }
                        if let (Some(endpoint), Some(m), Some((index, packet, _, _))) = (endpoint, mailbox, arrived[k].take()) {
                            unsafe { endpoint.post_slot_receive(packet as *mut u8, m.slot, index)?; }
                        }
                    }
                    for (k, (endpoint, mailbox)) in rails.iter_mut().enumerate() {
                        let range = [&rr, &rl][k / 2];
                        if range.is_empty() { continue; }
                        let bytes = offsets[range.start + 1] - offsets[range.start];
                        let first = Self::first_rail(bytes, dual[k / 2]);
                        let len = if k % 2 == 0 { first } else { bytes - first };
                        if len == 0 { continue; }
                        if let (Some(endpoint), Some(m)) = (endpoint, mailbox) {
                            while m.arrived.is_empty() {
                                endpoint.poll_slots(&mut m.sends, &mut m.arrived)?;
                                if Instant::now() >= deadline { return Err("RDMA mailbox receive timed out".into()); }
                                std::hint::spin_loop();
                            }
                            let (index, n) = m.arrived.pop_front().unwrap();
                            if index != m.next || n != len { return Err("RDMA mailbox message out of order or wrong length".into()); }
                            arrived[k] = Some((index, unsafe { base.add(m.offset + index * m.slot) }, n, range.start));
                            m.next ^= 1;
                            self.stats.received += n as u64;
                        }
                    }
                    if step + 1 == schedule.len() && p + 1 < pieces { producer_ns += prepare(p + 1, &mut rails, &mut producers)?; }
                    for (k, packet) in arrived.iter().enumerate() {
                        if let Some((_, packet, n, _)) = packet {
                            let range = [&rr, &rl][k / 2];
                            let bytes = offsets[range.start + 1] - offsets[range.start];
                            let at = if k % 2 == 0 { 0 } else { Self::first_rail(bytes, dual[k / 2]) };
                            consumers[k] = Some(timed(profiling, &mut consumer_ns, || receive(p, range.start, at, unsafe { std::slice::from_raw_parts(*packet, *n) }))?);
                        }
                    }
                }
            }
            for (k, (endpoint, mailbox)) in rails.iter_mut().enumerate() {
                if let Some(event) = consumers[k].take() { timed(profiling, &mut consumer_ns, || wait(event))?; }
                if let (Some(endpoint), Some(m), Some((index, packet, _, _))) = (endpoint, mailbox, arrived[k]) {
                    unsafe { endpoint.post_slot_receive(packet as *mut u8, m.slot, index)?; }
                }
            }
            Ok(())
        })();
        let mut drained = Ok(());
        for event in producers.into_iter().flatten().chain(consumers.into_iter().flatten()) {
            let result = timed(profiling, &mut consumer_ns, || wait(event));
            if drained.is_ok() { drained = result; }
        }
        let result = result.and(drained);
        self.stats.producer_ns += producer_ns;
        self.stats.consumer_ns += consumer_ns;
        if result.is_err() { self.abort(); }
        let elapsed = start.elapsed().as_nanos();
        self.stats.elapsed_ns += elapsed;
        self.stats.exchanges += offsets.len() as u64;
        if self.trace_transfers {
            let delta = self.stats.since(before);
            println!("seqpar_transfer rank={} exchange={} rdma=true sent_bytes={} received_bytes={} control_sent_bytes=0 control_received_bytes=0 elapsed_ns={elapsed} post_ns=0 wait_ns={elapsed} start_unix_ns={wall} mailbox=true ok={}",
                self.rank, self.stats.exchanges, delta.sent, delta.received, result.is_ok());
        }
        result
    }

}
