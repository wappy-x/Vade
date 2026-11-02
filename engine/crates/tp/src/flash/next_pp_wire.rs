//! Over RDMA: the layer pipeline's polled message wire over one RoCE RC queue
//! pair (spark-seqpar's mailbox slot API). Two receive slots stay posted; one send slot.
//! At most two messages wait unread per side (a chain's wire tile and its final tile), so two
//! receive slots never run short.
use spark_seqpar::transport::RdmaEndpoint;
use std::{
    collections::VecDeque,
    io::{Read, Write},
    net::TcpStream,
    time::{Duration, Instant},
};

pub struct RdmaWire {
    /// Declared before `memory`, so it drops first: destroying the queue pair ends every
    /// posted receive and deregisters the region before the storage is freed.
    endpoint: RdmaEndpoint,
    memory: Vec<u64>,
    slot: usize,
    sends: usize,
    arrived: VecDeque<(usize, usize)>,
    /// LINK_TIMEOUT: the longest a send waits for the previous one to complete.
    limit: Duration,
}

impl RdmaWire {
    /// Both ranks call this right after the TCP link is up (same order on both sides).
    /// `want` must match on both ranks; mismatch fails both.
    pub fn connect(stream: &mut TcpStream, want: bool, slot: usize, limit: Duration) -> Result<Option<Self>, String> {
        let e = |e: std::io::Error| e.to_string();
        spark_engine::flash::next_batched::pp_negotiate_wire(stream, want, limit)?;
        let mut peer = [0u8; 1];
        if !want {
            return Ok(None);
        }
        spark_engine::flash::next_batched::pp_negotiate_capacity(stream, slot)?;
        // The storage is declared before the endpoint, so every early return below drops
        // the endpoint (queue pair and registration) first and frees the storage after it.
        let mut memory = vec![0u64; (3 * slot).div_ceil(8)];
        let mut endpoint = RdmaEndpoint::open(stream)?;
        stream.write_all(&endpoint.hello()).map_err(e)?;
        let mut hello = [0u8; 24];
        stream.read_exact(&mut hello).map_err(e)?;
        endpoint.ready(&hello)?;
        let base = memory.as_mut_ptr().cast::<u8>();
        unsafe {
            endpoint.register(base, 3 * slot)?;
            endpoint.post_slot_receive(base, slot, 0)?;
            endpoint.post_slot_receive(base.add(slot), slot, 1)?;
        }
        // Neither side sends before the peer's receives are posted.
        stream.write_all(&[0x52]).map_err(e)?;
        stream.read_exact(&mut peer).map_err(e)?;
        Ok(Some(Self { endpoint, memory, slot, sends: 0, arrived: VecDeque::new(), limit }))
    }
}

impl spark_engine::flash::next_batched::PpWire for RdmaWire {
    fn capacity(&self) -> usize {
        self.slot
    }
    fn send(&mut self, parts: &[&[u8]]) -> Result<(), String> {
        let total: usize = parts.iter().map(|p| p.len()).sum();
        if total == 0 || total > self.slot {
            return Err("pipeline wire message size".into());
        }
        let started = Instant::now();
        while self.sends > 0 {
            self.endpoint.poll_slots(&mut self.sends, &mut self.arrived)?;
            if started.elapsed() > self.limit {
                return Err("pipeline wire send completion timed out".into());
            }
            std::hint::spin_loop();
        }
        let base = self.memory.as_mut_ptr().cast::<u8>();
        let mut at = 2 * self.slot;
        for p in parts {
            unsafe { std::ptr::copy_nonoverlapping(p.as_ptr(), base.add(at), p.len()) };
            at += p.len();
        }
        unsafe { self.endpoint.post_slot_send(base.add(2 * self.slot), total)? };
        self.sends += 1;
        Ok(())
    }
    fn try_recv(&mut self, out: &mut Vec<u8>) -> Result<bool, String> {
        if self.arrived.is_empty() {
            self.endpoint.poll_slots(&mut self.sends, &mut self.arrived)?;
        }
        let Some((index, bytes)) = self.arrived.pop_front() else { return Ok(false) };
        if index > 1 || bytes > self.slot {
            return Err("pipeline wire arrival out of range".into());
        }
        let base = self.memory.as_mut_ptr().cast::<u8>();
        out.clear();
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(base.add(index * self.slot), bytes) });
        unsafe { self.endpoint.post_slot_receive(base.add(index * self.slot), self.slot, index)? };
        Ok(true)
    }
}
