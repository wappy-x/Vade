//! Neighbour rendezvous and RDMA rail setup.
use super::{collective, pump, Duration, Instant, PeerLayout, Result, Ring, TransportStats, TIMEOUT};
use std::net::SocketAddr;
#[cfg(target_os = "linux")]
use super::{rdma, KvStaging, TcpStream};

impl Ring {
    /// Bind the incoming neighbour interface and connect to the next rank.
    /// Every process must share run_id. SEQPAR_RDMA=0 selects TCP, =1 requires
    /// RDMA, and the default auto mode falls back collectively during startup.
    pub fn connect(rank: usize, world: usize, listen: SocketAddr, next: SocketAddr, run_id: &str) -> Result<Self> {
        Self::connect_topology(rank, world, listen, next, run_id, false, TIMEOUT)
    }

    /// Open a chain when the last and first ranks have no physical closing link.
    /// Rank zero ignores listen; the last rank ignores next.
    pub fn connect_path(rank: usize, world: usize, listen: SocketAddr, next: SocketAddr, run_id: &str) -> Result<Self> {
        Self::connect_topology(rank, world, listen, next, run_id, true, TIMEOUT)
    }

    pub(crate) fn connect_topology(rank: usize, world: usize, listen: SocketAddr, next: SocketAddr, run_id: &str, path: bool, setup: Duration) -> Result<Self> {
        if rank >= world {
            return Err("rank must be less than a positive world".into());
        }
        if world > isize::MAX as usize / std::mem::size_of::<usize>() - 1 {
            return Err("rank offsets exceed host address space".into());
        }
        let deadline = Instant::now() + setup;
        let mut ring = Self {
            rank, world, link: PeerLayout::new(world, path), right: None, left: None, poisoned: false,
            trace_transfers: std::env::var("SEQPAR_TRACE_TRANSFERS").is_ok_and(|v| v == "1"),
            profile_layers: collective::layers_enabled(),
            setup_deadline: Some(deadline),
            #[cfg(target_os = "linux")]
            retain_regions: false,
            #[cfg(target_os = "linux")]
            registrations_sealed: false,
            stats: TransportStats::default(),
            profile: None,
            #[cfg(target_os = "linux")]
            endpoints: [None, None],
            #[cfg(target_os = "linux")]
            auxiliary: None,
            #[cfg(target_os = "linux")]
            second: [None, None],
            #[cfg(target_os = "linux")]
            staging: KvStaging::default(),
            #[cfg(target_os = "linux")]
            collective: Vec::new(),
            #[cfg(target_os = "linux")]
            collective_regions: Vec::new(),
            #[cfg(target_os = "linux")]
            rdma_ready_only: false,
            #[cfg(target_os = "linux")]
            mailbox: [None, None],
            #[cfg(target_os = "linux")]
            mailbox2: [None, None],
        };
        if world == 1 { ring.setup_deadline = None; return Ok(ring); }
        let mode = std::env::var("SEQPAR_RDMA").unwrap_or_else(|_| "auto".into());
        if !matches!(mode.as_str(), "auto" | "0" | "1") { return Err("SEQPAR_RDMA must be auto, 0, or 1".into()); }
        let (mut right, mut left) = crate::control::connect_peers(
            (!path || rank > 0).then_some(listen), (!path || rank + 1 < world).then_some(next), deadline)?;
        let run_hash = run_id.bytes().fold(0xcbf29ce484222325u64 ^ u64::from(path), |h, b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3));
        let mut hello = [0u8; 32];
        hello[..8].copy_from_slice(b"SEQPAR04");
        hello[8..16].copy_from_slice(&(rank as u64).to_le_bytes());
        hello[16..24].copy_from_slice(&(world as u64).to_le_bytes());
        hello[24..].copy_from_slice(&run_hash.to_le_bytes());
        if let Some(right) = &mut right {
            right.set_nonblocking(true).map_err(|e| e.to_string())?;
            pump(Some(right), None, &hello, &[], &mut [], &mut [], &mut ring.stats, Some(deadline))?;
        }
        if let Some(left) = &mut left {
            left.set_nonblocking(true).map_err(|e| e.to_string())?;
            pump(None, Some(left), &[], &[], &mut hello, &mut [], &mut ring.stats, Some(deadline))?;
            if &hello[..8] != b"SEQPAR04"
                || u64::from_le_bytes(hello[8..16].try_into().unwrap()) != ((rank + world - 1) % world) as u64
                || u64::from_le_bytes(hello[16..24].try_into().unwrap()) != world as u64
                || u64::from_le_bytes(hello[24..].try_into().unwrap()) != run_hash
            {
                return Err("ring protocol/rank/world/topology/run-id handshake mismatch".into());
            }
        }
        // Two ranks use one duplex socket; this also pairs one RC queue pair.
        if ring.link.duplex() {
            if rank == 0 { left = Some(right.as_ref().unwrap().try_clone().map_err(|e| e.to_string())?); }
            else { right = Some(left.as_ref().unwrap().try_clone().map_err(|e| e.to_string())?); }
        }
        for stream in right.iter().chain(left.iter()) {
            stream.set_nodelay(true).map_err(|e| e.to_string())?;
            stream.set_nonblocking(true).map_err(|e| e.to_string())?;
        }
        ring.right = right;
        ring.left = left;
        ring.initialize_rdma(&mode)?;
        #[cfg(target_os = "linux")]
        if ring.link.is_pair() && ring.uses_rdma() { ring.initialize_auxiliary()?; }
        #[cfg(target_os = "linux")]
        if path && world > 2 && ring.uses_rdma() { ring.initialize_second_rails()?; }
        ring.setup_deadline = None;
        ring.stats = TransportStats::default();
        if collective::enabled() { ring.profile = Some(Box::default()); }
        Ok(ring)
    }

    #[cfg(target_os = "linux")]
    fn initialize_auxiliary(&mut self) -> Result<()> {
        let socket = self.right.as_ref().or(self.left.as_ref()).unwrap();
        self.auxiliary = Self::open_second_rail(socket, &mut self.stats, self.setup_deadline)?;
        eprintln!("seqpar_links rank={} rails={}", self.rank, 1 + usize::from(self.auxiliary.is_some()));
        Ok(())
    }

    /// Open the second rail of each path link, right link first so the chain
    /// of neighbour negotiations resolves from the ends.
    #[cfg(target_os = "linux")]
    fn initialize_second_rails(&mut self) -> Result<()> {
        for link in 0..2 {
            let Some(socket) = [self.right.as_ref(), self.left.as_ref()][link] else { continue; };
            self.second[link] = Self::open_second_rail(socket, &mut self.stats, self.setup_deadline)?;
        }
        eprintln!("seqpar_links rank={} rails_right={} rails_left={}", self.rank,
            1 + usize::from(self.second[0].is_some()), 1 + usize::from(self.second[1].is_some()));
        Ok(())
    }

    /// Agree with the peer on one more cable rail (a second subnet both sides
    /// share) and pair a queue pair over it; None when either side lacks it.
    #[cfg(target_os = "linux")]
    fn open_second_rail(socket: &TcpStream, stats: &mut TransportStats, deadline: Option<Instant>) -> Result<Option<rdma::Endpoint>> {
        let primary = socket.local_addr().map_err(|e| e.to_string())?.ip();
        let mut local = [0u8; 40];
        for (slot, link) in local.chunks_exact_mut(5).zip(crate::discovery::links()?.into_iter()
            .filter(|link| std::net::IpAddr::V4(link.address) != primary)) {
            slot[..4].copy_from_slice(&link.address.octets()); slot[4] = link.prefix;
        }
        let mut peer = [0u8; 40];
        pump(Some(socket), Some(socket), &local, &[], &mut peer, &mut [], stats, deadline)?;
        let candidate = local.chunks_exact(5).flat_map(|a| peer.chunks_exact(5).filter_map(move |b| {
            let (a_ip, b_ip) = (u32::from_be_bytes(a[..4].try_into().unwrap()), u32::from_be_bytes(b[..4].try_into().unwrap()));
            let mask = if a[4] == 0 { 0 } else { u32::MAX << (32 - a[4].min(32)) };
            (a_ip != 0 && b_ip != 0 && a_ip != b_ip && a[4] == b[4] && a[4] <= 32 && a_ip & mask == b_ip & mask)
                .then_some(((a_ip.min(b_ip), a_ip.max(b_ip)), std::net::Ipv4Addr::from(a_ip)))
        })).min_by_key(|&(key, _)| key).map(|(_, address)| address);
        let mut auxiliary = candidate.and_then(|address| rdma::Endpoint::open_address(address).ok());
        let mut available = [0u8; 1];
        pump(Some(socket), Some(socket), &[u8::from(auxiliary.is_some())], &[], &mut available, &mut [], stats, deadline)?;
        if available[0] == 0 { return Ok(None); }
        let Some(endpoint) = auxiliary.as_mut() else { return Ok(None); };
        let mut hello = [0u8; 24];
        pump(Some(socket), Some(socket), &endpoint.hello(), &[], &mut hello, &mut [], stats, deadline)?;
        endpoint.ready(&hello)?;
        Ok(auxiliary)
    }

    fn initialize_rdma(&mut self, mode: &str) -> Result<()> {
        #[allow(unused_mut)]
        let mut available = false;
        #[allow(unused_mut)]
        let mut reason = String::from("RDMA requires Linux and a direct QSFP interface");
        #[cfg(target_os = "linux")]
        if mode != "0" {
            let result = (|| {
                self.endpoints[0] = self.right.as_ref().map(rdma::Endpoint::open).transpose()?;
                if !self.link.duplex() { self.endpoints[1] = self.left.as_ref().map(rdma::Endpoint::open).transpose()?; }
                Ok::<_, String>(())
            })();
            match result { Ok(()) => available = true, Err(error) => reason = error }
        }
        let mut votes = Vec::new();
        votes.try_reserve_exact(self.world * 2).map_err(|e| e.to_string())?;
        votes.resize(self.world * 2, 0u8);
        votes[self.rank * 2] = u8::from(available);
        votes[self.rank * 2 + 1] = u8::from(mode == "1");
        let offsets = (0..=self.world).map(|r| r * 2).collect::<Vec<_>>();
        self.gather_tcp(&mut votes, &offsets, false)?;
        if votes.chunks_exact(2).any(|v| v[0] == 0) {
            #[cfg(target_os = "linux")]
            { self.endpoints = [None, None]; }
            if votes.chunks_exact(2).any(|v| v[1] != 0) {
                return Err(format!("RDMA required but unavailable on one or more ranks; local: {reason}"));
            }
            if mode != "0" {
                let reason = if available { "another rank selected TCP or could not open RDMA" } else { &reason };
                eprintln!("TRANSPORT rank={} fallback=tcp reason={reason}", self.rank);
            }
            return Ok(());
        }
        #[cfg(target_os = "linux")]
        {
            let right = self.endpoints[0].as_ref().map(|e| e.hello());
            let left = self.endpoints[1].as_ref().map(|e| e.hello());
            let (mut from_left, mut from_right) = ([0u8; 24], [0u8; 24]);
            if self.link.duplex() {
                pump(self.right.as_ref(), self.left.as_ref(), &right.unwrap(), &[], &mut from_left, &mut [], &mut self.stats, self.setup_deadline)?;
                self.endpoints[0].as_mut().unwrap().ready(&from_left)?;
            } else {
                pump(self.right.as_ref(), self.left.as_ref(), right.as_ref().map_or(&[], |h| h.as_slice()), left.as_ref().map_or(&[], |h| h.as_slice()),
                    if left.is_some() { &mut from_left } else { &mut [] }, if right.is_some() { &mut from_right } else { &mut [] }, &mut self.stats, self.setup_deadline)?;
                if let Some(endpoint) = &mut self.endpoints[0] { endpoint.ready(&from_right)?; }
                if let Some(endpoint) = &mut self.endpoints[1] { endpoint.ready(&from_left)?; }
            }
        }
        Ok(())
    }

}
