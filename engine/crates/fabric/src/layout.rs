//! Link capabilities derived from a rank group and its existing topology.

/// Preserve the distinction between a pair and a pair sharing a duplex socket.
/// Some pair protocols also run on a path; socket sharing requires a ring.
#[derive(Clone, Copy, Debug)]
pub struct PeerLayout {
    pair: bool,
    path: bool,
}

impl PeerLayout {
    #[inline]
    pub fn new(world: usize, path: bool) -> Self {
        Self { pair: world == 2, path }
    }
    #[inline]
    pub fn is_pair(self) -> bool { self.pair }
    #[inline]
    pub fn duplex(self) -> bool { self.pair && !self.path }
    #[inline]
    pub fn is_path(self) -> bool { self.path }
}
