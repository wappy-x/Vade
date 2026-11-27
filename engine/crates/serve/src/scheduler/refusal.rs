//! Close a refused request without a connection reset.
//!
//! The server can refuse a request before it has read all of its bytes: a bad
//! head, an unknown route, a body over the limit, a shutdown. If it then drops
//! the socket while request bytes are unread, or while more are on the way,
//! Linux answers the peer with a reset. A reset can destroy the reply before
//! the client reads it. So a refused connection closes in two steps: the reply
//! and the write half first, then the socket.
use super::{Limits, Reading, Socket, Wire};
use std::time::{Duration, Instant};

/// Longest wait for the peer to close after a refusal. The connection's own
/// read deadline can end the wait sooner.
const LINGER: Duration = Duration::from_secs(1);

impl Socket {
    fn shutdown_write(&self) -> std::io::Result<()> {
        match self {
            Self::Tcp(socket) => socket.shutdown(std::net::Shutdown::Write),
            Self::Admin(socket) => socket.shutdown(std::net::Shutdown::Write),
        }
    }
}

impl Wire {
    /// Call after the reply is fully sent. Returns true when the socket may be
    /// dropped. The first call closes the write half, so the client reads the
    /// reply and then the end of the stream. Each call reads and discards the
    /// rest of the request. The wait ends when the peer closes, when `until`
    /// or the linger time passes, or when more than `limit` bytes are discarded.
    fn finish_refused(&mut self, until: Instant, limit: usize) -> bool {
        if self.linger.is_none() {
            // A failed half-close means the peer is gone: no reply is left to protect.
            if self.socket.shutdown_write().is_err() {
                return true;
            }
            self.linger = Some((until.min(Instant::now() + LINGER), 0));
        }
        let Some((deadline, discarded)) = &mut self.linger else { return true; };
        let mut buffer = [0; 8192];
        loop {
            match self.socket.read(&mut buffer) {
                Ok(0) => return true,
                Ok(count) => {
                    *discarded = discarded.saturating_add(count);
                    if *discarded > limit {
                        return true;
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Instant::now() >= *deadline,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => return true,
            }
        }
    }
}

impl Reading {
    /// Call after the reply is fully sent. Returns true when the connection
    /// may be dropped.
    ///
    /// Two cases close at once, as before: a request that was read whole, and
    /// a reply on a wire that the dispatcher handed back (it holds no request
    /// bytes). A request that was refused part way closes in two steps.
    pub(super) fn replied(&mut self, limits: &Limits) -> bool {
        let whole = self.head.as_ref().is_some_and(|(end, head)| self.data.len() == end + head.length);
        if self.data.is_empty() || whole {
            return true;
        }
        self.wire.finish_refused(self.deadline, limits.header_bytes.saturating_add(limits.body_bytes))
    }
}
