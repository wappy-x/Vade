//! Kept rows on the queued-tile acks. The last rank's KEEP ack carries its tile's rows. Over
//! more ranks a middle's KEEP ack carries the whole kept tiles that came back to it so far,
//! oldest first, so the rows reach the front while later tiles still run, not only with the
//! final reply. The rows, their order and the tiles the front observes do not change.
use super::*;

/// What the ack of a queued tile carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ack {
    /// No rows.
    Empty,
    /// This tile's rows (the peer is the last rank).
    Rows(usize),
    /// Zero or more whole kept tiles, oldest first: a middle peer passes up the kept rows that
    /// came back to it so far, so they do not wait for the final reply.
    Tiles,
}

impl Link {
    /// One header owed by the oldest queued tile: a heartbeat (false) or its ack (true).
    pub(super) fn queued_frame(&mut self, heard: &mut Instant) -> Result<bool> {
        let Some((status, len)) = self.frame(self.replied.wrapping_add(1), heard)? else { return Ok(false) };
        self.pending -= 1;
        let expect_rows = self.acks.pop_front().ok_or("pipeline queued ack without a tile")?;
        if status != 0 {
            let mut text = vec![0u8; len.min(1 << 16)];
            read_rest(&mut self.stream, &mut text, None, self.limit)?;
            return Err(format!("pipeline back failed a queued tile: {}", peer_text(&text)));
        }
        match expect_rows {
            Ack::Empty if len != 0 => return Err("pipeline queued-tile ack carries rows".into()),
            Ack::Rows(n) if len != n * W * 4 => return Err("pipeline kept ack size mismatch".into()),
            Ack::Tiles if len % (W * 4) != 0 => return Err("pipeline kept ack size mismatch".into()),
            Ack::Empty => (),
            Ack::Tiles if len == 0 => (),
            Ack::Rows(_) | Ack::Tiles => {
                // Validate before consuming payload.
                self.check_kept(len / (W * 4), matches!(expect_rows, Ack::Rows(_)))?;
                let mut data = vec![0f32; len / 4];
                read_rest(&mut self.stream, f32_bytes_mut(&mut data), None, self.limit)?;
                self.fill_kept(data);
            }
        }
        Ok(true)
    }
    /// Whether an ack may carry `rows` kept rows: at most the rows owed, and on the front whole
    /// tiles from its oldest tile without rows (`one`: exactly one tile).
    pub(super) fn check_kept(&self, rows: usize, one: bool) -> Result<()> {
        if rows > self.owed { return Err("pipeline kept ack exceeds owed rows".into()); }
        if self.relay_kept.is_some() { return Ok(()); }
        let mut tiles = self.kept.iter().filter(|t| t.data.is_none());
        let first = tiles.next().ok_or("pipeline kept ack with no tile")?.rows;
        if first == rows { return Ok(()); }
        if one || first > rows { return Err("pipeline kept ack does not match the tile".into()); }
        let mut at = first;
        for t in tiles {
            if at == rows { break; }
            at = at.checked_add(t.rows).filter(|&n| n <= rows).ok_or("pipeline kept ack splits a tile")?;
        }
        if at == rows { Ok(()) } else { Err("pipeline kept ack exceeds the posted tiles".into()) }
    }
    /// Kept rows from an ack (check_kept passed): a middle adds them to the rows it relays; the
    /// front puts them in its oldest tiles without rows, whole tiles in post order.
    fn fill_kept(&mut self, data: Vec<f32>) {
        self.owed -= data.len() / W;
        if let Some(relay) = self.relay_kept.as_mut() {
            relay.extend_from_slice(&data);
            return;
        }
        let mut at = 0;
        for t in self.kept.iter_mut().filter(|t| t.data.is_none()) {
            if at == data.len() { break; }
            let end = at + t.rows * W;
            t.data = Some(data[at..end].to_vec());
            at = end;
        }
    }
    /// What the ack of a tile of `n` rows will carry (`keep`: a KEEP tile).
    pub(super) fn kept_ack(&self, keep: bool, n: usize) -> Ack {
        if !keep { Ack::Empty } else if self.inline_kept { Ack::Rows(n) } else if self.chain_kept { Ack::Tiles } else { Ack::Empty }
    }
}

#[cfg(test)]
#[path = "next_pp_kept_tests.rs"]
mod tests;
