//! Copy gathers for logical tensor intervals. No dot product or reduction is split.
//! The layer driver must call each gather at its declared operation boundary.
use crate::Result;
use engine_model::plan::tensor_split::{Ownership,ProjectionRows};
use engine_fabric::exchange::Exchange;
use std::ops::Range;

fn need(ok: bool, why: &str) -> Result<()> { if ok { Ok(()) } else { Err(why.into()) } }
fn mul(a: usize, b: usize) -> Result<usize> { a.checked_mul(b).filter(|&n| n <= isize::MAX as usize).ok_or_else(|| "tensor wire product overflow".into()) }
fn size(n: u64) -> Result<usize> { usize::try_from(n).map_err(|_| "tensor wire dimension exceeds usize".into()) }

/// Rank-major wire packets, each in local row-major order. The global tensor
/// stays row-major even when the split axis follows token or expert axes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatherLayout {
    ownership: Ownership, element_bytes: usize, rows: usize, row_bytes: usize,
    spans: Vec<Range<usize>>, offsets: Vec<usize>, bytes: usize,
}
impl GatherLayout {
    /// `byte_limit` is an admission limit supplied by the caller. This checks
    /// one tensor and one wire buffer, not a complete model memory budget.
    pub fn new(ownership: Ownership, element_bytes: usize, byte_limit: usize) -> Result<Self> {
        need(element_bytes > 0, "tensor wire element size is empty")?;
        let (outer,axis,inner) = ownership.geometry().map_err(|e| e.to_string())?;
        let rows = size(outer)?;
        let stride = mul(size(inner)?,element_bytes)?;
        let row_bytes = mul(size(axis)?,stride)?;
        let bytes = mul(rows,row_bytes)?;
        need(bytes <= byte_limit, "tensor wire exceeds its byte limit")?;
        let mut spans = Vec::new();
        let mut offsets = vec![0usize];
        for owner in ownership.owners() {
            let span = mul(size(owner.interval.start)?,stride)?..mul(size(owner.interval.end)?,stride)?;
            let packet = mul(rows,span.len())?;
            let end = offsets.last().unwrap().checked_add(packet).ok_or("tensor wire offset overflow")?;
            spans.push(span); offsets.push(end);
        }
        need(offsets.last() == Some(&bytes), "tensor wire packets do not cover the tensor")?;
        Ok(Self { ownership,element_bytes,rows,row_bytes,spans,offsets,bytes })
    }
    pub fn ownership(&self) -> &Ownership { &self.ownership }
    pub fn bytes(&self) -> usize { self.bytes }
    pub fn element_bytes(&self) -> usize { self.element_bytes }
    pub fn offsets(&self) -> &[usize] { &self.offsets }
    pub fn packet(&self, rank: u32) -> Result<Range<usize>> {
        self.ownership.owner(rank).map_err(|e| e.to_string())?;
        let r = rank as usize; Ok(self.offsets[r]..self.offsets[r+1])
    }
    /// Copy a complete local tensor into its checked wire packet.
    pub fn pack(&self, rank: u32, local: &[u8], wire: &mut [u8]) -> Result<()> {
        let packet = self.packet(rank)?;
        need(wire.len() == self.bytes && local.len() == packet.len(), "tensor wire or local packet has the wrong byte extent")?;
        wire[packet].copy_from_slice(local); Ok(())
    }
    /// Copy packets to global coordinates. Arrival order never becomes tensor order.
    pub fn unpack(&self, wire: &[u8], global: &mut [u8]) -> Result<()> {
        need(wire.len() == self.bytes && global.len() == self.bytes, "tensor wire or global tensor has the wrong byte extent")?;
        for (rank,span) in self.spans.iter().enumerate() {
            for row in 0..self.rows {
                let from = self.offsets[rank]+row*span.len();
                let to = row*self.row_bytes+span.start;
                global[to..to+span.len()].copy_from_slice(&wire[from..from+span.len()]);
            }
        }
        Ok(())
    }
    /// The existing Exchange interface accepts these rank-major byte offsets.
    /// The caller must bind this layout and rank to the same checked group.
    /// A failed exchange leaves `global` unchanged. There is no reduce call.
    pub fn gather<E: Exchange>(&self, exchange: &mut E, rank: u32, local: &[u8], global: &mut [u8]) -> Result<()> {
        self.gather_with(rank,local,global,|wire,offsets| exchange.gather_bytes(wire,offsets))
    }
    /// A logical layer projection consumes one complete input tensor. Check K
    /// before exchange, then copy every feature into its global position.
    pub fn gather_projection_input<E: Exchange>(&self, projection: &ProjectionRows, exchange: &mut E,
        rank: u32, local: &[u8], global: &mut [u8]) -> Result<()> {
        need(self.ownership.elements() == projection.input_features(), "projection gather must cover the complete global input K")?;
        self.gather(exchange,rank,local,global)
    }
    /// Bind the concrete host ring's rank and group extent before communication.
    /// Equal ownership gives equal byte packets in this transport's all_gather.
    pub fn gather_ring(&self, ring: &mut crate::transport::Ring, local: &[u8], global: &mut [u8]) -> Result<()> {
        need(ring.world() == self.spans.len(), "tensor ownership and transport rank counts differ")?;
        self.gather_with(ring.rank() as u32,local,global,|wire,_| ring.all_gather(wire,1))
    }
    fn gather_with(&self, rank: u32, local: &[u8], global: &mut [u8], gather: impl FnOnce(&mut [u8],&[usize]) -> Result<()>) -> Result<()> {
        let packet = self.packet(rank)?;
        need(local.len() == packet.len() && global.len() == self.bytes, "tensor gather has the wrong local or global extent")?;
        let mut wire = Vec::new();
        wire.try_reserve_exact(self.bytes).map_err(|_| "tensor wire allocation failed")?;
        wire.resize(self.bytes,0);
        self.pack(rank,local,&mut wire)?;
        gather(&mut wire,&self.offsets)?;
        self.unpack(&wire,global)
    }
    /// Bind to the existing word-copy mailbox ABI. Packed BF16 is allowed only
    /// when every rank boundary covers whole words. No padding is invented.
    pub fn mailbox(&self, rank: u32, slot_bytes: usize) -> Result<MailboxRows> {
        self.packet(rank)?;
        let span = &self.spans[rank as usize];
        need(self.row_bytes.is_multiple_of(4) && span.start.is_multiple_of(4) && span.len().is_multiple_of(4),
            "tensor mailbox requires word-aligned row intervals")?;
        let packet_bytes = mul(self.rows,span.len())?;
        need(packet_bytes <= slot_bytes && packet_bytes <= u32::MAX as usize, "tensor interval exceeds the mailbox slot or length field")?;
        Ok(MailboxRows { ranks:self.spans.len(),rank:rank as usize,rows:self.rows,width:self.row_bytes/4,
            part:span.len()/4,own:span.start/4,words:self.bytes/4,packet_bytes })
    }
    /// Include the existing route header and slot constraints. A logical split
    /// can fit the host wire while the current device mailbox cannot serve it.
    pub fn mailbox_channels(&self, ring: bool) -> Result<spark_gpu_probe::tp_route::Channels> {
        use spark_gpu_probe::tp_route::{Channels,Links};
        let rows = self.mailbox(0,usize::MAX)?;
        Channels::new(&Links::new(self.spans.len(),ring)?,rows.packet_bytes/4)
    }
}

/// Exact arguments for tp_mailbox_put and tp_mailbox_take_from. Fields are
/// private so callers cannot change a checked tensor's origin or stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MailboxRows { ranks: usize, rank: usize, rows: usize, width: usize, part: usize, own: usize, words: usize, packet_bytes: usize }
impl MailboxRows {
    pub fn rank(&self) -> usize { self.rank }
    pub fn ranks(&self) -> usize { self.ranks }
    pub fn rows(&self) -> usize { self.rows }
    pub fn width(&self) -> usize { self.width }
    pub fn part(&self) -> usize { self.part }
    pub fn own(&self) -> usize { self.own }
    pub fn words(&self) -> usize { self.words }
    pub fn packet_bytes(&self) -> usize { self.packet_bytes }
    /// Global destination word for a packed packet word.
    pub fn destination(&self, word: usize) -> Result<usize> {
        need(word < self.packet_bytes/4, "tensor mailbox word is outside the packet")?;
        Ok((word/self.part)*self.width+self.own+word%self.part)
    }
    #[cfg(target_os="linux")]
    fn check(&self, mailbox: &spark_gpu_probe::driver::TpMailbox<'_>, vector: &spark_gpu_probe::driver::DeviceVector<'_>, seq: u64) -> Result<()> {
        use spark_gpu_probe::tp_mailbox_host::{SEND_LEN,SEND_SEQ,FAIL_OUT,FLAG_OUT,ERROR,RECV_SEQ};
        let layout = mailbox.layout();
        need(layout.world == self.ranks && layout.slots > 0 && layout.slots <= SEND_LEN-SEND_SEQ
            && layout.slots <= FAIL_OUT-FLAG_OUT && (self.ranks-1)*layout.slots <= ERROR-RECV_SEQ
            && layout.frame <= layout.slot_words, "tensor mailbox has an invalid route header")?;
        need(mailbox.slot_bytes() >= self.packet_bytes && vector.len() >= self.words && !vector.is_segmented(),
            "tensor mailbox binding has a different topology, slot, or vector extent")?;
        need(seq > 0 && !mailbox.failed(), "tensor mailbox sequence is empty or the mailbox has failed")
    }
    /// Queue only the copy. The device driver owns sequences, credits, stream
    /// order, completion, and failure cleanup. This does not admit a layer.
    #[cfg(target_os="linux")]
    pub fn put(&self, backend: &spark_gpu_probe::driver::Backend<'_>, mailbox: &spark_gpu_probe::driver::TpMailbox<'_>,
        vector: &spark_gpu_probe::driver::DeviceVector<'_>, seq: u64) -> Result<()> {
        self.check(mailbox,vector,seq)?;
        backend.tp_mailbox_put(mailbox,vector,self.rows,self.width,self.part,self.own,seq)
    }
    /// `self` describes the origin rank, not the destination rank. A receiver
    /// calls takes in global rank order at the logical gather boundary.
    #[cfg(target_os="linux")]
    pub fn take(&self, backend: &spark_gpu_probe::driver::Backend<'_>, mailbox: &spark_gpu_probe::driver::TpMailbox<'_>,
        vector: &mut spark_gpu_probe::driver::DeviceVector<'_>, receiver: usize, seq: u64, limit_ns: u64) -> Result<()> {
        self.check(mailbox,vector,seq)?;
        need(receiver < self.ranks && receiver != self.rank && limit_ns > 0 && limit_ns < 1<<47,
            "tensor mailbox receiver or timeout is invalid")?;
        need(vector.writable_from(0), "tensor mailbox destination must own writable storage")?;
        let channel = mailbox.layout().channel(self.rank,receiver);
        backend.tp_mailbox_take_from(mailbox,vector,self.rows,self.width,self.part,self.own,channel,seq,limit_ns)
    }
}
