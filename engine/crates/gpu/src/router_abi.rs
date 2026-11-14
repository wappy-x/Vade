//! Fixed corrected-sigmoid router geometry and disjoint device spans.
pub const MAX_EXPERTS: u32 = 1024;
pub const MAX_GROUPS: u32 = MAX_EXPERTS / 2;
pub const ROUTER_THREADS: u32 = 128;
pub const ROUTER_SHARED_BYTES: usize = (2 * MAX_EXPERTS + 2 * MAX_GROUPS) as usize * 4;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrectedSigmoidConfig {
    pub experts: u32, pub selected: u32, pub groups: u32, pub selected_groups: u32,
    pub normalize: u32, pub scale: f32,
}
impl CorrectedSigmoidConfig {
    pub fn checked(experts: u32, selected: u32, groups: u32, selected_groups: u32, normalize: bool, scale: f32) -> Option<Self> {
        if experts == 0 || experts > MAX_EXPERTS || groups == 0 || groups > experts || experts % groups != 0
            || experts / groups < 2 || selected_groups == 0 || selected_groups > groups
            || selected == 0 || selected > selected_groups * (experts / groups) || !scale.is_finite() || scale <= 0.0 { return None; }
        Some(Self { experts, selected, groups, selected_groups, normalize: u32::from(normalize), scale })
    }
    pub fn valid(self) -> bool {
        self.normalize <= 1 && Self::checked(self.experts,self.selected,self.groups,self.selected_groups,self.normalize != 0,self.scale).is_some()
    }
}

/// One block per token. IDs are u32 words; weights and input rows are FP32.
/// Status is sticky. The caller clears it at the start of a forward.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CorrectedSigmoidRoute {
    pub config: CorrectedSigmoidConfig,
    pub logits: u64, pub bias: u64, pub ids: u64, pub weights: u64, pub status: u64,
    pub tokens: u32,
}
impl CorrectedSigmoidRoute {
    /// Check geometry, byte arithmetic, alignment, and every pair of spans.
    /// The owner must also check allocation size, context, and lifetime.
    pub fn checked(self) -> Option<Self> {
        if !self.config.valid() || self.tokens == 0 || self.tokens > 65535 { return None; }
        let (t,e,k) = (self.tokens as u64,self.config.experts as u64,self.config.selected as u64);
        let spans = [(self.logits,t*e),(self.bias,e),(self.ids,t*k),(self.weights,t*k),(self.status,t)];
        for (i,&(a,n)) in spans.iter().enumerate() {
            if a == 0 || a % 4 != 0 { return None; }
            let end = a.checked_add(n.checked_mul(4)?)?;
            for &(b,m) in &spans[..i] { if a < b.checked_add(m.checked_mul(4)?)? && b < end { return None; } }
        }
        Some(self)
    }
}

/// BF16 down rows in router-slot order, reduced across ranks before this fold.
/// Shared is a separate completed down row. No gate or residual is applied.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct UngatedMoeFinish {
    pub ids: u64, pub weights: u64, pub partials: u64, pub shared: u64, pub output: u64, pub status: u64,
    pub selected: u32, pub hidden: u32, pub tokens: u32,
}
impl UngatedMoeFinish {
    /// IDs must also be unique and weights finite in each successful row.
    /// The owner must check allocation size, context, and lifetime.
    pub fn checked(self) -> Option<Self> {
        if self.selected == 0 || self.selected > MAX_EXPERTS || self.hidden == 0 || self.hidden % ROUTER_THREADS != 0
            || self.tokens == 0 || self.tokens > 65535 { return None; }
        let (t,h,k) = (self.tokens as u64,self.hidden as u64,self.selected as u64);
        let spans = [(self.ids,t*k),(self.weights,t*k),(self.partials,t.checked_mul(k)?.checked_mul(h)?),
            (self.shared,t*h),(self.output,t*h),(self.status,t)];
        // The one-dimensional output grid must fit a u32 block count.
        if (t*h).div_ceil(ROUTER_THREADS as u64) > u32::MAX as u64 { return None; }
        for (i,&(a,n)) in spans.iter().enumerate() {
            if a == 0 || a % 4 != 0 { return None; }
            let end = a.checked_add(n.checked_mul(4)?)?;
            for &(b,m) in &spans[..i] { if a < b.checked_add(m.checked_mul(4)?)? && b < end { return None; } }
        }
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn p() -> CorrectedSigmoidRoute {
        CorrectedSigmoidRoute { config: CorrectedSigmoidConfig::checked(4,2,2,1,true,2.5).unwrap(),
            logits: 0x1000,bias: 0x2000,ids: 0x3000,weights: 0x4000,status: 0x5000,tokens: 3 }
    }
    #[test]
    fn descriptor_bytes_and_offsets() {
        use core::mem::{size_of,align_of,offset_of};
        assert_eq!((size_of::<CorrectedSigmoidConfig>(),align_of::<CorrectedSigmoidConfig>()),(24,4));
        assert_eq!((size_of::<CorrectedSigmoidRoute>(),align_of::<CorrectedSigmoidRoute>()),(72,8));
        assert_eq!([offset_of!(CorrectedSigmoidRoute,config),offset_of!(CorrectedSigmoidRoute,logits),
            offset_of!(CorrectedSigmoidRoute,bias),offset_of!(CorrectedSigmoidRoute,ids),offset_of!(CorrectedSigmoidRoute,weights),
            offset_of!(CorrectedSigmoidRoute,status),offset_of!(CorrectedSigmoidRoute,tokens)],[0,24,32,40,48,56,64]);
        assert_eq!(ROUTER_SHARED_BYTES,12288); assert!(p().checked().is_some());
    }
    #[test]
    fn rejects_bad_geometry_and_byte_spans() {
        for (e,k,g,keep) in [(0,1,1,1),(1025,1,1,1),(4,0,1,1),(4,5,1,1),(4,1,0,1),
            (4,1,3,1),(4,1,4,1),(4,1,2,0),(4,1,2,3),(4,3,2,1)] {
            assert!(CorrectedSigmoidConfig::checked(e,k,g,keep,true,1.0).is_none());
        }
        for scale in [0.0,-1.0,f32::NAN,f32::INFINITY] { assert!(CorrectedSigmoidConfig::checked(4,2,1,1,true,scale).is_none()); }
        assert!(CorrectedSigmoidConfig::checked(1024,1024,512,512,false,0.5).is_some());
        for bad in [CorrectedSigmoidRoute { tokens: 0,..p() },CorrectedSigmoidRoute { tokens: 65536,..p() },
            CorrectedSigmoidRoute { ids: 0,..p() },CorrectedSigmoidRoute { ids: 1,..p() },
            CorrectedSigmoidRoute { ids: u64::MAX-3,..p() },CorrectedSigmoidRoute { weights: 0x3014,..p() },
            CorrectedSigmoidRoute { bias: 0x102c,..p() },CorrectedSigmoidRoute { status: 0x4000,..p() },
            CorrectedSigmoidRoute { config: CorrectedSigmoidConfig { normalize: 2,..p().config },..p() }] {
            assert!(bad.checked().is_none(),"{bad:?}");
        }
        // Touching ends are disjoint. Logits have 3*4*4 bytes.
        assert!(CorrectedSigmoidRoute { bias: 0x1030,..p() }.checked().is_some());
    }
    #[test]
    fn ungated_finish_bytes_and_spans() {
        use core::mem::{size_of,align_of,offset_of};
        assert_eq!((size_of::<UngatedMoeFinish>(),align_of::<UngatedMoeFinish>()),(64,8));
        assert_eq!([offset_of!(UngatedMoeFinish,ids),offset_of!(UngatedMoeFinish,weights),offset_of!(UngatedMoeFinish,partials),
            offset_of!(UngatedMoeFinish,shared),offset_of!(UngatedMoeFinish,output),offset_of!(UngatedMoeFinish,status),
            offset_of!(UngatedMoeFinish,selected),offset_of!(UngatedMoeFinish,hidden),offset_of!(UngatedMoeFinish,tokens)],
            [0,8,16,24,32,40,48,52,56]);
        let p = UngatedMoeFinish { ids: 0x1000,weights: 0x2000,partials: 0x3000,shared: 0x5000,output: 0x6000,status: 0x7000,
            selected: 3,hidden: 128,tokens: 2 };
        assert!(p.checked().is_some());
        for bad in [UngatedMoeFinish { selected: 0,..p },UngatedMoeFinish { selected: 1025,..p },
            UngatedMoeFinish { hidden: 0,..p },UngatedMoeFinish { hidden: 129,..p },UngatedMoeFinish { tokens: 0,..p },
            UngatedMoeFinish { tokens: 65536,..p },UngatedMoeFinish { output: 0x5000,..p },
            UngatedMoeFinish { shared: 0x3bfc,..p },UngatedMoeFinish { status: 0,..p },
            UngatedMoeFinish { output: u64::MAX-3,..p },UngatedMoeFinish { tokens: 65535,hidden: u32::MAX-127,..p }] {
            assert!(bad.checked().is_none(),"{bad:?}");
        }
    }
}
