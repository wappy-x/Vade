//! Output owners pinned by the state views of short KDA prompt groups.
use engine_model::plan::{TextDescription,description::TokenMixer,layer::Topology};
use crate::Result;

pub(super) struct Tail {planes:Vec<(usize,usize)>}
impl Tail {
    pub(super) fn new(description:&TextDescription,topology:Topology)->Result<Self> {
        let mut planes=Vec::new();
        for layer in &description.layers {
            if !matches!(layer.mixer,TokenMixer::Kda(_)) {continue;}
            let layout=crate::kda_heads::layout(layer.mixer,topology)?;
            let row=layout.local_heads().checked_mul(layout.key_dim()).and_then(|n|n.checked_mul(4)).ok_or("kept recurrent row overflow")?;
            let state=row.checked_mul(layout.key_dim()).ok_or("kept recurrent state overflow")?;
            planes.push((state,row));
        }Ok(Self {planes})
    }
    pub(super) fn bytes(&self,tokens:usize)->usize {
        // Device::prompt copies the state prefix only above this group size.
        // A parked view can come from any short tail, including a resumed turn.
        let tokens=tokens.min(crate::bounded_decoder::STATE_VIEW_ROWS);
        self.planes.iter().try_fold(0usize,|sum,&(state,row)| {
            let owner=state.checked_add(row.checked_mul(tokens)?)?.checked_add(4095)?/4096*4096;
            sum.checked_add(owner.checked_sub(state)?)
        }).unwrap_or(usize::MAX)
    }
}
