//! Host preparation for complete-head sparse attention. No device execution is admitted here.
#![forbid(unsafe_code)]
use crate::Result;
use engine_model::plan::{TextDescription,description::TokenMixer,layer::{Split,Topology}};
use std::ops::Range;

fn need(ok:bool,why:&str)->Result<()> {if ok {Ok(())} else {Err(why.into())}}
fn product(left:usize,right:usize)->Result<usize> {
    left.checked_mul(right).filter(|&n|n<=isize::MAX as usize/4 && n<=u32::MAX as usize).ok_or_else(||"sparse head extent exceeds device ABI".into())
}

/// A packed expansion row is `[head, key_dim + value_dim]`, with keys before values.
/// State latents and indexer planes stay replicated. Only expanded rows own these heads.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Layout {topology:Topology,heads:usize,key:usize,value:usize,first:usize,local:usize}
impl Layout {
    pub fn new(heads:u64,key_dim:u64,value_dim:u64,topology:Topology)->Result<Self> {
        topology.validate().map_err(|e|e.to_string())?;
        need(heads>0 && key_dim>0 && value_dim>0,"sparse head dimensions must be positive")?;
        let (_,first,local)=topology.interval(&[heads],Split::Axis {axis:0,unit:1}).map_err(|e|e.to_string())?.ok_or("sparse heads require an axis split")?;
        let size=|n|usize::try_from(n).map_err(|_|"sparse head dimension exceeds address space");
        let out=Self {topology,heads:size(heads)?,key:size(key_dim)?,value:size(value_dim)?,first:size(first)?,local:size(local)?};
        let width=out.key.checked_add(out.value).ok_or("sparse head width overflow")?;
        product(out.heads,width)?;Ok(out)
    }
    pub fn topology(self)->Topology {self.topology}
    pub fn heads(self)->Range<usize> {self.first..self.first+self.local}
    pub fn total_heads(self)->usize {self.heads}
    pub fn local_heads(self)->usize {self.local}
    pub fn key_dim(self)->usize {self.key}
    pub fn value_dim(self)->usize {self.value}
    pub fn row_words(self)->usize {self.local*(self.key+self.value)}
    pub fn global_row_words(self)->usize {self.heads*(self.key+self.value)}
    pub fn projection_rows(self)->Range<usize> {self.first*(self.key+self.value)..(self.first+self.local)*(self.key+self.value)}
    pub fn query_rows(self)->Range<usize> {self.first*self.key..(self.first+self.local)*self.key}
    pub fn attended_rows(self)->Range<usize> {self.first*self.value..(self.first+self.local)*self.value}
    pub fn expanded_words(self,rows:usize)->Result<usize> {product(rows,self.row_words())}
    /// Exact page charge for a held allocation or a memo slab, without its host key.
    /// Held rows are stored packed, two BF16 patterns a word: two bytes a feature.
    pub fn allocation_bytes(self,rows:usize)->Result<usize> {
        let bytes=self.expanded_words(rows)?.checked_mul(2).ok_or("sparse head byte overflow")?;
        bytes.checked_add(4095).and_then(|n|n.checked_div(4096)?.checked_mul(4096)).filter(|&n|n<=isize::MAX as usize).ok_or_else(||"sparse head allocation overflow".into())
    }
    #[allow(clippy::manual_is_multiple_of)] // A zero divisor must still panic in `%`.
    fn copy_rows<T:Copy>(self,source:&[T],width:usize)->Result<Vec<T>> {
        let full=product(self.heads,width)?;need(source.len()%full==0,"sparse head source row extent differs")?;
        let rows=source.len()/full;let local=product(self.local,width)?;let mut out=Vec::with_capacity(product(rows,local)?);
        for row in source.chunks_exact(full) {out.extend_from_slice(&row[self.first*width..(self.first+self.local)*width]);}Ok(out)
    }
    /// Host twin for a replicated-to-owned cache handoff. Copies raw elements only.
    pub fn local_expanded<T:Copy>(self,source:&[T])->Result<Vec<T>> {self.copy_rows(source,self.key+self.value)}
    pub fn local_queries<T:Copy>(self,source:&[T])->Result<Vec<T>> {self.copy_rows(source,self.key)}
}

/// Checkpoint-derived ownership, including any appended latent predictor layer.
pub fn layouts(description:&TextDescription,topology:Topology)->Result<Vec<(usize,Layout)>> {
    topology.validate().map_err(|e|e.to_string())?;
    description.layers.iter().enumerate().filter_map(|(layer,spec)|match spec.mixer {
        TokenMixer::LatentAttention(a)=>Some(Layout::new(a.heads,a.key_dim,a.value_dim,topology).map(|layout|(layer,layout))),_=>None,
    }).collect()
}

/// Host twin for the attended gather. Rank packets contain `[token, local_head, value]`.
/// The result is `[token, global_head, value]`; no floating-point operation is used.
pub fn gather_attended<T:Copy>(parts:&[(Layout,&[T])],tokens:usize)->Result<Vec<T>> {
    let first=parts.first().ok_or("sparse head gather is empty")?.0;
    need(parts.len()==first.topology.parts as usize,"sparse head gather has missing ranks")?;
    let full=product(first.heads,first.value)?;let mut out=Vec::with_capacity(product(tokens,full)?);
    for (rank,(layout,words)) in parts.iter().enumerate() {
        need(layout.topology.rank as usize==rank && layout.topology.parts==first.topology.parts && layout.heads==first.heads
            && layout.key==first.key && layout.value==first.value,"sparse head gather ownership differs")?;
        need(words.len()==product(tokens,layout.local*layout.value)?,"sparse head gather packet extent differs")?;
    }
    for token in 0..tokens {for (layout,words) in parts {let width=layout.local*layout.value;out.extend_from_slice(&words[token*width..(token+1)*width]);}}
    Ok(out)
}
