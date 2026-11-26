//! The weights of a model folder as host words: the weight calls of the driver for a run on the host backend.
//!
//! `HostWeights` reads each stored tensor through the file reader of the whole-model host form of the model crate
//! (`engine_model::loop_host::Folder`) and puts it into the layout of its `Weight` name (`mix.rs`). A dense weight
//! is read at the start and held as 32-bit words. An expert matrix is decoded when a visit picks it, by the one
//! decode of the model crate (`expert_widen`): the BF16 values of form A1 of the expert arithmetic. Decoded
//! matrices are kept up to a byte limit. The limit changes no word.
//!
//! This is a host path. A product runs on the backend with the matrix as host words (`mix::chain_product`). It is
//! the run of the driver on real weights that the whole-model host form is compared with, and it can serve a chat
//! on a CPU. It is not the device path: the operands of a loaded checkpoint (`operands`) keep each weight on the
//! device.
//!
//! Stored layouts that this file turns into the layouts of `Weight`. The stored names and layouts are those of the
//! public checkpoint of the family (Transformers v5.24.0, `models/inkling2_loop/`; read for the names and the
//! layouts, no code copied):
//! - Fixed convolution taps are `[channels, 1, taps]`, and `[taps, channels]` where a projection also makes taps
//!   from the token (reference `modeling_inkling2_loop.py` lines 285-455). The description says which stored tap
//!   meets the current token. The driver reads a channel, then the lag, and lag 0 meets the current token.
//! - The projection that makes taps has the stored row `head * taps + tap`. The driver reads `head * taps + lag`.
//! - The two message query tensors hold two rows a loop layer: (query, expert) and (key, value) (lines 145-173).
//!   The driver reads the order of `Message`.
//! - A stored gate-and-up matrix mixes its rows (`storage_groups::GATE_UP_ROWS`). The driver reads the gate rows
//!   first, then the up rows.
//! - `attn_queries.i` belongs to layer `i + 1` (lines 828-853 and 1022-1062).
use super::mix::{Message,Weight,Weights,chain_product,need};
use crate::{Result,bounded_decoder::Compute};
use engine_model::{expert_widen::{Stored,StoredMatrix,widen_values},loop_host::{Folder,Source,open_folder},
    plan::{description::{ConvTaps,LoopTextDescription,TapOrder},storage_groups::{ExpertRows,GATE_UP_ROWS}},safetensors::Dtype,shapes::Ffn};
use std::{collections::BTreeMap,path::Path,sync::{Arc,Mutex,MutexGuard}};

/// The first part of every stored name of the text model.
const TEXT:&str="model.llm.";
fn told<E:std::fmt::Display>(error:E)->String {format!("loop decoder host weights: {error}")}
fn size(value:u64)->Result<usize> {usize::try_from(value).map_err(told)}

/// A whole BF16 or 32-bit tensor of the weight files as 32-bit words. Widening changes no value.
fn floats(source:&dyn Source,name:&str)->Result<Vec<f32>> {
    let (dtype,shape)=source.tensor(name).map_err(told)?;
    let width=match dtype {Dtype::Bf16=>2,Dtype::F32=>4,other=>return Err(told(format!("{name} has the type {other:?}, not BF16 or 32-bit")))};
    let bytes=shape.iter().try_fold(width,|n:usize,&d|n.checked_mul(usize::try_from(d).ok()?)).ok_or_else(||told(format!("{name} is too large for this host")))?;
    let mut stored=vec![0u8;bytes];source.read(name,0,&mut stored).map_err(told)?;
    let words:Vec<f32>=if width==2 {stored.as_chunks::<2>().0.iter().map(|b|f32::from_bits(u32::from(u16::from_le_bytes(*b))<<16)).collect()}
        else {stored.as_chunks::<4>().0.iter().map(|b|f32::from_le_bytes(*b)).collect()};
    if words.iter().any(|v|!v.is_finite()) {return Err(told(format!("{name} holds a value that is not finite")));}
    Ok(words)
}
/// Fixed taps by lag: a channel, then the lag, and lag 0 meets the current token. A stored row is a channel
/// (`[channels, 1, taps]`) or a tap (`[taps, channels]`).
fn by_lag(stored:&[f32],channels:usize,taps:usize,order:TapOrder,row_is_tap:bool)->Result<Vec<f32>> {
    need(taps>0 && stored.len()==channels*taps,"stored taps differ from the description")?;
    Ok((0..channels*taps).map(|i|{
        let (channel,lag)=(i/taps,i%taps);let tap=match order {TapOrder::OldestFirst=>taps-1-lag,TapOrder::NewestFirst=>lag};
        if row_is_tap {stored[tap*channels+channel]} else {stored[channel*taps+tap]}
    }).collect())
}
/// Where the stored codes of one matrix kind of the experts of one layer are.
struct Stack {name:String,packed:bool,experts:usize,rows:usize,columns:usize,code_bytes:usize,scale_bytes:usize,amax:Vec<f32>,denominator:f32}
impl Stack {
    fn new(source:&dyn Source,name:String,experts:usize,rows:usize,columns:usize)->Result<Self> {
        let (dtype,shape)=source.tensor(&name).map_err(told)?;let (_,scale)=source.tensor(&format!("{name}_scale")).map_err(told)?;
        // The stored type of the codes tells the two formats apart: two 4-bit codes a byte, or one 8-bit float a value.
        let packed=match dtype {Dtype::U8=>true,Dtype::F8E4M3=>false,other=>return Err(told(format!("{name} has the type {other:?}, not expert codes")))};
        let stored_columns=if packed {columns/2} else {columns};
        need(shape==[experts as u64,rows as u64,stored_columns as u64] && scale.len()==3 && scale[..2]==shape[..2],"an expert stack differs from the description")?;
        let (amax,denominator)=if packed {(floats(source,&format!("{name}_amax"))?,floats(source,&format!("{name}_outer_scale_denominator"))?)} else {(Vec::new(),vec![0.0])};
        need((!packed || amax.len()==experts) && denominator.len()==1,"the second scales of an expert stack differ from its expert count")?;
        let (code_bytes,scale_bytes)=(rows.checked_mul(stored_columns),rows.checked_mul(size(scale[2])?));
        let (Some(code_bytes),Some(scale_bytes))=(code_bytes,scale_bytes) else {return Err(told("an expert stack is too large for this host"));};
        Ok(Self {name,packed,experts,rows,columns,code_bytes,scale_bytes,amax,denominator:denominator[0]})
    }
    /// One expert's matrix as BF16 values, `[rows, columns]` in stored row order.
    fn decode(&self,source:&dyn Source,expert:usize)->Result<Vec<f32>> {
        need(expert<self.experts,"an expert number is outside its stack")?;
        let (mut codes,mut scales)=(vec![0u8;self.code_bytes],vec![0u8;self.scale_bytes]);
        source.read(&self.name,(expert*self.code_bytes) as u64,&mut codes).map_err(told)?;
        source.read(&format!("{}_scale",self.name),(expert*self.scale_bytes) as u64,&mut scales).map_err(told)?;
        let data=if self.packed {Stored::Packed4 {codes:&codes,scales:&scales,amax:self.amax[expert],denominator:self.denominator}} else {Stored::Float8 {values:&codes,scales:&scales}};
        widen_values(&StoredMatrix {rows:self.rows,columns:self.columns,data}).map_err(told)
    }
}
/// Decoded expert matrices by layer, expert and kind (`true` = the down matrix). The entry that was used longest
/// ago leaves first.
#[derive(Default)]
struct Kept {held:BTreeMap<(usize,usize,bool),(Arc<Vec<f32>>,u64)>,bytes:usize,tick:u64,decoded:u64,hits:u64}
/// What the expert cache did so far, and the size of the dense weights.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Counts {
    /// Expert matrices decoded from their stored codes.
    pub decoded:u64,
    /// Products that found their matrix in the cache.
    pub hits:u64,
    /// Decoded matrices held now, and their bytes.
    pub held:usize,
    pub held_bytes:usize,
    /// 32-bit words of the dense weights.
    pub dense_words:usize,
}
/// The weights of one model folder for the driver on the host backend.
pub struct HostWeights {source:Folder,dense:BTreeMap<Weight,Vec<f32>>,stacks:Vec<(Stack,Stack)>,inner:Vec<usize>,width:usize,limit:usize,kept:Mutex<Kept>}
impl HostWeights {
    /// Read the description of the model folder `root` and every dense weight of its text model. The experts stay in
    /// the weight file. `expert_cache_bytes` is the most bytes of decoded expert matrices that stay in memory; with 0
    /// none is kept. It is a host resource limit and changes no word.
    pub fn open(root:&Path,expert_cache_bytes:usize)->Result<(LoopTextDescription,Self)> {
        let (d,_,source)=open_folder(root).map_err(told)?;let h=size(d.hidden)?;
        let read=|name:String|floats(&source,&format!("{TEXT}{name}"));
        let mut dense=BTreeMap::new();
        dense.insert(Weight::Embedding,read("embed.weight".into())?);
        dense.insert(Weight::EmbeddingNorm,read("embed_norm.weight".into())?);
        dense.insert(Weight::LastNorm,read("norm.weight".into())?);
        let mut head=read("unembed.weight".into())?;
        let head_words=size(d.head_rows)?.checked_mul(h).filter(|&words|words<=head.len()).ok_or_else(||told("the stored head has fewer rows than the model can write"))?;
        head.truncate(head_words);dense.insert(Weight::Head,head);
        dense.insert(Weight::MixOutput,read("attn_res.output_query".into())?);
        let order_of=|taps:ConvTaps|match taps {ConvTaps::Fixed {order}|ConvTaps::FixedAndToken {order,..}=>order};
        let stream_taps=size(d.conv.kernel)?;
        let pairs=if d.layers.iter().any(|layer|layer.message_gates) {Some((read("middle_cycle_cross_pass_queries".into())?,read("middle_cycle_cross_pass_kv_queries".into())?))} else {None};
        let (mut messages,mut loop_layer,mut stacks,mut inner)=(Vec::new(),0usize,Vec::new(),Vec::new());
        for (i,layer) in d.layers.iter().enumerate() {
            let a=layer.attention;let Ffn::Experts(e)=layer.ffn else {return Err(told(format!("layer {i} has no routed experts")));};
            let (kv,taps)=(size(a.kv_heads*a.head_dim)?,size(a.key_value_conv.kernel)?);
            let part=|name:&str|read(format!("layers.{i}.{name}"));
            for (weight,name) in [(Weight::Query(i),"attn.wq_du.weight"),(Weight::Key(i),"attn.wk_dv.weight"),(Weight::Value(i),"attn.wv_dv.weight"),
                (Weight::Relative(i),"attn.wr_du.weight"),(Weight::Sink(i),"attn.attn_sink_proj.weight"),(Weight::OutputGate(i),"attn.gate_attn.weight"),
                (Weight::Output(i),"attn.wo_ud.weight"),(Weight::QueryClip(i),"attn.q_l2_clip.scale"),(Weight::RelativeClip(i),"attn.rel_l2_clip.scale"),
                (Weight::RelativeTable(i),"attn.rel_logits_proj.proj"),(Weight::ExpertNorm(i),"mlp_norm.weight"),(Weight::Router(i),"mlp.gate.weight"),
                (Weight::RouterBias(i),"mlp.gate.bias")] {dense.insert(weight,part(name)?);}
            // A layer with one attention norm answers all three names with it.
            for (weight,input) in [(Weight::QueryNorm(i),"q"),(Weight::KeyNorm(i),"k"),(Weight::ValueNorm(i),"v")] {
                dense.insert(weight,part(&if a.separate_inputs {format!("{input}_input_norm.weight")} else {"attn_norm.weight".into()})?);
            }
            for (fixed,made,stream) in [(Weight::KeyTaps(i),Weight::KeyTapProjection(i),"k"),(Weight::ValueTaps(i),Weight::ValueTapProjection(i),"v")] {
                match a.key_value_conv.taps {
                    ConvTaps::Fixed {order}=>{dense.insert(fixed,by_lag(&part(&format!("attn.{stream}_sconv.weight"))?,kv,taps,order,false)?);},
                    ConvTaps::FixedAndToken {order,..}=>{
                        dense.insert(fixed,by_lag(&part(&format!("attn.{stream}_sconv.static_w"))?,kv,taps,order,true)?);
                        let stored=part(&format!("attn.{stream}_sconv.weight_proj"))?;let heads=size(a.kv_heads)?;
                        need(stored.len()==heads*taps*h,"a stored tap projection differs from the description")?;
                        let mut rows=Vec::with_capacity(stored.len());
                        for head in 0..heads {for lag in 0..taps {
                            let tap=match order {TapOrder::OldestFirst=>taps-1-lag,TapOrder::NewestFirst=>lag};
                            rows.extend_from_slice(&stored[(head*taps+tap)*h..(head*taps+tap+1)*h]);
                        }}
                        dense.insert(made,rows);
                    },
                }
            }
            dense.insert(Weight::ConvAfterAttention(i),by_lag(&part("attn_sconv.weight")?,h,stream_taps,order_of(d.conv.taps),false)?);
            dense.insert(Weight::ConvAfterExperts(i),by_lag(&part("mlp_sconv.weight")?,h,stream_taps,order_of(d.conv.taps),false)?);
            if layer.attention_queries>0 {
                need(i>0,"the first layer has no stored attention mix query")?;
                dense.insert(Weight::MixAttention(i),read(format!("attn_res.attn_queries.{}",i-1))?);
            }
            dense.insert(Weight::MixExperts(i),read(format!("attn_res.mlp_queries.{i}"))?);
            if layer.message_gates {
                for (message,gate) in [(Message::Query,"attn"),(Message::Key,"k"),(Message::Value,"v"),(Message::Expert,"mlp")] {
                    dense.insert(Weight::GateDown(i,message),part(&format!("controllers.{gate}.down_weight"))?);
                    dense.insert(Weight::GateForget(i,message),part(&format!("controllers.{gate}.forget_weight"))?);
                    dense.insert(Weight::GateGain(i,message),part(&format!("controllers.{gate}.gain_weight"))?);
                }
                let (first,second)=pairs.as_ref().ok_or_else(||told("the message queries are absent"))?;
                need(first.len()>=(2*loop_layer+2)*h && second.len()>=(2*loop_layer+2)*h,"the stored message queries differ from the loop range")?;
                // In the order of `Message`: query, key, value, expert.
                for (tensor,row) in [(first,2*loop_layer),(second,2*loop_layer),(second,2*loop_layer+1),(first,2*loop_layer+1)] {messages.extend_from_slice(&tensor[row*h..(row+1)*h]);}
                loop_layer+=1;
            }
            let (count,width)=(size(e.count)?,size(e.width)?);
            stacks.push((Stack::new(&source,format!("{TEXT}layers.{i}.mlp.experts.w13_weight"),count,2*width,h)?,
                Stack::new(&source,format!("{TEXT}layers.{i}.mlp.experts.w2_weight"),count,h,width)?));
            inner.push(width);
        }
        if pairs.is_some() {dense.insert(Weight::Messages,messages);}
        Ok((d,Self {source,dense,stacks,inner,width:h,limit:expert_cache_bytes,kept:Mutex::new(Kept::default())}))
    }
    fn lock(&self)->Result<MutexGuard<'_,Kept>> {self.kept.lock().map_err(|_|told("the expert cache lock is poisoned"))}
    /// What the expert cache did so far.
    pub fn counts(&self)->Result<Counts> {
        let kept=self.lock()?;
        Ok(Counts {decoded:kept.decoded,hits:kept.hits,held:kept.held.len(),held_bytes:kept.bytes,dense_words:self.dense.values().map(Vec::len).sum()})
    }
    /// One expert matrix in the layout of its `Weight` name, from the cache or from its stored codes.
    /// Two threads can decode the same matrix at the same time. Both get the same numbers.
    fn expert(&self,layer:usize,expert:usize,down:bool)->Result<Arc<Vec<f32>>> {
        let key=(layer,expert,down);
        {
            let mut kept=self.lock()?;kept.tick+=1;let tick=kept.tick;
            if let Some(entry)=kept.held.get_mut(&key) {entry.1=tick;let matrix=entry.0.clone();kept.hits+=1;return Ok(matrix);}
        }
        let (gate_up,down_stack)=self.stacks.get(layer).ok_or_else(||told("a weight names a layer outside the model"))?;
        let matrix=if down {down_stack.decode(&self.source,expert)?} else {
            let (stored,inner,h)=(gate_up.decode(&self.source,expert)?,self.inner[layer],self.width);
            // The driver reads the gate rows first, then the up rows.
            let row_of=|kind:usize,row:usize|match GATE_UP_ROWS {ExpertRows::Interleaved=>2*row+kind};
            let mut rows=Vec::with_capacity(stored.len());
            for kind in 0..2 {for row in 0..inner {rows.extend_from_slice(&stored[row_of(kind,row)*h..(row_of(kind,row)+1)*h]);}}
            rows
        };
        let (bytes,matrix)=(matrix.len()*std::mem::size_of::<f32>(),Arc::new(matrix));
        let mut kept=self.lock()?;kept.decoded+=1;
        if bytes<=self.limit {
            kept.tick+=1;let tick=kept.tick;
            if kept.held.insert(key,(matrix.clone(),tick)).is_none() {kept.bytes+=bytes;}
            while kept.bytes>self.limit {
                let Some(oldest)=kept.held.iter().min_by_key(|(_,entry)|entry.1).map(|(key,_)|*key) else {break};
                if let Some(gone)=kept.held.remove(&oldest) {kept.bytes-=gone.0.len()*std::mem::size_of::<f32>();}
            }
        }
        Ok(matrix)
    }
}
impl<C:Compute> Weights<C> for HostWeights {
    type RoutePlan = ();
    fn words(&self,weight:Weight)->Result<&[f32]> {
        self.dense.get(&weight).map(Vec::as_slice).ok_or_else(||told(format!("{weight:?} has no host words in this model folder (an expert matrix is read through its product)")))
    }
    fn product(&self,compute:&C,weight:Weight,x:&C::Value)->Result<C::Value> {
        match weight {
            Weight::GateUp(layer,expert)=>chain_product(compute,&self.expert(layer,expert,false)?,x),
            Weight::Down(layer,expert)=>chain_product(compute,&self.expert(layer,expert,true)?,x),
            other=>chain_product(compute,<Self as Weights<C>>::words(self,other)?,x),
        }
    }
}
