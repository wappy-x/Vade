//! A small made-up checkpoint for the driver: every size is small and comes from its own data. The tests of the
//! driver run on it, and a check program can run the driver on it on a device against the host run of the driver,
//! with no weight file and no load.
//!
//! `Fixture::new` takes a config text: `SMALL`, `WIDE` or another one. A config that the fixture cannot make is a
//! refusal that says why. No call of this file panics.
//!
//! A fixture is a config text and one tensor for each stored name that the text model reads. The config goes
//! through the same adapter as a released config, so the description has the same type and the same rules.
//! The names, the types and the shapes are those that the name index of the model crate expects for the config.
//! Each value is made from the name of its tensor and the index of its word, so a fixture has the same words on
//! every host and in every seat's tests.
//!
//! Two readers take the same stored bytes. The whole-model host form of the model crate reads them through
//! `Fixture::memory` and `Fixture::index`. The driver reads them through `mix::Weights`, which this file
//! implements: each `Weight` in the layout that its comment in `mix.rs` gives.
//!
//! Two shapes:
//! - `SMALL`: 2 blocks of 2 layers, the second block runs 2 times, width 32, window 4. Six visits a token.
//!   Its experts are wider than the stream.
//! - `WIDE`: 4 blocks of 2 layers, blocks 1 and 2 run 3 times, width 160, window 3. Sixteen visits a token.
//!   Its width is above 128 and its heads are wider than 32. So the two sum trees of a norm give different
//!   words, and the lanes of a clip matter. It also has layers after the loop, messages from two blocks, and
//!   3 taps in a convolution.
//!
//! Both shapes give the two layer kinds different head sizes, so a reader that takes the sizes of the wrong
//! kind gets a refused shape. The relative table of a full layer is shorter than the test chat, the position
//! scale starts inside the test chat, and its floor is not a power of two.
//!
//! Stored layouts that a reader must know:
//! - A BF16 tensor holds exact BF16 values, widened to 32-bit words. An F32 tensor holds any 32-bit value.
//! - The experts are stored as codes, as in a released checkpoint: packed 4-bit codes with block scales in the
//!   layers before the last block, 8-bit float codes with block exponents in the last block. `w13_weight` is
//!   [experts, 2 * width, hidden]: row `2i` is gate row `i` and row `2i + 1` is up row `i`. `w2_weight` is
//!   [experts, hidden, width]. The driver reads the BF16 values that the one decode of the model crate
//!   (`expert_widen`) makes from the codes: form A1 of the expert arithmetic.
//! - Fixed convolution taps are [channels, 1, taps]. Taps made from the token have `static_w` [taps, channels]
//!   and `weight_proj` [key heads * taps, hidden]. The description says which stored tap meets the current token.
//! - `attn_queries.i` belongs to layer `i + 1`. It has one row, or three rows in a layer with separate inputs.
//! - The two message query tensors are [2 * loop layers, hidden]: rows `2j` and `2j + 1` of the first are the
//!   query and the expert message query of loop layer `j`; of the second, the key and the value message query.
use super::mix::{Message,Weight,Weights,chain_product,need};
use crate::{Result,bounded_decoder::Compute};
use engine_model::{expert_widen::{StoredStack,widen_values},loop_host::{Memory,MemoryTensor},
    plan::{description::{ConvTaps,LoopTextDescription,TapOrder},inkling_index::{ExpertPart,IndexInput,Stored as Storage,StoredRole},storage_groups::StorageGroups},
    safetensors::Dtype,shapes::Ffn};
use spark_gpu_probe::causal_ops::{Op,Shape};
use std::collections::BTreeMap;

/// 2 blocks of 2 layers. Layers 2 and 3 run 2 times. Layers 0 and 2 see a window of 4 tokens.
pub const SMALL:&str=r#"{"model_type":"inkling2_loop","eos_token_id":1,"text_config":{"model_type":"inkling2_loop_text",
"model_max_length":64,"torch_dtype":"bfloat16","hidden_size":32,"num_hidden_layers":4,"vocab_size":24,"unpadded_vocab_size":20,
"num_attention_heads":6,"num_key_value_heads":2,"head_dim":4,"d_rel":4,"rel_extent":6,"q_bias":false,"o_bias":false,
"log_scaling_n_floor":5,"log_scaling_alpha":0.25,"rms_norm_eps":1e-06,"use_value_norm_no_gamma":true,
"qk_l2_clip_threshold":1.5,"rel_l2_clip_threshold":0.75,"local_layer_ids":[0,2],"sconv_kernel_size":4,
"final_logit_softcapping":30.0,"swa_head_dim":8,"swa_num_attention_heads":4,"swa_num_key_value_heads":2,
"sliding_window_size":4,"n_layers_per_attn_res_block":2,"attn_res_normalize_values":true,"n_routed_experts":6,
"num_experts_per_tok":2,"n_shared_experts":0,"intermediate_size":64,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,
"route_scale":2.0,"use_gate_bias":true,
"loop_config":{"core_start":2,"core_end":4,"num_passes":2,"gate_rank":3,"is_varloop":false}}}"#;
/// 4 blocks of 2 layers. Layers 2 to 5 run 3 times. The first layer of each block sees a window of 3 tokens.
pub const WIDE:&str=r#"{"model_type":"inkling2_loop","eos_token_id":2,"text_config":{"model_type":"inkling2_loop_text",
"model_max_length":48,"torch_dtype":"bfloat16","hidden_size":160,"num_hidden_layers":8,"vocab_size":40,"unpadded_vocab_size":33,
"num_attention_heads":4,"num_key_value_heads":2,"head_dim":40,"d_rel":6,"rel_extent":5,"q_bias":false,"o_bias":false,
"log_scaling_n_floor":3,"log_scaling_alpha":0.5,"rms_norm_eps":1e-06,"use_value_norm_no_gamma":true,
"qk_l2_clip_threshold":2.5,"rel_l2_clip_threshold":1.25,"local_layer_ids":[0,2,4,6],"sconv_kernel_size":3,
"final_logit_softcapping":30.0,"swa_head_dim":36,"swa_num_attention_heads":4,"swa_num_key_value_heads":2,
"sliding_window_size":3,"n_layers_per_attn_res_block":2,"attn_res_normalize_values":true,"n_routed_experts":5,
"num_experts_per_tok":3,"n_shared_experts":0,"intermediate_size":32,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,
"route_scale":3.0,"use_gate_bias":true,
"loop_config":{"core_start":2,"core_end":6,"num_passes":3,"gate_rank":4,"is_varloop":false}}}"#;
const TEXT:&str="model.llm.";
/// The most words or bytes that one made-up tensor holds. A config that asks for more is refused.
const TENSOR_LIMIT:usize=1<<24;

/// One stored tensor: its stored type, its shape, and its payload bytes as a weight file holds them (little-endian,
/// row-major). `words` holds the values of a BF16 or 32-bit tensor as 32-bit words. For the codes of an expert stack
/// `words` holds the BF16 values that the decode makes of the whole stack, as `[experts, rows, columns]` in stored
/// row order; the scale tensors beside the codes have no words of their own use.
#[derive(Debug)]
pub struct Stored {pub dtype:Dtype,pub shape:Vec<u64>,pub words:Vec<f32>,pub bytes:Vec<u8>}
/// A description, the tensor index of its config, the stored tensors, and each weight as the driver reads it.
pub struct Fixture {pub description:LoopTextDescription,pub index:IndexInput,tensors:BTreeMap<String,Stored>,weights:BTreeMap<Weight,Vec<f32>>}

fn seed(name:&str)->u64 {name.bytes().fold(0xcbf2_9ce4_8422_2325u64,|hash,byte|(hash^u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3))}
fn mixed(seed:u64,index:u64)->u64 {
    let mut z=seed.wrapping_add(index.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    z=(z^(z>>30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);z=(z^(z>>27)).wrapping_mul(0x94d0_49bb_1331_11eb);z^(z>>31)
}
/// One word with `bits` significant bits, a sign, and a binary exponent from `low` to `high`. The word is built
/// from its bit fields, so no host arithmetic can change it. 8 bits give an exact BF16 value.
fn word(seed:u64,index:usize,bits:u32,low:i32,high:i32)->f32 {
    let r=mixed(seed,index as u64);
    let fraction=((r>>8) as u32&((1<<(bits-1))-1))<<(24-bits);
    let exponent=(low+((r>>40)%(high-low+1) as u64) as i32+127) as u32;
    f32::from_bits(((r as u32&1)<<31)|(exponent<<23)|fraction)
}
/// `count` made-up words for a test, by a name of their own: 8 bits give BF16 values, 24 bits give 32-bit values.
/// The sizes are `[2^low, 2^(high + 1))`, with any sign.
pub fn made(name:&str,count:usize,bits:u32,low:i32,high:i32)->Vec<f32> {let seed=seed(name);(0..count).map(|index|word(seed,index,bits,low,high)).collect()}
/// The count of words or bytes of a shape. A count above the limit of a made-up tensor is a refusal.
fn count(name:&str,shape:&[u64])->Result<usize> {
    shape.iter().try_fold(1u64,|product,&size|product.checked_mul(size)).and_then(|cells|usize::try_from(cells).ok()).filter(|&cells|cells<=TENSOR_LIMIT)
        .ok_or_else(||format!("the fixture makes no tensor of the size of {name}"))
}
/// A tensor that is stored as it is: values of `[2^low, 2^(high + 1))` in size, with any sign.
struct Plain {name:String,dtype:Dtype,shape:Vec<u64>,low:i32,high:i32}
/// Every tensor that is stored as it is, and the name of each expert stack with its storage form.
fn listed(d:&LoopTextDescription)->Result<(Vec<Plain>,Vec<(String,Storage)>)> {
    use Dtype::{Bf16,F32};
    let (h,mut plain,mut stacks)=(d.hidden,Vec::new(),Vec::new());
    let mut add=|name:String,dtype:Dtype,shape:&[u64],low:i32,high:i32|plain.push(Plain {name,dtype,shape:shape.to_vec(),low,high});
    add(format!("{TEXT}embed.weight"),Bf16,&[d.vocab,h],-3,0);
    add(format!("{TEXT}unembed.weight"),Bf16,&[d.vocab,h],-4,-1);
    for name in ["embed_norm.weight","norm.weight"] {add(format!("{TEXT}{name}"),Bf16,&[h],-1,0);}
    add(format!("{TEXT}attn_res.output_query"),F32,&[h],-3,0);
    let looped=d.layers.iter().filter(|layer|layer.message_gates).count() as u64;
    if looped>0 {for name in ["middle_cycle_cross_pass_queries","middle_cycle_cross_pass_kv_queries"] {add(format!("{TEXT}{name}"),F32,&[2*looped,h],-3,0);}}
    for (i,layer) in d.layers.iter().enumerate() {
        let a=layer.attention;let Ffn::Experts(e)=layer.ffn else {return Err(format!("the fixture makes routed experts only, and layer {i} has none"));};
        let (q,kv,taps,p)=(a.query_heads*a.head_dim,a.kv_heads*a.head_dim,a.key_value_conv.kernel,format!("{TEXT}layers.{i}."));
        match (layer.attention_queries,i.checked_sub(1)) {(0,_)=>{},(_,None)=>return Err("the fixture has no attention mix query for layer 0".into()),
            (1,Some(before))=>add(format!("{TEXT}attn_res.attn_queries.{before}"),F32,&[h],-3,0),
            (rows,Some(before))=>add(format!("{TEXT}attn_res.attn_queries.{before}"),F32,&[rows,h],-3,0)}
        add(format!("{TEXT}attn_res.mlp_queries.{i}"),F32,&[h],-3,0);
        for (name,shape,low,high) in [("attn.wq_du.weight",vec![q,h],-5,-1),("attn.wk_dv.weight",vec![kv,h],-5,-1),("attn.wv_dv.weight",vec![kv,h],-5,-1),
            ("attn.wo_ud.weight",vec![h,q],-5,-1),("attn.gate_attn.weight",vec![q,h],-5,-1),("attn.wr_du.weight",vec![a.query_heads*a.relative.dim,h],-5,-1),
            ("attn.rel_logits_proj.proj",vec![a.relative.dim,a.relative.extent],-3,0),("attn.rel_l2_clip.scale",vec![a.query_heads,a.relative.dim],-2,0),
            ("attn.attn_sink_proj.weight",vec![a.query_heads,h],-5,-1),("attn.q_l2_clip.scale",vec![a.query_heads,a.head_dim],-2,0),
            ("attn_sconv.weight",vec![h,1,d.conv.kernel],-3,-1),("mlp_sconv.weight",vec![h,1,d.conv.kernel],-3,-1),("mlp_norm.weight",vec![h],-1,0),
            ("mlp.gate.weight",vec![e.count,h],-4,-1)] {
            add(format!("{p}{name}"),Bf16,&shape,low,high);
        }
        add(format!("{p}mlp.gate.bias"),F32,&[e.count],-6,-3);
        for stream in ["k","v"] {
            if matches!(a.key_value_conv.taps,ConvTaps::FixedAndToken {..}) {
                add(format!("{p}attn.{stream}_sconv.static_w"),Bf16,&[taps,kv],-3,-1);
                add(format!("{p}attn.{stream}_sconv.weight_proj"),Bf16,&[a.kv_heads*taps,h],-6,-3);
            } else {add(format!("{p}attn.{stream}_sconv.weight"),Bf16,&[kv,1,taps],-3,-1);}
        }
        if a.separate_inputs {for input in ["q","k","v"] {add(format!("{p}{input}_input_norm.weight"),Bf16,&[h],-1,0);}}
        else {add(format!("{p}attn_norm.weight"),Bf16,&[h],-1,0);}
        if layer.message_gates {for gate in ["attn","k","v","mlp"] {
            add(format!("{p}controllers.{gate}.down_weight"),Bf16,&[d.core.gate_rank,2*h],-5,-2);
            for part in ["forget_weight","gain_weight"] {add(format!("{p}controllers.{gate}.{part}"),Bf16,&[h,d.core.gate_rank],-4,-1);}
        }}
        // The layers of the last block store 8-bit float codes. The layers before it store packed 4-bit codes.
        let storage=if (i as u64)<(d.layers.len() as u64).saturating_sub(d.residual.block_layers) {Storage::Nvfp4} else {Storage::Mxfp8};
        for stack in ["w13_weight","w2_weight"] {stacks.push((format!("{p}mlp.experts.{stack}"),storage));}
    }
    Ok((plain,stacks))
}
/// The payload of one tensor of an expert stack. Every byte is a valid code, and the widened weights are small:
/// a packed 4-bit weight is under 0.3 in size, an 8-bit float weight under 0.25.
fn expert_part(name:&str,dtype:Dtype,part:ExpertPart,count:usize)->Result<(Vec<f32>,Vec<u8>)> {
    let seed=seed(name);let byte=|rule:fn(u64)->u8|(vec![],(0..count).map(|index|rule(mixed(seed,index as u64))).collect::<Vec<u8>>());
    Ok(match (dtype,part) {
        // Two 4-bit codes a byte: every byte is two codes.
        (Dtype::U8,ExpertPart::Codes)=>byte(|r|(r>>8) as u8),
        // One scale for 16 values: a positive 8-bit float with exponent field 5 to 7, so 0.25 to 1.875.
        (Dtype::F8E4M3,ExpertPart::BlockScale)=>byte(|r|((5+(r>>8)%3)<<3|((r>>16)&7)) as u8),
        // One 8-bit float a weight: exponent field 4 to 7, so 0.125 to 1.875, with any sign. Never the code of no number.
        (Dtype::F8E4M3,ExpertPart::Codes)=>byte(|r|((r&1)<<7|(4+(r>>8)%4)<<3|((r>>16)&7)) as u8),
        // One exponent byte for 32 values: 2^-6 to 2^-3.
        (Dtype::U8,ExpertPart::BlockScale)=>byte(|r|121+((r>>8)%4) as u8),
        // The scale of one expert, 2^-5 to 2^-3, over the divisor of the layer.
        (Dtype::F32,ExpertPart::ExpertScale)=>{let words=(0..count).map(|index|word(seed,index,24,-5,-4).abs()).collect::<Vec<_>>();let bytes=words.iter().flat_map(|v|v.to_le_bytes()).collect();(words,bytes)},
        (Dtype::F32,ExpertPart::ScaleDivisor)=>(vec![6.0;count],(0..count).flat_map(|_|6.0f32.to_le_bytes()).collect()),
        other=>return Err(format!("the fixture has no payload rule for {name}: {other:?}")),
    })
}
/// Fixed taps in the layout of the arm: `[channel, lag]`, and lag 0 meets the current token.
/// `row_is_tap`: the stored tensor is `[tap, channel]`; else it is `[channel, tap]`.
fn taps_by_lag(stored:&[f32],channels:usize,taps:usize,order:TapOrder,row_is_tap:bool)->Result<Vec<f32>> {
    need(stored.len()==channels*taps,"a fixture tap tensor differs from its channels and taps")?;
    let mut out=vec![0.0;channels*taps];
    for channel in 0..channels {for lag in 0..taps {
        let tap=match order {TapOrder::OldestFirst=>taps-1-lag,TapOrder::NewestFirst=>lag};
        out[channel*taps+lag]=if row_is_tap {stored[tap*channels+channel]} else {stored[channel*taps+tap]};
    }}
    Ok(out)
}
impl Fixture {
    /// The made-up checkpoint of a config text. A config that the fixture cannot make is a refusal that says why.
    pub fn new(config:&str)->Result<Self> {
        let description=LoopTextDescription::from_config(config.as_bytes()).map_err(|e|format!("the fixture config: {e}"))?;
        need(description.hidden>0 && description.vocab>0 && description.vocab<=u64::from(u32::MAX) && description.head_rows<=description.vocab,
            "the fixture config has no width, no token ID that fits 32 bits, or more head rows than token IDs")?;
        let (plain,stacks)=listed(&description)?;
        let mut index=IndexInput::from_description(&description,&StorageGroups {method:String::new(),model:Vec::new(),draft:Vec::new()}).map_err(|e|format!("the fixture index: {e}"))?;
        for (name,storage) in plain.iter().map(|t|(&t.name,if t.dtype==Dtype::F32 {Storage::F32} else {Storage::Bf16})).chain(stacks.iter().map(|(name,storage)|(name,*storage))) {
            if index.stored.insert(name.clone(),storage).is_some() {return Err(format!("the fixture makes the tensor {name} twice"));}
        }
        let expected=index.expected_main().map_err(|e|format!("the fixture index: {e}"))?;
        let mut tensors=BTreeMap::new();
        for t in plain {
            let e=expected.get(&t.name).ok_or_else(||format!("the index expects no tensor {}",t.name))?;
            if e.dtype!=t.dtype || e.shape!=t.shape {return Err(format!("the index expects another type or shape for {}",t.name));}
            let (seed,bits)=(seed(&t.name),if t.dtype==Dtype::F32 {24} else {8});
            let words=(0..count(&t.name,&t.shape)?).map(|index|word(seed,index,bits,t.low,t.high)).collect::<Vec<_>>();
            let bytes=if t.dtype==Dtype::F32 {words.iter().flat_map(|v|v.to_le_bytes()).collect()} else {words.iter().flat_map(|v|((v.to_bits()>>16) as u16).to_le_bytes()).collect()};
            tensors.insert(t.name,Stored {dtype:t.dtype,shape:t.shape,words,bytes});
        }
        for (name,e) in &expected {
            let StoredRole::Expert {part,..}=e.role else {if tensors.contains_key(name) {continue;} return Err(format!("the fixture makes no tensor {name}"));};
            let (words,bytes)=expert_part(name,e.dtype,part,count(name,&e.shape)?)?;
            tensors.insert(name.clone(),Stored {dtype:e.dtype,shape:e.shape.clone(),words,bytes});
        }
        if tensors.len()!=expected.len() {return Err("the fixture makes a tensor that the index does not expect".into());}
        let mut fixture=Self {description,index,tensors,weights:BTreeMap::new()};
        for (name,_) in &stacks {let words=fixture.widened(name)?;fixture.tensors.get_mut(name).ok_or_else(||format!("the fixture holds no tensor {name}"))?.words=words;}
        fixture.weights=fixture.read_weights()?;Ok(fixture)
    }
    /// `SMALL` for a test. Outside a test: `Fixture::new(SMALL)`.
    #[cfg(test)]
    pub(crate) fn small()->Self {Self::new(SMALL).expect("the small fixture")}
    /// `WIDE` for a test. Outside a test: `Fixture::new(WIDE)`.
    #[cfg(test)]
    pub(crate) fn wide()->Self {Self::new(WIDE).expect("the wide fixture")}
    /// A tensor by its stored name. A name that the fixture does not hold is a refusal that names it.
    pub fn stored(&self,name:&str)->Result<&Stored> {self.tensors.get(name).ok_or_else(||format!("the fixture holds no tensor {name}"))}
    /// Every stored name, in rising order.
    pub fn names(&self)->impl Iterator<Item=&str> {self.tensors.keys().map(String::as_str)}
    /// The stored tensors as the whole-model host form of the model crate reads them.
    pub fn memory(&self)->Memory {
        Memory {tensors:self.tensors.iter().map(|(name,t)|(name.clone(),MemoryTensor {dtype:t.dtype,shape:t.shape.clone(),bytes:t.bytes.clone()})).collect()}
    }
    /// A fixed chat of `count` token IDs. Token 5 is the last stored ID: the head cannot write it, an input can hold it.
    pub fn chat(&self,count:usize)->Vec<u32> {
        let (seed,vocab)=(seed("chat"),self.description.vocab.max(1));
        (0..count).map(|index|if index==5 {(vocab-1) as u32} else {(mixed(seed,index as u64)%vocab) as u32}).collect()
    }
    /// Rows `rows` of a stored matrix times one input row: one chain a row in rising column from +0.0, each
    /// product rounded before its add. The 32-bit sums are not rounded. The column count is the width of `x`.
    pub fn product<C:Compute>(&self,compute:&C,name:&str,rows:std::ops::Range<usize>,x:&C::Value)->Result<C::Value> {
        let (stored,cols)=(self.stored(name)?,compute.words(x));
        if cols==0 || !stored.words.len().is_multiple_of(cols) || rows.is_empty() || rows.end>stored.words.len()/cols {return Err(format!("the fixture product {name} has no such rows or columns"));}
        let shape=Shape {op:Op::Project,heads:0,dim:0,taps:0,rows:rows.len() as u32,cols:cols as u32,epsilon:0.0,bound:0.0};
        compute.operation(shape,[Some(x),None,None,None],&stored.words[rows.start*cols..rows.end*cols])
    }
    /// The BF16 values of one expert stack, `[experts, rows, columns]` in stored row order, made from the stored codes
    /// by the one decode of the model crate.
    fn widened(&self,name:&str)->Result<Vec<f32>> {
        let (codes,scales)=(self.stored(name)?,self.stored(&format!("{name}_scale"))?);
        let &[count,rows,columns]=&codes.shape[..] else {return Err(format!("the fixture stack {name} has no three sizes"));};
        let (count,rows,columns)=(count as usize,rows as usize,columns as usize);
        let (stored,columns)=match codes.dtype {
            Dtype::U8=>(StoredStack::Packed4 {codes:&codes.bytes,scales:&scales.bytes,amax:&self.stored(&format!("{name}_amax"))?.words,
                denominator:*self.stored(&format!("{name}_outer_scale_denominator"))?.words.first().ok_or_else(||format!("the fixture stack {name} has no scale divisor"))?},2*columns),
            _=>(StoredStack::Float8 {values:&codes.bytes,scales:&scales.bytes},columns),
        };
        let mut out=Vec::with_capacity(count*rows*columns);
        for expert in 0..count {out.extend(widen_values(&stored.expert(count,expert,rows,columns).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?);}
        Ok(out)
    }
    /// The gate-and-up matrix and the down matrix of one expert. The gate-and-up rows are in stored order: row `2i`
    /// is gate row `i`, row `2i + 1` is up row `i`.
    fn expert(&self,layer:usize,expert:usize)->Result<(Vec<f32>,Vec<f32>)> {
        let Some(Ffn::Experts(e))=self.description.layers.get(layer).map(|layer|layer.ffn) else {return Err(format!("the fixture has no routed experts in layer {layer}"));};
        let cells=(e.width*self.description.hidden) as usize;
        let part=|stack:&str,cells:usize|->Result<Vec<f32>> {
            Ok(self.stored(&format!("{TEXT}layers.{layer}.mlp.experts.{stack}"))?.words.get(expert*cells..(expert+1)*cells).ok_or_else(||format!("the fixture holds no expert {expert} in layer {layer}"))?.to_vec())
        };
        Ok((part("w13_weight",2*cells)?,part("w2_weight",cells)?))
    }
    /// Every weight of the driver, in the layout that its comment in `mix.rs` gives.
    fn read_weights(&self)->Result<BTreeMap<Weight,Vec<f32>>> {
        let d=&self.description;let (h,mut out)=(d.hidden as usize,BTreeMap::new());
        let words=|name:String|->Result<Vec<f32>> {Ok(self.stored(&format!("{TEXT}{name}"))?.words.clone())};
        out.insert(Weight::Embedding,words("embed.weight".into())?);
        out.insert(Weight::EmbeddingNorm,words("embed_norm.weight".into())?);
        out.insert(Weight::LastNorm,words("norm.weight".into())?);
        out.insert(Weight::Head,words("unembed.weight".into())?.get(..d.head_rows as usize*h).ok_or("the fixture head has more rows than the stored table")?.to_vec());
        out.insert(Weight::MixOutput,words("attn_res.output_query".into())?);
        if d.layers.iter().any(|layer|layer.message_gates) {
            // Rows `2j` and `2j + 1` of the first tensor: the query and the expert message query of loop layer `j`.
            // Of the second: its key and its value message query.
            let (first,second)=(words("middle_cycle_cross_pass_queries".into())?,words("middle_cycle_cross_pass_kv_queries".into())?);
            let mut merged=Vec::with_capacity(2*first.len());
            for (qe,kv) in first.chunks_exact(2*h).zip(second.chunks_exact(2*h)) {merged.extend_from_slice(&qe[..h]);merged.extend_from_slice(kv);merged.extend_from_slice(&qe[h..]);}
            out.insert(Weight::Messages,merged);
        }
        let fixed_order=|taps:ConvTaps|match taps {ConvTaps::Fixed {order}|ConvTaps::FixedAndToken {order,..}=>order};
        for (i,layer) in d.layers.iter().enumerate() {
            let a=layer.attention;let Ffn::Experts(e)=layer.ffn else {return Err(format!("the fixture has no routed experts in layer {i}"));};
            let (kv,taps,p)=((a.kv_heads*a.head_dim) as usize,a.key_value_conv.kernel as usize,format!("layers.{i}."));
            let part=|name:&str|words(format!("{p}{name}"));
            for (weight,name) in [(Weight::Query(i),"attn.wq_du.weight"),(Weight::Key(i),"attn.wk_dv.weight"),(Weight::Value(i),"attn.wv_dv.weight"),(Weight::Relative(i),"attn.wr_du.weight"),
                (Weight::Sink(i),"attn.attn_sink_proj.weight"),(Weight::OutputGate(i),"attn.gate_attn.weight"),(Weight::Output(i),"attn.wo_ud.weight"),(Weight::QueryClip(i),"attn.q_l2_clip.scale"),
                (Weight::RelativeClip(i),"attn.rel_l2_clip.scale"),(Weight::RelativeTable(i),"attn.rel_logits_proj.proj"),(Weight::ExpertNorm(i),"mlp_norm.weight"),(Weight::Router(i),"mlp.gate.weight"),
                (Weight::RouterBias(i),"mlp.gate.bias")] {out.insert(weight,part(name)?);}
            for (weight,separate) in [(Weight::QueryNorm(i),"q"),(Weight::KeyNorm(i),"k"),(Weight::ValueNorm(i),"v")] {
                out.insert(weight,part(&if a.separate_inputs {format!("{separate}_input_norm.weight")} else {"attn_norm.weight".into()})?);
            }
            for (fixed,made,stream) in [(Weight::KeyTaps(i),Weight::KeyTapProjection(i),"k"),(Weight::ValueTaps(i),Weight::ValueTapProjection(i),"v")] {
                match a.key_value_conv.taps {
                    ConvTaps::Fixed {order}=>{out.insert(fixed,taps_by_lag(&part(&format!("attn.{stream}_sconv.weight"))?,kv,taps,order,false)?);},
                    ConvTaps::FixedAndToken {order,..}=>{
                        out.insert(fixed,taps_by_lag(&part(&format!("attn.{stream}_sconv.static_w"))?,kv,taps,order,true)?);
                        // A stored row is `head * taps + tap`. The driver reads row `head * taps + lag`.
                        let stored=part(&format!("attn.{stream}_sconv.weight_proj"))?;let mut by_lag=Vec::with_capacity(stored.len());
                        need(stored.len()==a.kv_heads as usize*taps*h,"a fixture tap projection differs from its heads and taps")?;
                        for head in 0..a.kv_heads as usize {for lag in 0..taps {
                            let tap=match order {TapOrder::OldestFirst=>taps-1-lag,TapOrder::NewestFirst=>lag};by_lag.extend_from_slice(&stored[(head*taps+tap)*h..(head*taps+tap+1)*h]);
                        }}
                        out.insert(made,by_lag);
                    },
                }
            }
            let stream_taps=d.conv.kernel as usize;
            out.insert(Weight::ConvAfterAttention(i),taps_by_lag(&part("attn_sconv.weight")?,h,stream_taps,fixed_order(d.conv.taps),false)?);
            out.insert(Weight::ConvAfterExperts(i),taps_by_lag(&part("mlp_sconv.weight")?,h,stream_taps,fixed_order(d.conv.taps),false)?);
            if let (true,Some(before))=(layer.attention_queries>0,i.checked_sub(1)) {out.insert(Weight::MixAttention(i),words(format!("attn_res.attn_queries.{before}"))?);}
            out.insert(Weight::MixExperts(i),words(format!("attn_res.mlp_queries.{i}"))?);
            if layer.message_gates {for (message,gate) in [(Message::Query,"attn"),(Message::Key,"k"),(Message::Value,"v"),(Message::Expert,"mlp")] {
                out.insert(Weight::GateDown(i,message),part(&format!("controllers.{gate}.down_weight"))?);
                out.insert(Weight::GateForget(i,message),part(&format!("controllers.{gate}.forget_weight"))?);
                out.insert(Weight::GateGain(i,message),part(&format!("controllers.{gate}.gain_weight"))?);
            }}
            let width=e.width as usize;
            for expert in 0..e.count as usize {
                let (stored,down)=self.expert(i,expert)?;
                // The gate rows first, then the up rows.
                let mut gate_up=Vec::with_capacity(stored.len());
                need(stored.len()==2*width*h,"a fixture expert differs from its width")?;
                for half in 0..2 {for row in 0..width {gate_up.extend_from_slice(&stored[(2*row+half)*h..(2*row+half+1)*h]);}}
                out.insert(Weight::GateUp(i,expert),gate_up);out.insert(Weight::Down(i,expert),down);
            }
        }
        Ok(out)
    }
}
impl<C:Compute> Weights<C> for Fixture {
    fn words(&self,weight:Weight)->Result<&[f32]> {self.weights.get(&weight).map(Vec::as_slice).ok_or_else(||format!("the fixture holds no weight {weight:?}"))}
    fn product(&self,compute:&C,weight:Weight,x:&C::Value)->Result<C::Value> {chain_product(compute,<Self as Weights<C>>::words(self,weight)?,x)}
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounded_decoder::Host;
    use spark_gpu_probe::fixed_math::rn;
    #[test]
    fn the_small_fixture_has_the_ordered_shape() {
        let d=Fixture::small().description;
        assert_eq!((d.hidden,d.layers.len(),d.residual.block_layers,d.blocks()),(32,4,2,2));
        assert_eq!((d.core.start,d.core.end,d.core.passes,d.visits.len()),(2,4,2,6));
        assert_eq!(d.layers.iter().map(|layer|layer.attention.window).collect::<Vec<_>>(),[Some(4),None,Some(4),None]);
        assert_eq!(d.visits.iter().map(|visit|(visit.layer,visit.pass,visit.messages)).collect::<Vec<_>>(),[(0,0,false),(1,0,false),(2,1,false),(3,1,false),(2,2,true),(3,2,true)]);
    }
    #[test]
    fn the_wide_fixture_has_layers_after_the_loop_and_a_width_above_one_norm_chunk() {
        let d=Fixture::wide().description;
        assert_eq!((d.hidden,d.layers.len(),d.blocks(),d.core.passes,d.visits.len()),(160,8,4,3,16));
        assert!(d.hidden>128 && d.layers.iter().all(|layer|layer.attention.head_dim>32));
        assert_eq!(d.visits.iter().map(|visit|visit.layer).collect::<Vec<_>>(),[0,1,2,3,4,5,2,3,4,5,2,3,4,5,6,7]);
        assert_eq!(d.visits.iter().filter(|visit|visit.messages).count(),8);
    }
    #[test]
    fn the_two_layer_kinds_have_their_own_sizes() {
        for fixture in [Fixture::small(),Fixture::wide()] {
            let d=&fixture.description;let window=d.layers[0].attention;let full=d.layers[1].attention;
            assert!(window.window.is_some() && full.window.is_none() && !window.separate_inputs && full.separate_inputs);
            assert_ne!(window.query_heads*window.head_dim,full.query_heads*full.head_dim);
            assert_ne!(window.kv_heads*window.head_dim,full.kv_heads*full.head_dim);
            assert_ne!(window.relative.extent,full.relative.extent);
            assert!(window.head_dim%2==0 && full.head_dim%2==0,"a packed row holds two values a word");
            // The position scale starts inside a chat of 12 tokens, and the floor is not a power of two.
            let floor=full.position_scale.unwrap().floor;assert!(floor<12 && !floor.is_power_of_two());
            assert!(full.relative.extent<12 && window.window.unwrap()<12);
        }
    }
    #[test]
    fn every_tensor_has_its_shape_and_its_number_type() {
        for fixture in [Fixture::small(),Fixture::wide()] {
            let d=&fixture.description;let layers=d.layers.len();let looped=d.layers.iter().filter(|layer|layer.message_gates).count();
            let full=d.layers.iter().filter(|layer|layer.attention.separate_inputs).count();
            // 5 model tensors, 2 message tensors, and for each layer: 16 that every layer has, an attention mix
            // query after layer 0, 3 of a window layer or 7 of a full layer, and 12 gate tensors inside the loop.
            // The two expert stacks of a layer are 4 tensors each as packed 4-bit codes and 2 each as 8-bit floats.
            let last=d.residual.block_layers as usize;
            assert_eq!(fixture.names().count(),5+2+layers*16+(layers-1)+(layers-full)*3+full*7+looped*12+(layers-last)*8+last*4);
            assert_eq!(fixture.index.expected_main().unwrap().keys().map(String::as_str).collect::<Vec<_>>(),fixture.names().collect::<Vec<_>>());
            for name in fixture.names() {
                let t=fixture.stored(name).unwrap();
                assert_eq!(t.bytes.len() as u64,t.shape.iter().product::<u64>()*t.dtype.bytes(),"{name}");
                if name.contains(".mlp.experts.") {assert!(matches!(t.dtype,Dtype::U8|Dtype::F8E4M3|Dtype::F32),"{name}");continue;}
                assert_eq!(t.words.len() as u64,t.shape.iter().product::<u64>(),"{name}");
                assert!(t.words.iter().all(|v|v.is_finite() && *v!=0.0),"{name}");
                match t.dtype {
                    Dtype::Bf16=>assert!(t.words.iter().all(|&v|rn(v).to_bits()==v.to_bits()),"{name} is not exact BF16"),
                    Dtype::F32=>assert!(t.words.iter().any(|&v|rn(v).to_bits()!=v.to_bits()),"{name} has no 32-bit value"),
                    _=>panic!("{name} has a type the fixture does not make"),
                }
                assert_eq!(t.dtype==Dtype::F32,name.contains("attn_res.") || name.contains("middle_cycle") || name.ends_with("gate.bias"),"{name}");
            }
            let a=d.layers[1].attention;let Ffn::Experts(e)=d.layers[1].ffn else {panic!()};
            assert_eq!(fixture.stored("model.llm.layers.1.attn.wq_du.weight").unwrap().shape,[a.query_heads*a.head_dim,d.hidden]);
            assert_eq!(fixture.stored("model.llm.layers.1.attn.k_sconv.weight_proj").unwrap().shape,[a.kv_heads*a.key_value_conv.kernel,d.hidden]);
            assert_eq!(fixture.stored("model.llm.attn_res.attn_queries.0").unwrap().shape,[3,d.hidden]);
            assert_eq!(fixture.stored("model.llm.attn_res.attn_queries.1").unwrap().shape,[d.hidden]);
            // Layer 1 stores two 4-bit codes a byte. The last layer stores one 8-bit float a weight.
            assert_eq!(fixture.stored("model.llm.layers.1.mlp.experts.w13_weight").unwrap().shape,[e.count,2*e.width,d.hidden/2]);
            assert_eq!(fixture.stored("model.llm.layers.1.mlp.experts.w2_weight_scale").unwrap().shape,[e.count,d.hidden,e.width/16]);
            let stack=fixture.stored(&format!("model.llm.layers.{}.mlp.experts.w2_weight",layers-1)).unwrap();
            assert_eq!((stack.dtype,&stack.shape[..]),(Dtype::F8E4M3,&[e.count,d.hidden,e.width][..]));
            assert_eq!(fixture.stored("model.llm.layers.2.controllers.mlp.down_weight").unwrap().shape,[d.core.gate_rank,2*d.hidden]);
            assert!(fixture.stored("model.llm.layers.0.controllers.mlp.down_weight").unwrap_err().contains("layers.0.controllers"));
            assert!(fixture.stored(&format!("model.llm.attn_res.attn_queries.{}",layers-1)).is_err(),"layer 0 has no attention mix query");
        }
    }
    #[test]
    fn the_words_are_fixed_and_differ_between_tensors() {
        let (a,b)=(Fixture::small(),Fixture::small());
        for name in a.names() {assert_eq!(a.stored(name).unwrap().words.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),b.stored(name).unwrap().words.iter().map(|v|v.to_bits()).collect::<Vec<_>>());}
        let (key,value)=(a.stored("model.llm.layers.0.attn.wk_dv.weight").unwrap(),a.stored("model.llm.layers.0.attn.wv_dv.weight").unwrap());
        assert_ne!(key.words,value.words);
        // Known words: a change of the generator changes every test value of every seat, so it must be seen here.
        assert_eq!(a.stored("model.llm.embed_norm.weight").unwrap().words[..3].iter().map(|v|v.to_bits()).collect::<Vec<_>>(),[word(seed("model.llm.embed_norm.weight"),0,8,-1,0).to_bits(),word(seed("model.llm.embed_norm.weight"),1,8,-1,0).to_bits(),word(seed("model.llm.embed_norm.weight"),2,8,-1,0).to_bits()]);
        // These three numbers come from a second calculation outside this file (the same two hash steps, written again).
        assert_eq!(seed("chat"),0xf2a3_8d91_0b5b_348b);
        assert_eq!((word(7,0,8,-1,0).to_bits(),word(7,1,24,-3,0).to_bits()),(0xbf8d_0000,0x3e74_3c66));
    }
    #[test]
    fn the_chat_holds_stored_ids_and_one_id_the_head_cannot_write() {
        let fixture=Fixture::small();let chat=fixture.chat(12);let d=&fixture.description;
        assert_eq!(chat.len(),12);assert!(chat.iter().all(|&id|u64::from(id)<d.vocab));
        assert!(u64::from(chat[5])>=d.head_rows);assert_eq!(chat,fixture.chat(12));assert_eq!(chat[..4],fixture.chat(4)[..]);
        assert!(chat.iter().collect::<std::collections::BTreeSet<_>>().len()>6,"the chat repeats too many IDs");
    }
    #[test]
    fn a_product_is_one_chain_a_row_and_takes_a_part_of_a_stack() {
        let fixture=Fixture::small();let d=&fixture.description;let h=d.hidden as usize;
        let x=fixture.stored("model.llm.embed.weight").unwrap().words[h..2*h].to_vec();
        let name="model.llm.layers.0.attn.wk_dv.weight";let w=&fixture.stored(name).unwrap().words;
        let out=fixture.product(&Host,name,0..w.len()/h,&x).unwrap();
        let chain=|row:usize|{let mut sum=0.0f32;for j in 0..h {sum+=x[j]*w[row*h+j];}sum};
        assert_eq!(out.len(),w.len()/h);
        for (row,value) in out.iter().enumerate() {assert_eq!(value.to_bits(),chain(row).to_bits());}
        assert!(out.iter().any(|&v|rn(v).to_bits()!=v.to_bits()),"the sums are 32-bit and are not rounded");
        // A part of the rows: the same words as those rows of the whole product.
        let rows=w.len()/h;
        assert_eq!(fixture.product(&Host,name,1..rows-1,&x).unwrap(),out[1..rows-1]);
        assert!(fixture.product(&Host,name,1..1,&x).is_err() && fixture.product(&Host,name,0..rows+1,&x).is_err());
        assert!(fixture.product(&Host,"model.llm.none",0..1,&x).unwrap_err().contains("model.llm.none"));
        // The driver's call gives the same words as the stored rows.
        assert_eq!(<Fixture as Weights<Host>>::product(&fixture,&Host,Weight::Key(0),&x).unwrap(),out);
    }
    #[test]
    fn each_weight_of_the_driver_has_the_layout_of_its_name() {
        for fixture in [Fixture::small(),Fixture::wide()] {
            let d=&fixture.description;let h=d.hidden as usize;let words=|weight:Weight|<Fixture as Weights<Host>>::words(&fixture,weight).unwrap();
            let stored=|name:&str|&fixture.stored(&format!("{TEXT}{name}")).unwrap().words;
            assert_eq!(words(Weight::Head).len() as u64,d.head_rows*d.hidden);assert_eq!(words(Weight::Head),&stored("unembed.weight")[..d.head_rows as usize*h]);
            assert_eq!(words(Weight::Embedding).len() as u64,d.vocab*d.hidden);
            // A layer with one attention norm answers the three names with it. A layer with three has three.
            assert_eq!((words(Weight::QueryNorm(0)),words(Weight::ValueNorm(0))),(&stored("layers.0.attn_norm.weight")[..],&stored("layers.0.attn_norm.weight")[..]));
            assert_eq!((words(Weight::KeyNorm(1)),words(Weight::ValueNorm(1))),(&stored("layers.1.k_input_norm.weight")[..],&stored("layers.1.v_input_norm.weight")[..]));
            // Fixed taps: the last stored tap of a channel meets the current token, so it is lag 0.
            let (taps,fixed)=(d.conv.kernel as usize,stored("layers.0.attn_sconv.weight"));let by_lag=words(Weight::ConvAfterAttention(0));
            for channel in [0,1,h-1] {for lag in 0..taps {assert_eq!(by_lag[channel*taps+lag].to_bits(),fixed[channel*taps+taps-1-lag].to_bits());}}
            // The static taps of a layer that makes taps from the token: stored row 0 meets the current token.
            let a=d.layers[1].attention;let (kv,kv_taps)=((a.kv_heads*a.head_dim) as usize,a.key_value_conv.kernel as usize);
            let (stat,by_lag)=(stored("layers.1.attn.v_sconv.static_w"),words(Weight::ValueTaps(1)));
            for channel in [0,kv-1] {for lag in 0..kv_taps {assert_eq!(by_lag[channel*kv_taps+lag].to_bits(),stat[lag*kv+channel].to_bits());}}
            assert_eq!(words(Weight::KeyTapProjection(1)),&stored("layers.1.attn.k_sconv.weight_proj")[..]);
            assert!(<Fixture as Weights<Host>>::words(&fixture,Weight::KeyTapProjection(0)).unwrap_err().contains("KeyTapProjection(0)"));
            // The message queries: query, key, value, expert for each loop layer.
            let (first,second,merged)=(stored("middle_cycle_cross_pass_queries"),stored("middle_cycle_cross_pass_kv_queries"),words(Weight::Messages));
            assert_eq!(merged.len(),2*first.len());
            assert_eq!((&merged[4*h..5*h],&merged[5*h..6*h],&merged[6*h..7*h],&merged[7*h..8*h]),(&first[2*h..3*h],&second[2*h..3*h],&second[3*h..4*h],&first[3*h..4*h]));
            // One expert: the gate rows first, then the up rows; BF16 values; not all one size.
            let Ffn::Experts(e)=d.layers[0].ffn else {panic!()};let width=e.width as usize;
            for layer in [0,d.layers.len()-1] {
                let (stored_rows,down)=fixture.expert(layer,1).unwrap();let gate_up=words(Weight::GateUp(layer,1));
                assert_eq!((gate_up.len(),down.len()),(2*width*h,h*width));assert_eq!(words(Weight::Down(layer,1)),&down[..]);
                assert_eq!((&gate_up[h..2*h],&gate_up[(width+1)*h..(width+2)*h]),(&stored_rows[2*h..3*h],&stored_rows[3*h..4*h]));
                assert!(gate_up.iter().chain(&down).all(|&v|v.is_finite() && v.abs()<0.5 && rn(v).to_bits()==v.to_bits()));
                assert!(gate_up.iter().filter(|&&v|v!=0.0).map(|v|v.to_bits()).collect::<std::collections::BTreeSet<_>>().len()>16);
                assert_ne!(fixture.expert(layer,0).unwrap().0,stored_rows);
            }
        }
    }
    #[test]
    fn a_config_that_the_fixture_cannot_make_is_a_refusal_and_not_a_panic() {
        let refused=|config:&str|{let why=Fixture::new(config).err().expect("the config is refused");println!("refused: {why}");why};
        assert!(refused("{").contains("the fixture config"));
        // An expert width that no stored code form holds, a table that is no made-up tensor, and more head rows than IDs.
        refused(&SMALL.replace("\"intermediate_size\":64","\"intermediate_size\":40"));
        refused(&SMALL.replace("\"vocab_size\":24","\"vocab_size\":4000000000"));
        assert!(refused(&SMALL.replace("\"vocab_size\":24","\"vocab_size\":1000000")).contains("no tensor of the size of model.llm.embed.weight"));
        refused(&SMALL.replace("\"unpadded_vocab_size\":20","\"unpadded_vocab_size\":25"));
        refused(&SMALL.replace("\"hidden_size\":32","\"hidden_size\":0"));
        refused(&SMALL.replace("\"num_hidden_layers\":4","\"num_hidden_layers\":0"));
        refused(&SMALL.replace("\"n_layers_per_attn_res_block\":2","\"n_layers_per_attn_res_block\":9"));
        // The public call makes the same bytes as the call of the tests.
        for (config,test) in [(SMALL,Fixture::small()),(WIDE,Fixture::wide())] {
            let public=Fixture::new(config).unwrap();
            assert_eq!(public.names().collect::<Vec<_>>(),test.names().collect::<Vec<_>>());
            for name in public.names() {assert_eq!(public.stored(name).unwrap().bytes,test.stored(name).unwrap().bytes,"{name}");}
            assert_eq!(public.chat(12),test.chat(12));
        }
    }
    #[test]
    fn the_whole_model_host_form_of_the_model_crate_loads_the_fixture() {
        use engine_model::loop_host::{Limits,Model};
        for fixture in [Fixture::small(),Fixture::wide()] {
            let model=Model::new(&fixture.description,&fixture.index,Box::new(fixture.memory()),Limits {expert_cache_bytes:usize::MAX}).unwrap();
            assert_eq!((model.hidden() as u64,model.head_rows() as u64),(fixture.description.hidden,fixture.description.head_rows));
        }
    }
}
