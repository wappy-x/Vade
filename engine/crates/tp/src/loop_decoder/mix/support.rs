//! Test support of the mix and of the expert half: a made-up checkpoint, a hand form of each operation, and a
//! backend that compares each block arm with its hand form.
//!
//! The hand forms follow the written contract of the arms (the headers of `gpu/src/block_stream.rs` and
//! `gpu/src/block_experts.rs`). They are test code only and were written apart from the arms. `Stand` runs the arm
//! and refuses a launch whose words differ from the hand form. An arm that has no body for the contract runs on
//! its hand form, and `Stand` lists it as standing. The tests need that list to be empty.
use super::{Message,Weight,Weights,MESSAGES,chain_product};
use super::super::fixture::Fixture;
use crate::{Result,bounded_decoder::{ChatGather,Compute,Host},mxfp8_projection::Mxfp8Projection};
use engine_model::{plan::{admission::CheckpointAdmission,description::{ConvTaps,LoopTextDescription,TapOrder}},shapes::Ffn};
use spark_gpu_probe::{causal_ops::{Op,Shape},fixed_math::{exp,rn,sigmoid,tanh},norm_boundary::NormShape};
use std::{cell::{Cell,RefCell},collections::{BTreeMap,BTreeSet}};

/// A made-up config of the looped schema. Every size is small and belongs to no checkpoint.
/// The norm epsilon differs from the epsilon of a mix, so a test tells the two norms apart.
pub fn description(layers:usize,block:usize,start:usize,end:usize,passes:usize,window_layers:&[usize])->LoopTextDescription {
    let local=window_layers.iter().map(|l|l.to_string()).collect::<Vec<_>>().join(",");
    let text=format!(r#"{{"model_type":"inkling2_loop_text","torch_dtype":"bfloat16","model_max_length":64,"hidden_size":40,
        "num_hidden_layers":{layers},"vocab_size":48,"unpadded_vocab_size":44,"eos_token_id":7,"num_attention_heads":4,"num_key_value_heads":2,
        "head_dim":4,"d_rel":2,"rel_extent":8,"q_bias":false,"o_bias":false,"log_scaling_n_floor":16,"log_scaling_alpha":0.25,
        "rms_norm_eps":1e-05,"use_value_norm_no_gamma":true,"qk_l2_clip_threshold":5.5,"rel_l2_clip_threshold":2.0,
        "local_layer_ids":[{local}],"sconv_kernel_size":4,"final_logit_softcapping":30.0,"swa_head_dim":8,"swa_num_attention_heads":2,
        "swa_num_key_value_heads":1,"sliding_window_size":4,"n_layers_per_attn_res_block":{block},"attn_res_normalize_values":true,
        "n_routed_experts":5,"num_experts_per_tok":3,"n_shared_experts":0,"intermediate_size":6,"mlp_gate_softcap":4.0,
        "mlp_up_softcap":25.0,"route_scale":2.0,"use_gate_bias":true,
        "loop_config":{{"core_start":{start},"core_end":{end},"num_passes":{passes},"gate_rank":3,"is_varloop":false}}}}"#);
    LoopTextDescription::from_config(text.as_bytes()).unwrap()
}
/// Six layers in blocks of two, the loop over the second block, three passes. Layers 1, 3 and 5 read every earlier
/// token, so each has three attention queries.
pub fn small()->LoopTextDescription {description(6,2,2,4,3,&[0,2,4])}

/// Made-up numbers in a fixed order: the same seed gives the same words on every host.
pub struct Numbers(u64);
impl Numbers {
    pub fn new(seed:u64)->Self {Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)|1)}
    /// A value in `[-1, 1)`.
    pub fn next(&mut self)->f32 {
        self.0=self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0>>40) as f32/(1u32<<24) as f32)*2.0-1.0
    }
    /// 32-bit words in `[-scale, scale)`.
    pub fn words(&mut self,count:usize,scale:f32)->Vec<f32> {(0..count).map(|_|self.next()*scale).collect()}
    /// BF16 values in about `[-scale, scale]`.
    pub fn bf16(&mut self,count:usize,scale:f32)->Vec<f32> {(0..count).map(|_|rn(self.next()*scale)).collect()}
}
/// The weights of a made-up checkpoint, as host words. `asked` lists the products in the order they ran.
pub struct Made {pub words:BTreeMap<Weight,Vec<f32>>,pub asked:RefCell<Vec<Weight>>}
impl Made {
    pub fn new(d:&LoopTextDescription,seed:u64)->Self {
        let mut n=Numbers::new(seed);let mut words=BTreeMap::new();let width=d.hidden as usize;
        let (rank,taps,start,end)=(d.core.gate_rank as usize,d.conv.kernel as usize,d.core.start as usize,d.core.end as usize);
        for (layer,spec) in d.layers.iter().enumerate() {
            let Ffn::Experts(experts)=spec.ffn else {panic!("the made-up layers have routed experts")};let inner=experts.width as usize;
            if spec.attention_queries>0 {words.insert(Weight::MixAttention(layer),n.words(spec.attention_queries as usize*width,2.0));}
            words.insert(Weight::MixExperts(layer),n.words(width,2.0));
            words.insert(Weight::ExpertNorm(layer),(0..width).map(|_|rn(1.0+n.next()*0.25)).collect());
            words.insert(Weight::Router(layer),n.bf16(experts.count as usize*width,0.25));
            words.insert(Weight::RouterBias(layer),n.words(experts.count as usize,0.125));
            words.insert(Weight::ConvAfterExperts(layer),n.bf16(width*taps,0.5));
            for expert in 0..experts.count as usize {
                words.insert(Weight::GateUp(layer,expert),n.bf16(2*inner*width,0.375));
                words.insert(Weight::Down(layer,expert),n.bf16(width*inner,0.375));
            }
            if spec.message_gates {for which in [Message::Query,Message::Key,Message::Value,Message::Expert] {
                words.insert(Weight::GateDown(layer,which),n.bf16(rank*2*width,0.25));
                words.insert(Weight::GateForget(layer,which),n.bf16(width*rank,0.5));
                words.insert(Weight::GateGain(layer,which),n.bf16(width*rank,0.5));
            }}
        }
        words.insert(Weight::MixOutput,n.words(width,2.0));
        words.insert(Weight::Messages,n.words((end-start)*MESSAGES*width,2.0));
        Self {words,asked:RefCell::new(Vec::new())}
    }
    pub fn get(&self,weight:Weight)->&[f32] {&self.words[&weight]}
    /// The weights of the mix and of the expert half of a fixture, from its stored tensors. Each stored layout is put
    /// into the layout of its `Weight` name here, as a loader does it:
    /// - The message queries are stored as two tensors with two rows a loop layer: (query, expert) and (key, value).
    /// - A stored gate-and-up matrix mixes its rows: row `2i` is gate row `i`, row `2i + 1` is up row `i`.
    /// - Stored fixed taps are `[channels, 1, taps]`. The description says which stored tap meets the current token.
    pub fn stored(fixture:&Fixture)->Self {
        let d=&fixture.description;let (width,taps)=(d.hidden as usize,d.conv.kernel as usize);
        let read=|name:String|fixture.stored(&format!("model.llm.{name}")).unwrap().words.clone();
        let newest_first=matches!(d.conv.taps,ConvTaps::Fixed {order:TapOrder::NewestFirst}|ConvTaps::FixedAndToken {order:TapOrder::NewestFirst,..});
        let (mut words,mut messages,mut loop_layer)=(BTreeMap::new(),Vec::new(),0);
        for (layer,spec) in d.layers.iter().enumerate() {
            let Ffn::Experts(experts)=spec.ffn else {panic!("the fixture layers have routed experts")};let inner=experts.width as usize;
            if spec.attention_queries>0 {words.insert(Weight::MixAttention(layer),read(format!("attn_res.attn_queries.{}",layer-1)));}
            words.insert(Weight::MixExperts(layer),read(format!("attn_res.mlp_queries.{layer}")));
            words.insert(Weight::ExpertNorm(layer),read(format!("layers.{layer}.mlp_norm.weight")));
            words.insert(Weight::Router(layer),read(format!("layers.{layer}.mlp.gate.weight")));
            words.insert(Weight::RouterBias(layer),read(format!("layers.{layer}.mlp.gate.bias")));
            let conv=read(format!("layers.{layer}.mlp_sconv.weight"));
            words.insert(Weight::ConvAfterExperts(layer),(0..width*taps).map(|i|conv[i/taps*taps+if newest_first {i%taps} else {taps-1-i%taps}]).collect());
            let (gate_up,down)=(read(format!("layers.{layer}.mlp.experts.w13_weight")),read(format!("layers.{layer}.mlp.experts.w2_weight")));
            for expert in 0..experts.count as usize {
                let both=&gate_up[expert*2*inner*width..(expert+1)*2*inner*width];
                let rows=(0..2).flat_map(|kind|(0..inner).map(move |i|2*i+kind)).flat_map(|row|both[row*width..(row+1)*width].iter().copied()).collect();
                words.insert(Weight::GateUp(layer,expert),rows);
                words.insert(Weight::Down(layer,expert),down[expert*width*inner..(expert+1)*width*inner].to_vec());
            }
            if spec.message_gates {
                for (which,name) in [(Message::Query,"attn"),(Message::Key,"k"),(Message::Value,"v"),(Message::Expert,"mlp")] {
                    words.insert(Weight::GateDown(layer,which),read(format!("layers.{layer}.controllers.{name}.down_weight")));
                    words.insert(Weight::GateForget(layer,which),read(format!("layers.{layer}.controllers.{name}.forget_weight")));
                    words.insert(Weight::GateGain(layer,which),read(format!("layers.{layer}.controllers.{name}.gain_weight")));
                }
                let (first,second)=(read("middle_cycle_cross_pass_queries".into()),read("middle_cycle_cross_pass_kv_queries".into()));
                // In the order of `Message`: query, key, value, expert.
                for (tensor,row) in [(&first,2*loop_layer),(&second,2*loop_layer),(&second,2*loop_layer+1),(&first,2*loop_layer+1)] {
                    messages.extend_from_slice(&tensor[row*width..(row+1)*width]);
                }
                loop_layer+=1;
            }
        }
        words.insert(Weight::MixOutput,read("attn_res.output_query".into()));
        if loop_layer>0 {words.insert(Weight::Messages,messages);}
        Self {words,asked:RefCell::new(Vec::new())}
    }
}
impl<C:Compute> Weights<C> for Made {
    fn words(&self,weight:Weight)->Result<&[f32]> {self.words.get(&weight).map(Vec::as_slice).ok_or_else(||format!("made-up weight {weight:?} is absent"))}
    fn product(&self,compute:&C,weight:Weight,x:&C::Value)->Result<C::Value> {
        self.asked.borrow_mut().push(weight);chain_product(compute,<Self as Weights<C>>::words(self,weight)?,x)
    }
}

/// The hand form of each operation that the two halves launch.
pub mod hand {
    use super::{exp,rn,sigmoid,tanh};
    /// The sum of squares on the 32-lane tree: lane `i mod 32` in rising index, then the join at 16, 8, 4, 2, 1.
    pub fn tree(x:&[f32])->f32 {
        let mut lanes=[0.0f32;32];for (i,&v) in x.iter().enumerate() {lanes[i%32]+=v*v;}
        for distance in [16,8,4,2,1] {for lane in 0..distance {lanes[lane]+=lanes[lane+distance];}}
        lanes[0]
    }
    /// The sum of squares in the tree of the norm kernel: chunks of 128 values, each on the 32-lane tree, and then
    /// the chunk sums on a 32-lane tree with lane `chunk mod 32`.
    pub fn norm_tree(x:&[f32])->f32 {
        let mut lanes=[0.0f32;32];for (chunk,part) in x.chunks(128).enumerate() {lanes[chunk%32]+=tree(part);}
        for distance in [16,8,4,2,1] {for lane in 0..distance {lanes[lane]+=lanes[lane+distance];}}
        lanes[0]
    }
    /// The norm with a result that is rounded once: `rn((x * scale) * weight)`, or `rn(x * scale)` with no weight.
    pub fn rounded_norm(x:&[f32],weight:&[f32],epsilon:f32)->Vec<f32> {
        let scale=1.0/(norm_tree(x)/x.len() as f32+epsilon).sqrt();
        x.iter().enumerate().map(|(i,&v)|if weight.is_empty() {rn(v*scale)} else {rn((v*scale)*weight[i])}).collect()
    }
    /// The plain norm that stays 32-bit.
    pub fn plain_norm(x:&[f32],epsilon:f32)->Vec<f32> {let scale=1.0/(tree(x)/x.len() as f32+epsilon).sqrt();x.iter().map(|&v|v*scale).collect()}
    /// One chain in rising index from +0.0.
    pub fn dot(a:&[f32],b:&[f32])->f32 {assert_eq!(a.len(),b.len());let mut sum=0.0f32;for (x,y) in a.iter().zip(b) {sum+=x*y;}sum}
    /// A matrix times a vector: one chain a row.
    pub fn product(matrix:&[f32],x:&[f32])->Vec<f32> {matrix.chunks_exact(x.len()).map(|row|dot(row,x)).collect()}
    /// The softmax of the scores and the blend of the rows with its weights.
    pub fn blend(scores:&[f32],rows:&[&[f32]])->Vec<f32> {
        let mut largest=f32::NEG_INFINITY;for &score in scores {largest=largest.max(score);}
        let mut sum=0.0f32;let parts=scores.iter().map(|&score|{let e=exp(score-largest);sum+=e;e}).collect::<Vec<_>>();
        let mut out=vec![0.0f32;rows[0].len()];
        for (part,row) in parts.iter().zip(rows) {let weight=part/sum;for (total,&value) in out.iter_mut().zip(*row) {*total+=weight*value;}}
        out
    }
    /// One block mix for one query: every entry is normed here, as the reference does it.
    pub fn mix(query:&[f32],entries:&[&[f32]])->Vec<f32> {
        let normed=entries.iter().map(|entry|plain_norm(entry,super::super::MIX_NORM_EPSILON)).collect::<Vec<_>>();
        let scale=(1.0/(query.len() as f64).sqrt()) as f32;
        let scores=normed.iter().map(|row|dot(query,row)*scale).collect::<Vec<_>>();
        blend(&scores,&normed.iter().map(Vec::as_slice).collect::<Vec<_>>())
    }
    /// `(1 - tanh(F)) * value + tanh(G) * message`.
    pub fn gate_blend(forget:&[f32],gain:&[f32],value:&[f32],message:&[f32])->Vec<f32> {
        (0..value.len()).map(|i|(1.0-tanh(forget[i]))*value[i]+tanh(gain[i])*message[i]).collect()
    }
    /// `tanh(x * r) * c`: `r` is the reciprocal of the cap `c`.
    pub fn cap(x:f32,r:f32,c:f32)->f32 {tanh(x*r)*c}
    /// The reciprocal of a cap, made in 64-bit and rounded once. This is the hand form: it does not call the driver.
    pub fn inverse(c:f64)->f32 {(1.0/c) as f32}
    /// The expert activation of one element, rounded once. `gate` and `up` are each a cap and its reciprocal.
    pub fn active(gate:f32,up:f32,gate_cap:(f32,f32),up_cap:(f32,f32))->f32 {let (gate,up)=(rn(gate),rn(up));rn((cap(gate,gate_cap.0,gate_cap.1)*sigmoid(gate))*cap(up,up_cap.0,up_cap.1))}
    /// The sum of the picked experts, in pick order, rounded once.
    pub fn fold(weights:&[f32],downs:&[&[f32]])->Vec<f32> {
        (0..downs[0].len()).map(|i|{let mut sum=0.0f32;for (weight,down) in weights.iter().zip(downs) {sum+=rn(down[i])*weight;}rn(sum)}).collect()
    }
    /// One channel of a short convolution. `past` holds the inputs of this channel, the oldest first and the current
    /// token last. `taps` is by lag: `taps[0]` meets the current token.
    pub fn conv(past:&[f32],taps:&[f32],scale:f32)->f32 {
        let last=past.len()-1;let mut sum=0.0f32;
        for lag in (0..taps.len()).rev() {sum+=taps[lag]*rn(past[last-lag]);}
        rn((sum+rn(past[last]))*scale)
    }
}
/// A block arm by its written contract. `None`: not one of the arms that this seat launches.
fn stand_in(s:Shape,x:[&[f32];4],w:&[f32])->Result<Option<Vec<f32>>> {
    let (h,d,t,r,k)=(s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.cols as usize);
    let sized=|ok:bool|->Result<()> {if ok {Ok(())} else {Err(format!("stand-in arm {:?}: an extent differs",s.op))}};
    Ok(Some(match s.op {
        Op::PlainNorm=>{sized(h>0 && d>0 && x[0].len()==h*d)?;x[0].chunks_exact(d).flat_map(|row|hand::plain_norm(row,s.epsilon)).collect()},
        Op::RoundedNorm=>{sized(h>0 && d>0 && t<=1 && s.epsilon>0.0 && x[0].len()==h*d && w.len()==t*d)?;x[0].chunks_exact(d).flat_map(|row|hand::rounded_norm(row,w,s.epsilon)).collect()},
        Op::MixBlend=>{sized(h>0 && k>0 && d>0 && x[0].len()==h*k && x[1].len()==k*d)?;let rows=x[1].chunks_exact(d).collect::<Vec<_>>();
            x[0].chunks_exact(k).flat_map(|dots|hand::blend(&dots.iter().map(|&dot|dot*s.bound).collect::<Vec<_>>(),&rows)).collect()},
        Op::GateBlend=>{sized(d>0 && x[0].len()==2*d && x[1].len()==d && x[2].len()==d)?;hand::gate_blend(&x[0][..d],&x[0][d..],x[1],x[2])},
        Op::CappedGate=>{sized(r>0 && d>0 && x[0].len()==r*2*d && w.len()==2)?;
            x[0].chunks_exact(2*d).flat_map(|pick|(0..d).map(|i|hand::active(pick[i],pick[d+i],(w[0],s.epsilon),(w[1],s.bound))).collect::<Vec<_>>()).collect()},
        Op::RouteFold=>{sized(r>0 && d>0 && x[0].len()==r && x[1].len()==r*d)?;hand::fold(x[0],&x[1].chunks_exact(d).collect::<Vec<_>>())},
        Op::ShortConv=>{let width=h*d;let form=s.cols;
            sized(width>0 && t>0 && x[0].len()==t*width && x[3].len()==1 && w.len()==width*t)?;
            sized(x[1].len()==if form&1!=0 {h*t} else {0} && x[2].len()==if form&2!=0 {width} else {0})?;
            let slot=x[3][0] as usize;sized(x[3][0]>=0.0 && slot<t)?;
            (0..width).map(|c|{
                let past=(0..t).map(|age|x[0][((slot+1+age)%t)*width+c]).collect::<Vec<_>>();
                let taps=(0..t).map(|lag|w[c*t+lag]+if form&1!=0 {x[1][(c/d)*t+lag]} else {0.0}).collect::<Vec<_>>();
                let y=hand::conv(&past,&taps,s.bound);
                if form&2==0 {y} else if form&4!=0 {rn(x[2][c]+y)} else {x[2][c]+y}
            }).collect()},
        _=>return Ok(None),
    }))
}
/// The host backend. A launch of a block arm of this seat is also run on its hand form, and the two results must
/// have the same bits. An arm that refuses the contract shape, or takes another weight length than the contract,
/// has no body for the contract yet: its launch runs on the hand form and its name goes into `standing`.
/// `launches` counts each launch by its operation.
#[derive(Default)]
pub struct Stand {pub launches:RefCell<BTreeMap<u32,usize>>,pub standing:RefCell<BTreeSet<u32>>,pub compared:Cell<usize>,pub norms:Cell<usize>,pub reads:Cell<usize>}
impl Stand {
    pub fn new()->Self {Self::default()}
    pub fn count(&self,op:Op)->usize {self.launches.borrow().get(&(op as u32)).copied().unwrap_or(0)}
}
impl Compute for Stand {
    type Value=Vec<f32>;type Pass=<Host as Compute>::Pass;
    fn mhc_row_launches(&self)->u64 {0}
    fn rows_greedy(&self,logits:&Vec<f32>,rows:usize)->Result<Option<Vec<u32>>> {Host.rows_greedy(logits,rows)}
    fn has_latent_absorption(&self,name:&str)->bool {Host.has_latent_absorption(name)}
    fn gather_absorbed(&self,s:Shape,x:[Option<&Vec<f32>>;4],reserve:usize,expected:f32)->Result<Vec<f32>> {Host.gather_absorbed(s,x,reserve,expected)}
    fn latent_absorption(&self,name:&str,s:Shape,query:&Vec<f32>,latent:&Vec<f32>,count:&Vec<f32>,slots:usize)->Result<Vec<f32>> {Host.latent_absorption(name,s,query,latent,count,slots)}
    fn latent_absorption_rows(&self,name:&str,s:spark_gpu_probe::prompt_ops::Shape,query:&Vec<f32>,latent:&Vec<f32>,selected:&Vec<f32>)->Result<Vec<f32>> {Host.latent_absorption_rows(name,s,query,latent,selected)}
    fn latent_absorption_chats(&self,name:&str,s:spark_gpu_probe::prompt_ops::Shape,gathers:&[ChatGather<'_,Vec<f32>>],stride:usize,query:&Vec<f32>,selected:&Vec<f32>)->Result<Option<Vec<f32>>> {
        Host.latent_absorption_chats(name,s,gathers,stride,query,selected)
    }
    fn upload(&self,v:&[f32])->Result<Vec<f32>> {Host.upload(v)}
    fn read(&self,v:&Vec<f32>)->Result<Vec<f32>> {self.reads.set(self.reads.get()+1);Host.read(v)}
    fn select_known(&self,ids:&Vec<f32>,count:&Vec<f32>,row:&[f32])->Result<()> {Host.select_known(ids,count,row)}
    fn prompt_end(&self)->Result<()> {Host.prompt_end()}
    fn prompt_seeds(&self)->bool {Host.prompt_seeds()}
    fn prepare_beside(&self,c:&CheckpointAdmission,live:usize)->Result<()> {Host.prepare_beside(c,live)}
    fn words(&self,v:&Vec<f32>)->usize {v.len()}
    fn range(&self,v:&Vec<f32>,start:usize,len:usize)->Result<Vec<f32>> {Host.range(v,start,len)}
    fn join(&self,parts:&[&Vec<f32>])->Result<Vec<f32>> {Host.join(parts)}
    fn zeros(&self,len:usize)->Result<Vec<f32>> {Host.zeros(len)}
    fn operation(&self,s:Shape,x:[Option<&Vec<f32>>;4],w:&[f32])->Result<Vec<f32>> {
        *self.launches.borrow_mut().entry(s.op as u32).or_default()+=1;
        let Some(hand)=stand_in(s,x.map(|v|v.map_or(&[][..],Vec::as_slice)),w)? else {return Host.operation(s,x,w);};
        if !s.lengths().is_some_and(|lengths|lengths.1==w.len()) {
            self.standing.borrow_mut().insert(s.op as u32);
            return if hand.iter().all(|v|v.is_finite()) {Ok(hand)} else {Err("stand-in arm output is not finite".into())};
        }
        let out=Host.operation(s,x,w)?;self.compared.set(self.compared.get()+1);
        match bits(&out).iter().zip(bits(&hand)).position(|(a,b)|*a!=b) {
            Some(word)=>Err(format!("arm {:?} differs from its hand form at word {word}: {:?} and {:?}",s.op,out[word],hand[word])),
            None if out.len()!=hand.len()=>Err(format!("arm {:?} gives {} words, its hand form {}",s.op,out.len(),hand.len())),
            None=>Ok(out),
        }
    }
    fn norm(&self,name:&str,s:NormShape,x:&Vec<f32>,w:&[f32])->Result<Vec<f32>> {self.norms.set(self.norms.get()+1);Host.norm(name,s,x,w)}
    fn begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Vec<f32>)->Result<(Self::Pass,Vec<f32>)> {Host.begin(key,p,x)}
    fn finish(&self,p:Self::Pass,x:&Vec<f32>)->Result<Vec<f32>> {Host.finish(p,x)}
    fn mx(&self,p:&Mxfp8Projection<'_>,x:&Vec<f32>)->Result<Vec<f32>> {Host.mx(p,x)}
    fn round(&self,x:&Vec<f32>)->Result<Vec<f32>> {Host.round(x)}
    fn mean(&self,x:&Vec<f32>,streams:u32,hidden:u32)->Result<Vec<f32>> {Host.mean(x,streams,hidden)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&Vec<f32>>;5],w:&[f32])->Result<Vec<f32>> {Host.prompt(s,x,w)}
    fn rows_norm(&self,name:&str,s:NormShape,x:&Vec<f32>,m:usize,w:&[f32])->Result<Vec<f32>> {Host.rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Vec<f32>,m:usize)->Result<(Self::Pass,Vec<f32>)> {Host.rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:Self::Pass,x:&Vec<f32>,m:usize)->Result<Vec<f32>> {Host.rows_finish(p,x,m)}
}
/// The bits of a row, for a comparison that tells +0.0 from -0.0.
pub fn bits(x:&[f32])->Vec<u32> {x.iter().map(|v|v.to_bits()).collect()}
