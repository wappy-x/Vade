//! Greedy draft block on the same backend as its verifier. Projection weights stay in their
//! checkpoint format on disk. The device can pack projection weights for proposals. Context keys, values, recurrent state and vocabulary logits stay on device.
use super::{mix::{Weight,Weights},operands::{load::Store,weights::ResidentProduct}};
use crate::{Result,bounded_decoder::Compute};
use engine_model::mtp_draft::{MtpConfig,MtpModel};
use spark_gpu_probe::draft_ops::DraftOp;

fn need(ok:bool,why:&str)->Result<()> {if ok {Ok(())} else {Err(format!("loop draft: {why}"))}}
pub trait Lanes:Compute {
    fn draft_sum(&self,x:&Self::Value,y:&Self::Value)->Result<Self::Value> {self.bf16_sum(x,y)}
    fn lanes(&self,op:DraftOp,x:&Self::Value,y:Option<&Self::Value>)->Result<Self::Value>;
    /// `lanes` with a third input (`DraftOp::MarkovLogits`).
    fn lanes3(&self,op:DraftOp,x:&Self::Value,y:&Self::Value,z:&Self::Value)->Result<Self::Value>;
    fn draft_norm(&self,x:&Self::Value,rows:usize,width:usize,eps:f32,weight:&Self::Value)->Result<Self::Value>;
    /// Leave the greedy pick `[best, id, flag]` on the device for the next draft step.
    fn greedy_on_device(&self,logits:&Self::Value)->Result<Self::Value>;
}
/// Bound the shared immutable phase table independently of the advertised
/// context. Later positions keep the original calculation.
fn phase_rows(positions:usize,width:usize)->usize {
    const MAX_WORDS:usize=(8<<20)/size_of::<f32>();
    positions.min(MAX_WORDS/width.max(1))
}
#[test]
fn phase_table_respects_both_limits() {
    assert_eq!(phase_rows(128,64),128);
    assert_eq!(phase_rows(1<<20,128),16_384);
    assert_eq!(phase_rows(5,1<<22),0);
}
/// Baseline device allocations of the drafter, before optional BF16 panels.
pub fn device_weight_bytes(model:&MtpModel)->usize {
    packed_weight_bytes(model,None,None).expect("the BF16 estimate needs no packed unit layout")
}
/// Base device storage for these draft formats. The verifier keeps its own weights.
pub fn packed_weight_bytes(model:&MtpModel,layer_format:Option<ExpertCodes>,head_format:Option<ExpertCodes>)->Result<usize> {
    let cfg=model.config();
    let h=cfg.hidden_size;
    let mut layers=packed::storage(cfg.unpadded_vocab_size,h,head_format)?.unwrap_or(0);
    for l in model.layers() {
        let matrices=[(&l.q_proj,h),(&l.k_proj,h),(&l.v_proj,h),(&l.k_proj_ctx,h),(&l.v_proj_ctx,h),
            (&l.o_proj,cfg.q_width()),(&l.gate_proj,h),(&l.up_proj,h),(&l.down_proj,cfg.intermediate_size)];
        for (bits,columns) in matrices {
            layers+=packed::storage(bits.len()/columns,columns,layer_format)?.unwrap_or(bits.len()*2);
        }
        layers+=[&l.input_layernorm,&l.post_attention_layernorm,&l.q_norm,&l.k_norm,&l.k_norm_ctx,&l.dflare_norm]
            .into_iter().map(|v|v.len()*4).sum::<usize>();
    }
    let (rnn, rnn_bias) = model.rnn_proj();
    let (fold, fold_bias) = model.fold();
    let latent=cfg.markov_rank;
    let fold_panel=packed::panel(cfg.unpadded_vocab_size,latent,head_format);
    let rnn_bytes=if fold_panel.is_some()
        && let (Some(dyn_u),Some(h_u))=(packed::panel(3*latent,2*latent,layer_format),packed::panel(3*latent,h,layer_format))
    {
        (spark_gpu_probe::draft_ops::panel_words(dyn_u.rows,dyn_u.columns).ok_or("draft rnn dyn size")?
            +spark_gpu_probe::draft_ops::panel_words(h_u.rows,h_u.columns).ok_or("draft rnn h size")?*2)*4
    } else {
        packed::storage(3*latent,2*latent+2*h,layer_format)?.unwrap_or(rnn.len()*2)
    };
    let fold_bytes=match fold_panel {
        Some(unit)=>spark_gpu_probe::draft_ops::panel_words(unit.rows,unit.columns).ok_or("draft fold size")?*4,None=>fold.len()*2,
    };
    Ok(layers + fold_bytes + rnn_bytes + model.markov_w1().len() * 2
        + (rnn_bias.len() + fold_bias.len() + model.markov_norm().len() + model.final_norm().len()
            + model.mask_embed().len() * model.config().gamma()
            + phase_rows(model.config().max_position_embeddings,model.config().head_dim)*model.config().head_dim
            + model.fuse_alpha().iter().map(Vec::len).sum::<usize>()) * 4)
}

mod batch;
mod packed;
use engine_model::plan::loop_load::ExpertCodes;
/// `Fused`: the rows live in the layer's sibling panels, which project them in one launch.
enum Projection<V> { Bf16(V), Packed(packed::Packed<V>), Fused }
struct Matrix<V> {weight:Projection<V>,rows:usize,columns:usize}
fn bf16<S:Store>(store:&mut S,bits:&[u16],rows:usize,columns:usize)->Result<S::Value> {
    need(rows>0 && columns.is_multiple_of(2) && rows.checked_mul(columns)==Some(bits.len()),"draft weight shape")?;
    let bytes=bits.iter().flat_map(|b|b.to_le_bytes()).collect::<Vec<_>>();
    let mut value=store.create(bits.len()/2)?;store.write(&mut value,0,&bytes)?;
    store.register_bf16(&value,rows,columns,&bytes)?;
    Ok(value)
}
fn matrix<S:Store>(store:&mut S,bits:&[u16],rows:usize,columns:usize)->Result<Matrix<S::Value>> {
    Ok(Matrix {weight:Projection::Bf16(bf16(store,bits,rows,columns)?),rows,columns})
}
impl<V> Matrix<V> {
    fn dot<C:ResidentProduct<V>+Lanes>(&self,c:&C,x:&C::Value,n:usize)->Result<C::Value> {
        match &self.weight {
            Projection::Packed(pack)=>pack.dot(c,x,n),
            Projection::Bf16(value)=>c.resident_product_rows(value,0..self.rows*self.columns/2,self.rows,self.columns,x,n,true),
            Projection::Fused=>Err("loop draft: a fused projection runs with its siblings".into()),
        }
    }
    /// Reuse one quantized activation row set across sibling projections.
    fn prepare<C:ResidentProduct<V>+Lanes>(&self,c:&C,x:&C::Value,n:usize)->Result<Option<C::Value>> {
        if n<=spark_gpu_probe::draft_ops::PANEL_INPUTS_MAX && matches!(self.weight,Projection::Packed(packed::Packed::Panel{..})) {
            return Ok(Some(c.lanes(DraftOp::PackActivations(c.words(x)),x,None)?));
        }
        Ok(None)
    }
    fn rounded_shared<C:ResidentProduct<V>+Lanes>(&self,c:&C,x:&C::Value,n:usize,input:Option<&C::Value>)->Result<C::Value> {
        if let (Some(input),Projection::Packed(packed::Packed::Panel{unit,weight}))=(input,&self.weight)
            && n<=spark_gpu_probe::draft_ops::PANEL_INPUTS_MAX
        {
            return c.draft_panel(DraftOp::Panel{rows:unit.rows,columns:unit.columns,inputs:n},weight,input,None);
        }
        self.rounded(c,x,n)
    }
    fn rounded<C:ResidentProduct<V>+Lanes>(&self,c:&C,x:&C::Value,n:usize)->Result<C::Value> {
        if let Projection::Packed(packed::Packed::Panel{..})=&self.weight {self.dot(c,x,n)} else {c.round(&self.dot(c,x,n)?)}
    }
}
struct Layer<V> {q:Matrix<V>,k:Matrix<V>,v:Matrix<V>,k_ctx:Matrix<V>,v_ctx:Matrix<V>,o:Matrix<V>,gate:Matrix<V>,up:Matrix<V>,down:Matrix<V>,
    qkv:Option<packed::Siblings<V>>,gate_up:Option<packed::Siblings<V>>}
impl<V> Layer<V> {
    /// The rounded q, k and v rows of one normalized block: one launch when `n <= PANEL_INPUTS_MAX`, else 32-row slabs.
    fn qkv<C:ResidentProduct<V>+Lanes>(&self,c:&C,x:&C::Value,n:usize)->Result<[C::Value;3]> {
        const MAX:usize=spark_gpu_probe::draft_ops::PANEL_INPUTS_MAX;
        if let Some(fused)=&self.qkv {
            let [q,k,v]=fused.rows();
            if n<=MAX {
                let out=fused.project(c,x,n)?;
                return Ok([c.range(&out,0,q*n)?,c.range(&out,q*n,k*n)?,c.range(&out,(q+k)*n,v*n)?]);
            }
            let cols=fused.columns();
            let (mut qs,mut ks,mut vs)=(Vec::new(),Vec::new(),Vec::new());
            for start in (0..n).step_by(MAX) {
                let count=(n-start).min(MAX);
                let out=fused.project(c,&c.range(x,start*cols,count*cols)?,count)?;
                qs.push(c.range(&out,0,q*count)?);
                ks.push(c.range(&out,q*count,k*count)?);
                vs.push(c.range(&out,(q+k)*count,v*count)?);
            }
            return Ok([
                c.join(&qs.iter().collect::<Vec<_>>())?,
                c.join(&ks.iter().collect::<Vec<_>>())?,
                c.join(&vs.iter().collect::<Vec<_>>())?,
            ]);
        }
        let input=self.q.prepare(c,x,n)?;
        Ok([self.q.rounded_shared(c,x,n,input.as_ref())?,self.k.rounded_shared(c,x,n,input.as_ref())?,self.v.rounded_shared(c,x,n,input.as_ref())?])
    }
    /// The rounded gate and up rows of one normalized block: one launch when `n <= PANEL_INPUTS_MAX`, else 32-row slabs.
    fn gate_up<C:ResidentProduct<V>+Lanes>(&self,c:&C,y:&C::Value,n:usize)->Result<[C::Value;2]> {
        const MAX:usize=spark_gpu_probe::draft_ops::PANEL_INPUTS_MAX;
        if let Some(fused)=&self.gate_up {
            let [gate,up,_]=fused.rows();
            if n<=MAX {
                let out=fused.project(c,y,n)?;
                return Ok([c.range(&out,0,gate*n)?,c.range(&out,gate*n,up*n)?]);
            }
            let cols=fused.columns();
            let (mut gs,mut us)=(Vec::new(),Vec::new());
            for start in (0..n).step_by(MAX) {
                let count=(n-start).min(MAX);
                let out=fused.project(c,&c.range(y,start*cols,count*cols)?,count)?;
                gs.push(c.range(&out,0,gate*count)?);
                us.push(c.range(&out,gate*count,up*count)?);
            }
            return Ok([c.join(&gs.iter().collect::<Vec<_>>())?,c.join(&us.iter().collect::<Vec<_>>())?]);
        }
        let input=self.gate.prepare(c,y,n)?;
        Ok([self.gate.rounded_shared(c,y,n,input.as_ref())?,self.up.rounded_shared(c,y,n,input.as_ref())?])
    }
}
struct Norms<A> {input:A,post:A,q:A,k:A,k_ctx:A,dflare:A}
/// Factored NVFP4 panels of `rnn_proj [3*rank, 2*rank + 2*h]`:
/// - `dyn_part [3*rank, 2*rank]`: tiny L2-resident `[state, embedding]` step matrix (~257 KB)
/// - `draft_part [3*rank, h]`: batched once per proposal across all `depth` draft rows
/// - `feat_part [3*rank, h]`: projected once per proposal on `chat.feature`
struct SplitRnn<V> {dyn_part:V,draft_part:V,feat_part:V}
pub struct Drafter<'a,V,A> {
    model:&'a MtpModel,bytes:usize,head:Option<packed::Packed<V>>,layers:Vec<Layer<V>>,rnn:Option<Matrix<V>>,rnn_split:Option<SplitRnn<V>>,fold:V,
    w1:A,rnn_bias:A,fold_bias:Option<A>,mask:A,alpha:Vec<A>,
    norms:Vec<Norms<A>>,markov_norm:A,final_norm:A,phases:A,
}
pub struct DraftChat<A> {kv:Vec<Option<(A,A)>>,local_base:Vec<usize>,feature:Option<A>,len:usize,limit:usize}
/// Grow committed rows geometrically. Temporary proposal rows need only their
/// own tail, including when the committed prefix fills a power of two exactly.
fn kv_capacity(committed:usize,temporary:usize,limit:usize)->usize {
    committed.checked_next_power_of_two().unwrap_or(limit).saturating_add(temporary).min(limit)
}
/// Bounded sliding buffer capacity (in rows) for a local draft attention layer (`sliding_window_size`).
/// Holds `window - 1` history rows, `2 * block_size` temporary proposal rows, and ~1 window of append slack.
pub(crate) fn local_kv_cap(cfg:&MtpConfig,limit:usize)->usize {
    let keep=cfg.sliding_window_size.saturating_sub(1);
    let temp=cfg.block_size.saturating_mul(2);
    keep.saturating_add(temp).saturating_add(keep.max(64)).next_multiple_of(64).min(limit).max(1)
}
/// Paged capacity (in rows) for a global draft attention layer at `needed` rows and `page_bytes` granularity.
pub(crate) fn global_kv_rows(cfg:&MtpConfig,needed:usize,limit:usize,page_bytes:usize)->usize {
    let needed=needed.max(1).min(limit);
    let row_bytes=cfg.kv_width().saturating_mul(4);
    if page_bytes==0 || row_bytes==0 || limit.saturating_mul(row_bytes)<page_bytes {
        return kv_capacity(needed,0,limit);
    }
    let page_rows=page_bytes.div_ceil(row_bytes).max(1);
    needed.div_ceil(page_rows).saturating_mul(page_rows).min(limit)
}
/// Total draft KV bytes (K + V across all draft layers) reserved for a chat at `tokens` tokens.
pub(crate) fn draft_kv_bytes(cfg:&MtpConfig,tokens:usize,limit:usize,page_bytes:usize)->usize {
    let row_bytes=cfg.kv_width().saturating_mul(4);
    let temp=cfg.block_size.saturating_mul(2);
    let need=tokens.saturating_add(temp).min(limit);
    (0..cfg.num_layers).map(|i| {
        let rows=if cfg.is_local(i) {
            local_kv_cap(cfg,limit)
        } else {
            global_kv_rows(cfg,need,limit,page_bytes)
        };
        rows.saturating_mul(row_bytes).saturating_mul(2)
    }).sum()
}
#[test]
fn kv_growth_keeps_one_temporary_tail_at_a_power_of_two() {
    assert_eq!(kv_capacity(8192,8,32768),8200);
    assert_eq!(kv_capacity(8193,8,32768),16392);
    assert_eq!(kv_capacity(8192,8,8196),8196);
    assert_eq!(kv_capacity(8192,8,8192),8192);
}
#[test]
fn draft_kv_bytes_bounds_local_layers_and_pages_global_layer_without_256k_cliff() {
    // Inkling-2-Mini draft geometry: 2 local layers (window 1024) + 1 global layer, 8 kv heads * 128 head_dim.
    let cfg=MtpConfig {
        hidden_size:4096,intermediate_size:14336,num_layers:3,num_heads:32,num_kv_heads:8,head_dim:128,
        sliding_window_size:1024,block_size:8,markov_rank:256,vocab_size:201088,unpadded_vocab_size:200064,
        max_position_embeddings:1_048_576,rms_norm_eps:1e-6,rope_theta:1_000_000.0,final_logit_softcapping:30.0,
        local_layer_ids:vec![0,1],target_layer_ids:vec![0,15,30],
    };
    const VMM_PAGE:usize=2*1024*1024;
    // Local layer cap stays constant (~2,112 rows = ~16.5 MiB per layer, ~33 MiB for 2 local layers) at 256K & 512K.
    assert_eq!(local_kv_cap(&cfg,262_144),local_kv_cap(&cfg,524_288));
    assert!(local_kv_cap(&cfg,524_288)<3000);
    // At 1 token: 2 bounded local layers (~33 MiB) + 1 global VMM page (2 * 2 MiB = 4 MiB) ~= 37 MiB (NOT 6.44 GB!).
    let at_1_256k=draft_kv_bytes(&cfg,1,262_144,VMM_PAGE);
    let at_1_512k=draft_kv_bytes(&cfg,1,524_288,VMM_PAGE);
    assert_eq!(at_1_256k,at_1_512k,"no 256K reservation cliff");
    assert!(at_1_512k<48*1024*1024,"1-token draft reservation must stay under 48 MiB, got {at_1_512k}");
    // Global layer grows smoothly in 512-token (4 MiB K+V) pages as live tokens grow.
    let at_32k=draft_kv_bytes(&cfg,32_768,524_288,VMM_PAGE);
    let at_256k=draft_kv_bytes(&cfg,262_144,524_288,VMM_PAGE);
    assert!(at_1_512k<at_32k && at_32k<at_256k);
    assert!(at_256k<2_300*1024*1024,"256K draft KV uses only 1 global layer (~2.1 GB), not 3x (~6.4 GB)");
}
fn push_layer_kv_for<C:Compute>(cfg:&MtpConfig,c:&C,chat:&mut DraftChat<C::Value>,layer:usize,k:&C::Value,v:&C::Value,rows:usize)->Result<()> {
    let kv=cfg.kv_width();
    let end=chat.len.checked_add(rows).ok_or("draft context overflow")?;
    let temp=cfg.block_size.saturating_mul(2);
    let need_tail=temp.min(chat.limit.saturating_sub(end));
    if cfg.is_local(layer) {
        let keep_window=cfg.sliding_window_size.saturating_sub(1);
        let cap=local_kv_cap(cfg,chat.limit);
        let cap_words=cap.checked_mul(kv).ok_or("draft cache shape overflow")?;
        if chat.kv[layer].is_none() {
            let init_rows=if end.saturating_add(need_tail)>=cap {cap} else {kv_capacity(end,need_tail,cap)};
            let w=init_rows.checked_mul(kv).ok_or("draft cache shape overflow")?;
            chat.kv[layer]=Some((c.zeros(w)?,c.zeros(w)?));
        }
        let cur_cap=c.words(&chat.kv[layer].as_ref().ok_or("missing draft cache")?.0)/kv;
        let base=chat.local_base[layer];
        if end.saturating_sub(base).saturating_add(need_tail)<=cur_cap {
            let (keys,vals)=chat.kv[layer].as_mut().ok_or("missing draft cache")?;
            let off=(chat.len-base)*kv;
            c.state_fill(keys,off,k)?;c.state_fill(vals,off,v)?;
        } else if cur_cap<cap && end.saturating_sub(base).saturating_add(need_tail)<=cap {
            let grow_rows=kv_capacity(end-base,need_tail,cap);
            let w=grow_rows.checked_mul(kv).ok_or("draft cache shape overflow")?;
            let (mut keys,mut vals)=(c.zeros(w)?,c.zeros(w)?);
            let live=chat.len-base;
            if live>0 {
                let (old_k,old_v)=chat.kv[layer].as_ref().ok_or("missing draft cache")?;
                c.state_fill(&mut keys,0,&c.range(old_k,0,live*kv)?)?;
                c.state_fill(&mut vals,0,&c.range(old_v,0,live*kv)?)?;
            }
            c.state_fill(&mut keys,live*kv,k)?;c.state_fill(&mut vals,live*kv,v)?;
            chat.kv[layer]=Some((keys,vals));
        } else {
            let new_base=end.saturating_sub(keep_window);
            let (mut keys,mut vals)=(c.zeros(cap_words)?,c.zeros(cap_words)?);
            if new_base<chat.len {
                let keep_old=chat.len-new_base;
                let old_off=(new_base-base)*kv;
                let (old_k,old_v)=chat.kv[layer].as_ref().ok_or("missing draft cache")?;
                c.state_fill(&mut keys,0,&c.range(old_k,old_off,keep_old*kv)?)?;
                c.state_fill(&mut vals,0,&c.range(old_v,old_off,keep_old*kv)?)?;
                c.state_fill(&mut keys,keep_old*kv,k)?;
                c.state_fill(&mut vals,keep_old*kv,v)?;
            } else {
                let skip=new_base-chat.len;
                let keep_new=rows-skip;
                if keep_new>0 {
                    c.state_fill(&mut keys,0,&c.range(k,skip*kv,keep_new*kv)?)?;
                    c.state_fill(&mut vals,0,&c.range(v,skip*kv,keep_new*kv)?)?;
                }
            }
            chat.kv[layer]=Some((keys,vals));
            chat.local_base[layer]=new_base;
        }
        return Ok(());
    }
    let page_bytes=c.plane_page_bytes();
    let need_rows=end.saturating_add(temp).min(chat.limit);
    let want_rows=global_kv_rows(cfg,need_rows,chat.limit,page_bytes);
    let want_words=want_rows.checked_mul(kv).ok_or("draft cache shape overflow")?;
    let max_words=chat.limit.checked_mul(kv).ok_or("draft cache shape overflow")?;
    if chat.kv[layer].is_none() {
        let keys=c.plane_alloc(max_words,want_words)?;
        let vals=c.plane_alloc(max_words,want_words)?;
        chat.kv[layer]=Some((keys,vals));
    } else {
        let (keys,vals)=chat.kv[layer].as_mut().ok_or("missing draft cache")?;
        if c.words(keys)<want_words {c.plane_grow(keys,want_words,max_words)?;}
        if c.words(vals)<want_words {c.plane_grow(vals,want_words,max_words)?;}
    }
    let (keys,vals)=chat.kv[layer].as_mut().ok_or("missing draft cache")?;
    c.state_fill(keys,chat.len*kv,k)?;c.state_fill(vals,chat.len*kv,v)?;
    Ok(())
}
#[test]
fn draft_chat_sliding_local_and_paged_global_preserve_exact_history_words() {
    use crate::bounded_decoder::Host;
    let cfg=MtpConfig {
        hidden_size:16,intermediate_size:16,num_layers:2,num_heads:2,num_kv_heads:2,head_dim:4,
        sliding_window_size:8,block_size:4,markov_rank:4,vocab_size:32,unpadded_vocab_size:32,
        max_position_embeddings:512,rms_norm_eps:1e-6,rope_theta:10000.0,final_logit_softcapping:30.0,
        local_layer_ids:vec![0],target_layer_ids:vec![0],
    };
    let kv=cfg.kv_width();
    let row_k=|p:usize|(0..kv).map(|j|(p*100+j+1) as f32).collect::<Vec<_>>();
    let row_v=|p:usize|(0..kv).map(|j|(p*100+j+51) as f32).collect::<Vec<_>>();
    // Test both small incremental chunks (crossing local_kv_cap slides & global page boundaries) and a large chunk (> cap).
    for chunks in [vec![1usize;300],vec![150,7,1,120,19]] {
        let mut chat=DraftChat::new(cfg.num_layers,cfg.max_position_embeddings);
        let mut ref_k=Vec::new();let mut ref_v=Vec::new();
        for n in chunks {
            let start_pos=chat.len;
            let mut ks=Vec::with_capacity(n*kv);let mut vs=Vec::with_capacity(n*kv);
            for p in start_pos..start_pos+n {let (k,v)=(row_k(p),row_v(p));ks.extend_from_slice(&k);vs.extend_from_slice(&v);ref_k.extend_from_slice(&k);ref_v.extend_from_slice(&v);}
            chat.ensure_global_capacity(&Host,&cfg,start_pos+n).unwrap();
            push_layer_kv_for(&cfg,&Host,&mut chat,0,&ks,&vs,n).unwrap();
            push_layer_kv_for(&cfg,&Host,&mut chat,1,&ks,&vs,n).unwrap();
            chat.len+=n;
            // Local layer 0: bounded buffer capacity and bit-exact visible sliding window `[start..len)`.
            let cap=local_kv_cap(&cfg,chat.limit);
            let (lk,lv)=chat.kv[0].as_ref().unwrap();
            assert!(lk.len()<=cap*kv,"local buffer stays bounded: {} <= {}",lk.len(),cap*kv);
            let w_start=chat.len.saturating_sub(cfg.sliding_window_size-1);
            let base=chat.local_base[0];
            assert!(base<=w_start);
            let rel=(w_start-base)*kv;let span=(chat.len-w_start)*kv;
            assert_eq!(&lk[rel..rel+span],&ref_k[w_start*kv..chat.len*kv]);
            assert_eq!(&lv[rel..rel+span],&ref_v[w_start*kv..chat.len*kv]);
            // Global layer 1: exact full prefix `[0..len)` across paged growth.
            let (gk,gv)=chat.kv[1].as_ref().unwrap();
            assert_eq!(&gk[..chat.len*kv],&ref_k[..chat.len*kv]);
            assert_eq!(&gv[..chat.len*kv],&ref_v[..chat.len*kv]);
        }
        assert!(chat.local_base[0]>0,"local window actually slid past initial base");
    }
}
impl<A> DraftChat<A> {
    pub(super) fn new(layers:usize,limit:usize)->Self {
        Self {kv:(0..layers).map(|_|None).collect(),local_base:vec![0;layers],feature:None,len:0,limit}
    }
    pub fn len(&self)->usize {self.len}
    pub fn is_empty(&self)->bool {self.len==0}
    pub fn limit_to(&mut self,context:usize) {self.limit=self.limit.min(context);}
    /// Retained device bytes for a committed draft prefix of `len` rows: bounded sliding window on local
    /// layers + paged VMM pages (`page_bytes`) on global layers + recurrent feature.
    pub fn prefix_bytes(cfg:&MtpConfig,len:usize,limit:usize,page_bytes:usize)->usize {
        let page=|words:usize|words.saturating_mul(4).next_multiple_of(4096);
        let kv_w=cfg.kv_width();
        let kv:usize=if len==0 {0} else {
            (0..cfg.num_layers).map(|i| {
                let rows=if cfg.is_local(i) {
                    len.min(cfg.sliding_window_size.saturating_sub(1))
                } else {
                    global_kv_rows(cfg,len,limit,page_bytes)
                };
                if rows>0 {2*page(rows.saturating_mul(kv_w))} else {0}
            }).sum()
        };
        let feat=if len>0 {page(cfg.hidden_size)} else {0};
        kv.saturating_add(feat).saturating_add(std::mem::size_of::<Self>())
    }
    /// Private (non-paged-global) bytes retained for a committed draft prefix: local sliding windows + feature.
    pub fn retained_private_bytes(cfg:&MtpConfig,len:usize)->usize {
        let page=|words:usize|words.saturating_mul(4).next_multiple_of(4096);
        let kv_w=cfg.kv_width();
        let local_kv:usize=if len==0 {0} else {
            let keep=len.min(cfg.sliding_window_size.saturating_sub(1));
            let local_layers=(0..cfg.num_layers).filter(|&i|cfg.is_local(i)).count();
            if keep>0 {local_layers.saturating_mul(2).saturating_mul(page(keep.saturating_mul(kv_w)))} else {0}
        };
        let feat=if len>0 {page(cfg.hidden_size)} else {0};
        local_kv.saturating_add(feat).saturating_add(std::mem::size_of::<Self>())
    }
    /// Enumerate global paged draft KV allocations for CoW deduplication in `SessionCache`.
    pub fn collect_allocations<C:Compute<Value=A>>(&self,c:&C,cfg:&MtpConfig,committed_tokens:Option<usize>,out:&mut Vec<(usize,usize)>)->bool {
        let kv_w=cfg.kv_width();
        for (i,slot) in self.kv.iter().enumerate() {
            if cfg.is_local(i) {continue;}
            let Some((keys,vals))=slot else {continue;};
            let ok=match committed_tokens {
                Some(toks)=>{
                    let words=toks.min(self.limit).saturating_mul(kv_w);
                    c.plane_prefix_allocations(keys,words,out) && c.plane_prefix_allocations(vals,words,out)
                }
                None=>c.state_allocations(keys,out) && c.state_allocations(vals,out),
            };
            if !ok {return false;}
        }
        true
    }
    /// Copy the committed draft prefix and recurrent feature.
    /// Local layers keep only the visible sliding window (`sliding_window_size - 1` rows); global layers use
    /// zero-copy CoW `plane_fork` on complete VMM pages. No returned buffer mutably aliases the source.
    pub fn copy_prefix<C:Compute<Value=A>>(&self,c:&C,cfg:&MtpConfig,restore:bool)->Result<Self> {
        need(self.len>0 && self.len<=self.limit && self.kv.len()==cfg.num_layers && self.local_base.len()==cfg.num_layers,"invalid draft prefix state")?;
        let kv_width=cfg.kv_width();
        let temp=cfg.block_size.saturating_mul(2);
        let page_bytes=c.plane_page_bytes();
        let max_words=self.limit.checked_mul(kv_width).ok_or("draft prefix capacity overflow")?;
        let copy=|src:&A|c.rows_append(None,0,src,1);
        let mut kv=Vec::with_capacity(self.kv.len());
        let mut local_base=vec![0usize;self.kv.len()];
        for (i,slot) in self.kv.iter().enumerate() {
            let Some((keys,vals))=slot else {return Err("loop draft: missing draft KV for non-empty prefix".into());};
            if cfg.is_local(i) {
                let start=self.len.saturating_sub(cfg.sliding_window_size.saturating_sub(1));
                let base=self.local_base[i];
                need(base<=start,"draft local window base passed read start")?;
                let keep_rows=self.len-start;
                let rel_off=(start-base).checked_mul(kv_width).ok_or("draft prefix offset overflow")?;
                let keep_words=keep_rows.checked_mul(kv_width).ok_or("draft prefix size overflow")?;
                need(c.words(keys)>=rel_off+keep_words && c.words(vals)>=rel_off+keep_words,"draft local KV shorter than window")?;
                let cap=local_kv_cap(cfg,self.limit);
                let restore_rows=kv_capacity(keep_rows.max(1),temp,cap);
                let restore_words=restore_rows.checked_mul(kv_width).ok_or("draft local capacity overflow")?;
                let plane=|src:&A|->Result<A> {
                    if keep_words==0 {
                        return c.zeros(if restore {restore_words} else {0});
                    }
                    let live=c.range(src,rel_off,keep_words)?;
                    if restore && keep_words<restore_words {c.state_restore(&live,restore_words)} else {copy(&live)}
                };
                kv.push(Some((plane(keys)?,plane(vals)?)));
                local_base[i]=start;
            } else {
                let committed=self.len.checked_mul(kv_width).ok_or("draft prefix size overflow")?;
                need(c.words(keys)>=committed && c.words(vals)>=committed,"draft global KV shorter than len")?;
                let want_tokens=if restore {self.len.saturating_add(temp).min(self.limit)} else {self.len};
                let want_words=global_kv_rows(cfg,want_tokens,self.limit,page_bytes).checked_mul(kv_width).ok_or("draft global capacity overflow")?;
                kv.push(Some((
                    c.plane_fork(keys,committed,want_words,max_words)?,
                    c.plane_fork(vals,committed,want_words,max_words)?,
                )));
            }
        }
        let feature=match &self.feature {
            Some(feat)=>{
                need(c.words(feat)==cfg.hidden_size,"draft feature width differs")?;
                Some(copy(feat)?)
            }
            None if cfg.num_layers==0=>None,
            None=>return Err("loop draft: missing draft feature for non-empty prefix".into()),
        };
        Ok(Self {kv,local_base,feature,len:self.len,limit:self.limit})
    }
    /// Pre-grow global paged draft KV planes to cover `needed_tokens` tokens (call outside `compute.work`).
    pub fn ensure_global_capacity<C:Compute<Value=A>>(&mut self,c:&C,cfg:&MtpConfig,needed_tokens:usize)->Result<()> {
        let kv=cfg.kv_width();
        let temp=cfg.block_size.saturating_mul(2);
        let need_rows=needed_tokens.max(self.len).saturating_add(temp).min(self.limit);
        let page_bytes=c.plane_page_bytes();
        let want_rows=global_kv_rows(cfg,need_rows,self.limit,page_bytes);
        let want_words=want_rows.checked_mul(kv).ok_or("draft cache shape overflow")?;
        let max_words=self.limit.checked_mul(kv).ok_or("draft cache shape overflow")?;
        for i in 0..self.kv.len() {
            if cfg.is_local(i) {continue;}
            if let Some((keys,vals))=self.kv[i].as_mut() {
                if c.words(keys)<want_words {c.plane_grow(keys,want_words,max_words)?;}
                if c.words(vals)<want_words {c.plane_grow(vals,want_words,max_words)?;}
            } else {
                let keys=c.plane_alloc(max_words,want_words)?;
                let vals=c.plane_alloc(max_words,want_words)?;
                self.kv[i]=Some((keys,vals));
            }
        }
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn advance_len(&mut self,rows:usize) {self.len+=rows;}
}
impl<'a,V,A> Drafter<'a,V,A> {
    pub fn config(&self)->&MtpConfig {self.model.config()}
    /// Maximum neural draft depth supported by `propose_confident`: wide 3-tier block-causal DFlash up to 15
    /// when GPU-packed (`head.is_some()`, `head_dim == 128`, `gamma >= 7`), else `gamma`.
    pub fn wide_depth(&self)->usize {
        let cfg=self.model.config();
        if self.head.is_some() && cfg.head_dim==128 && cfg.gamma()>=7 {15} else {cfg.gamma()}
    }
    pub fn new<S:Store<Value=V>,C:Compute<Value=A>>(model:&'a MtpModel,store:&mut S,c:&C)->Result<Self> {
        Self::with_packed(model,store,c,None,None)
    }
    pub fn with_packed<S:Store<Value=V>,C:Compute<Value=A>>(model:&'a MtpModel,store:&mut S,c:&C,layer_format:Option<ExpertCodes>,head_format:Option<ExpertCodes>)->Result<Self> {
        let cfg=model.config();let (h,q,kv,inter,latent)=(cfg.hidden_size,cfg.q_width(),cfg.kv_width(),cfg.intermediate_size,cfg.markov_rank);
        let head=if let Some(f)=head_format {packed::Packed::new(store,model.head(),cfg.unpadded_vocab_size,h,f)?} else {None};
        let layer_matrix=|store:&mut S,bits:&[u16],rows,columns| {
            if let Some(f)=layer_format && let Some(packed)=packed::Packed::new(store,bits,rows,columns,f)? {
                return Ok(Matrix {weight:Projection::Packed(packed),rows,columns});
            }
            matrix(store,bits,rows,columns)
        };
        // Sibling panels of one input share one allocation and one launch; their own matrices are not stored again.
        let mut layers=Vec::with_capacity(model.layers().len());
        for l in model.layers() {
            let qkv=packed::siblings(store,&[(&l.q_proj,q),(&l.k_proj,kv),(&l.v_proj,kv)],h,layer_format)?;
            let gate_up=packed::siblings(store,&[(&l.gate_proj,inter),(&l.up_proj,inter)],h,layer_format)?;
            let own=|store:&mut S,fused:bool,bits:&[u16],rows,columns| if fused {Ok(Matrix {weight:Projection::Fused,rows,columns})} else {layer_matrix(store,bits,rows,columns)};
            let (a,b)=(qkv.is_some(),gate_up.is_some());
            layers.push(Layer {q:own(store,a,&l.q_proj,q,h)?,k:own(store,a,&l.k_proj,kv,h)?,v:own(store,a,&l.v_proj,kv,h)?,
                k_ctx:layer_matrix(store,&l.k_proj_ctx,kv,h)?,v_ctx:layer_matrix(store,&l.v_proj_ctx,kv,h)?,o:layer_matrix(store,&l.o_proj,h,q)?,
                gate:own(store,b,&l.gate_proj,inter,h)?,up:own(store,b,&l.up_proj,inter,h)?,down:layer_matrix(store,&l.down_proj,h,inter)?,qkv,gate_up});
        }
        let norms=model.layers().iter().map(|l|Ok(Norms {input:c.upload(&l.input_layernorm)?,post:c.upload(&l.post_attention_layernorm)?,
            q:c.upload(&l.q_norm)?,k:c.upload(&l.k_norm)?,k_ctx:c.upload(&l.k_norm_ctx)?,dflare:c.upload(&l.dflare_norm)?})).collect::<Result<Vec<_>>>()?;
        let (rnn,rnn_bias)=model.rnn_proj();let (fold,fold_bias)=model.fold();
        need(latent.is_multiple_of(2),"packed Markov embedding needs an even rank")?;
        let words=model.markov_w1().as_chunks::<2>().0.iter().map(|bits|f32::from_bits(u32::from(bits[0])|(u32::from(bits[1])<<16))).collect::<Vec<_>>();
        let phases=c.upload(&model.rope_phases(0,phase_rows(cfg.max_position_embeddings,cfg.head_dim)))?;
        let fold_panel=packed::panel(cfg.unpadded_vocab_size,latent,head_format);
        let (fold,fold_bias)=match fold_panel {
            Some(unit)=>(packed::upload_panel(store,unit,fold,fold_bias)?,None),
            None=>(bf16(store,fold,cfg.unpadded_vocab_size,latent)?,Some(c.upload(fold_bias)?)),
        };
        let (rnn,rnn_split)=if fold_panel.is_some()
            && let (Some(dyn_u),Some(h_u))=(packed::panel(3*latent,2*latent,layer_format),packed::panel(3*latent,h,layer_format))
        {
            let (rows,cols)=(3*latent,2*latent+2*h);
            need(rnn.len()==rows*cols,"draft rnn weight shape")?;
            let (mut dyn_bits,mut draft_bits,mut feat_bits)=(Vec::with_capacity(rows*2*latent),Vec::with_capacity(rows*h),Vec::with_capacity(rows*h));
            for row in rnn.chunks_exact(cols) {
                dyn_bits.extend_from_slice(&row[..2*latent]);
                draft_bits.extend_from_slice(&row[2*latent..2*latent+h]);
                feat_bits.extend_from_slice(&row[2*latent+h..]);
            }
            (None,Some(SplitRnn {
                dyn_part:packed::upload_panel(store,dyn_u,&dyn_bits,&[])?,
                draft_part:packed::upload_panel(store,h_u,&draft_bits,&[])?,
                feat_part:packed::upload_panel(store,h_u,&feat_bits,&[])?,
            }))
        } else {
            (Some(layer_matrix(store,rnn,3*latent,2*latent+2*h)?),None)
        };
        Ok(Self {model,bytes:packed_weight_bytes(model,layer_format,head_format)?,head,layers,norms,phases,markov_norm:c.upload(model.markov_norm())?,final_norm:c.upload(model.final_norm())?,
            rnn,rnn_split,fold,
            w1:c.upload(&words)?,rnn_bias:c.upload(rnn_bias)?,fold_bias,mask:c.upload(&model.mask_embed().repeat(cfg.block_size-1))?,
            alpha:model.fuse_alpha().iter().map(|row|c.upload(row)).collect::<Result<_>>()?})
    }
    pub fn bytes(&self)->usize { self.bytes }
    fn phases<C:Compute<Value=A>>(&self,c:&C,start:usize,rows:usize)->Result<A> {
        let width=self.config().head_dim;
        if start.checked_add(rows).is_some_and(|end|end<=c.words(&self.phases)/width) {
            c.range(&self.phases,start*width,rows*width)
        } else {
            c.upload(&self.model.rope_phases(start,rows))
        }
    }
    pub fn chat(&self)->DraftChat<A> {DraftChat::new(self.layers.len(),self.config().max_position_embeddings)}
    pub fn push<C:ResidentProduct<V,Value=A>+Lanes>(&self,c:&C,chat:&mut DraftChat<A>,rows:&[Vec<Vec<f32>>])->Result<()> {
        if rows.is_empty() {return Ok(());}
        let cfg=self.config();let width=cfg.hidden_size;let taps=cfg.target_layer_ids.len();
        need(rows.iter().all(|r|r.len()==taps && r.iter().all(|v|v.len()==width)),"draft tap shape")?;
        let values=(0..taps).map(|t|c.upload(&rows.iter().flat_map(|r|r[t].iter().copied()).collect::<Vec<_>>())).collect::<Result<Vec<_>>>()?;
        self.push_values(c,chat,&values,rows.len())
    }
    pub(super) fn push_layer_kv<C:Compute<Value=A>>(&self,c:&C,chat:&mut DraftChat<A>,layer:usize,k:&A,v:&A,rows:usize)->Result<()> {
        push_layer_kv_for(self.config(),c,chat,layer,k,v,rows)
    }
    pub fn push_values<C:ResidentProduct<V,Value=A>+Lanes>(&self,c:&C,chat:&mut DraftChat<A>,taps:&[A],rows:usize)->Result<()> {
        if rows==0 {return Ok(());}
        let cfg=self.config();let (h,hd,heads)=(cfg.hidden_size,cfg.head_dim,cfg.num_kv_heads);
        let end=chat.len.checked_add(rows).ok_or("draft context overflow")?;
        need(end<=chat.limit && taps.len()==cfg.target_layer_ids.len() && taps.iter().all(|t|c.words(t)==rows*h),"draft context shape")?;
        let stacked=c.join(&taps.iter().collect::<Vec<_>>())?;let phases=self.phases(c,chat.len,rows)?;
        let mix=|layer:usize,weight:&A|->Result<A> {
            let x=c.lanes(DraftOp::Mix{words:rows*h,taps:taps.len()},&stacked,Some(&self.alpha[layer]))?;
            c.draft_norm(&x,rows,h,cfg.rms_norm_eps,weight)
        };
        for (i,l) in self.layers.iter().enumerate() {
            let w=&self.norms[i];let x=mix(i,&w.dflare)?;
            let input=l.k_ctx.prepare(c,&x,rows)?;
            let k=l.k_ctx.rounded_shared(c,&x,rows,input.as_ref())?;let v=l.v_ctx.rounded_shared(c,&x,rows,input.as_ref())?;
            let k=c.draft_norm(&k,rows*heads,hd,cfg.rms_norm_eps,&w.k_ctx)?;
            let k=c.lanes(DraftOp::Rope{rows,heads,width:hd},&k,Some(&phases))?;
            self.push_layer_kv(c,chat,i,&k,&v,rows)?;
        }
        // Only the final committed row becomes the next recurrent feature.
        // Short verification batches can reuse the joined operands directly.
        let feature=if rows<=32 {
            let mixed=mix(cfg.num_layers,&self.markov_norm)?;
            c.range(&mixed,(rows-1)*h,h)?
        } else {
            let last=taps.iter().map(|tap|c.range(tap,(rows-1)*h,h)).collect::<Result<Vec<_>>>()?;
            let tail=c.join(&last.iter().collect::<Vec<_>>())?;
            let x=c.lanes(DraftOp::Mix{words:h,taps:taps.len()},&tail,Some(&self.alpha[cfg.num_layers]))?;
            c.draft_norm(&x,1,h,cfg.rms_norm_eps,&self.markov_norm)?
        };
        chat.feature=Some(feature);chat.len=end;Ok(())
    }
    pub fn propose<C:ResidentProduct<V,Value=A>+Lanes,W:Weights<C>>(&self,c:&C,w:&W,chat:&DraftChat<A>,anchor:u32,embed:&[f32],depth:usize,_threads:usize)->Result<Vec<u32>> {
        self.propose_value(c,w,chat,anchor,&c.upload(embed)?,depth)
    }
    pub fn propose_value<C:ResidentProduct<V,Value=A>+Lanes,W:Weights<C>>(&self,c:&C,w:&W,chat:&DraftChat<A>,anchor:u32,embed:&A,depth:usize)->Result<Vec<u32>> {
        let cfg=self.config();let (rows,h,hd,heads,kvheads,kv)=(cfg.block_size,cfg.hidden_size,cfg.head_dim,cfg.num_heads,cfg.num_kv_heads,cfg.kv_width());
        need(depth>0 && depth<rows && c.words(embed)==h && chat.len+rows<=cfg.max_position_embeddings,"draft block shape")?;
        let feature=chat.feature.as_ref().ok_or("draft context empty")?;
        let mut res=c.join(&[embed,&self.mask])?;
        let phases=self.phases(c,chat.len,rows)?;
        let norm=|x:&A,lines,width,weight:&A|c.draft_norm(x,lines,width,cfg.rms_norm_eps,weight);
        let flash_attn=hd==128 && self.head.is_some();
        for (i,l) in self.layers.iter().enumerate() {
            let last_layer=i+1==self.layers.len();
            let weights=&self.norms[i];let x=norm(&res,rows,h,&weights.input)?;
            let [q_all,k,v_new]=l.qkv(c,&x,rows)?;
            let (q_rows,q_in,q_phases,res_in)=if last_layer {
                (depth,c.range(&q_all,heads*hd,depth*heads*hd)?,c.range(&phases,hd,depth*hd)?,c.range(&res,h,depth*h)?)
            } else {
                (rows,q_all,c.range(&phases,0,rows*hd)?,res)
            };
            let q=norm(&q_in,q_rows*heads,hd,&weights.q)?;
            let q=c.lanes(DraftOp::Rope{rows:q_rows,heads,width:hd},&q,Some(&q_phases))?;
            let k=norm(&k,rows*kvheads,hd,&weights.k)?;
            let k=c.lanes(DraftOp::Rope{rows,heads:kvheads,width:hd},&k,Some(&phases))?;
            let start=if cfg.is_local(i) {chat.len.saturating_sub(cfg.sliding_window_size.saturating_sub(1))} else {0};
            let base=chat.local_base[i];
            need(base<=start,"draft local window base passed read start")?;
            let rel_start=start-base;
            let keys=chat.len-start+rows;let (ck,cv)=chat.kv[i].as_ref().ok_or("draft keys missing")?;
            let append=|ctx:&A,new:&A|c.join(&[&c.range(ctx,rel_start*kv,(chat.len-start)*kv)?,new]);
            let k=append(ck,&k)?;let v=append(cv,&v_new)?;
            let a=if flash_attn {
                let splits=if keys>=256 {16} else if keys>=64 {8} else {1};
                let part=c.lanes3(DraftOp::FlashAttn{rows:q_rows,heads,kv:kvheads,width:hd,keys,splits,causal_base:0,row_off:0},&q,&k,&v)?;
                if splits>1 {c.lanes(DraftOp::FlashAttnMerge{rows:q_rows,heads,width:hd,splits},&part,None)?} else {part}
            } else {
                let scores=c.lanes(DraftOp::Scores{rows:q_rows,heads,kv:kvheads,width:hd,keys},&q,Some(&k))?;
                let probs=c.lanes(DraftOp::Normalizer{queries:q_rows*heads,keys},&scores,None)?;
                c.lanes(DraftOp::Values{rows:q_rows,heads,kv:kvheads,width:hd,keys},&probs,Some(&v))?
            };
            res=c.draft_sum(&res_in,&l.o.rounded(c,&a,q_rows)?)?;
            let y=norm(&res,q_rows,h,&weights.post)?;
            let [gate,up]=l.gate_up(c,&y,q_rows)?;
            let act=c.lanes(DraftOp::Gate(c.words(&gate)),&gate,Some(&up))?;
            res=c.draft_sum(&res,&l.down.rounded(c,&act,q_rows)?)?;
        }
        let drafted_in=if self.layers.is_empty() {c.range(&res,h,depth*h)?} else {res};
        let drafted=norm(&drafted_in,depth,h,&self.final_norm)?;
        self.markov_walk(c,w,&drafted,feature,anchor,depth,false,0.0).map(|(ids,_)|ids)
    }
    /// Zero-copy proposal with per-step top-1 softmax probabilities: stages `1 + depth.max(gamma)` temporary K/V rows
    /// directly past `chat.len` in `chat.kv` when room is reserved (leaving `chat.len` unchanged), applying 3-tier
    /// block-causal masking when `depth > gamma` so block-1 rows `0..8` match a `gamma`-token proposal identically.
    /// When `abort_below > 0.0` and `depth > 1`, checks step 0 early and returns after 1 token if its probability is
    /// below `abort_below`, skipping the remaining Markov steps.
    #[allow(clippy::too_many_arguments)]
    pub fn propose_confident<C:ResidentProduct<V,Value=A>+Lanes,W:Weights<C>>(&self,c:&C,w:&W,chat:&mut DraftChat<A>,anchor:u32,embed:&A,depth:usize,abort_below:f32)->Result<(Vec<u32>,Vec<f32>)> {
        let cfg=self.config();let (h,hd,heads,kvheads,kv)=(cfg.hidden_size,cfg.head_dim,cfg.num_heads,cfg.num_kv_heads,cfg.kv_width());
        need(c.words(embed)==h && depth>0 && depth<=15,"draft block shape")?;
        let mask_rows=depth.max(cfg.gamma());
        let rows=1+mask_rows;
        let kv_pos=chat.len;
        need(kv_pos+rows<=cfg.max_position_embeddings,"draft block position overflow")?;
        let mask_slice=if mask_rows==cfg.gamma() {
            c.range(&self.mask,0,mask_rows*h)?
        } else {
            let g=cfg.gamma();
            let mut parts:Vec<A>=Vec::new();
            let mut rem=mask_rows;
            while rem>0 {
                let take=rem.min(g);
                parts.push(c.range(&self.mask,0,take*h)?);
                rem-=take;
            }
            c.join(&parts.iter().collect::<Vec<_>>())?
        };
        let mut res=c.join(&[embed,&mask_slice])?;
        let phases=self.phases(c,kv_pos,rows)?;
        let norm=|x:&A,lines,width,weight:&A|c.draft_norm(x,lines,width,cfg.rms_norm_eps,weight);
        let flash_attn=hd==128 && self.head.is_some();
        for (i,l) in self.layers.iter().enumerate() {
            let last_layer=i+1==self.layers.len();
            let weights=&self.norms[i];let x=norm(&res,rows,h,&weights.input)?;
            let [q_all,k,v_new]=l.qkv(c,&x,rows)?;
            let (q_rows,q_in,q_phases,res_in)=if last_layer {
                (depth,c.range(&q_all,heads*hd,depth*heads*hd)?,c.range(&phases,hd,depth*hd)?,c.range(&res,h,depth*h)?)
            } else {
                (rows,q_all,c.range(&phases,0,rows*hd)?,res)
            };
            let q=norm(&q_in,q_rows*heads,hd,&weights.q)?;
            let q=c.lanes(DraftOp::Rope{rows:q_rows,heads,width:hd},&q,Some(&q_phases))?;
            let k=norm(&k,rows*kvheads,hd,&weights.k)?;
            let k=c.lanes(DraftOp::Rope{rows,heads:kvheads,width:hd},&k,Some(&phases))?;
            let start=if cfg.is_local(i) {kv_pos.saturating_sub(cfg.sliding_window_size.saturating_sub(1))} else {0};
            let base=chat.local_base[i];
            need(base<=start,"draft local window base passed read start")?;
            let (rel_start,rel_pos)=(start-base,kv_pos-base);
            let keys=kv_pos-start+rows;let (ck,cv)=chat.kv[i].as_mut().ok_or("draft keys missing")?;
            let (k,v)=if c.words(ck)/kv>=rel_pos+rows {
                c.state_fill(ck,rel_pos*kv,&k)?;c.state_fill(cv,rel_pos*kv,&v_new)?;
                (c.range(ck,rel_start*kv,keys*kv)?,c.range(cv,rel_start*kv,keys*kv)?)
            } else {
                let append=|ctx:&A,new:&A|c.join(&[&c.range(ctx,rel_start*kv,(kv_pos-start)*kv)?,new]);
                (append(ck,&k)?,append(cv,&v_new)?)
            };
            let a=if flash_attn {
                let splits=if keys>=256 {16} else if keys>=64 {8} else {1};
                let causal_base=if rows>cfg.block_size {(kv_pos-start)+cfg.block_size} else {0};
                let row_off=usize::from(last_layer);
                let part=c.lanes3(DraftOp::FlashAttn{rows:q_rows,heads,kv:kvheads,width:hd,keys,splits,causal_base,row_off},&q,&k,&v)?;
                if splits>1 {c.lanes(DraftOp::FlashAttnMerge{rows:q_rows,heads,width:hd,splits},&part,None)?} else {part}
            } else {
                let scores=c.lanes(DraftOp::Scores{rows:q_rows,heads,kv:kvheads,width:hd,keys},&q,Some(&k))?;
                let probs=c.lanes(DraftOp::Normalizer{queries:q_rows*heads,keys},&scores,None)?;
                c.lanes(DraftOp::Values{rows:q_rows,heads,kv:kvheads,width:hd,keys},&probs,Some(&v))?
            };
            res=c.draft_sum(&res_in,&l.o.rounded(c,&a,q_rows)?)?;
            let y=norm(&res,q_rows,h,&weights.post)?;
            let [gate,up]=l.gate_up(c,&y,q_rows)?;
            let act=c.lanes(DraftOp::Gate(c.words(&gate)),&gate,Some(&up))?;
            res=c.draft_sum(&res,&l.down.rounded(c,&act,q_rows)?)?;
        }
        let feature=chat.feature.as_ref().ok_or("draft context empty")?;
        let drafted_in=if self.layers.is_empty() {c.range(&res,h,depth*h)?} else {res};
        let drafted=norm(&drafted_in,depth,h,&self.final_norm)?;
        self.markov_walk(c,w,&drafted,feature,anchor,depth,true,abort_below)
    }
    #[allow(clippy::too_many_arguments)]
    fn markov_walk<C:ResidentProduct<V,Value=A>+Lanes,W:Weights<C>>(&self,c:&C,w:&W,drafted:&A,feature:&A,anchor:u32,depth:usize,confidence:bool,abort_below:f32)->Result<(Vec<u32>,Vec<f32>)> {
        const PROB_CHUNK:usize=512;
        let cfg=self.config();let h=cfg.hidden_size;
        let combined=match &self.head {
            Some(packed::Packed::Panel{weight,..}) if self.fold_bias.is_none() && depth<=16=>Some(weight),_=>None,
        };
        let mut base=if combined.is_some() {None} else {Some(match &self.head {
            Some(head @ packed::Packed::Panel{..})=>head.dot(c,drafted,depth)?,
            Some(head)=>c.round(&head.dot(c,drafted,depth)?)?,
            None=>c.round(&w.product_rows(c,Weight::Head,drafted,depth)?)?,
        })};
        let vocab=base.as_ref().map_or(cfg.unpadded_vocab_size,|v|c.words(v)/depth);need(vocab>0 && vocab<=cfg.unpadded_vocab_size,"draft vocabulary differs")?;
        let rank=cfg.markov_rank;
        let (bound,epsilon)=(cfg.final_logit_softcapping as f32,engine_model::norm_cap::reciprocal(cfg.final_logit_softcapping));
        need((anchor as usize)<cfg.vocab_size,"draft ID outside embedding")?;
        let parse_pick=|row:&[f32]|->Result<u32> {
            need(row[2]==1.0 && row[0].is_finite() && row[1].is_finite(),"nonfinite draft logits")?;
            need(row[1]>=0.0 && row[1].fract()==0.0 && (row[1] as usize)<cfg.vocab_size,"draft ID outside embedding")?;
            Ok(row[1] as u32)
        };
        let parse_prob=|slice:&[f32]|->f32 {
            let sum:f64=slice.iter().map(|&v|f64::from(v)).sum();
            if sum>0.0 && sum.is_finite() {(1.0/sum).clamp(0.0,1.0) as f32} else {0.0}
        };
        let per_part=if !confidence {0} else if self.fold_bias.is_none() {1} else {vocab.div_ceil(PROB_CHUNK)};
        let early_check=confidence && abort_below>0.0 && depth>1;
        let mut picks:Vec<A>=Vec::with_capacity(depth);
        let mut parts:Vec<A>=Vec::with_capacity(if confidence {depth} else {0});
        let (mut first_id,mut first_prob)=(None,None);
        let factored=if let Some(split)=&self.rnn_split && self.fold_bias.is_none() && depth<=16 {
            let df=c.join(&[drafted,feature])?;
            let packed_df=c.lanes(DraftOp::PackActivations(c.words(&df)),&df,None)?;
            let h_packed=(h/64)*18;
            let packed_drafted=c.range(&packed_df,0,depth*h_packed)?;
            let packed_feat=c.range(&packed_df,depth*h_packed,h_packed)?;
            let draft_out=c.draft_panel(DraftOp::Panel{rows:3*rank,columns:h,inputs:depth},&split.draft_part,&packed_drafted,None)?;
            let feat_out=c.draft_panel(DraftOp::Panel{rows:3*rank,columns:h,inputs:1},&split.feat_part,&packed_feat,None)?;
            let static_base=c.lanes3(DraftOp::MarkovStaticBase{rows:depth,width:3*rank},&draft_out,&feat_out,&self.rnn_bias)?;
            Some((split,packed_drafted,static_base))
        } else {None};
        let mut state=if factored.is_none() {Some(c.zeros(rank)?)} else {None};
        let mut prev_result:Option<A>=None;
        let mut stop0:Option<A>=None;
        let stop_above_sum=if early_check {((1.0f64/f64::from(abort_below)) as f32).next_up()} else {0.0};
        for s in 0..depth {
            let head_row=|base:&Option<A>|->Result<A> {
                let first=usize::from(combined.is_some());
                c.range(base.as_ref().ok_or("missing draft head rows")?,(s-first)*vocab,vocab)
            };
            let pick=if let Some((split,packed_drafted,static_base))=&factored {
                let gated=stop0.is_some();
                let packed_dyn=match prev_result.as_ref() {
                    None=>c.lanes(DraftOp::MarkovPrep{latent:rank,vocab:cfg.vocab_size,anchor,from_pick:false},&self.w1,None)?,
                    Some(prev)=>c.lanes3(DraftOp::MarkovPrep{latent:rank,vocab:cfg.vocab_size,anchor:0,from_pick:true},&self.w1,prev,state.as_ref().ok_or("missing markov state")?)?,
                };
                let dyn_out=if let Some(flag)=stop0.as_ref() {
                    c.draft_panel(DraftOp::PanelGated{rows:3*rank,columns:2*rank,inputs:1},&split.dyn_part,&packed_dyn,Some(flag))?
                } else {
                    c.draft_panel(DraftOp::Panel{rows:3*rank,columns:2*rank,inputs:1},&split.dyn_part,&packed_dyn,None)?
                };
                let base_row=c.range(static_base,s*3*rank,3*rank)?;
                let stepped=match state.as_ref() {
                    None=>c.lanes(DraftOp::MarkovStep{latent:rank,has_state:false,gated},&dyn_out,Some(&base_row))?,
                    Some(st)=>c.lanes3(DraftOp::MarkovStep{latent:rank,has_state:true,gated},&dyn_out,&base_row,st)?,
                };
                state=Some(c.range(&stepped,0,rank)?);
                let packed_latent=c.range(&stepped,rank,(rank/64)*18+usize::from(gated))?;
                let tile_count=vocab.div_ceil(16);
                let tiles=if let Some(head)=combined.filter(|_|s==0) {
                    let head_in=c.join(&[packed_drafted,&packed_latent])?;
                    let op=DraftOp::HeadFold{rows:vocab,hidden:h,latent:rank,inputs:depth,epsilon,bound,confidence};
                    let output=c.draft_head_fold(op,head,&self.fold,&head_in)?;
                    let row_words=(depth-1)*vocab;
                    if row_words>0 {base=Some(c.range(&output,0,row_words)?);}
                    c.range(&output,row_words,tile_count*4)?
                } else {
                    c.draft_panel(DraftOp::FoldPick{rows:vocab,columns:rank,epsilon,bound,confidence,gated},&self.fold,&packed_latent,Some(&head_row(&base)?))?
                };
                let step_stop=if s==0 {stop_above_sum} else {0.0};
                let result=if tile_count>128 {
                    let blocks=tile_count.div_ceil(256).clamp(1,48).min(tile_count);
                    let reduced=c.lanes(DraftOp::FoldReduce{tiles:tile_count,blocks,confidence,gated},&tiles,stop0.as_ref())?;
                    c.lanes(DraftOp::FoldFinish{tiles:blocks,confidence,stop_above_sum:step_stop,gated},&reduced,stop0.as_ref())?
                } else {
                    c.lanes(DraftOp::FoldFinish{tiles:tile_count,confidence,stop_above_sum:step_stop,gated},&tiles,stop0.as_ref())?
                };
                if s==0 && early_check {stop0=Some(c.range(&result,4,1)?);}
                prev_result=Some(c.range(&result,0,5)?);
                if confidence {parts.push(c.range(&result,3,1)?);}
                c.range(&result,0,3)?
            } else {
                let st=state.as_ref().ok_or("missing markov state")?;
                let embedding=match picks.last() {
                    None=>c.lanes(DraftOp::Unpack(rank),&c.range(&self.w1,anchor as usize*(rank/2),rank/2)?,None)?,
                    Some(pick)=>c.lanes(DraftOp::UnpackAt{width:rank,rows:cfg.vocab_size},&self.w1,Some(pick))?,
                };
                let u=c.join(&[st,&embedding,&c.range(drafted,s*h,h)?,feature])?;
                let rnn=self.rnn.as_ref().ok_or("missing draft rnn")?;
                let proj=c.draft_sum(&rnn.dot(c,&u,1)?,&self.rnn_bias)?;
                let pair=c.lanes(DraftOp::Recurrent(rank),&proj,Some(st))?;
                state=Some(c.range(&pair,0,rank)?);let latent=c.range(&pair,rank,rank)?;
                if let Some(bias)=&self.fold_bias {
                    let head_row=head_row(&base)?;
                    let dot=c.resident_product_rows(&self.fold,0..vocab*rank/2,vocab,rank,&latent,1,true)?;
                    let logits=c.lanes3(DraftOp::MarkovLogits{n:vocab,epsilon,bound},&dot,&c.range(bias,0,vocab)?,&head_row)?;
                    let pick=c.greedy_on_device(&logits)?;
                    if confidence {parts.push(c.lanes(DraftOp::ProbPart{n:vocab,chunk:PROB_CHUNK},&logits,Some(&pick))?);}
                    pick
                } else {
                    let tiles=if let Some(head)=combined.filter(|_|s==0) {
                        let input=c.join(&[drafted,&latent])?;
                        let packed=c.lanes(DraftOp::PackActivations(c.words(&input)),&input,None)?;
                        let op=DraftOp::HeadFold{rows:vocab,hidden:h,latent:rank,inputs:depth,epsilon,bound,confidence};
                        let output=c.draft_head_fold(op,head,&self.fold,&packed)?;
                        let row_words=(depth-1)*vocab;
                        if row_words>0 {base=Some(c.range(&output,0,row_words)?);}
                        c.range(&output,row_words,vocab.div_ceil(16)*4)?
                    } else {
                        let input=c.lanes(DraftOp::PackActivations(rank),&latent,None)?;
                        c.draft_panel(DraftOp::FoldPick{rows:vocab,columns:rank,epsilon,bound,confidence,gated:false},&self.fold,&input,Some(&head_row(&base)?))?
                    };
                    let tile_count=vocab.div_ceil(16);
                    let result=if tile_count>128 {
                        let blocks=tile_count.div_ceil(256).clamp(1,48).min(tile_count);
                        let reduced=c.lanes(DraftOp::FoldReduce{tiles:tile_count,blocks,confidence,gated:false},&tiles,None)?;
                        c.lanes(DraftOp::FoldFinish{tiles:blocks,confidence,stop_above_sum:0.0,gated:false},&reduced,None)?
                    } else {
                        c.lanes(DraftOp::FoldFinish{tiles:tile_count,confidence,stop_above_sum:0.0,gated:false},&tiles,None)?
                    };
                    if confidence {parts.push(c.range(&result,3,1)?);}
                    c.range(&result,0,3)?
                }
            };
            picks.push(pick);
            if s==0 && early_check && factored.is_none() {
                let words=c.read(&c.join(&[&picks[0],&parts[0]])?)?;
                need(words.len()==3+per_part,"missing device draft pick")?;
                let (id0,p0)=(parse_pick(&words[..3])?,parse_prob(&words[3..]));
                if p0<abort_below {return Ok((vec![id0],vec![p0]));}
                first_id=Some(id0);first_prob=Some(p0);
            }
        }
        let start=usize::from(first_id.is_some());
        let rem=depth-start;
        let mut refs:Vec<&A>=picks[start..].iter().collect();
        if confidence {refs.extend(parts[start..].iter());}
        let words=c.read(&c.join(&refs)?)?;drop(picks);drop(parts);
        need(words.len()==3*rem+rem*per_part,"missing device draft pick")?;
        if early_check && first_id.is_none() {
            let (id0,p0)=(parse_pick(&words[..3])?,parse_prob(&words[3*rem..3*rem+per_part]));
            if p0<abort_below {return Ok((vec![id0],vec![p0]));}
        }
        let mut ids=Vec::with_capacity(depth);
        if let Some(id0)=first_id {ids.push(id0);}
        for row in words[..3*rem].as_chunks::<3>().0 {ids.push(parse_pick(row)?);}
        let mut probs=Vec::with_capacity(depth);
        if let Some(p0)=first_prob {probs.push(p0);}
        if confidence {
            for s in 0..rem {probs.push(parse_prob(&words[3*rem+s*per_part..3*rem+(s+1)*per_part]));}
        } else {probs.resize(depth,1.0);}
        Ok((ids,probs))
    }
}
#[cfg(test)]
mod prefix_tests {
    use super::*;
    use crate::bounded_decoder::Host;

    fn test_cfg(num_layers:usize)->MtpConfig {
        MtpConfig {
            hidden_size:8,intermediate_size:16,num_layers,num_heads:2,num_kv_heads:2,head_dim:4,
            sliding_window_size:4,block_size:8,markov_rank:4,vocab_size:32,unpadded_vocab_size:32,
            max_position_embeddings:64,rms_norm_eps:1e-6,rope_theta:10000.0,final_logit_softcapping:30.0,
            local_layer_ids:vec![0],target_layer_ids:vec![0],
        }
    }

    #[test]
    fn draft_chat_copy_prefix_packs_restores_and_isolates_buffers() {
        let cfg=test_cfg(2);
        let kv_w=cfg.kv_width();
        let len=5usize;
        let cap=kv_capacity(len,cfg.block_size*2,64);
        let mut src=DraftChat::new(2,64);
        for l in 0..2 {
            let mut k=vec![0.0f32;cap*kv_w];
            let mut v=vec![0.0f32;cap*kv_w];
            for i in 0..len*kv_w {k[i]=(l*100+i+1) as f32;v[i] = -((l*100+i+1) as f32);}
            for i in len*kv_w..cap*kv_w {k[i]=999.0;v[i] = -999.0;}
            src.kv[l]=Some((k,v));
        }
        src.feature=Some((1..=cfg.hidden_size).map(|i|i as f32).collect());
        src.len=len;

        let kept=src.copy_prefix(&Host,&cfg,false).unwrap();
        assert_eq!((kept.len,kept.limit),(len,64));
        // Local layer 0 keeps only the visible sliding window (sliding_window_size - 1 = 3 rows, local_base = 2).
        assert_eq!(kept.local_base,[2,0]);
        let (lk,lv)=kept.kv[0].as_ref().unwrap();
        assert_eq!((lk.len(),lv.len()),(3*kv_w,3*kv_w));
        assert_eq!(lk.as_slice(),&src.kv[0].as_ref().unwrap().0[2*kv_w..len*kv_w]);
        assert_eq!(lv.as_slice(),&src.kv[0].as_ref().unwrap().1[2*kv_w..len*kv_w]);
        // Global layer 1 keeps all committed rows (paged to 1 page of 8 rows on Host).
        let (gk,gv)=kept.kv[1].as_ref().unwrap();
        assert_eq!(&gk[..len*kv_w],&src.kv[1].as_ref().unwrap().0[..len*kv_w]);
        assert_eq!(&gv[..len*kv_w],&src.kv[1].as_ref().unwrap().1[..len*kv_w]);
        assert_eq!(kept.feature,src.feature);

        let mut restored=kept.copy_prefix(&Host,&cfg,true).unwrap();
        assert_eq!((restored.len,restored.limit,restored.local_base.as_slice()),(len,64,&[2,0][..]));
        let (rk,rv)=restored.kv[0].as_mut().unwrap();
        assert!(rk.len()>=3*kv_w && rv.len()>=3*kv_w);
        assert_eq!(&rk[..3*kv_w],kept.kv[0].as_ref().unwrap().0.as_slice());
        rk[0] = -42.0;rv[0]=42.0;
        assert_eq!(kept.kv[0].as_ref().unwrap().0[0],(2*kv_w+1) as f32,"mutating restored copy must not affect kept snapshot");
    }
}
