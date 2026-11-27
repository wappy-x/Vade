//! The Inkling kernel family: the third image (`sm_121a`, crate `spark-inkling-kernels`, Ruling R6h), loaded when one
//! of its entries is first launched or at a preload. Its entries are the Inkling 2 Mini block arms and the packed product; no other model
//! launches them, and the general image does not hold them.
use super::*;

impl<'d> Backend<'d> {
    pub fn draft_lanes(&self,op:crate::draft_ops::DraftOp,input:[Option<&DeviceVector<'_>>;3])->Result<DeviceVector<'d>,String> {
        if self.arm_capture() {return Err("draft lane operation inside a stream capture".into());}
        let (lens,out,lanes,[kind,a,b,c,d,e])=op.layout().ok_or("invalid draft operation dimensions")?;
        let allow_seg_kv = matches!(op, crate::draft_ops::DraftOp::FlashAttn { .. } | crate::draft_ops::DraftOp::Scores { .. } | crate::draft_ops::DraftOp::Values { .. });
        let mut ptr=[0;3];
        for (i,v) in input.into_iter().enumerate() {match v {
            Some(v) if lens[i]>0 && allow_seg_kv && i>=1=>{self.check_vector_prefix(v,lens[i])?;ptr[i]=v.device_address();}
            Some(v) if lens[i]>0 && !v.is_segmented()=>{self.check_vector(v,lens[i])?;ptr[i]=v.device_address();}
            None if lens[i]==0=>(),_=>return Err("draft lane input extent differs".into()),
        }}
        if matches!(kind,13|14|17|19|28) && (ptr[0]%16!=0 || (kind==17 && ptr[1]%16!=0)) {return Err("unaligned draft panel".into());}
        let result=self.vector(out)?;
        if result.is_segmented() {return Err("draft lane output must be contiguous".into());}
        if kind==20 && ((ptr[0]|ptr[1]|ptr[2]|result.device_address())%16!=0) {return Err("unaligned draft flash attention".into());}
        if kind==21 && ((ptr[0]|result.device_address())%16!=0) {return Err("unaligned draft flash attention merge".into());}
        use crate::resident_launch::{Command,Kind};
        let (kind,op)=(match kind {17|28=>Kind::DraftHeadFold,13|14|19=>Kind::DraftPanel,_=>Kind::DraftLanes},kind);
        self.require_arm_kernels(&[kind.kernel()])?;
        let words=[op,ptr[0],ptr[1],ptr[2],result.device_address(),lanes as u64,a,b,c,d,e,0];
        let mut command=Command::new(kind);
        for (i,word) in words.into_iter().enumerate() {command.word(i,word);}
        let threads=if matches!(op,13|19) && (((c>>16)&7)==0) && lanes<=128*128 {32u32} else {128u32};
        let blocks=u32::try_from(lanes.div_ceil(threads as usize)).map_err(|_|"draft lane grid overflow")?;
        let shared=if op==22 {64u32} else {0u32};
        if !self.record_resident_block(kind,command,[blocks,1,1,threads,shared])? {
            let table=self.launch_inkling_command(kind,command,blocks,threads,shared,0)?;
            self.resident_completion()?;
            drop(table);
        }
        Ok(result)
    }
    /// Rounded weighted RMS norm for the draft model (`Kind::FastNorm`), working both inside a recorded resident
    /// queue and during unrecorded draft pushes (`rows_unrecorded`), eliminating the `resident_launch_block_arms`
    /// fallback while keeping `warm_command!` prefetches strictly in bounds (`pad_commands = 4`).
    #[allow(clippy::too_many_arguments)]
    pub fn draft_fast_norm_into(
        &self,rows:usize,dim:usize,eps:f32,prefill:bool,
        input:&DeviceVector<'_>,weight:&DeviceVector<'_>,output:&mut DeviceVector<'_>,
    )->Result<(),String> {
        use crate::{fast_ops::{self,Span},resident_launch::{Command,Kind}};
        if self.arm_capture() {return Err("draft fast norm: refused inside a stream capture".into());}
        self.require_resident_stream()?;
        let span=|v:&DeviceVector<'_>|Span {address:v.device_address(),bytes:v.len() as u64*4};
        fast_ops::check_norm(rows,dim,true,eps,span(input),Some(span(weight)),span(output))?;
        self.check_vector(input,rows*dim)?;
        self.check_vector(weight,dim)?;
        self.check_vector(output,rows*dim)?;
        if input.is_segmented() || weight.is_segmented() {return Err("draft fast norm: input and weight must be contiguous".into());}
        if output.is_segmented() || !output.writable_from(0) {return Err("draft fast norm: output requires private contiguous storage".into());}
        self.require_arm_kernels(&[Kind::FastNorm.kernel()])?;
        self.resident_completion()?;
        let mut command=Command::new(Kind::FastNorm);
        let words=[
            input.device_address(),weight.device_address(),output.device_address(),
            rows as u64,dim as u64,1u64,u64::from(eps.to_bits()),
            0,0,u64::from(prefill),
        ];
        for (i,w) in words.into_iter().enumerate() {command.word(i,w);}
        let blocks=fast_ops::norm_blocks(rows,dim,true);
        let threads=fast_ops::NORM_THREADS as u32;
        if !self.record_resident_block(Kind::FastNorm,command,[blocks,1,1,threads,0])? {
            let table=self.launch_inkling_command(Kind::FastNorm,command,blocks,threads,0,4)?;
            self.resident_completion()?;
            drop(table);
        } else {
            self.resident_completion()?;
        }
        Ok(())
    }
    /// Batched draft flash attention across `1..=32` chats (op 27 in `inkling_draft_lanes`), reading each chat's
    /// committed KV prefix `(ck, cv)` directly (BF16-packed or FP32) plus its temporary `(k_new, v_new)` rows with
    /// zero staging copies. Uses 3-head GQA warp sharing when `heads == kv * 3`.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_flash_attn_batch(
        &self,
        q:&DeviceVector<'_>,k_new:&DeviceVector<'_>,v_new:&DeviceVector<'_>,
        chats:&[(&DeviceVector<'_>,&DeviceVector<'_>,usize,usize,usize)],
        heads:usize,kv:usize,width:usize,q_rows:usize,new_rows:usize,
        kv_bf16:bool,k_new_bf16:bool,causal_base:usize,row_off:usize,
    )->Result<DeviceVector<'d>,String> {
        if self.arm_capture() {return Err("draft flash attention batch inside a stream capture".into());}
        let batch=chats.len();
        if !(1..=32).contains(&batch) || width!=128 || kv==0 || heads==0 || !heads.is_multiple_of(kv)
            || q_rows==0 || q_rows>=65536 || new_rows==0 || new_rows>=65536 || row_off>=256
            || causal_base>u32::MAX as usize
        {
            return Err("invalid draft flash attention batch dimensions".into());
        }
        let kv_words=kv.checked_mul(if kv_bf16 {64} else {128}).ok_or("draft kv stride overflow")?;
        let k_new_words=kv.checked_mul(if k_new_bf16 {64} else {128}).ok_or("draft k_new stride overflow")?;
        let v_new_words=kv.checked_mul(128).ok_or("draft v_new stride overflow")?;
        let queries=batch.checked_mul(q_rows).and_then(|v|v.checked_mul(heads)).ok_or("draft query count overflow")?;
        let q_len=queries.checked_mul(128).ok_or("draft q extent overflow")?;
        let k_new_len=batch.checked_mul(new_rows).and_then(|v|v.checked_mul(k_new_words)).ok_or("draft k_new extent overflow")?;
        let v_new_len=batch.checked_mul(new_rows).and_then(|v|v.checked_mul(v_new_words)).ok_or("draft v_new extent overflow")?;
        if q.is_segmented() || k_new.is_segmented() || v_new.is_segmented() {return Err("draft flash attention batch requires contiguous q/k_new/v_new".into());}
        self.check_vector(q,q_len)?;
        self.check_vector(k_new,k_new_len)?;
        self.check_vector(v_new,v_new_len)?;
        if !(q.device_address()|k_new.device_address()|v_new.device_address()).is_multiple_of(16) {return Err("unaligned draft flash attention batch operand".into());}
        let mut max_splits=1usize;
        let mut all_kv_16_aligned=kv_bf16;
        let mut chat_words=[(0u64,0u64,0u64);32];
        let align_mask=if kv_bf16 {7u64} else {15u64};
        for (bi,&(ck,cv,rel_start,hist_keys,splits_b)) in chats.iter().enumerate() {
            if splits_b==0 || splits_b>u32::MAX as usize || hist_keys>u32::MAX as usize || causal_base>hist_keys.saturating_add(new_rows) {
                return Err("invalid draft flash attention batch chat entry".into());
            }
            max_splits=max_splits.max(splits_b);
            let (ck_addr,cv_addr)=if hist_keys>0 {
                let end_words=rel_start.checked_add(hist_keys).and_then(|r|r.checked_mul(kv_words)).ok_or("draft kv prefix overflow")?;
                self.check_vector_prefix(ck,end_words)?;
                self.check_vector_prefix(cv,end_words)?;
                let byte_off=(rel_start.checked_mul(kv_words).and_then(|w|w.checked_mul(4)).ok_or("draft kv byte offset overflow")?) as u64;
                let ka=ck.device_address().checked_add(byte_off).ok_or("draft ck address overflow")?;
                let va=cv.device_address().checked_add(byte_off).ok_or("draft cv address overflow")?;
                if ((ka|va)&align_mask)!=0 {return Err("unaligned draft kv plane".into());}
                if ((ka|va)&15)!=0 {all_kv_16_aligned=false;}
                (ka,va)
            } else {(0,0)};
            chat_words[bi]=(ck_addr,cv_addr,(hist_keys as u64)|((splits_b as u64)<<32));
        }
        let out=if max_splits==1 {q_len} else {queries.checked_mul(max_splits).and_then(|v|v.checked_mul(132)).ok_or("draft attn out overflow")?};
        let result=self.vector(out)?;
        if result.is_segmented() || !result.device_address().is_multiple_of(16) {return Err("draft flash attention batch output must be contiguous and aligned".into());}
        let tc_mode=all_kv_16_aligned;
        let (blocks,threads,shared,lanes)=if tc_mode {
            let group=heads/kv;
            let q_tiles=q_rows.checked_mul(group).ok_or("draft tc q_tiles overflow")?.div_ceil(32);
            let total_blocks=batch.checked_mul(max_splits).and_then(|v|v.checked_mul(kv)).and_then(|v|v.checked_mul(q_tiles)).ok_or("draft tc grid overflow")?;
            let blocks=u32::try_from(total_blocks).map_err(|_|"draft flash attention batch grid overflow")?;
            let lanes=total_blocks.checked_mul(256).ok_or("draft tc lanes overflow")?;
            (blocks,256u32,27_776u32,lanes)
        } else {
            let lanes=batch.checked_mul(q_rows).and_then(|v|v.checked_mul(heads)).and_then(|v|v.checked_mul(max_splits)).and_then(|v|v.checked_mul(32)).ok_or("draft attn lanes overflow")?;
            let blocks=u32::try_from(lanes.div_ceil(128)).map_err(|_|"draft flash attention batch grid overflow")?;
            (blocks,128u32,0u32,lanes)
        };
        use crate::resident_launch::{Command,Kind};
        self.require_arm_kernels(&[Kind::DraftLanes.kernel()])?;
        let mut command=Command::new(Kind::DraftLanes);
        let scale_bits=(1.0f32/(width as f32).sqrt()).to_bits() as u64;
        let meta6=scale_bits|((batch as u64)<<32)|(u64::from(kv_bf16)<<48)|(u64::from(k_new_bf16)<<49)|(u64::from(tc_mode)<<50);
        let meta7=(heads as u64)|((kv as u64)<<16)|((q_rows as u64)<<32)|((new_rows as u64)<<48);
        let meta8=(causal_base as u64)|((row_off as u64)<<32);
        let header=[27u64,q.device_address(),k_new.device_address(),v_new.device_address(),result.device_address(),lanes as u64,meta6,meta7,meta8,max_splits as u64,0,0];
        for (i,w) in header.into_iter().enumerate() {command.word(i,w);}
        for (bi,&(ka,va,m)) in chat_words.iter().take(batch).enumerate() {
            command.word(12+bi*3,ka);
            command.word(12+bi*3+1,va);
            command.word(12+bi*3+2,m);
        }
        if !self.record_resident_block(Kind::DraftLanes,command,[blocks,1,1,threads,shared])? {
            let table=self.launch_inkling_command(Kind::DraftLanes,command,blocks,threads,shared,0)?;
            self.resident_completion()?;
            drop(table);
        }
        Ok(result)
    }
    /// Batched split-K merge across `1..=32` chats (op 28 in `inkling_draft_lanes`), each chat merging its own `splits[bi]` partials.
    pub fn draft_flash_attn_merge_batch(
        &self,part:&DeviceVector<'_>,splits:&[usize],heads:usize,q_rows:usize,max_splits:usize,
    )->Result<DeviceVector<'d>,String> {
        if self.arm_capture() {return Err("draft flash attention merge batch inside a stream capture".into());}
        let batch=splits.len();
        if !(1..=32).contains(&batch) || heads==0 || q_rows==0 || max_splits<2 || max_splits>u32::MAX as usize
            || splits.iter().any(|&s|s==0 || s>max_splits)
        {
            return Err("invalid draft flash attention merge batch dimensions".into());
        }
        let queries=batch.checked_mul(q_rows).and_then(|v|v.checked_mul(heads)).ok_or("draft merge queries overflow")?;
        let part_len=queries.checked_mul(max_splits).and_then(|v|v.checked_mul(132)).ok_or("draft merge input overflow")?;
        if part.is_segmented() || !part.device_address().is_multiple_of(16) {return Err("draft merge input must be contiguous and aligned".into());}
        self.check_vector(part,part_len)?;
        let out=queries.checked_mul(128).ok_or("draft merge out overflow")?;
        let result=self.vector(out)?;
        if result.is_segmented() || !result.device_address().is_multiple_of(16) {return Err("draft merge output must be contiguous and aligned".into());}
        let lanes=queries.checked_mul(32).ok_or("draft merge lanes overflow")?;
        use crate::resident_launch::{Command,Kind};
        self.require_arm_kernels(&[Kind::DraftLanes.kernel()])?;
        let mut command=Command::new(Kind::DraftLanes);
        let header=[28u64,part.device_address(),0,0,result.device_address(),lanes as u64,heads as u64,q_rows as u64,max_splits as u64,batch as u64,0,0];
        for (i,w) in header.into_iter().enumerate() {command.word(i,w);}
        let mut split_bytes=[0u8;128];
        for (bi,&s) in splits.iter().enumerate() {split_bytes[bi*4..bi*4+4].copy_from_slice(&(s as u32).to_ne_bytes());}
        command.put(96,&split_bytes);
        let blocks=u32::try_from(lanes.div_ceil(128)).map_err(|_|"draft merge batch grid overflow")?;
        if !self.record_resident_block(Kind::DraftLanes,command,[blocks,1,1,128,0])? {
            let table=self.launch_inkling_command(Kind::DraftLanes,command,blocks,128,0,0)?;
            self.resident_completion()?;
            drop(table);
        }
        Ok(result)
    }
    /// Batched KV scatter across `1..=32` chats (op 31 in `inkling_draft_lanes`): copies `k_src` (`kv_u32_stride` words/row)
    /// and copies or BF16-packs `v_src` (`v_is_fp32`) into each chat's `(k_dst, v_dst)` plane in one launch.
    /// Returns `Ok(false)` when any destination plane's target tail is not privately writable (caller falls back).
    pub fn draft_scatter_kv_batch(
        &self,k_src:&DeviceVector<'_>,v_src:&DeviceVector<'_>,total_rows:usize,kv_u32_stride:usize,v_is_fp32:bool,
        chats:&mut [(&mut DeviceVector<'_>,&mut DeviceVector<'_>,usize,usize,usize)],
    )->Result<bool,String> {
        if self.arm_capture() {return Err("draft scatter kv batch inside a stream capture".into());}
        let batch=chats.len();
        if !(1..=32).contains(&batch) || total_rows==0 || kv_u32_stride==0 || !kv_u32_stride.is_multiple_of(2) {
            return Err("invalid draft scatter kv batch dimensions".into());
        }
        let k_len=total_rows.checked_mul(kv_u32_stride).ok_or("draft scatter k overflow")?;
        let v_len=k_len.checked_mul(if v_is_fp32 {2} else {1}).ok_or("draft scatter v overflow")?;
        if k_src.is_segmented() || v_src.is_segmented() {return Err("draft scatter requires contiguous sources".into());}
        self.check_vector(k_src,k_len)?;
        self.check_vector(v_src,v_len)?;
        if !k_src.device_address().is_multiple_of(8) || !v_src.device_address().is_multiple_of(if v_is_fp32 {16} else {8}) {
            return Err("unaligned draft scatter source".into());
        }
        let mut chat_words=[(0u64,0u64,0u64);32];
        let mut covered=0usize;
        for (bi,(dk,dv,dst_row,src_row,rows_b)) in chats.iter().enumerate() {
            if *rows_b==0 || *src_row!=covered || src_row.checked_add(*rows_b).is_none_or(|e|e>total_rows) {
                return Err("invalid draft scatter row partition".into());
            }
            covered+=*rows_b;
            let dst_off=dst_row.checked_mul(kv_u32_stride).ok_or("draft scatter dst off overflow")?;
            let dst_end=dst_off.checked_add(rows_b.checked_mul(kv_u32_stride).ok_or("draft scatter dst len overflow")?).ok_or("draft scatter dst end overflow")?;
            self.check_vector_prefix(dk,dst_end)?;
            self.check_vector_prefix(dv,dst_end)?;
            if !dk.writable_from(dst_off) || !dv.writable_from(dst_off) {return Ok(false);}
            let ka=dk.device_address().checked_add((dst_off as u64)*4).ok_or("draft scatter dk addr overflow")?;
            let va=dv.device_address().checked_add((dst_off as u64)*4).ok_or("draft scatter dv addr overflow")?;
            if !(ka|va).is_multiple_of(8) || ka==va {return Err("unaligned or aliased draft scatter dst".into());}
            chat_words[bi]=(ka,va,(*src_row as u64)|((*rows_b as u64)<<32));
        }
        if covered!=total_rows {return Err("draft scatter rows do not cover total_rows".into());}
        let lanes=total_rows.checked_mul(kv_u32_stride/2).ok_or("draft scatter lanes overflow")?;
        use crate::resident_launch::{Command,Kind};
        self.require_arm_kernels(&[Kind::DraftLanes.kernel()])?;
        let mut command=Command::new(Kind::DraftLanes);
        let header=[31u64,k_src.device_address(),v_src.device_address(),0,0,lanes as u64,kv_u32_stride as u64,batch as u64,u64::from(v_is_fp32),0,0,0];
        for (i,w) in header.into_iter().enumerate() {command.word(i,w);}
        for (bi,&(ka,va,m)) in chat_words.iter().take(batch).enumerate() {
            command.word(12+bi*3,ka);
            command.word(12+bi*3+1,va);
            command.word(12+bi*3+2,m);
        }
        let blocks=u32::try_from(lanes.div_ceil(128)).map_err(|_|"draft scatter grid overflow")?;
        if !self.record_resident_block(Kind::DraftLanes,command,[blocks,1,1,128,0])? {
            let table=self.launch_inkling_command(Kind::DraftLanes,command,blocks,128,0,0)?;
            self.resident_completion()?;
            drop(table);
        }
        Ok(true)
    }
    /// Fold the shipped projections in FP64. The last column holds the projected bias.
    pub fn fold_markov(&self,head:&[u16],weight:&[f32],bias:&[f32],rows:usize,hidden:usize,latent_width:usize)->Result<(Vec<u16>,Vec<f32>),String> {
        if self.arm_capture() {return Err("Markov fold inside a stream capture".into());}
        if rows==0 || hidden==0 || !hidden.is_multiple_of(2) || latent_width==0 || rows.checked_mul(hidden)!=Some(head.len())
            || hidden.checked_mul(latent_width)!=Some(weight.len()) || bias.len()!=hidden {return Err("invalid Markov fold shape".into());}
        let columns=latent_width.checked_add(1).ok_or("fold column overflow")?;
        let mut projection=Vec::with_capacity(hidden.checked_mul(columns).ok_or("fold projection overflow")?);
        for (row,&offset) in weight.chunks_exact(latent_width).zip(bias) {projection.extend_from_slice(row);projection.push(offset);}
        let projection=self.upload_vector(&projection)?;
        self.require_arm_kernels(&["inkling_markov_fold"])?;
        let mut weights=Vec::with_capacity(rows.checked_mul(latent_width).ok_or("fold weight overflow")?);
        let mut offsets=Vec::with_capacity(rows);
        for first in (0..rows).step_by(2048) {
            let count=(rows-first).min(2048);
            let bytes=head[first*hidden..(first+count)*hidden].iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>();
            let mut left=self.vector(bytes.len()/4)?;self.write_vector_bytes(&mut left,&bytes)?;
            let words=count.checked_mul(columns).ok_or("fold output overflow")?;
            let out=self.vector(words)?;
            if [&left,&projection,&out].iter().any(|v|v.is_segmented()) {return Err("fold requires contiguous matrices".into());}
            let blocks=u32::try_from(words.div_ceil(128)).map_err(|_|"fold grid overflow")?;
            let mut args=[left.device_address(),projection.device_address(),out.device_address(),count as u64,hidden as u64,columns as u64];
            self.launch_checked_expert_payload("inkling_markov_fold",blocks,1,128,0,&mut args)?;
            for row in self.read_vector_values(&out)?.chunks_exact(columns) {
                weights.extend(row[..latent_width].iter().map(|&x|(crate::bf16::rounded(x).to_bits()>>16) as u16));
                offsets.push(crate::bf16::rounded(row[latent_width]));
            }
        }
        Ok((weights,offsets))
    }
}

/// Entries that live in the Inkling image: the kernel names of its kinds (host test).
pub(crate) fn in_inkling_image(name:&str)->bool {matches!(name,"inkling_draft_lanes"|"inkling_draft_panel"|"inkling_draft_head_fold"|"inkling_markov_fold"|"resident_launch_block_arms"|"resident_launch_block_rows"|"resident_launch_plain_staged"|"resident_launch_plain_staged_rows"|"resident_launch_plain_staged_few"|"resident_launch_plain_staged_one"|"resident_launch_plain_staged_narrow"|"resident_launch_plain_packed"|"resident_launch_plain_packed_group"|"resident_launch_plain_packed_picked"|"resident_launch_block_scores"|"resident_launch_block_values"|"resident_launch_block_softmax"|"resident_launch_plain_tc_rows"|"resident_launch_plain_tc_big"|"resident_launch_plain_tc_panel64"|"resident_launch_plain_packed_tc_rows"|"resident_launch_prefill_packed_tc"|"resident_launch_prefill_flash_window"|"resident_launch_prefill_gate_fuse"|"resident_launch_moe_group_route"|"resident_launch_flash_window"|"resident_launch_flash_window_split16"|"resident_launch_flash_window_split32"|"resident_launch_flash_window_reduce"|"resident_launch_fast_mix_blend"|"resident_launch_fast_row_dots"|"resident_launch_fast_norm"|"resident_launch_fast_capped_gate"|"resident_launch_fast_router_pick"|"resident_launch_fast_stream_ops"|"resident_launch_plain_packed_rows"|"resident_launch_embed_rows")}

/// The entry of a rows launch of a block arm: `WindowScores` with its key plane shared takes its own entry
/// (`block_scores::stage`), `SinkSoftmax` with its scores per row its own (`block_attention::softmax_rows_stage`),
/// `WindowValues` with its value plane shared its own (`block_values::stage`);
/// every other rows launch takes the rows entry. One choice for the kernel check of
/// `resident_causal_into` and for `block_arm` (review REVSCORES, LOW 2).
pub(super) fn rows_kind(shape:crate::causal_ops::Shape,count:usize,shared:u32)->crate::resident_launch::Kind {
    use crate::resident_launch::Kind;
    if crate::block_scores::stage(shape,count,shared).is_some() {Kind::BlockScores}
    else if crate::block_attention::softmax_rows_stage(shape,count,shared).is_some() {Kind::BlockSoftmax}
    else if crate::block_values::stage(shape,count,shared).is_some() {Kind::BlockValues}
    else {Kind::BlockRows}
}

/// The backend's state for the Inkling image.
pub(super) struct InklingState<'a> {
    /// The image, loaded when one of its entries is first launched.
    module:std::cell::OnceCell<Module<'a>>,
    /// True while `capture_with_allowance` captures the stream (`capture_batch`, `capture_batch_bounded`,
    /// `capture_decode`). The general flag `context.capturing` is set only by `verify_segment`, and its meaning for the
    /// other families stays as it is; the arms read both (`arm_capture`).
    capturing:Cell<bool>,
    pub(super) panels:RefCell<std::collections::BTreeMap<u64,super::staged_few::Panel<'a>>>,
}
impl InklingState<'_> {
    pub(super) fn new()->Self {Self {module:std::cell::OnceCell::new(),capturing:Cell::new(false),panels:RefCell::new(std::collections::BTreeMap::new())}}
}
impl<'a> Backend<'a> {
    /// Record one scores or values arm with an independent plane for each chat.
    /// Every plane is checked against the complete output and its own one-row extent.
    pub fn block_planes_into(&self,shape:crate::causal_ops::Shape,x:[&DeviceVector<'_>;3],planes:&[&DeviceVector<'_>],output:&mut DeviceVector<'_>)->Result<bool,String> {
        use crate::{block_check::{Operand,operands},inkling_launch::{PLANES_AT,PLANES_MAX},resident_launch::Kind};
        let rows=planes.len();
        if !self.resident_launches_active() {return Ok(false);}
        self.require_resident_stream()?;
        if !(2..=PLANES_MAX).contains(&rows) || !matches!(shape.op,crate::causal_ops::Op::WindowScores|crate::causal_ops::Op::WindowValues) {
            return Err("per-chat planes require a scores or values arm and 2 to 16 chats".into());
        }
        if !output.writable_from(0) {return Err("per-chat attention requires a private output".into());}
        let (kind,stage)=if shape.op==crate::causal_ops::Op::WindowScores {(Kind::BlockScores,crate::block_scores::stage(shape,1,2))}
            else {(Kind::BlockValues,crate::block_values::stage(shape,1,2))};
        let Some((blocks,threads))=stage else {return Ok(false);};
        self.require_arm_kernels(&[kind.kernel()])?;
        let operand=|v:&DeviceVector<'_>|Operand {address:v.device_address(),words:v.len(),contiguous:!v.is_segmented(),own_context:self.check_vector(v,v.len()).is_ok()};
        let counts=crate::block_rows::lengths(shape,rows,2);
        let plane_words=counts.ok_or("invalid per-chat attention shape")?.0[1];
        let [a,b,c]=x.map(operand);
        let out=operand(output);
        let out_span=(output.device_address(),output.device_address()+4*output.len() as u64);
        let args=crate::causal_ops::Arguments {input:[a.address,4,b.address,c.address],weight:4,output:out.address,shape};
        let mut command=crate::resident_launch_encode::causal(args);
        command.kind=kind as u32;
        command.put(crate::block_ops::device::ROWS_AT,&(rows as u32).to_ne_bytes());
        command.put(crate::block_ops::device::SHARED_AT,&2u32.to_ne_bytes());
        for (i,plane) in planes.iter().enumerate() {
            self.check_vector_prefix(plane,plane_words)?;
            let p_end=plane.device_address().checked_add(4*plane.len() as u64).ok_or("per-chat plane ends past address space")?;
            if out_span.0<p_end && plane.device_address()<out_span.1 {return Err("block arm output overlaps an operand".into());}
            let plane=Operand {address:plane.device_address(),words:plane_words,contiguous:true,own_context:true};
            operands(counts,[Some(a),Some(plane),Some(b),Some(c)],None,out,self.arm_capture())?;
            command.put(PLANES_AT+i*8,&plane.address.to_ne_bytes());
        }
        let blocks=u32::try_from(blocks).map_err(|_|"per-chat attention grid overflow")?;
        self.record_resident_block(kind,command,[blocks,rows as u32,1,threads as u32,0])
    }
    /// Record `WindowScores` or `WindowValues` over a virtual chronological plane `(ring[start_slot..], delta)`
    /// without copying `ring` into a temporary buffer. Every operand extent, alignment, context ownership, and
    /// output non-overlap is checked before recording.
    #[allow(clippy::too_many_arguments)]
    pub fn block_ring_delta_into(
        &self, shape: crate::causal_ops::Shape, rows: usize, shared: u32,
        x: [&DeviceVector<'_>; 3], ring: &DeviceVector<'_>, delta: &DeviceVector<'_>,
        ring_rows: usize, start_slot: usize, prefix: usize, output: &mut DeviceVector<'_>,
    ) -> Result<bool, String> {
        use crate::{block_check::{Operand, operands}, inkling_launch::RING_DELTA_AT, resident_launch::Kind};
        if !self.resident_launches_active() { return Ok(false); }
        self.require_resident_stream()?;
        if !matches!(shape.op, crate::causal_ops::Op::WindowScores | crate::causal_ops::Op::WindowValues) {
            return Err("ring-delta attention requires WindowScores or WindowValues".into());
        }
        if !output.writable_from(0) { return Err("ring-delta attention requires a private output".into()); }
        let (kind, stage) = if shape.op == crate::causal_ops::Op::WindowScores {
            (Kind::BlockScores, crate::block_scores::stage(shape, rows, shared))
        } else {
            (Kind::BlockValues, crate::block_values::stage(shape, rows, shared))
        };
        let Some((blocks, threads)) = stage else { return Ok(false); };
        let view = shape.rows as usize;
        if ring_rows == 0 || prefix == 0 || prefix > ring_rows || prefix >= view || start_slot >= ring_rows {
            return Err("ring-delta attention layout is out of bounds".into());
        }
        let delta_rows = view - prefix;
        let row_words = (shape.heads / shape.taps) as usize * (shape.dim as usize / 2);
        let ring_words = ring_rows.checked_mul(row_words).ok_or("ring-delta ring overflow")?;
        let delta_words = delta_rows.checked_mul(row_words).ok_or("ring-delta delta overflow")?;
        self.require_arm_kernels(&[kind.kernel()])?;
        let operand = |v: &DeviceVector<'_>| Operand {
            address: v.device_address(), words: v.len(),
            contiguous: !v.is_segmented(), own_context: self.check_vector(v, v.len()).is_ok(),
        };
        let (mut lengths, weights, units, words) = crate::block_rows::lengths(shape, rows, shared)
            .ok_or("invalid ring-delta attention shape")?;
        let [a, b, c] = x.map(operand);
        let ring_op = operand(ring);
        let delta_op = operand(delta);
        let out = operand(output);
        lengths[1] = ring_words;
        operands(Some((lengths, weights, units, words)), [Some(a), Some(ring_op), Some(b), Some(c)], None, out, self.arm_capture())?;
        lengths[1] = delta_words;
        operands(Some((lengths, weights, units, words)), [Some(a), Some(delta_op), Some(b), Some(c)], None, out, self.arm_capture())?;
        self.skip_fill(output)?;
        self.resident_completion()?;
        let args = crate::causal_ops::Arguments {
            input: [a.address, ring_op.address, b.address, c.address],
            weight: 4, output: out.address, shape,
        };
        let mut command = crate::resident_launch_encode::causal(args);
        command.kind = kind as u32;
        command.put(crate::block_ops::device::ROWS_AT, &(rows as u32).to_ne_bytes());
        command.put(crate::block_ops::device::SHARED_AT, &shared.to_ne_bytes());
        for (i, w) in [delta_op.address, ring_rows as u64, start_slot as u64, prefix as u64].into_iter().enumerate() {
            command.put(RING_DELTA_AT + i * 8, &w.to_ne_bytes());
        }
        let blocks = u32::try_from(blocks).map_err(|_| "ring-delta attention grid overflow")?;
        let ok = self.record_resident_block(kind, command, [blocks, 1, 1, threads as u32, 0])?;
        self.resident_completion()?;
        Ok(ok)
    }
    pub(super) fn inkling_module(&self)->Result<&Module<'a>,String> {
        if let Some(module)=self.inkling.module.get() {return Ok(module);}
        let ptx=crate::inkling_ptx().ok_or("this build has no Inkling image: run `cargo kernels-inkling` before the host build")?;
        let module=Module::load_second(self.context,ptx)?;
        Ok(self.inkling.module.get_or_init(||module))
    }
    /// True when this build carries the Inkling image.
    pub fn inkling_image_present(&self)->bool {
        self.inkling.module.get().is_some() || {
            static PRESENT:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
            *PRESENT.get_or_init(||crate::inkling_ptx().is_some())
        }
    }
    /// Loads the Inkling image now when this build carries it, so the first arm launch after a start does not pay for
    /// its PTX compile. The same module and entries: no word changes.
    pub fn inkling_preload(&self)->Result<(),String> {if self.inkling_image_present() {self.inkling_module()?;} Ok(())}
    /// Marks a graph capture of `capture_with_allowance` (`on`) or its end; gives the mark before, to put back.
    pub(super) fn mark_graph_capture(&self,on:bool)->bool {self.inkling.capturing.replace(on)}
    /// True when any stream capture runs: a block arm is then refused, since its one-command table could not live
    /// until a replay (review REVIMAGE finding 1).
    pub(super) fn arm_capture(&self)->bool {self.inkling.capturing.get() || self.context.capturing.get()}
    /// One block-arm operation of `resident_causal_into` (Ruling R6h rule 1: the whole launch lives in this Inkling
    /// host file). Here, in this one function: every operand is checked (`block_check::operands`: exact counts of the
    /// arm or of its rows, this context, contiguous, aligned, the output disjoint, no capture), the command is built
    /// from the checked addresses, and then it is recorded, or uploaded as a table of one aligned Command and launched
    /// at once. Returns that table (None when recorded): the caller keeps it through the completion it waits for.
    /// The launch rests only on what this function checked (review REVIMAGE-2 finding 1).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn block_arm(&self,shape:crate::causal_ops::Shape,row_launch:Option<(usize,u32)>,input:[Option<&DeviceVector<'_>>;4],weight:Option<&DeviceVector<'_>>,output:&DeviceVector<'_>,units:usize,args:crate::causal_ops::Arguments)->Result<Option<DeviceVector<'a>>,String> {
        use crate::resident_launch::Kind;
        let operand=|v:&DeviceVector<'_>|crate::block_check::Operand {address:v.device_address(),words:v.len(),contiguous:!v.is_segmented(),own_context:self.check_vector(v,v.len()).is_ok()};
        let counts=match row_launch {Some((count,shared))=>crate::block_rows::lengths(shape,count,shared),None=>crate::block_ops::checked_lengths(shape)};
        let kv_plane_slot=matches!(shape.op,crate::causal_ops::Op::WindowScores|crate::causal_ops::Op::WindowValues);
        let mut checked_inputs=input.map(|v|v.map(operand));
        if kv_plane_slot {
            if let (Some(v),Some((lengths,_,_,_)))=(input[1],counts) {
                if self.check_vector_prefix(v,lengths[1]).is_ok() {
                    let p_end=v.device_address().checked_add(4*v.len() as u64).ok_or("block arm input 1 ends past the address space")?;
                    let out_end=output.device_address().checked_add(4*output.len() as u64).ok_or("block arm output ends past the address space")?;
                    if output.device_address()<p_end && v.device_address()<out_end {return Err("block arm output overlaps an operand".into());}
                    checked_inputs[1]=Some(crate::block_check::Operand {address:v.device_address(),words:lengths[1],contiguous:true,own_context:true});
                }
            }
        }
        let (_,_,checked_units,_)=crate::block_check::operands(counts,checked_inputs,weight.map(operand),operand(output),self.arm_capture())?;
        if checked_units!=units {return Err("block arm units differ from the checked counts".into());}
        let addresses=input.map(|v|v.map_or(4,|v|v.device_address()));
        // Every field of the shape in the command is the checked shape (review REVPACK finding 3), the floats by their bits.
        let same_shape=|a:crate::causal_ops::Shape,b:crate::causal_ops::Shape|a.op==b.op && a.heads==b.heads && a.dim==b.dim && a.taps==b.taps && a.rows==b.rows && a.cols==b.cols && a.epsilon.to_bits()==b.epsilon.to_bits() && a.bound.to_bits()==b.bound.to_bits();
        if args.input!=addresses || args.weight!=weight.map_or(4,|v|v.device_address()) || args.output!=output.device_address() || !same_shape(args.shape,shape) {return Err("block arm command differs from its checked operands".into());}
        let mut command=crate::resident_launch_encode::causal(args);
        let (kind,(blocks,threads))=match row_launch {
            Some((count,shared))=>{
                command.put(crate::block_ops::device::ROWS_AT,&(count as u32).to_ne_bytes());command.put(crate::block_ops::device::SHARED_AT,&shared.to_ne_bytes());
                // WindowScores at rows with its key plane shared (the bias shared or each row's own), and SinkSoftmax at
                // rows with its scores per row, and WindowValues at rows with its value plane shared: each its own entry (rows_kind,
                // block_scores::stage, softmax_rows_stage, block_values::stage).
                match (rows_kind(shape,count,shared),crate::block_scores::stage(shape,count,shared),crate::block_attention::softmax_rows_stage(shape,count,shared)) {
                    (Kind::BlockScores,Some((blocks,threads)),_)=>(Kind::BlockScores,(u32::try_from(blocks).map_err(|_|"block scores grid overflow")?,threads as u32)),
                    (Kind::BlockSoftmax,_,Some(blocks))=>(Kind::BlockSoftmax,(u32::try_from(blocks).map_err(|_|"block softmax grid overflow")?,crate::block_ops::SUM_THREADS as u32)),
                    (Kind::BlockValues,_,_)=>{let (blocks,threads)=crate::block_values::stage(shape,count,shared).ok_or("block values stage")?;(Kind::BlockValues,(u32::try_from(blocks).map_err(|_|"block values grid overflow")?,threads as u32))},
                    _ if shape.op==crate::causal_ops::Op::RouterPick && shape.rows<=128 && shared==0 =>
                        (Kind::BlockRows,(u32::try_from(count).map_err(|_|"router rows grid overflow")?,128)),
                    _ if count<=16 && crate::block_ops::dot_stage(shape).is_some() =>
                        (Kind::BlockRows,(u32::try_from(units).map_err(|_|"row dots grid overflow")?,128)),
                    _=>(Kind::BlockRows,(u32::try_from(units.div_ceil(128)).map_err(|_|"block rows grid overflow")?,128)),
                }
            },
            None if shape.op==crate::causal_ops::Op::RelBias && crate::block_attention::rel_part(shape.cols as usize)>1=>{
                // One lane per distance, so a warp reads a contiguous segment of each table row.
                let words=units*crate::block_attention::rel_part(shape.cols as usize);
                (Kind::BlockArms,(words.div_ceil(128) as u32,128))
            }
            None if shape.op==crate::causal_ops::Op::ShortConv && shape.heads==1 && units>=512=>{
                self.require_arm_kernels(&[Kind::FastStreamOps.kernel()])?;
                (Kind::BlockArms,(u32::try_from(units.div_ceil(32)).map_err(|_|"short conv grid overflow")?,32))
            }
            None=>(Kind::BlockArms,crate::block_ops::geometry(shape,units)),
        };
        command.kind=kind as u32;
        if self.record_resident_block(kind,command,[blocks,1,1,threads,0])? {return Ok(None);}
        self.launch_inkling_command(kind,command,blocks,threads,0,0).map(Some)
    }
    /// Upload and launch one checked Inkling command, optionally followed by `pad_commands` zero-padded commands so
    /// kernels that issue `warm_command!` prefetches stay strictly inside the allocation. The caller holds its
    /// operands and the returned table through completion, or leaves their drops to the queue.
    fn launch_inkling_command(&self,kind:crate::resident_launch::Kind,command:crate::resident_launch::Command,blocks:u32,threads:u32,shared:u32,pad_commands:usize)->Result<DeviceVector<'a>,String> {
        let total_words=crate::inkling_launch::COMMAND_WORDS.checked_mul(1+pad_commands).ok_or("command table overflow")?;
        let mut words=Vec::with_capacity(total_words);
        words.extend_from_slice(&[f32::from_bits(command.kind),0.0,0.0,0.0]);
        words.extend(command.payload.as_chunks::<4>().0.iter().map(|b|f32::from_bits(u32::from_ne_bytes(*b))));
        words.resize(total_words,0.0);
        let table=self.upload_vector(&words)?;
        let name=kind.kernel();
        if !in_inkling_image(name) || table.len()!=total_words || table.device_address()%16!=0 {return Err("command table is not an aligned command of an Inkling entry".into());}
        let (mut address,mut index)=(table.device_address(),0u64);
        // SAFETY: The callers check the input and output extents, context, command kind and grid for the chosen entry.
        // The table above is one aligned, initialized command plus `pad_commands` zero-padded trailing commands in this
        // context. The caller keeps the table and operands alive through completion. Inside the resident queue,
        // dropped allocations wait for a real stream drain.
        unsafe {self.launch_flash_shared(name,blocks,threads,shared,&mut [parameter(&mut address),parameter(&mut index)])}?;
        Ok(table)
    }
    /// The module that holds the entry `name`: the tensor-core image, the Inkling image, or the general image.
    pub(super) fn module_for(&self,name:&str)->Result<&Module<'a>,String> {
        if tc_family::in_tc_image(name) {self.tc_module()} else if in_inkling_image(name) {self.inkling_module()} else {Ok(&self.module)}
    }
}
#[cfg(test)] mod tests {
    use crate::{inkling_launch::{BLOCK_ARMS,BLOCK_ROWS,PLAIN_STAGED,PLAIN_STAGED_ROWS,PLAIN_STAGED_FEW,PLAIN_STAGED_ONE,PLAIN_STAGED_NARROW,PLAIN_PACKED,PLAIN_PACKED_GROUP,PLAIN_PACKED_PICKED,BLOCK_SCORES,BLOCK_VALUES,BLOCK_SOFTMAX,PLAIN_TC_ROWS,PLAIN_PACKED_TC_ROWS,PREFILL_PACKED_TC,PREFILL_FLASH_WINDOW,PREFILL_GATE_FUSE,MOE_GROUP_ROUTE,FLASH_WINDOW_ATTN,FLASH_WINDOW_SPLIT16,FLASH_WINDOW_SPLIT32,FLASH_WINDOW_REDUCE,FAST_MIX_BLEND,FAST_ROW_DOTS,FAST_NORM,FAST_CAPPED_GATE,FAST_ROUTER_PICK,FAST_STREAM_OPS,PLAIN_PACKED_ROWS,EMBED_ROWS,DRAFT_LANES,PLAIN_TC_BIG,PLAIN_TC_PANEL64,DRAFT_PANEL,DRAFT_HEAD_FOLD},resident_launch::Kind};
    #[test] fn the_inkling_entries_are_the_kernels_of_its_kinds() {
        for kind in Kind::ALL {
            assert_eq!(super::in_inkling_image(kind.kernel()),[BLOCK_ARMS,BLOCK_ROWS,PLAIN_STAGED,PLAIN_STAGED_ROWS,PLAIN_STAGED_FEW,PLAIN_STAGED_ONE,PLAIN_STAGED_NARROW,PLAIN_PACKED,PLAIN_PACKED_GROUP,PLAIN_PACKED_PICKED,BLOCK_SCORES,BLOCK_VALUES,BLOCK_SOFTMAX,PLAIN_TC_ROWS,PLAIN_PACKED_TC_ROWS,PREFILL_PACKED_TC,PREFILL_FLASH_WINDOW,PREFILL_GATE_FUSE,MOE_GROUP_ROUTE,FLASH_WINDOW_ATTN,FLASH_WINDOW_SPLIT16,FLASH_WINDOW_SPLIT32,FLASH_WINDOW_REDUCE,FAST_MIX_BLEND,FAST_ROW_DOTS,FAST_NORM,FAST_CAPPED_GATE,FAST_ROUTER_PICK,FAST_STREAM_OPS,PLAIN_PACKED_ROWS,EMBED_ROWS,DRAFT_LANES,PLAIN_TC_BIG,PLAIN_TC_PANEL64,DRAFT_PANEL,DRAFT_HEAD_FOLD].contains(&(kind as u32)),"{}",kind.kernel());
        }
        assert!(!super::in_inkling_image("causal_operation") && !super::in_inkling_image("resident_launch_causal"));
    }
    /// The one choice of the rows entry (review REVSOFTMAX, LOW 1): WindowScores with its key plane shared takes
    /// kind 97, SinkSoftmax with its scores per row kind 98, every other rows launch the rows entry.
    #[test] fn a_rows_launch_takes_the_entry_of_its_stage() {
        use crate::causal_ops::{Op,Shape};
        let softmax=Shape {op:Op::SinkSoftmax,heads:24,dim:0,taps:0,rows:8192,cols:0,epsilon:0.0,bound:0.0};
        let scores=Shape {op:Op::WindowScores,heads:24,dim:128,taps:3,rows:1024,cols:1024,epsilon:0.0,bound:0.088_388_35};
        assert_eq!(super::rows_kind(scores,1024,0b0010),Kind::BlockScores);
        assert_eq!(super::rows_kind(scores,1024,0),Kind::BlockRows);
        assert_eq!(super::rows_kind(softmax,1024,0),Kind::BlockSoftmax);
        assert_eq!(super::rows_kind(softmax,1024,0b0110),Kind::BlockSoftmax);
        // Shared scores, a launch above the rows limit, another arm: the rows entry.
        assert_eq!(super::rows_kind(softmax,1024,0b0001),Kind::BlockRows);
        assert_eq!(super::rows_kind(softmax,crate::block_rows::ROWS_MAX+1,0),Kind::BlockRows);
        assert_eq!(super::rows_kind(Shape {op:Op::WindowValues,..scores},1024,0b0010),Kind::BlockRows);
        // WindowValues with its value plane shared takes kind 99 (review REVVALUES, LOW 1); without the shared plane,
        // with 4 taps or with a dim that is not a multiple of 16 it is a rows launch the rows entry takes.
        let values=Shape {op:Op::WindowValues,heads:24,dim:128,taps:3,rows:1024,cols:0,epsilon:0.0,bound:0.0};
        assert_eq!(super::rows_kind(values,1024,0b0010),Kind::BlockValues);
        assert_eq!(super::rows_kind(values,1,0b0010),Kind::BlockValues);
        for (shape,shared) in [(values,0),(Shape {heads:24,taps:4,..values},0b0010),(Shape {dim:24,..values},0b0010)] {
            assert!(crate::block_rows::lengths(shape,1024,shared).is_some());
            assert_eq!(super::rows_kind(shape,1024,shared),Kind::BlockRows);
        }
    }
    /// Review REVIMAGE finding 1: an arm launched inside each graph capture is refused with an error, no launch, and
    /// the same arm launches after the capture.
    #[test]
    #[ignore = "skipped: no GPU; requires CUDA and the Inkling image"]
    fn an_arm_inside_each_graph_capture_is_refused() {
        use crate::causal_ops::{Op,Shape};
        crate::driver::with_backend(0,crate::layer_ptx().unwrap(),usize::MAX,|b| {
            let norm=Shape {op:Op::PlainNorm,heads:1,dim:64,taps:0,rows:0,cols:0,epsilon:1.0e-6,bound:0.0};
            let x=b.upload_vector(&[0.5f32;64])?;let position=b.upload_vector(&[0.0f32;4])?;let y=std::cell::RefCell::new(b.vector(64)?);
            let arm=||b.resident_causal_into(norm,[Some(&x),None,None,None],None,&mut y.borrow_mut());
            let refused=|graph:Result<crate::driver::DecodeGraph<'_>,String>,name:&str|->Result<(),String> {
                match graph {Err(e) if e.contains("stream capture")=>Ok(()),Err(e)=>Err(format!("{name}: other error {e}")),Ok(_)=>Err(format!("{name}: the arm was captured"))}
            };
            // SAFETY: each capture records nothing (the arm is refused before any launch, so the capture fails or is
            // empty) and every buffer the closure names (x, y, position) lives to the end of this test.
            let graphs=unsafe {[b.capture_batch(1,1,arm),b.capture_batch_bounded(64,arm),b.capture_decode(&position,0,1,1,arm)]};
            for (graph,name) in graphs.into_iter().zip(["capture_batch","capture_batch_bounded","capture_decode"]) {refused(graph,name)?;}
            if b.arm_capture() {return Err("the capture mark stayed set".into());}
            arm()
        }).unwrap();
    }
}
