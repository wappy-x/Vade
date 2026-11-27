//! The small-batch BF16 product (`few_product`, kind 101) on one CUDA device, small shapes. No model file.
//! FEW: the product of a made-up BF16 matrix with 2 to 16 input rows, into an output with guard words on both sides,
//! against the host ascending-K BF16 MMA chain and against one launch of
//! the row product (`project_staged_into`, kind 90) for each input row: every word must have the bits of both. The
//! shapes walk the edges of the launch geometry: 1 row, a warp less one, a whole warp, a warp and one, many warps;
//! 1, 2, 9 and 18 stages of columns; 2, 3, 7, 8, 9, 15 and 16 input rows; weights and input at 16-byte steps.
//! PARTS: one matrix as two parts that write their columns of one output in place (the output stride).
//! REFUSALS: a launch that the host check refuses records nothing and writes no word.
//! Small on purpose: it runs under the memory checker of the device tools
//! (`compute-sanitizer --tool memcheck few-check`).
#[cfg(target_os = "linux")]
mod extra;
#[cfg(target_os = "linux")]
mod check {
    use spark_gpu_probe::{driver::{Backend, DeviceVector}, resident_launch::Kind};

    const GUARD: usize = 64;
    const POISON: u32 = 0x7fc0_dead;
    /// (matrix rows, columns).
    const SHAPES: [(usize, usize); 13] = [(1, 64), (31, 64), (32, 128), (33, 576), (96, 1152), (130, 64), (257, 192), (64, 128), (31, 256), (1024, 2304), (3072, 2304), (40, 9216), (2304, 9216)];
    const INPUTS: [usize; 8] = [1, 2, 3, 7, 8, 9, 15, 16];

    /// BF16 patterns, finite: both signs, both zeros, values below the normal range, values near 2^31, and ordinary
    /// values. No value is large enough for a sum to leave the finite range.
    fn weight(i: usize) -> u16 {
        let n = (i as u32).wrapping_mul(2_654_435_761) >> 16;
        match i % 11 { 0 => 0x0000, 1 => 0x8000, 2 => 0x0001 | (n as u16 & 0x007f), 3 => 0x4f00 | (n as u16 & 0x007f), _ => (0x3800 + (n % 0x1000)) as u16 | ((n as u16 & 1) << 15) }
    }
    /// Input values with 24 bits in each value and both signs.
    fn input(k: usize) -> f32 { ((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0 }
    /// The matrix on the device: two BF16 values a word, the even column in the low half. `lead` words come first.
    fn matrix<'d>(b: &Backend<'d>, values: &[u16], lead: usize) -> Result<DeviceVector<'d>, String> {
        let mut vector = b.vector(lead + values.len() / 2)?;
        let bytes = vec![0u8; lead * 4].into_iter().chain(values.iter().flat_map(|v| v.to_le_bytes())).collect::<Vec<_>>();
        b.write_vector_bytes(&mut vector, &bytes)?;
        vector.into_resident_shared()
    }
    struct Count { cases: usize, words: usize, different: usize, row_different: usize, guards: usize, changed: usize }
    /// Compare the output between its guards with the host words, and the guards with the poison.
    fn compare(b: &Backend<'_>, whole: &DeviceVector<'_>, want: &[f32], count: &mut Count) -> Result<Vec<f32>, String> {
        let back = b.read_vector_values(whole)?;
        if back.len() != want.len() + 2 * GUARD { return Err(format!("{} words read back for {} words and the guards", back.len(), want.len())); }
        count.cases += 1; count.words += want.len(); count.guards += 2 * GUARD;
        count.different += back[GUARD..GUARD + want.len()].iter().zip(want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        count.changed += back[..GUARD].iter().chain(&back[GUARD + want.len()..]).filter(|word| word.to_bits() != POISON).count();
        Ok(back[GUARD..GUARD + want.len()].to_vec())
    }
    fn poisoned<'d>(b: &Backend<'d>, words: usize) -> Result<DeviceVector<'d>, String> { b.upload_vector(&vec![f32::from_bits(POISON); words + 2 * GUARD])?.into_resident_shared() }

    pub fn run() -> Result<(), String> {
        spark_gpu_probe::driver::with_backend(0, spark_gpu_probe::layer_ptx()?, usize::MAX, |b| {
            // Odd rank, a partial final block, and a second chunk of vocabulary rows.
            let (rows,hidden,latent_width)=(2051,18,5);
            let left=(0..rows*hidden).map(weight).collect::<Vec<_>>();
            let value=|i|f32::from_bits(u32::from(weight(i))<<16);
            let right=(0..hidden*latent_width).map(value).collect::<Vec<_>>();
            let bias=(0..hidden).map(|i|value(i+3)).collect::<Vec<_>>();
            let (fold,offset)=b.fold_markov(&left,&right,&bias,rows,hidden,latent_width)?;
            for v in 0..rows {for j in 0..=latent_width {
                let mut sum=0.0f64;
                for k in 0..hidden {sum+=f64::from(value(v*hidden+k))*f64::from(if j==latent_width {bias[k]} else {right[k*latent_width+j]});}
                let bits=spark_gpu_probe::bf16::rounded(sum as f32).to_bits();
                if bits!=if j==latent_width {offset[v].to_bits()} else {u32::from(fold[v*latent_width+j])<<16} {return Err(format!("fold mismatch at {v},{j}"));}
            }}
            println!("MARKOV FOLD PASS rows={rows} hidden={hidden} latent_width={latent_width}");
            use spark_gpu_probe::draft_ops::DraftOp::*;
            let ones=b.upload_vector(&[1.0;80])?;
            let twos=b.upload_vector(&[2.0;80])?;
            let shared=ones.into_resident_shared()?;let two=twos.into_resident_shared()?;
            let check=|op:spark_gpu_probe::draft_ops::DraftOp,want:&[f32]|->Result<(),String> {
                let (lens,_,_,_)=op.layout().ok_or("invalid test shape")?;
                let x=shared.resident_range(0..lens[0])?;
                let y=(lens[1]>0).then(||two.resident_range(0..lens[1])).transpose()?;
                let out=b.draft_lanes(op,[Some(&x),y.as_ref(),None])?;
                if b.read_vector_values(&out)?!=want {return Err("draft lane result differs".into());}Ok(())
            };
            check(Rope{rows:1,heads:2,width:2},&[0.0,4.0,0.0,4.0])?;
            check(Scores{rows:2,heads:2,kv:1,width:1,keys:3},&[2.0;12])?;
            check(Normalizer{queries:2,keys:3},&[1.0/3.0;6])?;
            // The 16-byte path of the scores (width a multiple of 4), and rows of more keys than a warp has lanes.
            check(Scores{rows:2,heads:2,kv:1,width:4,keys:3},&[4.0;12])?;
            check(Normalizer{queries:2,keys:40},&[1.0/40.0;80])?;
            check(Values{rows:2,heads:2,kv:1,width:1,keys:3},&[6.0;4])?;
            check(Mix{words:4,taps:3},&[6.0;4])?;
            check(Unpack(4),&[0.0,1.0,0.0,1.0])?;
            // UnpackAt: the row named by a device pick [best, id, flag] of a packed table of 3 rows of width 4; a
            // pick whose flag is not 1 gives zeros.
            for (pick,want) in [(&shared,[0.0,1.0,0.0,1.0]),(&two,[0.0;4])] {
                let (x,y)=(shared.resident_range(0..6)?,pick.resident_range(0..3)?);
                let out=b.draft_lanes(UnpackAt{width:4,rows:3},[Some(&x),Some(&y),None])?;
                if b.read_vector_values(&out)?!=want {return Err("draft lane UnpackAt result differs".into());}
            }
            let rn=spark_gpu_probe::bf16::rounded;
            let gate=rn(spark_gpu_probe::fixed_math::sigmoid(1.0));
            let cand=rn(spark_gpu_probe::fixed_math::tanh(1.0));
            check(Gate(3),&[rn(gate*2.0);3])?;
            // MarkovLogits: dot 1, bias 2, base 1: r(1 + r(1 + 2)) = 4, capped at 32 (epsilon 1/32).
            {let (x,y,z)=(shared.resident_range(0..5)?,two.resident_range(0..5)?,shared.resident_range(5..10)?);
             let out=b.draft_lanes(MarkovLogits{n:5,epsilon:0.03125,bound:32.0},[Some(&x),Some(&y),Some(&z)])?;
             let want=spark_gpu_probe::fixed_math::tanh(rn(4.0)*0.03125)*32.0;
             if b.read_vector_values(&out)?!=[want;5] {return Err("draft lane MarkovLogits result differs".into());}}
            check(Recurrent(2),&[rn(rn(gate*2.0)+rn(rn(1.0-gate)*cand)),rn(rn(gate*2.0)+rn(rn(1.0-gate)*cand)),cand,cand])?;
            // Fused register-only DraftOp::FlashAttn (kind 20) + Split-K FlashAttnMerge (kind 21) vs Scores -> Normalizer -> Values.
            for (rows,heads,kv,keys,splits) in [(3usize,24usize,8usize,137usize,1usize),(3,24,8,137,8),(3,24,8,1031,16),(8,24,8,65,8),(8,24,8,129,16)] {
                let width=128usize;
                let q_vals=(0..rows*heads*width).map(|i|input(i+17)*0.5).collect::<Vec<_>>();
                let k_vals=(0..keys*kv*width).map(|i|input(i+29)*0.5).collect::<Vec<_>>();
                let v_vals=(0..keys*kv*width).map(|i|input(i+43)*0.5).collect::<Vec<_>>();
                let (qd,kd,vd)=(b.upload_vector(&q_vals)?,b.upload_vector(&k_vals)?,b.upload_vector(&v_vals)?);
                let scores=b.draft_lanes(Scores{rows,heads,kv,width,keys},[Some(&qd),Some(&kd),None])?;
                let probs=b.draft_lanes(Normalizer{queries:rows*heads,keys},[Some(&scores),None,None])?;
                let ref_out=b.draft_lanes(Values{rows,heads,kv,width,keys},[Some(&probs),Some(&vd),None])?;
                let part=b.draft_lanes(FlashAttn{rows,heads,kv,width,keys,splits,causal_base:0,row_off:0},[Some(&qd),Some(&kd),Some(&vd)])?;
                let fused_out=if splits>1 {b.draft_lanes(FlashAttnMerge{rows,heads,width,splits},[Some(&part),None,None])?} else {part};
                let (want,got)=(b.read_vector_values(&ref_out)?,b.read_vector_values(&fused_out)?);
                let max_err=want.iter().zip(&got).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
                if !(max_err<=1e-3 && got.iter().all(|v|v.is_finite())) {
                    return Err(format!("draft FlashAttn rows={rows} keys={keys} splits={splits} max_err={max_err}"));
                }
            }
            // 3-tier block-causal FlashAttn (`causal_base > 0`): rows 0..8 attend to 0..causal_base,
            // rows 8..12 to 0..min(causal_base+4, keys), rows 12..16 to 0..keys (with row_off=0 and row_off=1).
            for (rows,row_off,keys,causal_base,splits) in [(16usize,0usize,144usize,136usize,8usize),(15,1,144,136,8),(14,0,62,56,1)] {
                let (heads,kv,width)=(24usize,8usize,128usize);
                let q_vals=(0..rows*heads*width).map(|i|input(i+71)*0.5).collect::<Vec<_>>();
                let k_vals=(0..keys*kv*width).map(|i|input(i+83)*0.5).collect::<Vec<_>>();
                let v_vals=(0..keys*kv*width).map(|i|input(i+97)*0.5).collect::<Vec<_>>();
                let (qd,kd,vd)=(b.upload_vector(&q_vals)?,b.upload_vector(&k_vals)?,b.upload_vector(&v_vals)?);
                let part=b.draft_lanes(FlashAttn{rows,heads,kv,width,keys,splits,causal_base,row_off},[Some(&qd),Some(&kd),Some(&vd)])?;
                let fused_out=if splits>1 {b.draft_lanes(FlashAttnMerge{rows,heads,width,splits},[Some(&part),None,None])?} else {part};
                let got=b.read_vector_values(&fused_out)?;
                for (tier_k,row_start,row_end) in [
                    (causal_base,0usize,8usize.saturating_sub(row_off).min(rows)),
                    ((causal_base+4).min(keys),8usize.saturating_sub(row_off).min(rows),12usize.saturating_sub(row_off).min(rows)),
                    (keys,12usize.saturating_sub(row_off).min(rows),rows),
                ] {
                    if row_start>=row_end {continue;}
                    let n=row_end-row_start;
                    let q_sub=b.upload_vector(&q_vals[row_start*heads*width..row_end*heads*width])?;
                    let k_sub=b.upload_vector(&k_vals[..tier_k*kv*width])?;
                    let v_sub=b.upload_vector(&v_vals[..tier_k*kv*width])?;
                    let scores=b.draft_lanes(Scores{rows:n,heads,kv,width,keys:tier_k},[Some(&q_sub),Some(&k_sub),None])?;
                    let probs=b.draft_lanes(Normalizer{queries:n*heads,keys:tier_k},[Some(&scores),None,None])?;
                    let ref_out=b.read_vector_values(&b.draft_lanes(Values{rows:n,heads,kv,width,keys:tier_k},[Some(&probs),Some(&v_sub),None])?)?;
                    let got_sub=&got[row_start*heads*width..row_end*heads*width];
                    let max_err=ref_out.iter().zip(got_sub).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
                    if !(max_err<=1e-3 && got_sub.iter().all(|v|v.is_finite())) {
                        return Err(format!("causal FlashAttn rows={rows} row_off={row_off} tier_k={tier_k} splits={splits} max_err={max_err}"));
                    }
                }
            }
            // Distinct code bytes, input columns and token rows catch lane permutations.
            {
                use spark_gpu_probe::{decode_panels,packed_product};
                let (rows,cols)=(79usize,128usize);
                let physical_rows=rows.next_multiple_of(64);
                let layout=packed_product::layout(physical_rows,cols,false);
                let mut bytes=vec![0u8;layout.bytes];
                let coeff=|row:usize,col:usize| (row*3+col/3+col*row)%8;
                for row in 0..rows {
                    for col in (0..cols).step_by(2) {
                        let at=packed_product::code_copy(row,col/64,0,cols,false)+(col%64)/2;
                        bytes[at]=(coeff(row,col)|(coeff(row,col+1)<<4)) as u8;
                    }
                }
                for v in 0..8 {
                    bytes[layout.table+v*2..layout.table+v*2+2].copy_from_slice(&(((v as f32).to_bits()>>16) as u16).to_le_bytes());
                }
                bytes[layout.scales..layout.table].fill(56);
                let panel=decode_panels::pack_draft(&bytes,physical_rows,cols,1.0);
                let mut weight=b.vector(panel.len()/4)?;
                b.write_vector_bytes(&mut weight,&panel)?;
                for inputs in [1,3,8,9,12,15,16,24,32,48,64,96] {
                    let x=(0..inputs*cols).map(|at| [0.0,1.5,3.0,4.0,6.0][(at/cols+3*at%cols)%5]).collect::<Vec<_>>();
                    let input=b.upload_vector(&x)?;
                    let input=b.draft_lanes(PackActivations(x.len()),[Some(&input),None,None])?;
                    let out=b.draft_lanes(Panel{rows,columns:cols,inputs},[Some(&weight),Some(&input),None])?;
                    let got=b.read_vector_values(&out)?;
                    for t in 0..inputs {for row in 0..rows {
                        let want=(0..cols).map(|col|[0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0][coeff(row,col)]*x[t*cols+col]).sum::<f32>();
                        if got[t*rows+row]!=rn(want) {return Err(format!("draft panel product differs at inputs={inputs} t={t} row={row}"));}
                    }}
                }
                let mut packed_bias=panel;
                packed_bias.extend_from_slice(&vec![0;4*rows]);
                let mut fold=b.vector(packed_bias.len()/4)?;
                b.write_vector_bytes(&mut fold,&packed_bias)?;
                // Batched FoldPickBatch + FoldReduceBatch + FoldFinishBatch vs single-chat FoldPick + FoldReduce + FoldFinish
                // for both cols=128 (general multi-slab) and cols=256 (chunks==4 register-resident fast path) up to batch=32.
                for batch in [2usize, 4, 8, 16, 32] {
                    let lat_vals=(0..batch*cols).map(|i|((i*17+i/cols*5)%37) as f32/64.0-0.25).collect::<Vec<_>>();
                    let base_vals=(0..batch*rows).map(|i|((i*11+i/rows*7)%29) as f32/16.0-0.5).collect::<Vec<_>>();
                    let lat_raw=b.upload_vector(&lat_vals)?;
                    let lat_packed=b.draft_lanes(PackActivations(lat_vals.len()),[Some(&lat_raw),None,None])?.into_resident_shared()?;
                    let base_dev=b.upload_vector(&base_vals)?.into_resident_shared()?;
                    let tile_count=rows.div_ceil(16);
                    let blocks=2usize.min(tile_count);
                    let b_tiles=b.draft_lanes(FoldPickBatch{rows,columns:cols,batch,epsilon:0.03125,bound:32.0,confidence:true,bf16_base:false},[Some(&fold),Some(&lat_packed),Some(&base_dev)])?;
                    let b_red=b.draft_lanes(FoldReduceBatch{batch,tiles:tile_count,blocks,confidence:true},[Some(&b_tiles),None,None])?;
                    let b_fin=b.read_vector_values(&b.draft_lanes(FoldFinishBatch{batch,tiles:blocks,confidence:true},[Some(&b_red),None,None])?)?;
                    let stride=20+blocks;
                    let col_pack=(cols/64)*18;
                    for bi in 0..batch {
                        let lat_b=lat_packed.resident_range(bi*col_pack..(bi+1)*col_pack)?;
                        let base_b=base_dev.resident_range(bi*rows..(bi+1)*rows)?;
                        let s_tiles=b.draft_lanes(FoldPick{rows,columns:cols,epsilon:0.03125,bound:32.0,confidence:true,gated:false,bf16_base:false},[Some(&fold),Some(&lat_b),Some(&base_b)])?;
                        let s_red=b.draft_lanes(FoldReduce{tiles:tile_count,blocks,confidence:true,gated:false},[Some(&s_tiles),None,None])?;
                        let s_fin=b.read_vector_values(&b.draft_lanes(FoldFinish{tiles:blocks,confidence:true,stop_above_sum:0.0,gated:false},[Some(&s_red),None,None])?)?;
                        let got=&b_fin[bi*stride..bi*stride+4];
                        if got.iter().map(|v|v.to_bits()).ne(s_fin[..4].iter().map(|v|v.to_bits())) {
                            return Err(format!("FoldPickBatch/Reduce/Finish cols={cols} batch={batch} chat={bi} differs: {got:?} vs {:?}",&s_fin[..4]));
                        }
                    }
                }
                // Also verify cols=256 (`chunks == 4` register-resident FoldPickBatch) at batch=16 and 32 with non-zero bias.
                {
                    let cols256=256usize;
                    let layout256=packed_product::layout(physical_rows,cols256,false);
                    let mut bytes256=vec![0u8;layout256.bytes];
                    for row in 0..rows {
                        for col in (0..cols256).step_by(2) {
                            let at=packed_product::code_copy(row,col/64,0,cols256,false)+(col%64)/2;
                            bytes256[at]=(coeff(row,col)|(coeff(row,col+1)<<4)) as u8;
                        }
                    }
                    for v in 0..8 {
                        bytes256[layout256.table+v*2..layout256.table+v*2+2].copy_from_slice(&(((v as f32).to_bits()>>16) as u16).to_le_bytes());
                    }
                    bytes256[layout256.scales..layout256.table].fill(56);
                    let mut panel256=decode_panels::pack_draft(&bytes256,physical_rows,cols256,1.0);
                    for r in 0..rows {
                        panel256.extend_from_slice(&((((r*7)%19) as f32/16.0-0.5).to_le_bytes()));
                    }
                    let mut fold256=b.vector(panel256.len()/4)?;
                    b.write_vector_bytes(&mut fold256,&panel256)?;
                    for batch in [16usize, 32] {
                        let lat_vals=(0..batch*cols256).map(|i|((i*19+i/cols256*7)%41) as f32/64.0-0.3).collect::<Vec<_>>();
                        let base_vals=(0..batch*rows).map(|i|((i*13+i/rows*5)%31) as f32/16.0-0.5).collect::<Vec<_>>();
                        let lat_raw=b.upload_vector(&lat_vals)?;
                        let lat_packed=b.draft_lanes(PackActivations(lat_vals.len()),[Some(&lat_raw),None,None])?.into_resident_shared()?;
                        let base_dev=b.upload_vector(&base_vals)?.into_resident_shared()?;
                        let tile_count=rows.div_ceil(16);
                        let blocks=2usize.min(tile_count);
                        let b_tiles=b.draft_lanes(FoldPickBatch{rows,columns:cols256,batch,epsilon:0.03125,bound:32.0,confidence:true,bf16_base:false},[Some(&fold256),Some(&lat_packed),Some(&base_dev)])?;
                        let b_red=b.draft_lanes(FoldReduceBatch{batch,tiles:tile_count,blocks,confidence:true},[Some(&b_tiles),None,None])?;
                        let b_fin=b.read_vector_values(&b.draft_lanes(FoldFinishBatch{batch,tiles:blocks,confidence:true},[Some(&b_red),None,None])?)?;
                        let stride=20+blocks;
                        let col_pack=(cols256/64)*18;
                        for bi in 0..batch {
                            let lat_b=lat_packed.resident_range(bi*col_pack..(bi+1)*col_pack)?;
                            let base_b=base_dev.resident_range(bi*rows..(bi+1)*rows)?;
                            let s_tiles=b.draft_lanes(FoldPick{rows,columns:cols256,epsilon:0.03125,bound:32.0,confidence:true,gated:false,bf16_base:false},[Some(&fold256),Some(&lat_b),Some(&base_b)])?;
                            let s_red=b.draft_lanes(FoldReduce{tiles:tile_count,blocks,confidence:true,gated:false},[Some(&s_tiles),None,None])?;
                            let s_fin=b.read_vector_values(&b.draft_lanes(FoldFinish{tiles:blocks,confidence:true,stop_above_sum:0.0,gated:false},[Some(&s_red),None,None])?)?;
                            let got=&b_fin[bi*stride..bi*stride+4];
                            if got.iter().map(|v|v.to_bits()).ne(s_fin[..4].iter().map(|v|v.to_bits())) {
                                return Err(format!("FoldPickBatch/Reduce/Finish cols=256 batch={batch} chat={bi} differs: {got:?} vs {:?}",&s_fin[..4]));
                            }
                        }
                    }
                }
                let zero=b.upload_vector(&vec![0.0;cols])?;
                let zero=b.draft_lanes(PackActivations(cols),[Some(&zero),None,None])?;
                let mut logits=vec![0.0;rows];
                logits[rows-2]=1.0;logits[rows-1]=1.0;
                let base=b.upload_vector(&logits)?;
                let tiles=b.draft_lanes(FoldPick{rows,columns:cols,epsilon:0.03125,bound:32.0,confidence:true,gated:false,bf16_base:false},[Some(&fold),Some(&zero),Some(&base)])?;
                let result=b.draft_lanes(FoldFinish{tiles:rows.div_ceil(16),confidence:true,stop_above_sum:0.0,gated:false},[Some(&tiles),None,None])?;
                let got=b.read_vector_values(&result)?;
                let best=spark_gpu_probe::fixed_math::tanh(0.03125)*32.0;
                let sum=2.0+(rows-2) as f32*spark_gpu_probe::fixed_math::exp(-best);
                if got[..3]!=[best,(rows-2) as f32,1.0] || (got[3]-sum).abs()>1e-4 || got[4]!=0.0 {return Err("draft fold pick differs".into());}
                // Device-side early stop check: threshold below sum sets stop=1.0; gated FoldPick/FoldFinish skip cleanly.
                let stopped=b.draft_lanes(FoldFinish{tiles:rows.div_ceil(16),confidence:true,stop_above_sum:sum*0.5,gated:false},[Some(&tiles),None,None])?.into_resident_shared()?;
                let stop_vals=b.read_vector_values(&stopped)?;
                if stop_vals[4]!=1.0 {return Err("FoldFinish stop_above_sum did not trigger".into());}
                let stop_flag=stopped.resident_range(4..5)?;
                let pg_skip=b.read_vector_values(&b.draft_lanes(PanelGated{rows,columns:cols,inputs:1},[Some(&weight),Some(&zero),Some(&stop_flag)])?)?;
                if pg_skip[rows]!=1.0 {return Err("PanelGated did not propagate stop=1.0".into());}
                let zero_flag=b.upload_vector(&[0.0f32])?;
                let pg_run=b.read_vector_values(&b.draft_lanes(PanelGated{rows,columns:cols,inputs:1},[Some(&weight),Some(&zero),Some(&zero_flag)])?)?;
                if pg_run[rows]!=0.0 {return Err("PanelGated did not propagate stop=0.0".into());}
                let zero_plus_stop=b.upload_vector(&vec![0.0f32;cols/64*18].into_iter().chain([1.0f32]).collect::<Vec<_>>())?;
                let gated_tiles=b.draft_lanes(FoldPick{rows,columns:cols,epsilon:0.03125,bound:32.0,confidence:true,gated:true,bf16_base:false},[Some(&fold),Some(&zero_plus_stop),Some(&base)])?;
                let gated_fin=b.read_vector_values(&b.draft_lanes(FoldFinish{tiles:rows.div_ceil(16),confidence:true,stop_above_sum:0.0,gated:true},[Some(&gated_tiles),Some(&stop_flag),None])?)?;
                if gated_fin[..5]!=[0.0,0.0,0.0,0.0,1.0] {return Err("gated FoldFinish did not emit stopped record".into());}
                for inputs in [1,3,8,9,12,15,16] {
                    let values=(0..cols*(inputs+1)).map(|i|((i*13+i/cols)%41) as f32/128.0).collect::<Vec<_>>();
                    let upload=|slice:&[f32]|->Result<_,String> {
                        let raw=b.upload_vector(slice)?;
                        b.draft_lanes(PackActivations(slice.len()),[Some(&raw),None,None])
                    };
                    let head_input=upload(&values[..inputs*cols])?;
                    let base=b.draft_lanes(Panel{rows,columns:cols,inputs},[Some(&weight),Some(&head_input),None])?;
                    let base=base.into_resident_shared()?;
                    let first=base.resident_range(0..rows)?;
                    let latent=upload(&values[inputs*cols..])?;
                    let all=upload(&values)?;
                    for confidence in [false,true] {
                        let tiles=b.draft_lanes(FoldPick{rows,columns:cols,epsilon:0.03125,bound:32.0,confidence,gated:false,bf16_base:false},[Some(&fold),Some(&latent),Some(&first)])?;
                        let got=b.draft_lanes(HeadFold{rows,hidden:cols,latent:cols,inputs,epsilon:0.03125,bound:32.0,confidence,bf16_base:false},[Some(&weight),Some(&fold),Some(&all)])?;
                        let mut expected=b.read_vector_values(&base)?[rows..].to_vec();
                        expected.extend(b.read_vector_values(&tiles)?);
                        if b.read_vector_values(&got)?!=expected {return Err("combined head/fold differs from separate projections".into());}
                        // Bit-exact BF16 base path: HeadFold(bf16_base=true) + FoldPick(bf16_base=true) matches FP32 bit for bit.
                        let got_bf16=b.draft_lanes(HeadFold{rows,hidden:cols,latent:cols,inputs,epsilon:0.03125,bound:32.0,confidence,bf16_base:true},[Some(&weight),Some(&fold),Some(&all)])?.into_resident_shared()?;
                        let bw=(inputs-1)*rows.div_ceil(2);
                        let got_t0=b.read_vector_values(&got_bf16.resident_range(bw..bw+rows.div_ceil(16)*4)?)?;
                        let want_t0=b.read_vector_values(&tiles)?;
                        if got_t0.iter().map(|v|v.to_bits()).ne(want_t0.iter().map(|v|v.to_bits())) {
                            return Err(format!("HeadFold bf16_base step0 differs at inputs={inputs} conf={confidence}"));
                        }
                        if inputs>1 {
                            let s1_bf16=got_bf16.resident_range(0..rows.div_ceil(2))?;
                            let s1_fp32=base.resident_range(rows..2*rows)?;
                            let t1_bf16=b.read_vector_values(&b.draft_lanes(FoldPick{rows,columns:cols,epsilon:0.03125,bound:32.0,confidence,gated:false,bf16_base:true},[Some(&fold),Some(&latent),Some(&s1_bf16)])?)?;
                            let t1_fp32=b.read_vector_values(&b.draft_lanes(FoldPick{rows,columns:cols,epsilon:0.03125,bound:32.0,confidence,gated:false,bf16_base:false},[Some(&fold),Some(&latent),Some(&s1_fp32)])?)?;
                            if t1_bf16.iter().map(|v|v.to_bits()).ne(t1_fp32.iter().map(|v|v.to_bits())) {
                                return Err(format!("FoldPick bf16_base step1 differs at inputs={inputs} conf={confidence}"));
                            }
                        }
                    }
                }
                // PanelBf16 + BF16 FoldPickBatch vs Panel + FP32 FoldPickBatch (odd rows=79, narrow & wide warps, batches 2..=32).
                for (batch,depth) in [(2usize,3usize),(4,7),(8,7),(16,3),(32,7)] {
                    let total=batch*depth;
                    let h_vals=(0..total*cols).map(|i|((i*17+i/cols*5)%37) as f32/64.0-0.25).collect::<Vec<_>>();
                    let l_vals=(0..batch*cols).map(|i|((i*13+i/cols*7)%41) as f32/64.0-0.2).collect::<Vec<_>>();
                    let raw_h=b.upload_vector(&h_vals)?;
                    let raw_l=b.upload_vector(&l_vals)?;
                    let packed_h=b.draft_lanes(PackActivations(h_vals.len()),[Some(&raw_h),None,None])?;
                    let packed_l=b.draft_lanes(PackActivations(l_vals.len()),[Some(&raw_l),None,None])?;
                    let ref_base=b.draft_lanes(Panel{rows,columns:cols,inputs:total},[Some(&weight),Some(&packed_h),None])?.into_resident_shared()?;
                    let bf_base=b.draft_lanes(PanelBf16{rows,columns:cols,inputs:total},[Some(&weight),Some(&packed_h),None])?.into_resident_shared()?;
                    for s in [0usize, depth - 1] {
                        let ref_bs=ref_base.resident_range(s*batch*rows..(s+1)*batch*rows)?;
                        let bf_bs=bf_base.resident_range(s*batch*rows.div_ceil(2)..(s+1)*batch*rows.div_ceil(2))?;
                        let ref_ts=b.read_vector_values(&b.draft_lanes(FoldPickBatch{rows,columns:cols,batch,epsilon:0.03125,bound:32.0,confidence:true,bf16_base:false},[Some(&fold),Some(&packed_l),Some(&ref_bs)])?)?;
                        let got_ts=b.read_vector_values(&b.draft_lanes(FoldPickBatch{rows,columns:cols,batch,epsilon:0.03125,bound:32.0,confidence:true,bf16_base:true},[Some(&fold),Some(&packed_l),Some(&bf_bs)])?)?;
                        if got_ts.iter().map(|v|v.to_bits()).ne(ref_ts.iter().map(|v|v.to_bits())) {
                            return Err(format!("PanelBf16+FoldPickBatch step {s} differs at batch={batch} depth={depth}"));
                        }
                    }
                }
            }
            for value in [1.125,f32::INFINITY,f32::NAN,f32::MAX] {
                let mut values=[0.0;35];values[34]=value;
                let x=b.upload_vector(&values)?;
                let sum=b.draft_lanes(CheckedSum(35),[Some(&x),Some(&x),None])?.into_resident_shared()?;
                let partials=sum.resident_range(35..41)?;
                let flag=b.draft_lanes(PickFinish(2),[Some(&partials),None,None])?;
                let want=rn(value+value);
                let got=b.read_vector_values(&sum)?[34];
                if (got!=want && !(got.is_nan() && want.is_nan())) || b.read_vector_values(&flag)?[2]!=u32::from(want.is_finite()) as f32 {return Err("draft residual or flag differs".into());}
            }
            // Sibling panels: one Panels launch against one Panel launch for each, every output word.
            {
                use spark_gpu_probe::{decode_panels,packed_product};
                let cols=128usize;
                let make=|rows:usize,seed:usize|->Vec<u8> {
                    let layout=packed_product::layout(rows,cols,false);
                    let mut bytes=vec![0u8;layout.bytes];
                    for row in 0..rows {for col in (0..cols).step_by(2) {
                        let at=packed_product::code_copy(row,col/64,0,cols,false)+(col%64)/2;
                        bytes[at]=(((row*7+col*seed+seed)%16)|(((row+col*3+seed*5)%16)<<4)) as u8;
                    }}
                    for (j,s) in bytes[layout.scales..layout.table].iter_mut().enumerate() {*s=(48+(j*seed)%16) as u8;}
                    decode_panels::pack_draft(&bytes,rows,cols,0.5+seed as f32/8.0)
                };
                for rows in [[64usize,128,64],[192,64,0],[128,128,0]] {
                    let parts=rows.iter().enumerate().filter(|(_,r)|**r>0).map(|(k,&r)|(r,make(r,k+1))).collect::<Vec<_>>();
                    let all=parts.iter().flat_map(|(_,p)|p.iter().copied()).collect::<Vec<_>>();
                    let mut joined=b.vector(all.len()/4)?;b.write_vector_bytes(&mut joined,&all)?;
                    for inputs in [1,5,8,9,12,13,16,24,32,64] {
                        let x=(0..inputs*cols).map(|at|[0.0,1.5,-3.0,4.0,6.0,-0.5][(at*7+at/cols)%6]).collect::<Vec<_>>();
                        let input=b.upload_vector(&x)?;
                        let input=b.draft_lanes(PackActivations(x.len()),[Some(&input),None,None])?;
                        let got=b.read_vector_values(&b.draft_lanes(Panels{rows,columns:cols,inputs},[Some(&joined),Some(&input),None])?)?;
                        let mut want=Vec::new();
                        for (r,panel) in &parts {
                            let mut weight=b.vector(panel.len()/4)?;b.write_vector_bytes(&mut weight,panel)?;
                            want.extend(b.read_vector_values(&b.draft_lanes(Panel{rows:*r,columns:cols,inputs},[Some(&weight),Some(&input),None])?)?);
                        }
                        if got.iter().map(|v|v.to_bits()).ne(want.iter().map(|v|v.to_bits())) {return Err(format!("sibling panels differ from single panels at {rows:?} inputs {inputs}"));}
                    }
                }
            }
            // The block finish against the one-warp order: lane L adds tiles L, L+32, ... then the butterfly.
            // Repeated values (ties broken by the lower id), both zeros, an odd tile count and a partial last pass.
            for (tiles,confidence) in [(12504usize,true),(12504,false),(1,true),(33,true),(4097,true)] {
                let mut table=vec![0.0f32;tiles*4];
                for t in 0..tiles {
                    let n=(t as u32).wrapping_mul(2_654_435_761)>>8;
                    let value=match t%7 {0=>0.0,1=>-0.0,_=>((n%4096) as f32/128.0)-16.0};
                    table[t*4..t*4+4].copy_from_slice(&[value,(tiles-1-t) as f32,1.0,1.0+(n%977) as f32/64.0]);
                }
                if tiles>2 {table[(tiles/2)*4]=16.0;table[(tiles/3)*4]=16.0;}
                let input=b.upload_vector(&table)?;
                let got=b.read_vector_values(&b.draft_lanes(FoldFinish{tiles,confidence,stop_above_sum:0.0,gated:false},[Some(&input),None,None])?)?;
                let (mut best,mut id)=(f32::NEG_INFINITY,u32::MAX);
                for t in 0..tiles {let (v,other)=(table[t*4],table[t*4+1] as u32);if v>best || (v==best && other<id) {best=v;id=other;}}
                let mut lanes=[0.0f32;32];
                if confidence {
                    for (lane,sum) in lanes.iter_mut().enumerate() {
                        for t in (lane..tiles).step_by(32) {*sum+=table[t*4+3]*spark_gpu_probe::fixed_math::exp(table[t*4]-best);}
                    }
                    for dist in [16,8,4,2,1] {lanes=core::array::from_fn(|l|lanes[l]+lanes[l^dist]);}
                }
                let want=[best,id as f32,1.0,lanes[0]];
                if got[..4].iter().zip(&want).any(|(a,b)|a.to_bits()!=b.to_bits()) {return Err(format!("draft fold finish differs at {tiles} tiles: {:?} {:?}",&got[..4],want));}
                if tiles>1 {
                    let blocks=tiles.div_ceil(256).clamp(1,48).min(tiles);
                    let reduced=b.draft_lanes(FoldReduce{tiles,blocks,confidence,gated:false},[Some(&input),None,None])?;
                    let two_stage=b.read_vector_values(&b.draft_lanes(FoldFinish{tiles:blocks,confidence,stop_above_sum:0.0,gated:false},[Some(&reduced),None,None])?)?;
                    if two_stage[..3].iter().zip(&want[..3]).any(|(a,b)|a.to_bits()!=b.to_bits())
                        || (confidence && (two_stage[3]-want[3]).abs()>want[3].abs()*1e-5+1e-5)
                        || (!confidence && two_stage[3]!=0.0)
                    {
                        return Err(format!("draft two-stage fold reduce differs at {tiles} tiles: {:?} {:?}",&two_stage[..4],want));
                    }
                    let flag_one=b.upload_vector(&[1.0f32])?;
                    let skipped=b.draft_lanes(FoldReduce{tiles,blocks,confidence,gated:true},[Some(&input),Some(&flag_one),None])?;
                    let fin_skip=b.read_vector_values(&b.draft_lanes(FoldFinish{tiles:blocks,confidence,stop_above_sum:0.0,gated:true},[Some(&skipped),Some(&flag_one),None])?)?;
                    if fin_skip[..5]!=[0.0,0.0,0.0,0.0,1.0] {return Err("gated FoldReduce+FoldFinish did not stop".into());}
                }
            }
            for (len,bad_at,value) in [(35usize,34usize,1.125f32),(35,34,f32::INFINITY),(35,34,f32::NAN),(35,34,f32::MAX),(128,77,1.25),(128,77,f32::MAX)] {
                let mut a=vec![0.5f32;len];let mut c_vec=vec![0.25f32;len];
                a[bad_at]=value;c_vec[bad_at]=value;
                let (xd,yd)=(b.upload_vector(&a)?,b.upload_vector(&c_vec)?);
                let got=b.read_vector_values(&b.draft_lanes(AddRound(len),[Some(&xd),Some(&yd),None])?)?;
                for k in 0..len {
                    let raw=a[k]+c_vec[k];
                    let want=if raw.is_finite() {let r=rn(raw); if r.is_finite() {r} else {f32::NAN}} else {f32::NAN};
                    if (got[k].to_bits()!=want.to_bits()) && !(got[k].is_nan() && want.is_nan()) {
                        return Err(format!("draft AddRound differs at len={len} k={k}: got={} want={}",got[k],want));
                    }
                }
            }
            // Fused Markov helpers: MarkovStaticBase, MarkovPrep (anchor + from_pick), MarkovStep vs unfused ops.
            for latent in [64usize, 256usize] {
                let (vocab,rows)=(4usize,3usize);
                let draft_out=(0..rows*3*latent).map(|k|((k*17+3)%31) as f32/16.0-1.0).collect::<Vec<_>>();
                let feat_out=(0..3*latent).map(|k|((k*13+5)%29) as f32/16.0-0.8).collect::<Vec<_>>();
                let bias=(0..3*latent).map(|k|((k*7+1)%19) as f32/32.0-0.25).collect::<Vec<_>>();
                let (dd,fd,bd)=(b.upload_vector(&draft_out)?,b.upload_vector(&feat_out)?,b.upload_vector(&bias)?);
                let sb=b.read_vector_values(&b.draft_lanes(MarkovStaticBase{rows,width:3*latent},[Some(&dd),Some(&fd),Some(&bd)])?)?;
                for r in 0..rows {for j in 0..3*latent {
                    let want=(draft_out[r*3*latent+j]+feat_out[j])+bias[j];
                    if sb[r*3*latent+j].to_bits()!=want.to_bits() {return Err(format!("MarkovStaticBase differs at {r},{j}"));}
                }}
                let w1_words=(0..vocab*latent/2).map(|k|{
                    let lo=((0x3e00+((k*29)%0x0380)) as u32) | (((k&1) as u32)<<15);
                    let hi=((0x3d80+((k*37)%0x0380)) as u32) | ((((k>>1)&1) as u32)<<15);
                    f32::from_bits(lo|(hi<<16))
                }).collect::<Vec<_>>();
                let w1=b.upload_vector(&w1_words)?;
                let st_vals=(0..latent).map(|k|((k*11+2)%23) as f32/8.0-1.25).collect::<Vec<_>>();
                let st=b.upload_vector(&st_vals)?;
                // Anchor mode (state = 0, tok = 2)
                let got0=b.read_vector_values(&b.draft_lanes(MarkovPrep{latent,vocab,anchor:2,from_pick:false},[Some(&w1),None,None])?)?;
                let w1_shared=w1.into_resident_shared()?;
                let emb2=b.read_vector_values(&b.draft_lanes(Unpack(latent),[Some(&w1_shared.resident_range(2*(latent/2)..3*(latent/2))?),None,None])?)?;
                let mut u0=vec![0.0f32;latent];u0.extend_from_slice(&emb2);
                let u0_d=b.upload_vector(&u0)?;
                let want0=b.read_vector_values(&b.draft_lanes(PackActivations(2*latent),[Some(&u0_d),None,None])?)?;
                if got0.iter().map(|v|v.to_bits()).ne(want0.iter().map(|v|v.to_bits())) {return Err("MarkovPrep anchor differs".into());}
                // Pick mode (state = st, tok = 3)
                let pick=b.upload_vector(&[5.0,3.0,1.0,2.0,0.0])?;
                let got1=b.read_vector_values(&b.draft_lanes(MarkovPrep{latent,vocab,anchor:0,from_pick:true},[Some(&w1_shared),Some(&pick),Some(&st)])?)?;
                let emb3=b.read_vector_values(&b.draft_lanes(Unpack(latent),[Some(&w1_shared.resident_range(3*(latent/2)..4*(latent/2))?),None,None])?)?;
                let mut u1=st_vals.clone();u1.extend_from_slice(&emb3);
                let u1_d=b.upload_vector(&u1)?;
                let want1=b.read_vector_values(&b.draft_lanes(PackActivations(2*latent),[Some(&u1_d),None,None])?)?;
                if got1.iter().map(|v|v.to_bits()).ne(want1.iter().map(|v|v.to_bits())) {return Err("MarkovPrep from_pick differs".into());}
                // MarkovStep (has_state = true, ungated and gated run/skip) vs AddRound + Recurrent + PackActivations
                let dyn_vals=(0..3*latent).map(|k|((k*19+7)%37) as f32/16.0-1.1).collect::<Vec<_>>();
                let base_vals=sb[..3*latent].to_vec();
                let (dynd,based)=(b.upload_vector(&dyn_vals)?,b.upload_vector(&base_vals)?);
                let stepped=b.read_vector_values(&b.draft_lanes(MarkovStep{latent,has_state:true,gated:false},[Some(&dynd),Some(&based),Some(&st)])?)?;
                let proj=b.draft_lanes(AddRound(3*latent),[Some(&dynd),Some(&based),None])?;
                let pair=b.draft_lanes(Recurrent(latent),[Some(&proj),Some(&st),None])?.into_resident_shared()?;
                let want_st=b.read_vector_values(&pair.resident_range(0..latent)?)?;
                let want_lat=b.read_vector_values(&b.draft_lanes(PackActivations(latent),[Some(&pair.resident_range(latent..2*latent)?),None,None])?)?;
                if stepped[..latent].iter().map(|v|v.to_bits()).ne(want_st.iter().map(|v|v.to_bits()))
                    || stepped[latent..latent+(latent/64)*18].iter().map(|v|v.to_bits()).ne(want_lat.iter().map(|v|v.to_bits()))
                {
                    return Err(format!("MarkovStep latent={latent} differs from unfused AddRound + Recurrent + PackActivations"));
                }
                let mut dyn_g0=dyn_vals.clone();dyn_g0.push(0.0);
                let mut dyn_g1=dyn_vals.clone();dyn_g1.push(1.0);
                let (dg0,dg1)=(b.upload_vector(&dyn_g0)?,b.upload_vector(&dyn_g1)?);
                let step_g0=b.read_vector_values(&b.draft_lanes(MarkovStep{latent,has_state:true,gated:true},[Some(&dg0),Some(&based),Some(&st)])?)?;
                let tile_w=(latent/64)*18;
                if step_g0[..latent].iter().map(|v|v.to_bits()).ne(want_st.iter().map(|v|v.to_bits()))
                    || step_g0[latent..latent+tile_w].iter().map(|v|v.to_bits()).ne(want_lat.iter().map(|v|v.to_bits()))
                    || step_g0[latent+tile_w]!=0.0
                {
                    return Err(format!("MarkovStep gated(0) latent={latent} differs"));
                }
                let step_g1=b.read_vector_values(&b.draft_lanes(MarkovStep{latent,has_state:true,gated:true},[Some(&dg1),Some(&based),Some(&st)])?)?;
                if step_g1[latent+tile_w]!=1.0 {
                    return Err(format!("MarkovStep gated(1) latent={latent} did not propagate stop"));
                }
                // Batched MarkovStaticBaseBatch, MarkovPrepBatch (anchor + pick), and MarkovStepBatch vs single-chat ops.
                for batch in [2usize, 4, 8, 16, 32] {
                    let depth=2usize;
                    let w_all=3*latent;
                    let b_draft=(0..depth*batch*w_all).map(|k|((k*23+5)%37) as f32/16.0-1.0).collect::<Vec<_>>();
                    let b_feat=(0..batch*w_all).map(|k|((k*13+9)%31) as f32/16.0-0.75).collect::<Vec<_>>();
                    let (bdd,bfd)=(b.upload_vector(&b_draft)?,b.upload_vector(&b_feat)?);
                    let b_sb=b.read_vector_values(&b.draft_lanes(MarkovStaticBaseBatch{batch,depth,width:w_all},[Some(&bdd),Some(&bfd),Some(&bd)])?)?;
                    for s in 0..depth {for bi in 0..batch {for j in 0..w_all {
                        let idx=(s*batch+bi)*w_all+j;
                        let want=(b_draft[idx]+b_feat[bi*w_all+j])+bias[j];
                        if b_sb[idx].to_bits()!=want.to_bits() {return Err(format!("MarkovStaticBaseBatch differs at batch={batch} s={s} bi={bi} j={j}"));}
                    }}}
                    let anchors=(0..batch).map(|bi|(bi%vocab) as f32).collect::<Vec<_>>();
                    let anc_d=b.upload_vector(&anchors)?;
                    let prep0=b.read_vector_values(&b.draft_lanes(MarkovPrepBatch{batch,latent,vocab,from_pick:false,pick_stride:0},[Some(&w1_shared),Some(&anc_d),None])?)?;
                    let two_tiles=(2*latent/64)*18;
                    for bi in 0..batch {
                        let want=b.read_vector_values(&b.draft_lanes(MarkovPrep{latent,vocab,anchor:(bi%vocab) as u32,from_pick:false},[Some(&w1_shared),None,None])?)?;
                        if prep0[bi*two_tiles..(bi+1)*two_tiles].iter().map(|v|v.to_bits()).ne(want.iter().map(|v|v.to_bits())) {
                            return Err(format!("MarkovPrepBatch anchor differs at batch={batch} bi={bi}"));
                        }
                    }
                    let stride=20usize;
                    let mut picks_all=vec![0.0f32;batch*stride];
                    let b_st=(0..batch*latent).map(|k|((k*17+3)%29) as f32/8.0-1.1).collect::<Vec<_>>();
                    for bi in 0..batch {
                        picks_all[bi*stride..bi*stride+5].copy_from_slice(&[4.0,((bi+1)%vocab) as f32,1.0,2.0,0.0]);
                    }
                    let (picks_d,b_st_d)=(b.upload_vector(&picks_all)?,b.upload_vector(&b_st)?);
                    let prep1=b.read_vector_values(&b.draft_lanes(MarkovPrepBatch{batch,latent,vocab,from_pick:true,pick_stride:stride},[Some(&w1_shared),Some(&picks_d),Some(&b_st_d)])?)?;
                    for bi in 0..batch {
                        let pk=b.upload_vector(&picks_all[bi*stride..bi*stride+5])?;
                        let st_bi=b.upload_vector(&b_st[bi*latent..(bi+1)*latent])?;
                        let want=b.read_vector_values(&b.draft_lanes(MarkovPrep{latent,vocab,anchor:0,from_pick:true},[Some(&w1_shared),Some(&pk),Some(&st_bi)])?)?;
                        if prep1[bi*two_tiles..(bi+1)*two_tiles].iter().map(|v|v.to_bits()).ne(want.iter().map(|v|v.to_bits())) {
                            return Err(format!("MarkovPrepBatch from_pick differs at batch={batch} bi={bi}"));
                        }
                    }
                    let b_dyn=(0..batch*w_all).map(|k|((k*19+11)%41) as f32/16.0-1.05).collect::<Vec<_>>();
                    let b_base=b_sb[..batch*w_all].to_vec();
                    let (b_dyn_d,b_base_d)=(b.upload_vector(&b_dyn)?,b.upload_vector(&b_base)?);
                    let b_step=b.read_vector_values(&b.draft_lanes(MarkovStepBatch{batch,latent,has_state:true},[Some(&b_dyn_d),Some(&b_base_d),Some(&b_st_d)])?)?;
                    for bi in 0..batch {
                        let d_bi=b.upload_vector(&b_dyn[bi*w_all..(bi+1)*w_all])?;
                        let ba_bi=b.upload_vector(&b_base[bi*w_all..(bi+1)*w_all])?;
                        let st_bi=b.upload_vector(&b_st[bi*latent..(bi+1)*latent])?;
                        let want=b.read_vector_values(&b.draft_lanes(MarkovStep{latent,has_state:true,gated:false},[Some(&d_bi),Some(&ba_bi),Some(&st_bi)])?)?;
                        if b_step[bi*latent..(bi+1)*latent].iter().map(|v|v.to_bits()).ne(want[..latent].iter().map(|v|v.to_bits()))
                            || b_step[batch*latent+bi*tile_w..batch*latent+(bi+1)*tile_w].iter().map(|v|v.to_bits()).ne(want[latent..latent+tile_w].iter().map(|v|v.to_bits()))
                        {
                            return Err(format!("MarkovStepBatch differs at latent={latent} batch={batch} bi={bi}"));
                        }
                    }
                }
            }
            println!("DRAFT LANES PASS");
            let mut panel_cases = Vec::new();
            for (rows, cols) in [(91usize, 128usize), (49, 256), (65, 2304)] {
                let values = (0..rows * cols).map(|i| {
                    let finite = 0x3a00 + (i.wrapping_mul(31415) as u16 & 0x07ff);
                    if i == cols * 17 { 0x8000 } else { finite | ((i as u16 & 1) << 15) }
                }).collect::<Vec<_>>();
                let raw = matrix(b, &values, 0)?;
                let panel = matrix(b, &values, 0)?;
                let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                let before = b.allocated_bytes();
                let size = b.register_bf16_panels(&panel, rows, cols, &bytes)?;
                if spark_gpu_probe::bf16_panels::storage_words(rows, cols).is_none_or(|max| size / 4 > max)
                    || b.allocated_bytes() - before != size
                {
                    return Err("BF16 panel allocation differs from the budget".into());
                }
                panel_cases.push((rows, cols, raw, panel));
            }
            // The one-row panel form (`resident_launch_plain_staged_one`, word 7 = 1) reads the same registered panels. Tile counts
            // 16 and 20 (one ring of 12 and a tail) and 144 (twelve rings: a full 2,304-column panel row); two bands past the
            // last full group; RAW tiles (every other 16-column tile has wide high bytes). Then two products after a norm.
            // The panels are registered here, before launches are recorded; `raw` is a second copy with no panels.
            let mut one_cases = Vec::new();
            for (rows, cols, mixed) in [(17usize, 256usize, true), (64, 256, true), (129, 256, true), (64, 320, true), (129, 2304, true), (129, 2304, false), (1024, 256, true), (4160, 256, false)] {
                let values = (0..rows*cols).map(|i| if mixed && i / 16 % 2 == 0 {
                    0x3c00 | ((i % 1024) as u16) | (((i % 3 == 0) as u16) << 15)
                } else { weight(i) }).collect::<Vec<_>>();
                let raw = matrix(b, &values, 0)?;
                let panel = matrix(b, &values, 0)?;
                let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                b.register_bf16_panels(&panel, rows, cols, &bytes)?;
                one_cases.push((rows, cols, mixed, raw, panel));
            }
            let queue = b.resident_queue()?;
            b.enable_resident_launches()?;
            let recorded = |kind: Kind| b.stats().recorded_kinds[kind as usize];
            let before=recorded(Kind::DraftLanes);
            check(Values{rows:2,heads:2,kv:1,width:1,keys:3},&[6.0;4])?;
            if recorded(Kind::DraftLanes)!=before+1 {return Err("draft lane did not enter the resident graph".into());}
            for bad in [false,true] {
                let mut logits=vec![-1.0;2051];
                logits[31]=0.0;logits[1024] = -0.0;logits[2050]=0.0;
                if bad {logits[2048]=f32::NAN;}
                let input=b.upload_vector(&logits)?;
                let partials=b.draft_lanes(PickPartials(logits.len()),[Some(&input),None,None])?;
                let pick=b.draft_lanes(PickFinish(partials.len()/3),[Some(&partials),None,None])?;
                let mut scratch=b.vector(partials.len())?;let mut reference=b.vector(3)?;
                b.enqueue_argmax_batch_device(&input,&mut scratch,&mut reference,1)?;
                let bits=|v:&DeviceVector<'_>|->Result<Vec<u32>,String> {
                    Ok(b.read_vector_values(v)?.iter().map(|value|value.to_bits()).collect())
                };
                if bits(&pick)?!=bits(&reference)? {return Err("recorded draft pick differs from the layer pick".into());}
                // ProbPart against the same pick: each lane sums `exp(logit - best)` over its words of the slice in
                // ascending order, then the warp adds across lanes (own word plus the other lane's, distances 16 to 1).
                // A pick whose flag is not 1 (the NaN row) gives zeros. Slices of 512 and of 32 (the last one short).
                let picked=b.read_vector_values(&pick)?;
                for chunk in [512,32] {
                    let parts=b.draft_lanes(ProbPart{n:logits.len(),chunk},[Some(&input),Some(&pick),None])?;
                    let want=(0..logits.len().div_ceil(chunk)).map(|part|{
                        let mut lanes=[0.0f32;32];
                        if picked[2]==1.0 {
                            for (lane,sum) in lanes.iter_mut().enumerate() {
                                let mut at=part*chunk+lane;
                                while at<(part*chunk+chunk).min(logits.len()) {
                                    *sum+=spark_gpu_probe::fixed_math::exp(logits[at]-picked[0]);at+=32;
                                }
                            }
                        }
                        for distance in [16,8,4,2,1] {let before=lanes;for (lane,sum) in lanes.iter_mut().enumerate() {*sum=before[lane]+before[lane^distance];}}
                        lanes[0].to_bits()
                    }).collect::<Vec<_>>();
                    if bits(&parts)?!=want {return Err(format!("draft lane ProbPart differs (chunk {chunk}, flag {})",picked[2]));}
                }
            }
            println!("DRAFT PICK PASS");
            let mut panel_words = 0;
            for (rows, cols, raw, panel) in &panel_cases {
                let (rows, cols) = (*rows, *cols);
                for inputs in 1..=16 {
                    for first in [0, 16] {
                        let window = first * cols / 2..rows * cols / 2;
                        let n = rows - first;
                        let x = b.upload_vector(&(0..inputs * cols).map(input).collect::<Vec<_>>())?;
                        let baseline = poisoned(b, n * inputs)?;
                        let candidate = poisoned(b, n * inputs)?;
                        b.project_staged_few_into(n, cols, inputs, n, &x, &raw.resident_range(window.clone())?,
                            &mut baseline.resident_range(GUARD..GUARD + n * inputs)?)?;
                        b.project_staged_few_into(n, cols, inputs, n, &x, &panel.resident_range(window)?,
                            &mut candidate.resident_range(GUARD..GUARD + n * inputs)?)?;
                        let want = b.read_vector_values(&baseline)?;
                        let got = b.read_vector_values(&candidate)?;
                        if let Some(at) = want.iter().zip(&got).position(|(a,b)| a.to_bits() != b.to_bits()) {
                            return Err(format!("BF16 panel mismatch {n} x {cols}, inputs={inputs}, first={first}, word={at}"));
                        }
                        panel_words += n * inputs;
                    }
                }
            }
            println!("BF16 PANEL PASS compared_words={panel_words} widths=1..16");
            // Multi-chat kept-ring `fast_short_conv_multi_into`: 4..=64 chats up to 128 total rows (taps=3 and taps=4)
            // against independent per-chat single-chat launches (exact output + every + guard words).
            {
                use spark_gpu_probe::fast_ops::ShortConvMulti;
                let spec_c16: Vec<(usize, usize)> = (0..16).map(|ci| (2 + (ci % 3), ci % 3)).collect(); // 48 rows
                let spec_c32: Vec<(usize, usize)> = (0..32).map(|ci| (2 + (ci % 2), ci % 4)).collect(); // 80 rows
                let spec_c64: Vec<(usize, usize)> = (0..64).map(|ci| (2, ci % 3)).collect(); // 128 rows
                for &(heads, dim, taps, form, chats_spec) in &[
                    (1usize, 320usize, 3usize, 6u32, &[(2usize, 0usize), (3, 1), (1, 2), (4, 0)][..]),
                    (4, 128, 4, 1, &[(2, 0), (2, 1), (2, 2), (2, 3), (2, 1), (2, 2), (2, 0), (2, 3)][..]),
                    (1, 320, 3, 6, spec_c16.as_slice()),
                    (4, 128, 4, 1, spec_c32.as_slice()),
                    (1, 320, 3, 6, spec_c64.as_slice()),
                ] {
                    let width = heads * dim;
                    let total_rows: usize = chats_spec.iter().map(|&(c, _)| c).sum();
                    let bound = 0.5f32;
                    let fixed_host = (0..width * taps).map(|k| input(k + 7) * 0.25).collect::<Vec<_>>();
                    let inputs_host = (0..total_rows * width).map(|k| input(k + 31)).collect::<Vec<_>>();
                    let token_host = (form & 1 != 0).then(|| (0..total_rows * heads * taps).map(|k| input(k + 59) * 0.125).collect::<Vec<_>>());
                    let add_host = (form & 2 != 0).then(|| (0..total_rows * width).map(|k| input(k + 97)).collect::<Vec<_>>());
                    let mut rings_host = Vec::with_capacity(chats_spec.len() * taps * width);
                    for (ci, _) in chats_spec.iter().enumerate() {
                        for k in 0..taps * width { rings_host.push(input(k + ci * 131 + 13)); }
                    }
                    let fixed_d = b.upload_vector(&fixed_host)?.into_resident_shared()?;
                    let inputs_d = b.upload_vector(&inputs_host)?.into_resident_shared()?;
                    let token_d = token_host.as_ref().map(|v| b.upload_vector(v)?.into_resident_shared()).transpose()?;
                    let add_d = add_host.as_ref().map(|v| b.upload_vector(v)?.into_resident_shared()).transpose()?;
                    let rings_d = b.upload_vector(&rings_host)?.into_resident_shared()?;
                    let (mut start_lo, mut start_hi, mut slots_lo, mut slots_hi, mut off) = (0u64, 0u64, 0u64, 0u64, 0usize);
                    let (mut want_out, mut want_every) = (Vec::new(), Vec::new());
                    for (ci, &(cnt, slot)) in chats_spec.iter().enumerate() {
                        if off < 64 { start_lo |= 1u64 << off; } else { start_hi |= 1u64 << (off - 64); }
                        if ci < 32 { slots_lo |= (slot as u64) << (2 * ci); } else { slots_hi |= (slot as u64) << (2 * (ci - 32)); }
                        let x_c = inputs_d.resident_range(off * width..(off + cnt) * width)?;
                        let r_c = rings_d.resident_range(ci * taps * width..(ci + 1) * taps * width)?;
                        let t_c = token_d.as_ref().map(|v| v.resident_range(off * heads * taps..(off + cnt) * heads * taps)).transpose()?;
                        let a_c = add_d.as_ref().map(|v| v.resident_range(off * width..(off + cnt) * width)).transpose()?;
                        let mut o_c = b.vector(cnt * width)?;
                        let mut e_c = b.vector(cnt * taps * width)?;
                        b.fast_short_conv_win_into(cnt, heads, dim, taps, form, bound, &x_c, t_c.as_ref(), a_c.as_ref(), &fixed_d, Some((&r_c, slot)), &mut o_c, Some((&mut e_c, slot)))?;
                        want_out.extend(b.read_vector_values(&o_c)?);
                        want_every.extend(b.read_vector_values(&e_c)?);
                        off += cnt;
                    }
                    let multi = ShortConvMulti { chats: chats_spec.len(), start_mask: [start_lo, start_hi, 0, 0], slots_lo, slots_hi };
                    let got_o = poisoned(b, total_rows * width)?;
                    let got_e = poisoned(b, total_rows * taps * width)?;
                    let mut got_o_s = got_o.resident_range(GUARD..GUARD + total_rows * width)?;
                    let mut got_e_s = got_e.resident_range(GUARD..GUARD + total_rows * taps * width)?;
                    b.fast_short_conv_multi_into(total_rows, heads, dim, taps, form, bound, &inputs_d, token_d.as_ref(), add_d.as_ref(), &fixed_d,
                        &rings_d, &mut got_o_s, &mut got_e_s, multi)?;
                    let back_o = b.read_vector_values(&got_o)?;
                    let back_e = b.read_vector_values(&got_e)?;
                    if back_o[..GUARD].iter().chain(&back_o[GUARD + want_out.len()..]).any(|w| w.to_bits() != POISON)
                        || back_e[..GUARD].iter().chain(&back_e[GUARD + want_every.len()..]).any(|w| w.to_bits() != POISON)
                    {
                        return Err("multi-chat short_conv_keep changed a guard word".into());
                    }
                    if back_o[GUARD..GUARD + want_out.len()].iter().zip(&want_out).any(|(a, b)| a.to_bits() != b.to_bits())
                        || back_e[GUARD..GUARD + want_every.len()].iter().zip(&want_every).any(|(a, b)| a.to_bits() != b.to_bits())
                    {
                        return Err(format!("multi-chat short_conv_keep differs for total_rows={total_rows} chats={} taps={taps}", chats_spec.len()));
                    }
                }
                println!("FEW CHECK MULTI-CHAT SHORT CONV KEEP cases=5 chats=4..64 rows=10..128 taps=3,4 exact output+every and guards");
            }
            let (few_before, row_before) = ((recorded(Kind::PlainStagedFew) + recorded(Kind::PlainStagedOne)), recorded(Kind::PlainStaged));
            let mut few = Count { cases: 0, words: 0, different: 0, row_different: 0, guards: 0, changed: 0 };
            let mut row_launches = 0usize;
            let (rows, cols) = (257, 256);
            let weights = matrix(b, &vec![0x3f80; rows * cols], 0)?;
            let mut x = vec![0.0; 2 * cols];
            x[0] = 1_073_741_824.0;
            x[64] = -x[0];
            x[80] = 1.0;
            x.copy_within(..cols, cols);
            let source = b.upload_vector(&x)?.into_resident_shared()?;
            let first = source.resident_range(0..cols)?;
            let second = source.resident_range(cols..2 * cols)?;
            let grouped = poisoned(b, 2 * rows)?;
            let batched = poisoned(b, 2 * rows)?;
            let prefill = poisoned(b, 2 * rows)?;
            b.project_staged_narrow_group_into(rows, cols, &[&first, &second], &[&weights; 2], &mut grouped.resident_range(GUARD..GUARD + 2 * rows)?)?;
            b.project_staged_few_into(rows, cols, 2, rows, &source, &weights, &mut batched.resident_range(GUARD..GUARD + 2 * rows)?)?;
            b.project_tc_rows_into(rows, cols, 2, &source, &weights, &mut prefill.resident_range(GUARD..GUARD + 2 * rows)?)?;
            let want = vec![1.0; 2 * rows];
            compare(b, &batched, &want, &mut few)?;
            let mut across = Count { cases: 0, words: 0, different: 0, row_different: 0, guards: 0, changed: 0 };
            compare(b, &grouped, &want, &mut across)?;
            compare(b, &prefill, &want, &mut across)?;
            // A scalar activation path retains the extra FP32 bit. All admitted batch widths must round it to BF16.
            for inputs in [7, 8, 16, 17] {
                let mut x = vec![0.0; inputs * cols];
                x.chunks_exact_mut(cols).for_each(|row| row[0] = 1.0 + 1.0 / 1024.0);
                let input = b.upload_vector(&x)?;
                let want = vec![1.0; inputs * rows];
                let out = poisoned(b, inputs * rows)?;
                let mut words = out.resident_range(GUARD..GUARD + inputs * rows)?;
                if inputs <= spark_gpu_probe::few_product::FEW_MAX {
                    b.project_staged_few_into(rows, cols, inputs, rows, &input, &weights, &mut words)?;
                    compare(b, &out, &want, &mut few)?;
                }
                b.project_tc_rows_into(rows, cols, inputs, &input, &weights, &mut words)?;
                compare(b, &out, &want, &mut across)?;
            }
            if across.different + across.changed != 0 {
                return Err(format!("BF16 entry comparison: {} different words, {} changed guards", across.different, across.changed));
            }
            println!("FEW CHECK CROSS-ENTRY cases={} words={} different={} guard_words_changed={}", across.cases, across.words, across.different, across.changed);
            for &(rows, cols) in &SHAPES { for inputs in INPUTS { for lead in [0usize, 4] {
                let values = (0..rows * cols).map(|i| weight(i + 3 * rows + inputs + lead)).collect::<Vec<_>>();
                let x = (0..inputs * cols).map(|k| input(k + 5 * inputs + lead)).collect::<Vec<_>>();
                let stored = matrix(b, &values, lead)?;
                let weights = stored.resident_range(lead..lead + rows * cols / 2)?;
                let source = b.upload_vector(&vec![0.0; lead].into_iter().chain(x.iter().copied()).collect::<Vec<_>>())?.into_resident_shared()?;
                let all = source.resident_range(lead..lead + inputs * cols)?;
                if !b.staged_few_ready(rows, cols, inputs, &all, &weights) { return Err(format!("the few product is not ready for {rows} x {cols}, {inputs} input rows")); }
                let whole = poisoned(b, inputs * rows)?;
                b.project_staged_few_into(rows, cols, inputs, rows, &all, &weights, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?)?;
                let (vals, xs) = (&values, &x);
                let narrow = spark_gpu_probe::staged_narrow::shape_admitted(rows, cols);
                let want: Vec<f32> = (0..inputs).flat_map(|t| (0..rows).map(move |r| {
                    let (wrow, xrow) = (&vals[r * cols..(r + 1) * cols], &xs[t * cols..(t + 1) * cols]);
                    spark_gpu_probe::tc_rows::tc_word(wrow, xrow)
                })).collect();
                let got = compare(b, &whole, &want, &mut few)?;
                // Check every input against the single-row entry for the same matrix shape.
                let wide = cols > spark_gpu_probe::staged_product::COLS_MAX;
                if !wide {
                    for t in 0..inputs {
                        let one = b.vector(rows)?.into_resident_shared()?;
                        let mut out = one.resident_range(0..rows)?;
                        let x = all.resident_range(t * cols..(t + 1) * cols)?;
                        if narrow {
                            b.project_staged_narrow_into(rows, cols, &x, &weights, &mut out)?;
                        } else {
                            b.project_staged_into(rows, cols, &x, &weights, &mut out)?;
                            row_launches += 1;
                        }
                        let back = b.read_vector_values(&one)?;
                        few.row_different += back.iter().zip(&got[t * rows..(t + 1) * rows]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    }
                }
            } } }
            // PARTS: a matrix of 512 rows as two parts of 256 rows that write their columns of one [inputs][512]
            // output in place (the stride), as a matrix above the envelope runs.
            for inputs in [2usize, 8, 16] {
                let (rows, cols, part) = (512usize, 128usize, 256usize);
                let values = (0..rows * cols).map(|i| weight(i + 11 * inputs)).collect::<Vec<_>>();
                let x = (0..inputs * cols).map(|k| input(k + 3 * inputs)).collect::<Vec<_>>();
                let stored = matrix(b, &values, 0)?;
                let all = b.upload_vector(&x)?.into_resident_shared()?;
                let whole = poisoned(b, inputs * rows)?;
                for first in [0usize, part] {
                    let weights = stored.resident_range(first * cols / 2..(first + part) * cols / 2)?;
                    b.project_staged_few_into(part, cols, inputs, rows, &all, &weights, &mut whole.resident_range(GUARD + first..GUARD + first + (inputs - 1) * rows + part)?)?;
                }
                let (vals, xs) = (&values, &x);
                let want: Vec<f32> = (0..inputs).flat_map(|t| (0..rows).map(move |r| {
                    let (wrow, xrow) = (&vals[r * cols..(r + 1) * cols], &xs[t * cols..(t + 1) * cols]);
                    spark_gpu_probe::tc_rows::tc_word(wrow, xrow)
                })).collect();
                compare(b, &whole, &want, &mut few)?;
            }
            let parts_extra = 3usize;
            println!("FEW CHECK FEW cases={} words={} different={} different_from_row_product={} guard_words={} guard_words_changed={} recorded_launches={} row_launches={}",
                few.cases, few.words, few.different, few.row_different, few.guards, few.changed, (recorded(Kind::PlainStagedFew) + recorded(Kind::PlainStagedOne)) - few_before, recorded(Kind::PlainStaged) - row_before);
            // REFUSALS: 1 and 17 input rows, an input off the 16-byte step, an input one word short, an output one
            // word short, weights one word short, columns not whole stages, a stride below the rows, an output that is
            // the input. Nothing is recorded and no word of the output changes.
            let (rows, cols, inputs) = (256usize, 128usize, 4usize);
            let values = (0..rows * cols).map(weight).collect::<Vec<_>>();
            let weights = matrix(b, &values, 0)?;
            let source = b.upload_vector(&(0..17 * cols + 8).map(input).collect::<Vec<_>>())?.into_resident_shared()?;
            let whole = poisoned(b, 17 * rows)?;
            let few_at = recorded(Kind::PlainStagedFew) + recorded(Kind::PlainStagedOne);
            let mut refused = 0usize;
            let mut expect = |what: &str, result: Result<(), String>| -> Result<(), String> {
                match result { Err(error) if error.starts_with("few") => { refused += 1; Ok(()) }, other => Err(format!("{what}: not refused by the host check: {other:?}")) }
            };
            let out = |words: usize| whole.resident_range(GUARD..GUARD + words);
            expect("zero input rows", b.project_staged_few_into(rows, cols, 0, rows, &source.resident_range(0..cols)?, &weights, &mut out(rows)?))?;
            expect("17 input rows", b.project_staged_few_into(rows, cols, 17, rows, &source.resident_range(0..17 * cols)?, &weights, &mut out(17 * rows)?))?;
            expect("input off the 16-byte step", b.project_staged_few_into(rows, cols, inputs, rows, &source.resident_range(1..1 + inputs * cols)?, &weights, &mut out(inputs * rows)?))?;
            expect("input one word short", b.project_staged_few_into(rows, cols, inputs, rows, &source.resident_range(0..inputs * cols - 1)?, &weights, &mut out(inputs * rows)?))?;
            expect("output one word short", b.project_staged_few_into(rows, cols, inputs, rows, &source.resident_range(0..inputs * cols)?, &weights, &mut out(inputs * rows - 1)?))?;
            expect("weights one word short", b.project_staged_few_into(rows, cols, inputs, rows, &source.resident_range(0..inputs * cols)?, &weights.resident_range(0..rows * cols / 2 - 1)?, &mut out(inputs * rows)?))?;
            expect("columns not whole stages", b.project_staged_few_into(rows, 96, inputs, rows, &source.resident_range(0..inputs * 96)?, &weights.resident_range(0..rows * 96 / 2)?, &mut out(inputs * rows)?))?;
            expect("stride below the rows", b.project_staged_few_into(rows, cols, inputs, rows - 1, &source.resident_range(0..inputs * cols)?, &weights, &mut out((inputs - 1) * (rows - 1) + rows)?))?;
            expect("the output is the input", b.project_staged_few_into(cols / 2, cols, 2, cols / 2, &source.resident_range(0..2 * cols)?, &weights.resident_range(0..cols * cols / 4)?, &mut source.resident_range(0..cols)?))?;
            let untouched = b.read_vector_values(&whole)?.iter().filter(|word| word.to_bits() != POISON).count();
            let stray = (recorded(Kind::PlainStagedFew) + recorded(Kind::PlainStagedOne)) - few_at;
            println!("FEW CHECK REFUSALS cases={refused} recorded_launches={stray} output_words_changed={untouched}");
            drop(queue);
            if few.different + few.row_different + few.changed + untouched > 0 || stray > 0 {
                return Err(format!("few: {} of {} words differ from the host chain, {} from the row product, {} guard words changed; refusals: {stray} launches recorded, {untouched} output words changed",
                    few.different, few.words, few.row_different, few.changed));
            }
            if (recorded(Kind::PlainStagedFew) + recorded(Kind::PlainStagedOne)) - few_before != (few.cases + parts_extra) as u128 || recorded(Kind::PlainStaged) - row_before != row_launches as u128 {
                return Err("a case did not record one launch of its entry".into());
            }
            // The weights of the 2-to-16-row gate blend cases below, each raw and with its BF16 panels (registered while no
            // queue runs): the decode's forget and gain stacks have panels, so the joined launch reads them.
            let blend_cases = [(384usize, 2usize), (608, 5), (384, 8), (2304, 3), (2304, 8), (384, 9), (608, 12), (2304, 16)].into_iter().map(|(dim, inputs)| {
                let (rows, cols) = (2 * dim, 64);
                let values = (0..rows*cols).map(|i| {
                    let v = input(i+5) * 0.125;
                    (spark_gpu_probe::bf16::rounded(v).to_bits() >> 16) as u16
                }).collect::<Vec<_>>();
                let raw = matrix(b, &values, 0)?;
                let panel = matrix(b, &values, 0)?;
                let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                b.register_bf16_panels(&panel, rows, cols, &bytes)?;
                Ok((dim, inputs, raw, panel))
            }).collect::<Result<Vec<_>, String>>()?;
            // Multi-part `FEW_PANEL_RING` matrices registered while the stream is idle: 1-warp multi-part and 4-warp split.
            let ring_cases = [(1024usize, 2304usize, &[1024usize][..]), (3072, 512, &[1024, 1024, 1024])].into_iter().map(|(rows, cols, splits)| {
                let mut w_host = (0..rows * cols).map(|k| (((k * 29 + 11) % 239) as u16) << 6 | 0x3c00).collect::<Vec<_>>();
                w_host[0] = 0x7f7f;
                let raw = matrix(b, &w_host, 0)?;
                let packed = matrix(b, &w_host, 0)?;
                let bytes = w_host.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                b.register_bf16_panels(&packed, rows, cols, &bytes)?;
                Ok((rows, cols, splits, raw, packed))
            }).collect::<Result<Vec<_>, String>>()?;
            // Preserve both projection halves as well as the blended row.
            // Force the reference launches apart by reading the projection.
            let queue = b.resident_queue()?;
            b.enable_resident_launches()?;
            for dim in [384, 608] {
                let cols = 64;
                let rows = 2 * dim;
                let weights = matrix(b, &(0..rows*cols).map(|i| {
                    let v = input(i+3) * 0.125;
                    (spark_gpu_probe::bf16::rounded(v).to_bits() >> 16) as u16
                }).collect::<Vec<_>>(), 0)?;
                let x = b.upload_vector(&(0..cols).map(input).collect::<Vec<_>>())?;
                let kept = b.upload_vector(&(0..dim).map(|i| input(i+17)).collect::<Vec<_>>())?;
                let message = b.upload_vector(&(0..dim).map(|i| input(i+91)).collect::<Vec<_>>())?;
                let run_pair = |split: bool| -> Result<(Vec<u32>,Vec<u32>),String> {
                    let gates = poisoned(b, rows)?;
                    let result = poisoned(b, dim)?;
                    let mut g = gates.resident_range(GUARD..GUARD+rows)?;
                    let mut y = result.resident_range(GUARD..GUARD+dim)?;
                    b.project_staged_few_into(rows,cols,1,rows,&x,&weights,&mut g)?;
                    if split { b.read_vector_values(&gates)?; }
                    b.fast_gate_blend_into(1,dim,&g,&kept,&message,&mut y)?;
                    let read = |v: &DeviceVector<'_>| -> Result<Vec<u32>,String> {
                        let words = b.read_vector_values(v)?.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                        if words[..GUARD].iter().chain(&words[words.len()-GUARD..]).any(|&w| w != POISON) {
                            return Err("gate blend changed a guard word".into());
                        }
                        Ok(words)
                    };
                    Ok((read(&gates)?,read(&result)?))
                };
                if run_pair(true)? != run_pair(false)? { return Err("gate blend changed a projection or output word".into()); }
            }
            // The 2-to-8-row attached form (few_product::FEW_GATE_BLEND): the few-row product and the blend of its rows
            // recorded one after the other (joined) against the same two launches apart (the projection read between
            // them), every projection word, blended word and guard word, raw and panel weights; the two weights also
            // give the same words.
            for (dim, inputs, raw, panel) in &blend_cases {
                let (dim, inputs, cols) = (*dim, *inputs, 64);
                let rows = 2 * dim;
                let x = b.upload_vector(&(0..inputs*cols).map(|i| input(i+29)).collect::<Vec<_>>())?;
                let kept = b.upload_vector(&(0..inputs*dim).map(|i| input(i+17)).collect::<Vec<_>>())?;
                let message = b.upload_vector(&(0..inputs*dim).map(|i| input(i+91)).collect::<Vec<_>>())?;
                let run_rows = |weights: &DeviceVector<'_>, split: bool| -> Result<(Vec<u32>,Vec<u32>),String> {
                    let gates = poisoned(b, inputs*rows)?;
                    let result = poisoned(b, inputs*dim)?;
                    let mut g = gates.resident_range(GUARD..GUARD+inputs*rows)?;
                    let mut y = result.resident_range(GUARD..GUARD+inputs*dim)?;
                    b.project_staged_few_into(rows,cols,inputs,rows,&x,weights,&mut g)?;
                    if split { b.read_vector_values(&gates)?; }
                    b.fast_gate_blend_into(inputs,dim,&g,&kept,&message,&mut y)?;
                    let read = |v: &DeviceVector<'_>| -> Result<Vec<u32>,String> {
                        let words = b.read_vector_values(v)?.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                        if words[..GUARD].iter().chain(&words[words.len()-GUARD..]).any(|&w| w != POISON) {
                            return Err("gate blend rows changed a guard word".into());
                        }
                        Ok(words)
                    };
                    Ok((read(&gates)?,read(&result)?))
                };
                let apart = run_rows(raw, true)?;
                for (form, weights) in [("raw", raw), ("panel", panel)] {
                    if run_rows(weights, false)? != apart || run_rows(weights, true)? != apart {
                        return Err(format!("gate blend rows changed a projection or output word (dim {dim}, {inputs} rows, {form} weights)"));
                    }
                }
            }
            for (rows, cols, _, raw, packed) in one_cases.iter().filter(|case| case.2) {
                let (rows, cols) = (*rows, *cols);
                let input = b.upload_vector(&(0..cols).map(input).collect::<Vec<_>>())?;
                let original = poisoned(b, rows)?;
                let panel = poisoned(b, rows)?;
                b.project_staged_few_into(rows,cols,1,rows,&input,raw,&mut original.resident_range(GUARD..GUARD+rows)?)?;
                b.project_staged_one_panels_into(rows,cols,rows,&input,packed,&mut panel.resident_range(GUARD..GUARD+rows)?)?;
                let expect = b.read_vector_values(&original)?;
                let actual = b.read_vector_values(&panel)?;
                for (index,(left,right)) in expect.iter().zip(actual.iter()).enumerate() {
                    if left.to_bits() != right.to_bits() ||
                        (!(GUARD..GUARD+rows).contains(&index) && right.to_bits()!=POISON) {
                        return Err(format!("decode panels differ at word {index} for {rows} rows"));
                    }
                }
            }
            println!("FEW CHECK PANELS cases=5 compact and outlier tiles, exact rows and guards");
            // The panel form for 1 to 8 input rows (`project_staged_panels_rows_into`): every one-row case, every input
            // count, the input staged whole and (2,304 columns) in parts of 12, 48 and 72 column tiles, the output at its
            // own stride and at a wider one; split outputs; then the automatic choice on the tall matrix. Each against
            // the raw matrix in the few-row (or row) forms, every word of the outputs, guards and stride gaps included.
            let mut rows_cases = 0usize;
            for (rows, cols, _, raw, packed) in &one_cases {
                let (rows, cols) = (*rows, *cols);
                let parts: &[Option<usize>] = if cols == 2304 { &[None, Some(12), Some(48), Some(72)] } else { &[None] };
                for inputs in 1..=8usize {
                    let x = b.upload_vector(&(0..inputs * cols).map(|k| input(k + 77 * inputs)).collect::<Vec<_>>())?;
                    for &part in parts {
                        for stride in [rows, rows + 3] {
                            let words = (inputs - 1) * stride + rows;
                            let (want, got) = (poisoned(b, words)?, poisoned(b, words)?);
                            b.project_staged_few_into(rows, cols, inputs, stride, &x, raw, &mut want.resident_range(GUARD..GUARD + words)?)?;
                            b.project_staged_panels_rows_into(rows, cols, inputs, stride, part, &x, packed, &mut got.resident_range(GUARD..GUARD + words)?)?;
                            let (want, got) = (b.read_vector_values(&want)?, b.read_vector_values(&got)?);
                            if let Some(at) = want.iter().zip(&got).position(|(a, b)| a.to_bits() != b.to_bits()) {
                                return Err(format!("panel rows differ at word {at}: {rows} x {cols}, inputs {inputs}, part {part:?}, stride {stride}"));
                            }
                            rows_cases += 1;
                        }
                    }
                    if [2usize, 5, 8].contains(&inputs) {
                        let widths = [rows / 3, rows / 3, rows - 2 * (rows / 3)];
                        for &part in parts {
                            let run = |panel_form: bool| -> Result<Vec<u32>, String> {
                                let outs = widths.iter().map(|&w| poisoned(b, inputs * w)).collect::<Result<Vec<_>, _>>()?;
                                let mut views = outs.iter().zip(widths).map(|(o, w)| o.resident_range(GUARD..GUARD + inputs * w)).collect::<Result<Vec<_>, _>>()?;
                                let mut refs = views.iter_mut().zip(widths).collect::<Vec<_>>();
                                if panel_form { b.project_staged_panels_split_into(rows, cols, inputs, part, &x, packed, &mut refs)?; }
                                else { b.project_staged_few_split_into(rows, cols, inputs, &x, raw, &mut refs)?; }
                                let mut bits = Vec::new();
                                for out in &outs { bits.extend(b.read_vector_values(out)?.into_iter().map(f32::to_bits)); }
                                Ok(bits)
                            };
                            if run(false)? != run(true)? {
                                return Err(format!("panel split differs: {rows} x {cols}, inputs {inputs}, part {part:?}"));
                            }
                            rows_cases += 1;
                        }
                    }
                }
            }
            // The automatic choice: the tall matrix (at least 32 rows a multiprocessor) takes the panel form for 2 to 8
            // input rows as for one.
            let (rows, cols, _, raw, packed) = one_cases.iter().find(|case| case.0 == 4160).ok_or("few-check: no tall panel case")?;
            for inputs in [2usize, 8] {
                let (rows, cols) = (*rows, *cols);
                let x = b.upload_vector(&(0..inputs * cols).map(|k| input(k + 5 * inputs)).collect::<Vec<_>>())?;
                let (want, got) = (poisoned(b, inputs * rows)?, poisoned(b, inputs * rows)?);
                b.project_staged_few_into(rows, cols, inputs, rows, &x, raw, &mut want.resident_range(GUARD..GUARD + inputs * rows)?)?;
                let before = recorded(Kind::PlainStagedOne);
                b.project_staged_few_into(rows, cols, inputs, rows, &x, packed, &mut got.resident_range(GUARD..GUARD + inputs * rows)?)?;
                if recorded(Kind::PlainStagedOne) != before + 1 { return Err(format!("the tall panel matrix did not take the panel form for {inputs} rows")); }
                if b.read_vector_values(&want)?.iter().zip(&b.read_vector_values(&got)?).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    return Err(format!("the automatic panel form differs for {inputs} rows"));
                }
                rows_cases += 1;
            }
            println!("FEW CHECK PANEL ROWS cases={rows_cases} inputs=1..8, staged parts, strides, split outputs, exact words and guards");
            // A panel product right after a norm is a dependent launch (`resident_launch::dependent_launch`): its first
            // panel loads run while the norm runs, and it reads the norm's output after griddepcontrol.wait. The same
            // product launched after the norm has ended, and the row product of the raw matrix, give the same words.
            for (rows, cols, _, raw, packed) in one_cases.iter().filter(|case| !case.2) {
                let (rows, cols) = (*rows, *cols);
                let x = b.upload_vector(&(0..cols).map(|k| 6.0 * input(k + 409)).collect::<Vec<_>>())?;
                let run = |split: bool| -> Result<(Vec<f32>, Vec<f32>), String> {
                    // The output is placed first: an upload between the two launches would end the recorded piece.
                    let out = poisoned(b, rows)?;
                    let mut normed = b.vector(cols)?;
                    b.fast_norm_into(1, cols, false, 1e-6, false, &x, None, &mut normed)?;
                    if split { b.read_vector_values(&normed)?; }
                    b.project_staged_one_panels_into(rows, cols, rows, &normed, packed, &mut out.resident_range(GUARD..GUARD+rows)?)?;
                    let back = b.read_vector_values(&out)?;
                    if back[..GUARD].iter().chain(&back[GUARD+rows..]).any(|w| w.to_bits() != POISON) {
                        return Err(format!("decode panels after a norm changed a guard word for {rows} rows"));
                    }
                    Ok((b.read_vector_values(&normed)?, back[GUARD..GUARD+rows].to_vec()))
                };
                let ((normed, joined), (_, split)) = (run(false)?, run(true)?);
                let original = poisoned(b, rows)?;
                b.project_staged_few_into(rows,cols,1,rows,&b.upload_vector(&normed)?,raw,&mut original.resident_range(GUARD..GUARD+rows)?)?;
                let expect = b.read_vector_values(&original)?;
                if joined.iter().zip(&split).zip(&expect[GUARD..GUARD+rows]).any(|((a, s), e)| a.to_bits() != s.to_bits() || a.to_bits() != e.to_bits()) {
                    return Err(format!("decode panels after a norm differ for {rows} rows"));
                }
            }
            println!("FEW CHECK PANELS AFTER NORM cases=2 dependent and ordered launches, exact rows and guards");
            {
                use spark_gpu_probe::packed_product::layout;
                // Chunk counts 4 (under the ring depth of 6) and 10 (a full ring and a tail).
                for (rows,cols,experts) in [(128usize,256usize,2usize),(128,640,2)] {
                let shape = layout(rows,cols,false);
                let mut source = vec![0u8; experts * shape.bytes];
                let mut alternate = Vec::new();
                for (expert,unit) in source.chunks_exact_mut(shape.bytes).enumerate() {
                    for (i,v) in unit[..shape.table].iter_mut().enumerate() {
                        *v = (i.wrapping_mul(37) + (i>>5) + expert*71) as u8;
                    }
                    for (i,word) in unit[shape.table..].as_chunks_mut::<2>().0.iter_mut().enumerate() {
                        let v = spark_gpu_probe::bf16::rounded(input(i+expert+1));
                        word.copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
                    }
                    alternate.extend(spark_gpu_probe::decode_panels::pack_expert(unit,rows,cols));
                }
                let upload = |bytes:&[u8]| {
                    let mut vec = b.vector(bytes.len()/4)?;
                    b.write_vector_bytes(&mut vec,bytes)?;
                    vec.into_resident_shared()
                };
                let raw = upload(&source)?;
                let panels = upload(&alternate)?;
                for route in [[1.0,0.0],[1.0,1.0]] {
                    let ids = b.upload_vector(&route)?;
                    for gate in [false,true] {
                        let input_len = if gate {cols} else {cols*2};
                        let x = b.upload_vector(&(0..input_len).map(input).collect::<Vec<_>>())?;
                        let caps = gate.then_some([3.0,3.0,1.0/3.0,1.0/3.0]);
                        let run = |packed| -> Result<Vec<u32>,String> {
                            let out = poisoned(b,2*rows)?;
                            let mut slice = out.resident_range(GUARD..GUARD+2*rows)?;
                            b.project_packed_picked_step_into(rows,cols,false,gate,&x,gate,&raw,experts,&ids,&mut slice,caps,true,packed)?;
                            let words = b.read_vector_values(&out)?.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                            let writes = if gate { rows } else { 2*rows };
                            if words[..GUARD].iter().chain(&words[GUARD+writes..]).any(|&word|word!=POISON) {
                                return Err("picked expert panels changed a guard word".into());
                            }
                            Ok(words)
                        };
                        if run(None)? != run(Some(&panels))? { return Err("picked expert panels changed the product".into()); }
                    }
                }
                }
                println!("FEW CHECK EXPERT PANELS cases=8 ordinary and duplicate picks, exact products and guards");
            }
            {
                // In-place tagged NVFP4 MMA-lane panels (zero extra VRAM, panels=None) across M=1,2,3,4,8
                // covering one-shot X and multi-slab X, gated & ungated, plus direct PlainPacked & PlainPackedRows.
                use spark_gpu_probe::packed_product::layout;
                for (rows, cols, count, gate) in [
                    (128usize, 256usize, 1usize, false),
                    (128, 640, 2, true),
                    (128, 1152, 3, false),
                    (128, 1152, 4, true),
                    (128, 1152, 8, false),
                    (128, 2304, 8, true),
                ] {
                    let shape = layout(rows, cols, false);
                    let mut raw_bytes = vec![0u8; 2 * shape.bytes];
                    let mut tagged_bytes = Vec::with_capacity(2 * shape.bytes);
                    for (expert, unit) in raw_bytes.chunks_exact_mut(shape.bytes).enumerate() {
                        for (i, v) in unit[..shape.scales].iter_mut().enumerate() {
                            *v = (i.wrapping_mul(41) + (i >> 4) + expert * 73) as u8;
                        }
                        for (i, v) in unit[shape.scales..shape.table].iter_mut().enumerate() {
                            let mut s = ((i.wrapping_mul(17) + expert * 53) & 0xff) as u8;
                            if (s & 0x7f) == 0x7f { s ^= 1; }
                            *v = s;
                        }
                        for (i, word) in unit[shape.table..].as_chunks_mut::<2>().0.iter_mut().enumerate() {
                            let v = spark_gpu_probe::bf16::rounded(input(i + expert * 19 + 5));
                            word.copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
                        }
                        tagged_bytes.extend(spark_gpu_probe::decode_panels::pack_expert_tagged(unit, rows, cols));
                    }
                    let upload = |bytes: &[u8]| {
                        let mut vec = b.vector(bytes.len() / 4)?;
                        b.write_vector_bytes(&mut vec, bytes)?;
                        vec.into_resident_shared()
                    };
                    let raw = upload(&raw_bytes)?;
                    let tagged = upload(&tagged_bytes)?;
                    let mut route = vec![0.0f32; count];
                    route.push(1.0);
                    let picks = route.len();
                    let ids = b.upload_vector(&route)?.into_resident_shared()?;
                    let x = b.upload_vector(&(0..picks * cols).map(|k| input(k + rows + count * 7)).collect::<Vec<_>>())?.into_resident_shared()?;
                    let caps = gate.then_some([4.0, 25.0, 0.25, 0.04]);
                    let run = |stack: &DeviceVector<'_>| -> Result<Vec<u32>, String> {
                        let out = poisoned(b, picks * rows)?;
                        let mut slice = out.resident_range(GUARD..GUARD + picks * rows)?;
                        b.project_packed_picked_step_into(rows, cols, false, gate, &x, false, stack, 2, &ids, &mut slice, caps, count == 1, None)?;
                        let words = b.read_vector_values(&out)?.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                        let writes = if gate { picks * (rows / 2) } else { picks * rows };
                        if words[..GUARD].iter().chain(&words[GUARD + writes..]).any(|&w| w != POISON) {
                            return Err(format!("in-place NVFP4 panel changed a guard word for {rows}x{cols} count={count}"));
                        }
                        Ok(words)
                    };
                    if run(&tagged)? != run(&raw)? {
                        return Err(format!("in-place NVFP4 panel mismatch for {rows}x{cols} count={count} gate={gate}"));
                    }
                    if count == 1 {
                        let u_raw = raw.resident_range(0..shape.bytes / 4)?;
                        let u_tag = tagged.resident_range(0..shape.bytes / 4)?;
                        let x0 = x.resident_range(0..cols)?;
                        let (o_raw, o_tag) = (poisoned(b, rows)?, poisoned(b, rows)?);
                        b.project_packed_into(rows, cols, false, gate, &x0, &u_raw, &mut o_raw.resident_range(GUARD..GUARD + rows)?)?;
                        b.project_packed_into(rows, cols, false, gate, &x0, &u_tag, &mut o_tag.resident_range(GUARD..GUARD + rows)?)?;
                        if b.read_vector_values(&o_raw)?.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                            != b.read_vector_values(&o_tag)?.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        {
                            return Err(format!("direct PlainPacked in-place panel mismatch for {rows}x{cols}"));
                        }
                        let (r_raw, r_tag) = (poisoned(b, picks * rows)?, poisoned(b, picks * rows)?);
                        b.project_packed_rows_grouped_into(rows, cols, false, gate, rows, &[(&u_raw, picks)], &x, None, &mut r_raw.resident_range(GUARD..GUARD + picks * rows)?, None)?;
                        b.project_packed_rows_grouped_into(rows, cols, false, gate, rows, &[(&u_tag, picks)], &x, None, &mut r_tag.resident_range(GUARD..GUARD + picks * rows)?, None)?;
                        if b.read_vector_values(&r_raw)?.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                            != b.read_vector_values(&r_tag)?.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        {
                            return Err(format!("PlainPackedRows in-place panel mismatch for {rows}x{cols}"));
                        }
                    }
                }
                println!("FEW CHECK IN-PLACE NVFP4 PANELS cases=6 M=1..8 one-shot+multi-slab+direct+rows exact and guards");
            }
            // Sub-48-block 1-warp panel mode (`rows = 1024, cols = 256`, `inputs = 1, 2, 4, 8`) + `FEW_PANEL_RING`
            // (`inputs = 9, 12, 16`, 1-warp single/multi-part + 4-warp + split outputs) bit-exact vs raw.
            {
                let (rows, cols, _, raw, packed) = one_cases.iter().find(|case| case.0 == 1024).ok_or("few-check: no 1-warp panel case")?;
                let (rows, cols) = (*rows, *cols);
                for inputs in [1usize, 2, 4, 8, 9, 12, 16] {
                    let x = b.upload_vector(&(0..inputs * cols).map(|k| input(k + 19 * inputs)).collect::<Vec<_>>())?;
                    let (want, got) = (poisoned(b, inputs * rows)?, poisoned(b, inputs * rows)?);
                    b.project_staged_few_into(rows, cols, inputs, rows, &x, raw, &mut want.resident_range(GUARD..GUARD + inputs * rows)?)?;
                    let expected_kind = if inputs <= 8 { Kind::PlainStagedOne } else { Kind::PlainStagedFew };
                    let before = recorded(expected_kind);
                    b.project_staged_few_into(rows, cols, inputs, rows, &x, packed, &mut got.resident_range(GUARD..GUARD + inputs * rows)?)?;
                    if recorded(expected_kind) != before + 1 { return Err(format!("1-warp panel matrix did not take expected entry for {inputs} rows")); }
                    if b.read_vector_values(&want)?.iter().zip(&b.read_vector_values(&got)?).any(|(a, b)| a.to_bits() != b.to_bits()) {
                        return Err(format!("1-warp panel matrix differs for {inputs} rows"));
                    }
                }
                // Multi-part `FEW_PANEL_RING`: 1-warp (`1024 x 2304`, part=72) and 4-warp split (`3072 x 512`).
                for (rows, cols, splits, raw, packed) in &ring_cases {
                    let (rows, cols) = (*rows, *cols);
                    for inputs in [9usize, 16] {
                        let x = b.upload_vector(&(0..inputs * cols).map(|k| input(k + rows + 7 * inputs)).collect::<Vec<_>>())?;
                        let want = poisoned(b, inputs * rows)?;
                        b.project_staged_few_into(rows, cols, inputs, rows, &x, raw, &mut want.resident_range(GUARD..GUARD + inputs * rows)?)?;
                        let want_v = b.read_vector_values(&want)?;
                        if splits.len() == 1 {
                            let got = poisoned(b, inputs * rows)?;
                            b.project_staged_few_into(rows, cols, inputs, rows, &x, packed, &mut got.resident_range(GUARD..GUARD + inputs * rows)?)?;
                            if want_v.iter().zip(&b.read_vector_values(&got)?).any(|(a, b)| a.to_bits() != b.to_bits()) {
                                return Err(format!("FEW_PANEL_RING {rows}x{cols} m={inputs} differs"));
                            }
                        } else {
                            let bufs = splits.iter().map(|&w| poisoned(b, inputs * w)).collect::<Result<Vec<_>, _>>()?;
                            let mut slices = bufs.iter().zip(*splits).map(|(v, &w)| Ok((v.resident_range(GUARD..GUARD + inputs * w)?, w))).collect::<Result<Vec<_>, String>>()?;
                            let mut refs = slices.iter_mut().map(|(v, w)| (v, *w)).collect::<Vec<_>>();
                            b.project_staged_few_split_into(rows, cols, inputs, &x, packed, &mut refs)?;
                            let mut off = 0usize;
                            for (buf, &w) in bufs.iter().zip(*splits) {
                                let back = b.read_vector_values(buf)?;
                                if back[..GUARD].iter().chain(&back[GUARD + inputs * w..]).any(|v| v.to_bits() != POISON) {
                                    return Err(format!("FEW_PANEL_RING split {rows}x{cols} m={inputs} changed guard"));
                                }
                                for t in 0..inputs {
                                    for r in 0..w {
                                        if back[GUARD + t * w + r].to_bits() != want_v[GUARD + t * rows + off + r].to_bits() {
                                            return Err(format!("FEW_PANEL_RING split {rows}x{cols} m={inputs} t={t} r={r} differs"));
                                        }
                                    }
                                }
                                off += w;
                            }
                        }
                    }
                }
                println!("FEW CHECK 1-WARP + PANEL RING cases=11 inputs=1..16 1-warp/4-warp/multi-part/split bit-exact and guards unchanged");
            }
            // Grouped 3-way dual-input narrow launch (`project_staged_few_group_pair_into`) vs individual `project_staged_few_pair_into`.
            {
                for &(rows, cols, inputs, round_out) in &[(64usize, 4608usize, 2usize, true), (64, 4608, 4, true), (64, 4608, 8, false)] {
                    let half = cols / 2;
                    let (mut ws, mut x1s, mut x2s) = (Vec::new(), Vec::new(), Vec::new());
                    let mut want = Vec::with_capacity(3 * inputs * rows);
                    for g in 0..3usize {
                        let w_host: Vec<u16> = (0..rows * cols).map(|i| (((i * 31 + 19 + g * 53) % 241) as u16) << 6 | 0x3c00).collect();
                        let x1_host: Vec<f32> = (0..inputs * half).map(|i| input(i + g * 101 + 17)).collect();
                        let x2_host: Vec<f32> = (0..inputs * half).map(|i| input(i + g * 137 + 31)).collect();
                        let (w_dev, x1_dev, x2_dev) = (matrix(b, &w_host, 0)?, b.upload_vector(&x1_host)?.into_resident_shared()?, b.upload_vector(&x2_host)?.into_resident_shared()?);
                        let mut one = b.vector(inputs * rows)?;
                        b.project_staged_few_pair_into(rows, cols, inputs, round_out, &x1_dev, &x2_dev, &w_dev, &mut one)?;
                        want.extend(b.read_vector_values(&one)?);
                        ws.push(w_dev); x1s.push(x1_dev); x2s.push(x2_dev);
                    }
                    let items: Vec<_> = (0..3).map(|g| (&ws[g], &x1s[g], &x2s[g])).collect();
                    let got_buf = poisoned(b, 3 * inputs * rows)?;
                    b.project_staged_few_group_pair_into(rows, cols, inputs, round_out, &items,
                        &mut got_buf.resident_range(GUARD..GUARD + 3 * inputs * rows)?)?;
                    let back = b.read_vector_values(&got_buf)?;
                    if back[..GUARD].iter().chain(&back[GUARD + want.len()..]).any(|w| w.to_bits() != POISON) {
                        return Err(format!("few_group_pair r={rows} m={inputs} changed a guard word"));
                    }
                    if back[GUARD..GUARD + want.len()].iter().zip(&want).any(|(a, b)| a.to_bits() != b.to_bits()) {
                        return Err(format!("few_group_pair r={rows} c={cols} m={inputs} rnd={round_out} differs"));
                    }
                }
                println!("FEW CHECK GROUP PAIR cases=3 inputs=2,4,8 bit-exact and guards unchanged");
            }
            crate::extra::run(b, &recorded)?;
            drop(queue);
            // 3-way fused 4-warp cooperative Vade-BF11-BP panel group (`Q+K+V`), single-row dual-input `staged_narrow`
            // (with PDL after `fast_norm_into`), and LUT-free SM121a register NVFP4 (`NVFP4_FACTOR_MAGIC`).
            {
                let (rows, cols) = (768usize, 512usize);
                let mut raw3 = Vec::new();
                let mut pan3 = Vec::new();
                for g in 0..3usize {
                    let vals: Vec<u16> = (0..rows * cols).map(|i| {
                        let k = i + g * 19937;
                        if (k / 16) % 2 == 0 { 0x3c00 | ((k % 1024) as u16) | (((k % 3 == 0) as u16) << 15) } else { weight(k) }
                    }).collect();
                    let raw = matrix(b, &vals, 0)?;
                    let pan = matrix(b, &vals, 0)?;
                    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                    b.register_bf16_panels(&pan, rows, cols, &bytes)?;
                    raw3.push(raw);
                    pan3.push(pan);
                }
                let queue = b.resident_queue()?;
                b.enable_resident_launches()?;
                let xs: Vec<_> = (0..3usize).map(|g| b.upload_vector(&(0..cols).map(|k| input(k + 503 * (g + 1))).collect::<Vec<_>>())).collect::<Result<Vec<_>, _>>()?;
                let mut want3 = Vec::new();
                for g in 0..3usize {
                    let out = poisoned(b, rows)?;
                    b.project_staged_few_into(rows, cols, 1, rows, &xs[g], &raw3[g], &mut out.resident_range(GUARD..GUARD + rows)?)?;
                    want3.push(b.read_vector_values(&out)?);
                }
                let got3: Vec<_> = (0..3).map(|_| poisoned(b, rows)).collect::<Result<Vec<_>, _>>()?;
                let mut normed = b.vector(cols)?;
                b.fast_norm_into(1, cols, false, 1e-6, false, &xs[0], None, &mut normed)?;
                let x0_ref = b.read_vector_values(&normed)?;
                let want0 = poisoned(b, rows)?;
                b.project_staged_few_into(rows, cols, 1, rows, &b.upload_vector(&x0_ref)?, &raw3[0], &mut want0.resident_range(GUARD..GUARD + rows)?)?;
                want3[0] = b.read_vector_values(&want0)?;
                let before = recorded(Kind::PlainStagedOne);
                b.fast_norm_into(1, cols, false, 1e-6, false, &xs[0], None, &mut normed)?;
                b.project_staged_few_into(rows, cols, 1, rows, &normed, &pan3[0], &mut got3[0].resident_range(GUARD..GUARD + rows)?)?;
                b.project_staged_few_into(rows, cols, 1, rows, &xs[1], &pan3[1], &mut got3[1].resident_range(GUARD..GUARD + rows)?)?;
                b.project_staged_few_into(rows, cols, 1, rows, &xs[2], &pan3[2], &mut got3[2].resident_range(GUARD..GUARD + rows)?)?;
                if recorded(Kind::PlainStagedOne) != before + 1 {
                    return Err(format!("3-way coop panel group recorded {} launches instead of 1", recorded(Kind::PlainStagedOne) - before));
                }
                for g in 0..3usize {
                    let got = b.read_vector_values(&got3[g])?;
                    if got.iter().zip(&want3[g]).any(|(a, w)| a.to_bits() != w.to_bits()) {
                        return Err(format!("3-way coop panel group item {g} differs from raw"));
                    }
                }
                // Single-row dual-input `staged_narrow` (single + 3-way group, round_out true/false, PDL after norm).
                for round_out in [false, true] {
                    let (nr, nc) = (64usize, 4608usize);
                    let half = nc / 2;
                    let (mut ws, mut x1s, mut x2s) = (Vec::new(), Vec::new(), Vec::new());
                    let mut want = Vec::with_capacity(3 * nr);
                    for g in 0..3usize {
                        let wh: Vec<u16> = (0..nr * nc).map(|i| (((i * 29 + 13 + g * 47) % 241) as u16) << 6 | 0x3c00).collect();
                        let a_h: Vec<f32> = (0..half).map(|i| input(i + g * 89 + 7)).collect();
                        let b_h: Vec<f32> = (0..half).map(|i| input(i + g * 113 + 19)).collect();
                        let (wd, ad, bd) = (matrix(b, &wh, 0)?, b.upload_vector(&a_h)?.into_resident_shared()?, b.upload_vector(&b_h)?.into_resident_shared()?);
                        let mut ref_out = b.vector(nr)?;
                        b.project_staged_few_pair_into(nr, nc, 1, round_out, &ad, &bd, &wd, &mut ref_out)?;
                        let ref_vals = b.read_vector_values(&ref_out)?;
                        if g == 0 {
                            let one_buf = poisoned(b, nr)?;
                            let mut n1 = b.vector(half)?;
                            b.fast_norm_into(1, half, false, 1e-6, false, &ad, None, &mut n1)?;
                            let n1_ref = b.read_vector_values(&n1)?;
                            let mut one_want = b.vector(nr)?;
                            b.project_staged_few_pair_into(nr, nc, 1, round_out, &b.upload_vector(&n1_ref)?, &bd, &wd, &mut one_want)?;
                            let one_want = b.read_vector_values(&one_want)?;
                            b.fast_norm_into(1, half, false, 1e-6, false, &ad, None, &mut n1)?;
                            b.project_staged_narrow_pair_into(nr, nc, round_out, &n1, &bd, &wd, &mut one_buf.resident_range(GUARD..GUARD + nr)?)?;
                            let got = b.read_vector_values(&one_buf)?;
                            if got[..GUARD].iter().chain(&got[GUARD + nr..]).any(|w| w.to_bits() != POISON)
                                || got[GUARD..GUARD + nr].iter().zip(&one_want).any(|(a, w)| a.to_bits() != w.to_bits())
                            {
                                return Err(format!("narrow_pair PDL rnd={round_out} differs"));
                            }
                        }
                        want.extend(ref_vals);
                        ws.push(wd); x1s.push(ad); x2s.push(bd);
                    }
                    let items: Vec<_> = (0..3).map(|g| (&ws[g], &x1s[g], &x2s[g])).collect();
                    let grp = poisoned(b, 3 * nr)?;
                    b.project_staged_narrow_group_pair_into(nr, nc, round_out, &items, &mut grp.resident_range(GUARD..GUARD + 3 * nr)?)?;
                    let back = b.read_vector_values(&grp)?;
                    if back[..GUARD].iter().chain(&back[GUARD + want.len()..]).any(|w| w.to_bits() != POISON)
                        || back[GUARD..GUARD + want.len()].iter().zip(&want).any(|(a, w)| a.to_bits() != w.to_bits())
                    {
                        return Err(format!("narrow_group_pair rnd={round_out} differs"));
                    }
                }
                // LUT-free SM121a register NVFP4 (`NVFP4_FACTOR_MAGIC_LO/HI`) vs 8 KB LUT reference across M=1,4,8 and `PlainPackedRows`.
                {
                    use spark_gpu_probe::packed_product::{NVFP4_FACTOR_MAGIC_HI, NVFP4_FACTOR_MAGIC_LO, layout};
                    const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
                    let e4m3 = |s: u8| -> f32 {
                        let sign = if (s & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
                        let exp = (s >> 3) & 15;
                        let mant = s & 7;
                        let mag = if exp == 0 { (mant as f32) * 2.0f32.powi(-9) } else { (1.0 + mant as f32 / 8.0) * 2.0f32.powi(exp as i32 - 7) };
                        sign * mag
                    };
                    for (rows, cols, count, gate) in [(128usize, 640usize, 1usize, false), (128, 1152, 4, true), (128, 2304, 8, true)] {
                        let shape = layout(rows, cols, false);
                        let mut lut_bytes = vec![0u8; 2 * shape.bytes];
                        let mut reg_bytes = Vec::with_capacity(2 * shape.bytes);
                        for (expert, unit) in lut_bytes.chunks_exact_mut(shape.bytes).enumerate() {
                            let factor = if expert == 0 { 0.022781372f32 } else { 0.041931152f32 };
                            for (i, v) in unit[..shape.scales].iter_mut().enumerate() { *v = (i.wrapping_mul(37) + (i >> 3) + expert * 61) as u8; }
                            for (i, v) in unit[shape.scales..shape.table].iter_mut().enumerate() {
                                let mut s = ((i.wrapping_mul(19) + expert * 43) & 0xff) as u8;
                                if (s & 0x7f) == 0x7f { s ^= 1; }
                                *v = s;
                            }
                            for s in 0..256usize {
                                let sc = if (s & 0x7f) == 0x7f { 0.0f32 } else { e4m3(s as u8) };
                                for c in 0..16usize {
                                    let val = if (c & 8) != 0 { -E2M1[c & 7] } else { E2M1[c & 7] };
                                    let w = spark_gpu_probe::bf16::rounded((val * sc) * factor);
                                    let off = shape.table + s * 32 + c * 2;
                                    unit[off..off + 2].copy_from_slice(&((w.to_bits() >> 16) as u16).to_le_bytes());
                                }
                            }
                            let mut tagged = spark_gpu_probe::decode_panels::pack_expert_tagged(unit, rows, cols);
                            let tag = shape.table + 0xff * 32;
                            tagged[tag..tag + 4].copy_from_slice(&NVFP4_FACTOR_MAGIC_LO.to_le_bytes());
                            tagged[tag + 4..tag + 8].copy_from_slice(&NVFP4_FACTOR_MAGIC_HI.to_le_bytes());
                            tagged[tag + 8..tag + 12].copy_from_slice(&factor.to_bits().to_le_bytes());
                            reg_bytes.extend(tagged);
                        }
                        let upload = |bytes: &[u8]| {
                            let mut v = b.vector(bytes.len() / 4)?;
                            b.write_vector_bytes(&mut v, bytes)?;
                            v.into_resident_shared()
                        };
                        let (raw_lut, reg_pan) = (upload(&lut_bytes)?, upload(&reg_bytes)?);
                        let mut route = vec![0.0f32; count];
                        route.push(1.0);
                        let picks = route.len();
                        let ids = b.upload_vector(&route)?.into_resident_shared()?;
                        let x = b.upload_vector(&(0..picks * cols).map(|k| input(k + rows + count * 13)).collect::<Vec<_>>())?.into_resident_shared()?;
                        let caps = gate.then_some([4.0, 25.0, 0.25, 0.04]);
                        let run_picked = |stack: &DeviceVector<'_>| -> Result<Vec<u32>, String> {
                            let out = poisoned(b, picks * rows)?;
                            let mut slice = out.resident_range(GUARD..GUARD + picks * rows)?;
                            b.project_packed_picked_step_into(rows, cols, false, gate, &x, false, stack, 2, &ids, &mut slice, caps, count == 1, None)?;
                            let words: Vec<u32> = b.read_vector_values(&out)?.into_iter().map(f32::to_bits).collect();
                            let writes = if gate { picks * (rows / 2) } else { picks * rows };
                            if words[..GUARD].iter().chain(&words[GUARD + writes..]).any(|&w| w != POISON) {
                                return Err(format!("reg NVFP4 guard changed {rows}x{cols} count={count}"));
                            }
                            Ok(words)
                        };
                        if run_picked(&reg_pan)? != run_picked(&raw_lut)? {
                            return Err(format!("reg NVFP4 picked mismatch {rows}x{cols} count={count} gate={gate}"));
                        }
                        if count == 1 {
                            let (u_lut, u_reg) = (raw_lut.resident_range(0..shape.bytes / 4)?, reg_pan.resident_range(0..shape.bytes / 4)?);
                            let (o_lut, o_reg) = (poisoned(b, picks * rows)?, poisoned(b, picks * rows)?);
                            b.project_packed_rows_grouped_into(rows, cols, false, gate, rows, &[(&u_lut, picks)], &x, None, &mut o_lut.resident_range(GUARD..GUARD + picks * rows)?, None)?;
                            b.project_packed_rows_grouped_into(rows, cols, false, gate, rows, &[(&u_reg, picks)], &x, None, &mut o_reg.resident_range(GUARD..GUARD + picks * rows)?, None)?;
                            if b.read_vector_values(&o_lut)?.iter().map(|v| v.to_bits()).ne(b.read_vector_values(&o_reg)?.iter().map(|v| v.to_bits())) {
                                return Err(format!("reg NVFP4 PlainPackedRows mismatch {rows}x{cols}"));
                            }
                        }
                    }
                }
                drop(queue);
                println!("FEW CHECK COOP-TRIPLE + NARROW-PAIR + REG-NVFP4 PASS");
            }
            println!("FEW CHECK GATE BLEND cases=18 one row, and 2 to 16 rows raw and with panels, exact intermediate and blended words, guards unchanged");
            println!("FEW CHECK PASS");
            Ok(())
        })
    }
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = check::run() { eprintln!("FAIL {e}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("few-check requires Linux CUDA"); std::process::exit(1); }
}
