//! Kernel kit, dense family, premise D1: one plain BF16 matrix of the model on the EXISTING tensor-core BF16 kernels
//! of the production image, in the canonical order (`Backend::matmul_bf16_device`: four fixed column ranges and a
//! balanced merge, BF16 output boundary). The shape is `kv_b_proj` on one rank: 16,384 rows x 512 columns.
//!
//! Part 1, words. A token row ALONE (one row a call) against the same token row in a batch of 2 to 1,040 rows at three
//! placements: every output word must be equal. The public call picks a different kernel by row count, so this is the
//! device proof that those kernels share one word function. Sampled words are also checked against the CPU twin
//! (`tc_twin_bf16`).
//! Part 2, time. The whole public call of today's prompt path (the one-at-a-time row kernel on FP32 weights, the BF16
//! round and the transpose) against the tensor-core call, at 1, 16, 256 and 1,024 rows.
use std::{fs,path::Path,time::Instant};
use spark_tp::Result;
use spark_gpu_probe::{driver::{Backend,DeviceVector,with_backend},prompt_ops::{Op as Row,Shape as RowShape},tc_twin_bf16 as twin};
fn mixed(index:usize,salt:usize)->u32 {
    let mut value=(index as u32).wrapping_add((salt as u32+1).wrapping_mul(0x9e37_79b9));
    value=(value^(value>>16)).wrapping_mul(0x85eb_ca6b);value=(value^(value>>13)).wrapping_mul(0xc2b2_ae35);value^(value>>16)
}
/// One BF16 pattern. Classes by `class`: values near one; a wide exponent spread; subnormals and zeros among small
/// values; signed zeros; plus and minus one in turn (sums that cancel); mantissas of ones.
fn pattern(index:usize,salt:usize,class:usize)->u16 {
    let h=mixed(index,salt);let sign=((h>>16)&0x8000) as u16;let frac=(h&127) as u16;
    match class%6 {
        0=>sign|((124+(h>>8)%6) as u16)<<7|frac,
        1=>sign|((97+(h>>8)%60) as u16)<<7|frac,
        2=>sign|match h&3 {0=>0,1=>frac,_=>((1+(h>>8)%6) as u16)<<7|frac},
        3=>if h&7==0 {sign} else {sign|((120+(h>>8)%12) as u16)<<7|frac},
        4=>if index.is_multiple_of(2) {0x3f80} else {0xbf80},
        _=>sign|((122+(h>>8)%8) as u16)<<7|0x7f,
    }
}
fn value(bits:u16)->f32 {f32::from_bits(u32::from(bits)<<16)}
/// The activation rows: token `t` has class `t % 7`; class 6 is a general FP32 row (it has a low BF16 part), the others are BF16 values.
fn token_row(cols:usize,token:usize)->Vec<f32> {
    (0..cols).map(|c|{let class=token%7;
        if class==6 {let h=mixed(c,token+5000);f32::from_bits((h&0x807f_ffff)|((120+(h>>23)%12)<<23))} else {value(pattern(c,token+3000,class))}}).collect()
}
/// The kernel `matmul_bf16_device` launches for this shape and row count (the branch order of `batch.rs`).
fn entry(rows:usize,cols:usize,m:usize)->&'static str {
    if m.is_multiple_of(256) && rows.is_multiple_of(128) && cols.is_multiple_of(256) {"gemm_bf16_mma_t256n128_exact"}
    else if m<=64 && rows.is_multiple_of(64) && cols.is_multiple_of(256) {if m<=16 && rows<=8192 && cols>=4096 {"gemm_bf16_decode_n32"} else {"gemm_bf16_decode"}}
    else if rows>=4096 && rows.is_multiple_of(64) && cols.is_multiple_of(256) {"verify_bf16_blocked_mma"} else {"verify_bf16_f32_mma"}
}
struct Fixture<'a> {rows:usize,weight:Vec<u16>,matrix:spark_gpu_probe::driver::Matrix<'a>,pack:DeviceVector<'a>,scratch:DeviceVector<'a>}
fn fixture<'a>(backend:&Backend<'a>,rows:usize,cols:usize,most:usize)->Result<Fixture<'a>> {
    let weight:Vec<u16>=(0..rows*cols).map(|i|pattern(i,17,i/cols)).collect();
    let matrix=backend.upload_packed_bf16(rows,cols,1<<20,|offset,bytes|{
        for (i,pair) in bytes.as_chunks_mut::<2>().0.iter_mut().enumerate() {pair.copy_from_slice(&weight[offset/2+i].to_le_bytes());}Ok(())})?;
    // The packed activation scratch of the canonical call: 32 fragments of 4 words a 16 x 16 tile, two planes.
    let pack=backend.vector(most.div_ceil(16)*(cols/16)*32*4)?;
    Ok(Fixture {rows,weight,matrix,pack,scratch:backend.vector(1)?})
}
/// The tensor-core call on these token rows: output [tokens][rows], BF16 values.
fn product<'a>(backend:&Backend<'a>,f:&mut Fixture<'a>,input:&DeviceVector<'_>,m:usize)->Result<DeviceVector<'a>> {
    let mut output=backend.vector(m*f.rows)?;
    backend.matmul_bf16_device(&f.matrix,input,&mut output,&mut f.scratch,&mut f.pack,m,false)?;Ok(output)
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;
    let mut receipt=String::from("DENSE_TC_ROWS host_path=Backend::matmul_bf16_device order=canonical_four_ranges timing=complete_public_call\n");
    with_backend(0,spark_gpu_probe::layer_ptx()?,32<<30,|backend| {
        backend.require_batch_kernels()?;
        const MOST:usize=1040;
        // Part 1: words. The small shape checks every word against the twin; the model's shape samples rows.
        for (rows,cols) in [(256usize,256usize),(16384,512)] {
            let mut f=fixture(backend,rows,cols,MOST)?;
            let inputs:Vec<Vec<f32>>=(0..MOST).map(|t|token_row(cols,t)).collect();
            // Alone: one token row a call.
            let mut alone=Vec::with_capacity(MOST);
            for row in &inputs {let x=backend.upload_vector(row)?;let y=product(backend,&mut f,&x,1)?;alone.push(backend.read_vector_values(&y)?);}
            // The twin, on the alone words.
            let sampled:Vec<usize>=if rows<=256 {(0..rows).collect()} else {vec![0,1,7,8,63,64,127,128,129,4095,4096,8191,8192,rows-2,rows-1]};
            let (mut twin_checked,mut twin_differ,mut zero_sign)=(0usize,0usize,0usize);let mut first=String::new();
            // By activation class (token % 7; class 6 = general FP32 rows, the others are BF16 values): the first box run found the
            // twin's old low-part model wrong on class 6 only; the counts stay in the receipt.
            let mut by_class=[0usize;7];
            for (t,row) in inputs.iter().enumerate() {
                for &r in &sampled {
                    let want=twin::word(twin::canonical(&f.weight[r*cols..(r+1)*cols],row));let have=alone[t][r].to_bits();twin_checked+=1;
                    if have!=want {by_class[t%7]+=1;if (have|want)&0x7fff_ffff==0 {zero_sign+=1;} else {twin_differ+=1;if first.is_empty() {first=format!("token={t} row={r} device={have:08x} twin={want:08x}");}}}
                }
            }
            let line=format!("DENSE TC TWIN rows={rows} cols={cols} tokens={MOST} sampled_rows={} words={twin_checked} differ={twin_differ} zero_sign_only={zero_sign} differ_by_class={by_class:?} first=[{first}]\n",sampled.len());
            eprint!("{line}");receipt.push_str(&line);
            if twin_differ+zero_sign!=0 {return Err(format!("DENSE TC TWIN DIFFERS rows={rows} cols={cols} {first}"));}
            // Batches: the same token rows in a batch must give the same words, at every size and placement.
            let (mut batch_words,mut batch_differ)=(0usize,0usize);let mut first=String::new();
            for m in [2usize,3,15,16,17,32,33,64,65,128,255,256,257,512,768,1024,1040] {
                for placement in 0..3 {
                    let order:Vec<usize>=match placement {0=>(0..m).collect(),1=>(0..m).map(|i|MOST-1-i).collect(),_=>(0..m).map(|i|(i*7+3)%MOST).collect()};
                    let x=backend.upload_vector(&order.iter().flat_map(|&t|inputs[t].iter().copied()).collect::<Vec<_>>())?;
                    let y=backend.read_vector_values(&product(backend,&mut f,&x,m)?)?;
                    let mut differ=0usize;
                    for (slot,&t) in order.iter().enumerate() {for r in 0..rows {
                        if y[slot*rows+r].to_bits()!=alone[t][r].to_bits() {differ+=1;if first.is_empty() {first=format!("m={m} placement={placement} slot={slot} token={t} row={r} batch={:08x} alone={:08x}",y[slot*rows+r].to_bits(),alone[t][r].to_bits());}}
                    }}
                    batch_words+=m*rows;batch_differ+=differ;
                    if placement==0 {let line=format!("DENSE TC BATCH rows={rows} cols={cols} m={m} kernel={} alone_kernel={} differ_forward={differ}\n",entry(rows,cols,m),entry(rows,cols,1));eprint!("{line}");receipt.push_str(&line);}
                }
            }
            let line=format!("DENSE TC ALONE-BATCH rows={rows} cols={cols} alone_words={} batch_words={batch_words} differ={batch_differ} first=[{first}]\n",MOST*rows);
            eprint!("{line}");receipt.push_str(&line);
            super::all_equal(&format!("DENSE TC ALONE-BATCH rows={rows} cols={cols}"),batch_words,batch_differ,&first)?;
            // Today's prompt kernel on the same rows: how many words the new order moves (a report, not a gate).
            let expanded=backend.upload_vector(&f.weight.iter().map(|&w|value(w)).collect::<Vec<_>>())?;
            let m=256usize;let bf16_rows:Vec<usize>=(0..MOST).filter(|t|t%7!=6).take(m).collect();
            let x=backend.upload_vector(&bf16_rows.iter().flat_map(|&t|inputs[t].iter().copied()).collect::<Vec<_>>())?;
            let mut s=RowShape::new(Row::ProjectRows,m).ok_or("prompt rows exceed ABI")?;s.rows=rows as u32;s.cols=cols as u32;
            let old=backend.read_vector_values(backend.round_bf16_lanes(&backend.prompt_operation(s,[Some(&x),None,None,None,None],Some(&expanded))?)?.vector())?;
            let mut moved=0usize;for (slot,&t) in bf16_rows.iter().enumerate() {for r in 0..rows {if old[r*m+slot].to_bits()!=alone[t][r].to_bits() {moved+=1;}}}
            let line=format!("DENSE TC OLD-ORDER rows={rows} cols={cols} m={m} words={} differ_from_today={moved} share={:.4}\n",m*rows,moved as f64/(m*rows) as f64);
            eprint!("{line}");receipt.push_str(&line);
        }
        // Part 2: time of the whole public call at the model's shape.
        let (rows,cols)=(16384usize,512usize);let mut f=fixture(backend,rows,cols,1024)?;
        let expanded=backend.upload_vector(&f.weight.iter().map(|&w|value(w)).collect::<Vec<_>>())?;
        for m in [1usize,16,256,1024] {
            let x=backend.upload_vector(&(0..m).filter(|t|t%7!=6).chain(0..m).take(m).flat_map(|t|token_row(cols,t)).collect::<Vec<_>>())?;
            let median=|run:&mut dyn FnMut()->Result<()>|->Result<f64> {
                let mut times=Vec::new();
                for repeat in 0..12 {backend.synchronize()?;let started=Instant::now();run()?;backend.synchronize()?;if repeat>=3 {times.push(started.elapsed().as_secs_f64()*1e3);}}
                times.sort_by(|a,b|a.partial_cmp(b).unwrap());Ok(times[times.len()/2])
            };
            let before=spark_gpu_probe::prompt_launches::line();
            let today=median(&mut ||{
                let mut s=RowShape::new(Row::ProjectRows,m).ok_or("prompt rows exceed ABI")?;s.rows=rows as u32;s.cols=cols as u32;
                let y=backend.round_bf16_lanes(&backend.prompt_operation(s,[Some(&x),None,None,None,None],Some(&expanded))?)?.into_vector();
                if m>1 {let mut t=RowShape::new(Row::Transpose,1).ok_or("prompt transpose shape")?;t.rows=rows as u32;t.cols=m as u32;drop(backend.prompt_operation(t,[Some(&y),None,None,None,None],None)?);}
                Ok(())})?;
            let kernels:Vec<String>=before.split_whitespace().zip(spark_gpu_probe::prompt_launches::line().split_whitespace()).filter(|(a,b)|a!=b).map(|(_,b)|b.split('=').next().unwrap_or("").to_string()).collect();
            let tensor=median(&mut ||{drop(product(backend,&mut f,&x,m)?);Ok(())})?;
            // Bus floor: today's call reads 4 bytes a weight, the tensor-core call 2; both write 4 bytes a word.
            let floor=|weight_bytes:usize|(rows*cols*weight_bytes+m*cols*4+m*rows*4) as f64/235e9*1e3;
            let line=format!("DENSE TC TIME rows={rows} cols={cols} m={m} today_ms={today:.4} tensor_core_ms={tensor:.4} ratio={:.3} today_kernels={} tc_kernel={} today_bus_floor_ms={:.4} tc_bus_floor_ms={:.4} tc_times_floor={:.2}\n",
                today/tensor,kernels.join("+"),entry(rows,cols,m),floor(4),floor(2),tensor/floor(2));
            eprint!("{line}");receipt.push_str(&line);
        }
        Ok(())
    })?;
    receipt.push_str("DENSE_TC_ROWS COMPLETE\n");
    fs::write(out.join("DENSE_TC_ROWS"),receipt).map_err(|error|error.to_string())
}
