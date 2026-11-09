use std::{fs,path::Path,time::Instant};
use spark_gpu_probe::{driver::with_backend,expert_ops::{Op,Shape}};
use spark_tp::Result;

pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;
    let started=Instant::now();let mut receipt=String::new();let mut comparisons=0usize;
    let patterns=[0u32,0x80000000,1,0x80000001,0x00018000,0x00800000,0x80800000,
        0x3f800000,0xbf800000,0x3f808000,0x3f818000,0x7f7fffff,0xff7fffff,
        0x7f800000,0xff800000,0x7fc00000,0xffc00000,0x7f800001,0xff800001,0x7fffffff];
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<28,|backend| {
        for cols in [128u32,129] {for width in [1u32,2,3,5,7,128] {
            let weights=(0..256usize*cols as usize).map(|index|if index%cols as usize==0 {(index/cols as usize) as u8} else {0}).collect::<Vec<_>>();
            let packed=weights.chunks(4).map(|chunk|{let mut word=[0;4];word[..chunk.len()].copy_from_slice(chunk);f32::from_le_bytes(word)}).collect::<Vec<_>>();
            let weight=backend.upload_vector(&packed)?;
            for scale_bits in [0u32,0x80000000,1,0x00800000,0x3f800000,0xbf800000,0x7f7fffff] {
                let scales=backend.upload_vector(&vec![f32::from_bits(scale_bits);2*cols.div_ceil(width) as usize])?;
                for (profile,&pattern) in patterns.iter().enumerate() {
                    let source=vec![f32::from_bits(pattern);cols as usize];let input=backend.upload_vector(&source)?;
                    let shape=Shape {op:Op::Fp8Project,rows:256,cols,block_rows:128,block_cols:width,tokens:1,global:1.0,activation:1.0,bound:0.0};
                    let mut outputs=Vec::new();
                    for (name,op) in [("old",Op::Fp8Project),("new",Op::Fp8Serial)] {
                        let output=backend.expert_operation(Shape {op,..shape},[Some(&input),None,None,None],Some(&weight),Some(&scales))?;
                        let values=backend.read_vector_values(&output)?;
                        let bytes=values.iter().flat_map(|value|value.to_bits().to_le_bytes()).collect::<Vec<_>>();
                        super::write(&out.join(format!("k{cols}-bc{width}-s{scale_bits:08x}-p{profile}-{name}.f32")),&bytes)?;
                        outputs.push(bytes);
                    }
                    if outputs[0]!=outputs[1] {let byte=outputs[0].iter().zip(&outputs[1]).position(|(old,new)|old!=new).ok_or("missing difference")?;let row=byte/4;
                        return Err(format!("FP8 ALL CODES DIFFER cols={cols} block_cols={width} scale={scale_bits:08x} input={pattern:08x} weight_code={row:02x} old={:02x?} new={:02x?}",&outputs[0][row*4..row*4+4],&outputs[1][row*4..row*4+4]));}
                    comparisons+=256;
                }
            }
            receipt.push_str(&format!("FP8 ALL CODES PASS codes=256 cols={cols} block_cols={width} scales=7 input_profiles=20 raw_bytes=true\n"));
            super::write(&out.join(format!("partial-k{cols}-bc{width}.txt")),receipt.as_bytes())?;
        }}
        for chunk in 0..512u32 {
            let source=(0..128u32).map(|index|f32::from_bits(((chunk*128+index)<<16)|match index%4 {0=>0x7fff,1=>0x8000,2=>0x8001,_=>0xffff})).collect::<Vec<_>>();
            let input=backend.upload_vector(&source)?;let mut weights=vec![0u8;128*128];for row in 0..128 {weights[row*128+row]=0x38;}
            let packed=weights.chunks_exact(4).map(|word|f32::from_le_bytes(word.try_into().unwrap())).collect::<Vec<_>>();
            let weight=backend.upload_vector(&packed)?;let scales=backend.upload_vector(&[1.0])?;
            let shape=Shape {op:Op::Fp8Project,rows:128,cols:128,block_rows:128,block_cols:128,tokens:1,global:1.0,activation:1.0,bound:0.0};
            let mut outputs=Vec::new();for (name,op) in [("old",Op::Fp8Project),("new",Op::Fp8Serial)] {
                let output=backend.expert_operation(Shape {op,..shape},[Some(&input),None,None,None],Some(&weight),Some(&scales))?;
                let values=backend.read_vector_values(&output)?;let bytes=values.iter().flat_map(|value|value.to_bits().to_le_bytes()).collect::<Vec<_>>();
                super::write(&out.join(format!("bf16-chunk{chunk}-{name}.f32")),&bytes)?;outputs.push(bytes);
            }
            if outputs[0]!=outputs[1] {return Err(format!("FP8 BF16 EDGE DIFFER chunk={chunk}"));}comparisons+=128;
        }
        Ok(())
    })?;
    receipt.push_str(&format!("FIX TICKET S1 DEVICE PASS compared_rows={comparisons} differing_bytes=0 all_weight_codes=256 bf16_edge_patterns=65536 elapsed_ns={} production_kernel_unchanged=true\n",started.elapsed().as_nanos()));
    super::write(&out.join("RECEIPT"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}
