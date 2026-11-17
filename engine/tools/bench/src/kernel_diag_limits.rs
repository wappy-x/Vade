//! Cost: short one-device calibration and replay fixtures. No served path changes.
//! Gain: consumed, checked instruction and read/write rates beside prompt kernels.
#![forbid(unsafe_code)]
use super::{Gpu, Result, ptr, sha256};
use std::time::Instant;
fn median(mut v:Vec<f64>)->f64 {v.sort_by(f64::total_cmp);v[v.len()/2]}
fn tensor_ptx(kind:&str,chains:usize)->String {
 let (k,code)=match kind {"e4m3"=>(32,"0x38383838"),"bf16"=>(16,"0x3f803f80"),_=>(64,"0x22222222")};
 let mut s=format!(".version 8.8\n.target sm_121a\n.address_size 64\n.visible .entry diag_mma(.param .u64 out,.param .u32 loops,.param .u32 seed) {{\n.reg .pred p; .reg .b32 a<4>,b<2>,i,n,t,scale,salt; .reg .f32 c<{}>; .reg .b64 o,w;\nld.param.u64 o,[out];ld.param.u32 n,[loops];ld.param.u32 salt,[seed];\n",chains*4);
 for j in 0..4 {s+=&format!("mov.b32 a{j},{code};\n");}for j in 0..2{s+=&format!("mov.b32 b{j},{code};\n");}
 for j in 0..4{s+=&format!("xor.b32 a{j},a{j},salt;\n");}for j in 0..2{s+=&format!("xor.b32 b{j},b{j},salt;\n");}
 s+="mov.b32 scale,0x38383838;\n";for j in 0..chains*4{s+=&format!("mov.f32 c{j},0f00000000;\n");}s+="mov.u32 i,0;L:\n";
 for j in 0..chains {let at=j*4;let d=format!("{{c{},c{},c{},c{}}}",at,at+1,at+2,at+3);
  s+=&if kind=="fp4" {format!("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {d},{{a0,a1,a2,a3}},{{b0,b1}},{d},scale,{{0,0}},scale,{{0,0}};\n")}
  else{format!("mma.sync.aligned.m16n8k{k}.row.col.f32.{kind}.{kind}.f32 {d},{{a0,a1,a2,a3}},{{b0,b1}},{d};\n")};
 }
 s+="add.u32 i,i,1;setp.lt.u32 p,i,n;@p bra L;\nmov.u32 t,%ctaid.x;mov.u32 i,%ntid.x;mov.u32 n,%tid.x;mad.lo.u32 t,t,i,n;\n";
 s+=&format!("mul.wide.u32 w,t,{};add.u64 o,o,w;\n",chains*16);
 for j in 0..chains*4{s+=&format!("st.global.f32 [o+{}],c{j};\n",j*4);}s+="ret;}\n";s
}
fn saturation(g:&mut Gpu)->Result<()> {
 for kind in ["e4m3","fp4","bf16"] {for chains in [1usize,4,8] {
  let ptx=tensor_ptx(kind,chains);let module=g.module(ptx.as_bytes())?;let f=g.function(module,"diag_mma")?;
  for blocks in [24u32,48,96,192,384] {let threads=256u32;let loops=4096u32;let n=blocks as usize*threads as usize*chains*4;let mut out=g.upload(&vec![0xa5;n*4])?;
   let mut iters=loops;let mut seed=0u32;
   if blocks==24 {
    seed=match kind {"e4m3"=>0x08080808,"bf16"=>0x00800080,_=>0x11111111};
    g.launch(f,[blocks,1,1],threads,0,&mut [ptr(&mut out),ptr(&mut iters),ptr(&mut seed)])?;g.sync()?;
    let k=match kind {"e4m3"=>32.0,"bf16"=>16.0,_=>64.0};let factor=if kind=="fp4"{2.25}else{0.25};let want=(k*loops as f32*factor).to_bits();
    let control=g.read(out,n*4)?;if !control.chunks_exact(4).all(|b|u32::from_le_bytes(b.try_into().unwrap())==want){return Err(format!("DIAG SAT changed-input control {kind} chains={chains}"));}
    println!("DIAG SAT CONTROL kind={kind} chains={chains} changed_words={n} check=PASS");seed=0;
   }
   g.launch(f,[blocks,1,1],threads,0,&mut [ptr(&mut out),ptr(&mut iters),ptr(&mut seed)])?;g.sync()?;
   let mut times=vec![];for _ in 0..7 {let t=Instant::now();g.launch(f,[blocks,1,1],threads,0,&mut [ptr(&mut out),ptr(&mut iters),ptr(&mut seed)])?;g.sync()?;times.push(t.elapsed().as_secs_f64());}
   let k=match kind {"e4m3"=>32usize,"bf16"=>16,_=>64};let want=(k as f32*loops as f32).to_bits();let data=g.read(out,n*4)?;
   let mut checksum=0u64;for (i,b) in data.chunks_exact(4).enumerate(){let w=u32::from_le_bytes(b.try_into().unwrap());if w!=want{return Err(format!("DIAG SAT checksum {kind} chains={chains} index={i} have={w:08x} want={want:08x}"));}checksum=checksum.wrapping_add(w as u64);}
   let sec=median(times);let ops=blocks as f64*8.0*chains as f64*loops as f64*2.0*16.0*8.0*k as f64;
   println!("DIAG SAT kind={kind} chains={chains} blocks={blocks} threads={threads} loops={loops} ms={:.9} TFLOPs={:.9} checked_words={n} checksum={checksum} check=PASS",sec*1e3,ops/sec/1e12);g.clear()?;
  }
 }}Ok(())
}
const MOVE:&str=r#".version 8.8
.target sm_121
.address_size 64
.visible .entry diag_move(.param .u64 input,.param .u64 output,.param .u64 bytes,.param .u32 mode) {
 .reg .pred p,q; .reg .b16 h<4>; .reg .b32 r<16>; .reg .b64 d<16>;
 ld.param.u64 d0,[input];ld.param.u64 d1,[output];ld.param.u64 d2,[bytes];ld.param.u32 r8,[mode];
 mov.u32 r0,%ctaid.x;mov.u32 r1,%ntid.x;mov.u32 r2,%tid.x;mov.u32 r3,%nctaid.x;
 mad.lo.u32 r4,r0,r1,r2;mul.lo.u32 r5,r3,r1;mul.wide.u32 d3,r4,16;mul.wide.u32 d4,r5,16;
 L:setp.ge.u64 p,d3,d2;@p bra E;add.u64 d5,d0,d3;add.u64 d6,d1,d3;
 ld.global.v4.u32 {r9,r10,r11,r12},[d5];
 setp.eq.u32 q,r8,1;@q bra ROUND;bra STORE;
 ROUND:cvt.rn.bf16.f32 h0,r9;cvt.rn.bf16.f32 h1,r10;cvt.rn.bf16.f32 h2,r11;cvt.rn.bf16.f32 h3,r12;
 cvt.u32.u16 r13,h0;cvt.u32.u16 r14,h1;cvt.u32.u16 r15,h2;cvt.u32.u16 r7,h3;
 shl.b32 r9,r13,16;shl.b32 r10,r14,16;shl.b32 r11,r15,16;shl.b32 r12,r7,16;
 STORE:st.global.v4.u32 [d6],{r9,r10,r11,r12};add.u64 d3,d3,d4;bra L;
 E:ret;
}
"#;
fn memory(g:&mut Gpu)->Result<()> {
 let module=g.module(MOVE.as_bytes())?;let f=g.function(module,"diag_move")?;
 for bytes in [1usize<<20,16<<20,64<<20,256<<20] {for mode in [0u32,1] {
  let value=0x3f800123u32;let input=g.upload(&vec![value;bytes/4].iter().flat_map(|w|w.to_le_bytes()).collect::<Vec<_>>())?;
  let output=g.upload(&vec![0xa5;bytes])?;let mut m=mode;let mut n=bytes as u64;let mut src=input;let mut dst=output;
  let mut times=vec![];for pass in 0..10 {let t=Instant::now();for _ in 0..16 {g.launch(f,[192,1,1],256,0,&mut [ptr(&mut src),ptr(&mut dst),ptr(&mut n),ptr(&mut m)])?;}g.sync()?;if pass>0{times.push(t.elapsed().as_secs_f64()/16.0);}}
  let got=g.read(output,bytes)?;let want=if mode==0{value}else{0x3f800000};let mut checksum=0u64;
  for b in got.chunks_exact(4){let w=u32::from_le_bytes(b.try_into().unwrap());if w!=want{return Err(format!("DIAG memory checksum mode={mode}"));}checksum=checksum.wrapping_add(w as u64);}
  let sec=median(times);println!("DIAG MEMORY mode={mode} bytes_read={bytes} bytes_written={bytes} grid=192 threads=256 ms={:.9} aggregate_GBs={:.9} checksum={checksum} check=PASS",sec*1e3,2.0*bytes as f64/sec/1e9);g.clear()?;
 }}Ok(())
}
pub(crate) fn run(image:&str,tc_image:&str,mode:&str)->Result<()> {
 let primary=std::fs::read(image).map_err(|e|e.to_string())?;let tc=std::fs::read(tc_image).map_err(|e|e.to_string())?;
 println!("DIAG ID primary_sha256={} tc_sha256={} mode={mode}",sha256::digest(&primary),sha256::digest(&tc));
 let mut g=Gpu::new()?;
 if mode=="scalar" || mode=="rates2" {scalar(&mut g)?;}
 if mode=="rates2" {mixed_memory(&mut g)?;}
 if mode=="rest" {let m=g.module(&primary)?;super::steptile::diag_replay(&mut g,m)?;super::readp::diag_rest(&mut g,m)?;super::absorbtc::diag_replay(&mut g,m)?;}
 if mode=="limits" || mode=="all" {saturation(&mut g)?;memory(&mut g)?;}
 if mode=="replay" || mode=="all" {let m=g.module(&primary)?;let t=g.module(&tc)?;super::mxtc::diag_replay(&mut g,m)?;super::fp4cohort::diag_replay(&mut g,t)?;super::mhc::fp8tc::diag_replay(&mut g,m)?;super::readp::diag_replay(&mut g,m)?;}
 else if !["limits","scalar","rates2","rest"].contains(&mode){return Err("diag mode must be limits, replay, or all".into());}
 println!("DIAG COMPLETE mode={mode}");Ok(())
}
#[cfg(test)] mod tests {use super::*;
 #[test]fn every_accumulator_is_stored(){for kind in ["e4m3","fp4","bf16"]{for chains in [1,4,8]{let s=tensor_ptx(kind,chains);assert_eq!(s.matches("st.global.f32").count(),chains*4);assert_eq!(s.matches("mma.sync").count(),chains);}}}
}

/// Consume all timed outputs. A cold sample is preceded by an untimed 128 MiB
/// read/write sweep. Report cold and warm separately; do not infer DRAM counters.
pub(in crate::cuda) fn replay<F:Fn(&Gpu)->Result<()>>(g:&mut Gpu,name:&str,descriptor:&str,out:u64,words:usize,expected:Option<u32>,launch:&F)->Result<()> {
 launch(g)?;g.sync()?;let initial=g.read(out,words*4)?;
 let check=|data:&[u8]|->Result<u64>{let mut sum=0u64;for (i,b) in data.chunks_exact(4).enumerate(){let w=u32::from_le_bytes(b.try_into().unwrap());if let Some(want)=expected {if w!=want{return Err(format!("DIAG REPLAY check {name} index={i} have={w:08x} want={want:08x}"));}}else if !f32::from_bits(w).is_finite(){return Err(format!("DIAG REPLAY nonfinite {name} index={i}"));}sum=sum.wrapping_add(w as u64);}Ok(sum)};
 let want_sum=check(&initial)?;let flush_bytes=64usize<<20;
 let mut src=g.words(&vec![1f32.to_bits();flush_bytes/4])?;let mut dst=g.upload(&vec![0xa5;flush_bytes])?;
 let module=g.module(MOVE.as_bytes())?;let mover=g.function(module,"diag_move")?;let mut bytes=flush_bytes as u64;let mut mode=0u32;
 for cold in [false,true]{let mut times=vec![];for _ in 0..9 {
  if cold{g.launch(mover,[192,1,1],256,0,&mut [ptr(&mut src),ptr(&mut dst),ptr(&mut bytes),ptr(&mut mode)])?;g.sync()?;}
  let t=Instant::now();launch(g)?;g.sync()?;times.push(t.elapsed().as_secs_f64()*1e3);
 }
 let data=g.read(out,words*4)?;let checksum=check(&data)?;if checksum!=want_sum{return Err(format!("DIAG REPLAY unstable checksum {name}"));}
 let ms=median(times);println!("DIAG REPLAY name={name} {descriptor} cold={cold} median_ms={ms:.9} checked_words={words} checksum={checksum} check=PASS");
 }
 let flushed=g.read(dst,flush_bytes)?;if !flushed.chunks_exact(4).all(|b|u32::from_le_bytes(b.try_into().unwrap())==1f32.to_bits()){return Err("DIAG flush checksum".into());}Ok(())
}

fn scalar(g:&mut Gpu)->Result<()> {
 for chains in [1usize,4,16,32] {
  let mut s=format!(".version 8.8\n.target sm_121\n.address_size 64\n.visible .entry diag_scalar(.param .u64 output,.param .u32 loops,.param .f32 scale,.param .f32 offset) {{.reg .pred p; .reg .b32 i,n,t; .reg .b64 o,w; .reg .f32 a,b,c<{}>;ld.param.u64 o,[output];ld.param.u32 n,[loops];ld.param.f32 a,[scale];ld.param.f32 b,[offset];",chains);
  for j in 0..chains{s+=&format!("mov.f32 c{j},0f00000000;");}s+="mov.u32 i,0;L:";
  for j in 0..chains{s+=&format!("mul.rn.f32 c{j},c{j},a;add.rn.f32 c{j},c{j},b;");}
  s+="add.u32 i,i,1;setp.lt.u32 p,i,n;@p bra L;mov.u32 t,%ctaid.x;mov.u32 i,%ntid.x;mov.u32 n,%tid.x;mad.lo.u32 t,t,i,n;";s+=&format!("mul.wide.u32 w,t,{};add.u64 o,o,w;",chains*4);
  for j in 0..chains{s+=&format!("st.global.f32 [o+{}],c{j};",j*4);}s+="ret;}";
  let m=g.module(s.as_bytes())?;let f=g.function(m,"diag_scalar")?;
  for blocks in [48u32,192,384]{let loops=4096u32;let words=blocks as usize*256*chains;let mut out=g.upload(&vec![0xa5;words*4])?;let(mut a,mut b,mut n)=(1f32,1f32,loops);
   g.launch(f,[blocks,1,1],256,0,&mut [ptr(&mut out),ptr(&mut n),ptr(&mut a),ptr(&mut b)])?;g.sync()?;
   let mut ts=vec![];for _ in 0..7{let t=Instant::now();g.launch(f,[blocks,1,1],256,0,&mut [ptr(&mut out),ptr(&mut n),ptr(&mut a),ptr(&mut b)])?;g.sync()?;ts.push(t.elapsed().as_secs_f64());}
   let v=g.read(out,words*4)?;if !v.chunks_exact(4).all(|w|u32::from_le_bytes(w.try_into().unwrap())==(loops as f32).to_bits()){return Err("DIAG scalar checksum".into());}
   b=2.0;g.launch(f,[blocks,1,1],256,0,&mut [ptr(&mut out),ptr(&mut n),ptr(&mut a),ptr(&mut b)])?;g.sync()?;let changed=g.read(out,words*4)?;
   if !changed.chunks_exact(4).all(|w|u32::from_le_bytes(w.try_into().unwrap())==(2.0*loops as f32).to_bits()){return Err("DIAG scalar changed-input control".into());}
   let sec=median(ts);println!("DIAG SCALAR chains={chains} blocks={blocks} loops={loops} ms={:.9} TFLOPs={:.9} checked_words={words} changed_words={words} check=PASS",sec*1e3,words as f64*loops as f64*2.0/sec/1e12);g.clear()?;
  }
 }Ok(())
}

const MIX:&str=r#".version 8.8
.target sm_121
.address_size 64
.visible .entry diag_mix(.param .u64 input,.param .u64 output,.param .u64 reads,.param .u64 writes) {
 .reg .pred p; .reg .b32 r<16>; .reg .b64 d<16>;
 ld.param.u64 d0,[input];ld.param.u64 d1,[output];ld.param.u64 d2,[reads];ld.param.u64 d8,[writes];
 mov.u32 r0,%ctaid.x;mov.u32 r1,%ntid.x;mov.u32 r2,%tid.x;mov.u32 r3,%nctaid.x;
 mad.lo.u32 r4,r0,r1,r2;mul.lo.u32 r5,r3,r1;mul.wide.u32 d3,r4,16;mul.wide.u32 d4,r5,16;mov.u32 r6,0;
 L:setp.ge.u64 p,d3,d2;@p bra W;add.u64 d5,d0,d3;ld.global.v4.u32 {r9,r10,r11,r12},[d5];
 add.u32 r6,r6,r9;add.u32 r6,r6,r10;add.u32 r6,r6,r11;add.u32 r6,r6,r12;add.u64 d3,d3,d4;bra L;
 W:mul.wide.u32 d3,r4,16;
 S:setp.ge.u64 p,d3,d8;@p bra E;add.u64 d6,d1,d3;st.global.v4.u32 [d6],{r6,r6,r6,r6};add.u64 d3,d3,d4;bra S;
 E:ret;
}
"#;
fn mixed_memory(g:&mut Gpu)->Result<()> {
 let mut cases=vec![];
 let mut read=0usize;let mut write=0usize;let mut calls=0usize;
 for(n,k,m,c)in[(4096usize,4096usize,1024usize,816usize),(2048,8192,1024,272),(6144,4096,256,192),(2048,12288,256,96)]{read+=(n*k*33/32+4*m*k)*c;write+=4*m*n*c;calls+=c;}
 cases.push(("mx_tc_tile",read/calls,write/calls));
 for (name,n,k) in [("fp4_cohort",1024usize,4096usize),("fp4_cohort_t64",2048,2048)]{cases.push((name,287*n*k*9/16+16384*k*9/16,4*16384*n));}
 read=0;write=0;calls=0;
 for(n,k,c)in[(1024usize,4096usize,672usize),(2048,2048,336),(2048,16384,88),(8192,1536,88),(768,4096,88),(256,4096,88)]{read+=(n*k+4*n.div_ceil(128)*k.div_ceil(128)+4*1024*k)*c;write+=4*1024*n*c;calls+=c;}
 cases.push(("fp8_rows_tile3",read/calls,write/calls));
 cases.push(("mhc_boundary_block_no_normalized",85573740,100892672));cases.push(("mhc_boundary_block_normalized",85573740,168001536));
 cases.push(("step_rows_tile",71434240,16777216));cases.push(("prompt_place",50331648,50331648));cases.push(("bf16_lanes_round",11246032,11246032));
 let module=g.module(MIX.as_bytes())?;let f=g.function(module,"diag_mix")?;
 for(name,r,w)in cases {let r=r.div_ceil(16)*16;let w=w.div_ceil(16)*16;let mut input=g.words(&vec![1u32;r/4])?;let mut output=g.upload(&vec![0xa5;w])?;let(mut reads,mut writes)=(r as u64,w as u64);
  let mut samples=vec![];for pass in 0..10 {let t=Instant::now();g.launch(f,[192,1,1],256,0,&mut [ptr(&mut input),ptr(&mut output),ptr(&mut reads),ptr(&mut writes)])?;g.sync()?;if pass>0{samples.push(t.elapsed().as_secs_f64());}}
  let data=g.read(output,w)?;let threads=192usize*256;let mut sum=0u64;
  for(i,b)in data.chunks_exact(16).enumerate(){let worker=i%threads;let n=if worker>=r/16{0}else{(r/16-1-worker)/threads+1};let want=(n*4)as u32;for v in b.chunks_exact(4){let x=u32::from_le_bytes(v.try_into().unwrap());if x!=want{return Err(format!("DIAG MIX checksum {name} worker={worker} have={x} want={want}"));}sum=sum.wrapping_add(x as u64);}}
  let sec=median(samples);println!("DIAG MIX name={name} bytes_read={r} bytes_written={w} ms={:.9} aggregate_GBs={:.9} checksum={sum} check=PASS bytes_basis=ESTIMATE_STREAMING_PAYLOAD",sec*1e3,(r+w)as f64/sec/1e9);g.clear()?;
 }Ok(())
}
