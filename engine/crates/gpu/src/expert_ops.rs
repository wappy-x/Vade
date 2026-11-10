//! Calibrated NVFP4 and weight-only FP8 operations. Host and device share math.
use crate::next_math::{bf16,exp};
#[inline(always)]
pub fn round(x:f32)->f32 {f32::from_bits(bf16(x)<<16)}
#[inline(always)]
pub fn e2m1(n:u8)->f32 {
    let magnitude=u32::from(n&7);
    let bits=if magnitude<2 {magnitude*0x3f00_0000} else {((magnitude/2+126)<<23)|((magnitude&1)<<22)};
    f32::from_bits(bits|(u32::from(n&8)<<28))
}
#[inline(always)]
pub fn e4m3(b:u8)->f32 {
    let e=(b>>3)&15;let f=b&7;
    let v=if e==0 {f as f32/512.0} else if b&127==127 {f32::NAN}
        else {f32::from_bits(((u32::from(e)+120)<<23)|(u32::from(f)<<20))};
    if b&128==0 {v} else {-v}
}
/// Nearest-even finite encoding. Distances are exact for adjacent FP formats.
pub fn nearest_e4m3(x:f32)->u8 {
    let sign=if x.is_sign_negative() {128} else {0};let x=x.abs();let (mut lo,mut hi)=(0u8,126u8);
    while lo<hi {let m=lo+(hi-lo)/2;if e4m3(m)<x {lo=m+1;} else {hi=m;}}
    if lo>0 {let a=x-e4m3(lo-1);let b=e4m3(lo)-x;if a<b || a==b && (lo-1)%2==0 {lo-=1;}}
    sign|lo
}
pub fn nearest_e2m1(x:f32)->u8 {
    let sign=if x.is_sign_negative() {8} else {0};let x=x.abs();if x>=6.0 {return sign|7;}let mut best=0;
    for n in 1..8 {let a=(x-e2m1(best)).abs();let b=(x-e2m1(n)).abs();if b<a || b==a && n%2==0 {best=n;}}
    sign|best
}
#[repr(u32)]
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Op {Quantize,NvProject,Fp8Project,SwiGlu,Fold,QuantizeRows,NvRows,Fp8Rows,Fp8Serial,PlainSerial}
#[repr(C)]
#[derive(Debug,Clone,Copy)]
pub struct Shape {pub op:Op,pub rows:u32,pub cols:u32,pub block_rows:u32,pub block_cols:u32,pub tokens:u32,
    pub global:f32,pub activation:f32,pub bound:f32}
impl Shape {
    pub fn lengths(self)->Option<([usize;4],usize,usize,usize,usize)> {
        let (r,k,br,bc)=(self.rows as usize,self.cols as usize,self.block_rows as usize,self.block_cols as usize);
        if r==0 || k==0 || !self.global.is_finite() || self.global<=0.0 || !self.activation.is_finite()
            || self.activation<=0.0 || !self.bound.is_finite() {return None;}
        let rk=r.checked_mul(k)?;
        let m=self.tokens as usize;if m==0 || m>crate::prompt_ops::MAX_ROWS {return None;}
        let result=match self.op {
            Op::Quantize if k%16==0=>([k,0,0,0],0,0,k/16,17),
            Op::NvProject if k%16==0=>([k/16*17,0,0,0],rk/2,rk/16,r,1),
            Op::Fp8Project|Op::Fp8Serial if br>0 && bc>0=>([k,0,0,0],rk,r.div_ceil(br).checked_mul(k.div_ceil(bc))?.checked_mul(4)?,r,1),
            Op::PlainSerial=>([k,0,0,0],rk.checked_mul(4)?,0,r,1),
            Op::SwiGlu if self.bound>0.0=>([r,r,0,0],0,0,r,1),
            Op::Fold if k<=1024=>([rk,k,k,r],0,0,r,1),
            Op::QuantizeRows if k%16==0=>([m.checked_mul(k)?,0,0,0],0,0,m.checked_mul(k/16)?,17),
            Op::NvRows if k%16==0=>([m.checked_mul(k/16)?.checked_mul(17)?,0,0,0],rk/2,rk/16,r.checked_mul(m)?,1),
            Op::Fp8Rows if br>0 && bc>0=>([m.checked_mul(k)?,0,0,0],rk,r.div_ceil(br).checked_mul(k.div_ceil(bc))?.checked_mul(4)?,r.checked_mul(m)?,1),
            _=>return None,

        };
        if result.0.into_iter().chain([result.1,result.2,result.3.checked_mul(result.4)?]).any(|n|n>u32::MAX as usize || n>isize::MAX as usize/4) {return None;}
        Some(result)
    }
}
/// A complete warp can schedule fewer independent long NVFP4 rows per block.
/// The inactive upper lanes never obtain an output slice.
pub const NV_ROWS_PER_WARP:u32=16;
/// Shape-only device-proof envelope. Other valid shapes keep the original
/// arithmetic and eight active rows. Calibration and ignored metadata fields
/// do not select the layout. All batch jobs have equal rows and columns.
pub const NV_PACKED_COLS:[u32;11]=[16,32,48,64,128,256,512,1024,2048,4096,4112];
#[inline(always)]
pub fn nv_packed_shape(rows:u32,cols:u32,count:u32)->bool {
    matches!(count,1|2|8|16) && (matches!((rows,cols),(1024,4096)|(2048,2048))
        || rows>0 && rows<=257 && NV_PACKED_COLS.contains(&cols)
        || rows==65536 && cols==16 && count==1)
}
#[inline(always)]
pub fn nv_rows_per_warp(rows:u32,cols:u32,count:u32)->u32 {if nv_packed_shape(rows,cols,count) {NV_ROWS_PER_WARP} else {8}}
pub fn output_index(op:Op,thread:usize)->Option<usize> {
    if op==Op::NvProject {let lane=thread%32;if lane>=8 {None} else {Some(thread/32*8+lane)}} else {Some(thread)}
}
#[repr(C)]
#[derive(Clone,Copy)]
pub struct Arguments {pub input:[u64;4],pub weight:u64,pub scales:u64,pub output:u64,pub shape:Shape}
#[repr(C)]
#[derive(Clone,Copy)]
pub struct LaunchArguments {pub single:Arguments,pub batch:u64,pub count:u32}
pub fn projection_batch_index(rows:u32,count:u32,thread:usize)->Option<(usize,usize)> {
    projection_row_index(rows,count,thread,8)
}
pub fn nv_projection_batch_index(rows:u32,cols:u32,count:u32,thread:usize)->Option<(usize,usize)> {
    projection_row_index(rows,count,thread,nv_rows_per_warp(rows,cols,count))
}
fn projection_row_index(rows:u32,count:u32,thread:usize,active:u32)->Option<(usize,usize)> {
    let blocks=rows.div_ceil(active) as usize;let lane=thread%32;
    if blocks==0 || lane>=active as usize {return None;}
    let job=thread/32/blocks;let row=thread/32%blocks*active as usize+lane;
    if job>=count as usize || row>=rows as usize {None} else {Some((job,row))}
}
pub fn serial_projection_index(rows:u32,count:u32,thread:usize)->Option<(usize,usize)> {
    let blocks=rows.div_ceil(32) as usize;if blocks==0 {return None;}
    let job=thread/32/blocks;let row=thread/32%blocks*32+thread%32;
    if job>=count as usize || row>=rows as usize {None} else {Some((job,row))}
}
/// Decode-time block-FP8 rows may run in the shared-memory tile kernel (`fp8_tile`): the same one
/// ascending FP32 chain per output row, with the products made by many threads and staged in
/// shared memory. The tile launch is TILE_THREADS threads per block and TILE_SHARED dynamic shared
/// bytes; one block owns TILE_ROWS consecutive rows of one job.
pub const FP8_TILE_ROWS:u32=16;
pub const FP8_TILE_COLS:u32=128;
pub const FP8_TILE_DEPTH:u32=4;
/// Chunks ahead of the copies at which each row's next line is named to the L2 cache (0 = none).
pub const FP8_TILE_PREFETCH:u32=2;
pub const FP8_TILE_THREADS:u32=32+FP8_TILE_ROWS*FP8_TILE_COLS/16;
pub const FP8_TILE_SHARED:u32=fp8_tile_shared(FP8_TILE_ROWS,FP8_TILE_COLS,FP8_TILE_DEPTH);
/// Dynamic shared bytes of one tile block: raw weight stages, scale stages, raw input stages, three rounded
/// input buffers and two product buffers (row pitch cols+4 words).
pub const fn fp8_tile_shared(rows:u32,cols:u32,depth:u32)->u32 {
    depth*rows*cols+depth*(rows*cols/16)*4+depth*cols*4+3*cols*4+2*rows*(cols+4)*4
}
/// The shape-only launch predicate of the tile kernel. Every other Fp8Serial shape keeps the
/// one-thread-per-row kernel. The device byte sweep covers this envelope; it reads nothing but the shape.
pub const FP8_TILE_MAX_ROWS:u32=16384;
pub const FP8_TILE_MAX_COLS:u32=32768;
/// Jobs in one batch the tile launch is proven for (the device sweep runs batches of 1 to 8 jobs and single launches).
pub const FP8_TILE_MAX_JOBS:u32=8;
pub fn fp8_tile_admits(shape:&Shape)->bool {
    shape.op==Op::Fp8Serial && (1..=FP8_TILE_MAX_ROWS).contains(&shape.rows) && (16..=FP8_TILE_MAX_COLS).contains(&shape.cols) && shape.cols%16==0
        && (1..=1024).contains(&shape.block_rows) && shape.block_cols.is_power_of_two() && (16..=4096).contains(&shape.block_cols)
}
/// The launch shape (blocks, threads, dynamic shared bytes) of these jobs: the shared-memory tile kernel when every
/// job is an admitted block-FP8 decode shape with 16-byte aligned weight and input owners (a property of shape and
/// owners, never of the model), otherwise `grid` blocks of one warp.
#[cfg(not(target_arch="nvptx64"))]
pub fn launch_geometry(jobs:&[Arguments],grid:u32,count:u32)->Result<(u32,u32,u32),String> {
    if (1..=FP8_TILE_MAX_JOBS).contains(&count) && jobs.len()==count as usize
        && jobs.iter().all(|job|fp8_tile_admits(&job.shape) && (job.weight|job.input[0])&15==0) {
        let blocks=fp8_tile_blocks(jobs[0].shape.rows).checked_mul(count).ok_or("expert projection tile grid overflow")?;
        Ok((blocks,FP8_TILE_THREADS,FP8_TILE_SHARED))
    } else {Ok((grid,32,0))}
}
/// Keep the NV stream geometry separate from the block-FP8 geometry and its fallback.
#[cfg(not(target_arch="nvptx64"))]
pub fn batch_launch_geometry(jobs:&[Arguments],grid:u32,count:u32)->Result<(bool,u32,u32,u32),String> {
    let owners=jobs.iter().map(|job|(job.weight,job.scales)).collect::<Vec<_>>();
    if jobs.first().is_some_and(|job|crate::nv_stream::projection_admitted(job.shape,count,&owners)) {
        if nv_tile_shape_proven(jobs[0].shape.rows,jobs[0].shape.cols,count) {
            let blocks=crate::nv_stream::tile_blocks(jobs[0].shape.rows as usize,count as usize).ok_or("NV tile grid overflow")?;
            return Ok((true,blocks,crate::nv_stream::TILE_THREADS,crate::nv_stream::TILE_SHARED));
        }
        let blocks=crate::nv_stream::blocks(jobs[0].shape.rows as usize,count as usize).ok_or("NV stream grid overflow")?;
        Ok((true,blocks,64,0))
    } else {let (blocks,threads,shared)=launch_geometry(jobs,grid,count)?;Ok((false,blocks,threads,shared))}
}
/// New-body shape cases in the retained NV batch sweep; other proven NV shapes keep the stream kernel.
pub fn nv_tile_shape_proven(rows:u32,cols:u32,jobs:u32)->bool {
    nv_packed_shape(rows,cols,jobs) && matches!(rows,1|2|31|32|33|63|64|65|127|128|129|191|255|1024|2048)
        && ((u64::from(rows)*u64::from(cols)*u64::from(jobs)<=40_000_000) || (rows==1024 && cols==4096 && jobs==16))
}
/// Blocks per job of a tile launch.
pub const fn fp8_tile_blocks(rows:u32)->u32 {rows.div_ceil(FP8_TILE_ROWS)}
#[cfg(not(target_arch="nvptx64"))]
impl Arguments {
    pub fn descriptor_words(self)->[u32;24] {
        let mut words=[0;24];
        for (index,address) in self.input.into_iter().chain([self.weight,self.scales,self.output]).enumerate() {
            words[index*2]=address as u32;words[index*2+1]=(address>>32) as u32;
        }
        words[14..23].copy_from_slice(&[self.shape.op as u32,self.shape.rows,self.shape.cols,self.shape.block_rows,self.shape.block_cols,
            self.shape.tokens,self.shape.global.to_bits(),self.shape.activation.to_bits(),self.shape.bound.to_bits()]);words
    }
}
/// One expert's token group inside a whole-layer launch. Units are contiguous
/// and ascending by `first`. Each unit runs the ordinary `unit` arithmetic.
#[repr(C)]
#[derive(Clone,Copy)]
pub struct GroupEntry {pub shape:Shape,pub weight:u64,pub scales:u64,pub first:u32,pub first_row:u32,pub input_words:u32,pub weight_bytes:u32,pub scale_bytes:u32,pub pad:u32}
#[repr(C)]
#[derive(Clone,Copy)]
pub struct GroupArguments {pub table:u64,pub input:u64,pub output:u64,pub lut:u64,pub count:u32,pub units:u32,pub words:u32,pub stride:u32}
/// The entry that owns one thread's unit. The host keeps `first` ascending from zero.
#[inline(always)]
pub fn group_entry(table:&[GroupEntry],unit:u32)->&GroupEntry {
    let (mut low,mut high)=(0,table.len());
    while high-low>1 {let middle=low+(high-low)/2;if table[middle].first<=unit {low=middle;} else {high=middle;}}
    &table[low]
}
/// Token tiles keep a tile's activations inside the first-level cache while every
/// weight row sweeps them. Measured peaks: NVFP4 at 8 tokens, FP8 at 16.
#[inline(always)]
fn token_tile(op:Op)->usize {if op==Op::NvRows {8} else {16}}
/// Order threads by token tile. This moves threads only. Each output keeps its fold.
#[inline(always)]
pub fn group_unit(s:Shape,local:usize)->usize {
    if !matches!(s.op,Op::NvRows|Op::Fp8Rows) {return local;}
    let (m,r,t)=(s.tokens as usize,s.rows as usize,token_tile(s.op));
    if m<=t {return local;}
    let full=m/t;let body=full*r*t;
    let (row,token)=if local<body {((local/t)%r,local/(r*t)*t+local%t)} else {let rest=local-body;let tail=m%t;(rest/tail,full*t+rest%tail)};
    row*m+token
}
/// Words in the decode table: one (low nibble, high nibble) pair per weight byte,
/// then one E4M3 value per byte.
pub const LUT_WORDS:usize=768;
/// The same bits as e2m1 and e4m3, one lookup per byte and no decode branch.
#[cfg(not(target_arch="nvptx64"))]
pub fn decode_lut()->Vec<f32> {
    let mut v=Vec::with_capacity(LUT_WORDS);
    for b in 0..=255u8 {v.push(e2m1(b&15));v.push(e2m1(b>>4));}
    for b in 0..=255u8 {v.push(e4m3(b));}v
}
/// One weight row for one token through the table. Each add keeps its feature
/// order and each product keeps its grouping, so the bits equal `nv_row`.
#[inline(always)]
fn nv_row_table(s:Shape,row:usize,x:&[f32],w:&[u8],scales:&[u8],lut:&[f32;LUT_WORDS])->f32 {
    let k=s.cols as usize;let blocks=k/16;let global=s.global*s.activation;let mut sum=0.0f32;
    let bytes=&w[row*(k/2)..(row+1)*(k/2)];let scale=&scales[row*blocks..(row+1)*blocks];
    for ((a,weights),&code) in x.chunks_exact(17).zip(bytes.chunks_exact(8)).zip(scale) {
        let fold=(lut[512+code as usize]*a[16])*global;
        for pair in 0..8 {
            let at=weights[pair] as usize*2;
            sum+=(lut[at]*a[2*pair])*fold;
            sum+=(lut[at+1]*a[2*pair+1])*fold;
        }
    }round(sum)
}
/// One weight row for one token. `x` holds BF16-rounded lanes, so this skips
/// the round that `fp8_row` applies. Rounding a rounded value changes no bit.
#[inline(always)]
fn fp8_row_table(s:Shape,row:usize,x:&[f32],w:&[u8],scales:&[u8],lut:&[f32;LUT_WORDS])->f32 {
    let k=s.cols as usize;let mut sum=0.0f32;let br=s.block_rows as usize;let bc=s.block_cols as usize;let blocks=k.div_ceil(bc);
    let bytes=&w[row*k..(row+1)*k];
    for (block,(input,weights)) in x.chunks(bc).zip(bytes.chunks(bc)).enumerate() {let at=((row/br)*blocks+block)*4;
        let scale=f32::from_le_bytes(scales[at..at+4].try_into().unwrap());
        for (&value,&code) in input.iter().zip(weights) {sum+=(lut[512+code as usize]*scale)*value;}
    }round(sum)
}
/// One unit of a whole-layer launch.
#[inline(always)]
pub fn group_run(s:Shape,index:usize,x:&[f32],w:&[u8],scales:&[u8],lut:&[f32;LUT_WORDS],out:&mut[f32]) {
    match s.op {
        Op::NvRows=>{let m=s.tokens as usize;let k=s.cols as usize;let token=index%m;let stride=k/16*17;
            out[0]=nv_row_table(s,index/m,&x[token*stride..(token+1)*stride],w,scales,lut);},
        Op::Fp8Rows=>{let m=s.tokens as usize;let k=s.cols as usize;let token=index%m;
            out[0]=fp8_row_table(s,index/m,&x[token*k..(token+1)*k],w,scales,lut);},
        _=>unit(s,index,[x,&[],&[],&[]],w,scales,out),
    }
}
#[inline(always)]
fn nv_row(s:Shape,index:usize,x:&[f32],w:&[u8],scales:&[u8])->f32 {
    let k=s.cols as usize;
    let global=s.global*s.activation;let mut sum=0.0f32;
    for block in 0..k/16 {
        let a=&x[block*17..block*17+17];let fold=(e4m3(scales[index*(k/16)+block])*a[16])*global;
        for pair in 0..8 {
            let packed=w[index*(k/2)+block*8+pair];
            sum+=(e2m1(packed&15)*a[2*pair])*fold;
            sum+=(e2m1(packed>>4)*a[2*pair+1])*fold;
        }
    }round(sum)
}
#[inline(always)]
fn fp8_row(s:Shape,index:usize,x:&[f32],w:&[u8],scales:&[u8])->f32 {
    let k=s.cols as usize;
    let mut sum=0.0f32;let br=s.block_rows as usize;let bc=s.block_cols as usize;let blocks=k.div_ceil(bc);
    for block in 0..blocks {let at=((index/br)*blocks+block)*4;
        let scale=f32::from_le_bytes(scales[at..at+4].try_into().unwrap());
        for j in block*bc..((block+1)*bc).min(k) {sum+=(e4m3(w[index*k+j])*scale)*round(x[j]);}
    }round(sum)
}
/// A single decode-time `Op::Quantize` may run one lane per element (`nv_quantize`, 128-thread blocks of eight 16-value
/// blocks); the retained sweep covers 16..1024 by 16 and eight larger widths. Every other shape keeps the original launch.
pub const QUANTIZE_LANES_THREADS:u32=128;
pub const QUANTIZE_LANES_MAX_COLS:u32=16384;
pub fn quantize_lanes_admitted(shape:&Shape)->bool {
    shape.op==Op::Quantize && shape.tokens==1 && crate::quantize_sweep::single_admitted(shape.cols)
}
/// Quantized blocks contain sixteen exact decoded E2M1 values and an E4M3 scale.
/// Fold follows ascending expert IDs. It rounds each contribution and addition.
pub fn unit(s:Shape,index:usize,x:[&[f32];4],w:&[u8],scales:&[u8],out:&mut[f32]) {
    let k=s.cols as usize;
    match s.op {
        Op::Quantize|Op::QuantizeRows=>{
            let row=&x[0][index*16..index*16+16];let mut maximum=0.0f32;
            for &v in row {maximum=maximum.max(round(v).abs());}
            let scale=nearest_e4m3((maximum/s.activation)/6.0);let block=e4m3(scale);
            for pair in 0..8 {
                let q=|v:f32| {let code=if block==0.0 {0} else {nearest_e2m1((round(v).abs()/s.activation)/block)};code|if v<0.0 {8} else {0}};
                out[2*pair]=e2m1(q(row[2*pair]));out[2*pair+1]=e2m1(q(row[2*pair+1]));
            }out[16]=block;
        },
        Op::NvProject=>{
            let global=s.global*s.activation;let mut sum=0.0f32;
            for block in 0..k/16 {
                let a=&x[0][block*17..block*17+17];let fold=(e4m3(scales[index*(k/16)+block])*a[16])*global;
                for pair in 0..8 {
                    let packed=w[index*(k/2)+block*8+pair];
                    sum+=(e2m1(packed&15)*a[2*pair])*fold;
                    sum+=(e2m1(packed>>4)*a[2*pair+1])*fold;
                }
            }out[0]=round(sum);
        },
        Op::PlainSerial=>{
            let mut sum=0.0f32;
            for (column,bytes) in w[index*k*4..(index+1)*k*4].chunks_exact(4).enumerate() {
                sum+=f32::from_le_bytes([bytes[0],bytes[1],bytes[2],bytes[3]])*x[0][column];
            }out[0]=sum;
        },
        Op::Fp8Project|Op::Fp8Serial=>{
            let mut sum=0.0f32;let br=s.block_rows as usize;let bc=s.block_cols as usize;let blocks=k.div_ceil(bc);
            for block in 0..blocks {let at=((index/br)*blocks+block)*4;
                let scale=f32::from_le_bytes(scales[at..at+4].try_into().unwrap());
                for j in block*bc..((block+1)*bc).min(k) {sum+=(e4m3(w[index*k+j])*scale)*round(x[0][j]);}
            }out[0]=round(sum);
        },
        Op::NvRows=>{
            // Adjacent token lanes read the same immutable weight row. Each
            // owns one accumulator and keeps the scalar feature fold.
            let m=s.tokens as usize;let token=index%m;let row=index/m;let stride=k/16*17;
            out[0]=nv_row(s,row,&x[0][token*stride..(token+1)*stride],w,scales);
        },
        Op::Fp8Rows=>{
            let m=s.tokens as usize;let token=index%m;let row=index/m;
            out[0]=fp8_row(s,row,&x[0][token*k..(token+1)*k],w,scales);
        },
        Op::SwiGlu=>{let g=x[0][index].min(s.bound);let u=x[1][index].clamp(-s.bound,s.bound);
            out[0]=round(round(g*(1.0/(1.0+exp(-g))))*u);},
        Op::Fold=>{
            let mut previous=-1.0f32;let mut sum=0.0;
            for _ in 0..k {let mut chosen=0;let mut id=f32::INFINITY;
                for slot in 0..k {if x[1][slot]>previous && x[1][slot]<id {chosen=slot;id=x[1][slot];}}
                sum=round(sum+round(x[0][chosen*s.rows as usize+index]*x[2][chosen]));previous=id;
            }out[0]=round(sum+x[3][index]);
        },
    }
}
#[cfg(not(target_arch="nvptx64"))]
pub fn reference(s:Shape,x:[&[f32];4],w:&[u8],scales:&[u8])->Result<Vec<f32>,String> {
    let (lengths,wb,sb,units,words)=s.lengths().ok_or("invalid expert operation shape")?;
    if x.iter().zip(lengths).any(|(a,n)|a.len()!=n) || w.len()!=wb || scales.len()!=sb
        || x.iter().flat_map(|v|v.iter()).any(|v|!v.is_finite()) {return Err("expert operation operand extent or value differs".into());}
    if matches!(s.op,Op::Quantize|Op::QuantizeRows) || matches!(s.op,Op::Fp8Project|Op::Fp8Rows|Op::Fp8Serial) {
        if x[0].iter().any(|&v|!round(v).is_finite()) {return Err("expert source overflows BF16".into());}
    }
    if matches!(s.op,Op::Quantize|Op::QuantizeRows) && x[0].chunks_exact(16).any(|b|
        !(b.iter().fold(0.0f32,|a,&v|a.max(round(v).abs()))/s.activation/6.0).is_finite()) {
        return Err("expert block scale exceeds the calibrated range".into());
    }
    if matches!(s.op,Op::NvProject|Op::NvRows) && !(s.global*s.activation).is_finite() {return Err("expert combined global scale is not finite".into());}
    if matches!(s.op,Op::NvProject|Op::NvRows) && (scales.iter().any(|&b|b>=127) || x[0].chunks_exact(17).any(|b|
        b[..16].iter().any(|&v|e2m1(nearest_e2m1(v)).to_bits()!=v.to_bits())
        || b[16].is_sign_negative() || e4m3(nearest_e4m3(b[16])).to_bits()!=b[16].to_bits())) {
        return Err("invalid NVFP4 decoded activation or weight scale".into());
    }
    if s.op==Op::PlainSerial && w.chunks_exact(4).any(|bytes|!f32::from_le_bytes(bytes.try_into().unwrap()).is_finite()) {return Err("invalid plain weight".into());}
    if matches!(s.op,Op::Fp8Project|Op::Fp8Rows|Op::Fp8Serial) && (w.iter().any(|b|b&127==127) || scales.chunks_exact(4).any(|b|
        {let v=f32::from_le_bytes(b.try_into().unwrap());!v.is_finite() || v<=0.0})) {return Err("invalid FP8 block weight or scale".into());}
    if s.op==Op::Fold && (x[1].iter().any(|&v|v<0.0 || v>=16_777_216.0 || v.fract()!=0.0)
        || (0..x[1].len()).any(|i|x[1][i+1..].contains(&x[1][i]))) {return Err("invalid or duplicate expert fold IDs".into());}
    let mut out=vec![0.0;units*words];for (i,o) in out.chunks_exact_mut(words).enumerate() {unit(s,i,x,w,scales,o);}
    if out.iter().any(|v|!v.is_finite()) {return Err("expert operation output is not finite".into());}Ok(out)
}
#[cfg(test)]
mod tests {
    #[test]
    fn batch_descriptor_abi_and_complete_warp_row_owners() {
        use std::mem::{size_of,align_of,offset_of};
        assert_eq!((size_of::<super::Arguments>(),align_of::<super::Arguments>(),offset_of!(super::Arguments,shape)),(96,8,56));
        assert_eq!((size_of::<super::LaunchArguments>(),offset_of!(super::LaunchArguments,batch),offset_of!(super::LaunchArguments,count)),(112,96,104));
        let shape=super::Shape {op:super::Op::NvProject,rows:33,cols:128,block_rows:1,block_cols:16,tokens:1,global:0.75,activation:0.125,bound:0.0};
        let arguments=super::Arguments {input:[0x1234_5678_9abc_def0,4,4,4],weight:0x2345_6789_abcd_ef01,scales:0x3456_789a_bcde_f012,output:0x4567_89ab_cdef_0123,shape};
        let words=arguments.descriptor_words();assert_eq!(&words[..2],&[0x9abc_def0,0x1234_5678]);
        assert_eq!(&words[8..14],&[0xabcd_ef01,0x2345_6789,0xbcde_f012,0x3456_789a,0xcdef_0123,0x4567_89ab]);
        assert_eq!(&words[14..],&[super::Op::NvProject as u32,33,128,1,16,1,0.75f32.to_bits(),0.125f32.to_bits(),0,0]);
        for rows in [1u32,7,8,9,17,33,1024,2048] {for count in [1u32,2,8,16] {
            let mut seen=vec![0;rows as usize*count as usize];
            for thread in 0..rows.div_ceil(8) as usize*count as usize*32 {
                if let Some((job,row))=super::projection_batch_index(rows,count,thread) {assert!(thread%32<8);seen[job*rows as usize+row]+=1;}
            }assert!(seen.iter().all(|&owners|owners==1));
        }}
        assert_eq!(super::projection_batch_index(0,1,0),None);assert_eq!(super::projection_batch_index(8,0,0),None);
        assert_eq!(super::projection_batch_index(8,1,32),None);
    }
    use super::*;
    fn s(op:Op)->Shape {Shape {op,tokens:1,rows:2,cols:16,block_rows:1,block_cols:16,global:1.0,activation:1.0,bound:10.0}}
    fn job(op:Op,rows:u32,cols:u32,br:u32,bc:u32)->Arguments {
        Arguments {input:[0x1000,4,4,4],weight:0x2000,scales:0x3000,output:0x4000,shape:Shape {op,rows,cols,block_rows:br,block_cols:bc,tokens:1,global:1.0,activation:1.0,bound:0.0}}
    }
    fn geometry(jobs:&[Arguments])->(u32,u32,u32) {launch_geometry(jobs,77,jobs.len() as u32).unwrap()}
    #[test]
    fn batch_geometry_preserves_nv_stream_fp8_tile_and_original_fallback() {
        let mut nv=job(Op::NvProject,1024,4096,0,0);nv.scales=512;
        assert_eq!(batch_launch_geometry(&[nv;16],77,16).unwrap(),(true,1024,160,24064));
        for rows in [3,17,256,257] {let other=Arguments {shape:Shape {rows,..nv.shape},..nv};
            assert_eq!(batch_launch_geometry(&[other;2],77,2).unwrap(),(true,rows.div_ceil(64)*2,64,0));}
        let down=Arguments {shape:Shape {rows:2048,cols:2048,..nv.shape},..nv};
        assert_eq!(batch_launch_geometry(&[down;8],77,8).unwrap(),(true,1024,160,24064));
        assert_eq!(batch_launch_geometry(&[down;16],77,16).unwrap(),(true,512,64,0));
        let fp8=job(Op::Fp8Serial,1024,4096,128,128);
        assert_eq!(batch_launch_geometry(&[fp8;2],77,2).unwrap(),(false,128,160,30720));
        // The job count is part of the admission: 8 jobs take the tile, 9 and more keep the original launch.
        assert_eq!(batch_launch_geometry(&[fp8;8],77,8).unwrap(),(false,64*8,160,30720));
        for jobs in [9usize,12,16] {
            let list=vec![fp8;jobs];
            assert_eq!(batch_launch_geometry(&list,77,jobs as u32).unwrap(),(false,77,32,0),"{jobs} jobs");
            assert_eq!(launch_geometry(&list,77,jobs as u32).unwrap(),(77,32,0),"{jobs} jobs");
        }
        assert_eq!(launch_geometry(&[fp8;8],77,8).unwrap(),(64*8,FP8_TILE_THREADS,FP8_TILE_SHARED));
        assert_eq!(FP8_TILE_MAX_JOBS,8);
        let mut outside=nv;outside.shape.rows=258;
        assert_eq!(batch_launch_geometry(&[outside],77,1).unwrap(),(false,77,32,0));
        let mut unaligned=fp8;unaligned.weight+=4;
        assert_eq!(batch_launch_geometry(&[fp8,unaligned],77,2).unwrap(),(false,77,32,0));
    }
    #[test]
    fn fp8_tile_batch_count_keeps_unproven_jobs_on_the_original_launch() {
        let fp8=job(Op::Fp8Serial,17,256,128,128);
        for count in [1,8,9] {
            let jobs=vec![fp8;count as usize];let legacy=count;
            let expected=if count<=FP8_TILE_MAX_JOBS {(2*count,FP8_TILE_THREADS,FP8_TILE_SHARED)} else {(legacy,32,0)};
            assert_eq!(launch_geometry(&jobs,legacy,count).unwrap(),expected);
            // Both the eager and recorded batch paths use this shared decision.
            let (stream,blocks,threads,shared)=batch_launch_geometry(&jobs,legacy,count).unwrap();
            assert!(!stream);assert_eq!((blocks,threads,shared),expected);
        }
        for (jobs,count) in [(1,0),(8,9),(9,8),(2,1)] {
            assert_eq!(launch_geometry(&vec![fp8;jobs],77,count).unwrap(),(77,32,0));
            assert_eq!(batch_launch_geometry(&vec![fp8;jobs],77,count).unwrap(),(false,77,32,0));
        }
    }
    #[test]
    fn tile_geometry_is_chosen_by_shape_and_alignment_only() {
        let old=(77,32,0);
        for (rows,cols,br,bc) in [(1024,4096,128,128),(2048,2048,128,128),(2048,16384,128,128),(8192,1536,128,128),(768,4096,128,128),(1,16,1,16),(17,48,3,16),(16384,32768,1024,4096)] {
            assert_eq!(geometry(&[job(Op::Fp8Serial,rows,cols,br,bc)]),(rows.div_ceil(16),FP8_TILE_THREADS,FP8_TILE_SHARED),"{rows}x{cols}");
            assert_eq!(launch_geometry(&[job(Op::Fp8Serial,rows,cols,br,bc);3],77,3).unwrap(),(rows.div_ceil(16)*3,FP8_TILE_THREADS,FP8_TILE_SHARED));
        }
        for shape in [(Op::Fp8Project,1024,4096,128,128),(Op::PlainSerial,1024,4096,128,128),(Op::NvProject,1024,4096,1,16),(Op::Fp8Rows,1024,4096,128,128),
            (Op::Fp8Serial,1024,4100,128,128),(Op::Fp8Serial,1024,8,128,128),(Op::Fp8Serial,1024,0,128,128),(Op::Fp8Serial,0,4096,128,128),
            (Op::Fp8Serial,1024,4096,128,100),(Op::Fp8Serial,1024,4096,128,8),(Op::Fp8Serial,1024,4096,128,8192),(Op::Fp8Serial,1024,4096,0,128),
            (Op::Fp8Serial,1024,4096,1025,128),(Op::Fp8Serial,16385,4096,128,128),(Op::Fp8Serial,1024,32784,128,128)] {
            assert_eq!(geometry(&[job(shape.0,shape.1,shape.2,shape.3,shape.4)]),old,"{shape:?}");
        }
        let mut weight=job(Op::Fp8Serial,1024,4096,128,128);weight.weight|=8;
        let mut input=job(Op::Fp8Serial,1024,4096,128,128);input.input[0]|=4;
        assert_eq!(geometry(&[weight]),old);assert_eq!(geometry(&[input]),old);
        let good=job(Op::Fp8Serial,1024,4096,128,128);
        assert_eq!(launch_geometry(&[good,weight],77,2).unwrap(),(77,32,0));
        assert_eq!(launch_geometry(&[good,job(Op::Fp8Serial,1024,4100,128,128)],77,2).unwrap(),(77,32,0));
        assert_eq!(launch_geometry(&[],77,0).unwrap(),old);
    }
    #[test]
    fn tile_launch_constants_fit_the_default_shared_limit() {
        assert_eq!(FP8_TILE_THREADS,160);assert_eq!(FP8_TILE_SHARED,30720);
        assert_eq!(fp8_tile_shared(16,128,4),4*16*128+4*128*4+4*128*4+3*128*4+2*16*132*4);
        assert!(FP8_TILE_SHARED<=48*1024 && FP8_TILE_THREADS%32==0);
    }
    #[test]
    fn serial_projection_grid_and_reference_keep_complete_rows() {
        for rows in [1u32,7,8,9,17,31,32,33,1024] {for count in [1u32,2,8] {
            let mut seen=vec![0;rows as usize*count as usize];
            for thread in 0..rows.div_ceil(32) as usize*count as usize*32 {if let Some((job,row))=serial_projection_index(rows,count,thread) {seen[job*rows as usize+row]+=1;}}
            assert!(seen.iter().all(|&owners|owners==1));
        }}
        assert_eq!(serial_projection_index(0,1,0),None);assert_eq!(serial_projection_index(7,0,0),None);
        let shape=s(Op::Fp8Project);let source=[-0.0,0.0,f32::MIN_POSITIVE,1.0,-1.0,0.5,-0.5,2.0,0.25,3.0,4.0,5.0,6.0,7.0,8.0,9.0];
        let weights=vec![0x38;32];let scales=1.0f32.to_le_bytes().repeat(2);
        assert_eq!(reference(shape,[&source,&[],&[],&[]],&weights,&scales).unwrap().iter().map(|value|value.to_bits()).collect::<Vec<_>>(),
            reference(Shape {op:Op::Fp8Serial,..shape},[&source,&[],&[],&[]],&weights,&scales).unwrap().iter().map(|value|value.to_bits()).collect::<Vec<_>>());
    }
    #[test]
    fn group_entries_cover_each_unit_once() {
        let entry=|first|GroupEntry {shape:s(Op::NvRows),weight:0,scales:0,first,first_row:0,input_words:0,weight_bytes:0,scale_bytes:0,pad:0};
        for firsts in [vec![0],vec![0,3],vec![0,1,2],vec![0,5,6,100,101,4096]] {
            let table=firsts.iter().map(|&f|entry(f)).collect::<Vec<_>>();
            for unit in 0..4200u32 {let want=firsts.iter().rposition(|&f|f<=unit).unwrap();assert_eq!(group_entry(&table,unit).first,firsts[want]);}
        }
    }
    #[test]
    fn table_rows_keep_the_scalar_bits() {
        let lut:[f32;LUT_WORDS]=decode_lut().try_into().unwrap();
        for b in 0..=255u8 {assert_eq!(lut[2*b as usize].to_bits(),e2m1(b&15).to_bits());assert_eq!(lut[2*b as usize+1].to_bits(),e2m1(b>>4).to_bits());assert_eq!(lut[512+b as usize].to_bits(),e4m3(b).to_bits());}
        for tokens in [1u32,2,5,33] {for cols in [16u32,32,96] {
            let base=Shape {rows:17,cols,tokens,..s(Op::NvRows)};let (k,r,m)=(cols as usize,17usize,tokens as usize);
            let x=(0..m*k).map(|i|if i%19==0 {-0.0} else {(i as i32%37-18) as f32/8.0}).collect::<Vec<_>>();
            let q=reference(Shape {op:Op::QuantizeRows,..base},[&x,&[],&[],&[]],&[],&[]).unwrap();
            let w=(0..r*k/2).map(|i|(i*53+17) as u8).collect::<Vec<_>>();let scales=(0..r*k/16).map(|i|0x10+(i%64) as u8).collect::<Vec<_>>();
            let expected=reference(base,[&q,&[],&[],&[]],&w,&scales).unwrap();
            for index in 0..r*m {let mut out=[f32::NAN];group_run(base,index,&q,&w,&scales,&lut,&mut out);assert_eq!(out[0].to_bits(),expected[index].to_bits());}
        }}
        for tokens in [1u32,2,5,33] {for (cols,bc) in [(33u32,16u32),(128,128),(515,128),(7,1024)] {for br in [1u32,2] {
            let base=Shape {op:Op::Fp8Rows,rows:5,cols,block_rows:br,block_cols:bc,tokens,global:1.0,activation:1.0,bound:0.0};let (k,r,m)=(cols as usize,5usize,tokens as usize);
            let x=(0..m*k).map(|i|if i%9==0 {-0.0} else {(i as i32%23-11) as f32*0.037}).collect::<Vec<_>>();
            let rounded=x.iter().copied().map(round).collect::<Vec<_>>();
            let w=(0..r*k).map(|i|((i*53+17)%127) as u8|(((i/7)%2) as u8*128)).collect::<Vec<_>>();
            let grid=k.div_ceil(bc as usize);let scales=(0..r.div_ceil(br as usize)*grid).flat_map(|i|(0.125+(i%17) as f32*0.03125).to_le_bytes()).collect::<Vec<_>>();
            let expected=reference(base,[&x,&[],&[],&[]],&w,&scales).unwrap();
            for index in 0..r*m {let mut out=[f32::NAN];group_run(base,index,&rounded,&w,&scales,&lut,&mut out);assert_eq!(out[0].to_bits(),expected[index].to_bits());}
        }}}
    }
    #[test]
    fn token_tiles_visit_each_output_once() {
        for m in [1u32,2,31,32,33,63,64,65,96,130,1024] {for r in [1u32,3,16,1024] {
            let shape=Shape {tokens:m,rows:r,..s(Op::NvRows)};let total=(m*r) as usize;let mut seen=vec![false;total];
            for local in 0..total {let at=group_unit(shape,local);assert!(!seen[at]);seen[at]=true;}
            let mut seen=vec![false;total];for local in 0..total {let at=group_unit(Shape {op:Op::Fp8Rows,..shape},local);assert!(!seen[at]);seen[at]=true;}
            if m>=16 && r>1 {for lane in 0..8 {assert_eq!(group_unit(shape,lane+8),m as usize+lane);}}
        }}
    }
    #[test]
    fn quantize_lanes_are_chosen_by_shape_only() {
        let q=|op,cols|Shape {op,rows:1,cols,block_rows:0,block_cols:0,tokens:1,global:1.0,activation:0.5,bound:0.0};
        for cols in [16,32,1024,2048,4096,16384] {assert!(quantize_lanes_admitted(&q(Op::Quantize,cols)),"{cols}");}
        for cols in [0,8,17,1040,1088,4100,5120,16400,32768] {assert!(!quantize_lanes_admitted(&q(Op::Quantize,cols)),"{cols}");}
        let retained:Vec<u32>=(1..=64).map(|n|n*16).chain([1536,2048,3072,4096,7168,8192,12288,16384]).collect();
        let admitted:Vec<u32>=(1..=QUANTIZE_LANES_MAX_COLS).filter(|&cols|quantize_lanes_admitted(&q(Op::Quantize,cols))).collect();
        assert_eq!(admitted,retained);assert_eq!(admitted.len(),72);
        assert!(!quantize_lanes_admitted(&Shape {tokens:2,..q(Op::Quantize,4096)}));
        for op in [Op::QuantizeRows,Op::NvProject,Op::Fp8Serial,Op::SwiGlu] {assert!(!quantize_lanes_admitted(&q(op,4096)),"{op:?}");}
        assert_eq!(QUANTIZE_LANES_THREADS%16,0);assert_eq!(QUANTIZE_LANES_THREADS%32,0);
    }
    #[test]
    fn e2m1_bits_match_all_codes_and_signs() {
        let values=[0.0f32,0.5,1.0,1.5,2.0,3.0,4.0,6.0];
        for n in 0..=255u8 {let v=values[(n&7) as usize];let old=if n&8==0 {v} else {-v};
            assert_eq!(e2m1(n).to_bits(),old.to_bits(),"nibble {n}");
        }
    }
    #[test]
    fn output_lanes_cover_each_row_once_at_warp_and_tail_boundaries() {
        for op in [Op::Quantize,Op::NvProject,Op::NvRows,Op::Fp8Project,Op::SwiGlu,Op::Fold] {
            for units in [1usize,7,8,9,15,16,17,31,32,33,1024,4097] {
                let active=if op==Op::NvProject {8} else {32};
                let rows=(0..units.div_ceil(active)*32).filter_map(|i|output_index(op,i)).filter(|&i|i<units).collect::<Vec<_>>();
                assert_eq!(rows,(0..units).collect::<Vec<_>>());
            }
        }
    }
    #[test]
    fn nv_batch_owns_each_row_once_with_full_physical_warps() {
        assert!([8,16,32].contains(&NV_ROWS_PER_WARP));
        for rows in [0u32,1,7,8,9,15,16,17,31,32,33,65,255,256,257,258,1024,2048] {for cols in [16u32,80,2048,4096] {for count in [0u32,1,2,3,8,16,17] {
            let active=nv_rows_per_warp(rows,cols,count);let grid=rows.div_ceil(active) as usize*count as usize;
            let mut seen=vec![0;rows as usize*count as usize];
            for thread in 0..(grid+1)*32 {if let Some((job,row))=nv_projection_batch_index(rows,cols,count,thread) {
                assert!(thread<grid*32);assert!(thread%32<active as usize);seen[job*rows as usize+row]+=1;
            }}assert!(seen.iter().all(|&owners|owners==1));
        }}}
    }
    #[test]
    fn nv_packed_envelope_is_the_device_swept_family() {
        for rows in 1..=257 {for cols in NV_PACKED_COLS {for count in [1,2,8,16] {assert!(nv_packed_shape(rows,cols,count));}}}
        for (rows,cols) in [(1024,4096),(2048,2048)] {for count in [1,2,8,16] {assert!(nv_packed_shape(rows,cols,count));}}
        assert!(nv_packed_shape(65536,16,1));assert!(!nv_packed_shape(65536,16,2));
        for rows in [0,258,1023,1025,2047,2049,65535,65537,u32::MAX/8] {assert!(!nv_packed_shape(rows,16,1));}
        for count in [0,3,4,7,15,17,u32::MAX] {assert!(!nv_packed_shape(16,16,count));}
    }
    #[test]
    fn nv_unproven_shape_keeps_original_arithmetic_and_geometry() {
        let shape=Shape {op:Op::NvProject,rows:2,cols:80,block_rows:1,block_cols:16,tokens:1,global:1.0,activation:1.0,bound:0.0};
        for count in [1u32,3] {
            assert!(!nv_packed_shape(shape.rows,shape.cols,count));assert_eq!(nv_rows_per_warp(shape.rows,shape.cols,count),8);
            for thread in 0..(count+1) as usize*32 {assert_eq!(nv_projection_batch_index(2,80,count,thread),projection_batch_index(2,count,thread));}
        }
        assert_eq!(reference(shape,[&[1.0;85],&[],&[],&[]],&[0x22;80],&[0x38;10]).unwrap(),[80.0;2]);
        let largest=Shape {rows:u32::MAX/8,cols:16,..shape};assert!(largest.lengths().is_some());
        assert_eq!(nv_rows_per_warp(largest.rows,16,1),8);
        let last=u32::MAX as usize-1;assert_eq!(nv_projection_batch_index(1,16,u32::MAX,last*32),Some((last,0)));
    }
    #[test]
    fn exact_encodings_ties_and_signed_zero() {
        for b in 0..=254u8 {if b&127!=127 {assert_eq!(nearest_e4m3(e4m3(b)),b);}}
        for n in 0..16 {assert_eq!(nearest_e2m1(e2m1(n)),n);}
        for (x,n) in [(0.25,0),(0.75,2),(1.25,2),(1.75,4),(2.5,4),(3.5,6),(5.0,6)] {
            assert_eq!(nearest_e2m1(x),n);assert_eq!(nearest_e2m1(-x),n|8);
        }
        assert_eq!(nearest_e4m3(1.0625),0x38);assert_eq!(nearest_e4m3(1.1875),0x3a);
    }
    #[test]
    fn calibrated_bytes_and_global_fold() {
        let mut x=[0.0;16];x[..8].copy_from_slice(&[6.0,0.25,0.75,1.25,1.75,2.5,3.5,5.0]);x[8]=-0.0;
        let q=reference(s(Op::Quantize),[&x,&[],&[],&[]],&[],&[]).unwrap();
        let expected=[6.0f32,0.0,1.0,1.0,2.0,2.0,4.0,4.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0];
        assert_eq!(q.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.map(f32::to_bits));
        let mut shape=s(Op::NvProject);shape.global=2.0;
        let v=reference(shape,[&q,&[],&[],&[]],&[0x22;16],&[0x38;2]).unwrap();assert_eq!(v,[40.0;2]);
        let mut shape=s(Op::Quantize);shape.activation=2.0;
        let scaled=reference(shape,[&x,&[],&[],&[]],&[],&[]).unwrap();assert_eq!(scaled[..16].iter().map(|v|v.to_bits()).collect::<Vec<_>>(),q[..16].iter().map(|v|v.to_bits()).collect::<Vec<_>>());assert_eq!(scaled[16],0.5);
    }
    #[test]
    fn decoded_activation_bits_match_the_original_numeric_code_oracle() {
        for width in [16,32,128] {for activation in [0.125,1.0,4.0] {
            let mut shape=s(Op::Quantize);shape.cols=width;shape.activation=activation;
            let input=(0..width as usize).map(|i|if i%7==0 {-0.0} else {(i as i32%31-15) as f32*0.0625}).collect::<Vec<_>>();
            let actual=reference(shape,[&input,&[],&[],&[]],&[],&[]).unwrap();
            for (row,lanes) in input.chunks_exact(16).zip(actual.chunks_exact(17)) {
                let maximum=row.iter().fold(0.0f32,|a,&v|a.max(round(v).abs()));
                let scale=nearest_e4m3((maximum/activation)/6.0);let block=e4m3(scale);
                let old=|v:f32|if v==0.0 {0} else if block==0.0 {if v<0.0 {8} else {0}} else {nearest_e2m1((round(v)/activation)/block)};
                for j in 0..16 {assert_eq!(lanes[j].to_bits(),e2m1(old(row[j])).to_bits());}
                assert_eq!(lanes[16].to_bits(),e4m3(scale).to_bits());
            }
        }}
    }
    #[test]
    fn quantizer_keeps_negative_underflow_but_not_exact_negative_zero() {
        let mut input=[0.0;64];input[..8].copy_from_slice(&[6.0,0.0,-0.0,0.125,-0.125,f32::from_bits(1),-f32::from_bits(1),-0.25]);
        input[16..32].fill(-0.0);input[32..48].fill(-f32::from_bits(1));input[48..64].fill(-f32::from_bits(0x0001_0000));
        for op in [Op::Quantize,Op::QuantizeRows] {let mut shape=s(op);shape.cols=64;
            let q=reference(shape,[&input,&[],&[],&[]],&[],&[]).unwrap();
            let expected=[6.0f32,0.0,0.0,0.0,-0.0,0.0,-0.0,-0.0];
            assert_eq!(q[..8].iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.map(f32::to_bits));
            assert!(q[17..34].iter().all(|v|v.to_bits()==0));
            for block in q[34..].chunks_exact(17) {assert!(block[..16].iter().all(|v|v.to_bits()==(-0.0f32).to_bits()));assert_eq!(block[16].to_bits(),0);}
        }
    }
    #[test]
    fn decoded_activations_keep_the_original_packed_ascending_feature_fold() {
        for rows in [1,17,65] {for cols in [16,32,128] {
            let mut shape=s(Op::NvProject);shape.rows=rows;shape.cols=cols;shape.global=0.75;shape.activation=0.125;
            let k=cols as usize;let packed=(0..k/16*9).map(|i|if i%9==8 {0x20+(i%31) as u8} else {((i*37+9)%256) as u8}).collect::<Vec<_>>();
            let x=packed.chunks_exact(9).flat_map(|b|(0..16).map(|j|e2m1(if j%2==0 {b[j/2]&15} else {b[j/2]>>4})).chain([e4m3(b[8])])).collect::<Vec<_>>();
            let w=(0..rows as usize*k/2).map(|i|((i*53+17)%256) as u8).collect::<Vec<_>>();
            let scales=(0..rows as usize*k/16).map(|i|0x10+(i%64) as u8).collect::<Vec<_>>();
            let actual=reference(shape,[&x,&[],&[],&[]],&w,&scales).unwrap();
            for (row,&value) in actual.iter().enumerate() {let global=shape.global*shape.activation;let mut sum=0.0f32;
                for block in 0..k/16 {let a=&packed[block*9..block*9+9];let fold=(e4m3(scales[row*(k/16)+block])*e4m3(a[8]))*global;
                    for j in 0..16 {let packed=w[row*(k/2)+block*8+j/2];let input=a[j/2];
                        let weight=e2m1(if j%2==0 {packed&15} else {packed>>4});let input=e2m1(if j%2==0 {input&15} else {input>>4});sum+=(weight*input)*fold;
                    }
                }assert_eq!(value.to_bits(),round(sum).to_bits(),"rows={rows} cols={cols} row={row}");
            }
        }}
    }
    #[test]
    fn decoded_planes_reject_old_extents_and_non_format_values() {
        let shape=s(Op::NvProject);let mut x=vec![0.0;17];let weight=[0x22;16];let scales=[0x38;2];
        assert!(reference(shape,[&x[..9],&[],&[],&[]],&weight,&scales).is_err());
        x[0]=0.25;assert!(reference(shape,[&x,&[],&[],&[]],&weight,&scales).is_err());
        x[0]=-0.0;x[16]=-0.0;assert!(reference(shape,[&x,&[],&[],&[]],&weight,&scales).is_err());
        x[16]=0.0;assert!(reference(shape,[&x,&[],&[],&[]],&weight,&scales).is_ok());
    }
    #[test]
    fn hoisted_fp8_scales_keep_each_feature_add_at_partial_blocks() {
        for rows in [1,3,17] {for cols in [1,17,129,515] {for br in [1,3,64] {for bc in [1,7,128,1024] {
            let mut shape=s(Op::Fp8Project);shape.rows=rows;shape.cols=cols;shape.block_rows=br;shape.block_cols=bc;
            let (r,k)=(rows as usize,cols as usize);
            let x=(0..k).map(|i|if i%11==0 {-0.0} else {(i as i32%23-11) as f32*0.037}).collect::<Vec<_>>();
            let w=(0..r*k).map(|i|((i*53+17)%127) as u8|(((i/7)%2) as u8*128)).collect::<Vec<_>>();
            let grid=k.div_ceil(bc as usize);let scales=(0..r.div_ceil(br as usize)*grid)
                .flat_map(|i|(0.125+(i%17) as f32*0.03125).to_le_bytes()).collect::<Vec<_>>();
            let got=reference(shape,[&x,&[],&[],&[]],&w,&scales).unwrap();
            for (row,&value) in got.iter().enumerate() {let mut sum=0.0f32;
                for j in 0..k {let at=((row/br as usize)*grid+j/bc as usize)*4;
                    let scale=f32::from_le_bytes(scales[at..at+4].try_into().unwrap());
                    sum+=(e4m3(w[row*k+j])*scale)*round(x[j]);
                }assert_eq!(value.to_bits(),round(sum).to_bits(),"row={row} cols={cols} br={br} bc={bc}");
            }
        }}}}
    }
    #[test]
    fn fp8_partial_blocks_and_fixed_fold_order() {
        let mut shape=s(Op::Fp8Project);shape.rows=3;shape.cols=3;shape.block_rows=2;shape.block_cols=2;
        let scales=[1.0f32,2.0,3.0,4.0].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
        assert_eq!(reference(shape,[&[1.0;3],&[],&[],&[]],&[0x38;9],&scales).unwrap(),[4.0,4.0,10.0]);
        let mut shape=s(Op::Fold);shape.rows=1;shape.cols=3;
        assert_eq!(reference(shape,[&[256.0,-256.0,1.0],&[0.0,2.0,1.0],&[1.0;3],&[0.0]],&[],&[]).unwrap(),[0.0]);
        assert!(reference(shape,[&[0.0;3],&[1.0;3],&[1.0;3],&[0.0]],&[],&[]).is_err());
        assert!(reference(s(Op::Quantize),[&[f32::NAN;16],&[],&[],&[]],&[],&[]).is_err());
    }
    #[test]
    fn grouped_rows_keep_scalar_quantization_folds_and_scale_tails() {
        let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        for tokens in [1,2,3,4,5,8,33] {for cols in [16,32,96] {
            let base=Shape {rows:33,cols,tokens:1,..s(Op::NvProject)};let k=cols as usize;let r=base.rows as usize;
            let x=(0..tokens as usize*k).map(|i|if i%19==0 {-0.0} else {(i as i32%37-18) as f32/8.0}).collect::<Vec<_>>();
            let q=reference(Shape {op:Op::QuantizeRows,tokens,..base},[&x,&[],&[],&[]],&[],&[]).unwrap();
            let w=(0..r*k/2).map(|i|(i*17+31) as u8).collect::<Vec<_>>();let scales=(0..r*k/16).map(|i|0x28+(i%24) as u8).collect::<Vec<_>>();
            let got=reference(Shape {op:Op::NvRows,tokens,..base},[&q,&[],&[],&[]],&w,&scales).unwrap();
            for token in 0..tokens as usize {
                let scalar_q=reference(Shape {op:Op::Quantize,..base},[&x[token*k..(token+1)*k],&[],&[],&[]],&[],&[]).unwrap();
                assert_eq!(bits(&q[token*k/16*17..(token+1)*k/16*17]),bits(&scalar_q));
                let expected=reference(base,[&scalar_q,&[],&[],&[]],&w,&scales).unwrap();
                for row in 0..r {assert_eq!(got[row*tokens as usize+token].to_bits(),expected[row].to_bits());}
            }
        }}
        for tokens in [1,2,3,4,5,8,33] {let base=Shape {op:Op::Fp8Project,rows:5,cols:33,block_rows:2,block_cols:16,tokens:1,..s(Op::Fp8Project)};
            let k=base.cols as usize;let r=base.rows as usize;let x=(0..tokens as usize*k).map(|i|if i%9==0 {-0.0} else {(i as i32%23-11) as f32/16.0}).collect::<Vec<_>>();
            let w=(0..r*k).map(|i|(i%120+1) as u8).collect::<Vec<_>>();let scales=(0..9).flat_map(|i|(0.25+(i as f32)/16.0).to_le_bytes()).collect::<Vec<_>>();
            let got=reference(Shape {op:Op::Fp8Rows,tokens,..base},[&x,&[],&[],&[]],&w,&scales).unwrap();
            for token in 0..tokens as usize {let expected=reference(base,[&x[token*k..(token+1)*k],&[],&[],&[]],&w,&scales).unwrap();for row in 0..r {assert_eq!(got[row*tokens as usize+token].to_bits(),expected[row].to_bits());}}
        }
        assert!(Shape {tokens:(crate::prompt_ops::MAX_ROWS+1) as u32,..s(Op::NvRows)}.lengths().is_none());
    }

}
