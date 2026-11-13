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
/// `role` of an NvProject / NvRows shape names the WORD FUNCTION of the product (`tc_contract`), fixed by the role
/// of the matrix in the model. `NV_FIRST` (0) = a FIRST matrix (expert gate, expert up): one chain a word.
/// `NV_SECOND` (1) = a SECOND matrix (expert down): the quarter chain, four chains joined
/// fl(fl(a0+a1)+fl(a2+a3)); cols must be a multiple of 256. The quantizer of a matrix's input may carry the matrix's
/// role (its shape is made from the matrix's); every other operation carries `NV_FIRST` (0).
pub const NV_FIRST:u32=0;
pub const NV_SECOND:u32=1;
/// `FP8_QUARTER` (2), on `Op::Fp8Serial` only: the job is ONE COLUMN QUARTER of a block-FP8 second matrix (shared expert
/// down, latent o) in the pair split. Its word is the RAW FP32 accumulator of today's chain over the job's own columns,
/// from +0.0: no BF16 round at the end (`tc_twin_quarter::fp8_chain`). The quarter join and the one BF16 round of the
/// sublayer come after it (`nv_quarter_fold`, `tp_reduce_take`). Only the tile kernel has this form.
pub const FP8_QUARTER:u32=2;
/// Chains a word of an NVFP4 product shape: 1 or 4.
#[inline(always)]
pub fn nv_chains(s:&Shape)->u32 {if s.role==NV_SECOND {crate::tc_contract::CHAINS_SECOND} else {crate::tc_contract::CHAINS_FIRST}}
/// `role` is the tenth word. It sits where `Arguments` and `GroupEntry` had four bytes of padding, so no other
/// field of either moved, and a table written with zero padding names a first matrix.
#[repr(C)]
#[derive(Debug,Clone,Copy)]
pub struct Shape {pub op:Op,pub rows:u32,pub cols:u32,pub block_rows:u32,pub block_cols:u32,pub tokens:u32,
    pub global:f32,pub activation:f32,pub bound:f32,pub role:u32}
impl Shape {
    pub fn lengths(self)->Option<([usize;4],usize,usize,usize,usize)> {
        let (r,k,br,bc)=(self.rows as usize,self.cols as usize,self.block_rows as usize,self.block_cols as usize);
        if r==0 || k==0 || !self.global.is_finite() || self.global<=0.0 || !self.activation.is_finite()
            || self.activation<=0.0 || !self.bound.is_finite() || self.role>FP8_QUARTER
            // A quantizer shape is made from its matrix's shape and carries that role; it does not read it.
            || self.role==NV_SECOND && !matches!(self.op,Op::NvProject|Op::NvRows|Op::Quantize|Op::QuantizeRows)
            // A raw quarter job is a block-FP8 decode projection of whole scale blocks.
            || self.role==FP8_QUARTER && (self.op!=Op::Fp8Serial || bc==0 || k%bc!=0) {return None;}
        let rk=r.checked_mul(k)?;
        let m=self.tokens as usize;if m==0 || m>crate::prompt_ops::MAX_ROWS {return None;}
        let result=match self.op {
            Op::Quantize if k%16==0=>([k,0,0,0],0,0,k/16,17),
            Op::NvProject|Op::NvRows if self.role==NV_SECOND && k%256!=0=>return None,
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
/// COLUMN-TILE storage of an NVFP4 matrix. `block_rows == NV_TILE_ROWS` on an NvProject or NvRows shape says how the
/// raw owners are laid out, and nothing else: values, products, their order and every rounding stay as in the
/// row-major form. A tile is 128 consecutive rows. Codes: column j of tile t is the 64 bytes at `(t*cols + j)*64`
/// (row 2k of the tile in the low nibble of byte k, row 2k+1 in the high nibble). Scales: 16-column block b of tile t
/// is the 128 bytes at `(t*cols/16 + b)*128`, one byte a row. The byte counts equal the row-major form.
/// Why: a decode kernel can then leave out the columns whose quantized activation is zero (`nv_cols`).
pub const NV_TILE_ROWS:u32=128;
/// The storage rule, shape only: whole tiles, whole 64-column rounds, and activations that fit the kernel's shared stage.
pub const NV_TILE_MAX_COLS:u32=4096;
#[inline(always)]
pub fn nv_tile_layout_admits(rows:usize,cols:usize)->bool {rows>0 && rows%NV_TILE_ROWS as usize==0 && cols>=64 && cols%64==0 && cols<=NV_TILE_MAX_COLS as usize}
#[inline(always)]
pub fn nv_tiled(s:&Shape)->bool {matches!(s.op,Op::NvProject|Op::NvRows) && s.block_rows==NV_TILE_ROWS}
/// Byte index and bit shift of the code of (row, column) in either layout.
#[inline(always)]
pub fn nv_code_at(s:&Shape,row:usize,col:usize)->(usize,u32) {
    let k=s.cols as usize;
    if nv_tiled(s) {let t=NV_TILE_ROWS as usize;(((row/t)*k+col)*(t/2)+(row%t)/2,4*(row as u32&1))} else {(row*(k/2)+col/2,4*(col as u32&1))}
}
/// Byte index of the scale of (row, 16-column block) in either layout.
#[inline(always)]
pub fn nv_scale_at(s:&Shape,row:usize,block:usize)->usize {
    let blocks=s.cols as usize/16;
    if nv_tiled(s) {let t=NV_TILE_ROWS as usize;((row/t)*blocks+block)*t+row%t} else {row*blocks+block}
}
/// The decode kernel of the column-tile layout (`nv_cols`): one block of NV_COLS_THREADS threads a tile a job, a thread a
/// row; a round is NV_COLS_ROUND_BLOCKS 16-column blocks; asynchronous copies run NV_COLS_DEPTH rounds deep.
pub const NV_COLS_THREADS:u32=128;
pub const NV_COLS_ROUND_BLOCKS:u32=4;
pub const NV_COLS_DEPTH:u32=3;
pub const NV_COLS_MAX_JOBS:u32=16;
/// Dynamic shared bytes of one block: the staged activations and NV_COLS_DEPTH rounds of codes and scales.
pub const fn nv_cols_shared(cols:u32)->u32 {(cols/16*17*4).div_ceil(16)*16+NV_COLS_DEPTH*NV_COLS_ROUND_BLOCKS*(1024+128)}
/// Blocks of a launch: job-major, one a tile.
pub fn nv_cols_blocks(rows:u32,jobs:u32)->Option<u32> {(rows/NV_TILE_ROWS).checked_mul(jobs)}
/// True when equal decode jobs of this shape run in `nv_cols`: a column-tile NvProject of one token inside the storage rule,
/// with 16-byte aligned raw owners and a 4-byte aligned input. Every other column-tile launch keeps the scalar body.
pub fn nv_cols_admitted(shape:&Shape,jobs:u32,owners:&[(u64,u64)])->bool {
    shape.op==Op::NvProject && nv_tiled(shape) && shape.tokens==1 && nv_tile_layout_admits(shape.rows as usize,shape.cols as usize)
        && (1..=NV_COLS_MAX_JOBS).contains(&jobs) && owners.len()==jobs as usize && owners.iter().all(|(w,s)|(w|s)&15==0)
}
/// Arguments of the load-time kernel `nv_tile_repack`: row-major NVFP4 planes in, column tiles out. The four planes are
/// different allocations of rows*cols/2 code bytes and rows*cols/16 scale bytes, 16-byte aligned.
#[repr(C)]
#[derive(Clone,Copy)]
pub struct TileRepack {pub source_codes:u64,pub source_scales:u64,pub codes:u64,pub scales:u64,pub rows:u32,pub cols:u32}
/// A block of `nv_tile_repack` moves one 128-row tile x NV_REPACK_COLS columns (4 scale blocks) with 128 threads.
pub const NV_REPACK_THREADS:u32=128;
pub const NV_REPACK_COLS:u32=64;
/// Shared bytes of a block: the staged code rows (32 bytes a row, 8 more a group of 8 rows) and scale rows (one word a row).
pub const NV_REPACK_CODE_STAGE:u32=128*32+8*16;
pub const NV_REPACK_SHARED:u32=NV_REPACK_CODE_STAGE+128*4;
/// Blocks of a launch: tile-major, cols/64 chunks a tile.
pub fn nv_tile_repack_blocks(rows:usize,cols:usize)->Option<u32> {u32::try_from((rows/NV_TILE_ROWS as usize).checked_mul(cols/NV_REPACK_COLS as usize)?).ok()}
/// Shared byte offset of staged code row `row` (0..128) of a block. The 8 bytes a group of 8 rows put the rows that
/// one output word needs (8g .. 8g+7 for g = 0..16) in different shared-memory banks at equal column offsets.
#[inline(always)]
pub const fn nv_repack_code_slot(row:usize)->usize {row*32+8*(row/8)}
/// Shared byte offset (from the scale stage) of the staged scale word of `row`: rows 4g+k for g = 0..32 fall in 32 banks.
#[inline(always)]
pub const fn nv_repack_scale_slot(row:usize)->usize {4*(row/4+32*(row%4))}
/// The code unit of `thread` in `step` (0..8): (column 0..64 of the chunk, word 0..16 of the column). A word is 4 output
/// bytes = row pairs 4w..4w+3 = rows 8w..8w+7. Half a warp writes the 16 words of one column, the other half of the
/// column 8 further on; over 8 steps the 4 warps cover the 64 columns once.
#[inline(always)]
pub const fn nv_repack_code_unit(thread:usize,step:usize)->(usize,usize) {let lane=thread%32;((thread/32)*16+step+8*(lane/16),lane%16)}
/// Row-major NVFP4 codes (rows x cols/2 bytes) to column tiles, IN PLACE. A tile keeps its own byte range in both
/// layouts (128 rows of cols/2 bytes = cols columns of 64 bytes), so one tile of scratch is all the memory this needs.
#[cfg(not(target_arch="nvptx64"))]
pub fn nv_tile_codes_in_place(planes:&mut [u8],rows:usize,cols:usize) {
    // The scratch rows are 64 bytes longer than a row: with a power-of-two pitch the 64 bytes of one column fall in
    // one cache set and the second step runs 2.4 times slower (measured at 4096 columns).
    let t=NV_TILE_ROWS as usize;let half=cols/2;let pitch=cols+64;let mut pairs=vec![0u8;(t/2)*pitch];
    for tile in planes[..rows*half].chunks_exact_mut(t*half) {
        // Step 1 (straight runs): rows 2p and 2p+1 to one byte a column, low nibble = even row.
        for (pair,out) in pairs.chunks_exact_mut(pitch).enumerate() {
            let (even,odd)=tile[2*pair*half..(2*pair+2)*half].split_at(half);
            for ((two,&a),&b) in out[..cols].chunks_exact_mut(2).zip(even).zip(odd) {two[0]=(a&15)|((b&15)<<4);two[1]=(a>>4)|(b&0xf0);}
        }
        // Step 2: the byte matrix [pair][column] to [column][pair].
        for (column,out) in tile.chunks_exact_mut(t/2).enumerate() {for (pair,byte) in out.iter_mut().enumerate() {*byte=pairs[pair*pitch+column];}}
    }
}
/// Row-major scales (rows x cols/16 bytes) to block-major tiles, IN PLACE (a tile keeps its own byte range).
#[cfg(not(target_arch="nvptx64"))]
pub fn nv_tile_scales_in_place(planes:&mut [u8],rows:usize,cols:usize) {
    let t=NV_TILE_ROWS as usize;let blocks=cols/16;let mut source=vec![0u8;t*blocks];
    for tile in planes[..rows*blocks].chunks_exact_mut(t*blocks) {
        source.copy_from_slice(tile);
        for (block,out) in tile.chunks_exact_mut(t).enumerate() {for (row,byte) in out.iter_mut().enumerate() {*byte=source[row*blocks+block];}}
    }
}
/// Column-tile copy of row-major NVFP4 codes (tests and device checks; the loader repacks in place).
#[cfg(not(target_arch="nvptx64"))]
pub fn nv_tile_codes(row_major:&[u8],rows:usize,cols:usize)->Vec<u8> {let mut out=row_major[..rows*(cols/2)].to_vec();nv_tile_codes_in_place(&mut out,rows,cols);out}
/// Block-major tile copy of row-major scales.
#[cfg(not(target_arch="nvptx64"))]
pub fn nv_tile_scales(row_major:&[u8],rows:usize,cols:usize)->Vec<u8> {let mut out=row_major[..rows*(cols/16)].to_vec();nv_tile_scales_in_place(&mut out,rows,cols);out}
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
    // The one-warp body rounds every word: a raw quarter job is never sent to it.
    } else if jobs.iter().any(|job|job.shape.role==FP8_QUARTER) {Err("a raw FP8 quarter job needs the tile kernel (admitted shape, 16-byte aligned owners, 1 to 8 jobs)".into())
    } else {Ok((grid,32,0))}
}
/// True when the jobs of a projection batch may share one launch: one operation, one geometry and one word function
/// (`role`). The decode kernel reads the word function from the first job; a job of the other role would trap on the device.
#[cfg(not(target_arch="nvptx64"))]
pub fn batch_shapes_agree(shapes:impl IntoIterator<Item=Shape>)->bool {
    let mut shapes=shapes.into_iter();let Some(first)=shapes.next() else {return false;};
    shapes.all(|s|s.op==first.op && s.rows==first.rows && s.cols==first.cols && s.role==first.role)
}
/// Keep the NV stream geometry separate from the block-FP8 geometry and its fallback.
#[cfg(not(target_arch="nvptx64"))]
pub fn batch_launch_geometry(jobs:&[Arguments],grid:u32,count:u32)->Result<(bool,u32,u32,u32),String> {
    let owners=jobs.iter().map(|job|(job.weight,job.scales)).collect::<Vec<_>>();
    // Column-tile jobs: the tile kernel of that layout, or the scalar body. Never a row-major kernel.
    if jobs.first().is_some_and(|job|nv_tiled(&job.shape)) {
        if nv_cols_admitted(&jobs[0].shape,count,&owners) && jobs.iter().all(|job|nv_tiled(&job.shape) && job.input[0]&3==0) {
            let blocks=nv_cols_blocks(jobs[0].shape.rows,count).ok_or("NV column grid overflow")?;
            return Ok((true,blocks,NV_COLS_THREADS,nv_cols_shared(jobs[0].shape.cols)));
        }
        return Ok((false,grid,32,0));
    }
    // TENSOR-CORE REBASE: the row-major stream and tile kernels (nv_rows_stream, nv_tile) keep the rc.8 chain, so no
    // NVFP4 launch reaches them; a row-major NVFP4 batch takes the generic body (`unit`, the contract's words).
    if jobs.first().is_some_and(|job|job.shape.op!=Op::NvProject && crate::nv_stream::projection_admitted(job.shape,count,&owners)) {
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
        words[14..24].copy_from_slice(&[self.shape.op as u32,self.shape.rows,self.shape.cols,self.shape.block_rows,self.shape.block_cols,
            self.shape.tokens,self.shape.global.to_bits(),self.shape.activation.to_bits(),self.shape.bound.to_bits(),self.shape.role]);words
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
#[allow(dead_code)] // rc.8 table form, kept with `nv_row` for the history tests.
fn nv_row_table(s:Shape,row:usize,x:&[f32],w:&[u8],scales:&[u8],lut:&[f32;LUT_WORDS])->f32 {
    let k=s.cols as usize;let blocks=k/16;let global=s.global*s.activation;let mut sum=0.0f32;
    if nv_tiled(&s) {
        // Column tiles: the same folds, products and feature order; only the byte addresses differ.
        for (block,a) in x.chunks_exact(17).enumerate() {
            let fold=(lut[512+scales[nv_scale_at(&s,row,block)] as usize]*a[16])*global;
            for j in 0..16 {let (at,shift)=nv_code_at(&s,row,block*16+j);sum+=(lut[((w[at]>>shift)&15) as usize*2]*a[j])*fold;}
        }
        return round(sum);
    }
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
            let _=lut;out[0]=nv_word(s,index/m,&x[token*stride..(token+1)*stride],w,scales);},
        Op::Fp8Rows=>{let m=s.tokens as usize;let k=s.cols as usize;let token=index%m;
            out[0]=fp8_row_table(s,index/m,&x[token*k..(token+1)*k],w,scales,lut);},
        _=>unit(s,index,[x,&[],&[],&[]],w,scales,out),
    }
}
/// One NVFP4 product word on CONTRACT (b) (`tc_contract`): the word function of the tensor-core family, for either
/// storage layout. `a` is the token's quantized row (17 words a 16-column block). 64-column steps in ascending
/// column order from column 0; a last step with fewer than four blocks has no term for the absent blocks. One chain
/// (first matrices) or four quarter chains joined fl(fl(a0+a1)+fl(a2+a3)) (second matrices), then the epilogue
/// bf16(fl(acc * fl(global * activation))). Host reference and generic device body; every specialized kernel
/// (`nv_cols`, `fp4_cohort*`) must give this word.
#[inline(always)]
pub fn nv_word(s:Shape,row:usize,a:&[f32],w:&[u8],scales:&[u8])->f32 {
    use crate::tc_contract as tc;
    let k=s.cols as usize;let blocks=k/16;let steps=blocks.div_ceil(4);
    let chains=nv_chains(&s) as usize;let per_chain=if chains==4 {steps/4} else {steps};
    let (mut parts,mut acc)=([0u32;4],0u32);
    let mut step=0;
    while step<steps {
        let (mut sum,mut live,mut wp,mut ap)=([0i32;4],[false;4],[(0u32,-6i32);4],[(0u32,-6i32);4]);
        let mut q=0;
        while q<4 && step*4+q<blocks {
            let block=step*4+q;let x=&a[block*17..block*17+17];
            let mut j=0;
            while j<16 {
                let (at,shift)=nv_code_at(&s,row,block*16+j);
                let product=tc::half_units(u32::from((w[at]>>shift)&15))*tc::half_units_f32(x[j].to_bits());
                sum[q]+=product;live[q]|=product!=0;j+=1;
            }
            wp[q]=tc::ue4m3_parts(u32::from(scales[nv_scale_at(&s,row,block)]));ap[q]=tc::ue4m3_parts_f32(x[16].to_bits());
            q+=1;
        }
        acc=tc::chain_step(acc,sum,live,wp,ap);
        step+=1;
        if chains==4 && step%per_chain==0 {parts[(step/per_chain-1)&3]=acc;acc=0;}
    }
    if chains==4 {
        let f=|x:u32|f32::from_bits(x);
        acc=((f(parts[0])+f(parts[1]))+(f(parts[2])+f(parts[3]))).to_bits();
    }
    round(f32::from_bits(acc)*(s.global*s.activation))
}
/// rc.8 chain (before the tensor-core rebase): kept for the bench baseline and the history tests only.
#[inline(always)]
#[allow(dead_code)]
fn nv_row(s:Shape,index:usize,x:&[f32],w:&[u8],scales:&[u8])->f32 {
    let k=s.cols as usize;
    let global=s.global*s.activation;let mut sum=0.0f32;let tiled=nv_tiled(&s);
    for block in 0..k/16 {
        let a=&x[block*17..block*17+17];let fold=(e4m3(scales[nv_scale_at(&s,index,block)])*a[16])*global;
        if tiled {for j in 0..16 {let (at,shift)=nv_code_at(&s,index,block*16+j);sum+=(e2m1((w[at]>>shift)&15)*a[j])*fold;}continue;}
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
        Op::NvProject=>{out[0]=nv_word(s,index,x[0],w,scales);},
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
            }out[0]=if s.role==FP8_QUARTER {sum} else {round(sum)};
        },
        Op::NvRows=>{
            // Adjacent token lanes read the same immutable weight row. Each
            // owns one accumulator and keeps the scalar feature fold.
            let m=s.tokens as usize;let token=index%m;let row=index/m;let stride=k/16*17;
            out[0]=nv_word(s,row,&x[0][token*stride..(token+1)*stride],w,scales);
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
        let shape=super::Shape {op:super::Op::NvProject,rows:33,cols:128,block_rows:1,block_cols:16,tokens:1,global:0.75,activation:0.125,bound:0.0,role:0};
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
    fn s(op:Op)->Shape {Shape {op,tokens:1,rows:2,cols:16,block_rows:1,block_cols:16,global:1.0,activation:1.0,bound:10.0,role:0}}
    fn job(op:Op,rows:u32,cols:u32,br:u32,bc:u32)->Arguments {
        Arguments {input:[0x1000,4,4,4],weight:0x2000,scales:0x3000,output:0x4000,shape:Shape {op,rows,cols,block_rows:br,block_cols:bc,tokens:1,global:1.0,activation:1.0,bound:0.0,role:0}}
    }
    fn geometry(jobs:&[Arguments])->(u32,u32,u32) {launch_geometry(jobs,77,jobs.len() as u32).unwrap()}
    #[test]
    fn batch_geometry_preserves_nv_stream_fp8_tile_and_original_fallback() {
        // TENSOR-CORE REBASE: a row-major NVFP4 batch never takes the rc.8 stream or tile kernel; it keeps the generic body.
        let mut nv=job(Op::NvProject,1024,4096,0,0);nv.scales=512;
        assert!(!batch_launch_geometry(&[nv;16],77,16).unwrap().0);
        for rows in [3,17,256,257] {let other=Arguments {shape:Shape {rows,..nv.shape},..nv};
            assert!(!batch_launch_geometry(&[other;2],77,2).unwrap().0);}
        let down=Arguments {shape:Shape {rows:2048,cols:2048,..nv.shape},..nv};
        assert!(!batch_launch_geometry(&[down;8],77,8).unwrap().0);
        assert!(!batch_launch_geometry(&[down;16],77,16).unwrap().0);
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
            let base=Shape {op:Op::Fp8Rows,rows:5,cols,block_rows:br,block_cols:bc,tokens,global:1.0,activation:1.0,bound:0.0,role:0};let (k,r,m)=(cols as usize,5usize,tokens as usize);
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
        let q=|op,cols|Shape {op,rows:1,cols,block_rows:0,block_cols:0,tokens:1,global:1.0,activation:0.5,bound:0.0,role:0};
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
        let shape=Shape {op:Op::NvProject,rows:2,cols:80,block_rows:1,block_cols:16,tokens:1,global:1.0,activation:1.0,bound:0.0,role:0};
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
    fn decoded_activations_give_the_contract_word_of_the_independent_twin() {
        // TENSOR-CORE REBASE: the reference word is contract (b) (`nv_word`); the independent model is `tc_twin`.
        // Both storage layouts, both word functions, whole and partial 64-column steps.
        for (rows,cols,second) in [(1u32,16u32,false),(17,32,false),(65,128,false),(128,256,false),(128,256,true),(256,1024,true),(3,80,false)] {
            for tiled in [false,true] {
                if tiled && !nv_tile_layout_admits(rows as usize,cols as usize) {continue;}
                let mut shape=s(Op::NvProject);shape.rows=rows;shape.cols=cols;shape.global=0.75;shape.activation=0.125;
                shape.role=if second {NV_SECOND} else {NV_FIRST};
                let (k,r)=(cols as usize,rows as usize);let packed=(0..k/16*9).map(|i|if i%9==8 {0x20+(i%31) as u8} else {((i*37+9)%256) as u8}).collect::<Vec<_>>();
                let x=packed.chunks_exact(9).flat_map(|b|(0..16).map(|j|e2m1(if j%2==0 {b[j/2]&15} else {b[j/2]>>4})).chain([e4m3(b[8])])).collect::<Vec<_>>();
                let w=(0..r*k/2).map(|i|((i*53+17)%256) as u8).collect::<Vec<_>>();
                let scales=(0..r*k/16).map(|i|if i%23==0 {(i%8) as u8} else {0x10+(i%64) as u8}).collect::<Vec<_>>();
                let (planes,scale_planes)=if tiled {shape.block_rows=NV_TILE_ROWS;(nv_tile_codes(&w,r,k),nv_tile_scales(&scales,r,k))} else {(w.clone(),scales.clone())};
                let actual=reference(shape,[&x,&[],&[],&[]],&planes,&scale_planes).unwrap();
                let a=packed.chunks_exact(9).flat_map(|b|(0..16).map(|j|if j%2==0 {b[j/2]&15} else {b[j/2]>>4})).collect::<Vec<_>>();
                let sa=packed.chunks_exact(9).map(|b|b[8]).collect::<Vec<_>>();
                for (row,&value) in actual.iter().enumerate() {
                    let codes=w[row*(k/2)..(row+1)*(k/2)].iter().flat_map(|b|[b&15,b>>4]).collect::<Vec<_>>();
                    let acc=crate::tc_twin::accumulator(nv_chains(&shape),&codes,&a,&scales[row*(k/16)..(row+1)*(k/16)],&sa);
                    assert_eq!(value.to_bits(),crate::tc_twin::word(acc,shape.global,shape.activation),"rows={rows} cols={cols} second={second} tiled={tiled} row={row}");
                }
            }
        }
        // Review finding U2: a batch of both roles is refused on the host (the kernel takes the role of the first job).
        let (mut first,mut second)=(s(Op::NvProject),s(Op::NvProject));first.cols=256;second.cols=256;second.role=NV_SECOND;
        assert!(batch_shapes_agree([first,first]) && batch_shapes_agree([second,second,second]) && batch_shapes_agree([first]));
        assert!(!batch_shapes_agree([first,second]) && !batch_shapes_agree([second,second,first]) && !batch_shapes_agree(Vec::new()));
        let mut wider=first;wider.cols=512;assert!(!batch_shapes_agree([first,wider]));
        // A second matrix needs whole quarters of 64-column steps.
        let mut shape=s(Op::NvProject);shape.cols=128;shape.role=NV_SECOND;assert!(shape.lengths().is_none());
        // The role is a named word: only 0 and 1 exist, and only an NVFP4 product has a second role.
        let mut shape=s(Op::NvProject);shape.cols=256;shape.role=NV_SECOND;assert!(shape.lengths().is_some());shape.role=2;assert!(shape.lengths().is_none());
        let mut shape=s(Op::SwiGlu);assert!(shape.lengths().is_some());shape.role=NV_SECOND;assert!(shape.lengths().is_none());
        // The quantizer of a second matrix's input keeps the role word of the shape it was made from.
        for op in [Op::Quantize,Op::QuantizeRows] {let mut shape=s(op);shape.cols=256;shape.role=NV_SECOND;assert!(shape.lengths().is_some(),"{op:?}");shape.role=2;assert!(shape.lengths().is_none());}
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
    #[test]
    fn column_tiles_keep_every_nvfp4_output_bit_and_the_byte_counts() {
        // The same logical matrix in both layouts gives the same words from the scalar reference (decode and token rows)
        // and from the table row of the group body. Activations include exact zeros of both signs.
        let mut state=0x9e37_79b9_7f4a_7c15u64;
        let mut next=||{state^=state<<13;state^=state>>7;state^=state<<17;state};
        for (rows,cols) in [(128usize,64usize),(256,128),(384,1024)] {
            assert!(nv_tile_layout_admits(rows,cols));
            let w=(0..rows*cols/2).map(|_|next() as u8).collect::<Vec<_>>();
            let scales=(0..rows*cols/16).map(|_|0x28+(next()%40) as u8).collect::<Vec<_>>();
            let (tw,ts)=(nv_tile_codes(&w,rows,cols),nv_tile_scales(&scales,rows,cols));
            assert_eq!((tw.len(),ts.len()),(w.len(),scales.len()));
            let flat=Shape {op:Op::NvProject,rows:rows as u32,cols:cols as u32,block_rows:1,block_cols:16,tokens:1,global:0.75,activation:0.125,bound:0.0,role:0};
            let tiled=Shape {block_rows:NV_TILE_ROWS,..flat};
            assert!(nv_tiled(&tiled) && !nv_tiled(&flat) && !nv_tiled(&Shape {op:Op::Fp8Serial,..tiled}));
            // Every code and every scale of the in-place repack, not a sample.
            for row in 0..rows {for col in 0..cols {
                let (a,sa)=nv_code_at(&flat,row,col);let (b,sb)=nv_code_at(&tiled,row,col);
                assert_eq!((w[a]>>sa)&15,(tw[b]>>sb)&15,"{rows}/{cols}/{row}/{col}");
                assert_eq!(scales[nv_scale_at(&flat,row,col/16)],ts[nv_scale_at(&tiled,row,col/16)]);
            }}
            let quant=|seed:u64|{let mut v=Vec::new();let mut z=seed|1;
                for i in 0..cols/16*17 {z^=z<<13;z^=z>>7;z^=z<<17;
                    v.push(if i%17==16 {e4m3(0x30+(z%8) as u8)} else if z%5==0 {if z&32==0 {0.0} else {-0.0}} else {e2m1((z>>8) as u8&15)});}v};
            let x=quant(7);
            let a=reference(flat,[&x,&[],&[],&[]],&w,&scales).unwrap();let b=reference(tiled,[&x,&[],&[],&[]],&tw,&ts).unwrap();
            assert_eq!(a.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),b.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            let tokens=3;let rows_x=(0..tokens).flat_map(|t|quant(11+t as u64)).collect::<Vec<_>>();
            let (fr,tr)=(Shape {op:Op::NvRows,tokens:tokens as u32,..flat},Shape {op:Op::NvRows,tokens:tokens as u32,..tiled});
            let a=reference(fr,[&rows_x,&[],&[],&[]],&w,&scales).unwrap();let b=reference(tr,[&rows_x,&[],&[],&[]],&tw,&ts).unwrap();
            assert_eq!(a.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),b.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            let lut:[f32;LUT_WORDS]=decode_lut().try_into().unwrap();
            for unit in [0,1,tokens,rows*tokens-1] {let (mut p,mut q)=([0.0f32],[0.0f32]);
                group_run(fr,unit,&rows_x,&w,&scales,&lut,&mut p);group_run(tr,unit,&rows_x,&tw,&ts,&lut,&mut q);
                assert_eq!(p[0].to_bits(),q[0].to_bits());assert_eq!(p[0].to_bits(),a[unit].to_bits());}
        }
        assert!(!nv_tile_layout_admits(127,4096) && !nv_tile_layout_admits(1024,4160) && !nv_tile_layout_admits(1024,8192) && !nv_tile_layout_admits(0,64));
        assert!(nv_tile_layout_admits(1024,4096) && nv_tile_layout_admits(2048,2048) && nv_tile_layout_admits(512,4096) && nv_tile_layout_admits(4096,2048));
    }
    #[test]
    fn device_repack_moves_every_byte_to_the_place_of_the_host_repack() {
        // A host model of one `nv_tile_repack` block: the same stage slots, units and byte moves as the kernel.
        let mut state=0x2545_f491_4f6c_dd1du64;
        let mut next=||{state^=state<<13;state^=state>>7;state^=state<<17;state};
        for (rows,cols) in [(128usize,64usize),(256,192),(128,4096),(384,1024)] {
            let (half,blocks)=(cols/2,cols/16);
            let source=(0..rows*half).map(|_|next() as u8).collect::<Vec<_>>();
            let source_scales=(0..rows*blocks).map(|_|next() as u8).collect::<Vec<_>>();
            let (mut codes,mut scales)=(vec![0u8;source.len()],vec![0u8;source_scales.len()]);
            let (mut code_writers,mut scale_writers)=(vec![0u8;source.len()/4],vec![0u8;source_scales.len()/4]);
            let chunks=cols/NV_REPACK_COLS as usize;
            for block in 0..nv_tile_repack_blocks(rows,cols).unwrap() as usize {
                let (tile,chunk)=(block/chunks,block%chunks);let mut stage=vec![0u8;NV_REPACK_SHARED as usize];
                for piece in 0..128*4 {let (row,part)=(piece/4,piece%4);let from=(tile*128+row)*half+chunk*32+part*8;let to=nv_repack_code_slot(row)+part*8;
                    stage[to..to+8].copy_from_slice(&source[from..from+8]);}
                for row in 0..128 {let from=(tile*128+row)*blocks+chunk*4;let to=NV_REPACK_CODE_STAGE as usize+nv_repack_scale_slot(row);stage[to..to+4].copy_from_slice(&source_scales[from..from+4]);}
                for thread in 0..NV_REPACK_THREADS as usize {
                    for step in 0..8 {
                        let (column,word)=nv_repack_code_unit(thread,step);let shift=4*(column as u32&1);let base=nv_repack_code_slot(8*word)+column/2;
                        let b:[u32;8]=core::array::from_fn(|k|u32::from(stage[base+32*k]));
                        let pair=|even:u32,odd:u32|((even>>shift)&15)|(((odd>>shift)&15)<<4);
                        let value=pair(b[0],b[1])|(pair(b[2],b[3])<<8)|(pair(b[4],b[5])<<16)|(pair(b[6],b[7])<<24);
                        let at=(tile*cols+chunk*64+column)*64+word*4;codes[at..at+4].copy_from_slice(&value.to_le_bytes());code_writers[at/4]+=1;
                    }
                    let (scale_block,word)=(thread/32,thread%32);
                    let at_row=|row:usize|u32::from(stage[NV_REPACK_CODE_STAGE as usize+nv_repack_scale_slot(row)+scale_block]);
                    let value=at_row(4*word)|(at_row(4*word+1)<<8)|(at_row(4*word+2)<<16)|(at_row(4*word+3)<<24);
                    let at=(tile*blocks+chunk*4+scale_block)*128+word*4;scales[at..at+4].copy_from_slice(&value.to_le_bytes());scale_writers[at/4]+=1;
                }
            }
            assert!(code_writers.iter().all(|&n|n==1) && scale_writers.iter().all(|&n|n==1),"{rows}/{cols}: one writer a word");
            assert!(codes==nv_tile_codes(&source,rows,cols),"{rows}/{cols} codes");assert!(scales==nv_tile_scales(&source_scales,rows,cols),"{rows}/{cols} scales");
        }
        // No shared-memory bank is hit by two different words in one load of a warp (bank = word index mod 32).
        for warp in 0..4 {for step in 0..8 {for k in 0..8 {
            let mut banks=std::collections::BTreeMap::new();
            for lane in 0..32 {let (column,word)=nv_repack_code_unit(warp*32+lane,step);let index=(nv_repack_code_slot(8*word+k)+column/2)/4;
                assert!(banks.insert(index%32,index).is_none_or(|other|other==index),"code load warp={warp} step={step} k={k}");}
        }}}
        for scale_block in 0..4 {for k in 0..4 {let mut banks=std::collections::BTreeSet::new();
            for word in 0..32 {assert!(banks.insert((NV_REPACK_CODE_STAGE as usize+nv_repack_scale_slot(4*word+k)+scale_block)/4%32),"scale load");}}}
        assert!((0..128).all(|row|nv_repack_code_slot(row)%8==0 && nv_repack_code_slot(row)+32<=NV_REPACK_CODE_STAGE as usize && nv_repack_scale_slot(row)+4<=128*4));
    }
    #[test]
    fn column_tile_jobs_never_get_a_row_major_kernel() {
        let tiled=Shape {op:Op::NvProject,rows:1024,cols:4096,block_rows:NV_TILE_ROWS,block_cols:16,tokens:1,global:1.0,activation:1.0,bound:0.0,role:0};
        let job=|shape:Shape,weight:u64|Arguments {input:[256,4,4,4],weight,scales:512,output:1024,shape};
        let jobs=vec![job(tiled,4096);16];
        assert_eq!(batch_launch_geometry(&jobs,64*16,16).unwrap(),(true,8*16,NV_COLS_THREADS,nv_cols_shared(4096)));
        assert_eq!(nv_cols_shared(4096),17408+12*1152);assert!(nv_cols_shared(4096)<=33*1024 && nv_cols_shared(2048)<=33*1024);
        // An unaligned owner or too many jobs keep the scalar body of the same layout.
        assert_eq!(batch_launch_geometry(&vec![job(tiled,4100);16],1024,16).unwrap(),(false,1024,32,0));
        assert_eq!(batch_launch_geometry(&vec![job(tiled,4096);17],1088,17).unwrap(),(false,1088,32,0));
        assert!(!crate::nv_stream::projection_admitted(tiled,16,&[(4096,512);16]));
        assert!(crate::nv_stream::projection_admitted(Shape {block_rows:1,..tiled},16,&[(4096,512);16]));
        assert!(!nv_cols_admitted(&Shape {block_rows:1,..tiled},16,&[(4096,512);16]) && !nv_cols_admitted(&Shape {tokens:2,..tiled},16,&[(4096,512);16]));
    }
}
