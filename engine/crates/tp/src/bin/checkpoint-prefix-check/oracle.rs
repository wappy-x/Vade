//! Fixed primitive oracle. This does not qualify a framework's runtime math.
use super::*;
use spark_gpu_probe::{mhc_boundary,norm_boundary::NormShape};
pub fn embedding(bytes:&[u8])->Vec<f32> {bytes.as_chunks::<2>().0.iter().map(|b|f32::from_bits(u32::from(u16::from_le_bytes(*b))<<16)).collect()}
fn sum(mut lanes:[f32;32])->f32 {
    for distance in [16,8,4,2,1] {let old=lanes;for lane in 0..32 {lanes[lane]=old[lane]+old[lane^distance];}}lanes[0]
}
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
pub fn norm(s:NormShape,x:&[f32],weight:&[f32])->Vec<f32> {
    let mut partials=Vec::new();
    for chunk in x.chunks(128) {let mut lanes=[0.0;32];for lane in 0..32 {
        let mut i=lane;while i<chunk.len() {lanes[lane]+=chunk[i]*chunk[i];i+=32;}
    }partials.push(sum(lanes));}
    let mut lanes=[0.0;32];for lane in 0..32 {let mut i=lane;while i<partials.len() {lanes[lane]+=partials[i];i+=32;}}
    let scale=1.0/(sum(lanes)/s.width() as f32+s.epsilon()).sqrt();
    x.iter().zip(weight).map(|(&x,&w)|s.weight(x*scale,w)).collect()
}
pub fn prefix(row:&[u8],block:&ResidualBlockOperands)->Result<[Vec<f32>;5]> {
    let m=block.mhc();let s=m.shape();
    let embedding=embedding(row).repeat(s.streams as usize);
    let mut normalized=vec![0.0;s.wide()];let mut coefficients=vec![0.0;s.coefficients()];
    mhc_math::prepare(mhc_params::Shape {rows:s.rows,streams:s.streams,hidden:s.hidden,iterations:s.iterations,rms_eps:s.rms_eps,eps:s.eps},
        &embedding,m.weights(),m.base(),m.scale(),&mut normalized,&mut coefficients);
    let collapsed=(0..s.hidden as usize).map(|i|mhc_boundary::collapse(&embedding,&coefficients,s.streams as usize,s.hidden as usize,i)).collect::<Vec<_>>();
    let norm=norm(block.norm().shape(),&collapsed,block.norm().weight());
    let values=[embedding,normalized,coefficients,collapsed,norm];
    for v in &values {floats(v)?;}Ok(values)
}
/// Sum all owners conservatively. Count all temporary vectors as live together.
/// The file reader slots and both old and new upload staging are separate.
#[cfg(any(target_os="linux",test))]
pub fn budget(block:&ResidualBlockOperands)->Result<(usize,usize)> {
    let s=block.mhc().shape();let n=block.norm().shape().width();
    let add=|a:usize,b:usize|a.checked_add(b).ok_or_else(||"prefix budget overflow".to_owned());
    let mut words=0;let mut largest=0;
    for count in [block.mhc().weights().len(),s.coefficients(),3,n,s.streams as usize,
        s.wide(),s.wide(),s.wide(),s.coefficients(),n,n,n.div_ceil(128),n] {
        words=add(words,count)?;largest=largest.max(count);
    }
    let device=add(words.checked_mul(4).ok_or("prefix budget overflow")?,n.checked_mul(2).ok_or("prefix budget overflow")?)?;
    let transfer=largest.checked_mul(4).ok_or("prefix staging overflow")?;
    let capacity=(add(transfer,63)?/64*64).checked_next_power_of_two().ok_or("prefix staging overflow")?;
    let mut pinned=add(spark_gpu_probe::staging::UPLOAD_BYTES,spark_gpu_probe::staging::WEIGHT_ARENA_BYTES)?;
    if capacity>spark_gpu_probe::staging::UPLOAD_BYTES {pinned=add(pinned,capacity)?;}
    Ok((device,add(device,pinned)?))
}

#[cfg(target_os="linux")]
pub fn device_embedding<'d>(b:&spark_gpu_probe::driver::Backend<'d>,p:&Prefix<'_>,id:u32)->Result<spark_gpu_probe::driver::DeviceVector<'d>> {
    use spark_gpu_probe::load::{WeightRange,WeightFile,WeightDtype};
    let file=p.checkpoint.component_file(&p.embedding).map_err(|e|e.to_string())?;
    let tensor=file.header().tensors.get(&p.embedding).ok_or("embedding header is absent")?;
    let count=p.width.checked_mul(2).ok_or("embedding row size overflow")?;
    let offset=tensor.start.checked_add(u64::from(id)*count as u64).and_then(|n|n.checked_add(file.header().data_start)).ok_or("embedding offset overflow")?;
    let ranges=[WeightRange {file_offset:offset,destination:0,bytes:count,row_bytes:0,row_stride:0}];
    let arena=b.load_weight_arena(count,&[WeightFile {file:file.file(),ranges:&ranges}])?;
    let table=b.weight_matrix(&arena,0,1,p.width,WeightDtype::Bf16)?;
    // Zero addresses the retained original row. It does not change the input ID.
    let row_ids=b.upload_vector(&vec![0.0;p.block.mhc().shape().streams as usize])?;
    let mut embedded=b.vector(p.block.mhc().shape().wide())?;
    let launched=b.batch_embedding_device(&table,&row_ids,&mut embedded);
    let completed=b.synchronize();launched.and(completed)?;Ok(embedded)
}
