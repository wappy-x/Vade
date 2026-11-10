//! Memo slab and resident scatter screen. No checkpoint is loaded.
//! The scatter copies raw words to disjoint destinations and refuses overlaps before any
//! write. The memo copies kept rows into its own funded slots, never rewrites a slot that a
//! memo entry or a live value reads, reuses released slots with no new buffer, keeps its
//! chunks within its limit, and skips rows it cannot fund.
use super::*;
use spark_tp::bounded_decoder::{Compute,Device,Value,SLAB_ROWS};

fn bits(values:&[f32])->Vec<u32> {values.iter().map(|v|v.to_bits()).collect()}
/// Raw words a value-based copy could change: signed zero, a subnormal, NaN payloads, infinities.
fn word(id:usize,at:usize)->f32 {
    match (id*13+at*7)%10 {0=>-0.0,1=>f32::from_bits(1),2=>f32::from_bits(0x7fc0_4321),3=>f32::from_bits(0xff80_0001),4=>f32::INFINITY,k=>(id as f32)-(at as f32)/(k as f32)}
}
fn row(id:usize,width:usize)->Vec<f32> {(0..width).map(|at|word(id,at)).collect()}
fn same<'b,'d>(d:&Device<'b,'d>,value:&Value<'b,'d>,want:&[f32],why:&str)->Result<()> {
    if bits(&d.read(value)?)!=bits(want) {return Err(format!("memo slab changed raw words: {why}"));}Ok(())
}

fn scatter(b:&spark_gpu_probe::driver::Backend<'_>)->Result<()> {
    let width=1000;let mut table=None;
    let source_words=(0..3).flat_map(|id|row(id,width)).collect::<Vec<_>>();let fill=row(9,4*width);
    let source=b.upload_vector(&source_words)?.into_resident_shared()?;let target=b.upload_vector(&fill)?.into_resident_shared()?;
    let read=|v:&spark_gpu_probe::driver::DeviceVector<'_>|->Result<Vec<u32>> {Ok(bits(&b.read_vector_values(v)?))};
    // Rows 0 and 2 land in target rows 3 and 1; rows 0 and 2 of the target keep their words.
    b.resident_scatter(&[(&source,0),(&source,2*width)],&mut [target.resident_range(3*width..4*width)?,target.resident_range(width..2*width)?],&mut table)?;
    let mut want=fill.clone();want[3*width..].copy_from_slice(&source_words[..width]);want[width..2*width].copy_from_slice(&source_words[2*width..]);
    if read(&target)?!=bits(&want) {return Err("resident scatter changed raw words".into());}
    // Every refusal happens before any write.
    let refusals:[(&str,Result<()>);4]=[
        ("overlapping destinations",b.resident_scatter(&[(&source,0),(&source,width)],&mut [target.resident_range(0..width)?,target.resident_range(width/2..width/2+width)?],&mut table)),
        ("destination over a source row",b.resident_scatter(&[(&source,0)],&mut [source.resident_range(width/2..width/2+width)?],&mut table)),
        ("source row past its vector",b.resident_scatter(&[(&source,2*width+1)],&mut [target.resident_range(0..width)?],&mut table)),
        ("row count differs",b.resident_scatter(&[(&source,0),(&source,width)],&mut [target.resident_range(0..width)?],&mut table)),
    ];
    for (why,result) in refusals {if result.is_ok() {return Err(format!("resident scatter accepted {why}"));}}
    if read(&target)?!=bits(&want) || read(&source)?!=bits(&source_words) {return Err("a refused scatter wrote words".into());}
    b.resident_scatter(&[],&mut [],&mut table)?;Ok(())
}

fn slab<'b,'d>(b:&'b spark_gpu_probe::driver::Backend<'d>,lines:&mut String,recording:bool)->Result<()> {
    let d=Device::new(b);let width=32768usize;let latent=512usize;
    let chunk_bytes=SLAB_ROWS*width*4;let limit=2*chunk_bytes;d.set_projection_memo_limit(limit)?;
    let counts=||{let s=b.vector_pool_stats();(s.allocated_count,s.allocated_bytes)};
    let next_id=std::cell::Cell::new(0usize);let mut inputs:Vec<(Value<'b,'d>,usize)>=Vec::new();
    // Remember `rows` new rows as one batch, the way ExpandKv keeps one group.
    let remember=|d:&Device<'b,'d>,rows:usize,inputs:&mut Vec<(Value<'b,'d>,usize)>|->Result<(usize,usize)> {
        let mut batch=Vec::new();
        for _ in 0..rows {let id=next_id.get();next_id.set(id+1);batch.push((d.upload(&row(id+50_000,latent))?,d.upload(&row(id,width))?,id));}
        let before=counts();
        let kept=batch.iter().map(|(x,output,_)|(x,output)).collect::<Vec<_>>();d.remember_projections("kv",true,&kept)?;
        let after=counts();for (x,_,id) in batch {inputs.push((x,id));}Ok((after.0-before.0,after.1-before.1))
    };
    let hit=|d:&Device<'b,'d>,x:&Value<'b,'d>|d.projection_origin("kv",x,true);
    // The first batch funds one chunk. The row read back is the projection output's words.
    let (first_buffers,first_bytes)=remember(&d,8,&mut inputs)?;
    if d.projection_slab_bytes()!=chunk_bytes {return Err(format!("first batch slab bytes {} != one chunk",d.projection_slab_bytes()));}
    for (x,id) in &inputs {same(&d,&hit(&d,x)?.ok_or("a kept row was not found")?,&row(*id,width),"kept row")?;}
    // A memo hit's prefix never grows in place: an append to it copies, and the memo row
    // keeps its words. The parent hit is dropped first, so the prefix is the only live view.
    {let parent=hit(&d,&inputs[1].0)?.ok_or("append row was not found")?;let prefix=d.range(&parent,0,1024)?;drop(parent);
        let suffix=d.upload(&row(70_000,1024))?;let grown=d.cache_append(&prefix,&suffix,1,1024,1024)?;
        if grown.device_address()==prefix.device_address() {return Err("a memo hit prefix was appended in place".into());}
        let mut want=row(inputs[1].1,width)[..1024].to_vec();want.extend(row(70_000,1024));same(&d,&grown,&want,"append to a memo prefix")?;
        drop((grown,prefix,suffix));same(&d,&hit(&d,&inputs[1].0)?.ok_or("append row left the memo")?,&row(inputs[1].1,width),"memo row after an append to its prefix")?;}
    // A live range of a hit, with the hit itself dropped, pins its slot through every later eviction.
    let part={let parent=hit(&d,&inputs[0].0)?.ok_or("pinned row was not found")?;d.range(&parent,5,7)?};let pinned_row=row(inputs[0].1,width);
    let mut steady=(0usize,0usize);let mut batches=0;
    while next_id.get()<20*8 {let (n,bytes)=remember(&d,8,&mut inputs)?;batches+=1;
        if d.projection_slab_bytes()>limit {return Err("memo slab grew past its limit".into());}
        if d.projection_slab_bytes()==limit && batches>16 {steady.0+=n;steady.1+=bytes;}}
    if d.projection_slab_bytes()!=limit {return Err(format!("slab bytes {} != limit {limit}",d.projection_slab_bytes()));}
    if steady!=(0,0) {return Err(format!("released slots were not reused: {steady:?} new buffers and bytes"));}
    if hit(&d,&inputs[0].0)?.is_some() {return Err("the oldest batch was not evicted".into());}
    same(&d,&part,&pinned_row[5..12],"pinned range after its entry left and its slot's neighbours were reused")?;
    drop(part);
    let kept_now=inputs.iter().filter_map(|(x,id)|hit(&d,x).transpose().map(|v|v.map(|v|(v,*id)))).collect::<Result<Vec<_>>>()?;
    for (v,id) in &kept_now {same(&d,v,&row(*id,width),"kept row after reuse")?;}
    // Every slot in use: the next rows are not kept and no chunk past the limit is made.
    let pinned_count=kept_now.len();let before=inputs.len();
    let (n,bytes)=remember(&d,16,&mut inputs)?;
    let fresh=inputs[before..].iter().map(|(x,id)|hit(&d,x).map(|v|v.map(|v|(v,*id)))).collect::<Result<Vec<_>>>()?;
    let kept_fresh=fresh.iter().flatten().count();
    // At most one new buffer: the scatter's descriptor table when more rows fit than before.
    if n>1 || bytes>4096 || d.projection_slab_bytes()!=limit {return Err(format!("a full slab allocated {n} buffers, {bytes} bytes"));}
    if kept_fresh+pinned_count>SLAB_ROWS*2 || kept_fresh==16 {return Err(format!("full slab kept {kept_fresh} of 16 rows with {pinned_count} pinned"));}
    for (v,id) in fresh.iter().flatten() {same(&d,v,&row(*id,width),"row kept by a full slab")?;}
    for (v,id) in &kept_now {same(&d,v,&row(*id,width),"pinned row while the slab was full")?;}
    if d.set_projection_memo_limit(chunk_bytes).is_ok() {return Err("memo limit was lowered below its slab".into());}
    drop((kept_now,fresh));d.clear_projection_memo()?;
    if d.projection_slab_bytes()!=0 {return Err("cleared memo kept its slab".into());}
    lines.push_str(&format!("MEMO SLAB recording={recording} width={width} chunk_bytes={chunk_bytes} limit={limit} first_new_buffers={first_buffers} first_new_bytes={first_bytes} steady_new_buffers={} steady_new_bytes={} pinned={pinned_count} full_slab_kept={kept_fresh}_of_16\n",steady.0,steady.1));
    drop(d);padded(b,lines,recording)
}

/// A row width whose chunk is not page aligned: the slab charges the allocator's whole
/// pages, and the bytes the allocator really adds equal the charge.
fn padded<'b,'d>(b:&'b spark_gpu_probe::driver::Backend<'d>,lines:&mut String,recording:bool)->Result<()> {
    let width=17usize;let logical=SLAB_ROWS*width*4;let charged=8192usize;
    let counts=||{let s=b.vector_pool_stats();(s.allocated_count,s.allocated_bytes)};
    let d=Device::new(b);
    // Warm the shared copy table with a two-part join, so only the chunk is new below.
    drop(d.join(&[&d.upload(&row(1,width))?,&d.upload(&row(2,width))?])?);
    let x=d.upload(&row(80_000,8))?;let output=d.upload(&row(3,width))?;
    // A limit of the logical chunk bytes does not fund the padded chunk: nothing is kept or allocated.
    d.set_projection_memo_limit(logical)?;
    let before=counts();d.remember_projections("kv",true,&[(&x,&output)])?;let after=counts();
    if after!=before || d.projection_slab_bytes()!=0 || d.projection_origin("kv",&x,true)?.is_some() {return Err(format!("a {logical}-byte limit kept a padded chunk: {:?}",(after.0-before.0,after.1-before.1)));}
    d.set_projection_memo_limit(charged)?;
    let before=counts();d.remember_projections("kv",true,&[(&x,&output)])?;let after=counts();
    let (buffers,bytes)=(after.0-before.0,after.1-before.1);
    if d.projection_slab_bytes()!=charged || buffers!=1 || bytes!=charged {return Err(format!("padded chunk charged {} but allocated {buffers} buffers, {bytes} bytes",d.projection_slab_bytes()));}
    same(&d,&d.projection_origin("kv",&x,true)?.ok_or("padded row was not kept")?,&row(3,width),"padded row")?;
    lines.push_str(&format!("MEMO SLAB PADDED recording={recording} width={width} logical_chunk_bytes={logical} charged_chunk_bytes={charged} allocated_new_bytes={bytes} logical_limit_kept=0\n"));
    Ok(())
}

pub fn check(out:&Path)->Result<()> {
    fs::create_dir_all(out).map_err(|e|e.to_string())?;let mut lines=String::new();
    for recording in [false,true] {
        spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
            if !b.vector_pool_configured() {b.configure_vector_pool(1<<30);}
            let queue=if recording {let q=b.resident_queue()?;b.enable_resident_launches()?;Some(q)} else {None};
            scatter(b)?;slab(b,&mut lines,recording)?;
            drop(queue);b.synchronize()?;Ok(())
        })?;
    }
    lines.push_str("MEMO SLAB PASS raw_bits=true recorder_off_and_on=true scatter_refusals_before_write=true append_to_memo_prefix_copies=true range_only_pin_kept=true padded_chunk_charged=true limit_below_slab_refused=true released_slots_reused=true steady_new_buffers_zero=true chunks_within_limit=true full_slab_skips_rows=true cleared=true no_model=true\n");
    write(&out.join("RECEIPT"),lines.as_bytes())?;print!("{lines}");Ok(())
}
