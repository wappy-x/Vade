//! Index key and gate append screen at the indexer width. No checkpoint is loaded.
//! Two caches of `WIDTH` words a row append one row a step from 1,040 and 4,158 held
//! rows, as the decode Pool step does. Each append must equal a plain join in raw words,
//! allocate only at growth, and keep its spare bytes inside the funded room.
use super::*;
use spark_tp::bounded_decoder::{Compute,Device,Value,HELD_ROOM};

const WIDTH:usize=128;
const APPENDS:usize=256;
const STARTS:[usize;2]=[1040,4158];
/// The funded spare of one cache in one bank: round_up_4096(HELD_ROOM x WIDTH x 4).
const FUNDED_ROOM_BYTES:usize=65_536;

fn bits(values:&[f32])->impl Iterator<Item=u32>+'_ {values.iter().map(|v|v.to_bits())}
fn same<'b,'d>(d:&Device<'b,'d>,value:&Value<'b,'d>,want:&[f32],why:&str)->Result<()> {
    if bits(&d.read(value)?).ne(bits(want)) {return Err(format!("index append changed raw words: {why}"));}Ok(())
}
/// Raw words that a value-based copy could change: signed zero, a subnormal, NaN payloads, infinities.
fn word(cache:usize,row:usize,column:usize)->f32 {
    match (row*7+column*3+cache)%11 {
        0=>-0.0,1=>f32::from_bits(1),2=>f32::from_bits(0x7fc0_1234),3=>f32::from_bits(0xffa0_0001),4=>f32::NEG_INFINITY,
        k=>(row as f32)*0.5-(column as f32)/(k as f32),
    }
}
fn row(cache:usize,at:usize)->Vec<f32> {(0..WIDTH).map(|column|word(cache,at,column)).collect()}
fn round_page(bytes:usize)->usize {bytes.div_ceil(4096)*4096}

pub fn check(out:&Path)->Result<()> {
    fs::create_dir_all(out).map_err(|e|e.to_string())?;
    assert_eq!(round_page(HELD_ROOM*WIDTH*4),FUNDED_ROOM_BYTES);
    let room=HELD_ROOM*WIDTH;let mut lines=String::new();
    for recording in [false,true] {
        spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
            if !b.vector_pool_configured() {b.configure_vector_pool(1<<30);}
            let queue=if recording {let q=b.resident_queue()?;b.enable_resident_launches()?;Some(q)} else {None};
            let d=Device::new(b);
            let counts=||{let s=b.vector_pool_stats();(s.allocated_count,s.allocated_bytes)};
            for start in STARTS {
                // Keys are cache 0, gates are cache 1. Both start as plain uploads, like a prompt's state.
                let mut want=[0,1].map(|cache|(0..start).flat_map(|at|row(cache,at)).collect::<Vec<_>>());
                let mut current=[d.upload(&want[0])?,d.upload(&want[1])?];
                let (mut growths,mut in_place,mut max_slack)=([0usize;2],[0usize;2],0usize);
                for step in 0..APPENDS {
                    let rows=start+step;
                    for cache in 0..2 {
                        let new=d.upload(&row(cache,rows))?;
                        let before=counts();let next=d.cache_append(&current[cache],&new,rows,WIDTH,room)?;let after=counts();
                        let (buffers,bytes)=(after.0-before.0,after.1-before.1);
                        want[cache].extend(row(cache,rows));
                        same(&d,&next,&want[cache],"append")?;
                        let joined=d.join(&[&current[cache],&new])?;same(&d,&joined,&want[cache],"plain join")?;drop(joined);
                        same(&d,&current[cache],&want[cache][..rows*WIDTH],"held prefix")?;
                        // A growth moves the cache to new storage; an in-place append keeps its address.
                        if next.device_address()==current[cache].device_address() {
                            need_eq(buffers,0,"in-place append new_buffers")?;need_eq(bytes,0,"in-place append new_bytes")?;in_place[cache]+=1;
                        } else {
                            need_eq(buffers,1,"growth new_buffers")?;
                            let slack=bytes.checked_sub(round_page((rows+1)*WIDTH*4)).ok_or("growth allocated less than its rows")?;
                            if slack>FUNDED_ROOM_BYTES {return Err(format!("growth spare {slack} exceeds the funded {FUNDED_ROOM_BYTES} bytes"));}
                            max_slack=max_slack.max(slack);growths[cache]+=1;
                        }
                        current[cache]=next;
                    }
                }
                // The first append copies the plain upload with room; the next growth comes HELD_ROOM rows later.
                let expected=1+(APPENDS-1)/(HELD_ROOM+1);
                if growths!=[expected;2] || in_place!=[APPENDS-expected;2] {return Err(format!("growths {growths:?} in place {in_place:?}, want {expected} growths per cache"));}
                lines.push_str(&format!("INDEX APPEND START rows={start} width={WIDTH} appends={APPENDS} recording={recording} growths_keys={} growths_gates={} in_place_keys={} in_place_gates={} max_growth_spare_bytes={max_slack} funded_room_bytes={FUNDED_ROOM_BYTES}\n",growths[0],growths[1],in_place[0],in_place[1]));
            }
            ownership(&d)?;
            drop(d);drop(queue);b.synchronize()?;Ok(())
        })?;
    }
    lines.push_str("INDEX APPEND PASS raw_bits=true recorder_off_and_on=true plain_join_equal=true prefix=true suffix=true competing_branch=true rollback=true new_buffers_zero_between_growths=true one_buffer_per_growth=true spare_within_funded_room=true no_model=true\n");
    write(&out.join("RECEIPT"),lines.as_bytes())?;print!("{lines}");Ok(())
}
fn need_eq(have:usize,want:usize,why:&str)->Result<()> {if have!=want {return Err(format!("{why}: {have} != {want}"));}Ok(())}

/// A retained prefix, a live suffix and a competing branch at the indexer width.
fn ownership<'b,'d>(d:&Device<'b,'d>)->Result<()> {
    let start=STARTS[0];let room=HELD_ROOM*WIDTH;
    let base=(0..start).flat_map(|at|row(0,at)).collect::<Vec<_>>();let plain=d.upload(&base)?;
    let first_row=row(0,start);let first=d.cache_append(&plain,&d.upload(&first_row)?,start,WIDTH,room)?;let address=first.device_address();
    let mut want=base.clone();want.extend(&first_row);
    let second_row=row(1,start+1);let second=d.cache_append(&first,&d.upload(&second_row)?,start+1,WIDTH,room)?;
    if second.device_address()!=address {return Err("index append copied within its room".into());}
    // A live suffix of the first branch blocks a second branch from writing over it.
    let suffix=d.range(&second,(start+1)*WIDTH,WIDTH)?;drop(second);
    let alternative=row(0,start+7);let branch=d.cache_append(&first,&d.upload(&alternative)?,start+1,WIDTH,room)?;
    if branch.device_address()==address {return Err("index append overwrote a retained suffix".into());}
    same(d,&suffix,&second_row,"live suffix")?;
    let mut branch_want=want.clone();branch_want.extend(&alternative);same(d,&branch,&branch_want,"competing branch")?;
    drop((branch,suffix));
    let rollback=d.cache_append(&first,&d.upload(&alternative)?,start+1,WIDTH,room)?;
    if rollback.device_address()!=address {return Err("index append did not reuse a released tail".into());}
    same(d,&rollback,&branch_want,"rollback")?;same(d,&first,&want,"retained prefix")?;same(d,&plain,&base,"plain source")?;
    if d.cache_append(&first,&d.upload(&alternative)?,start+2,WIDTH,room).is_ok() {return Err("index append accepted a wrong extent".into());}
    Ok(())
}
