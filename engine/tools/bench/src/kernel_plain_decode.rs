//! Recorded plain projections of one decode row, timed on the device stream.
use super::*;
use super::super::{capture_graph,sample,TIMER};

#[derive(Clone,Copy)]
struct Call { entry:H, kind:u32, geometry:(u32,u32) }

fn cycles(g:&mut Gpu,calls:&[Call],tables:&[Vec<u64>])->Result<Vec<f64>> {
    let module=g.module(TIMER.as_bytes())?;
    let timer=g.function(module,"kb_stamp")?;
    let ticks=g.upload(&[0;16])?;
    let graphs=calls.iter().zip(tables).map(|(call,inputs)|capture_graph(g,|| {
        let mut at=ticks;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut at)])?;
        for &input in inputs {launch_replay_t(g,call.entry,input,call.geometry)?;}
        at+=8;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut at)])
    })).collect::<Result<Vec<_>>>()?;
    for _ in 0..3 {for graph in &graphs {sample(graph,ticks)?;}}
    let mut times=vec![Vec::new();graphs.len()];
    for repeat in 0..9 {for k in 0..graphs.len() {
        let i=(k+repeat)%graphs.len();
        times[i].push(sample(&graphs[i],ticks)?.1/tables[i].len() as f64);
    }}
    Ok(times.into_iter().map(median).collect())
}

fn cpu(weights:&[f32],x:&[f32],round:bool)->Vec<f32> {
    weights.chunks_exact(x.len()).map(|row| {
        let value=row.iter().zip(x).fold(0.0f32,|sum,(w,x)|sum+w*x);
        if !round {return value;}
        let word=value.to_bits();
        f32::from_bits(word.wrapping_add(0x7fff+((word>>16)&1))&0xffff_0000)
    }).collect()
}

fn guarded(words:&[f32])->Vec<u8> { [GUARD.as_slice(),&le(words),GUARD.as_slice()].concat() }

fn single(g:&mut Gpu,(base,packed):(H,H),rows:usize,cols:usize)->Result<()> {
    let (tiles,threads)=plain_geo(rows,cols);
    let calls=[
        Call {entry:base,kind:3,geometry:(tiles,threads)},
        Call {entry:base,kind:3,geometry:(rows.div_ceil(32) as u32,32)},
        Call {entry:base,kind:3,geometry:(tiles*2,threads)},
        Call {entry:packed,kind:26,geometry:head_geo(rows)},
    ];
    let mut commands=vec![Vec::new();calls.len()];
    let copies=(192usize<<20).div_ceil(rows*cols*4).clamp(4,256);
    for i in 0..copies {
        let (narrow,wide)=bf16_patterns(&pattern([0,2,5,6,8][i%5],rows*cols,41+i as u64));
        let x=pattern([0,2,5,7,8][i%5],cols,401+i as u64);
        let input=Buffer::new(g,&x)?;
        let wide_address=Buffer::new(g,&wide)?.address;
        let packed_address=upload16(g,&narrow)?;
        let out=Buffer::new(g,&vec![f32::NAN;rows])?;
        let expected=guarded(&cpu(&wide,&x,false));
        for (call,tables) in calls.iter().zip(&mut commands) {
            let weights=if call.kind==3 {wide_address} else {packed_address};
            let payload=[weights,input.address,out.address,rows as u64,cols as u64]
                .into_iter().flat_map(u64::to_le_bytes).collect::<Vec<_>>();
            let table=g.upload(&command(call.kind,&payload))?;
            out.poison(g)?;
            launch_replay_t(g,call.entry,table,call.geometry)?;g.sync()?;
            if out.read(g)?!=expected {return Err(format!("PLAIN DECODE word difference {rows}x{cols}, arm {}, fixture {i}",call.kind));}
            tables.push(table);
        }
    }
    let time=cycles(g,&calls,&commands)?;
    println!("PLAIN DECODE rows={rows} cols={cols} fp32_us={:.3} fp32_warp_us={:.3} fp32_double_grid_us={:.3} bf16_us={:.3} cpu_equal=yes cases={copies}",time[0],time[1],time[2],time[3]);
    g.clear()
}

fn index_batch(g:&mut Gpu,(base,fold):(H,H),rows:[usize;3])->Result<()> {
    const COLS:usize=4096;
    let calls=[
        Call {entry:base,kind:17,geometry:(rows.iter().map(|n|n.div_ceil(128) as u32).sum(),128)},
        Call {entry:fold,kind:35,geometry:(rows.iter().map(|n|n.div_ceil(4) as u32).sum(),160)},
    ];
    let copies=(192usize<<20).div_ceil(rows.iter().sum::<usize>()*COLS*4).clamp(4,256);
    let mut commands=vec![Vec::new();calls.len()];
    for i in 0..copies {
        let x=pattern([0,2,5,7,8][i%5],COLS,560+i as u64);
        let input=Buffer::new(g,&x)?;
        let out=Buffer::new(g,&vec![f32::NAN;rows.iter().sum()])?;
        let mut batches=[Batch {round:1,plain:7,..Default::default()};2];
        let mut expected=Vec::new();
        for (j,n) in rows.into_iter().enumerate() {
            let (narrow,wide)=bf16_patterns(&pattern([0,2,5,6,8][i%5],n*COLS,23+i as u64*3+j as u64));
            let weight=Buffer::new(g,&wide)?.address;
            let packed=upload16(g,&narrow)?;
            let output=out.address+(expected.len()*4) as u64;
            expected.extend(cpu(&wide,&x,true));
            for (arm,batch) in batches.iter_mut().enumerate() {
                let (weights,scales,blocks)=if arm==0 {(weight,4,n.div_ceil(128))} else {(packed,2,n.div_ceil(4))};
                batch.layer.projections[j]=MxProjection {weights,scales,input:input.address,output,rows:n as u32,
                    cols:COLS as u32,tokens:1,input_bf16:0,round:0,mode:0};
                batch.layer.ends[j+1]=batch.layer.ends[j]+blocks as u32;
                batch.layer.count+=1;
            }
        }
        for ((call,batch),tables) in calls.iter().zip(batches).zip(&mut commands) {
            let table=g.upload(&command(call.kind,&batch_bytes(&batch)))?;
            out.poison(g)?;
            launch_replay_t(g,call.entry,table,call.geometry)?;g.sync()?;
            if out.read(g)?!=guarded(&expected) {return Err(format!("INDEX BATCH word difference, rows {rows:?}, arm {}, fixture {i}",call.kind));}
            tables.push(table);
        }
    }
    let time=cycles(g,&calls,&commands)?;
    println!("INDEX BATCH rows={rows:?} cols={COLS} fp32_us={:.3} bf16_fold_us={:.3} cpu_equal=yes cases={copies}",time[0],time[1]);
    g.clear()
}

pub(super) fn run(g:&mut Gpu,(old,new):(H,H))->Result<()> {
    let base=g.function(old,"resident_launch_plain")?;
    let packed=g.function(new,"resident_launch_plain_bf16")?;
    let mixed=g.function(old,"resident_launch_mixed")?;
    let fold=g.function(new,"resident_launch_plain_batch")?;
    index_batch(g,(mixed,fold),[64,64,16])?;
    index_batch(g,(mixed,fold),[32,32,8])?;
    for (rows,cols) in [(8,4096),(16,4096),(32,4096),(64,4096),(72,4096),(144,4096),
        (2048,128),(4096,128),(1024,1536),(2048,1536)] {
        single(g,(base,packed),rows,cols)?;
    }
    Ok(())
}
