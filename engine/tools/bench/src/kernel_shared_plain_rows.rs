//! Standalone plain row projection alternatives, including the one-row packed reader.
#![forbid(unsafe_code)]
use super::*;
use super::super::super::{capture_graph,sample,TIMER};
pub(in super::super::super) fn run(path:&str)->Result<()> {
    let image=std::fs::read(path).map_err(|e|e.to_string())?;let mut g=Gpu::new()?;let module=g.module(&image)?;
    let names=["plain_rows_small","plain_rows_tile","prompt_rows","project_bf16_ordered"];
    let fs=names.iter().map(|n|g.function(module,n)).collect::<Result<Vec<_>>>()?;let tm=g.module(TIMER.as_bytes())?;let timer=g.function(tm,"kb_stamp")?;
    for m in 2..=4 {let (rows,cols)=(2048usize,1536usize);let copies=(256usize<<20).div_ceil(rows*cols*4);let mut sets=Vec::new();let mut equal=[true;4];
        for copy in 0..copies {
            let values=pattern(5,rows*cols,copy as u64+51).into_iter().map(|v|f32::from_bits(v.to_bits()&0xffff0000)).collect::<Vec<_>>();
            let packed=g.upload(&values.iter().flat_map(|v|((v.to_bits()>>16) as u16).to_le_bytes()).collect::<Vec<_>>())?;
            let weight=Buffer::new(&mut g,&values)?;let input=Buffer::new(&mut g,&pattern(0,m*cols,copy as u64+75))?;
            let mut out=Vec::new();for _ in 0..4 {out.push(Buffer::new(&mut g,&vec![f32::NAN;rows*m])?);}
            let a=args(&input,&weight,&out[0],rows,m,cols);
            let run=|arm:usize|->Result<()> {let mut a=Arguments {output:out[arm].address,..a};match arm {
                0=>launch_small(&g,fs[arm],&mut a),1=>launch_tile(&g,fs[arm],&mut a),2=>launch_reference(&g,fs[arm],&mut a),
                _=>{for t in 0..m {let (mut w,mut x,mut y,mut r,mut c)=(packed,input.address+(t*cols*4) as u64,out[arm].address+(t*rows*4) as u64,rows as u32,cols as u32);
                    g.launch(fs[arm],[rows.div_ceil(128) as u32,1,1],128,0,&mut [ptr(&mut w),ptr(&mut x),ptr(&mut y),ptr(&mut r),ptr(&mut c)])?;}Ok(())}
            }};
            for arm in 0..4 {run(arm)?;}g.sync()?;let want=out[0].read(&g)?;
            for arm in 0..4 {let mut bytes=out[arm].read(&g)?;if bytes[..16]!=GUARD||bytes[bytes.len()-16..]!=GUARD {return Err("plain forms guard".into());}
                if arm==3 {let raw=bytes.clone();for r in 0..rows {for t in 0..m {let from=16+(t*rows+r)*4;let to=16+(r*m+t)*4;bytes[to..to+4].copy_from_slice(&raw[from..from+4]);}}}equal[arm]&=want==bytes;}
            sets.push((a,packed,out));
        }
        let times=g.upload(&[0;16])?;let mut samples=vec![Vec::new();4];
        {let mut graphs=Vec::new();for arm in 0..4 {graphs.push(capture_graph(&g,|| {let mut stamp=times;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;
            for (base,packed,out) in &sets {let mut a=Arguments {output:out[arm].address,..*base};match arm {
                0=>launch_small(&g,fs[arm],&mut a)?,1=>launch_tile(&g,fs[arm],&mut a)?,2=>launch_reference(&g,fs[arm],&mut a)?,
                _=>{for t in 0..m {let (mut w,mut x,mut y,mut r,mut c)=(*packed,a.input[0]+(t*cols*4) as u64,out[arm].address+(t*rows*4) as u64,rows as u32,cols as u32);
                    g.launch(fs[arm],[rows.div_ceil(128) as u32,1,1],128,0,&mut [ptr(&mut w),ptr(&mut x),ptr(&mut y),ptr(&mut r),ptr(&mut c)])?;}}
            }}stamp=times+8;g.launch(timer,[1,1,1],1,0,&mut [ptr(&mut stamp)])?;Ok(())})?);}
            for _ in 0..3 {for graph in &graphs {sample(graph,times)?;}}
            for repeat in 0..11 {for k in 0..4 {let arm=(repeat+k)%4;samples[arm].push(sample(&graphs[arm],times)?.1/copies as f64);}}
        }
        for arm in 0..4 {println!("FORMS family=plain_index_rows m={m} current=plain_rows_small arm={} us={:.3} equal={} copies={copies} clock=graph_stamps layout_cost_included={} samples={:?}",names[arm],median(samples[arm].clone()),equal[arm],arm!=3,samples[arm]);}g.clear()?;
    }Ok(())
}
