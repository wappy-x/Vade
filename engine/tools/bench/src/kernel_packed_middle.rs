//! The packed middle's new quantizer width: each rank packs its columns, then the byte rows join.
//! The baseline packs whole rows. The candidate packs the same blocks apart, into poisoned arrays.
use super::{Buffer,Gpu,H,QPack,Result,GUARD,Place,PLACE_TILE,bits,le,ptr,grouping,place_table,args,empty,PShape,place_case};
use std::time::Instant;

pub(super) fn bytes(g:&mut Gpu,ma:H,mb:H,started:Instant)->Result<()> {
    let (old,new)=(g.function(ma,"nv_quantize_pack")?,g.function(mb,"nv_quantize_pack")?);
    let (base,placed)=(g.function(ma,"prompt_rows")?,g.function(mb,"prompt_place")?);
    for (tokens,d) in [1usize,16,256,1024].into_iter().flat_map(|tokens|[512,1024].map(|d|(tokens,d))) {
        let case=Place {tokens,k:8,experts:288,d,parts:1};
        place_case(g,base,placed,&case,179,&[(1<<20,1<<20),(2,3)])?;
        let (mut table,rows)=grouping(&case,179);let source=Buffer::new(g,&bits(rows*d,91))?;
        let shape=PShape {op:PLACE_TILE,tokens:1,heads:0,dim:d as u32,taps:1,rows:rows as u32,cols:d as u32,first:0,slots:(rows*d) as u32,inner:empty()};
        let out=Buffer::new(g,&vec![0.0;rows*d])?;
        let run=|g:&mut Gpu,table:&[f32]|->Result<Vec<u8>> {
            let table=Buffer::new(g,&place_table(table))?;out.poison(g)?;
            let mut a=args(shape,[source.address,table.address,4,4,4],[rows*d,4*rows,0,0,0],out.address,rows*d,1);
            g.launch(placed,[rows.div_ceil(32) as u32,(d/32) as u32,1],256,0,&mut [ptr(&mut a)])?;g.sync()?;out.read(g)
        };
        let want=run(g,&table)?;
        for i in 0..3 {table.swap(i,i+3);}
        if run(g,&table)?==want {return Err(format!("PACKED MIDDLE PLACE CONTROL DID NOT FAIL tokens={tokens}"));}
        println!("PACKED MIDDLE place bytes equal tokens={tokens} local={d} parts=1 wrong_table_control=detected");g.clear()?;
    }
    let mut cases=0;
    for &(rows,local,parts) in &[(8usize,64usize,2usize),(16,128,4),(24,1024,2),(128,1024,2),(2048,1024,2),(8192,1024,2),
        (8,512,4),(16,512,4),(32,512,4),(128,512,4),(2048,512,4),(8192,512,4)] {
        for raw in [false,true] {
            cases+=1;let cols=local*parts;
            let values=if raw {bits(rows*cols,cases)} else {(0..rows*cols).map(|i|if i%37<16 {0.0} else {((i.wrapping_mul(2654435761)>>9)%4001) as f32/173.0-11.5}).collect()};
            let scales=(0..rows).map(|r|0.004+(r%11) as f32*0.0031).collect::<Vec<_>>();
            let acts=Buffer::new(g,&scales)?;
            // Every result has whole words, including the scales. Check both guards and every byte.
            let pack=|g:&mut Gpu,fun:H,input:&[f32],width:usize,poison:bool|->Result<(Vec<u8>,Vec<u8>)> {
                let source=Buffer::new(g,input)?;
                let (codes,scale)= (Buffer::new(g,&vec![0.0;rows*width/8])?,Buffer::new(g,&vec![0.0;rows*width/64])?);
                if poison {codes.poison(g)?;scale.poison(g)?;}
                let mut a=QPack {input:source.address,codes:codes.address,scales:scale.address,activations:acts.address,rows:rows as u32,cols:width as u32};
                g.launch(fun,[(rows*width/16).div_ceil(128) as u32,1,1],128,0,&mut [ptr(&mut a)])?;g.sync()?;
                let (c,s)=(codes.read(g)?,scale.read(g)?);
                for b in [&c,&s] {if b[..16]!=GUARD || b[b.len()-16..]!=GUARD {return Err("PACKED MIDDLE output guard changed".into());}}
                let before=[GUARD.as_slice(),le(input).as_slice(),GUARD.as_slice()].concat();
                if source.read(g)?!=before {return Err("PACKED MIDDLE source changed".into());}
                Ok((c[16..c.len()-16].to_vec(),s[16..s.len()-16].to_vec()))
            };
            let want=pack(g,old,&values,cols,false)?;
            let mut halves=Vec::new();
            for rank in 0..parts {
                let own=values.chunks_exact(cols).flat_map(|row|row[rank*local..(rank+1)*local].iter().copied()).collect::<Vec<_>>();
                halves.push(pack(g,new,&own,local,true)?);
            }
            let join=|halves:&[(Vec<u8>,Vec<u8>)],scale:bool| {
                let width=if scale {local/16} else {local/2};let mut joined=Vec::new();
                for row in 0..rows {for half in halves {let plane=if scale {&half.1} else {&half.0};joined.extend_from_slice(&plane[row*width..(row+1)*width]);}}
                joined
            };
            if (join(&halves,false),join(&halves,true))!=want {return Err(format!("PACKED MIDDLE BYTE FAIL rows={rows} local={local} parts={parts} raw={raw}"));}
            // The served path joins rank packets with RankRows (op 26), first over the complete code-width blocks,
            // then over the scale planes. Check those device moves, not only the host reconstruction above.
            let ranked=|g:&mut Gpu,input:&[u8],tokens:usize,width:usize|->Result<Vec<u8>> {
                let words=input.chunks_exact(4).map(|w|f32::from_bits(u32::from_le_bytes([w[0],w[1],w[2],w[3]]))).collect::<Vec<_>>();
                let source=Buffer::new(g,&words)?;let out=Buffer::new(g,&vec![0.0;words.len()])?;out.poison(g)?;
                let shape=PShape {op:26,tokens:tokens as u32,heads:0,dim:0,taps:0,rows:parts as u32,cols:width as u32,first:0,slots:0,inner:empty()};
                let mut a=args(shape,[source.address,4,4,4,4],[words.len(),0,0,0,0],out.address,words.len(),1);
                if width.is_multiple_of(4) {g.launch(g.function(mb,"prompt_permute")?,[(tokens*parts) as u32,1,1],256,0,&mut [ptr(&mut a)])?;}
                else {super::base_launch(g,g.function(mb,"prompt_rows")?,&mut a)?;}
                g.sync()?;let got=out.read(g)?;
                if got[..16]!=GUARD || got[got.len()-16..]!=GUARD {return Err("PACKED MIDDLE rank row guard changed".into());}
                Ok(got[16..got.len()-16].to_vec())
            };
            let exchange=halves.iter().flat_map(|half|half.0.iter().chain(&half.1).copied()).collect::<Vec<_>>();
            let mut device=ranked(g,&exchange,rows*9/8,local/8)?;
            let scales_apart=halves.iter().flat_map(|half|half.1.iter().copied()).collect::<Vec<_>>();
            let joined_scales=ranked(g,&scales_apart,rows,local/64)?;
            device[rows*cols/2..].copy_from_slice(&joined_scales);
            if device!=[want.0.as_slice(),want.1.as_slice()].concat() {return Err(format!("PACKED MIDDLE DEVICE JOIN FAIL rows={rows} local={local} raw={raw}"));}
            // Wrong rank order must fail. It catches a join that loses ownership information.
            halves.swap(0,1);
            if (join(&halves,false),join(&halves,true))==want {return Err(format!("PACKED MIDDLE CONTROL DID NOT FAIL rows={rows} local={local} raw={raw}"));}
            if acts.read(g)?!=[GUARD.as_slice(),le(&scales).as_slice(),GUARD.as_slice()].concat() {return Err("PACKED MIDDLE scales changed".into());}
            println!("PACKED MIDDLE bytes equal rows={rows} local={local} parts={parts} raw={raw} code_bytes={} scale_bytes={} wrong_rank_control=detected",want.0.len(),want.1.len());
            g.clear()?;
        }
    }
    println!("PACKED MIDDLE BYTES PASS cases={cases} controls={cases} elapsed_s={:.1}",started.elapsed().as_secs_f64());Ok(())
}
