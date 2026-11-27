//! Check mode `route-rows-guard OUT` (GPU 0): read arm 2, the router pick of many rows in one rows launch (arm
//! `RouterPick`, entry `resident_launch_block_rows`), as Ruling R6h asks for a changed Inkling entry. Each launch
//! writes its output into a view inside a larger buffer with `GUARD_WORDS` guard words on both sides. The guards must
//! keep their bits, and every word must equal the host: the rows unit of each row (`block_rows::unit`, the words the
//! entry writes) and the first 2k words of the general `Op::Route` unit of that row (`latent_ops::unit`). The units are
//! compared, not `reference`, so that logits that are not finite run too.
//! Cases: 1 to 1,024 rows and `ROWS_MAX` rows, four expert forms, one pick, the logits shared by all rows (mask bit 0),
//! rows of NaN, infinities and -0.0, the expert cap, and one launch of the single-arm entry. Then the launches the host
//! must refuse (review REVARM2 finding 3). Run it under compute-sanitizer memcheck for the memcheck receipt.
use std::{fmt::Write as _,fs,path::Path};
use spark_gpu_probe::causal_ops::{Op,Shape};
use spark_tp::Result;
/// Guard words on each side of an output, and their bits: a quiet NaN that the arm does not write, and a mark for
/// the output words before the launch.
const GUARD_WORDS:usize=64;
const GUARD:u32=0x7fc0_dead;
const MARK:u32=0x7fa0_beef;
/// A spread of logits that are not BF16 values, with both signs. A value of the check only.
fn value(i:usize,seed:usize)->f32 {(((i*7919+seed*104_729)%2003) as f32-1001.0)/97.0+0.000_3}
fn shape(experts:usize,picks:u32,normalize:u32,scale:f32)->Shape {Shape {op:Op::RouterPick,heads:1,dim:1,taps:normalize,rows:experts as u32,cols:picks,epsilon:0.0,bound:scale}}
/// The words of the rows launch on the host, unit by unit as the entry writes them, and the count of rows whose
/// words equal the first 2k words of `Op::Route` of that row.
fn host(s:Shape,rows:usize,shared:u32,logits:&[f32],bias:&[f32])->Result<(Vec<f32>,usize)> {
    let (_,_,units,words)=spark_gpu_probe::block_rows::lengths(s,rows,shared).ok_or("route rows: the host refuses a case shape")?;
    let (e,k)=(s.rows as usize,s.cols as usize);
    let mut out=vec![f32::NAN;units*words];let mut equal=0;
    for row in 0..units {
        let unit=&mut out[row*words..(row+1)*words];
        spark_gpu_probe::block_rows::unit(s,rows,shared,row,[logits,&[],&[],&[]],bias,unit);
        let own=if shared&1==1 {logits} else {&logits[row*e..(row+1)*e]};
        let mut route=vec![f32::NAN;2*k+2*e+2];spark_gpu_probe::latent_ops::unit(Shape {op:Op::Route,..s},0,[own,&[],&[],&[]],bias,&mut route);
        if route[..2*k].iter().zip(unit.iter()).all(|(a,b)|a.to_bits()==b.to_bits()) {equal+=1;}
    }
    Ok((out,equal))
}
#[cfg(target_os="linux")]
pub fn run(out:&Path)->Result<()> {
    use spark_gpu_probe::{block_experts::PICK_EXPERTS_MAX,block_rows::{LIST,ROWS_MAX}};
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let (mut text,mut failed,mut checked)=(String::new(),0usize,0usize);
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,4<<30,|b| {
        // (label, shape, rows, mask, logits of one row by row number, bias, single-arm entry)
        type Logits=Box<dyn Fn(usize,usize)->Vec<f32>>;
        let spread:Logits=Box::new(|row,e|(0..e).map(|i|if row%5==4 {0.5} else {value(row*e+i,e)}).collect());
        let odd:Logits=Box::new(|row,e|(0..e).map(|i|match (row+i)%7 {0=>f32::NAN,1=>f32::INFINITY,2=>f32::NEG_INFINITY,3=>-0.0,_=>value(row*e+i,3)}).collect());
        let flat:Logits=Box::new(|_,e|vec![0.0;e]);
        let mut cases:Vec<(String,Shape,usize,u32,&Logits,Vec<f32>,bool)>=Vec::new();
        for (experts,picks,normalize,scale) in [(256usize,8u32,1u32,2.5f32),(128,8,1,1.0),(64,4,0,0.75),(2,2,1,1.0),(256,1,1,2.5)] {
            let bias:Vec<f32>=(0..experts).map(|i|if i%3==0 {0.0} else {value(i,7)*0.01}).collect();
            for rows in [1usize,2,16,63,64,65,1024] {cases.push(("spread".into(),shape(experts,picks,normalize,scale),rows,0,&spread,bias.clone(),false));}
            cases.push(("logits shared by all rows".into(),shape(experts,picks,normalize,scale),16,1,&spread,bias.clone(),false));
            cases.push(("NaN, infinities and -0.0".into(),shape(experts,picks,normalize,scale),65,0,&odd,bias.clone(),false));
            cases.push(("single-arm entry".into(),shape(experts,picks,normalize,scale),1,0,&spread,bias.clone(),true));
        }
        cases.push(("most rows".into(),shape(8,2,1,1.0),ROWS_MAX,0,&spread,(0..8).map(|i|i as f32*0.01).collect(),false));
        // The expert cap: the top three numbers tie on the largest choice.
        let mut top=vec![0.0f32;PICK_EXPERTS_MAX];for i in [PICK_EXPERTS_MAX-3,PICK_EXPERTS_MAX-2,PICK_EXPERTS_MAX-1] {top[i]=2.0;}top[1]=1.0;
        cases.push(("expert cap, ties at the top".into(),shape(PICK_EXPERTS_MAX,4,1,1.0),2,0,&flat,top,false));
        for (label,s,rows,mask,logits_of,bias,single) in &cases {
            let (e,k)=(s.rows as usize,s.cols as usize);
            let logits:Vec<f32>=if mask&1==1 {logits_of(0,e)} else {(0..*rows).flat_map(|row|logits_of(row,e)).collect()};
            let (expected,route_rows)=host(*s,*rows,*mask,&logits,bias)?;
            let n=expected.len();
            let mut words=vec![f32::from_bits(GUARD);GUARD_WORDS];words.extend(std::iter::repeat_n(f32::from_bits(MARK),n));words.extend(std::iter::repeat_n(f32::from_bits(GUARD),GUARD_WORDS));
            let buffer=b.upload_vector(&words)?.into_resident_shared()?;
            let mut view=buffer.resident_range(GUARD_WORDS..GUARD_WORDS+n)?;
            let x=b.upload_vector(&logits)?;let w=b.upload_vector(bias)?;
            if *single {b.resident_causal_into(*s,[Some(&x),None,None,None],Some(&w),&mut view)?;} else {b.block_rows_into(*s,*rows,*mask,[Some(&x),None,None,None],Some(&w),&mut view)?;}
            b.synchronize()?;drop(view);
            let got=b.read_vector_values(&buffer)?;
            let low=got[..GUARD_WORDS].iter().filter(|w|w.to_bits()!=GUARD).count();
            let high=got[GUARD_WORDS+n..].iter().filter(|w|w.to_bits()!=GUARD).count();
            let words_got=&got[GUARD_WORDS..GUARD_WORDS+n];
            // Against the host: a NaN of the GPU and a NaN of the host may carry other payload bits; those words are
            // counted apart. Every other word must be equal in its bits.
            let differ=words_got.iter().zip(&expected).filter(|(a,b)|a.to_bits()!=b.to_bits() && !(a.is_nan() && b.is_nan())).count();
            let nan_payload=words_got.iter().zip(&expected).filter(|(a,b)|a.to_bits()!=b.to_bits() && a.is_nan() && b.is_nan()).count();
            // Against today's path on the GPU: the general Op::Route of each row, one launch a row, its first 2k
            // words, bit for bit (NaN payloads included). For up to 65 rows and up to 511 experts (its ranked form).
            let mut device_rows=0usize;let mut device_checked=0usize;
            if *rows<=65 && e<=511 {
                for row in 0..*rows {
                    let own=if mask&1==1 {logits.clone()} else {logits[row*e..(row+1)*e].to_vec()};
                    let x=b.upload_vector(&own)?;
                    let route=b.causal_operation(Shape {op:Op::Route,..*s},[Some(&x),None,None,None],Some(&w))?;
                    let words=b.read_vector_values(&route)?;
                    device_checked+=1;
                    if words[..2*k].iter().zip(&words_got[row*2*k..(row+1)*2*k]).all(|(a,b)|a.to_bits()==b.to_bits()) {device_rows+=1;}
                }
            }
            let ids_below=words_got.chunks(2*k).all(|unit|unit[..k].iter().all(|&id|id>=0.0 && id.fract()==0.0 && (id as usize)<e));
            let pass=got.len()==n+2*GUARD_WORDS && low==0 && high==0 && differ==0 && route_rows==*rows && ids_below && device_rows==device_checked;
            if !pass {failed+=1;}
            checked+=1;
            let _=writeln!(text,"ROUTE-ROWS GUARD case=\"{label}\" experts={e} picks={k} normalize={} rows={rows} mask={mask} entry={} words={n} guards_changed_below={low} guards_changed_above={high} differing_words={differ} nan_payload_words={nan_payload} rows_equal_to_route={route_rows} rows_equal_to_device_route={device_rows}_of_{device_checked} ids_below_count={ids_below} {}",
                s.taps,if *single {"arms"} else {"rows"},if pass {"PASS"} else {"MISS"});
        }
        // The host refuses each of these before the GPU sees it.
        let s=shape(64,4,1,1.0);let rows=16;
        let up=|n:usize|b.upload_vector(&vec![0.25f32;n.max(1)]);
        let (full,short,bias,bias_short)=(up(rows*64)?,up(rows*64-1)?,up(64)?,up(63)?);
        let (shared_one,cap_logits,cap_bias)=(up(64)?,up(PICK_EXPERTS_MAX+1)?,up(PICK_EXPERTS_MAX+1)?);
        let mut room=b.vector(rows*8)?;let mut small=b.vector(rows*8-1)?;let mut one=b.vector(8)?;let mut cap_room=b.vector(8)?;
        let refusals:Vec<(&str,bool)>=vec![
            ("logits one word short",b.block_rows_into(s,rows,0,[Some(&short),None,None,None],Some(&bias),&mut room).is_err()),
            ("bias one word short",b.block_rows_into(s,rows,0,[Some(&full),None,None,None],Some(&bias_short),&mut room).is_err()),
            ("output one word short",b.block_rows_into(s,rows,0,[Some(&full),None,None,None],Some(&bias),&mut small).is_err()),
            ("no rows",b.block_rows_into(s,0,0,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("one row above ROWS_MAX",b.block_rows_into(s,ROWS_MAX+1,0,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("list bit",b.block_rows_into(s,rows,LIST,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("shared bit of an unused input",b.block_rows_into(s,rows,1<<1,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("shared logits of the wrong count",b.block_rows_into(s,rows,1,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("two groups",b.block_rows_into(Shape {heads:2,..s},rows,0,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("two kept groups",b.block_rows_into(Shape {dim:2,..s},rows,0,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("one expert",b.block_rows_into(Shape {rows:1,..s},1,0,[Some(&shared_one),None,None,None],Some(&bias),&mut one).is_err()),
            ("no pick",b.block_rows_into(Shape {cols:0,..s},rows,0,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("more picks than experts",b.block_rows_into(Shape {cols:65,..s},rows,0,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("taps 2",b.block_rows_into(Shape {taps:2,..s},rows,0,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("scale 0",b.block_rows_into(Shape {bound:0.0,..s},rows,0,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("scale below 0",b.block_rows_into(Shape {bound:-1.0,..s},rows,0,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("scale infinite",b.block_rows_into(Shape {bound:f32::INFINITY,..s},rows,0,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("scale NaN",b.block_rows_into(Shape {bound:f32::NAN,..s},rows,0,[Some(&full),None,None,None],Some(&bias),&mut room).is_err()),
            ("one expert above the cap",b.block_rows_into(shape(PICK_EXPERTS_MAX+1,4,1,1.0),1,0,[Some(&cap_logits),None,None,None],Some(&cap_bias),&mut cap_room).is_err()),
        ];
        for (label,refused) in refusals {
            if !refused {failed+=1;}
            checked+=1;
            let _=writeln!(text,"ROUTE-ROWS REFUSAL case=\"{label}\" {}",if refused {"REFUSED PASS"} else {"LAUNCHED MISS"});
        }
        b.synchronize()?;
        Ok(())
    })?;
    let _=writeln!(text,"ROUTE-ROWS GUARDS {} launches={checked} failed={failed}",if failed==0 {"PASS"} else {"MISS"});
    eprint!("{text}");
    fs::write(out.join("RESULT.txt"),&text).map_err(|e|e.to_string())?;
    if failed>0 {return Err(format!("{failed} of the route rows checks did not pass; see RESULT.txt"));}
    fs::write(out.join("COMPLETE"),"").map_err(|e|e.to_string())
}
