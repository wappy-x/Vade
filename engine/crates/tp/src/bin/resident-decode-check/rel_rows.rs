//! Check mode `rel-rows-time OUT` (GPU 0): read arm 4, `RelBias` in a rows launch. For each shape: one launch whose
//! words must equal the host words of the rows launch (`block_rows::reference`), bit for bit; then the GPU time of
//! one launch alone on the device: 3 warm launches, then 10 timed, each from an idle device to its end; the middle
//! value. The model form is 24 heads (one rank), a relative state of 16 values and an extent of 1,024 distances.
use std::{fmt::Write as _,fs,path::Path};
use spark_gpu_probe::causal_ops::{Op,Shape};
use spark_tp::Result;
/// A step of xorshift: the values of the check only.
fn next(state:&mut u32)->u32 {*state^=*state<<13;*state^=*state>>17;*state^=*state<<5;*state}
/// Check mode `rel-rows-guard OUT` (GPU 0; review REVARM4): `RelBias` in a rows launch (entry
/// `resident_launch_block_rows`) with 8 distances a unit (extent 1,024) and one (extents 52 and 9), the state own to
/// each row and shared by all rows (mask bit 0). Each launch writes into a view inside a larger buffer with
/// `GUARD_WORDS` guard words on both sides; the guards keep their bits and every word equals the host unit of the rows
/// entry (`block_rows::unit`; NaN words count as equal NaN, their payloads aside).
/// Then launches the host must refuse. Run it under compute-sanitizer memcheck for the memcheck receipt.
#[cfg(target_os="linux")]
pub fn guard(out:&Path)->Result<()> {
    const GUARD_WORDS:usize=64;const GUARD:u32=0x7fc0_dead;const MARK:u32=0x7fa0_beef;
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let (mut text,mut failed,mut checked)=(String::new(),0usize,0usize);
    let mut state=0x6a09_e667u32;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b| {
        for (heads,dim,extent,count) in [(24usize,16usize,1024usize,1usize),(24,16,1024,3),(24,16,1024,65),(3,2,8,7),(24,16,52,5),(2,3,9,64),(1,1,1,2)] {
            for (mask,specials) in [(0u32,false),(1,false),(0,true),(1,true)] {
                let s=Shape {op:Op::RelBias,heads:heads as u32,dim:dim as u32,taps:0,rows:0,cols:extent as u32,epsilon:0.0,bound:0.0};
                let state_rows=if mask&1==1 {1} else {count};
                let state_words:Vec<f32>=(0..state_rows*heads*dim).map(|i|match if specials {i%41} else {0} {5=>f32::NAN,9=>f32::INFINITY,13=>-0.0,_=>((next(&mut state)>>8) as f32/8_388_608.0-1.0)*4.0}).collect();
                let table:Vec<f32>=(0..dim*extent).map(|_|{let v=((next(&mut state)>>8) as f32/8_388_608.0-1.0)*2.0;f32::from_bits(v.to_bits()&0xffff_0000)}).collect();
                // The host unit of the rows entry, launch unit by launch unit: `reference` refuses a state that is not
                // finite, and these cases hold NaN and infinities on purpose.
                let (_,_,launch,each)=spark_gpu_probe::block_rows::lengths(s,count,mask).ok_or("rel rows guard: the host refuses a case shape")?;
                let mut expected=vec![f32::NAN;launch*each];
                for index in 0..launch {spark_gpu_probe::block_rows::unit(s,count,mask,index,[&state_words,&[],&[],&[]],&table,&mut expected[index*each..(index+1)*each]);}
                let n=expected.len();
                let mut words=vec![f32::from_bits(GUARD);GUARD_WORDS];words.extend(std::iter::repeat_n(f32::from_bits(MARK),n));words.extend(std::iter::repeat_n(f32::from_bits(GUARD),GUARD_WORDS));
                let buffer=b.upload_vector(&words)?.into_resident_shared()?;
                let mut view=buffer.resident_range(GUARD_WORDS..GUARD_WORDS+n)?;
                let (x,w)=(b.upload_vector(&state_words)?,b.upload_vector(&table)?);
                b.block_rows_into(s,count,mask,[Some(&x),None,None,None],Some(&w),&mut view)?;
                b.synchronize()?;drop(view);
                let got=b.read_vector_values(&buffer)?;
                let low=got[..GUARD_WORDS].iter().filter(|v|v.to_bits()!=GUARD).count();
                let high=got[GUARD_WORDS+n..].iter().filter(|v|v.to_bits()!=GUARD).count();
                // A NaN made by GPU arithmetic is 0x7fffffff, and `rn` of it carries into the sign: -0.0. The host NaN
                // (0x7fc00000) stays a NaN. Those words (host NaN, GPU NaN or -0.0) are counted apart; with a finite
                // state every word must have the host bits.
                let nan_word=|a:&f32,b:&f32|b.is_nan() && (a.is_nan() || a.to_bits()==0x8000_0000);
                let nan_words=got[GUARD_WORDS..GUARD_WORDS+n].iter().zip(&expected).filter(|(a,b)|a.to_bits()!=b.to_bits() && nan_word(a,b)).count();
                let differ=got[GUARD_WORDS..GUARD_WORDS+n].iter().zip(&expected).filter(|(a,b)|a.to_bits()!=b.to_bits() && !nan_word(a,b)).count();
                let pass=got.len()==n+2*GUARD_WORDS && low==0 && high==0 && differ==0 && (specials || nan_words==0);
                if !pass {failed+=1;}
                checked+=1;
                let part=spark_gpu_probe::block_attention::rel_part(extent);
                let _=writeln!(text,"REL-ROWS GUARD heads={heads} dim={dim} extent={extent} part={part} rows={count} mask={mask} specials={specials} words={n} guards_changed_below={low} guards_changed_above={high} differing_words={differ} host_nan_gpu_nan_or_minus_zero={nan_words} {}",if pass {"PASS"} else {"MISS"});
            }
        }
        // The host refuses a launch whose counts differ from block_rows::lengths, before the GPU sees it.
        let s=Shape {op:Op::RelBias,heads:24,dim:16,taps:0,rows:0,cols:1024,epsilon:0.0,bound:0.0};
        let count=4;
        let x=b.upload_vector(&vec![0.5f32;count*24*16])?;let short=b.upload_vector(&vec![0.5f32;count*24*16-1])?;
        let w=b.upload_vector(&vec![0.25f32;16*1024])?;let w_short=b.upload_vector(&vec![0.25f32;16*1024-1])?;
        let mut room=b.vector(count*24*1024)?;let mut small=b.vector(count*24*1024-1)?;
        for (label,refused) in [
            ("state one word short",b.block_rows_into(s,count,0,[Some(&short),None,None,None],Some(&w),&mut room).is_err()),
            ("table one word short",b.block_rows_into(s,count,0,[Some(&x),None,None,None],Some(&w_short),&mut room).is_err()),
            ("output one word short",b.block_rows_into(s,count,0,[Some(&x),None,None,None],Some(&w),&mut small).is_err()),
            ("an input the arm does not read (mask bit 1)",b.block_rows_into(s,count,2,[Some(&x),None,None,None],Some(&w),&mut room).is_err()),
            ("taps 1",b.block_rows_into(Shape {taps:1,..s},count,0,[Some(&x),None,None,None],Some(&w),&mut room).is_err()),
        ] {
            if !refused {failed+=1;}
            checked+=1;
            let _=writeln!(text,"REL-ROWS REFUSAL case=\"{label}\" {}",if refused {"REFUSED PASS"} else {"LAUNCHED MISS"});
        }
        b.synchronize()?;
        Ok(())
    })?;
    let _=writeln!(text,"REL-ROWS GUARDS {} launches={checked} failed={failed}",if failed==0 {"PASS"} else {"MISS"});
    eprint!("{text}");
    fs::write(out.join("RESULT.txt"),&text).map_err(|e|e.to_string())?;
    if failed>0 {return Err(format!("{failed} of the rel rows guard checks did not pass; see RESULT.txt"));}
    fs::write(out.join("COMPLETE"),"").map_err(|e|e.to_string())
}
#[cfg(target_os="linux")]
pub fn time(out:&Path)->Result<()> {
    use std::time::Instant;
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let (mut text,mut failed)=(String::new(),0usize);
    let mut state=0x2e1b_1a50u32;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,2<<30,|b| {
        // (heads, state width, extent, rows of the launch): the model form at 1,024 and 16 rows, an extent that is
        // no multiple of the part.
        for (heads,dim,extent,count) in [(24usize,16usize,1024usize,1024usize),(24,16,1024,16),(6,10,52,64)] {
            let s=Shape {op:Op::RelBias,heads:heads as u32,dim:dim as u32,taps:0,rows:0,cols:extent as u32,epsilon:0.0,bound:0.0};
            let state_words:Vec<f32>=(0..count*heads*dim).map(|_|((next(&mut state)>>8) as f32/8_388_608.0-1.0)*4.0).collect();
            let table:Vec<f32>=(0..dim*extent).map(|_|{let v=((next(&mut state)>>8) as f32/8_388_608.0-1.0)*2.0;f32::from_bits(v.to_bits()&0xffff_0000)}).collect();
            let expected=spark_gpu_probe::block_rows::reference(s,count,0,[&state_words,&[],&[],&[]],&table)?;
            let (x,w)=(b.upload_vector(&state_words)?,b.upload_vector(&table)?);
            let mut output=b.vector(expected.len())?;
            let mut times=Vec::new();
            for round in 0..13 {
                b.synchronize()?;
                let t=Instant::now();
                b.block_rows_into(s,count,0,[Some(&x),None,None,None],Some(&w),&mut output)?;
                b.synchronize()?;
                if round>=3 {times.push(t.elapsed().as_secs_f64()*1e3);}
            }
            let got=b.read_vector_values(&output)?;
            let differ=got.iter().zip(&expected).filter(|(a,b)|a.to_bits()!=b.to_bits()).count()+got.len().abs_diff(expected.len());
            if differ>0 {failed+=1;}
            times.sort_by(f64::total_cmp);
            let _=writeln!(text,"REL-ROWS-TIME rows={count} heads={heads} dim={dim} extent={extent} words={} differing_words={differ} {} launch_ms={:.3} least_ms={:.3} most_ms={:.3}",
                expected.len(),if differ==0 {"PASS"} else {"MISS"},times[times.len()/2],times[0],times[times.len()-1]);
        }
        Ok(())
    })?;
    let _=writeln!(text,"REL-ROWS {} failed={failed}",if failed==0 {"PASS"} else {"MISS"});
    eprint!("{text}");
    fs::write(out.join("RESULT.txt"),&text).map_err(|e|e.to_string())?;
    if failed>0 {return Err(format!("{failed} of the rel rows shapes did not give the host words; see RESULT.txt"));}
    fs::write(out.join("COMPLETE"),"").map_err(|e|e.to_string())
}
