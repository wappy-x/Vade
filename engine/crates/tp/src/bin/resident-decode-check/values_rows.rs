//! Check mode `values-rows-time OUT` (GPU 0): read arm 7, `WindowValues` in a rows launch of the prompt read, alone
//! on the device. For each shape: one launch whose words must equal the host unit of the rows entry (`block_rows::unit`) bit for bit on a sample, then the
//! GPU time of one launch: 3 warm launches, then 10 timed, each from an idle device to its end; the middle value.
//! The model form: 24 query heads of 128, 3 query heads a value head; the value plane shared by the rows (mask bit 1);
//! each row its own weights over the view, its own gate and its own live count and slot (the rows of one chat).
use std::{fmt::Write as _,fs,path::Path};
use spark_gpu_probe::causal_ops::{Op,Shape};
use spark_tp::Result;
/// A step of xorshift: the values of the check only.
fn next(state:&mut u32)->u32 {*state^=*state<<13;*state^=*state>>17;*state^=*state<<5;*state}
/// Check mode `values-rows-guard OUT` (GPU 0; read arm 7): `WindowValues` in a rows launch with the value plane shared
/// (mask bit 1), the launch of the entry `resident_launch_block_values` when `block_values::stage` takes it. Each launch
/// writes into a view inside a larger buffer with `GUARD_WORDS` guard words on both sides; the guards keep their bits
/// and every word equals the host unit of the rows entry (`block_rows::unit`; NaN words count as equal NaN, their
/// payloads aside). The step words: one prompt chunk (slots one apart, a ring or a full view), random counts and slots
/// (rows that read the plane itself), rows with no count, a chunk whose last row has no count; with and without
/// infinities and NaN in the weights, the plane and the gates. Then launches the host must refuse. Run it under
/// compute-sanitizer memcheck for the memcheck receipt.
#[cfg(target_os="linux")]
pub fn guard(out:&Path)->Result<()> {
    const GUARD_WORDS:usize=64;const GUARD:u32=0x7fc0_dead;const MARK:u32=0x7fa0_beef;
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let (mut text,mut failed,mut checked)=(String::new(),0usize,0usize);
    let mut state=0x510e_527fu32;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,2<<30,|b| {
        // (heads, dim, taps, stored rows, query rows, first position of a chunk): the model form at a ring of 1,024 and
        // at a full view of 8,192 late in the read, taps 1 to 3, views below, at and above a block of distances, query
        // rows that are no multiple of 32, one stored row, and a launch whose dim takes no slice (the rows entry).
        for (heads,dim,group,view,count,start) in [(24usize,128usize,3usize,1024usize,37usize,7168usize),(24,128,3,8192,40,7000),(6,16,3,64,33,50),(4,32,2,33,40,10),(2,16,1,5,3,0),(3,48,3,2100,70,2040),(1,16,1,1100,65,1030),(2,16,2,1,1,0),(6,8,3,64,9,3)] {
            for (case,specials) in [(0usize,false),(1,false),(2,false),(3,false),(0,true),(1,true)] {
                let s=Shape {op:Op::WindowValues,heads:heads as u32,dim:dim as u32,taps:group as u32,rows:view as u32,cols:0,epsilon:0.0,bound:0.0};
                let odd=|v:u32,x:f32|if !specials {x} else {match v%53 {0=>f32::NAN,1=>f32::INFINITY,2=>f32::NEG_INFINITY,3=>-0.0,_=>x}};
                let steps:Vec<f32>=(0..count).flat_map(|row|{let v=next(&mut state) as usize;let position=start+row;let chunk=[((position+1).min(view)) as f32,(position%view) as f32];
                    match case {0=>chunk,1=>[(v%(view+1)) as f32,((v>>8)%view) as f32],2=>if row%5==1 {[0.5,0.0]} else if row%7==2 {[1.0,view as f32]} else {chunk},_=>if row+1==count || row%32==31 {[f32::NAN,0.0]} else {chunk}}}).collect();
                let weights:Vec<f32>=(0..count*heads*view).map(|_|{let v=next(&mut state);odd(v>>4,f32::from_bits(((((v>>9) as f32/4_194_304.0)-1.0)*0.01).to_bits()&0xffff_0000))}).collect();
                let plane:Vec<f32>=(0..view*(heads/group)*dim/2).map(|_|{let bits=next(&mut state);let w=(bits&0x3fff_3fff)|0x3c00_3c00|((bits&1)<<15)|((bits&2)<<30);f32::from_bits(if specials && bits>>26==0 {w|0x7f80_0000} else {w})}).collect();
                let gates:Vec<f32>=(0..count*heads*dim).map(|_|{let v=next(&mut state);odd(v>>3,((v>>8) as f32/8_388_608.0-1.0)*3.0)}).collect();
                let staged=spark_gpu_probe::block_values::stage(s,count,1<<1).is_some();
                let (_,_,launch,each)=spark_gpu_probe::block_rows::lengths(s,count,1<<1).ok_or("values rows guard: the host refuses a case shape")?;
                let mut expected=vec![f32::NAN;launch*each];
                for index in 0..launch {spark_gpu_probe::block_rows::unit(s,count,1<<1,index,[&weights,&plane,&gates,&steps],&[],&mut expected[index*each..(index+1)*each]);}
                let n=expected.len();
                let mut words=vec![f32::from_bits(GUARD);GUARD_WORDS];words.extend(std::iter::repeat_n(f32::from_bits(MARK),n));words.extend(std::iter::repeat_n(f32::from_bits(GUARD),GUARD_WORDS));
                let buffer=b.upload_vector(&words)?.into_resident_shared()?;
                let mut view_words=buffer.resident_range(GUARD_WORDS..GUARD_WORDS+n)?;
                let (w,p,g,st)=(b.upload_vector(&weights)?,b.upload_vector(&plane)?,b.upload_vector(&gates)?,b.upload_vector(&steps)?);
                b.block_rows_into(s,count,1<<1,[Some(&w),Some(&p),Some(&g),Some(&st)],None,&mut view_words)?;
                b.synchronize()?;drop(view_words);
                let got=b.read_vector_values(&buffer)?;
                let low=got[..GUARD_WORDS].iter().filter(|v|v.to_bits()!=GUARD).count();
                let high=got[GUARD_WORDS+n..].iter().filter(|v|v.to_bits()!=GUARD).count();
                // A NaN made by GPU arithmetic is 0x7fffffff, and `rn` of it carries into the sign: -0.0. The host NaN
                // (0x7fc00000) stays a NaN. Those words are counted apart; without odd inputs every word must have the
                // host bits.
                let nan_word=|a:&f32,b:&f32|b.is_nan() && (a.is_nan() || a.to_bits()==0x8000_0000);
                let nan_words=got[GUARD_WORDS..GUARD_WORDS+n].iter().zip(&expected).filter(|(a,b)|a.to_bits()!=b.to_bits() && nan_word(a,b)).count();
                let differ=got[GUARD_WORDS..GUARD_WORDS+n].iter().zip(&expected).filter(|(a,b)|a.to_bits()!=b.to_bits() && !nan_word(a,b)).count();
                let marks=got[GUARD_WORDS..GUARD_WORDS+n].iter().filter(|v|v.to_bits()==MARK).count();
                let pass=got.len()==n+2*GUARD_WORDS && low==0 && high==0 && differ==0 && marks==0 && (specials || nan_words==0);
                if !pass {failed+=1;}
                checked+=1;
                let _=writeln!(text,"VALUES-ROWS GUARD heads={heads} dim={dim} taps={group} view={view} rows={count} first_position={start} case={case} specials={specials} staged={staged} words={n} guards_changed_below={low} guards_changed_above={high} words_not_written={marks} differing_words={differ} host_nan_gpu_nan_or_minus_zero={nan_words} {}",if pass {"PASS"} else {"MISS"});
            }
        }
        // The host refuses a launch whose counts differ from block_rows::lengths, before the GPU sees it.
        let (heads,dim,group,view,count)=(24usize,128usize,3usize,64usize,5usize);
        let s=Shape {op:Op::WindowValues,heads:heads as u32,dim:dim as u32,taps:group as u32,rows:view as u32,cols:0,epsilon:0.0,bound:0.0};
        let steps:Vec<f32>=(0..count).flat_map(|row|[(row+1) as f32,row as f32]).collect();
        let w=b.upload_vector(&vec![0.25f32;count*heads*view])?;let w_short=b.upload_vector(&vec![0.25f32;count*heads*view-1])?;
        let p=b.upload_vector(&vec![1.0f32;view*(heads/group)*dim/2])?;let p_short=b.upload_vector(&vec![1.0f32;view*(heads/group)*dim/2-1])?;
        let g=b.upload_vector(&vec![0.5f32;count*heads*dim])?;let g_short=b.upload_vector(&vec![0.5f32;count*heads*dim-1])?;
        let st=b.upload_vector(&steps)?;let st_short=b.upload_vector(&steps[..2*count-1])?;
        let mut room=b.vector(count*heads*dim)?;let mut small=b.vector(count*heads*dim-1)?;
        for (label,refused) in [
            ("weights one word short",b.block_rows_into(s,count,1<<1,[Some(&w_short),Some(&p),Some(&g),Some(&st)],None,&mut room).is_err()),
            ("plane one word short",b.block_rows_into(s,count,1<<1,[Some(&w),Some(&p_short),Some(&g),Some(&st)],None,&mut room).is_err()),
            ("gates one word short",b.block_rows_into(s,count,1<<1,[Some(&w),Some(&p),Some(&g_short),Some(&st)],None,&mut room).is_err()),
            ("step words one short",b.block_rows_into(s,count,1<<1,[Some(&w),Some(&p),Some(&g),Some(&st_short)],None,&mut room).is_err()),
            ("output one word short",b.block_rows_into(s,count,1<<1,[Some(&w),Some(&p),Some(&g),Some(&st)],None,&mut small).is_err()),
            ("taps 4",b.block_rows_into(Shape {taps:4,..s},count,1<<1,[Some(&w),Some(&p),Some(&g),Some(&st)],None,&mut room).is_err()),
        ] {
            if !refused {failed+=1;}
            checked+=1;
            let _=writeln!(text,"VALUES-ROWS REFUSAL case=\"{label}\" {}",if refused {"REFUSED PASS"} else {"LAUNCHED MISS"});
        }
        b.synchronize()?;
        Ok(())
    })?;
    let _=writeln!(text,"VALUES-ROWS GUARDS {} launches={checked} failed={failed}",if failed==0 {"PASS"} else {"MISS"});
    eprint!("{text}");
    fs::write(out.join("RESULT.txt"),&text).map_err(|e|e.to_string())?;
    if failed>0 {return Err(format!("{failed} of the values rows guard checks did not pass; see RESULT.txt"));}
    fs::write(out.join("COMPLETE"),"").map_err(|e|e.to_string())
}
#[cfg(target_os="linux")]
pub fn time(out:&Path)->Result<()> {
    use std::time::Instant;
    fs::create_dir_all(out).map_err(|e|format!("{}: {e}",out.display()))?;
    let (mut text,mut failed)=(String::new(),0usize);
    let mut state=0x3c6e_f372u32;
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,6<<30,|b| {
        let (heads,dim,group)=(24usize,128usize,3usize);
        // (rows of the view, rows of the launch, first position of the launch): a sliding window of 1,024 at the
        // first step and later; a full view of 8,192 at step 1 and step 8 of an 8,192-token read (643 rows: the
        // chunk the decoder makes at that view with a key budget of 1 GiB).
        for (view,count,start) in [(1024usize,1024usize,0usize),(1024,1024,7168),(8192,1024,0),(8192,643,7168)] {
            let s=Shape {op:Op::WindowValues,heads:heads as u32,dim:dim as u32,taps:group as u32,rows:view as u32,cols:0,epsilon:0.0,bound:0.0};
            // Weights: a softmax-like row, BF16 values, zero past the live count.
            let steps:Vec<f32>=(0..count).flat_map(|row|{let position=start+row;[((position+1).min(view)) as f32,(position%view) as f32]}).collect();
            let weights:Vec<f32>=(0..count).flat_map(|row|{let live=steps[2*row] as usize;let mut state_row=state^(row as u32).wrapping_mul(0x9e37_79b9)|1;
                (0..heads*view).map(move |i|if i%view<live {f32::from_bits((((next(&mut state_row)>>9) as f32/4_194_304.0)/live as f32*2.0).to_bits()&0xffff_0000)} else {0.0}).collect::<Vec<f32>>()}).collect();
            let plane:Vec<f32>=(0..view*(heads/group)*dim/2).map(|_|{let bits=next(&mut state);f32::from_bits((bits&0x3fff_3fff)|0x3c00_3c00|((bits&1)<<15)|((bits&2)<<30))}).collect();
            let gate:Vec<f32>=(0..count*heads*dim).map(|_|((next(&mut state)>>8) as f32/8_388_608.0-1.0)*3.0).collect();
            let (_,_,launch,each)=spark_gpu_probe::block_rows::lengths(s,count,1<<1).ok_or("values rows: the host refuses a case shape")?;
            let (w,p,g,st)=(b.upload_vector(&weights)?,b.upload_vector(&plane)?,b.upload_vector(&gate)?,b.upload_vector(&steps)?);
            let mut output=b.vector(launch*each)?;
            let mut times=Vec::new();
            for round in 0..13 {
                b.synchronize()?;
                let t=Instant::now();
                b.block_rows_into(s,count,1<<1,[Some(&w),Some(&p),Some(&g),Some(&st)],None,&mut output)?;
                b.synchronize()?;
                if round>=3 {times.push(t.elapsed().as_secs_f64()*1e3);}
            }
            let got=b.read_vector_values(&output)?;
            // The host unit of the rows entry on a sample of launch units (every 61st, and the last): the whole host
            // sum at a view of 8,192 would take minutes of one CPU core.
            let mut differ=got.len().abs_diff(launch*each);let mut sampled=0usize;
            for index in (0..launch).step_by(61).chain([launch-1]) {
                let mut word=vec![f32::NAN;each];
                spark_gpu_probe::block_rows::unit(s,count,1<<1,index,[&weights,&plane,&gate,&steps],&[],&mut word);
                differ+=word.iter().zip(&got[index*each..(index+1)*each]).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();sampled+=each;
            }
            if differ>0 {failed+=1;}
            times.sort_by(f64::total_cmp);
            let weight_mb=(count*heads*view*4) as f64/1.0e6;
            let _=writeln!(text,"VALUES-ROWS-TIME rows={count} heads={heads} dim={dim} view={view} staged={} first_position={start} weight_mb={weight_mb:.1} sampled_words={sampled} differing_words={differ} {} launch_ms={:.3} least_ms={:.3} most_ms={:.3}",
                spark_gpu_probe::block_values::stage(s,count,1<<1).is_some(),if differ==0 {"PASS"} else {"MISS"},times[times.len()/2],times[0],times[times.len()-1]);
        }
        Ok(())
    })?;
    let _=writeln!(text,"VALUES-ROWS {} failed={failed}",if failed==0 {"PASS"} else {"MISS"});
    eprint!("{text}");
    fs::write(out.join("RESULT.txt"),&text).map_err(|e|e.to_string())?;
    if failed>0 {return Err(format!("{failed} of the values rows shapes did not give the host words; see RESULT.txt"));}
    fs::write(out.join("COMPLETE"),"").map_err(|e|e.to_string())
}
