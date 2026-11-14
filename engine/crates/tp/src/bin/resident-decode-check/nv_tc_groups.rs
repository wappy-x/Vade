//! Kernel kit: NVFP4 prompt groups on the tensor-core contract through the REAL host path (`Backend::expert_group`).
//! The word function comes from the matrix shape (`expert_ops::nv_chains`: first matrix = one chain, second matrix =
//! quarter chains). Part 1: every output word equals the CPU twin (`tc_twin`) on the serving choice (tensor-core
//! image for dense groups, decode jobs for groups of few tokens) AND with the tensor-core image switched off (decode
//! jobs or the generic body): three device paths, one word. Part 2: prompt groups of 256, 512 and 1,024 rows at the
//! real local shapes with 288 experts, timed against the bus floor of the launch.
use std::{fs,path::Path,time::Instant};
use spark_tp::Result;
use spark_gpu_probe::{driver::{Backend,DeviceVector,with_backend},expert_ops::{self,Op,Shape},tc_twin};
fn mixed(index:usize,job:usize)->u32 {
    let mut value=(index as u32).wrapping_add((job as u32+1).wrapping_mul(0x9e37_79b9));
    value=(value^(value>>16)).wrapping_mul(0x85eb_ca6b);value=(value^(value>>13)).wrapping_mul(0xc2b2_ae35);value^(value>>16)
}
fn upload<'a>(backend:&Backend<'a>,bytes:&[u8])->Result<DeviceVector<'a>> {
    backend.upload_vector(&bytes.chunks(4).map(|chunk|{let mut word=[0;4];word[..chunk.len()].copy_from_slice(chunk);f32::from_le_bytes(word)}).collect::<Vec<_>>())
}
/// One expert: its codes (one a byte, row-major) and scale codes for the twin, and its column-tile owners on the device.
struct Expert<'a> {codes:Vec<u8>,scales:Vec<u8>,weight:DeviceVector<'a>,scale:DeviceVector<'a>,shape:Shape}
fn expert<'a>(backend:&Backend<'a>,rows:u32,cols:u32,job:usize,keep:bool,chains:u32)->Result<Expert<'a>> {
    let (r,k)=(rows as usize,cols as usize);
    let bytes=(0..r*k/2).map(|index|mixed(index,job) as u8).collect::<Vec<_>>();
    let scales=(0..r*k/16).map(|index|if index%17==0 {0} else {0x28+(mixed(index,job+31)%64) as u8}).collect::<Vec<_>>();
    let weight=upload(backend,&expert_ops::nv_tile_codes(&bytes,r,k))?;let scale=upload(backend,&expert_ops::nv_tile_scales(&scales,r,k))?;
    let shape=Shape {op:Op::NvRows,rows,cols,block_rows:expert_ops::NV_TILE_ROWS,block_cols:16,tokens:1,global:0.75+(job%16) as f32/16.0,activation:0.125+(job%16) as f32/32.0,bound:0.0,role:if chains==4 {expert_ops::NV_SECOND} else {expert_ops::NV_FIRST}};
    // The twin reads one code a byte. Big timing fixtures keep only the rows the samples read.
    let codes=if keep {bytes.iter().flat_map(|b|[b&15,b>>4]).collect()} else {bytes[..k/2].iter().chain(&bytes[k/2..k]).chain(&bytes[(r-1)*k/2..]).flat_map(|b|[b&15,b>>4]).collect()};
    let scales=if keep {scales} else {scales[..k/16].iter().chain(&scales[k/16..k/8]).chain(&scales[(r-1)*k/16..]).copied().collect()};
    Ok(Expert {codes,scales,weight,scale,shape})
}
/// One quantized activation row as the quantizer writes it (17 words a block) and its codes for the twin.
fn token_row(cols:usize,token:usize,sparse:bool)->(Vec<f32>,Vec<u8>,Vec<u8>) {
    let (mut words,mut codes,mut scales)=(Vec::with_capacity(cols/16*17),Vec::with_capacity(cols),Vec::with_capacity(cols/16));
    for block in 0..cols/16 {
        for e in 0..16 {
            let h=mixed(block*16+e,token+1000);
            let code=if sparse && h%5<2 {(h>>8&8) as u8} else {(h>>4&15) as u8};
            codes.push(code);words.push(if code&8!=0 {-expert_ops::e2m1(code&7)} else {expert_ops::e2m1(code)});
        }
        let scale=if mixed(block,token+7).is_multiple_of(29) {(mixed(block,token+9)%8) as u8} else {0x30+(mixed(block,token+11)%20) as u8};
        scales.push(scale);words.push(expert_ops::e4m3(scale));
    }
    (words,codes,scales)
}
fn count(name:&str)->u64 {spark_gpu_probe::prompt_launches::line().split_whitespace().find_map(|word|word.strip_prefix(&format!("{name}="))).and_then(|value|value.parse().ok()).unwrap_or(0)}
/// Top-8 routing of `group` tokens over `experts` experts with a skew, as the kernel bench: the token list of each expert.
fn route(group:usize,experts:usize,seed:usize)->Vec<Vec<usize>> {
    let weights:Vec<f64>=(0..experts).map(|i|((i+1) as f64).powf(-0.5)).collect();let total:f64=weights.iter().sum();
    let mut lists=vec![Vec::new();experts];let mut draw=0usize;
    for token in 0..group {
        let mut chosen:Vec<usize>=Vec::new();
        while chosen.len()<8.min(experts) {
            draw+=1;let mut u=f64::from(mixed(draw,seed))/4294967296.0*total;let mut pick=experts-1;
            for (i,w) in weights.iter().enumerate() {if u<*w {pick=i;break;}u-=w;}
            if !chosen.contains(&pick) {chosen.push(pick);lists[pick].push(token);}
        }
    }
    lists
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;
    let mut receipt=String::from("NV_TC_GROUPS host_path=Backend::expert_group word_function=from_the_matrix_shape timing=complete_public_call_including_allocation_table_and_pack\n");
    with_backend(0,spark_gpu_probe::layer_ptx()?,32<<30,|backend| {
        if !backend.tc_image_present() {return Err("this build has no tensor-core image (cargo kernels --tc)".into());}
        let lut=backend.upload_vector(&expert_ops::decode_lut())?;
        // Part 1: every word against the CPU twin, both word functions, through the host path.
        for chains in [1u32,4] { for tensor_core in [true,false] {
            backend.set_nv_tc_groups(tensor_core);
            for (rows,cols) in [(128u32,256u32),(256,1024),(1024,4096),(2048,2048)] {
                // The generic body is one thread a word: keep its cases small.
                if !tensor_core && rows>256 {continue;}
                let (r,k)=(rows as usize,cols as usize);let stride=k/16*17;
                let experts=(0..5).map(|job|expert(backend,rows,cols,job,true,chains)).collect::<Result<Vec<_>>>()?;
                let mut checked=0usize;
                for (case,tokens) in [vec![8usize,3,9],vec![5,16],vec![1,1,1,1],vec![130,1,64],vec![200],vec![2,2,1,129,7]].into_iter().enumerate() {
                    let total:usize=tokens.iter().sum();
                    let rows_in:Vec<(Vec<f32>,Vec<u8>,Vec<u8>)>=(0..total).map(|t|token_row(k,t+37*case,(t+case)%3==0)).collect();
                    let input=backend.upload_vector(&rows_in.iter().flat_map(|row|row.0.iter().copied()).collect::<Vec<_>>())?;
                    let mut first=0;let mut parts=Vec::new();
                    for (part,&m) in tokens.iter().enumerate() {parts.push((Shape {tokens:m as u32,..experts[part].shape},Some((&experts[part].weight,&experts[part].scale)),first));first+=m;}
                    let before=count("fp4_cohort");
                    let output=backend.expert_group(&parts,&input,stride,&lut)?;backend.synchronize()?;
                    // Dense lists take the tensor-core image; lists of one token a part take decode jobs (the serving choice).
                    let dense=total>=2*tokens.len();
                    if tensor_core && dense && count("fp4_cohort")!=before+1 {return Err(format!("the tensor-core group kernel did not run rows={rows} cols={cols} case={case}"));}
                    if !tensor_core && count("fp4_cohort")!=before {return Err(format!("the tensor-core group kernel ran with the image switched off rows={rows} cols={cols} case={case}"));}
                    let got=backend.read_vector_values(&output)?;
                    let (mut at,mut first)=(0usize,0usize);
                    for (part,&m) in tokens.iter().enumerate() {
                        let e=&experts[part];
                        for row in 0..r {for token in 0..m {
                            let (_,codes,scales)=&rows_in[first+token];
                            let acc=tc_twin::accumulator(chains,&e.codes[row*k..][..k],codes,&e.scales[row*k/16..][..k/16],scales);
                            let want=tc_twin::word(acc,e.shape.global,e.shape.activation);
                            let have=got[at+row*m+token].to_bits();
                            if have!=want {return Err(format!("NV TC BITS DIFFER chains={chains} rows={rows} cols={cols} case={case} part={part} row={row} token={token} device={have:08x} twin={want:08x}"));}
                            checked+=1;
                        }}
                        at+=r*m;first+=m;
                    }
                }
                let line=format!("NV_TC_WORDS_PASS chains={chains} rows={rows} cols={cols} words={checked} differ=0 path={}\n",if !tensor_core {"decode jobs or generic body (tensor-core image off)"} else if chains==4 {"fp4_cohort_t64 + tc_nv_pack (dense), nv_cols jobs (few tokens)"} else {"fp4_cohort + tc_nv_pack (dense), nv_cols jobs (few tokens)"});
                print!("{line}");receipt.push_str(&line);
            }
        } }
        backend.set_nv_tc_groups(true);
        // Part 2: prompt groups of 256, 512 and 1,024 rows, 288 experts, both local shapes: today's kernel against
        // the tensor-core path, the same public call. Words: rows 0, 1 and the last of the first and last part.
        for (rows,cols,label,chains) in [(1024u32,4096u32,"gate_up",1u32),(2048,2048,"down",4)] {
            let (r,k)=(rows as usize,cols as usize);let stride=k/16*17;
            let experts=(0..288).map(|job|expert(backend,rows,cols,job,false,chains)).collect::<Result<Vec<_>>>()?;
            for group in [256usize,512,1024] {
                let lists:Vec<(usize,Vec<usize>)>=route(group,288,group).into_iter().enumerate().filter(|(_,l)|!l.is_empty()).collect();
                let routed:usize=lists.iter().map(|(_,l)|l.len()).sum();
                let mut words=Vec::with_capacity(routed*stride);let mut twin_rows=Vec::new();
                for (pi,(_,list)) in lists.iter().enumerate() {for &t in list {let (w,c,s)=token_row(k,t,t%3==0);words.extend(w);if pi==0 || pi+1==lists.len() {twin_rows.push((c,s));}}}
                let input=backend.upload_vector(&words)?;drop(words);
                let mut first=0;let mut parts=Vec::new();
                for (e,list) in &lists {parts.push((Shape {tokens:list.len() as u32,..experts[*e].shape},Some((&experts[*e].weight,&experts[*e].scale)),first));first+=list.len();}
                let time=||->Result<(f64,Vec<f32>)> {
                    let mut best=f64::MAX;let mut last=Vec::new();
                    for repeat in 0..6 {
                        let start=Instant::now();let output=backend.expert_group(&parts,&input,stride,&lut)?;backend.synchronize()?;
                        let seconds=start.elapsed().as_secs_f64();if repeat>0 {best=best.min(seconds);}
                        if repeat==5 {last=backend.read_vector_values(&output)?;}
                    }
                    Ok((best,last))
                };
                let (new,got)=time()?;
                // Sampled words of the tensor-core output against the twin.
                let (mut checked,mut at)=(0usize,0usize);let last_start=routed-lists[lists.len()-1].1.len();let _=last_start;
                let mut twin_at=0usize;
                for (pi,(e,list)) in lists.iter().enumerate() {
                    let m=list.len();
                    if pi==0 || pi+1==lists.len() {
                        let x=&experts[*e];
                        for (sample,row) in [0usize,1,r-1].into_iter().enumerate() {for token in 0..m {
                            let (codes,scales)=&twin_rows[twin_at+token];
                            let acc=tc_twin::accumulator(chains,&x.codes[sample*k..][..k],codes,&x.scales[sample*k/16..][..k/16],scales);
                            let want=tc_twin::word(acc,x.shape.global,x.shape.activation);let have=got[at+row*m+token].to_bits();
                            if have!=want {return Err(format!("NV TC BITS DIFFER {label} group={group} part={pi} row={row} token={token} device={have:08x} twin={want:08x}"));}
                            checked+=1;
                        }}
                        twin_at+=m;
                    }
                    at+=r*m;
                }
                // Bytes the product launch must move: each live expert's codes and scales once, the packed operands, the output.
                let bytes=lists.len()*(r*k/2+r*k/16)+routed*(k/2+k/16)+routed*r*4;let floor=bytes as f64/235e9;
                let line=format!("NV_TC_GROUP {label} chains={chains} rows={rows} cols={cols} group={group} parts={} routed_rows={routed} tensor_core_ms={:.3} bus_floor_ms={:.3} tensor_core_over_floor={:.2} sampled_words={checked} differ=0\n",
                    lists.len(),new*1e3,floor*1e3,new/floor);
                print!("{line}");receipt.push_str(&line);
            }
        }
        Ok(())
    })?;
    let line=format!("NV_TC_GROUPS_PASS launches: {}\n",spark_gpu_probe::prompt_launches::line());
    print!("{line}");receipt.push_str(&line);
    fs::write(out.join("RECEIPT"),receipt).map_err(|error|error.to_string())?;Ok(())
}

/// Kernel kit (feature `kernel-bench`): ONE prompt projection call of a 1,024-token group, made as `Weights::apply_experts`
/// makes it (quantize; gate product; up product on the shared quantized rows; SwiGlu. And: quantize; down product), 288
/// experts at the real local shapes, top-8 routing with the bench skew. Split by launch: `Backend::launch_split` drains the
/// stream before and after every launch, so each time is that launch alone. The whole call is also timed with one drain at
/// its end; the difference to the sum of the launches is host work (allocation, tables, uploads) and the drains.
#[cfg(feature="kernel-bench")]
pub fn split(out:&Path)->Result<()> {
    use std::collections::BTreeMap;
    fs::create_dir(out).map_err(|error|error.to_string())?;
    let mut receipt=String::from("NV_PROMPT_SPLIT host_path=Backend::expert_group,causal_operation order=Weights::apply_experts timing=wall_per_launch_drained_before_and_after\n");
    with_backend(0,spark_gpu_probe::layer_ptx()?,32<<30,|backend| {
        if !backend.tc_image_present() {return Err("this build has no tensor-core image (cargo kernels --tc)".into());}
        let lut=backend.upload_vector(&expert_ops::decode_lut())?;
        let median=|v:&mut Vec<f64>|{v.sort_by(|a,b|a.partial_cmp(b).unwrap());v[v.len()/2]};
        for (label,rows,cols,chains,matrices) in [("gate_up",1024u32,4096u32,1u32,2usize),("down",2048,2048,4,1)] { for (path,fused) in [("two_step",false),("one_pass",true)] {
            backend.set_nv_quantize_pack(fused);
            let (r,k)=(rows as usize,cols as usize);let stride=k/16*17;
            // Matrix mi of expert e is fixture job e + 288 * mi: the same input scale bits as matrix 0 (288 is a multiple of 16), so
            // gate and up share the quantized rows, as `apply_experts` does when the scale bits are equal.
            let experts=(0..matrices*288).map(|job|expert(backend,rows,cols,job,false,chains)).collect::<Result<Vec<_>>>()?;
            let lists:Vec<(usize,Vec<usize>)>=route(1024,288,1024).into_iter().enumerate().filter(|(_,l)|!l.is_empty()).collect();
            let routed:usize=lists.iter().map(|(_,l)|l.len()).sum();let most=lists.iter().map(|(_,l)|l.len()).max().unwrap_or(0);
            // FP32 activation rows, expert-major, as the gather hands them over.
            let raw:Vec<f32>=(0..routed*k).map(|i|(f64::from(mixed(i,4242)%4001)-2000.0) as f32/512.0).collect();
            let input=backend.upload_vector(&raw)?;drop(raw);
            let call=||->Result<DeviceVector<'_>> {
                let (mut q,mut p,mut first)=(Vec::new(),vec![Vec::new();matrices],0usize);
                for (e,list) in &lists {
                    q.push((Shape {op:Op::QuantizeRows,tokens:list.len() as u32,..experts[*e].shape},None,first));
                    for (mi,p) in p.iter_mut().enumerate() {let x=&experts[*e+288*mi];p.push((Shape {tokens:list.len() as u32,..x.shape},Some((&x.weight,&x.scale)),first));}
                    first+=list.len();
                }
                // The serving order of `Weights::apply_experts`: the one-pass operands when the backend takes them, else the two steps.
                let lists=p.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let (a,c)=match backend.expert_group_quantized(&lists,&input,k)? {
                    Some(mut outputs)=>{if !fused {return Err("the one-pass path ran while it was switched off".into());}let c=if matrices==2 {outputs.pop()} else {None};(outputs.pop().ok_or("no product")?,c)},
                    None=>{if fused {return Err("the group did not take the one-pass path".into());}
                        let packed=backend.expert_group(&q,&input,k,&lut)?;let a=backend.expert_group(&p[0],&packed,stride,&lut)?;
                        let c=if matrices==2 {Some(backend.expert_group(&p[1],&packed,stride,&lut)?)} else {None};(a,c)},
                };
                let Some(c)=c else {return Ok(a);};
                let s=spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::SwiGlu,heads:0,dim:0,taps:0,rows:a.len() as u32,cols:0,epsilon:0.0,bound:10.0};
                backend.causal_operation(s,[Some(&a),Some(&c),None,None],None)
            };
            drop(call()?);backend.synchronize()?;
            let mut whole=Vec::new();
            for _ in 0..9 {let start=Instant::now();let output=call()?;backend.synchronize()?;whole.push(start.elapsed().as_secs_f64()*1e3);drop(output);}
            // name -> (launches a call, the summed ms of that name in each repeat), in first-launch order.
            let mut names:Vec<String>=Vec::new();let mut times:BTreeMap<String,(usize,Vec<f64>)>=BTreeMap::new();
            for _ in 0..9 {
                backend.launch_split(true);let output=call()?;backend.synchronize()?;let log=backend.launch_split(false);drop(output);
                let mut sums:BTreeMap<String,(usize,f64)>=BTreeMap::new();
                for (name,ns) in log {if !names.contains(&name) {names.push(name.clone());}let row=sums.entry(name).or_default();row.0+=1;row.1+=ns as f64/1e6;}
                for (name,(count,ms)) in sums {let row=times.entry(name).or_insert((count,Vec::new()));if row.0!=count {return Err("the launch count of a call changed between repeats".into());}row.1.push(ms);}
            }
            let rows_ms:Vec<(String,usize,f64)>=names.iter().map(|name|{let (count,v)=times.get_mut(name).unwrap();(name.clone(),*count,median(v))}).collect();
            let sum:f64=rows_ms.iter().map(|row|row.2).sum();
            let weight_bytes=matrices*lists.len()*(r*k/2+r*k/16);
            for (name,count,ms) in &rows_ms {
                let line=format!("NV_SPLIT {label} path={path} launch={name} launches_a_call={count} ms_each={:.3} ms_a_call={ms:.3} share_of_launches={:.1}%\n",ms/ *count as f64,ms/sum*100.0);
                print!("{line}");receipt.push_str(&line);
            }
            let best=whole.iter().copied().fold(f64::MAX,f64::min);let middle=median(&mut whole);
            let line=format!("NV_SPLIT {label} path={path} rows={rows} cols={cols} chains={chains} group=1024 parts={} routed_rows={routed} most={most} sum_of_launches_ms={sum:.3} whole_call_ms_median={middle:.3} whole_call_ms_best={best:.3} weight_bytes={weight_bytes} weight_bus_floor_ms_at_235e9={:.3}\n",
                lists.len(),weight_bytes as f64/235e9*1e3);
            print!("{line}");receipt.push_str(&line);
        } }
        backend.set_nv_quantize_pack(true);
        Ok(())
    })?;
    let line=format!("NV_PROMPT_SPLIT_DONE launches: {}\n",spark_gpu_probe::prompt_launches::line());
    print!("{line}");receipt.push_str(&line);
    fs::write(out.join("RECEIPT"),receipt).map_err(|error|error.to_string())?;Ok(())
}

/// Kernel kit: `nv_quantize_pack` (the one-pass operands of a dense prompt group) on the device.
/// Part 1, BYTES: its packed operands against the CPU twin (`expert_ops::quantize_pack_rows`) AND against the two-step
/// path of main on the same device (the quantizer through `Backend::expert_group`, read back, then the pack's code map):
/// every byte, on finite rows and on hard rows (NaN, infinities, signed zeros, subnormals, saturating values), groups of
/// 1 row to 8,192 rows, parts of 1 to 1,024 rows at different input scales.
/// Part 2, WORDS: `Backend::expert_group_quantized` against the two-step calls, every output word, both word functions,
/// edge groups and the real local shapes with 288 experts; a group that is not dense must not take the path. The kept
/// operand scratch is filled with 0xA5 before every one-pass launch, so a byte the kernel did not store would show.
/// Part 3, AN IMAGE WITHOUT THE ENTRY: the same primary image with only `nv_quantize_pack` removed (an older kernel file
/// under this host): `Backend::expert_group_products`, the call `Weights::apply_experts` makes, must complete through
/// the two-step path with the words of the one-pass path, bit for bit.
pub fn quantize_pack(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;
    let mut receipt=String::from("NV_QUANTIZE_PACK kernel=nv_quantize_pack twin=expert_ops::quantize_pack_rows main_path=expert_group(QuantizeRows)+tc_nv_pack\n");
    with_backend(0,spark_gpu_probe::layer_ptx()?,32<<30,|backend| {
        if !backend.tc_image_present() {return Err("this build has no tensor-core image (cargo kernels --tc)".into());}
        let lut=backend.upload_vector(&expert_ops::decode_lut())?;
        let hard=[0.0f32,-0.0,f32::MIN_POSITIVE,-f32::MIN_POSITIVE,f32::from_bits(1),-f32::from_bits(1),1e-40,-1e-40,0.25,0.75,1.25,1.75,2.5,3.5,5.0,-0.25,-0.75,-1.25,-1.75,-2.5,-3.5,-5.0,
            6.0,-6.0,7.0,-7.0,3e38,-3e38,f32::MAX,-f32::MAX,f32::INFINITY,f32::NEG_INFINITY,f32::NAN,-f32::NAN,f32::from_bits(0x7f80_0001),f32::from_bits(0xffc0_1234)];
        let mut bytes_checked=0usize;
        // Row counts: 1, odd counts, counts that leave a partial CUDA block (rows * cols / 16 not a multiple of 128), the largest group.
        for (case,(rows,cols)) in [(1usize,16usize),(1,4096),(3,2048),(7,64),(33,256),(41,48),(1024,4096),(8192,2048)].into_iter().enumerate() {
            for kind in ["finite","hard","dedicated"] {
                // `dedicated`: every row is one class, so whole blocks are all zero, all minus zero, all subnormal, tiny
                // (a subnormal block scale), the E2M1 midpoints (ties), saturating, NaN, infinite; with extreme input scales.
                let class=|row:usize,i:usize|->f32 {let h=mixed(i,case+91);let sign=if h&1==0 {1.0f32} else {-1.0};match row%10 {
                    0=>0.0,1=>-0.0,2=>sign*f32::from_bits(1+(h>>8)%0x7f_ffff),3=>sign*((h>>8)%1024) as f32/65536.0,
                    4=>sign*[0.25f32,0.75,1.25,1.75,2.5,3.5,5.0,0.5,1.0,6.0][(h>>8) as usize%10]*0.125,5=>sign*(1e30+((h>>8)%1000) as f32*1e27),
                    6=>if (h>>8).is_multiple_of(3) {f32::NAN} else if (h>>8)%3==1 {-f32::NAN} else {f32::from_bits(0x7f80_0001+(h>>12))},
                    7=>sign*f32::INFINITY,8=>if (h>>4).is_multiple_of(16) {sign*3.0} else {0.0},_=>hard[(h>>8) as usize%hard.len()]}};
                let input:Vec<f32>=(0..rows*cols).map(|i|{let h=mixed(i,case+17);
                    if kind=="dedicated" {class(i/cols,i)} else if kind=="hard" && h.is_multiple_of(5) {hard[(h>>8) as usize%hard.len()]} else if h.is_multiple_of(37) {0.0} else if h.is_multiple_of(41) {-0.0} else {((h>>4)%8001) as f32/1000.0-4.0}}).collect();
                // Parts of 1 to 1,024 rows, cumulative, each at its own input scale.
                let (mut parts,mut activations,mut first)=(Vec::new(),Vec::with_capacity(rows),0usize);
                while first<rows {
                    let tokens=[1usize,3,2,130,1,1024,64,7][parts.len()%8].min(rows-first);
                    let activation=if kind=="dedicated" {[0.125f32,1e-30,1e30,f32::MIN_POSITIVE,65536.0,0.0037,1.0][parts.len()%7]} else {0.0625+(mixed(parts.len(),case)%16) as f32/32.0};
                    parts.push((Shape {op:Op::QuantizeRows,rows:128,cols:cols as u32,block_rows:expert_ops::NV_TILE_ROWS,block_cols:16,tokens:tokens as u32,global:1.0,activation,bound:0.0,role:expert_ops::NV_FIRST},None,first));
                    activations.extend(std::iter::repeat_n(activation,tokens));first+=tokens;
                }
                let device=backend.upload_vector(&input)?;
                let (codes,scales)=backend.nv_quantize_pack_bytes(&device,cols,&activations)?;
                let (twin_codes,twin_scales)=expert_ops::quantize_pack_rows(&input,cols,&activations)?;
                let quantized=backend.expert_group(&parts,&device,cols,&lut)?;backend.synchronize()?;
                let words=backend.read_vector_values(&quantized)?;
                let (mut main_codes,mut main_scales)=(Vec::with_capacity(codes.len()),Vec::with_capacity(scales.len()));
                for block in words.as_chunks::<17>().0 {let (lo,hi,scale)=expert_ops::pack_block(block);main_codes.extend_from_slice(&lo.to_le_bytes());main_codes.extend_from_slice(&hi.to_le_bytes());main_scales.push(scale as u8);}
                let differ=|a:&[u8],b:&[u8]|if a.len()!=b.len() {usize::MAX} else {a.iter().zip(b).filter(|(x,y)|x!=y).count()};
                let (twin_differ,main_differ)=(differ(&codes,&twin_codes).saturating_add(differ(&scales,&twin_scales)),differ(&codes,&main_codes).saturating_add(differ(&scales,&main_scales)));
                let line=format!("NV_QPACK_BYTES rows={rows} cols={cols} kind={kind} parts={} bytes={} differ_vs_cpu_twin={twin_differ} differ_vs_main_two_step_on_device={main_differ}\n",parts.len(),codes.len()+scales.len());
                print!("{line}");receipt.push_str(&line);
                if twin_differ!=0 || main_differ!=0 {
                    if let Some(at)=codes.iter().zip(&main_codes).position(|(x,y)|x!=y) {println!("NV_QPACK first code byte differing from main at {at}: one_pass={:02x} main={:02x} twin={:02x} input_block={:?}",codes[at],main_codes[at],twin_codes[at],&input[at/8*16..at/8*16+16]);}
                    return Err(format!("NV QUANTIZE PACK BYTES DIFFER rows={rows} cols={cols} kind={kind}"));
                }
                bytes_checked+=codes.len()+scales.len();
            }
        }
        let line=format!("NV_QPACK_BYTES_PASS bytes={bytes_checked} differ=0 (device one pass = CPU twin = device two-step path of main)\n");print!("{line}");receipt.push_str(&line);
        // Part 2: every output word of the group call, one-pass operands against the two steps.
        let mut words_checked=0usize;
        for (rows,cols,chains,matrices,experts_count,groups) in [
            // Matrix mi of expert e is fixture job e + experts_count * mi; a count that is a multiple of 16 gives both matrices
            // of an expert the same input scale bits (the fixture's scale is a function of job % 16), as gate and up have.
            (128u32,256u32,1u32,2usize,16usize,vec![vec![8usize,3,9],vec![5,16],vec![130,1,64],vec![200],vec![2,2,1,129,7]]),
            (256,1024,4,1,16,vec![vec![8,3,9],vec![130,1,64],vec![2,2,1,129,7]]),
            (1024,4096,1,2,288,vec![]),(2048,2048,4,1,288,vec![])] {
            let (r,k)=(rows as usize,cols as usize);let stride=k/16*17;
            let experts=(0..matrices*experts_count).map(|job|expert(backend,rows,cols,job,false,chains)).collect::<Result<Vec<_>>>()?;
            // The real shapes run one 1,024-token group routed over 288 experts; the small shapes run the edge lists.
            let groups:Vec<Vec<(usize,usize)>>=if groups.is_empty() {vec![route(1024,288,1024).into_iter().enumerate().filter(|(_,l)|!l.is_empty()).map(|(e,l)|(e,l.len())).collect()]}
                else {groups.into_iter().map(|tokens|tokens.into_iter().enumerate().collect()).collect()};
            for (case,group) in groups.iter().enumerate() { for kind in ["finite","hard"] {
                let routed:usize=group.iter().map(|g|g.1).sum();
                let raw:Vec<f32>=(0..routed*k).map(|i|{let h=mixed(i,4242+case);if kind=="hard" && h.is_multiple_of(7) {hard[(h>>8) as usize%hard.len()]} else {(f64::from(h%4001)-2000.0) as f32/512.0}}).collect();
                let input=backend.upload_vector(&raw)?;drop(raw);
                let (mut q,mut p,mut first)=(Vec::new(),vec![Vec::new();matrices],0usize);
                for &(e,tokens) in group {
                    q.push((Shape {op:Op::QuantizeRows,tokens:tokens as u32,..experts[e].shape},None,first));
                    for (mi,p) in p.iter_mut().enumerate() {let x=&experts[e+experts_count*mi];p.push((Shape {tokens:tokens as u32,..x.shape},Some((&x.weight,&x.scale)),first));}
                    first+=tokens;
                }
                let lists=p.iter().map(Vec::as_slice).collect::<Vec<_>>();
                backend.set_nv_quantize_pack(false);
                if backend.expert_group_quantized(&lists,&input,k)?.is_some() {return Err("the one-pass path ran while it was switched off".into());}
                let packed=backend.expert_group(&q,&input,k,&lut)?;
                let main:Vec<Vec<f32>>=p.iter().map(|list|{let output=backend.expert_group(list,&packed,stride,&lut)?;backend.synchronize()?;backend.read_vector_values(&output)}).collect::<Result<_>>()?;
                backend.set_nv_quantize_pack(true);
                // The two-step calls above left their operands in the kept scratch: overwrite them, so the one-pass
                // launch below must store every operand byte itself.
                backend.nv_tc_scratch_poison()?;
                let before=count("nv_quantize_pack");
                let outputs=backend.expert_group_quantized(&lists,&input,k)?.ok_or("a dense group did not take the one-pass path")?;backend.synchronize()?;
                if count("nv_quantize_pack")!=before+1 {return Err("the one-pass kernel did not run exactly once for the group".into());}
                for (mi,output) in outputs.iter().enumerate() {
                    let got=backend.read_vector_values(output)?;
                    if got.len()!=main[mi].len() || got.len()!=r*routed {return Err("the one-pass output extent differs".into());}
                    if let Some(at)=got.iter().zip(&main[mi]).position(|(x,y)|x.to_bits()!=y.to_bits()) {
                        return Err(format!("NV QUANTIZE PACK WORDS DIFFER rows={rows} cols={cols} chains={chains} case={case} kind={kind} matrix={mi} word={at} one_pass={:08x} main={:08x}",got[at].to_bits(),main[mi][at].to_bits()));
                    }
                    words_checked+=got.len();
                }
                let line=format!("NV_QPACK_WORDS rows={rows} cols={cols} chains={chains} matrices={matrices} parts={} routed_rows={routed} kind={kind} words={} differ=0\n",group.len(),matrices*r*routed);
                print!("{line}");receipt.push_str(&line);
            } }
            // A group of one token a part is not dense: it keeps the two-step path (decode jobs read the quantizer's rows).
            let sparse:Vec<_>=(0..4).map(|e|(Shape {tokens:1,..experts[e].shape},Some((&experts[e].weight,&experts[e].scale)),e)).collect();
            let input=backend.upload_vector(&vec![0.5f32;4*k])?;
            if backend.expert_group_quantized(&[&sparse],&input,k)?.is_some() {return Err("a group that is not dense took the one-pass path".into());}
            // Two lists whose parts have different input scales do not share operands: the path refuses them.
            let list=|shift:usize|(0..2).map(|e|(Shape {tokens:2,..experts[e+shift].shape},Some((&experts[e+shift].weight,&experts[e+shift].scale)),2*e)).collect::<Vec<_>>();
            let (left,right)=(list(0),list(1));
            if left.iter().zip(&right).all(|(x,y)|x.0.activation.to_bits()==y.0.activation.to_bits()) {return Err("the fixture's two lists have equal input scales".into());}
            if backend.expert_group_quantized(&[&left,&right],&input,k)?.is_some() {return Err("lists with different input scales shared one-pass operands".into());}
            if backend.expert_group_quantized(&[&left],&input,k)?.is_none() {return Err("a dense list did not take the one-pass path alone".into());}
        }
        let line=format!("NV_QPACK_WORDS_PASS words={words_checked} differ=0 (one-pass operands = two-step operands through the same product launch)\n");print!("{line}");receipt.push_str(&line);
        Ok(())
    })?;
    // Part 3: the same host, a primary image that lacks only the entry.
    let full=spark_gpu_probe::layer_ptx()?;
    let text=full.to_str().map_err(|error|error.to_string())?;
    let start=text.find(".visible .entry nv_quantize_pack(").ok_or("the primary image has no nv_quantize_pack entry")?;
    let end=start+text[start..].find("\n}\n").ok_or("the entry has no end")?+3;
    let stripped=format!("{}{}",&text[..start],&text[end..]);
    if stripped.contains(".entry nv_quantize_pack(") || text.len()-stripped.len()!=end-start {return Err("the image edit removed something else".into());}
    let stripped=std::ffi::CString::new(stripped).map_err(|error|error.to_string())?;
    let mut words:Vec<Vec<Vec<f32>>>=Vec::new();
    for (label,image,one_pass) in [("full",full,true),("without_entry",stripped.as_c_str(),false)] {
        let got=with_backend(0,image,32<<30,|backend| {
            let present=backend.require_arm_kernels(&["nv_quantize_pack"]).is_ok();
            if present!=one_pass || backend.require_arm_kernels(&["expert_group"]).is_err() {return Err(format!("image {label}: the entry is {} and expert_group must be present",if present {"present"} else {"absent"}));}
            let lut=backend.upload_vector(&expert_ops::decode_lut())?;
            let mut all=Vec::new();
            for (rows,cols,chains,matrices) in [(1024u32,4096u32,1u32,2usize),(2048,2048,4,1)] {
                let k=cols as usize;
                let experts=(0..matrices*288).map(|job|expert(backend,rows,cols,job,false,chains)).collect::<Result<Vec<_>>>()?;
                let group:Vec<(usize,usize)>=route(1024,288,1024).into_iter().enumerate().filter(|(_,l)|!l.is_empty()).map(|(e,l)|(e,l.len())).collect();
                let routed:usize=group.iter().map(|g|g.1).sum();
                let input=backend.upload_vector(&(0..routed*k).map(|i|(f64::from(mixed(i,777)%4001)-2000.0) as f32/512.0).collect::<Vec<_>>())?;
                let (mut p,mut first)=(vec![Vec::new();matrices],0usize);
                for &(e,tokens) in &group {for (mi,p) in p.iter_mut().enumerate() {let x=&experts[e+288*mi];p.push((Shape {tokens:tokens as u32,..x.shape},Some((&x.weight,&x.scale)),first));}first+=tokens;}
                let before=[count("nv_quantize_pack"),count("expert_group"),count("fp4_cohort")];
                // The call of `Weights::apply_experts`.
                let outputs=backend.expert_group_products(&p.iter().map(Vec::as_slice).collect::<Vec<_>>(),&input,k,&lut)?;backend.synchronize()?;
                let after=[count("nv_quantize_pack"),count("expert_group"),count("fp4_cohort")];
                let want=if one_pass {[before[0]+1,before[1],before[2]+matrices as u64]} else {[before[0],before[1]+1,before[2]+matrices as u64]};
                if after!=want {return Err(format!("image {label}: launches nv_quantize_pack, expert_group, fp4_cohort went {before:?} -> {after:?}, expected {want:?}"));}
                for output in &outputs {all.push(backend.read_vector_values(output)?);}
            }
            Ok(all)
        })?;
        words.push(got);
    }
    let (mut compared,mut differ)=(0usize,0usize);
    for (a,b) in words[0].iter().zip(&words[1]) {if a.len()!=b.len() {return Err("the two images gave outputs of different extents".into());}compared+=a.len();differ+=a.iter().zip(b).filter(|(x,y)|x.to_bits()!=y.to_bits()).count();}
    let line=format!("NV_QPACK_NO_ENTRY image_without_nv_quantize_pack: expert_group_products completed through the two-step path (0 nv_quantize_pack launches, 1 quantizer launch a group); outputs={} words={compared} differ_vs_full_image_one_pass={differ}\n",words[1].len());
    print!("{line}");receipt.push_str(&line);
    if differ!=0 || words[0].len()!=3 || words[1].len()!=3 {return Err("NV QUANTIZE PACK: the image without the entry gave different words".into());}
    let line=format!("NV_QUANTIZE_PACK_PASS launches: {}\n",spark_gpu_probe::prompt_launches::line());
    print!("{line}");receipt.push_str(&line);
    fs::write(out.join("RECEIPT"),receipt).map_err(|error|error.to_string())?;Ok(())
}
