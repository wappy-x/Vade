//! Kernel kit, plain BF16 format: the absorbed latent attention on the tensor cores (`gpu/src/absorb_tc.rs`), the four
//! fold operations (query, scores, values, project) through the public calls of the served path.
//!
//! Part 1, the twin. Device words against the CPU twin (`prompt_ops::reference`, which is `tc_twin_bf16::fold` and the
//! operation's last step): every word of five small shapes (short last steps, odd widths, more than one block and more
//! than one staged chunk), and sampled words of the model's shape. Operand classes: six BF16 classes, general FP32
//! words (the low half of an operand is never read), rare infinities and NaNs (NaN places must agree), shares in
//! [0, 1), and crossed pairs of classes (`CLASSES`).
//! Part 2, a control: the device words against the OLD word function (32 strips of FP32 sums) must differ.
//! Part 3, alone = group (rule 2). A token row alone (one row a call) against the same row in groups of 2 to 1,040
//! rows at three placements: every word of the four planes.
//! Part 4, step form = prompt form (rule 3). The decode step's kernels on the gathered rows of one token against the
//! same token's row of the prompt form: every word of the four planes.
#![forbid(unsafe_code)]
use std::{fs,path::Path};
use spark_tp::Result;
use spark_gpu_probe::{driver::{Backend,DeviceVector,with_backend},prompt_ops::{self,Op,Shape},causal_ops::{Op as StepOp,Shape as StepShape},latent_absorb_rows};
fn mixed(index:usize,salt:usize)->u32 {
    let mut value=(index as u32).wrapping_add((salt as u32+1).wrapping_mul(0x9e37_79b9));
    value=(value^(value>>16)).wrapping_mul(0x85eb_ca6b);value=(value^(value>>13)).wrapping_mul(0xc2b2_ae35);value^(value>>16)
}
/// One BF16 pattern (the classes of `dense_tc_rows`): near one; a wide exponent spread; subnormals and zeros; signed
/// zeros; plus and minus one in turn; mantissas of ones.
fn pattern(index:usize,salt:usize,class:usize)->u16 {
    let h=mixed(index,salt);let sign=((h>>16)&0x8000) as u16;let frac=(h&127) as u16;
    match class%6 {
        0=>sign|((124+(h>>8)%6) as u16)<<7|frac,
        1=>sign|((97+(h>>8)%60) as u16)<<7|frac,
        2=>sign|match h&3 {0=>0,1=>frac,_=>((1+(h>>8)%6) as u16)<<7|frac},
        3=>if h&7==0 {sign} else {sign|((120+(h>>8)%12) as u16)<<7|frac},
        4=>if index.is_multiple_of(2) {0x3f80} else {0xbf80},
        _=>sign|((122+(h>>8)%8) as u16)<<7|0x7f,
    }
}
/// An operand word. Class 6: a general FP32 word. Class 7: class 0 with a rare infinity or NaN. Class 8: a share in [0, 1).
fn word(index:usize,salt:usize,class:usize)->f32 {
    let h=mixed(index,salt+900);
    match class {
        6=>f32::from_bits((h&0x807f_ffff)|((120+(h>>23)%12)<<23)),
        7=>match h%1499 {0=>f32::INFINITY,1=>f32::NEG_INFINITY,2=>f32::NAN,_=>f32::from_bits(u32::from(pattern(index,salt,0))<<16)},
        8=>f32::from_bits((((h>>8) as f32/16_777_216.0/64.0).to_bits()+0x8000)&0xffff_0000),
        _=>f32::from_bits(u32::from(pattern(index,salt,class))<<16),
    }
}
fn plane(len:usize,salt:usize,class:usize)->Vec<f32> {(0..len).map(|i|word(i,salt,class)).collect()}
/// Key and value weights: BF16 patterns, two a word.
fn weights(len:usize,salt:usize,class:usize)->Vec<f32> {(0..len/2).map(|i|f32::from_bits(u32::from(pattern(2*i,salt,class))|u32::from(pattern(2*i+1,salt,class))<<16)).collect()}
/// Equal bits, or two NaNs (a NaN's payload is outside the contract).
fn same(a:f32,b:f32)->bool {a.to_bits()==b.to_bits() || a.is_nan() && b.is_nan()}
#[derive(Clone,Copy)]
struct Case {m:usize,h:usize,d:usize,v:usize,k:usize,r:usize,c:usize,n:usize}
impl Case {
    fn shape(&self,op:Op,m:usize)->Result<Shape> {
        let mut s=Shape::new(op,m).ok_or("absorbed shape")?;
        s.heads=self.h as u32;s.dim=self.d as u32;s.taps=self.v as u32;s.cols=if op==Op::AbsorbSoftmax {self.c} else {self.k} as u32;
        s.rows=self.r as u32;s.slots=self.n as u32;s.first=self.c as u32;Ok(s)
    }
    /// Selection rows. `varied`: counts of every residue (token 1 has none), IDs spread with a repeat; else full counts.
    fn selection(&self,tokens:&[usize],varied:bool)->Vec<f32> {
        let mut out=vec![0.0f32;tokens.len()*(self.n+1)];
        for (slot,&t) in tokens.iter().enumerate() {
            let count=if varied || t%5==3 {if t==1 || t==8 {0} else {(t*37+self.c)%(self.c+1)}} else {self.c};
            let row=&mut out[slot*(self.n+1)..(slot+1)*(self.n+1)];
            for (j,id) in row.iter_mut().enumerate().take(count) {*id=((j*389+t*17)%self.r) as f32;}
            if count>1 {row[1]=row[0];}
            row[self.n]=count as f32;
        }
        out
    }
}
/// The four planes of one prompt-form call on the device: absorbed query, scores, mixed latent, output.
fn planes(b:&Backend<'_>,case:&Case,m:usize,query:&DeviceVector<'_>,latent:&DeviceVector<'_>,selected:&DeviceVector<'_>,weight:&DeviceVector<'_>)->Result<[Vec<f32>;4]> {
    let q=b.prompt_operation(case.shape(Op::AbsorbQuery,m)?,[Some(query),None,None,None,None],Some(weight))?;
    let scores=b.prompt_operation(case.shape(Op::AbsorbScores,m)?,[Some(&q),Some(latent),Some(selected),None,None],None)?;
    let probability=b.prompt_operation(case.shape(Op::AbsorbSoftmax,m)?,[Some(&scores),Some(selected),None,None,None],None)?;
    let values=b.prompt_operation(case.shape(Op::AbsorbValues,m)?,[Some(&probability),Some(latent),Some(selected),None,None],None)?;
    let output=b.prompt_operation(case.shape(Op::AbsorbProject,m)?,[Some(&values),Some(selected),None,None,None],Some(weight))?;
    Ok([b.read_vector_values(&q)?,b.read_vector_values(&scores)?,b.read_vector_values(&values)?,b.read_vector_values(&output)?])
}
const PLANES:[&str;4]=["query","scores","values","project"];
/// Operand classes of the twin sweep, (A operand, B operand): each class with itself, then crossed pairs (tiny and
/// subnormal values against a wide exponent spread and against values near one, both ways; signed zeros; general words).
const CLASSES:[(usize,usize);18]=[(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(2,1),(1,2),(2,0),(0,2),(2,6),(6,2),(3,1),(1,8),(2,8)];
/// The OLD word of one score (SUM-ORDER v1: 32 strips of FP32 sums, joined at distances 16, 8, 4, 2, 1), for the control.
fn old_score(query:&[f32],row:&[f32],key_dim:usize)->f32 {
    let mut strips=[0.0f32;32];
    for j in 0..query.len() {strips[(j/8)%32]+=query[j]*row[j];}
    for distance in [16,8,4,2,1] {for lane in 0..distance {strips[lane]+=strips[lane+distance];}}
    strips[0]*(1.0/(key_dim as f32).sqrt())
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;
    let mut receipt=String::from("ABSORB_TC_ROWS host_path=Backend::prompt_operation+causal_operation word=absorb_tc twin=tc_twin_bf16::fold\n");
    let mut failures=Vec::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,32<<30,|b| {
        let line=|text:String,receipt:&mut String| {eprint!("{text}");receipt.push_str(&text);};
        // Part 1: the twin.
        let small=[Case {m:3,h:5,d:16,v:32,k:34,r:40,c:33,n:40},Case {m:5,h:3,d:48,v:20,k:64,r:700,c:600,n:640},Case {m:33,h:2,d:32,v:16,k:130,r:90,c:70,n:75},
            Case {m:2,h:33,d:16,v:16,k:18,r:50,c:17,n:30},Case {m:1,h:1,d:17,v:9,k:2,r:31,c:13,n:19}];
        let model=Case {m:24,h:32,d:256,v:256,k:512,r:4096,c:2051,n:2051};
        for (number,case) in small.iter().chain([&model]).enumerate() {
            let whole=number<small.len();let tokens:Vec<usize>=(0..case.m).collect();
            let mut by_class=[[0usize;2];CLASSES.len()];let (mut words,mut differ)=(0usize,0usize);let mut first=String::new();
            for (class,&(left,right)) in CLASSES.iter().enumerate() {
                let salt=number*100+class;let select=case.selection(&tokens,class%2==0);
                // `left`: the class of the A operand (weights, cache rows). `right`: the class of the B operand (the first input).
                let weight=weights(case.h*(case.d+case.v)*case.k,salt+1,left);
                let inputs:[(Op,Vec<f32>);4]=[(Op::AbsorbQuery,plane(case.m*case.h*case.d,salt+2,right)),(Op::AbsorbScores,plane(case.m*case.h*case.k,salt+3,right)),
                    (Op::AbsorbValues,plane(case.m*case.h*case.c,salt+4,right)),(Op::AbsorbProject,plane(case.m*case.h*case.k,salt+5,right))];
                let latent=plane(case.r*case.k,salt+6,left);
                let (dl,ds,dw)=(b.upload_vector(&latent)?,b.upload_vector(&select)?,b.upload_vector(&weight)?);
                for (op,x) in &inputs {
                    let s=case.shape(*op,case.m)?;let dx=b.upload_vector(x)?;
                    let (device,host):([Option<&DeviceVector<'_>>;5],[&[f32];5])=match op {
                        Op::AbsorbQuery=>([Some(&dx),None,None,None,None],[x,&[],&[],&[],&[]]),
                        Op::AbsorbProject=>([Some(&dx),Some(&ds),None,None,None],[x,&select,&[],&[],&[]]),
                        _=>([Some(&dx),Some(&dl),Some(&ds),None,None],[x,&latent,&select,&[],&[]]),
                    };
                    let dense=matches!(op,Op::AbsorbQuery|Op::AbsorbProject);
                    let have=b.read_vector_values(&b.prompt_operation(s,device,dense.then_some(&dw))?)?;
                    let w:&[f32]=if dense {&weight} else {&[]};
                    let mut check=|index:usize,want:f32| {
                        words+=1;by_class[class][0]+=1;
                        if !same(have[index],want) {differ+=1;by_class[class][1]+=1;
                            if first.is_empty() {first=format!("op={op:?} classes={left},{right} index={index} device={:08x} twin={:08x}",have[index].to_bits(),want.to_bits());}}
                    };
                    if whole {
                        let want=prompt_ops::reference(s,host,w)?;
                        if want.len()!=have.len() {return Err("absorbed plane extent differs".into());}
                        for (index,&value) in want.iter().enumerate() {check(index,value);}
                    } else {
                        for sample in 0..3000 {let index=mixed(sample,salt+7) as usize%have.len();let mut one=[0.0f32];
                            latent_absorb_rows::unit(s,index,host,w,&mut one);check(index,one[0]);}
                    }
                }
            }
            line(format!("ABSORB TC TWIN case={number} tokens={} heads={} key={} value={} latent={} cache_rows={} first={} every_word={whole} words={words} differ={differ} differ_by_class={:?} first=[{first}]\n",
                case.m,case.h,case.d,case.v,case.k,case.r,case.c,by_class.map(|c|c[1])),&mut receipt);
            if differ!=0 {failures.push(format!("TWIN case={number} differ={differ} {first}"));}
        }
        // Part 2: the control. The same scores against the old word function must differ somewhere.
        {
            let case=model;let tokens:Vec<usize>=(0..case.m).collect();let select=case.selection(&tokens,false);
            let (x,latent)=(plane(case.m*case.h*case.k,801,0),plane(case.r*case.k,802,0));
            let (dx,dl,ds)=(b.upload_vector(&x)?,b.upload_vector(&latent)?,b.upload_vector(&select)?);
            let have=b.read_vector_values(&b.prompt_operation(case.shape(Op::AbsorbScores,case.m)?,[Some(&dx),Some(&dl),Some(&ds),None,None],None)?)?;
            let (mut words,mut differ)=(0usize,0usize);
            for sample in 0..20000 {let index=mixed(sample,803) as usize%have.len();let (token,head,slot)=(index/(case.h*case.c),index/case.c%case.h,index%case.c);
                let id=select[token*(case.n+1)+slot] as usize;
                let old=old_score(&x[(token*case.h+head)*case.k..(token*case.h+head+1)*case.k],&latent[id*case.k..(id+1)*case.k],case.d);
                words+=1;if have[index].to_bits()!=old.to_bits() {differ+=1;}}
            line(format!("ABSORB TC CONTROL old_word=sum_order_v1 scores words={words} differ_from_old={differ} control_seen={}\n",differ>0),&mut receipt);
            if differ==0 {failures.push("CONTROL the old word function equals the device on every sampled word".into());}
        }
        // Parts 3 and 4 share the token rows of the model's shape.
        const MOST:usize=1040;
        let case=Case {m:MOST,..model};
        let weight=weights(case.h*(case.d+case.v)*case.k,901,0);let dw=b.upload_vector(&weight)?;
        // Cache rows by class (row % 7: six BF16 classes and general FP32 words), token rows the same.
        let latent:Vec<f32>=(0..case.r).flat_map(|row|(0..case.k).map(move |j|word(row*case.k+j,902,row%7))).collect();let dl=b.upload_vector(&latent)?;
        let row_of=|t:usize|->Vec<f32> {(0..case.h*case.d).map(|j|word(t*case.h*case.d+j,903,t%7)).collect()};
        let mut alone=Vec::with_capacity(MOST);
        for t in 0..MOST {
            let (dq,ds)=(b.upload_vector(&row_of(t))?,b.upload_vector(&case.selection(&[t],false))?);
            alone.push(planes(b,&case,1,&dq,&dl,&ds,&dw)?);
        }
        // Part 3: groups.
        let (mut group_words,mut group_differ)=(0usize,0usize);let mut first=String::new();
        for m in [2usize,3,8,16,17,31,32,33,64,255,256,257,1024,1040] {
            for placement in 0..3 {
                if placement>0 && m>257 {continue;}
                let order:Vec<usize>=match placement {0=>(0..m).collect(),1=>(0..m).map(|i|MOST-1-i).collect(),_=>(0..m).map(|i|(i*7+3)%MOST).collect()};
                let (dq,ds)=(b.upload_vector(&order.iter().flat_map(|&t|row_of(t)).collect::<Vec<_>>())?,b.upload_vector(&case.selection(&order,false))?);
                let group=planes(b,&case,m,&dq,&dl,&ds,&dw)?;
                let mut differ=[0usize;4];
                for (plane,(whole,name)) in group.iter().zip(PLANES).enumerate() {
                    let width=whole.len()/m;
                    for (slot,&t) in order.iter().enumerate() {for at in 0..width {
                        if whole[slot*width+at].to_bits()!=alone[t][plane][at].to_bits() {differ[plane]+=1;
                            if first.is_empty() {first=format!("m={m} placement={placement} plane={name} slot={slot} token={t} at={at} group={:08x} alone={:08x}",whole[slot*width+at].to_bits(),alone[t][plane][at].to_bits());}}
                    }}
                    group_words+=whole.len();
                }
                group_differ+=differ.iter().sum::<usize>();
                if placement==0 {line(format!("ABSORB TC GROUP tokens={m} differ_forward query={} scores={} values={} project={}\n",differ[0],differ[1],differ[2],differ[3]),&mut receipt);}
            }
        }
        line(format!("ABSORB TC ALONE-GROUP tokens_alone={MOST} group_words={group_words} differ={group_differ} first=[{first}]\n"),&mut receipt);
        if group_differ!=0 {failures.push(format!("ALONE-GROUP differ={group_differ} {first}"));}
        // Part 4: the step form on the gathered rows of one token. Twice: the three direct entries, then under the launch
        // recorder, which replays the two recorded entries (resident_launch_absorb_dense, resident_launch_absorb_attention)
        // with the bucket grid of the scores, as the served decode step does.
        let count_of=|name:&str|->u64 {spark_gpu_probe::prompt_launches::line().split_whitespace().find_map(|w|w.strip_prefix(&format!("{name}="))).and_then(|n|n.parse().ok()).unwrap_or(0)};
        for recorded in [false,true] {
            b.synchronize()?;let before=count_of("absorb_step_recorded");
            let queue=if recorded {let queue=b.resident_queue()?;b.enable_resident_launches()?;Some(queue)} else {None};
            let (mut step_words,mut step_differ,mut step_tokens)=(0usize,0usize,0usize);let mut first=String::new();
            for t in (0..MOST).filter(|t|t%13==0 || t%5==3) {
                let select=case.selection(&[t],false);let live=select[case.n] as usize;if live==0 {continue;}
                let gathered:Vec<f32>=(0..live).flat_map(|j|{let id=select[j] as usize;latent[id*case.k..(id+1)*case.k].iter().copied()}).collect();
                let (dq,dg,dc)=(b.upload_vector(&row_of(t))?,b.upload_vector(&gathered)?,b.upload_vector(&[live as f32])?);
                let s=StepShape {op:StepOp::AbsorbQuery,heads:case.h as u32,dim:case.d as u32,taps:case.v as u32,rows:live as u32,cols:case.k as u32,epsilon:0.0,bound:0.0};
                let q=b.causal_operation(s,[Some(&dq),None,None,None],Some(&dw))?;
                let scores=b.causal_operation(StepShape {op:StepOp::AbsorbScores,..s},[Some(&q),Some(&dg),Some(&dc),None],None)?;
                let probability=b.causal_operation(StepShape {op:StepOp::Softmax,taps:0,cols:0,..s},[Some(&scores),Some(&dc),None,None],None)?;
                let values=b.causal_operation(StepShape {op:StepOp::AbsorbValues,..s},[Some(&probability),Some(&dg),Some(&dc),None],None)?;
                let output=b.causal_operation(StepShape {op:StepOp::AbsorbProject,..s},[Some(&values),None,None,None],Some(&dw))?;
                let step=[b.read_vector_values(&q)?,b.read_vector_values(&scores)?,b.read_vector_values(&values)?,b.read_vector_values(&output)?];
                for (plane,name) in PLANES.iter().enumerate() {
                    // Scores: the step form holds [heads][live], the prompt form [heads][first] with zeros past the count.
                    let pairs:Vec<(usize,usize)>=if plane==1 {(0..case.h).flat_map(|head|(0..live).map(move |j|(head*live+j,head*case.c+j))).collect()} else {(0..step[plane].len()).map(|i|(i,i)).collect()};
                    for (a,r) in pairs {step_words+=1;
                        if step[plane][a].to_bits()!=alone[t][plane][r].to_bits() {step_differ+=1;
                            if first.is_empty() {first=format!("token={t} selected={live} plane={name} at={a} step={:08x} prompt={:08x}",step[plane][a].to_bits(),alone[t][plane][r].to_bits());}}}
                }
                step_tokens+=1;
            }
            b.synchronize()?;drop(queue);
            let took=count_of("absorb_step_recorded")-before;
            line(format!("ABSORB TC STEP-PROMPT recorded_entries={recorded} tokens={step_tokens} words={step_words} differ={step_differ} recorder_took={took} first=[{first}]\n"),&mut receipt);
            if step_differ!=0 {failures.push(format!("STEP-PROMPT recorded={recorded} differ={step_differ} {first}"));}
            if recorded!=(took>0) {failures.push(format!("STEP-PROMPT recorded={recorded} but the recorder took {took} step calls"));}
        }
        line(format!("ABSORB TC LAUNCHES {}\n",spark_gpu_probe::prompt_launches::line().split_whitespace().filter(|w|w.starts_with("latent_absorption")||w.starts_with("absorb_step")||w.starts_with("prompt_softmax_rows")).collect::<Vec<_>>().join(" ")),&mut receipt);
        Ok(())
    })?;
    if !failures.is_empty() {
        receipt.push_str(&format!("ABSORB_TC_ROWS FAILED {}\n",failures.join(" | ")));
        fs::write(out.join("ABSORB_TC_ROWS"),&receipt).map_err(|error|error.to_string())?;
        return Err(format!("ABSORB TC ROWS FAILED: {}",failures.join(" | ")));
    }
    receipt.push_str("ABSORB_TC_ROWS COMPLETE\n");
    fs::write(out.join("ABSORB_TC_ROWS"),receipt).map_err(|error|error.to_string())
}
