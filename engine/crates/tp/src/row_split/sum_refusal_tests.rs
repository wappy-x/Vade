//! SUM REFUSAL: a device residual sum that the host form refuses is refused with the host form's error. The sums go
//! through the served forwarders (&Split over &Device: the `&C` and `Split` forwards of `bf16_sum`, then `Device`).
//! Each bad case runs three ways, and each must end with the error the host form gives on the same device: the same sums
//! outside a work scope, where no resident queue runs and `Device::bf16_sum` takes the host form, as every sum did before
//! the device form. (The Host compute is no reference: it refuses a NaN or infinite word at its upload.)
//! - PICK: the sum, a later launch that reads it, then the read a pick makes. No words may leave.
//! - LATER ERROR: the sum, a later launch, then a later step of the same scope fails first.
//! - SCOPE END: the sum, then the scope ends with no read.
//!
//! The device form must have queued each bad sum (its `bf16_sum` returned), so the refusal is the deferred one.
//! After every refusal a clean case must give the host form's words, so no check is carried past its scope.
use super::*;
use super::kept_charge_tests::Local;
use crate::bounded_decoder::Device;
use spark_gpu_probe::driver::with_backend;

/// x and y of `words` words, finite but for what `kind` puts at word `at`: 1 NaN in x, 2 +inf in y, 3 -inf in x,
/// 4 two finite words whose sum is finite in FP32 and rounds past BF16.
fn operands(words:usize,kind:usize,at:usize)->(Vec<f32>,Vec<f32>) {
    let mut x:Vec<f32>=(0..words).map(|i|((i*37%101) as f32-50.0)/8.0).collect();
    let mut y:Vec<f32>=(0..words).map(|i|((i*53%97) as f32-48.0)/16.0).collect();
    match kind {1=>x[at]=f32::NAN,2=>y[at]=f32::INFINITY,3=>x[at]=f32::NEG_INFINITY,4=>{x[at]=1.7e38;y[at]=1.7e38;},_=>{}}
    (x,y)
}
fn same<T:PartialEq>(got:&Result<T>,want:&Result<T>)->bool {
    match (got,want) {(Ok(a),Ok(b))=>a==b,(Err(a),Err(b))=>a==b,_=>false}
}
/// The words as bits, so equal means the same bytes.
fn bits(v:Result<Vec<f32>>)->Result<Vec<u32>> {v.map(|words|words.iter().map(|w|w.to_bits()).collect())}
/// The two sums and the read, outside a work scope (the host form). `at_add` says whether the first sum itself refused.
fn host_sums<C:Compute>(c:&C,xs:&[f32],ys:&[f32],at_add:&mut bool)->Result<Vec<f32>> {
    let (x,y)=(c.upload(xs)?,c.upload(ys)?);let s=c.bf16_sum(&x,&y);*at_add=s.is_err();let t=c.bf16_sum(&s?,&y)?;c.read(&t)
}

#[test]
#[ignore="requires an authorized CUDA device; refusal witness, not served token IDs"]
fn device_sum_refusal_is_the_host_refusal_through_the_forwarders() {
    with_backend(0,spark_gpu_probe::layer_ptx().unwrap(),1usize<<30,|backend| {
        let device=Device::new(backend);let split=Split::new(&device,Local)?;let served=&split;
        let (mut cases,mut at_adds,mut deferred,mut equal,mut clean,mut wrong)=(0usize,0usize,0usize,[0usize;3],0usize,Vec::new());
        for words in [1usize,1023,1024,1025,6144] {
            let (cx,cy)=operands(words,0,0);let mut at_add=false;let want_clean=bits(host_sums(served,&cx,&cy,&mut at_add));
            if want_clean.is_err() || at_add {return Err(format!("words={words}: the host form refuses the clean case"));}
            for kind in 1..=4 {
                for at in [0,words/2,words-1] {
                    let (xs,ys)=operands(words,kind,at);let want=bits(host_sums(served,&xs,&ys,&mut at_add));cases+=1;at_adds+=usize::from(at_add);
                    let Err(refusal)=want.clone() else {return Err(format!("words={words} kind={kind} at={at}: the host form passed a bad case"));};
                    let mut queued=false;
                    let pick=bits(served.work(|| {let (x,y)=(served.upload(&xs)?,served.upload(&ys)?);let s=served.bf16_sum(&x,&y)?;queued=true;let t=served.bf16_sum(&s,&y)?;served.read(&t)}));
                    let later:Result<()>=served.work(|| {let (x,y)=(served.upload(&xs)?,served.upload(&ys)?);let s=served.bf16_sum(&x,&y)?;served.bf16_sum(&s,&y)?;Err("a later step failed".to_string())});
                    let ended:Result<()>=served.work(|| {let (x,y)=(served.upload(&xs)?,served.upload(&ys)?);served.bf16_sum(&x,&y).map(|_|())});
                    let after=bits(served.work(|| {let (x,y)=(served.upload(&cx)?,served.upload(&cy)?);let s=served.bf16_sum(&x,&y)?;let t=served.bf16_sum(&s,&y)?;served.read(&t)}));
                    deferred+=usize::from(queued);
                    let checks=[("pick",same(&pick,&want),pick.as_ref().err().cloned()),("later error",same(&later,&Err(refusal.clone())),later.as_ref().err().cloned()),
                        ("scope end",same(&ended,&Err(refusal.clone())),ended.as_ref().err().cloned())];
                    for (n,(name,ok,got)) in checks.into_iter().enumerate() {
                        if ok {equal[n]+=1;} else {wrong.push(format!("words={words} kind={kind} at={at} {name}: got {got:?}, want {refusal:?}"));}
                    }
                    if same(&after,&want_clean) {clean+=1;} else {wrong.push(format!("words={words} kind={kind} at={at}: the clean case after the refusal differs"));}
                }
            }
        }
        println!("SUM REFUSAL MEASURED cases={cases} host_refused_at_add={at_adds} deferred={deferred} pick_equal={} later_error_equal={} scope_end_equal={} clean_after_equal={clean} wrong={}",equal[0],equal[1],equal[2],wrong.len());
        for line in wrong.iter().take(8) {println!("SUM REFUSAL WRONG {line}");}
        if !wrong.is_empty() || deferred!=cases || at_adds!=cases {return Err(format!("{} of {} checks wrong, {deferred} of {cases} sums queued, {at_adds} host refusals at the add",wrong.len(),4*cases));}
        Ok(())
    }).unwrap();
}
