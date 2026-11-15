//! Byte verdicts use separate poison values and repeat every launch with new poison.
#![forbid(unsafe_code)]
use crate::Result;
const GUARD: [u8; 16] = [0x5a; 16];
const POISONS: [[u8; 4]; 2] = [[0xa7, 0x3d, 0x96, 0xe1], [0x6b, 0xd2, 0x48, 0xbc]];
#[derive(Clone, Copy)]
pub(crate) struct Span { pub address: u64, pub bytes: usize }
fn overlap(a: Span, b: Span) -> bool {
    a.address < b.address.saturating_add(b.bytes as u64) && b.address < a.address.saturating_add(a.bytes as u64)
}
fn layout<const N:usize>(outputs: &[Vec<Span>; N], inputs: &[Span]) -> Result<()> {
    let all: Vec<_> = outputs.iter().flatten().copied().collect();
    if N<2 || N>4 || outputs[0].is_empty() || outputs.iter().any(|s|s.len()!=outputs[0].len()) { return Err("empty or unequal output planes".into()); }
    for (i, &s) in all.iter().enumerate() {
        if s.bytes==0 || s.bytes%4!=0 || s.address.checked_add(s.bytes as u64).is_none() { return Err("invalid output extent".into()); }
        if inputs.iter().chain(all[..i].iter()).any(|&x| overlap(s,x)) { return Err("output aliases an input or another output".into()); }
    }
    Ok(())
}
fn equal(label: &str, a: &[Vec<u8>], b: &[Vec<u8>]) -> Result<()> {
    if a.len()!=b.len() || a.is_empty() { return Err(format!("{label}: plane count differs or is empty")); }
    for (plane,(x,y)) in a.iter().zip(b).enumerate() {
        if x.len()!=y.len() {return Err(format!("{label}: plane={plane} length differs"));}
        if let Some(byte)=x.iter().zip(y).position(|(x,y)|x!=y) {
            let word=byte.saturating_sub(16)/4;let start=16+word*4;
            return Err(format!("{label}: plane={plane} first_word={word} byte={byte} left={:02x?} right={:02x?}",&x[start..(start+4).min(x.len())],&y[start..(start+4).min(y.len())]));
        }
    }Ok(())
}
fn guards(label: &str, planes: &[Vec<u8>], spans: &[Span]) -> Result<()> {
    if planes.len()!=spans.len() { return Err(format!("{label}: missing output plane")); }
    for (i,(p,s)) in planes.iter().zip(spans).enumerate() {
        if p.len()!=s.bytes+32 || p[..16]!=GUARD || p[p.len()-16..]!=GUARD { return Err(format!("{label}: guard or extent failed plane={i}")); }
    }Ok(())
}
fn verdict<const N:usize>(label: &str, runs: &[[Vec<Vec<u8>>; N]; 2], spans: &[Vec<Span>;N]) -> Result<()> {
    for run in runs {for arm in 0..N {guards(label,&run[arm],&spans[arm])?;equal(label,&run[0],&run[arm])?;}}
    // This also resolves a real answer equal to a poison word. A skipped word
    // changes with its poison. A written answer must keep its exact bits.
    for arm in 0..N {equal(label,&runs[0][arm],&runs[1][arm])?;}Ok(())
}
fn faults<const N:usize>(label: &str, runs: &[[Vec<Vec<u8>>;N];2], spans: &[Vec<Span>;N]) -> Result<()> {
    for arm in 0..N {for plane in 0..spans[arm].len() {for fault in 0..3 {
        let mut bad=runs.clone();
        for round in 0..2 {
            let p=&mut bad[round][arm][plane];let end=p.len()-16;
            match fault {0=>p[16..end].fill(POISONS[round][arm]),1=>p[end-4..end].fill(POISONS[round][arm]),_=>p[end-4]^=1}
        }
        if verdict(label,&bad,spans).is_ok() {return Err(format!("{label}: checker accepted fault={fault} plane={plane}"));}
    }}}Ok(())
}
/// The callback initializes logical outputs with `poison`, launches one arm,
/// and downloads each complete guarded output. Inputs must be restored if mutable.
#[cfg_attr(not(target_os="linux"), allow(dead_code))]
pub(crate) fn arms<const N:usize>(label: &str, spans: [Vec<Span>;N], inputs: &[Span], mut run: impl FnMut(usize,u8)->Result<Vec<Vec<u8>>>) -> Result<Vec<Vec<u8>>> {
    layout(&spans,inputs).map_err(|e|format!("{label}: {e}"))?;
    let mut runs:[[Vec<Vec<u8>>;N];2]=std::array::from_fn(|_|std::array::from_fn(|_|Vec::new()));
    for round in 0..2 {for arm in 0..N {runs[round][arm]=run(arm,POISONS[round][arm])?;}}
    verdict(label,&runs,&spans)?;
    faults(label,&runs,&spans)?;
    Ok(runs[0][0].clone())
}
pub(crate) fn self_test() -> Result<()> {
    let spans=std::array::from_fn::<_,4,_>(|arm|vec![Span{address:256*(arm as u64+1),bytes:8}]);
    let data=[GUARD.as_slice(),&[0xa7;4],&[0;4],GUARD.as_slice()].concat();
    let runs=std::array::from_fn::<_,2,_>(|_|std::array::from_fn::<_,4,_>(|_|vec![data.clone()]));
    verdict("self-test",&runs,&spans)?;faults("self-test",&runs,&spans)?;
    for edge in [0,39] {let mut bad=runs.clone();bad[0][1][0][edge]^=1;if verdict("guard",&bad,&spans).is_ok(){return Err("guard test passed".into());}}
    if layout(&spans,&[Span{address:260,bytes:4}]).is_ok(){return Err("alias test passed".into());}
    if layout(&[vec![],vec![]],&[]).is_ok(){return Err("empty test passed".into());}
    let result=arms("broken finish",spans.clone(),&[],|arm,poison| {
        let mut p=data.clone();if arm==3 {p[20..24].fill(poison);}Ok(vec![p])
    });
    if result.is_ok() {return Err("broken finish self-test passed".into());}
    println!("CHECKER SELF TEST PASS faults=no-store,one-word-omitted,one-bit guards=both alias=rejected poison-collision=accepted");Ok(())
}
#[cfg(test)] mod tests { #[test] fn rejects_faults_and_accepts_poison_collision(){super::self_test().unwrap();} }
