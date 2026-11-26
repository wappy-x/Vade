//! DRAFT CONFIDENCE: the draft head's pick with its probability, for a head split over the ranks.
use super::{Rows,merge_argmax};
use crate::Result;
/// DRAFT CONFIDENCE: `merge_argmax` over four-word parts (best, local ID, valid, softmax mass against the part's
/// best: `logits_argmax_mass`). The pick is `merge_argmax`'s; the second result is its probability under the whole
/// row, 1 / (the parts' masses moved to the winning logit). A score only: no kept token reads it.
#[cfg(any(target_os="linux",test))]
pub fn merge_argmax_mass(values:&[f32],rows:&[Rows])->Result<(u32,f32)> {
    if rows.is_empty() || values.len()!=rows.len()*4 {return Err("resident argmax extent differs".into());}
    let picks=values.as_chunks::<4>().0.iter().flat_map(|part|part[..3].iter().copied()).collect::<Vec<_>>();
    let id=merge_argmax(&picks,rows)?;
    let best=values.as_chunks::<4>().0.iter().map(|part|part[0]).fold(f32::NEG_INFINITY,f32::max);
    let mut mass=0f64;
    for part in values.as_chunks::<4>().0 {
        if !part[3].is_finite() || part[3]<0.0 {return Err("resident argmax mass differs".into());}
        if part[3]>0.0 {mass+=f64::from(part[3])*(f64::from(part[0])-f64::from(best)).exp();}
    }
    // The winning logit is in its own part's mass with weight 1: a mass under 1 is a broken part.
    if mass<1.0-1e-3 {return Err("resident argmax mass is under its best logit".into());}
    Ok((id,(1.0/mass) as f32))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_mass_pick_is_the_plain_pick_and_its_probability_is_under_the_whole_row() {
        let rows=[Rows {start:4096,count:4096,total:8192,split:true},Rows {start:0,count:4096,total:8192,split:true}];
        // Part one: best 7 and mass 1.5 against it; part two: best 6 and mass 2 against 6. The pick is part one's, as
        // merge_argmax gives it, and its probability is 1 / (1.5 + 2 exp(-1)).
        for (values,id) in [([7.0,0.0,1.0,1.5,6.0,4095.0,1.0,2.0],4096),([6.0,0.0,1.0,2.0,7.0,4095.0,1.0,1.5],4095)] {
            let (pick,p)=merge_argmax_mass(&values,&rows).unwrap();
            assert_eq!(pick,id);assert_eq!(pick,merge_argmax(&[values[0],values[1],values[2],values[4],values[5],values[6]],&rows).unwrap());
            assert!((f64::from(p)-1.0/(1.5+2.0*(-1.0f64).exp())).abs()<1e-6,"{p}");
        }
        // A sure pick: every other logit is far under it.
        let (_,p)=merge_argmax_mass(&[7.0,0.0,1.0,1.0,-90.0,1.0,1.0,1.0],&rows).unwrap();assert_eq!(p,1.0);
        // A broken mass is refused: not finite, negative, or under the winner's own weight.
        for mass in [f32::NAN,f32::INFINITY,-1.0,0.5] {assert!(merge_argmax_mass(&[7.0,0.0,1.0,mass,6.0,1.0,1.0,1.0],&rows).is_err());}
        assert!(merge_argmax_mass(&[7.0,0.0,1.0,1.0],&rows).is_err());assert!(merge_argmax_mass(&[],&[]).is_err());
    }
}
