//! Complete index pools depend only on their own rows. Keep earlier pools and
//! run the unchanged scalar fold only for pools this group completes.
use super::*;

fn completed(first:usize,rows:usize,pool:usize)->Result<std::ops::Range<usize>> {
    need(pool>0,"prompt pool width is zero")?;
    let last=first.checked_add(rows).ok_or("prompt pool context overflow")?;
    Ok(first/pool..last/pool)
}

pub(super) fn append<C:Compute>(b:&C,s:Shape,first:usize,m:usize,x:[&C::Value;4],prior:&C::Value,w:&[f32])->Result<[C::Value;3]> {
    let (dim,pool)=(s.dim as usize,s.taps as usize);
    let groups=completed(first,m,pool)?;
    let keys=b.join(&[x[0],x[2]])?;let gates=b.join(&[x[1],x[3]])?;
    let earlier=b.range(prior,0,groups.start.checked_mul(dim).ok_or("prompt pool prefix overflow")?)?;
    let mut parts=vec![earlier];
    for group in groups {
        let offset=group.checked_mul(pool).and_then(|v|v.checked_mul(dim)).ok_or("prompt pool offset overflow")?;
        let before=(pool-1).checked_mul(dim).ok_or("prompt pool span overflow")?;
        let old_keys=b.range(&keys,offset,before)?;let old_gates=b.range(&gates,offset,before)?;
        let key=b.range(&keys,offset+before,dim)?;let gate=b.range(&gates,offset+before,dim)?;
        let folded=b.operation(Shape {rows:number(pool as u64)?,..s},[Some(&old_keys),Some(&old_gates),Some(&key),Some(&gate)],w)?;
        parts.push(b.range(&folded,2*pool*dim,dim)?);
    }
    let pooled=if parts.len()==1 {parts.remove(0)} else {b.join(&parts.iter().collect::<Vec<_>>())?};
    Ok([keys,gates,pooled])
}

#[cfg(test)] mod tests {
    use super::*;
    use spark_gpu_probe::{causal_ops::reference,latent_ops};
    #[test] fn completed_pools_cover_only_new_full_groups() {
        for pool in 1..=32 {for first in 1..=257 {for rows in 1..=16 {
            let groups=completed(first,rows,pool).unwrap().collect::<Vec<_>>();
            let expected=(0..(first+rows)/pool).filter(|&group|(group+1)*pool>first).collect::<Vec<_>>();
            assert_eq!(groups,expected);
        }}}
        assert!(completed(0,1,0).is_err());assert!(completed(usize::MAX,1,1).is_err());
    }
    #[test] fn appended_pool_folds_equal_full_history_bits() {
        for pool in [1usize,2,4,7,16] {for dim in [1usize,4,17] {for first in [1usize,3,4,15,16,31] {for rows in 1..=16 {
            let total=first+rows;let mut s=shape(Op::Pool);s.dim=dim as u32;s.taps=pool as u32;s.rows=total as u32;
            let keys=(0..total*dim).map(|i|f32::from_bits([0,0x80000000,0x3f800000,0xbf800000,0x3eaaaaab][i%5])).collect::<Vec<_>>();
            let gates=(0..total*dim).map(|i|((i*7%23) as f32-11.0)*0.0625).collect::<Vec<_>>();
            let w=(0..pool*dim).map(|i|((i%5) as f32-2.0)*0.125).collect::<Vec<_>>();
            let full=reference(s,[&keys[..(total-1)*dim],&gates[..(total-1)*dim],&keys[(total-1)*dim..],&gates[(total-1)*dim..]],&w).unwrap();
            let old=reference(Shape {rows:first as u32,..s},[&keys[..(first-1)*dim],&gates[..(first-1)*dim],&keys[(first-1)*dim..first*dim],&gates[(first-1)*dim..first*dim]],&w).unwrap();
            let mut pooled=old[2*first*dim..].to_vec();
            for group in completed(first,rows,pool).unwrap() {
                let offset=group*pool*dim;let end=offset+(pool-1)*dim;
                let part=reference(Shape {rows:pool as u32,..s},[&keys[offset..end],&gates[offset..end],&keys[end..end+dim],&gates[end..end+dim]],&w).unwrap();
                pooled.extend_from_slice(&part[2*pool*dim..]);
                // Both production phases preserve the same pool fold as the scalar reference.
                let channels=reference(Shape {op:Op::PoolChannels,rows:pool as u32,..s},[&keys[offset..end],&gates[offset..end],&keys[end..end+dim],&gates[end..end+dim]],&w).unwrap();
                let layout=reference(Shape {op:Op::PoolLayout,rows:pool as u32,..s},[&channels,&[],&[],&[]],&[]).unwrap();
                assert_eq!(part.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),layout.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            }
            assert_eq!(pooled.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),full[2*total*dim..].iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            assert!(latent_ops::lengths(s).is_some());
        }}}}
    }
}
