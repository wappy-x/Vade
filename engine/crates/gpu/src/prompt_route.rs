//! Stable prompt routing metadata. Integer tables keep expert and original-slot order.
pub const THREADS:u32=256;
pub const MAX_ROWS:u32=1024;
pub const MAX_EXPERTS:u32=1024;
pub const MAX_PICKS:u32=8;
#[repr(C)]
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Shape {pub rows:u32,pub picks:u32,pub experts:u32}
impl Shape {
    pub fn admitted(self)->bool {self.rows>0 && self.rows<=MAX_ROWS && self.picks>0 && self.picks<=MAX_PICKS && self.picks<=self.experts && self.experts<=MAX_EXPERTS}
    pub fn pairs(self)->usize {self.rows as usize*self.picks as usize}
    /// An upper bound on sum(ceil(count[expert] / tile)), including empty experts.
    pub fn work_capacity(self,tile:u32)->usize {self.experts as usize+self.pairs().div_ceil(tile as usize)}
    pub fn lengths(self)->Option<[usize;9]> {
        if !self.admitted() {return None;}
        let n=self.pairs();let e=self.experts as usize;
        Some([e,e+1,n,n,4*n,4*n,2*self.work_capacity(128),2*self.work_capacity(64),1])
    }
    /// Every private plane, rounded to a 4 KiB allocation boundary. IDs have an existing owner.
    pub fn workspace_bytes(self)->Option<usize> {self.lengths()?.into_iter().try_fold(0usize,|sum,n|sum.checked_add(n.checked_mul(4)?.checked_add(4095)?/4096*4096))}
}
#[repr(C)]
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Tile {pub expert:u32,pub tile:u32}
pub const EMPTY:Tile=Tile {expert:u32::MAX,tile:u32::MAX};
#[repr(C)]
#[derive(Clone,Copy)]
pub struct Build {pub ids:u64,pub counts:u64,pub starts:u64,pub order:u64,pub pairs:u64,pub grouped:u64,pub placed:u64,pub work128:u64,pub work64:u64,pub status:u64,pub shape:Shape,pub pad:u32}
const _:()={assert!(core::mem::size_of::<Build>()==96);assert!(core::mem::offset_of!(Build,shape)==80);assert!(core::mem::size_of::<Tile>()==8);};
/// Static header of one row set. Work names the route's sorted expert/tile map.
/// `order`: DIRECT PLACE, the route's pairs plane (the original pair of each
/// sorted row), read only when the build writes a row map; else 0.
#[repr(C)]
#[derive(Clone,Copy)]
pub struct NativeSet {pub counts:u64,pub starts:u64,pub work:u64,pub output:u64,pub activations:u64,pub order:u64,pub first:u32,pub pairs:u32,pub work_capacity:u32,pub pad:u32}
/// Build native parts, their common work list and optional calibrated row scales.
/// A zero table/work/output builds only activations; zero activations skips that store.
/// `merge` 1 (PAIRED PARTS, work only): one work interval an expert over the
/// tokens of all its sets, each record naming the expert's first part
/// (e*count); the product launch takes `NvRows::merge` = count. 0: an interval
/// a (expert, set), as before.
/// `rowmap` (DIRECT PLACE, with a table): for every sorted row i of set s, the
/// u64 address of word (row 0) of original pair `order[i]` in the set's output
/// (output + 4 * pair * rows), at index s.first + i; 0 = none. Every pair is
/// checked whole and below the set's pairs before its address is stored.
#[repr(C)]
#[derive(Clone,Copy)]
pub struct NativeBuild {pub templates:u64,pub sets:u64,pub table:u64,pub work:u64,pub rowmap:u64,pub count:u32,pub experts:u32,pub rows:u32,pub tokens:u32,pub capacity:u32,pub merge:u32}
pub const MAX_SETS:u32=8;
/// PAIRED PARTS: an upper bound on sum over experts of ceil((sum over sets of
/// count) / tile): experts + ceil(all pairs / tile). Never above the sum of
/// the sets' own capacities.
#[cfg(not(target_arch="nvptx64"))]
pub fn merged_work_capacity(experts:u32,pairs:usize,tile:u32)->usize {experts as usize+pairs.div_ceil(tile as usize)}
/// Full retained epoch: IDs and nine route planes per set, three calibration
/// planes and their headers, and gate/up/down parts, work and launch headers.
/// Immutable weight templates are resident startup storage. Product/packed
/// scratch remains in the existing product workspace, never this metadata sum.
#[cfg(not(target_arch="nvptx64"))]
pub fn epoch_workspace_bytes(sets:&[Shape])->Option<usize> {
    if sets.is_empty() || sets.len()>MAX_SETS as usize || sets.iter().any(|s|!s.admitted() || s.experts!=sets[0].experts) {return None;}
    let page=|words:usize|words.checked_mul(4)?.checked_add(4095).map(|n|n/4096*4096);
    let mut bytes=0usize;let mut pairs=0usize;let mut work=0usize;
    for &s in sets {bytes=bytes.checked_add(s.workspace_bytes()?)?.checked_add(page(s.pairs())?)?;pairs=pairs.checked_add(s.pairs())?;work=work.checked_add(s.work_capacity(64))?;}
    let headers=page(sets.len().checked_mul(16)?)?;
    // A phase: row scales, headers, parts, work, launch headers and (DIRECT PLACE) one u64 row address a pair.
    let phase=page(pairs)?.checked_add(headers)?.checked_add(page((sets[0].experts as usize).checked_mul(sets.len())?.checked_mul(10)?)?)?.checked_add(page(work.checked_mul(2)?)?)?.checked_add(headers)?.checked_add(page(pairs.checked_mul(2)?)?)?;
    bytes.checked_add(phase.checked_mul(3)?)
}
const _:()={assert!(core::mem::size_of::<NativeSet>()==64);assert!(core::mem::size_of::<NativeBuild>()==64);};

#[cfg(not(target_arch="nvptx64"))]
pub struct Reference {pub counts:Vec<u32>,pub starts:Vec<u32>,pub order:Vec<f32>,pub pairs:Vec<f32>,pub grouped:Vec<f32>,pub placed:Vec<f32>,pub work128:Vec<Tile>,pub work64:Vec<Tile>}
/// Independent host twin: sort (expert, original pair), then construct each table.
#[cfg(not(target_arch="nvptx64"))]
pub fn reference(ids:&[f32],s:Shape)->Result<Reference,String> {
    if !s.admitted() || ids.len()!=s.pairs() {return Err("prompt route extent differs".into());}
    let n=ids.len();let mut sorted=Vec::with_capacity(n);
    for (pair,&id) in ids.iter().enumerate() {
        if !(id>=0.0 && id<s.experts as f32 && id==(id as u32) as f32) {return Err("prompt expert ID is outside the recipe".into());}
        let token=pair/s.picks as usize;let slot=pair%s.picks as usize;
        if ids[token*s.picks as usize..token*s.picks as usize+slot].contains(&id) {return Err("prompt route repeats an expert".into());}
        sorted.push((id as u32,pair));
    }
    sorted.sort_unstable();let mut counts=vec![0u32;s.experts as usize];for &(e,_) in &sorted {counts[e as usize]+=1;}
    let mut starts=vec![0;s.experts as usize+1];for (i,&count) in counts.iter().enumerate() {starts[i+1]=starts[i]+count;}
    let mut order=vec![0.0;n];let mut pairs=vec![0.0;n];let mut grouped=vec![0.0;4*n];let mut placed=vec![0.0;4*n];
    for (at,&(e,pair)) in sorted.iter().enumerate() {
        let first=starts[e as usize] as usize;let rank=at-first;
        let entry=[first as f32,counts[e as usize] as f32,rank as f32];
        order[at]=(pair/s.picks as usize) as f32;pairs[at]=pair as f32;
        grouped[3*at..3*at+3].copy_from_slice(&entry);placed[3*pair..3*pair+3].copy_from_slice(&entry);
        grouped[3*n+at]=at as f32;placed[3*n+at]=pair as f32;
    }
    let work=|tile:u32| {let mut out=Vec::with_capacity(s.work_capacity(tile));for (expert,&count) in counts.iter().enumerate() {for at in 0..count.div_ceil(tile) {out.push(Tile {expert:expert as u32,tile:at});}}out.resize(s.work_capacity(tile),EMPTY);out};
    let (work128,work64)=(work(128),work(64));Ok(Reference {counts,starts,order,pairs,grouped,placed,work128,work64})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn stable_groups_include_empty_experts_and_inverse_tables() {
        let s=Shape {rows:3,picks:2,experts:4};let r=reference(&[2.0,0.0, 1.0,2.0, -0.0,3.0],s).unwrap();
        assert_eq!(r.counts,[2,1,2,1]);assert_eq!(r.starts,[0,2,3,5,6]);assert_eq!(r.pairs,[1.0,4.0,2.0,0.0,3.0,5.0]);assert_eq!(r.order,[0.0,2.0,1.0,0.0,1.0,2.0]);
        for table in [&r.grouped,&r.placed] {for (row,entry) in table[..18].as_chunks::<3>().0.iter().enumerate() {assert_eq!(table[18+(entry[0]+entry[2]) as usize],row as f32);}}
        assert_eq!(r.work128[..4],[Tile {expert:0,tile:0},Tile {expert:1,tile:0},Tile {expert:2,tile:0},Tile {expert:3,tile:0}]);assert_eq!(r.work128[4],EMPTY);
    }
    #[test] fn malformed_ids_and_unfunded_extents_refuse() {
        let s=Shape {rows:1,picks:2,experts:4};for ids in [[0.0,0.0],[0.0,-0.0],[0.0,f32::NAN],[0.0,f32::INFINITY],[0.0,-1.0],[0.0,0.5],[0.0,4.0]] {assert!(reference(&ids,s).is_err());}
        for s in [Shape {rows:0,..s},Shape {rows:MAX_ROWS+1,..s},Shape {picks:0,..s},Shape {experts:1,..s},Shape {experts:MAX_EXPERTS+1,..s}] {assert!(s.lengths().is_none() && s.workspace_bytes().is_none());}
    }
    #[test] fn tile_capacities_and_charges_cover_every_plane() {
        for rows in [1,2,7,31,32,33,127,128,129,1024] {for picks in [1,2,4,8] {let s=Shape {rows,picks,experts:32};
            let ids=(0..rows).flat_map(|t|(0..picks).map(move |p|((t+p)%32) as f32)).collect::<Vec<_>>();let r=reference(&ids,s).unwrap();
            for (tile,work) in [(128,&r.work128),(64,&r.work64)] {assert_eq!(work.len(),s.work_capacity(tile));assert!(work.iter().take_while(|v|**v!=EMPTY).count()<=work.len());}
            assert!(s.workspace_bytes().unwrap()>=s.lengths().unwrap().iter().sum::<usize>()*4);
        }}
    }
    #[test] fn epoch_charge_keeps_all_sets_and_all_phase_headers() {
        let s=Shape {rows:1024,picks:8,experts:512};
        let one=epoch_workspace_bytes(&[s]).unwrap();let two=epoch_workspace_bytes(&[s,s]).unwrap();
        assert!(one>s.workspace_bytes().unwrap()+s.pairs()*4);assert!(two>one);
        assert!(epoch_workspace_bytes(&[s;MAX_SETS as usize]).unwrap()>=two);
        assert!(epoch_workspace_bytes(&[]).is_none());assert!(epoch_workspace_bytes(&[s;MAX_SETS as usize+1]).is_none());
        assert!(epoch_workspace_bytes(&[s,Shape {experts:8,..s}]).is_none());
    }
}
