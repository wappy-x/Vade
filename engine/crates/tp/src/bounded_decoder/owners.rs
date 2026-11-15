//! Physical state owners. Build lists only for cache publication and parking.
use std::collections::BTreeMap;

#[derive(Clone,Debug)]
pub struct Owners {allocations:Vec<(usize,usize)>,private:usize,bytes:usize}
impl Owners {
    pub fn new(mut allocations:Vec<(usize,usize)>)->Self {
        allocations.sort_unstable();let mut used=0;let mut private=0usize;
        for at in 0..allocations.len() {
            let (id,raw)=allocations[at];if raw==0 {continue;}
            let bytes=raw.checked_add(4095).map_or(usize::MAX,|n|n/4096*4096);
            if id==0 {private=private.saturating_add(bytes);continue;}
            if used>0 && allocations[used-1].0==id {allocations[used-1].1=allocations[used-1].1.max(bytes);}
            else {allocations[used]=(id,bytes);used+=1;}
        }
        allocations.truncate(used);
        let bytes=allocations.iter().fold(private,|sum,&(_,n)|sum.saturating_add(n));
        Self {allocations,private,bytes}
    }
    pub fn bytes(&self)->usize {self.bytes}
    pub fn private_bytes(&self)->usize {self.private}
    pub fn allocations(&self)->&[(usize,usize)] {&self.allocations}
}

#[derive(Default)]
pub struct Ledger {owners:BTreeMap<usize,(usize,usize)>,bytes:usize}
impl Ledger {
    pub const fn new()->Self {Self {owners:BTreeMap::new(),bytes:0}}
    /// Constant-time read. No owner walk or allocation on a pressure pass.
    pub fn bytes(&self)->usize {self.bytes}
    #[cfg(all(test,target_os="linux"))]
    pub fn owner_count(&self)->usize {self.owners.len()}
    pub fn additional(&self,owners:&Owners)->Option<usize> {
        owners.allocations.iter().try_fold(owners.private,|sum,&(id,bytes)|match self.owners.get(&id) {
            Some(&(prior,_)) if prior==bytes=>Some(sum),Some(_)=>None,None=>sum.checked_add(bytes)})
    }
    /// Call only after additional and the cap check succeed, with no intervening change.
    pub fn retain(&mut self,owners:&Owners) {
        let added=self.additional(owners).expect("validated state owner identities");
        self.bytes=self.bytes.checked_add(added).expect("validated retained state total");
        for &(id,bytes) in &owners.allocations {let entry=self.owners.entry(id).or_insert((bytes,0));entry.1+=1;}
    }
    pub fn release(&mut self,owners:&Owners) {
        self.bytes-=owners.private;
        for &(id,_) in &owners.allocations {
            let entry=self.owners.get_mut(&id).expect("retained state owner exists");entry.1-=1;
            if entry.1==0 {let bytes=entry.0;self.owners.remove(&id);self.bytes-=bytes;}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_and_unrelated_owners_leave_only_on_the_last_release() {
        let first=Owners::new(vec![(1,4096),(2,8192)]);let longer=Owners::new(vec![(1,4096),(3,12288)]);
        let unrelated=Owners::new(vec![(4,4096),(5,8192)]);let mut ledger=Ledger::new();
        ledger.retain(&first);assert_eq!(ledger.bytes(),12288);
        ledger.retain(&first);assert_eq!(ledger.bytes(),12288);
        ledger.retain(&longer);assert_eq!(ledger.bytes(),24576);
        ledger.retain(&unrelated);assert_eq!(ledger.bytes(),36864);
        ledger.release(&first);assert_eq!(ledger.bytes(),36864);
        ledger.release(&first);assert_eq!(ledger.bytes(),28672);
        ledger.release(&longer);assert_eq!(ledger.bytes(),12288);
        ledger.release(&unrelated);assert_eq!(ledger.bytes(),0);
    }
    #[test]
    fn unproven_identity_is_private_and_rounding_covers_unpooled_bytes() {
        let owners=Owners::new(vec![(1,4097),(1,4097),(2,1),(0,1)]);
        assert_eq!((owners.bytes(),owners.private_bytes()),(16384,4096));
        let mut ledger=Ledger::new();ledger.retain(&owners);ledger.retain(&owners);
        assert_eq!(ledger.bytes(),20480);ledger.release(&owners);assert_eq!(ledger.bytes(),16384);
        ledger.release(&owners);assert_eq!(ledger.bytes(),0);
        assert_eq!(Owners::new(vec![(1,usize::MAX)]).bytes(),usize::MAX);
        ledger.retain(&Owners::new(vec![(1,4096)]));
        assert_eq!(ledger.additional(&Owners::new(vec![(1,8192)])),None,"conflicting identity cannot earn sharing credit");
    }
}
