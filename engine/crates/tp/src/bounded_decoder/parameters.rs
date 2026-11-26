//! Immutable parameter identities for scalar recipe operations. No device arithmetic.
#![forbid(unsafe_code)]
use std::{collections::BTreeMap,hash::{Hash,Hasher},rc::Rc};
use crate::Result;
use engine_model::plan::layer::{Kind,Step};
type Table=Rc<BTreeMap<String,Vec<f32>>>;
/// One ordered weight list from one immutable checkpoint parameter table.
/// Fields are private. The cache retains a weak identity, not another checkpoint copy.
pub struct StaticParameters {table:Table,names:Vec<String>,values:Vec<f32>,bits:Vec<u32>,hash:u64}
impl StaticParameters {
    fn new(table:Table,names:Vec<String>)->Result<Self> {
        let mut values=Vec::new();for name in &names {values.extend_from_slice(table.get(name).ok_or_else(||format!("missing static parameter {name}"))?);}
        let bits=values.iter().map(|v|v.to_bits()).collect::<Vec<_>>();let mut hash=std::collections::hash_map::DefaultHasher::new();bits.hash(&mut hash);
        Ok(Self {table,names,values,bits,hash:hash.finish()})
    }
    pub fn values(&self)->&[f32] {&self.values}
    pub fn bits(&self)->&[u32] {&self.bits}
    pub fn content_hash(&self)->u64 {self.hash}
    fn check(&self,table:&Table,step:&Step) {
        // The decoder and Prepared retain the same immutable table. Neither exposes
        // mutable parameter slices. Only the token-dependent shape and inputs vary.
        debug_assert!(Rc::ptr_eq(&self.table,table),"static parameter table changed");
        debug_assert!(self.names.iter().map(String::as_str).eq(step.weights.iter().map(|w|w.name.as_str())),"static parameter order changed");
    }
}
#[derive(Default)]
pub(super) struct StaticParameterPlan {layers:BTreeMap<usize,BTreeMap<String,Rc<StaticParameters>>>}
impl StaticParameterPlan {
    pub(super) fn admits(step:&Step)->bool {matches!(step.kind,Kind::Conv|Kind::KdaPrepare|Kind::GatedNorm|Kind::KeyNorm|Kind::Pool|Kind::Route) && !step.weights.is_empty()}
    pub(super) fn new(table:Table,jobs:Vec<(usize,String,Vec<String>)>)->Result<Self> {
        let mut result=Self::default();for (layer,id,names) in jobs {let weights=Rc::new(StaticParameters::new(table.clone(),names)?);
            if result.layers.entry(layer).or_default().insert(id,weights).is_some() {return Err("duplicate static parameter operation".into());}}
        Ok(result)
    }
    pub(super) fn get(&self,layer:usize,table:&Table,step:&Step)->Option<&Rc<StaticParameters>> {
        let value=self.layers.get(&layer)?.get(&step.id)?;value.check(table,step);Some(value)
    }
}
#[derive(Clone,Copy,Default,Debug,PartialEq,Eq)]
pub struct ParameterCounts {pub key_builds:u64,pub static_hits:u64,pub static_binds:u64}
/// The content map preserves the original upload deduplication. Static identities
/// reach that same owner once, then bypass content scans. Hashes alone are never keys.
#[cfg(any(target_os="linux",test))]
pub(super) struct ParameterCache<T> {contents:BTreeMap<Vec<u32>,Rc<T>>,bindings:BTreeMap<(usize,u64),(std::rc::Weak<StaticParameters>,Rc<T>)>,counts:ParameterCounts}
#[cfg(any(target_os="linux",test))]
impl<T> Default for ParameterCache<T> {fn default()->Self {Self {contents:BTreeMap::new(),bindings:BTreeMap::new(),counts:ParameterCounts::default()}}}
#[cfg(any(target_os="linux",test))]
impl<T> ParameterCache<T> {
    pub(super) fn counts(&self)->ParameterCounts {self.counts}
    #[cfg(any(feature="kernel-bench",test))]
    pub(super) fn clear(&mut self) {self.bindings.clear();self.contents.clear();self.counts=ParameterCounts::default();}
    pub(super) fn resolve(&mut self,values:&[f32],prepared:Option<&Rc<StaticParameters>>,upload:impl FnOnce(&[f32])->Result<T>)->Result<Option<Rc<T>>> {
        if values.is_empty() {return Ok(None);}
        if let Some(p)=prepared {
            debug_assert!(std::ptr::eq(values.as_ptr(),p.values().as_ptr()) && values.len()==p.values().len(),"static parameter slice changed");
            let identity=(Rc::as_ptr(p) as usize,p.content_hash());
            if let Some((owner,value))=self.bindings.get(&identity) {debug_assert_eq!(owner.as_ptr(),Rc::as_ptr(p));self.counts.static_hits+=1;return Ok(Some(value.clone()));}
            // A weak pointer holds the identity allocation until removal, so its
            // address cannot alias another table. Reap ended sessions on cold binds.
            self.bindings.retain(|_,(owner,_)|owner.strong_count()!=0);
            let value=if let Some(value)=self.contents.get(p.bits()) {value.clone()} else {
                let value=Rc::new(upload(values)?);self.contents.insert(p.bits().to_vec(),value.clone());value};
            self.bindings.insert(identity,(Rc::downgrade(p),value.clone()));self.counts.static_binds+=1;return Ok(Some(value));
        }
        self.counts.key_builds+=1;let key=values.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        if let Some(value)=self.contents.get(&key) {return Ok(Some(value.clone()));}
        let value=Rc::new(upload(values)?);self.contents.insert(key,value.clone());Ok(Some(value))
    }
}
#[cfg(test)] #[path="parameters/tests.rs"] mod tests;
