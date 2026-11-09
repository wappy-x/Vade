//! Immutable row identity. Copies keep origins; arithmetic creates a fresh origin.
//! This metadata cannot assert equality of separately computed values.
use crate::Result;
#[derive(Debug,Clone,PartialEq,Eq,PartialOrd,Ord)]
pub struct Span {source:u64,start:usize,words:usize}
#[derive(Debug,Clone,Default,PartialEq,Eq,PartialOrd,Ord)]
pub struct Lineage {spans:Vec<Span>,words:usize}
impl Lineage {
    pub fn fresh(source:u64,words:usize)->Self {Self {spans:if words==0 {Vec::new()} else {vec![Span {source,start:0,words}]},words}}
    pub fn bytes(&self)->usize {self.spans.len()*std::mem::size_of::<Span>()}
    pub fn range(&self,start:usize,words:usize)->Result<Self> {
        let end=start.checked_add(words).filter(|&n|n<=self.words).ok_or("immutable range exceeds source")?;
        let mut out=Self::default();let mut offset=0;
        for span in &self.spans {let next=offset+span.words;let lo=start.max(offset);let hi=end.min(next);
            if lo<hi {out.push(Span {source:span.source,start:span.start+lo-offset,words:hi-lo})?;}offset=next;
        }
        Ok(out)
    }
    fn push(&mut self,span:Span)->Result<()> {
        self.words=self.words.checked_add(span.words).ok_or("immutable join overflow")?;
        if let Some(last)=self.spans.last_mut() {if last.source==span.source && last.start.checked_add(last.words)==Some(span.start) {last.words+=span.words;return Ok(());}}
        if span.words>0 {self.spans.push(span);}Ok(())
    }
    pub fn join<'a>(parts:impl IntoIterator<Item=&'a Self>)->Result<Self> {
        let mut out=Self::default();for part in parts {for span in &part.spans {out.push(span.clone())?;}}Ok(out)
    }
    /// Only live gather rows have an origin. Padding is a fresh immutable zero.
    pub fn gather(&self,width:usize,ids:&[usize],slots:usize,padding:u64)->Result<Self> {
        if width==0 || ids.len()>slots {return Err("immutable gather geometry differs".into());}
        let mut parts=Vec::with_capacity(ids.len()+1);
        for &id in ids {let start=id.checked_mul(width).ok_or("immutable gather offset overflow")?;parts.push(self.range(start,width)?);}
        let words=(slots-ids.len()).checked_mul(width).ok_or("immutable gather padding overflow")?;parts.push(Self::fresh(padding,words));
        Self::join(&parts)
    }
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn range_and_join_preserve_original_identity() {
        let x=Lineage::fresh(1,12);assert!(x.bytes()>0);let a=x.range(0,4).unwrap();let b=x.range(4,8).unwrap();
        assert_eq!(Lineage::join([&a,&b]).unwrap(),x);
        assert_eq!(Lineage::join([&a,&b]).unwrap().range(3,7).unwrap(),x.range(3,7).unwrap());
        assert_ne!(x,Lineage::fresh(2,12));assert!(x.range(usize::MAX,1).is_err());assert!(x.range(12,1).is_err());
        assert_eq!(x.range(12,0).unwrap(),Lineage::default());
    }
    #[test] fn gather_preserves_selected_rows_and_fresh_padding() {
        let x=Lineage::fresh(1,12);let y=x.gather(3,&[3,1],3,2).unwrap();
        assert_eq!(y.range(0,3).unwrap(),x.range(9,3).unwrap());assert_eq!(y.range(3,3).unwrap(),x.range(3,3).unwrap());
        assert_ne!(y.range(6,3).unwrap(),Lineage::fresh(3,3));assert!(x.gather(3,&[4],1,2).is_err());
        assert!(x.gather(3,&[0,1],1,2).is_err());assert!(x.gather(0,&[],1,2).is_err());
    }
}
