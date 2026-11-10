//! Immutable row identity. Copies keep origins; arithmetic creates a fresh origin.
//! This metadata cannot assert equality of separately computed values.
use crate::Result;
/// `end` is the word of the lineage where this span ends: the words of every span up to and
/// with this one. It follows from the spans before it, so two lineages with equal spans have
/// equal ends and compare as they did without it. It lets a range find its first span by a
/// binary search: a selection reads thousands of rows of one lineage in a decode step.
#[derive(Debug,Clone,PartialEq,Eq,PartialOrd,Ord,Hash)]
pub struct Span {source:u64,start:usize,words:usize,end:usize}
/// The spans in order: `head`, then `tail`. Most values have one span (a fresh result, one
/// cache row), and that one lives in the value itself: no allocation to make, copy or drop
/// it. `head` is absent only when there is no span, so equal span lists have equal fields
/// and the derived order is the order of the span lists, then the words.
#[derive(Debug,Clone,Default,PartialEq,Eq,PartialOrd,Ord,Hash)]
pub struct Lineage {head:Option<Span>,tail:Vec<Span>,words:usize}
/// The memo charges a span as its identity (source, start, words). The end offset is derived,
/// so the charge and with it the rows the memo keeps stay as they were.
const SPAN_CHARGE:usize=24;
impl Lineage {
    pub fn fresh(source:u64,words:usize)->Self {Self {head:(words>0).then_some(Span {source,start:0,words,end:words}),tail:Vec::new(),words}}
    fn spans(&self)->impl Iterator<Item=&Span> {self.head.iter().chain(&self.tail)}
    fn count(&self)->usize {usize::from(self.head.is_some())+self.tail.len()}
    pub fn bytes(&self)->usize {self.count()*SPAN_CHARGE}
    pub fn range(&self,start:usize,words:usize)->Result<Self> {let mut out=Self::default();self.range_into(start,words,&mut out)?;Ok(out)}
    /// Append the identity of `words` words from `start` to `out`, as `range` then `join` do.
    pub fn range_into(&self,start:usize,words:usize,out:&mut Self)->Result<()> {
        let end=start.checked_add(words).filter(|&n|n<=self.words).ok_or("immutable range exceeds source")?;
        // Every span before the first one that ends past `start` lies before the range.
        let Some(head)=&self.head else {return Ok(());};
        let first=if head.end>start {0} else {1+self.tail.partition_point(|span|span.end<=start)};
        for span in std::iter::once(head).chain(&self.tail).skip(first) {let offset=span.end-span.words;if offset>=end {break;}
            let lo=start.max(offset);let hi=end.min(span.end);
            if lo<hi {out.push(Span {source:span.source,start:span.start+lo-offset,words:hi-lo,end:0})?;}
        }
        Ok(())
    }
    /// Empty this lineage and keep its storage, so a probe for many rows allocates nothing.
    pub fn clear(&mut self) {self.head=None;self.tail.clear();self.words=0;}
    fn push(&mut self,span:Span)->Result<()> {
        self.words=self.words.checked_add(span.words).ok_or("immutable join overflow")?;
        if let Some(last)=self.tail.last_mut().or(self.head.as_mut())&& last.source==span.source && last.start.checked_add(last.words)==Some(span.start) {last.words+=span.words;last.end=self.words;return Ok(());}
        if span.words>0 {let span=Span {end:self.words,..span};if self.head.is_none() {self.head=Some(span);} else {self.tail.push(span);}}Ok(())
    }
    pub fn join<'a>(parts:impl IntoIterator<Item=&'a Self>)->Result<Self> {
        let mut out=Self::default();for part in parts {for span in part.spans() {out.push(span.clone())?;}}Ok(out)
    }
    /// Only live gather rows have an origin. Padding is a fresh immutable zero.
    pub fn gather(&self,width:usize,ids:&[usize],slots:usize,padding:u64)->Result<Self> {
        if width==0 || ids.len()>slots {return Err("immutable gather geometry differs".into());}
        let mut out=Self::default();
        for &id in ids {let start=id.checked_mul(width).ok_or("immutable gather offset overflow")?;self.range_into(start,width,&mut out)?;}
        let words=(slots-ids.len()).checked_mul(width).ok_or("immutable gather padding overflow")?;
        if words>0 {out.push(Span {source:padding,start:0,words,end:0})?;}Ok(out)
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
    /// The slow forms the fast ones replace: a scan of every span for a range, and a join of one range a row for a gather.
    fn scan(x:&Lineage,start:usize,words:usize)->Lineage {
        let end=start+words;let mut out=Lineage::default();let mut offset=0;
        for span in x.spans() {let next=offset+span.words;let lo=start.max(offset);let hi=end.min(next);
            if lo<hi {out.push(Span {source:span.source,start:span.start+lo-offset,words:hi-lo,end:0}).unwrap();}offset=next;}
        out
    }
    #[test] fn searched_ranges_and_gathers_equal_the_scanned_forms() {
        // A cache as a decode step holds it: prompt groups of several rows, then one source a token.
        let width=3;let mut parts=Vec::new();let mut source=1;
        for rows in [5usize,1,7,2,4] {parts.push(Lineage::fresh(source,rows*width));source+=1;}
        for _ in 0..9 {parts.push(Lineage::fresh(source,width));source+=1;}
        let cache=Lineage::join(&parts).unwrap();let rows=cache.words/width;assert_eq!(rows,28);
        let mut ends=0;for span in cache.spans() {ends+=span.words;assert_eq!(span.end,ends);}assert_eq!(ends,cache.words);assert_eq!(cache.count(),14);
        for start in 0..=cache.words {for words in 0..=cache.words-start {
            let got=cache.range(start,words).unwrap();assert_eq!(got,scan(&cache,start,words),"range {start}+{words}");
            let mut sum=0;for span in got.spans() {sum+=span.words;assert_eq!(span.end,sum);}assert_eq!(sum,got.words);assert_eq!(got.head.is_none(),got.count()==0);
        }}
        // Selections in any order, with runs that merge and rows that repeat a source.
        for ids in [vec![],vec![0],vec![27,26,25],vec![3,4,5,6,20,21,9,0,1,2],(0..rows).collect::<Vec<_>>(),(0..rows).rev().collect::<Vec<_>>(),vec![4,5,12,13,14,15,16,27]] {
            for slots in [ids.len(),ids.len()+2] {
                let slow={let mut all=ids.iter().map(|&id|scan(&cache,id*width,width)).collect::<Vec<_>>();all.push(Lineage::fresh(99,(slots-ids.len())*width));Lineage::join(&all).unwrap()};
                let got=cache.gather(width,&ids,slots,99).unwrap();assert_eq!(got,slow,"gather {ids:?} into {slots}");
                // A row of the gathered value has the identity of the cache row it copies: the memo key.
                let mut probe=Lineage::default();
                for (slot,&id) in ids.iter().enumerate() {
                    assert_eq!(got.range(slot*width,width).unwrap(),cache.range(id*width,width).unwrap());
                    probe.clear();got.range_into(slot*width,width,&mut probe).unwrap();assert_eq!(probe,scan(&got,slot*width,width));
                }
            }
        }
        assert_eq!(Lineage::fresh(7,2).bytes(),24,"the memo charge of a span does not count its derived end");
    }
    /// The order of two lineages is the order of their span lists, then their words, as it was with one list.
    #[test] fn the_order_is_the_order_of_the_span_lists() {
        let list=|x:&Lineage|(x.spans().map(|s|(s.source,s.start,s.words)).collect::<Vec<_>>(),x.words);
        let a=Lineage::fresh(1,4);let b=Lineage::fresh(2,4);let mut all=vec![Lineage::default(),a.clone(),b.clone(),a.range(1,2).unwrap(),a.range(0,2).unwrap(),
            Lineage::join([&a,&b]).unwrap(),Lineage::join([&b,&a]).unwrap(),Lineage::join([&a,&b,&a]).unwrap(),Lineage::join([&a.range(0,2).unwrap(),&b]).unwrap(),Lineage::join([&a,&a.range(2,2).unwrap()]).unwrap()];
        for x in &all {for y in &all {assert_eq!(x.cmp(y),list(x).cmp(&list(y)),"{x:?} {y:?}");assert_eq!(x==y,list(x)==list(y));}}
        all.sort();assert!(all.windows(2).all(|pair|list(&pair[0])<=list(&pair[1])));
        assert_eq!(Lineage::join([&a,&b]).unwrap().bytes(),48);assert_eq!(Lineage::default().bytes(),0);
    }
}
