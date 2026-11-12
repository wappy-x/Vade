//! Which words of one shared allocation live views still read.
use std::{cell::RefCell,rc::{Rc,Weak}};
/// The storage word where each view of one allocation ends. A view is live as long as any
/// handle of it is, whoever holds the handle. The words past the last live end are read by
/// nobody, so an append may write them.
pub struct Ends<T> {views:RefCell<Vec<(Weak<T>,usize)>>,
    /// The list length at which dead views are next dropped: twice the live views of the last
    /// pass. A decode step lists hundreds of views of one cache; a pass at every add would
    /// read the whole list each time. `last` reads only live views, so its answer is the same.
    purge:std::cell::Cell<usize>}
impl<T> Default for Ends<T> {fn default()->Self {Self {views:RefCell::new(Vec::new()),purge:std::cell::Cell::new(0)}}}
impl<T> Ends<T> {
    /// List a view that ends at storage word `end`.
    pub fn add(&self,view:&Rc<T>,end:usize) {let mut views=self.views.borrow_mut();
        if views.len()>=self.purge.get() {views.retain(|v|v.0.strong_count()>0);self.purge.set((views.len()*2).max(16));}
        views.push((Rc::downgrade(view),end));}
    /// The largest end of a live view, 0 when no view lives.
    pub fn last(&self)->usize {self.views.borrow().iter().filter(|v|v.0.strong_count()>0).map(|v|v.1).max().unwrap_or(0)}
}
/// Whether an append may write its new rows into held storage itself. Only storage made
/// for appends (a cache's held rows) qualifies; a memo slot never does, so a prefix of a
/// memo hit always copies. The held rows must start the storage and be exactly `have`
/// words, the result must fit, and no live view may end past the held rows.
pub fn append_in_place(appendable:bool,view_start:usize,view_words:usize,have:usize,needed:usize,capacity:usize,last_end:usize)->bool {
    appendable && view_start==0 && view_words==have && needed<=capacity && last_end==have
}
/// Bytes of one memo slab chunk as allocated: whole 4 KiB pages, the device allocator's
/// unit. The chunk is allocated at exactly this size, so its page rounding is charged.
pub fn slab_chunk_bytes(rows:usize,width:usize)->Option<usize> {
    rows.checked_mul(width)?.checked_mul(4)?.checked_add(4095).map(|n|n/4096*4096)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_memo_slot_prefix_never_appends_in_place() {
        // A prefix of a memo hit whose parent is gone: starts the storage, is the only live
        // view and the suffix fits. Cache storage may append; a memo slot must copy.
        assert!(append_in_place(true,0,256,256,512,1024,256));
        assert!(!append_in_place(false,0,256,256,512,1024,256));
        assert!(!append_in_place(false,0,1024,1024,1024,1024,1024),"not even an empty suffix");
        // The cache rules themselves are unchanged.
        assert!(!append_in_place(true,1,256,256,512,1024,256),"a view that does not start the storage");
        assert!(!append_in_place(true,0,255,256,512,1024,256),"a view that is not exactly the held rows");
        assert!(!append_in_place(true,0,256,256,1025,1024,256),"a result past the capacity");
        assert!(!append_in_place(true,0,256,256,512,1024,300),"a live view past the held rows");
    }
    #[test]
    fn slab_chunk_bytes_include_the_allocator_page_rounding() {
        // Independent literals: 64 rows x 17 words x 4 B = 4,352 B, held as two 4 KiB pages.
        assert_eq!(slab_chunk_bytes(64,17),Some(8192));
        assert_eq!(slab_chunk_bytes(64,32768),Some(8_388_608),"GLM rows are page aligned already");
        assert_eq!(slab_chunk_bytes(1,1),Some(4096));
        assert_eq!(slab_chunk_bytes(0,17),Some(0));
        assert_eq!(slab_chunk_bytes(usize::MAX,2),None);
        // A cap of the logical 4,352 B does not admit the chunk; its real 8,192 B do.
        let cap=4352;assert!(slab_chunk_bytes(64,17).unwrap()>cap);
    }
    #[test]
    fn the_last_live_end_follows_every_view_and_every_handle() {
        let ends=Ends::<u8>::default();assert_eq!(ends.last(),0);
        let full=Rc::new(0u8);ends.add(&full,4);
        // A prefix, a suffix and a view inside the suffix. The suffix ends where the whole ends.
        let (prefix,suffix,nested)=(Rc::new(0u8),Rc::new(0u8),Rc::new(0u8));ends.add(&prefix,2);ends.add(&suffix,4);ends.add(&nested,4);
        drop(full);assert_eq!(ends.last(),4,"the suffix still reads words 2..4");
        drop(suffix);assert_eq!(ends.last(),4,"the view inside the suffix still reads word 3");
        // A second handle of a view keeps it live after the first handle is gone.
        let copy=nested.clone();drop(nested);assert_eq!(ends.last(),4);drop(copy);assert_eq!(ends.last(),2);
        // A rollback prefix is the last live end once the longer branch is gone; a competing branch is not.
        let (ahead,rolled)=(Rc::new(0u8),Rc::new(0u8));ends.add(&ahead,6);ends.add(&rolled,3);assert_eq!(ends.last(),6);
        drop(ahead);assert_eq!(ends.last(),3);drop(rolled);drop(prefix);assert_eq!(ends.last(),0);
    }
    #[test]
    fn many_adds_keep_the_last_live_end_and_a_bounded_list() {
        // A step's views of one cache: each lives for a while and goes. The list never holds
        // more than twice the live views plus the floor, and the last live end is exact.
        let ends=Ends::<u8>::default();let mut live:Vec<(Rc<u8>,usize)>=Vec::new();let mut state=5u32;
        for at in 0..4000usize {
            state=state.wrapping_mul(1664525).wrapping_add(1013904223);let end=1+(state>>8) as usize%500;
            let view=Rc::new(0u8);ends.add(&view,end);live.push((view,end));
            if at%3!=0 {live.remove((state>>20) as usize%live.len());}
            if at%200==199 {live.truncate(5);}
            assert_eq!(ends.last(),live.iter().map(|v|v.1).max().unwrap_or(0));
            assert!(ends.views.borrow().len()<=(2*live.len()+2).max(18)+200,"list {} live {}",ends.views.borrow().len(),live.len());
        }
        live.clear();assert_eq!(ends.last(),0);
    }
}
