//! The rows each session asked of one projection last: the source of the answers made ahead of a selection read.
use std::rc::Weak;
/// Lists kept for one projection: one a session, as many as a service has sessions open.
pub(super) const SESSIONS:usize=8;
/// One projection's lists, the one written longest ago first. `T` is what a session's cache is made of. A list holds
/// a weak handle to the cache it was asked of: the cache of a session that waits for its turn lives in that session's
/// state, the cache of a session that ended is gone. So the two can be told apart without a call when a session closes.
pub(super) struct Asked<T> {lists:Vec<(u64,Weak<T>,Vec<u32>)>}
impl<T> Default for Asked<T> {fn default()->Self {Self {lists:Vec::new()}}}
impl<T> Asked<T> {
    /// The list of the session whose cache starts with the source `cache`.
    pub(super) fn get(&self,cache:u64)->Option<&[u32]> {self.lists.iter().find(|list|list.0==cache).map(|list|list.2.as_slice())}
    pub(super) fn len(&self)->usize {self.lists.len()}
    /// Writes one session's list. When SESSIONS other lists stand, one goes: a list of a session that ended, the one
    /// written longest ago of those; a list of an open session only when every list is of an open session.
    pub(super) fn set(&mut self,cache:u64,alive:Weak<T>,rows:Vec<u32>) {
        self.lists.retain(|list|list.0!=cache);
        if self.lists.len()>=SESSIONS {let gone=self.lists.iter().position(|list|list.1.strong_count()==0).unwrap_or(0);self.lists.remove(gone);}
        self.lists.push((cache,alive,rows));
    }
}
#[cfg(test)] mod tests {
    use super::{Asked,SESSIONS};
    use std::rc::Rc;
    /// SESSIONS open sessions, written in order: source `k` has the list `[k]`.
    fn full()->(Asked<()>,Vec<Option<Rc<()>>>) {
        let mut asked=Asked::default();let mut open=Vec::new();
        for k in 0..SESSIONS as u64 {let cache=Rc::new(());asked.set(k,Rc::downgrade(&cache),vec![k as u32]);open.push(Some(cache));}
        (asked,open)
    }
    #[test] fn the_list_of_a_session_that_ended_goes_before_the_list_of_an_open_session() {
        let (mut asked,mut open)=full();
        // Session 0 waits longest. Session 3 ends. A ninth session asks: the list of 3 goes, not the list of 0.
        open[3]=None;let ninth=Rc::new(());asked.set(100,Rc::downgrade(&ninth),vec![100]);
        assert_eq!(asked.len(),SESSIONS);assert!(asked.get(3).is_none());assert_eq!(asked.get(0),Some(&[0u32][..]));assert_eq!(asked.get(100),Some(&[100u32][..]));
        // Two more end (5, then 1 is still open): of the ended ones the one written longest ago goes first.
        open[5]=None;open[6]=None;let tenth=Rc::new(());asked.set(101,Rc::downgrade(&tenth),vec![101]);
        assert!(asked.get(5).is_none());assert!(asked.get(6).is_some());assert!(asked.get(0).is_some());
    }
    /// The key of a session's list is the first source of its CACHE. A gather's output starts with the first selected
    /// row: when that row is past the cache's first span, the output's first source is another one, and a list
    /// stored under it is never found by the next step (the first MEMO-SHARE build stored it there).
    #[test] fn a_selection_that_starts_past_the_first_span_reads_the_list_it_wrote() {
        use super::super::lineage::Lineage;
        // Rows of 3 words: rows 0 to 3 are the first span (source 7), rows 4 and 5 were appended (sources 8 and 9).
        let cache=Lineage::join([&Lineage::fresh(7,12),&Lineage::fresh(8,3),&Lineage::fresh(9,3)]).unwrap();
        let output=cache.gather(3,&[4,5],2,99).unwrap();
        assert_eq!((cache.first_source(),output.first_source()),(Some(7),Some(8)));
        let alive=Rc::new(());let mut asked=Asked::default();
        asked.set(cache.first_source().unwrap(),Rc::downgrade(&alive),vec![4,5]);
        // The next step appends a row and looks the list up with its longer cache: the same key, the current list.
        let longer=Lineage::join([&cache,&Lineage::fresh(10,3)]).unwrap();
        assert_eq!(asked.get(longer.first_source().unwrap()),Some(&[4u32,5][..]));
        // Under the output's first source the list would be out of reach of that lookup.
        let mut wrong=Asked::default();wrong.set(output.first_source().unwrap(),Rc::downgrade(&alive),vec![4,5]);
        assert!(wrong.get(longer.first_source().unwrap()).is_none());
    }
    #[test] fn with_every_session_open_the_list_written_longest_ago_goes() {
        let (mut asked,_open)=full();
        let ninth=Rc::new(());asked.set(100,Rc::downgrade(&ninth),vec![100]);
        assert_eq!(asked.len(),SESSIONS);assert!(asked.get(0).is_none());
        for k in 1..SESSIONS as u64 {assert_eq!(asked.get(k),Some(&[k as u32][..]));}
    }
    #[test] fn a_session_that_asks_again_replaces_its_own_list_and_takes_no_place() {
        let (mut asked,open)=full();
        // Session 0 asks again: its list is the newest now, and no other list goes.
        asked.set(0,Rc::downgrade(open[0].as_ref().unwrap()),vec![7,8]);
        assert_eq!(asked.len(),SESSIONS);assert_eq!(asked.get(0),Some(&[7u32,8][..]));
        let ninth=Rc::new(());asked.set(100,Rc::downgrade(&ninth),vec![100]);
        assert!(asked.get(1).is_none());assert_eq!(asked.get(0),Some(&[7u32,8][..]));
    }
}
