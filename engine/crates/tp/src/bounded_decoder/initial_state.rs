//! Initial state metadata belongs to prepared checkpoint parameters, never to a request's values.
use std::{cell::RefCell,rc::Rc};
use super::Result;
/// Full-head state names and word counts in layer order. Local head ownership is applied at allocation time.
pub(super) type Layout=Vec<Vec<(String,usize)>>;
#[derive(Default)]
pub(super) struct Layouts {zero:RefCell<Option<Rc<Layout>>>}
impl Layouts {
    pub(super) fn get(&self,position:u64,build:impl FnOnce()->Result<Layout>)->Result<Rc<Layout>> {
        if position==0 {if let Some(layout)=self.zero.borrow().as_ref() {return Ok(layout.clone());}}
        let layout=Rc::new(build()?);
        if position==0 {*self.zero.borrow_mut()=Some(layout.clone());}
        Ok(layout)
    }
}
#[cfg(test)] mod tests {
    use super::*;
    fn layout(words:usize)->Layout {vec![vec![("state".into(),words)]]}
    #[test] fn prepared_owners_share_one_successful_initial_layout() {
        let cache=Rc::new(Layouts::default());let prepared=cache.clone();
        let a=cache.get(0,||Ok(layout(12))).unwrap();
        let b=prepared.get(0,||Err("initial layout was rebuilt".into())).unwrap();
        assert!(Rc::ptr_eq(&a,&b));assert_eq!(a[0][0].1,12);
    }
    #[test] fn diagnostic_positions_neither_seed_nor_reuse_the_initial_layout() {
        let cache=Layouts::default();
        assert_eq!(cache.get(7,||Ok(layout(70))).unwrap()[0][0].1,70);
        let initial=cache.get(0,||Ok(layout(12))).unwrap();
        assert_eq!(cache.get(7,||Ok(layout(71))).unwrap()[0][0].1,71);
        assert!(Rc::ptr_eq(&initial,&cache.get(0,||Err("zero changed".into())).unwrap()));
    }
    #[test] fn failed_build_is_not_kept_and_success_remains_reusable() {
        let cache=Layouts::default();assert!(cache.get(0,||Err("invalid layout".into())).is_err());
        let good=cache.get(0,||Ok(layout(8))).unwrap();
        assert!(Rc::ptr_eq(&good,&cache.get(0,||Err("good layout was rebuilt".into())).unwrap()));
    }
    #[test] fn separate_checkpoint_owners_do_not_share_layouts() {
        let (a,b)=(Layouts::default(),Layouts::default());
        assert_eq!(a.get(0,||Ok(layout(8))).unwrap()[0][0].1,8);
        assert_eq!(b.get(0,||Ok(layout(16))).unwrap()[0][0].1,16);
    }
}
