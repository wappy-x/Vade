//! Recurrent projection metadata stays with one owner. Loaded row extents are checked on every use.
use super::{Result,Rows};
use crate::sparse_heads::Layout;
use std::collections::BTreeSet;
type Entries=Vec<(String,Layout)>;
#[derive(Default)]
pub(super) struct Plans {key:Option<usize>,entries:Entries}
impl Plans {
    // The containing Weights fixes its topology before preparing any matrices. A plan never leaves that owner.
    pub(super) fn prepare(&mut self,owner:usize,build:impl FnOnce()->Result<Entries>,
        mut rows:impl FnMut(&str)->Option<Rows>,local:&mut BTreeSet<String>)->Result<()> {
        if let Some(key)=self.key {
            if key!=owner {return Err("recurrent projection metadata owner changed".into());}
        } else {self.entries=build()?;self.key=Some(owner);}
        // Predictor weights can arrive after the plan. Do not cache their presence or the validation result.
        for (name,layout) in &self.entries {
            let Some(rows)=rows(name) else {continue;};
            let heads=layout.heads();
            if rows.total%layout.total_heads()!=0 {return Err("recurrent projection head extent differs".into());}
            let width=rows.total/layout.total_heads();
            if rows.start!=heads.start*width || rows.count!=layout.local_heads()*width {return Err("recurrent projection ownership differs".into());}
            local.insert(name.clone());
        }Ok(())
    }
}
#[cfg(test)] mod tests {
    use super::*;
    use engine_model::plan::layer::Topology;
    fn topology()->Topology {Topology {rank:1,parts:2}}
    fn entries()->Entries {vec![("projection".into(),Layout::new(8,4,4,topology()).unwrap())]}
    fn valid()->Rows {Rows {start:16,count:16,total:32,split:true}}
    #[test] fn later_loaded_rows_are_checked_without_rebuilding_metadata() {
        let mut plan=Plans::default();let mut local=BTreeSet::new();
        plan.prepare(1,||Ok(entries()),|_|None,&mut local).unwrap();assert!(local.is_empty());
        let no_build=||Err("metadata rebuilt".into());
        assert_eq!(plan.prepare(1,no_build,|_|Some(Rows {total:31,..valid()}),&mut local).unwrap_err(),"recurrent projection head extent differs");
        assert_eq!(plan.prepare(1,no_build,|_|Some(Rows {start:0,..valid()}),&mut local).unwrap_err(),"recurrent projection ownership differs");
        assert_eq!(plan.prepare(1,no_build,|_|Some(Rows {count:8,..valid()}),&mut local).unwrap_err(),"recurrent projection ownership differs");
        assert!(local.is_empty());plan.prepare(1,no_build,|_|Some(valid()),&mut local).unwrap();
        assert_eq!(local,BTreeSet::from(["projection".into()]));
    }
    #[test] fn repeated_sessions_reuse_the_plan_and_recheck_loaded_rows() {
        let mut plan=Plans::default();let mut local=BTreeSet::new();
        plan.prepare(1,||Ok(entries()),|_|Some(valid()),&mut local).unwrap();
        plan.prepare(1,||Err("metadata rebuilt".into()),|_|Some(valid()),&mut local).unwrap();
        assert!(plan.prepare(1,||Err("metadata rebuilt".into()),|_|Some(Rows {count:0,..valid()}),&mut local).is_err());
        assert_eq!(local.len(),1);
    }
    #[test] fn separate_owners_keep_separate_layouts() {
        let mut left=Plans::default();let mut right=Plans::default();let mut local=BTreeSet::new();
        left.prepare(1,||Ok(entries()),|_|Some(valid()),&mut local).unwrap();
        let layout=Layout::new(8,4,4,Topology {rank:0,parts:2}).unwrap();
        right.prepare(2,||Ok(vec![("projection".into(),layout)]),|_|Some(Rows {start:0,..valid()}),&mut local).unwrap();
        left.prepare(1,||Err("metadata rebuilt".into()),|_|Some(valid()),&mut local).unwrap();
        right.prepare(2,||Err("metadata rebuilt".into()),|_|Some(Rows {start:0,..valid()}),&mut local).unwrap();
    }
    #[test] fn failed_build_can_retry_but_a_ready_owner_cannot_change() {
        let mut plan=Plans::default();let mut local=BTreeSet::new();
        assert!(plan.prepare(1,||Err("bad plan".into()),|_|None,&mut local).is_err());
        plan.prepare(1,||Ok(entries()),|_|Some(valid()),&mut local).unwrap();
        assert_eq!(plan.prepare(2,||Ok(Vec::new()),|_|None,&mut local).unwrap_err(),"recurrent projection metadata owner changed");
    }
}
