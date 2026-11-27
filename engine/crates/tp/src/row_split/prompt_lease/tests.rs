use super::*;
use crate::bounded_decoder::{Host,TestCheckpoint};
use engine_model::safetensors::Dtype;
use spark_serve::{Engine as _,Session as _};
#[derive(Default)]
struct Observed {spares:RefCell<Vec<u8>>,returns:Cell<usize>,phases:Cell<usize>,fail:Cell<bool>}
// Delegate arithmetic to the host twin. Only scratch ownership and faults differ.
macro_rules! host {($(fn $name:ident($($arg:ident:$ty:ty),*)->$out:ty;)*)=>{$(
    fn $name(&self,$($arg:$ty),*)->$out {Host.$name($($arg),*)}
)*};}
impl Compute for Observed {
    type Value=Vec<f32>;type Pass=<Host as Compute>::Pass;
    host! {
        fn mhc_row_launches()->u64;
        fn prepare_beside(c:&CheckpointAdmission,live:usize)->Result<()>;
        fn rows_greedy(x:&Self::Value,rows:usize)->Result<Option<Vec<u32>>>;
        fn has_latent_absorption(name:&str)->bool;
        fn latent_absorption(name:&str,s:Shape,a:&Self::Value,b:&Self::Value,c:&Self::Value,n:usize)->Result<Self::Value>;
        fn latent_absorption_rows(name:&str,s:spark_gpu_probe::prompt_ops::Shape,a:&Self::Value,b:&Self::Value,c:&Self::Value)->Result<Self::Value>;
        fn latent_absorption_chats(name:&str,s:spark_gpu_probe::prompt_ops::Shape,a:&[crate::bounded_decoder::ChatGather<'_,Self::Value>],n:usize,b:&Self::Value,c:&Self::Value)->Result<Option<Self::Value>>;
        fn upload(x:&[f32])->Result<Self::Value>;
        fn read(x:&Self::Value)->Result<Vec<f32>>;
        fn select_known(a:&Self::Value,b:&Self::Value,x:&[f32])->Result<()>;
        fn range(x:&Self::Value,start:usize,len:usize)->Result<Self::Value>;
        fn words(x:&Self::Value)->usize;
        fn join(x:&[&Self::Value])->Result<Self::Value>;
        fn zeros(n:usize)->Result<Self::Value>;
        fn operation(s:Shape,x:[Option<&Self::Value>;4],w:&[f32])->Result<Self::Value>;
        fn gather_absorbed(s:Shape,x:[Option<&Self::Value>;4],n:usize,f:f32)->Result<Self::Value>;
        fn norm(name:&str,s:NormShape,x:&Self::Value,w:&[f32])->Result<Self::Value>;
        fn begin(key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Self::Value)->Result<(Self::Pass,Self::Value)>;
        fn finish(p:Self::Pass,x:&Self::Value)->Result<Self::Value>;
        fn mx(p:&Mxfp8Projection<'_>,x:&Self::Value)->Result<Self::Value>;
        fn round(x:&Self::Value)->Result<Self::Value>;
        fn mean(x:&Self::Value,streams:u32,hidden:u32)->Result<Self::Value>;
        fn prompt(s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&Self::Value>;5],w:&[f32])->Result<Self::Value>;
        fn prompt_end()->Result<()>;
        fn prompt_seeds()->bool;
        fn rows_norm(name:&str,s:NormShape,x:&Self::Value,m:usize,w:&[f32])->Result<Self::Value>;
        fn rows_begin(key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Self::Value,m:usize)->Result<(Self::Pass,Self::Value)>;
        fn rows_finish(p:Self::Pass,x:&Self::Value,m:usize)->Result<Self::Value>;
    }
    fn state_allocations(&self,v:&Self::Value,out:&mut Vec<(usize,usize)>)->bool {Host.state_allocations(v,out)}
    fn memory_bytes(&self)->Option<(usize,usize,usize)> {Some((self.spares.borrow().capacity(),0,0))}
    fn rows_keep(&self)->Result<()> {self.spares.replace(Vec::new());self.returns.set(self.returns.get()+1);Ok(())}
    fn rows_phase_end(&self)->Result<()> {self.phases.set(self.phases.get()+1);Ok(())}
    fn rows_embedding(&self,_:&[u32],_:usize)->Result<Option<Self::Value>> {
        self.spares.replace(vec![0;4096]);
        if self.fail.replace(false) {return Err("injected prompt allocation failure".into());}Ok(None)
    }
}
struct Wire {peer:Cell<usize>,commands:RefCell<Vec<u32>>}
impl Wire {fn new()->Self {Self {peer:Cell::new(1024),commands:RefCell::new(Vec::new())}}}
impl Exchange for Wire {
    fn topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn gather(&mut self,local:&[f32],n:usize)->Result<Vec<f32>> {
        assert_eq!(local.len(),n);let mut out=local.to_vec();
        if n==4 {let own=(0..4).map(|i|(local[i] as u64)<<(16*i)).sum::<u64>() as usize;let min=own.min(self.peer.get());
            for (i,v) in out.iter_mut().enumerate() {*v=((min as u64>>(16*i))&65535) as f32;}}
        Ok(out)
    }
    fn command(&mut self,words:&mut [u8])->Result<()> {if words.len()==16 {self.commands.borrow_mut().push(u32::from_le_bytes(words[..4].try_into().unwrap()));}Ok(())}
    fn complete(&mut self)->Result<()> {Ok(())}
}
fn engine(c:&CheckpointAdmission)->Factory<'_,Observed,Wire> {
    let mut e=Factory::new(c,Split::new(Observed::default(),Wire::new()).unwrap(),8193,vec![0]).unwrap().with_prompt_keep();
    // Test geometry has no draft layer. Exercise the workspace policy without
    // enabling predictor arithmetic in the tiny host model.
    e.predictor=3;e.memo_admitted=true;e.init_prompt_lease_plan().unwrap();e
}
/// WIDE PHASE: the drain of a wide pass reaches the compute through both wrappers (`Split` and `&C`) and is not the
/// group's end: the pool's spares stay. A wrapper that did not forward it would fall to the default, which trims them.
#[test]
fn phase_completion_crosses_both_wrappers_without_returning_group_spares() {
    let compute=Observed::default();compute.spares.replace(vec![0;4096]);
    let split=Split::new(&compute,Wire::new()).unwrap();
    split.rows_phase_end().unwrap();
    assert_eq!((compute.phases.get(),compute.returns.get()),(1,0));
    assert!(compute.spares.borrow().capacity()>=4096,"a phase end keeps the pool's spares");
    split.rows_keep().unwrap();assert_eq!((compute.phases.get(),compute.returns.get()),(1,1));
    assert_eq!(compute.spares.borrow().capacity(),0);
}
#[test]
fn returned_lease_releases_actual_host_scratch_allocation() {
    let c=checkpoint();let e=engine(&c.admission);let mut s=e.session().unwrap();
    let workspace=e.workspace;let bytes=e.prompt_lease_extra(1024).unwrap();e.prepare_prompt_workspace(bytes).unwrap();assert_eq!(s.prefill_chunk_size(8192),1024);
    s.prefill_advance(&vec![1;1024]).unwrap();assert_eq!(e.split.compute.spares.borrow().capacity(),0);assert!(e.split.compute.returns.get()>0);
    assert_eq!((e.lease_bytes.get(),e.lease_rows.get(),e.workspace),(0,0,workspace));assert_eq!(s.prefill_chunk_size(8192),256);
}
#[test]
fn unequal_rank_offers_choose_smaller_and_eight_chats_cannot_borrow() {
    let c=checkpoint();let e=engine(&c.admission);let a=e.session().unwrap();
    e.split.exchange.borrow().peer.set(512);e.prepare_prompt_workspace(usize::MAX).unwrap();assert_eq!(e.current_prompt_group(),512);e.finish_prompt_workspace().unwrap();
    e.split.exchange.borrow().peer.set(1024);let others=(0..7).map(|_|e.session().unwrap()).collect::<Vec<_>>();
    e.prepare_prompt_workspace(usize::MAX).unwrap();assert_eq!(e.current_prompt_group(),256);assert_eq!(e.lease_bytes.get(),0);drop(others);drop(a);
}
#[test]
fn arrival_after_large_group_has_original_pool_and_group() {
    let c=checkpoint();let e=engine(&c.admission);let mut a=e.session().unwrap();
    e.prepare_prompt_workspace(usize::MAX).unwrap();a.prefill_advance(&vec![1;1024]).unwrap();
    // Scheduler intake resumes only after this synchronous call. The arriving
    // session must see freed scratch, not just a lower ledger charge.
    let b=e.session().unwrap();assert_eq!(e.split.compute.spares.borrow().capacity(),0);
    e.prepare_prompt_workspace(0).unwrap();assert_eq!(a.prefill_chunk_size(8192),256);drop(b);
}
#[test]
fn unused_offer_cancel_and_mid_prompt_error_release_once() {
    let c=checkpoint();let e=engine(&c.admission);let mut a=e.session().unwrap();
    e.prepare_prompt_workspace(usize::MAX).unwrap();let n=e.split.compute.returns.get();
    e.finish_prompt_workspace().unwrap();e.finish_prompt_workspace().unwrap();assert_eq!(e.split.compute.returns.get(),n+1);
    e.prepare_prompt_workspace(usize::MAX).unwrap();e.split.compute.fail.set(true);
    assert!(a.prefill_advance(&vec![1;1024]).unwrap_err().contains("injected"));
    assert_eq!(e.split.compute.spares.borrow().capacity(),0);assert_eq!(e.lease_bytes.get(),0);assert!(e.failed.get());
}
#[test]
fn cached_8192_resume_borrows_only_for_its_suffix() {
    let c=checkpoint();let e=engine(&c.admission);let mut a=e.session().unwrap();
    for _ in 0..8 {e.prepare_prompt_workspace(usize::MAX).unwrap();a.prefill_advance(&vec![1;1024]).unwrap();}
    let kept=a.fork().unwrap();drop(a);let mut resumed=kept.fork().unwrap();
    assert_eq!(resumed.state.as_ref().unwrap().position,8192);
    e.prepare_prompt_workspace(usize::MAX).unwrap();resumed.prefill(&[1]).unwrap();
    assert_eq!(resumed.state.as_ref().unwrap().position,8193);assert_eq!(e.lease_bytes.get(),0);assert_eq!(e.split.compute.spares.borrow().capacity(),0);
}
/// GROUP LEASE: the borrow and its return read no predictor state, so a service without the predictor borrows as one
/// with it. With nothing spare it sends no command and keeps its startup group.
#[test]
fn predictor_off_borrows_when_bytes_are_spare_and_sends_nothing_when_none_are() {
    let c=checkpoint();let mut e=engine(&c.admission);e.predictor=0;let a=e.session().unwrap();
    e.prepare_prompt_workspace(0).unwrap();assert_eq!(e.current_prompt_group(),256);
    assert!(!e.split.exchange.borrow().commands.borrow().contains(&25));
    e.prepare_prompt_workspace(usize::MAX).unwrap();assert_eq!(e.current_prompt_group(),1024);
    assert!(e.split.exchange.borrow().commands.borrow().contains(&25));e.finish_prompt_workspace().unwrap();
    assert_eq!((e.lease_bytes.get(),e.lease_rows.get(),e.current_prompt_group()),(0,0,256));drop(a);
}
/// GROUP LEASE: 2,048, 3,072 and 4,096 rows are borrowed at exactly their own workspace difference, one byte less
/// funds the next smaller size, the peer's smaller offer wins, and the startup workspace never changes. (Expanded
/// latent rows: every size is charged as a whole group.)
#[test]
fn lease_only_sizes_need_their_whole_workspace_difference() {
    let c=checkpoint();let e=engine(&c.admission);let a=e.session().unwrap();e.split.exchange.borrow().peer.set(8192);
    let base=e.workspace;let [eight,four,three,two,one,half]=e.lease_extra;assert!(eight>four && four>three && three>two && two>one && one>half && half>0);
    let workspace=|rows|crate::service_memory::workspace_for_prompt_group(&c.admission,8193,base,rows).unwrap();
    assert_eq!((eight,four,three,two),(workspace(8192)-base,workspace(4096)-base,workspace(3072)-base,workspace(2048)-base));
    assert_eq!(SERVED,8192);
    for (room,rows) in [(two-1,1024),(two,2048),(three-1,2048),(three,3072),(four-1,3072),(four,4096),(eight-1,4096),(eight,8192),(usize::MAX,8192)] {
        e.prepare_prompt_workspace(room).unwrap();assert_eq!(e.current_prompt_group(),rows,"room {room}");assert_eq!(e.workspace,base);
        assert_eq!(e.lease_bytes.get(),e.prompt_lease_extra(rows).unwrap());
        e.finish_prompt_workspace().unwrap();assert_eq!((e.lease_bytes.get(),e.lease_rows.get()),(0,0));
    }
    e.split.exchange.borrow().peer.set(2048);e.prepare_prompt_workspace(usize::MAX).unwrap();assert_eq!((e.current_prompt_group(),e.lease_bytes.get()),(2048,two));
    e.finish_prompt_workspace().unwrap();
    e.split.exchange.borrow().peer.set(3072);e.prepare_prompt_workspace(usize::MAX).unwrap();assert_eq!((e.current_prompt_group(),e.lease_bytes.get()),(3072,three));
    e.finish_prompt_workspace().unwrap();
    // A second live chat: nothing is borrowed, whatever is offered.
    let b=e.session().unwrap();e.prepare_prompt_workspace(usize::MAX).unwrap();assert_eq!((e.current_prompt_group(),e.lease_bytes.get()),(256,0));drop(b);drop(a);
}
/// WIDE PHASE: a service whose base latent layers are all absorbed borrows every wide lease size (2,048..=8,192) at
/// its strictly monotone phase charge (`wide_phase_workspace`). One byte still decides the size, and the peer's smaller offer wins.
#[test]
fn an_absorbed_wide_lease_borrows_its_phase_charge_to_the_byte() {
    use crate::bounded_decoder::{group_workspace_bytes_with,LatentRows,PROMPT_GROUP,PROMPT_GROUP_MAX};
    let c=checkpoint();let mut e=engine(&c.admission);
    let rows=LatentRows::Absorbed {draft:false};e.latent_rows=rows;e.init_prompt_lease_plan().unwrap();
    let a=e.session().unwrap();e.split.exchange.borrow().peer.set(8192);
    let [eight,four,three,two,one,half]=e.lease_extra;assert!(eight>four && four>three && three>two && two>one && one>half && half>0);
    let charge=|group|group_workspace_bytes_with(&c.admission,8193,group,rows).unwrap();
    assert_eq!(four,charge(4096)+crate::service_memory::state_copy_bytes(&c.admission).unwrap()-charge(PROMPT_GROUP));
    assert_eq!(eight,charge(8192)+crate::service_memory::state_copy_bytes(&c.admission).unwrap()-charge(PROMPT_GROUP));
    assert!(charge(4096)>=charge(PROMPT_GROUP_MAX),"the phase charge holds a whole run");
    let sizes=[(512usize,half),(1024,one),(2048,two),(3072,three),(4096,four),(8192,eight)];
    let offered=|room:usize|sizes.iter().take_while(|(_,extra)|*extra<=room).last().map_or(256,|(rows,_)|*rows);
    for room in [half-1,half,one,two-1,two,three-1,three,four-1,four,eight-1,eight,usize::MAX] {
        e.prepare_prompt_workspace(room).unwrap();assert_eq!(e.current_prompt_group(),offered(room),"room {room}");
        assert_eq!(e.lease_bytes.get(),e.prompt_lease_extra(offered(room)).unwrap());assert!(e.lease_bytes.get()<=room);e.finish_prompt_workspace().unwrap();
    }
    assert_eq!((offered(eight),offered(usize::MAX)),(8192,8192));
    e.split.exchange.borrow().peer.set(2048);e.prepare_prompt_workspace(usize::MAX).unwrap();
    assert_eq!((e.current_prompt_group(),e.lease_bytes.get()),(2048,two));e.finish_prompt_workspace().unwrap();drop(a);
}
/// NOT MONOTONE: if a price list is non-monotone, a size is offered only when every smaller size is funded too.
#[test]
fn a_peers_smaller_size_is_funded_when_the_prices_are_not_monotone() {
    let c=checkpoint();let mut e=engine(&c.admission);
    let (eight,four,three,two)=(9_500_000_000usize,7_982_780_416usize,8_484_896_768usize,4_242_448_384usize);assert!(four<three && two<four);
    e.lease_extra=[eight,four,three,two,0,0];let a=e.session().unwrap();
    let band=8_100_000_000usize;assert!(four<=band && band<three);
    for (peer,room,rows,bytes) in [
        (3072,band,2048,two),
        (4096,band,2048,two),(4096,four,2048,two),(4096,three-1,2048,two),(2048,band,2048,two),
        (4096,three,4096,four),(3072,three,3072,three),(2048,three,2048,two),
        (4096,usize::MAX,4096,four),(3072,usize::MAX,3072,three),(2048,usize::MAX,2048,two),
        (4096,two-1,1024,0),(4096,two,2048,two),
    ] {
        e.split.exchange.borrow().peer.set(peer);e.prepare_prompt_workspace(room).unwrap();
        assert_eq!((e.current_prompt_group(),e.lease_bytes.get()),(rows,bytes),"peer {peer} room {room}");
        assert!(e.lease_bytes.get()<=room,"peer {peer}: {} rows borrow {} above the room {room}",e.current_prompt_group(),e.lease_bytes.get());
        e.finish_prompt_workspace().unwrap();assert_eq!((e.lease_bytes.get(),e.lease_rows.get()),(0,0));
    }
    drop(a);
}
/// GROUP LEASE: a leased call runs whole groups of 8,192, 4,096, 3,072 or 2,048 rows and a rest of at most 1,024.
#[test]
fn a_leased_call_is_cut_into_lease_sizes_and_a_small_rest() {
    use crate::bounded_decoder::lease_groups;
    for rows in [1usize,64,256,512,1024] {for tokens in [1usize,63,256,1000,1024,1025,5000] {
        assert_eq!(lease_groups(tokens,rows),(0..tokens).step_by(rows).map(|at|rows.min(tokens-at)).collect::<Vec<_>>(),"{tokens} {rows}");}}
    assert_eq!(lease_groups(8192,8192),[8192]);assert_eq!(lease_groups(8192,4096),[4096,4096]);assert_eq!(lease_groups(8192,2048),[2048;4]);
    assert_eq!(lease_groups(4096,4096),[4096]);assert_eq!(lease_groups(4095,4096),[3072,1023]);
    assert_eq!(lease_groups(8192,3072),[3072,3072,2048]);assert_eq!(lease_groups(5000,3072),[3072,1024,904]);
    assert_eq!(lease_groups(3072,3072),[3072]);assert_eq!(lease_groups(3071,3072),[2048,1023]);assert_eq!(lease_groups(4096,3072),[3072,1024]);
    assert_eq!(lease_groups(3000,4096),[2048,952]);assert_eq!(lease_groups(3000,2048),[2048,952]);
    assert_eq!(lease_groups(2047,2048),[1024,1023]);assert_eq!(lease_groups(100,4096),[100]);assert_eq!(lease_groups(0,4096),Vec::<usize>::new());
    for rows in [2048usize,3072,4096,8192] {for tokens in 1..=9000usize {let cut=lease_groups(tokens,rows);
        assert_eq!(cut.iter().sum::<usize>(),tokens);assert!(cut.iter().all(|&n|n>0 && (n<=1024 || n==2048 || n==3072 || n==4096 || n==8192) && n<=rows));}}
}
/// GROUP LEASE: the bits of a prompt do not depend on its groups. Every wide group above 1,024 drains at each mixer run
/// and each routed layer's end.
#[test]
fn borrowed_groups_of_4096_3072_and_2048_rows_give_the_bits_of_smaller_groups() {
    let c=checkpoint();let vocab=u32::try_from(c.admission.description().vocab).unwrap();assert!(vocab>=2);let tokens=(0..4096u32).map(|i|(i*7+3)%vocab.min(13)).collect::<Vec<_>>();
    let read=|peer:usize,room:usize| {
        let e=engine(&c.admission);e.split.exchange.borrow().peer.set(peer);let mut a=e.session().unwrap();
        e.prepare_prompt_workspace(room).unwrap();let rows=e.current_prompt_group();let logits=a.prefill(&tokens).unwrap();
        assert_eq!((e.lease_bytes.get(),e.lease_rows.get()),(0,0),"the call returns its borrow");
        let groups=e.split.exchange.borrow().commands.borrow().iter().filter(|&&code|code==5).count();
        let layers=c.admission.description().layers.len();
        let wide_groups=crate::bounded_decoder::lease_groups(tokens.len(),rows);
        let phases=wide_groups.iter().filter(|&&g|g>1024).map(|&g|(g/1024)*layers+crate::bounded_decoder::wide_cut_layers(&c.admission)).sum::<usize>();
        assert_eq!(e.split.compute.phases.get(),phases,"{rows} rows: every wide group above 1,024 rows drains at each mixer run and each routed layer's end");
        (rows,groups,logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),a.state.as_ref().unwrap().position)
    };
    let four=read(4096,usize::MAX);let three=read(3072,usize::MAX);let big=read(2048,usize::MAX);let mid=read(1024,usize::MAX);let small=read(4096,0);
    assert_eq!((four.0,four.1),(4096,1));assert_eq!((three.0,three.1),(3072,2));assert_eq!((big.0,big.1),(2048,2));assert_eq!((mid.0,mid.1),(1024,4));assert_eq!((small.0,small.1),(256,16));
    assert_eq!((four.3,three.3,big.3),(4096,4096,4096));assert_eq!(four.2,mid.2);assert_eq!(three.2,mid.2);assert_eq!(big.2,mid.2);assert_eq!(big.2,small.2);
}
#[test]
fn exact_remainder_threshold_preserves_the_startup_workspace_charge() {
    let c=checkpoint();let e=engine(&c.admission);let a=e.session().unwrap();
    let base=e.workspace;let half=e.prompt_lease_extra(512).unwrap();let full=e.prompt_lease_extra(1024).unwrap();assert!(full>half && half>0);
    for (room,rows) in [(half-1,256),(half,512),(full-1,512),(full,1024)] {
        e.prepare_prompt_workspace(room).unwrap();assert_eq!(e.current_prompt_group(),rows);assert_eq!(e.workspace,base);
        e.finish_prompt_workspace().unwrap();assert_eq!(e.lease_bytes.get(),0);
    }drop(a);
}

fn checkpoint()->TestCheckpoint {
    TestCheckpoint::build_with(Dtype::Bf16,true,8193,false,|d| {
        d.hidden=32;d.layers.truncate(1);
        if let engine_model::plan::description::TokenMixer::Kda(k)=&mut d.layers[0].mixer {k.heads=1;}
    })
}
#[test]
fn memo_startup_prepares_the_immutable_lease_plan() {
    let c=checkpoint();let mut e=engine(&c.admission);let expected=e.lease_extra;
    e.memo_admitted=false;e.lease_extra=[0;SIZES.len()];let e=e.with_memo_bound().unwrap();
    assert_eq!(e.lease_extra,expected);assert!(expected.iter().all(|&bytes|bytes>0));
    let a=e.session().unwrap();e.prepare_prompt_workspace(expected[3]).unwrap();assert_eq!(e.current_prompt_group(),1024);
    e.finish_prompt_workspace().unwrap();drop(a);
}
