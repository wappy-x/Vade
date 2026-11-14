#[path="../src/hybrid/memory.rs"] mod memory;
use std::{alloc::{GlobalAlloc, Layout, System}, cell::Cell};
struct Tracked;
thread_local! { static BYTES: Cell<Option<(usize,usize)>> = const { Cell::new(None) }; }
// Track only this test's thread; allocations include bucket controls and queue
// storage. The default realloc also counts simultaneous old/new queue buffers.
unsafe impl GlobalAlloc for Tracked {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr=unsafe { System.alloc(layout) };
        if !ptr.is_null() { let _=BYTES.try_with(|b| { if let Some((live,peak))=b.get() {
            let live=live+layout.size(); b.set(Some((live,peak.max(live))));
        } }); }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _=BYTES.try_with(|b| { if let Some((live,peak))=b.get() { b.set(Some((live-layout.size(),peak))); } });
        unsafe { System.dealloc(ptr,layout) };
    }
}
#[global_allocator] static ALLOCATOR: Tracked = Tracked;
#[test] fn copy_index_peak_funds_buckets_queue_and_replacement() {
    use spark_engine::hybrid_draft::{CopyIndex, Options};
    struct Tracking;
    impl Drop for Tracking { fn drop(&mut self) { BYTES.with(|b| b.set(None)); } }
    let on=Options { copy_first:true, gated_depth:false };
    let history: Vec<u32>=(0..262144u32).map(|n| n.wrapping_mul(0x9e3779b9).rotate_left(13)).collect();
    let mut worst=0;
    for _ in 0..8 {
        let _tracking=Tracking;
        let mut copy=CopyIndex::default();
        BYTES.with(|b| b.set(Some((0,0))));
        for end in 0..=history.len() {
            assert!(copy.propose(&history[..end],0,0).is_empty());
            let peak=BYTES.with(|b| b.get().unwrap().1);
            assert!(peak<=CopyIndex::reservation_bytes(end,on).unwrap(), "end={end} peak={peak}");
            worst=worst.max(peak);
        }
        drop(copy);
        assert_eq!(BYTES.with(|b| b.replace(None).unwrap().0),0);
    }
    assert!(worst>1051648, "must exercise the formerly underfunded peak");
    println!("copy_index worst_peak_bytes={worst} reservation_bytes={}",CopyIndex::reservation_bytes(history.len(),on).unwrap());
}
#[test] fn configured_context_and_pooled_owners() {
    let c=engine_model::qwen3_5::Config::parse(include_bytes!("../../model/tests/fixtures/qwen/Qwen3.6-35B-A3B-FP8.config.json")).unwrap();
    let (scratch,prompt,decode,pool)=(4525376,9027723264,588208128,17761639143usize);
    assert_eq!((memory::private_rows(false),memory::private_rows(true)),(4096,16)); assert_eq!(memory::workspace(&c,256,262144,prompt),Some(prompt)); for (fixed,draft,active,pooled) in [(true,false,Some(1),true),(true,false,Some(8),true),(true,true,Some(8),true),(false,false,None,false)] { assert_eq!(memory::pool_prompt(fixed,draft,active),pooled); }
    for context in [8192,32768,131072] { let p=memory::workspace(&c,256,context,prompt).unwrap(); let d=memory::workspace(&c,16,context,decode).unwrap(); assert_eq!(p,prompt-256*16*(262144-context)*8);
        let before=memory::request(&c,context,context,256,scratch,prompt,false,0).unwrap(); assert_eq!(before,9757490496+20512*context+memory::PREFIX_SNAPSHOTS*c.state_bytes() as usize);
        for chats in [1,2,8,16] { let pooled=chats>1; let rows=memory::private_rows(pooled); let after=memory::request(&c,context,context,rows,scratch,if pooled {d} else {p},false,0).unwrap(); let available=pool+decode-d-memory::host_logits(&c,16).unwrap()-if pooled {p+memory::host_logits(&c,256).unwrap()} else {0}; println!("chats={chats} context={context} before={before} after={after} total_before={} total_after={} available_after={available}",chats*before,chats*after); if chats==8 && context==32768 { assert!(chats*before>pool && chats*after<=available); } }
    }
    let context=16384; let p=memory::workspace(&c,256,context,prompt).unwrap(); let d=memory::workspace(&c,16,context,decode).unwrap(); assert!(16*memory::request(&c,context,context,16,scratch,d,false,0).unwrap()<=pool+decode-d-memory::host_logits(&c,16).unwrap()-p-memory::host_logits(&c,256).unwrap());
    for (tokens,ceiling) in [(0,32768),(32769,32768),(1,262145),(usize::MAX,usize::MAX)] { assert!(memory::request(&c,tokens,ceiling,16,scratch,decode,false,0).is_none()); } assert!(memory::workspace(&c,16,0,decode).is_none()); assert!(memory::workspace(&c,16,262145,decode).is_none()); assert!(memory::workspace(&c,256,1,1).is_none()); assert!(memory::host_logits(&c,usize::MAX-1).is_none());
    // The literals above exercise host arithmetic; Linux must fund the live executors.
    #[cfg(target_os = "linux")]
    let (prompt,decode) = {
        use spark_engine::hybrid::Model;
        (Model::fixed_decode_bytes(&c,memory::private_rows(false)).unwrap(),Model::fixed_decode_bytes(&c,16).unwrap())
    };
    for context in [8192,32768] {
        let p=memory::workspace(&c,memory::private_rows(false),context,prompt).unwrap();
        let d=memory::workspace(&c,16,context,decode).unwrap();
        let prompt=p+memory::host_logits(&c,256).unwrap();
        let pool=54127801959usize-35777954688-prompt-d-memory::host_logits(&c,16).unwrap();
        let bytes=|retain,draft,rows| memory::request_with_retention(&c,context,context,rows,scratch,if rows==16 {d} else {p},draft,0,retain).unwrap();
        #[cfg(target_os = "linux")] {
            use spark_engine::hybrid::Model;
            assert_eq!(prompt,Model::fixed_prompt_bytes(&c,context).unwrap());
            assert_eq!(p,Model::fixed_decode_bytes_at(&c,memory::private_rows(false),context).unwrap());
            assert_eq!(d,Model::fixed_decode_bytes_at(&c,16,context).unwrap());
        }
        for draft in [false,true] { for rows in [16,memory::private_rows(false)] { assert_eq!(bytes(true,draft,rows)-bytes(false,draft,rows),memory::PREFIX_SNAPSHOTS*c.state_bytes() as usize); } }
        for retain in [false,true] { let request=bytes(retain,false,16); let slots=(pool/request).min(32); assert!(slots>0 && slots*request<=pool); if slots<32 { assert!((slots+1)*request>pool); } }
    }
}
