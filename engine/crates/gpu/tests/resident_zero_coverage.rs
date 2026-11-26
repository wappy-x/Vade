//! Host model of backend/resident_launch.rs::fuse_fresh_fills. No CUDA calls.
use spark_gpu_probe::resident_launch::stage;

type CopyRow=(u64,u64,u32);
const ZERO_WORDS:usize=1<<15;
const ZERO:u64=1<<48;
fn span(target:u64,words:u32)->(u64,u64) {(target,target+words as u64*4)}

// Keep the containment checks, sorted complement and chunk arithmetic equal to
// fuse_fresh_fills. None means settle_fresh_fills keeps the full head fill.
fn fused_plan(waiting:&[(u64,u32)],rows:&[CopyRow])->Option<Vec<CopyRow>> {
    if rows.is_empty() {return None;}
    let mut fused=vec![false;waiting.len()];
    for &(source,target,words) in rows {
        let ((a,b),(sa,sb))=(span(target,words),span(source,words));let mut inside=None;
        for (i,&(t,w)) in waiting.iter().enumerate() {let (s,e)=span(t,w);
            if sa<e && s<sb {return None;}
            if a>=s && b<=e {inside=Some(i);} else if a<e && s<b {return None;}
        }
        if let Some(i)=inside {fused[i]=true;}
    }
    if !fused.contains(&true) {return None;}
    let chunk=rows.iter().map(|r|r.2).max().unwrap_or(0).clamp(128,ZERO_WORDS as u32) as u64;
    let mut table=rows.to_vec();
    for (i,&(t,w)) in waiting.iter().enumerate() {
        if !fused[i] {continue;}
        let (s,e)=span(t,w);
        let mut covered=rows.iter().map(|&(_,t,w)|span(t,w)).filter(|&(a,b)|a>=s && b<=e).collect::<Vec<_>>();covered.sort_unstable();
        let mut cursor=s;let mut gaps=Vec::new();
        for &(a,b) in &covered {if a>cursor {gaps.push((cursor,a));}cursor=cursor.max(b);}
        if cursor<e {gaps.push((cursor,e));}
        for (mut a,b) in gaps {while a<b {let n=((b-a)/4).min(chunk);table.push((ZERO,a,n as u32));a+=n*4;}}
    }
    if table.len()>u16::MAX as usize || table.len()*24>stage::BYTES/4 {return None;}
    // Unmatched waiting outputs retain their full head fill.
    for (&(t,w),fused) in waiting.iter().zip(fused) {if !fused {table.push((ZERO,t,w));}}
    Some(table)
}

#[derive(Clone,Copy,Debug)]
enum Path {Normal,PartialTake,FailedTake,EarlyExit}
fn check_output(row_count:usize,width:usize,local:&[std::ops::Range<usize>],path:Path) {
    let words=row_count*width;let base=0x1000;let other=base+(words as u64+16)*4;
    // The second output has no copy, so it must keep its full head fill.
    let waiting=[(base,words as u32),(other,3)];
    let mut rows=local.iter().filter(|r|!r.is_empty()).map(|r|
        (ZERO+4096+r.start as u64*4,base+r.start as u64*4,r.len() as u32)).collect::<Vec<_>>();
    rows.reverse(); // The complement must sort copy targets, not trust table order.
    let plan=fused_plan(&waiting,&rows).unwrap_or_else(||waiting.iter().map(|&(t,w)|(ZERO,t,w)).collect());
    let mut copied=vec![false;words];
    for r in local {for word in r.clone() {assert!(!copied[word],"overlapping input copy rows");copied[word]=true;}}
    for &(target,len) in &waiting {
        let mut coverage=vec![0u8;len as usize];let mut values=vec![u32::MAX;len as usize];
        for &(source,t,w) in &plan {
            let (a,b)=span(t,w);let (s,e)=span(target,len);
            if a>=e || b<=s {continue;}
            assert!(a>=s && b<=e,"plan row crosses an output edge");
            assert!(source!=ZERO || w as usize<=ZERO_WORDS || rows.is_empty() || target==other);
            for word in ((a-s)/4) as usize..((b-s)/4) as usize {
                coverage[word]+=1;values[word]=if source==ZERO {0} else {11};
            }
        }
        // Every word in each logical row has exactly one producing copy or zero.
        for (word,&count) in coverage.iter().enumerate() {
            assert_eq!(count,1,"rows={row_count} width={width} path={path:?} output={target:#x} word={word}");
        }
        for (word,value) in values.iter_mut().enumerate() {
            let local=target==base && copied[word];
            let takes_peer=target==base && !local && match path {
                Path::Normal=>true,Path::PartialTake=>word%3==0,Path::FailedTake|Path::EarlyExit=>false,
            };
            // Take runs after the producing launch. It can overwrite zero words.
            if takes_peer {*value=23;}
            let expected=if local {11} else if takes_peer {23} else {0};
            assert_eq!(*value,expected,"rows={row_count} width={width} path={path:?} word={word}");
        }
    }
}

#[test]
#[allow(clippy::single_range_in_vec_init)] // a one-range Vec is intended.
fn zero_inside_copy_covers_exchange_outputs_on_every_path() {
    let mut shapes=Vec::new();
    for width in [1,3,127,128,129] {for rows in [0,1,2,3,5,7,8,9,11,16,99] {shapes.push((rows,width));}}
    // Exchange-output word counts: DEAD-ZERO census/zc3-c15-rank0.txt,
    // fresh=1, resident_weights.rs:99 and :318. Include zero-source boundaries.
    for words in [2336,4096,4384,16384,20480,24576,24896,32768,32769] {shapes.push((1,words));}
    shapes.push((3,ZERO_WORDS));
    for (rows,width) in shapes {
        let words=rows*width;
        let layouts=[vec![0..words/2],vec![words/2..words],
            (0..rows).filter(|r|r%2==0).map(|r|r*width..(r+1)*width).collect(),
            vec![words/3..words*2/3],Vec::new()];
        for local in &layouts {for path in [Path::Normal,Path::PartialTake,Path::FailedTake,Path::EarlyExit] {
            check_output(rows,width,local,path);
        }}
    }
}

#[test]
fn zero_inside_copy_keeps_head_fill_for_unsafe_spans() {
    let waiting=[(0x1000,8),(0x2000,8)];
    for rows in [vec![],vec![(ZERO+4096,0x0ffc,2)],vec![(ZERO+4096,0x101c,2)],
        vec![(0x101c,0x2000,2)],vec![(0x0ffc,0x2000,2)],vec![(0x1000,0x2000,1)],
        vec![(ZERO+4096,0x3000,4)]] {
        assert!(fused_plan(&waiting,&rows).is_none(),"must keep the full head fill: {rows:?}");
    }
}
