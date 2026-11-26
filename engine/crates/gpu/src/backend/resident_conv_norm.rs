//! Join a checked short convolution and its unweighted head clip or norm.
use crate::{block_ops::device::{Op,OP_AT},resident_launch::Command};

pub(super) fn join(conv:&mut Command,norm:&Command)->bool {
    let field=|c:&Command,i:usize|u32::from_ne_bytes(c.payload[OP_AT+4*i..OP_AT+4*i+4].try_into().unwrap());
    let addr=|c:&Command,i:usize|u64::from_ne_bytes(c.payload[8*i..8*i+8].try_into().unwrap());
    let op=field(norm,0);
    if field(conv,0)!=Op::ShortConv as u32 || conv.reserved[0]!=0 ||
        !(op==Op::RoundedNorm as u32 || op==Op::HeadClip as u32) ||
        field(norm,3)!=0 || field(norm,5)!=0 || field(conv,5)>1 ||
        field(conv,2)!=crate::block_ops::SUM_THREADS as u32 || field(norm,2)!=field(conv,2) ||
        field(norm,1)!=field(conv,1) || addr(conv,5)!=addr(norm,0) {return false;}
    let heads=u64::from(field(conv,1));
    let width=heads*u64::from(field(conv,2));
    let taps=u64::from(field(conv,3));
    let dst=addr(norm,5);
    let Some(end)=dst.checked_add(4*width) else {return false};
    // Both original calls passed block_check. The extra condition for simultaneous
    // execution is that the later output cannot overwrite any input of the producer.
    for (i,words) in [width*taps,heads*taps*u64::from(field(conv,5)),0,1,width*taps,width].into_iter().enumerate() {
        let base=addr(conv,i);
        if words!=0 && base<end && base.checked_add(4*words).is_none_or(|limit|dst<limit) {return false;}
    }
    conv.reserved[0]=op;
    conv.reserved[1]=if op==Op::RoundedNorm as u32 {field(norm,6)} else {field(norm,7)};
    conv.word(16,dst);
    true
}

/// Join two finish commands (`conv_finish_into`, the key and then the value of one token) of the same shape and the
/// same step word into one launch of twice the grid (`inkling_launch::CONV_FINISH_PAIR`). Each passed its own checks;
/// the extra condition for running them at once is that the words one writes (its ring, output and plane row) meet
/// no word the other reads or writes.
pub(super) fn pair(first:&mut Command,second:&Command)->bool {
    use crate::{block_stream::CONV_TOKEN_TAPS,inkling_launch::{CONV_FINISH,CONV_FINISH_PAIR,FINISH_PLANE_WORD,FINISH_PRODUCT_WORD,PAIR_AT}};
    let field=|c:&Command,i:usize|u32::from_ne_bytes(c.payload[OP_AT+4*i..OP_AT+4*i+4].try_into().unwrap());
    let addr=|c:&Command,i:usize|u64::from_ne_bytes(c.payload[8*i..8*i+8].try_into().unwrap());
    if first.reserved[2]!=CONV_FINISH || second.reserved[2]!=CONV_FINISH || field(first,0)!=Op::ShortConv as u32
        || first.payload[OP_AT..OP_AT+32]!=second.payload[OP_AT..OP_AT+32] || addr(first,3)!=addr(second,3) {return false;}
    let (heads,dim,taps)=(u64::from(field(first,1)),u64::from(field(first,2)),u64::from(field(first,3)));
    let width=heads*dim;
    let made=if field(first,5)&CONV_TOKEN_TAPS!=0 {heads*taps} else {0};
    // (start, words) of the ranges each command reads and writes.
    let ranges=|c:&Command|->([(u64,u64);5],[(u64,u64);3]) {(
        [(addr(c,0),width*taps),(addr(c,1),made),(addr(c,3),1),(addr(c,4),width*taps),(addr(c,FINISH_PRODUCT_WORD),width)],
        [(addr(c,0),width*taps),(addr(c,5),width),(addr(c,FINISH_PLANE_WORD),width/2)])};
    let meet=|a:(u64,u64),b:(u64,u64)|a.1!=0 && b.1!=0 && (a.0 as u128)<b.0 as u128+4*b.1 as u128 && (b.0 as u128)<a.0 as u128+4*a.1 as u128;
    let ((read1,write1),(read2,write2))=(ranges(first),ranges(second));
    for w in write1 {if read2.iter().chain(&write2).any(|&r|meet(w,r)) {return false;}}
    for w in write2 {if read1.iter().any(|&r|meet(w,r)) {return false;}}
    first.reserved[2]=CONV_FINISH_PAIR;
    for (i,a) in [addr(second,0),addr(second,1),addr(second,4),addr(second,5),addr(second,FINISH_PRODUCT_WORD),addr(second,FINISH_PLANE_WORD)].into_iter().enumerate() {
        first.word(PAIR_AT+i,a);
    }
    first.word(PAIR_AT+6,u64::from(second.reserved[0])|(u64::from(second.reserved[1])<<32));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{resident_launch::Kind,causal_ops::{Arguments,Shape}};
    fn pair(op:crate::causal_ops::Op)->(Command,Command) {
        let shape=Shape {op:crate::causal_ops::Op::ShortConv,heads:2,dim:128,taps:4,rows:0,cols:1,epsilon:0.0,bound:0.5};
        let mut conv=crate::resident_launch_encode::causal(Arguments {input:[0x1000,0x3000,4,0x4000],weight:0x5000,output:0x8000,shape});
        let mut norm=crate::resident_launch_encode::causal(Arguments {input:[0x8000,4,4,4],weight:4,output:0xa000,
            shape:Shape {op,taps:0,cols:0,epsilon:0.01,bound:0.75,..shape}});
        conv.kind=Kind::BlockArms as u32;norm.kind=conv.kind;
        (conv,norm)
    }
    #[test]
    fn producer_inputs_stay_live_through_the_norm_store() {
        for op in [crate::causal_ops::Op::HeadClip,crate::causal_ops::Op::RoundedNorm] {
            let (conv,norm)=pair(op);
            for addr in [0x0ffc,0x1ffc,0x3004,0x4000,0x5ffc,0x8004] {
                let (mut prev,mut next)=(conv,norm);next.word(5,addr);
                assert!(!join(&mut prev,&next));
            }
            let mut prev=conv;assert!(join(&mut prev,&norm));assert!(!join(&mut prev,&norm));
            assert_eq!(prev.reserved[0],op as u32);
        }
    }
    /// A finish command at `base`: ring at +0 (4 rows of 256 words), fixed taps at +0x2000, output at +0x4000, product at
    /// +0x5000, plane row at +0x6000; the step word at 0x100.
    fn finish(base:u64,op:Op)->Command {
        use crate::inkling_launch::{CONV_FINISH,FINISH_PLANE_WORD,FINISH_PRODUCT_WORD};
        let shape=Shape {op:crate::causal_ops::Op::ShortConv,heads:2,dim:128,taps:4,rows:0,cols:0,epsilon:0.0,bound:0.2};
        let mut c=crate::resident_launch_encode::causal(Arguments {input:[base,4,4,0x100],weight:base+0x2000,output:base+0x4000,shape});
        c.kind=Kind::BlockArms as u32;c.reserved=[op as u32,0x3f00_0000,CONV_FINISH];
        c.word(FINISH_PRODUCT_WORD,base+0x5000);c.word(FINISH_PLANE_WORD,base+0x6000);c
    }
    #[test]
    fn a_key_and_value_finish_pair_only_when_no_write_meets_the_other() {
        use crate::inkling_launch::{CONV_FINISH_PAIR,FINISH_PRODUCT_WORD,PAIR_AT};
        let (key,value)=(finish(0x10000,Op::HeadClip),finish(0x20000,Op::RoundedNorm));
        let joins=|mut a:Command,b:&Command|super::pair(&mut a,b);
        let mut first=key;assert!(super::pair(&mut first,&value));
        assert_eq!(first.reserved[2],CONV_FINISH_PAIR);
        let word=|c:&Command,i:usize|u64::from_ne_bytes(c.payload[8*i..8*i+8].try_into().unwrap());
        assert_eq!((0..7).map(|i|word(&first,PAIR_AT+i)).collect::<Vec<_>>(),
            [0x20000,4,0x22000,0x24000,0x25000,0x26000,u64::from(Op::RoundedNorm as u32)|(0x3f00_0000u64<<32)]);
        assert!(!joins(first,&value),"a pair takes no third command");
        // The other step word, another shape, a value read inside a key write, a key read inside a value write.
        let mut other=value;other.word(3,0x104);assert!(!joins(key,&other));
        let mut other=value;other.payload[OP_AT+12]=3;assert!(!joins(key,&other));
        for (at,address) in [(FINISH_PRODUCT_WORD,0x14000),(0,0x16000),(4,0x10ffc)] {
            let mut other=value;other.word(at,address);assert!(!joins(key,&other));
        }
        let mut early=key;early.word(FINISH_PRODUCT_WORD,0x24000);assert!(!joins(early,&value));
    }
}
