//! A one-row stream convolution of the Inkling block arms (arm `ShortConv` over the whole width, one head: the
//! convolution after attention and the one after the experts) takes the small calls right before it in a token step,
//! so they run inside its launch and no launch of their own runs:
//! - `ring`: the copy of the new input into one row of the ring it reads (`state::write_ring`). The thread of
//!   channel `i` writes word `i` of that row before it reads its channel of the ring.
//! - `fold`: before that copy, the one-row route fold (`fast_ops::STREAM_OP_ROUTE_FOLD`) whose output the copy reads
//!   (the expert sum of `experts::close`). The thread of channel `i` makes word `i` of the fold by the fold's own
//!   chain (`rn` of each expert word times its weight, added in pick order from 0.0, then `rn`) and writes it to the
//!   fold's output and to the ring row.
//!
//! Why the words are the same: each join takes only the call right before the launch (no other call between them
//! reads or writes these words); the moved writes are the words those calls wrote, made by the same arithmetic; and
//! every read operand of a moved call is checked here to meet no word the launch writes, while each word the launch
//! reads from the ring row is written first by the thread that reads it.
use crate::{block_ops::device::{Op,OP_AT},resident_launch::{Command,Kind}};

/// Command words (8-byte) of the joined calls: the ring input and the ring row; the fold's weights, expert rows and
/// pick count.
pub(crate) const RING_SOURCE_AT:usize=19;
pub(crate) const RING_ROW_AT:usize=20;
pub(crate) const FOLD_WEIGHTS_AT:usize=21;
pub(crate) const FOLD_ROWS_AT:usize=22;
pub(crate) const FOLD_PICKS_AT:usize=23;
/// Bits of `reserved[2]` of the launch: the ring row is written from `RING_SOURCE_AT`; that source is the output of a
/// route fold the launch makes itself.
pub(crate) const RING:u32=4;
pub(crate) const FOLD:u32=8;

fn field(c:&Command,i:usize)->u32 {u32::from_ne_bytes(c.payload[OP_AT+4*i..OP_AT+4*i+4].try_into().unwrap())}
fn addr(c:&Command,i:usize)->u64 {u64::from_ne_bytes(c.payload[8*i..8*i+8].try_into().unwrap())}
fn meets(a:u64,an:u64,b:u64,bn:u64)->bool {an!=0 && bn!=0 && a!=0 && b!=0 && a<b.saturating_add(4*bn) && b<a.saturating_add(4*an)}
/// The width of a one-head ShortConv command that nothing has joined yet. None for any other command.
fn stream_width(conv:&Command)->Option<u64> {
    (field(conv,0)==Op::ShortConv as u32 && field(conv,1)==1 && conv.reserved==[0;3]).then(||u64::from(field(conv,2)))
}
/// The copy of the new input into one row of the ring, right before the one-row convolution `conv` (not recorded yet).
/// The row must be a whole row of the ring the convolution reads; the source must meet nothing the launch writes.
pub(super) fn ring(conv:&mut Command,copy_kind:Kind,copy:&Command)->bool {
    let Some(width)=stream_width(conv) else {return false};
    let (ring,taps)=(addr(conv,0),u64::from(field(conv,3)));
    if copy_kind!=Kind::CopyWords || width==0 || taps==0 || addr(copy,2)!=width {return false;}
    let (source,target)=(addr(copy,0),addr(copy,1));
    let Some(offset)=target.checked_sub(ring) else {return false};
    if !offset.is_multiple_of(4*width) || offset/(4*width)>=taps || source==0 || !source.is_multiple_of(4) {return false;}
    // The launch writes its output and the ring row; the source is read. It must not be the ring or the output.
    if meets(source,width,ring,width*taps) || meets(source,width,addr(conv,5),width) {return false;}
    conv.word(RING_SOURCE_AT,source);conv.word(RING_ROW_AT,target);conv.reserved[2]=RING;
    true
}
/// The one-row route fold right before the joined ring copy of `conv`, whose output is that copy's source. Its weights
/// and expert rows must meet nothing the launch writes (the ring, the output, the fold's output).
pub(super) fn fold(conv:&mut Command,kind:Kind,fold:&Command)->bool {
    if kind!=Kind::FastStreamOps || field(conv,0)!=Op::ShortConv as u32 || field(conv,1)!=1 || conv.reserved!=[0,0,RING] {return false;}
    let width=u64::from(field(conv,2));
    // Words of the stream op: op, weights, expert rows, -, -, output, rows, dim, picks.
    let (weights,downs,picks)=(addr(fold,1),addr(fold,2),addr(fold,8));
    if addr(fold,0)!=crate::fast_ops::STREAM_OP_ROUTE_FOLD || addr(fold,6)!=1 || addr(fold,7)!=width || addr(fold,5)!=addr(conv,RING_SOURCE_AT)
        || picks==0 || picks>crate::fast_ops::ROUTER_PICKS_MAX as u64 || weights==0 || downs==0 {return false;}
    let taps=u64::from(field(conv,3));
    let written=[(addr(conv,0),width*taps),(addr(conv,5),width),(addr(conv,RING_SOURCE_AT),width)];
    if [(weights,picks),(downs,picks*width)].iter().any(|&(base,n)|written.iter().any(|&(w,m)|meets(base,n,w,m))) {return false;}
    conv.word(FOLD_WEIGHTS_AT,weights);conv.word(FOLD_ROWS_AT,downs);conv.word(FOLD_PICKS_AT,picks);conv.reserved[2]|=FOLD;
    true
}
/// Rewrite a fused 1-head `BlockArms` `ShortConv` (`RING` or `RING | FOLD`, no token taps, `2..=4` taps)
/// into a zero-stack full-SM (`72 x 32`) `Kind::FastStreamOps` `STREAM_OP_RING_CONV_ONE` command.
pub(super) fn to_fast_stream(conv:&Command)->Option<(Command,[u32;5])> {
    if field(conv,0)!=Op::ShortConv as u32 || field(conv,1)!=1 || conv.reserved[0]!=0 || conv.reserved[2]&RING==0 {return None;}
    let (dim,taps,cols,bound_bits)=(field(conv,2) as usize,field(conv,3) as usize,field(conv,5),field(conv,7));
    if cols&crate::block_stream::CONV_TOKEN_TAPS!=0
        || !crate::fast_ops::stream_ops_admitted(crate::fast_ops::STREAM_OP_RING_CONV_ONE,1,dim,taps,crate::fast_ops::ROUTE_FOLD_CONV_THREADS)
    {
        return None;
    }
    let has_fold=conv.reserved[2]&FOLD!=0;
    let has_add=cols&crate::block_stream::CONV_ADDEND!=0;
    let mut c=Command::new(Kind::FastStreamOps);
    let words=[
        crate::fast_ops::STREAM_OP_RING_CONV_ONE,
        addr(conv,0),
        if has_add {addr(conv,2)} else {0},
        addr(conv,3),
        addr(conv,4),
        addr(conv,5),
        1,
        dim as u64,
        taps as u64,
        u64::from(cols),
        if has_fold {addr(conv,FOLD_PICKS_AT)} else {0},
        u64::from(bound_bits),
        addr(conv,RING_SOURCE_AT),
        addr(conv,RING_ROW_AT),
        if has_fold {addr(conv,FOLD_WEIGHTS_AT)} else {0},
        if has_fold {addr(conv,FOLD_ROWS_AT)} else {0},
    ];
    for (i,w) in words.into_iter().enumerate() {c.word(i,w);}
    let threads=crate::fast_ops::ROUTE_FOLD_CONV_THREADS as u32;
    let blocks=u32::try_from(dim.div_ceil(threads as usize)).ok()?;
    Some((c,[blocks,1,1,threads,0]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_ops::{Arguments,Shape};
    /// A convolution after attention or after the experts: one head of 256 values, 4 taps, ring at 0x1000 (4 rows of
    /// 1 KiB), addend at 0x6000, output at 0x8000.
    fn conv()->Command {
        let shape=Shape {op:crate::causal_ops::Op::ShortConv,heads:1,dim:256,taps:4,rows:0,cols:2,epsilon:0.0,bound:0.5};
        let mut conv=crate::resident_launch_encode::causal(Arguments {input:[0x1000,4,0x6000,0x4000],weight:0x5000,output:0x8000,shape});
        conv.kind=Kind::BlockArms as u32;conv
    }
    fn copy(source:u64,target:u64,words:u64)->Command {let mut c=Command::new(Kind::CopyWords);c.word(0,source);c.word(1,target);c.word(2,words);c}
    fn folding(weights:u64,downs:u64,output:u64,rows:u64,dim:u64,picks:u64)->Command {
        let mut c=Command::new(Kind::FastStreamOps);for (i,w) in [crate::fast_ops::STREAM_OP_ROUTE_FOLD,weights,downs,0,0,output,rows,dim,picks,0,0,0].into_iter().enumerate() {c.word(i,w);}c
    }
    #[test]
    fn the_ring_row_is_a_whole_row_of_the_ring_and_its_source_is_only_read() {
        for target in [0x1000u64,0x1400,0x1800,0x1c00] {let mut c=conv();assert!(ring(&mut c,Kind::CopyWords,&copy(0xe000,target,256)),"{target:#x}");
            assert_eq!((addr(&c,RING_SOURCE_AT),addr(&c,RING_ROW_AT),c.reserved),(0xe000,target,[0,0,RING]));}
        for (source,target,words) in [(0xe000,0x2000,256),(0xe000,0x1004,256),(0xe000,0x0c00,256),(0xe000,0x1000,128),(0x1400,0x1000,256),(0x8100,0x1000,256),(0,0x1000,256),(0xe002,0x1000,256)] {
            let mut c=conv();assert!(!ring(&mut c,Kind::CopyWords,&copy(source,target,words)),"{source:#x} {target:#x} {words}");assert_eq!(c.reserved,[0;3]);
        }
        let mut c=conv();assert!(!ring(&mut c,Kind::Copy,&copy(0xe000,0x1000,256)));
        // A key or value convolution (8 heads of 128) or a joined one takes none.
        let mut kv=conv();kv.put(OP_AT+4,&8u32.to_ne_bytes());kv.put(OP_AT+8,&32u32.to_ne_bytes());assert!(!ring(&mut kv,Kind::CopyWords,&copy(0xe000,0x1000,256)));
        let mut c=conv();c.reserved[0]=1;assert!(!ring(&mut c,Kind::CopyWords,&copy(0xe000,0x1000,256)));
        let mut c=conv();assert!(ring(&mut c,Kind::CopyWords,&copy(0xe000,0x1000,256)));assert!(!ring(&mut c,Kind::CopyWords,&copy(0xe800,0x1400,256)),"one row a launch");
    }
    #[test]
    fn the_fold_joins_only_before_its_own_ring_row() {
        let ringed=||{let mut c=conv();assert!(ring(&mut c,Kind::CopyWords,&copy(0xe000,0x1400,256)));c};
        let mut c=conv();assert!(!fold(&mut c,Kind::FastStreamOps,&folding(0x9000,0x20000,0xe000,1,256,4)),"a ring row first");
        for (i,f) in [folding(0x9000,0x20000,0xe400,1,256,4),folding(0x9000,0x20000,0xe000,2,256,4),folding(0x9000,0x20000,0xe000,1,128,4),
            folding(0x9000,0x20000,0xe000,1,256,0),folding(0x9000,0x20000,0xe000,1,256,33),folding(0x1000,0x20000,0xe000,1,256,4),
            folding(0x9000,0x7c00,0xe000,1,256,4),folding(0x9000,0xdc00,0xe000,1,256,4)].iter().enumerate() {
            let mut c=ringed();assert!(!fold(&mut c,Kind::FastStreamOps,f),"{i}");assert_eq!(c.reserved,[0,0,RING]);
        }
        let mut c=ringed();assert!(!fold(&mut c,Kind::FastNorm,&folding(0x9000,0x20000,0xe000,1,256,4)));
        let mut c=ringed();assert!(fold(&mut c,Kind::FastStreamOps,&folding(0x9000,0x20000,0xe000,1,256,4)));
        assert_eq!((c.reserved,addr(&c,FOLD_WEIGHTS_AT),addr(&c,FOLD_ROWS_AT),addr(&c,FOLD_PICKS_AT)),([0,0,RING|FOLD],0x9000,0x20000,4));
        assert!(!fold(&mut c,Kind::FastStreamOps,&folding(0x9000,0x20000,0xe000,1,256,4)),"one fold a launch");
    }
    #[test]
    fn to_fast_stream_rewrites_ring_and_fold_one_head_convs() {
        let mut c=conv();
        assert!(to_fast_stream(&c).is_none(),"requires RING");
        assert!(ring(&mut c,Kind::CopyWords,&copy(0xe000,0x1400,256)));
        let (f,g)=to_fast_stream(&c).expect("RING rewrites");
        assert_eq!((f.kind,g),(Kind::FastStreamOps as u32,[8,1,1,32,0]));
        assert_eq!((addr(&f,0),addr(&f,1),addr(&f,2),addr(&f,3),addr(&f,4),addr(&f,5),addr(&f,10),addr(&f,12),addr(&f,13)),
            (crate::fast_ops::STREAM_OP_RING_CONV_ONE,0x1000,0x6000,0x4000,0x5000,0x8000,0,0xe000,0x1400));
        assert!(fold(&mut c,Kind::FastStreamOps,&folding(0x9000,0x20000,0xe000,1,256,8)));
        let (f,g)=to_fast_stream(&c).expect("RING|FOLD rewrites");
        assert_eq!((f.kind,g),(Kind::FastStreamOps as u32,[8,1,1,32,0]));
        assert_eq!((addr(&f,10),addr(&f,14),addr(&f,15)),(8,0x9000,0x20000));
        c.put(OP_AT+20,&(crate::block_stream::CONV_TOKEN_TAPS|2).to_ne_bytes());
        assert!(to_fast_stream(&c).is_none(),"token taps stay on BlockArms");
    }
}
