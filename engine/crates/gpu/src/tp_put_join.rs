//! The joined put: the shape host and kernel share. One launch sends one exchange message and
//! writes every word of the gathered output that this rank knows: its own rows at their final
//! places and zero in every other word (the words a later take fills). The message is the parts
//! in order; a part may ask for the BF16 round of its words first. Integer layout only.
/// A row moves four-word units when the host allows it, the count divides and both ends are
/// 16-byte aligned (a zero row has no source). Host and kernel use this one rule.
pub const fn quad(source:u64,destination:u64,words:usize,wide:bool)->bool {wide && words.is_multiple_of(4) && (source|destination).is_multiple_of(16)}
/// A BF16-packed wire slot row moves four FP32 source words into two packed BF16 u32 words (8 bytes)
/// when the count divides 4, the source is 16-byte aligned, and the packed slot destination is 8-byte aligned.
pub const fn quad_bf16_slot(source:u64,destination:u64,words:usize,wide:bool)->bool {wide && words.is_multiple_of(4) && source.is_multiple_of(16) && destination.is_multiple_of(8)}
/// Bit 31 of a table row's word count: round each word to BF16 first.
pub const ROUND:u32=1<<31;
#[cfg(not(target_arch="nvptx64"))]
pub use host::*;
#[cfg(not(target_arch="nvptx64"))]
mod host {
    use super::*;
    /// The most table rows of one message. A row can be cut once at each part edge.
    pub const MOST_ROWS:usize=2048;
    /// The shapes the joined put takes: a decode step's messages and multi-row decode/verify steps.
    pub const fn admitted(parts:usize,rows:usize,message:usize,output:usize)->bool {
        parts>=1 && parts<=MOST_PARTS && rows<=MOST_JOBS && message>=1 && message<=MOST_MESSAGE && output>=1 && output<=MOST_OUTPUT
    }
    pub const MOST_PARTS:usize=32;
    pub const MOST_JOBS:usize=512;
    pub const MOST_MESSAGE:usize=1<<20;
    /// Output words covered by OUTPUT_BLOCKS x THREADS threads (up to 4 four-word units/thread at 1M words).
    pub const MOST_OUTPUT:usize=1<<20;
    #[derive(Clone,Copy,Debug,PartialEq,Eq)]
    pub enum Target {Slot(usize),Output(usize)}
    /// `words` words from word `source.1` of part `source.0` (None: zero) go to `target`.
    #[derive(Clone,Copy,Debug,PartialEq,Eq)]
    pub struct Row {pub source:Option<(usize,usize)>,pub target:Target,pub words:usize,pub round:bool}
    /// Slot rows (each part at its message offset), then the own placement (`rows` are [message
    /// offset, output offset, words]; a row is cut where it crosses a part), then the zero rows
    /// that cover the rest of the output. Output ranges are disjoint and together they are the
    /// whole output. `parts` are (words, round).
    pub fn table(parts:&[(usize,bool)],output:usize,rows:&[[usize;3]])->Result<Vec<Row>,String> {
        if parts.is_empty() || parts.iter().any(|part|part.0==0) || output==0 || output>=ROUND as usize {return Err("joined put extent differs".into());}
        let mut table=Vec::with_capacity(parts.len()+rows.len()*2+1);let mut starts=Vec::with_capacity(parts.len());let mut message=0usize;
        for (part,&(words,round)) in parts.iter().enumerate() {
            table.push(Row {source:Some((part,0)),target:Target::Slot(message),words,round});starts.push(message);
            message=message.checked_add(words).filter(|&n|n<ROUND as usize).ok_or("joined put message extent overflow")?;
        }
        let mut placed=Vec::with_capacity(rows.len());
        for &[from,to,n] in rows {
            if n==0 || from.checked_add(n).is_none_or(|end|end>message) || to.checked_add(n).is_none_or(|end|end>output) {return Err("joined put row exceeds storage".into());}
            placed.push((to,to+n));let mut done=0;
            while done<n {
                let part=starts.partition_point(|&start|start<=from+done)-1;let inside=from+done-starts[part];let words=(n-done).min(parts[part].0-inside);
                table.push(Row {source:Some((part,inside)),target:Target::Output(to+done),words,round:parts[part].1});done+=words;
            }
        }
        placed.sort_unstable();if placed.windows(2).any(|pair|pair[0].1>pair[1].0) {return Err("joined put output rows overlap".into());}
        let mut cursor=0;
        for (start,end) in placed {if start>cursor {table.push(Row {source:None,target:Target::Output(cursor),words:start-cursor,round:false});}cursor=end;}
        if cursor<output {table.push(Row {source:None,target:Target::Output(cursor),words:output-cursor,round:false});}
        if table.len()>MOST_ROWS {return Err("joined put row count exceeds the table".into());}
        Ok(table)
    }
    /// The encoded table. Slot rows come first; `units` are [slot units, output units].
    pub struct Encoded {pub bytes:Vec<u8>,pub slot_rows:usize,pub units:[u64;2]}
    /// HybridCopyRow bytes. A row is: source address (0: zero row), target address, words with
    /// ROUND, and (in the padding field) the units before this row among the slot rows, or among
    /// the output rows.
    pub fn encode(table:&[Row],parts:&[u64],slot:u64,output:u64,wide:bool)->Result<Encoded,String> {
        encode_wire(table,parts,slot,output,wide,false)
    }
    /// HybridCopyRow bytes with optional lossless 2-BF16-per-u32 slot packing (`bf16_wire`).
    /// Output rows always remain full FP32; slot rows use 2-byte offsets and verify zero low 16 bits on device.
    pub fn encode_wire(table:&[Row],parts:&[u64],slot:u64,output:u64,wide:bool,bf16_wire:bool)->Result<Encoded,String> {
        let address=|base:u64,offset:usize,stride:u64|u64::try_from(offset).ok().and_then(|v|v.checked_mul(stride)).and_then(|v|base.checked_add(v)).ok_or_else(||"joined put address overflow".to_string());
        let mut bytes=Vec::with_capacity(table.len()*24);let mut units=[0u64;2];let slot_rows=table.iter().take_while(|row|matches!(row.target,Target::Slot(_))).count();
        if slot_rows==0 || slot_rows==table.len() || table[slot_rows..].iter().any(|row|matches!(row.target,Target::Slot(_))) {return Err("joined put table order differs".into());}
        for (index,row) in table.iter().enumerate() {
            let is_slot=index<slot_rows;
            let from=match row.source {Some((part,offset))=>{let base=*parts.get(part).ok_or("joined put part differs")?;address(base,offset+row.words,4)?;address(base,offset,4)?},None=>0};
            let (base,offset,stride)=match row.target {Target::Slot(offset)=>(slot,offset,if bf16_wire {2} else {4}),Target::Output(offset)=>(output,offset,4)};
            if is_slot && bf16_wire && (!offset.is_multiple_of(2) || !row.words.is_multiple_of(2)) {return Err("joined put BF16 slot row requires even offset and word count".into());}
            address(base,offset+row.words,stride)?;let to=address(base,offset,stride)?;
            if row.words==0 || row.words>=ROUND as usize {return Err("joined put row extent differs".into());}
            let space=&mut units[usize::from(!is_slot)];
            bytes.extend_from_slice(&from.to_ne_bytes());bytes.extend_from_slice(&to.to_ne_bytes());
            bytes.extend_from_slice(&(row.words as u32|if row.round {ROUND} else {0}).to_ne_bytes());
            bytes.extend_from_slice(&u32::try_from(*space).map_err(|_|"joined put unit count overflow")?.to_ne_bytes());
            *space+=if is_slot && bf16_wire {
                if quad_bf16_slot(from,to,row.words,wide) {row.words/4} else {row.words/2}
            } else if quad(from,to,row.words,wide) {row.words/4} else {row.words} as u64;
        }
        if units[0]>u32::MAX as u64 || units[1]>u32::MAX as u64 {return Err("joined put unit count overflow".into());}
        Ok(Encoded {bytes,slot_rows,units})
    }
    /// The launch: [output blocks, threads]. Block 0 moves the message and publishes it, as the
    /// one-block put; the output blocks after it fill the output, one unit a thread while the
    /// output has at most OUTPUT_BLOCKS*THREADS units (measured on GB10: put-join-check).
    pub fn grid(units:[u64;2])->[u32;2] {
        match SHAPE.load(core::sync::atomic::Ordering::Relaxed) {0=>{},forced=>return [forced>>16,forced&0xffff]}
        [(units[1].div_ceil(THREADS as u64) as u32).clamp(1,OUTPUT_BLOCKS),THREADS]
    }
    pub const THREADS:u32=1024;
    /// The most output blocks of one launch.
    pub const OUTPUT_BLOCKS:u32=64;
    /// Proof tools only: output blocks<<16|threads forces the launch shape (0: the rule).
    pub static SHAPE:core::sync::atomic::AtomicU32=core::sync::atomic::AtomicU32::new(0);
}
#[cfg(all(test,not(target_arch="nvptx64")))]
mod tests {
    use super::*;
    use crate::tp_put_wide::{first,stride};
    /// The kernel's walk over an encoded table. Slot blocks take slot units, output blocks take
    /// output units: first, first+stride, ... of that space; a binary search finds the first
    /// row, later rows are found forward.
    fn run(e:&Encoded,[output_blocks,threads]:[u32;2],wide:bool,mut store:impl FnMut(u64,u64,bool)) {let slot_blocks=1;
        let rows=e.bytes.as_chunks::<24>().0.iter().map(|row|(u64::from_ne_bytes(row[0..8].try_into().unwrap()),u64::from_ne_bytes(row[8..16].try_into().unwrap()),
            u32::from_ne_bytes(row[16..20].try_into().unwrap()),u32::from_ne_bytes(row[20..24].try_into().unwrap()) as usize)).collect::<Vec<_>>();
        for block in 0..slot_blocks+output_blocks {for thread in 0..threads {
            let slot=block<slot_blocks;let (rows,total,grid,block)=if slot {(&rows[..e.slot_rows],e.units[0] as usize,slot_blocks,block)} else {(&rows[e.slot_rows..],e.units[1] as usize,output_blocks,block-slot_blocks)};
            let mut unit=first(block,thread,threads);if unit>=total {continue;}
            let (mut low,mut high)=(0,rows.len());while high-low>1 {let middle=(low+high)/2;if rows[middle].3<=unit {low=middle;} else {high=middle;}}
            let mut row=low;
            loop {
                let (from,to,words,before)=rows[row];let n=(words&!ROUND) as usize;let quad=quad(from,to,n,wide);let at=unit-before;
                if at>=if quad {n/4} else {n} {row+=1;continue;}
                for k in 0..if quad {4} else {1} {let word=(if quad {4*at} else {at}+k) as u64;store(if from==0 {0} else {from+4*word},to+4*word,words&ROUND!=0);}
                unit+=stride(grid,threads);if unit>=total {break;}
            }
        }}
    }
    #[test]
    fn every_slot_and_output_word_has_one_writer_and_the_head_value() {
        // (own words a job, jobs, parts): the step's message shapes, an odd shape, and the three-word argmax message.
        for (count,jobs,cut) in [(2048usize,1usize,1usize),(2048,6,1),(2048,6,6),(1168,1,1),(8192,1,1),(2048,8,2),(4096,4,3),(4840,16,1),(512,25,25),(3,1,1),(1037,3,2)] {
            let words=count*jobs;let src=(0..words).map(|i|(i as u32).wrapping_mul(2654435761)|1).collect::<Vec<_>>();
            // Parts: `cut` nearly equal pieces of the message; odd parts ask for the round.
            let parts=(0..cut).map(|part|(words*(part+1)/cut-words*part/cut,part%2==1)).collect::<Vec<_>>();
            let starts=(0..cut).map(|part|words*part/cut).collect::<Vec<_>>();
            for rank in 0..2 {
                let rows=(0..jobs).map(|job|[job*count,job*2*count+rank*count,count]).collect::<Vec<_>>();
                let table=table(&parts,2*words,&rows).unwrap();
                // Addresses: part p at 0x1000_0000*(p+1) (some unaligned), slot at 0x9000_0000, output at 0xa000_0000.
                let bases=(0..cut).map(|part|0x1000_0000*(part as u64+1)+if part%3==2 {4} else {0}).collect::<Vec<_>>();
                let value=|address:u64,round:bool| {let part=(address>>28) as usize-1;let word=src[starts[part]+((address-bases[part])/4) as usize];if round {word&0xffff_0000} else {word}};
                let mut expected_slot=Vec::new();for (part,&(n,round)) in parts.iter().enumerate() {expected_slot.extend((0..n).map(|i|Some(value(bases[part]+4*i as u64,round))));}
                let mut expected=vec![Some(0u32);2*words];for &[from,to,n] in &rows {expected[to..to+n].copy_from_slice(&expected_slot[from..from+n]);}
                for shape in [[1,1024],[1,256],[5,256],[7,256],[8,1024],[32,1024],[90,256],[2,128]] {let [blocks,threads]=shape;for wide in [false,true] {
                    let encoded=encode(&table,&bases,0x9000_0000,0xa000_0000,wide).unwrap();let (mut slot,mut out)=(vec![None;words],vec![None;2*words]);
                    run(&encoded,shape,wide,|from,to,round| {
                        let (target,at)=if to>=0xa000_0000 {(&mut out,((to-0xa000_0000)/4) as usize)} else {(&mut slot,((to-0x9000_0000)/4) as usize)};
                        assert!(target[at].is_none(),"word {at} has two writers");target[at]=Some(if from==0 {0} else {value(from,round)});
                    });
                    assert_eq!(slot,expected_slot,"slot count={count} jobs={jobs} parts={cut} grid={blocks}x{threads} wide={wide}");
                    assert_eq!(out,expected,"output count={count} jobs={jobs} parts={cut} rank={rank} grid={blocks}x{threads} wide={wide}");
                }}
            }
        }
    }
    #[test]
    fn a_rank_without_rows_fills_the_whole_output_with_zero() {
        assert_eq!(table(&[(8,false)],24,&[]).unwrap(),vec![Row {source:Some((0,0)),target:Target::Slot(0),words:8,round:false},Row {source:None,target:Target::Output(0),words:24,round:false}]);
    }
    #[test]
    fn the_table_refuses_overlap_and_rows_outside_storage() {
        let one=[(8,false)];
        assert!(table(&one,16,&[[0,0,4],[4,2,4]]).is_err());assert!(table(&one,16,&[[6,0,4]]).is_err());assert!(table(&one,16,&[[0,14,4]]).is_err());
        assert!(table(&one,16,&[[0,0,0]]).is_err());assert!(table(&one,0,&[]).is_err());assert!(table(&[],8,&[]).is_err());assert!(table(&[(0,false)],8,&[]).is_err());
        let rows=table(&[(4,false),(4,true)],16,&[[2,8,4]]).unwrap();
        assert_eq!(rows[2..],[Row {source:Some((0,2)),target:Target::Output(8),words:2,round:false},Row {source:Some((1,0)),target:Target::Output(10),words:2,round:true},
            Row {source:None,target:Target::Output(0),words:8,round:false},Row {source:None,target:Target::Output(12),words:4,round:false}]);
    }
    #[test]
    fn encoded_rows_are_addresses_words_round_and_units_before() {
        let Encoded {bytes,slot_rows,units}=encode(&table(&[(8,true)],16,&[[4,8,4]]).unwrap(),&[1024],4096,8192,true).unwrap();assert_eq!((bytes.len(),slot_rows,units),(96,1,[2,1+2+1]));
        assert_eq!(&bytes[..8],&1024u64.to_ne_bytes());assert_eq!(&bytes[8..16],&4096u64.to_ne_bytes());assert_eq!(&bytes[16..20],&(8|ROUND).to_ne_bytes());assert_eq!(&bytes[20..24],&0u32.to_ne_bytes());
        assert_eq!(&bytes[24..32],&1040u64.to_ne_bytes());assert_eq!(&bytes[32..40],&8224u64.to_ne_bytes());assert_eq!(&bytes[44..48],&0u32.to_ne_bytes());
        assert_eq!(&bytes[48..56],&0u64.to_ne_bytes());assert_eq!(&bytes[68..72],&1u32.to_ne_bytes());assert_eq!(&bytes[92..96],&3u32.to_ne_bytes());
        assert_eq!(encode(&table(&[(8,true)],16,&[[4,8,4]]).unwrap(),&[1024],4096,8192,false).unwrap().units,[8,4+8+4]);
        assert!(encode(&table(&[(8,false)],16,&[]).unwrap(),&[u64::MAX-8],0,0,true).is_err());assert!(encode(&table(&[(8,false)],16,&[]).unwrap()[..1],&[64],0,0,true).is_err());
    }
    #[test]
    fn only_step_sized_messages_are_admitted() {
        assert!(admitted(1,1,3,6) && admitted(9,9,3584,7168) && admitted(2,25,11264,22528) && admitted(1,8,16384,32768) && admitted(1,16,77440,154880) && admitted(32,MOST_JOBS,MOST_MESSAGE,MOST_OUTPUT));
        assert!(!admitted(0,1,8,16) && !admitted(MOST_PARTS+1,1,8,16) && !admitted(1,MOST_JOBS+1,8,16) && !admitted(1,1,MOST_MESSAGE+1,16) && !admitted(1,1,8,MOST_OUTPUT+1) && !admitted(1,1,0,16) && !admitted(1,1,8,0));
        // An admitted table is short: every part row, every job row cut at each part edge, and a gap before each and after the last.
        const _:()=assert!(MOST_PARTS+2*(MOST_JOBS+MOST_PARTS)<MOST_ROWS);
    }
    #[test]
    fn bf16_wire_encoding_halves_slot_stride_and_preserves_output_units() {
        let tbl=table(&[(8,false),(4,true)],24,&[[0,0,8],[8,16,4]]).unwrap();
        let e=encode_wire(&tbl,&[0x1000,0x2000],0x9000,0xa000,true,true).unwrap();
        assert_eq!(e.slot_rows,2);
        // Slot row 0: 8 words -> 2 quad units (4 words -> 2 packed u32), to = 0x9000.
        // Slot row 1: 4 words -> 1 quad unit, to = 0x9000 + 8*2 = 0x9010 (8-byte aligned!).
        assert_eq!(e.units[0],3);
        assert_eq!(&e.bytes[8..16],&0x9000u64.to_ne_bytes());
        assert_eq!(&e.bytes[32..40],&0x9010u64.to_ne_bytes());
        // Odd slot word count is refused on bf16_wire.
        let odd=table(&[(3,false)],6,&[[0,0,3]]).unwrap();
        assert!(encode_wire(&odd,&[0x1000],0x9000,0xa000,true,true).is_err());
    }
    #[test]
    fn the_launch_shape_follows_the_output_units() {
        assert_eq!(grid([1,1]),[1,1024]);assert_eq!(grid([4096,1024]),[1,1024]);assert_eq!(grid([4096,1025]),[2,1024]);assert_eq!(grid([4096,8192]),[8,1024]);assert_eq!(grid([4096,1<<30]),[OUTPUT_BLOCKS,1024]);
    }
}
