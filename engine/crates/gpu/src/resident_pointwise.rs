use crate::resident_launch::{Command,Kind};
const LIMIT:usize=31;
#[derive(Clone,Copy,Debug)]
struct Move {source:u64,target:u64,words:u32,round:bool,origin:i64}
#[derive(Clone)]
pub(crate) struct Plan {moves:Vec<Move>}
fn word(bytes:&[u8],at:usize)->Option<u64> {Some(u64::from_ne_bytes(bytes.get(at..at+8)?.try_into().ok()?))}
fn small(bytes:&[u8],at:usize)->Option<u32> {Some(u32::from_ne_bytes(bytes.get(at..at+4)?.try_into().ok()?))}
fn overlaps(left:u64,left_words:u32,right:u64,right_words:u32)->bool {
    left<right.saturating_add(right_words as u64*4) && right<left.saturating_add(left_words as u64*4)
}
impl Plan {
    pub(crate) fn read(command:&Command,staged:&[u8],stage:u64,rows:u32)->Option<Self> {
        let bytes=&command.payload;let mut plan=Self {moves:Vec::new()};
        match command.kind {
            kind if kind==Kind::Pointwise as u32=>{
                let count=usize::try_from(word(bytes,0)?).ok()?;if count==0 || count>LIMIT {return None;}
                for index in 0..count {let at=16+index*32;
                    plan.moves.push(Move {source:word(bytes,at)?,target:word(bytes,at+8)?,words:small(bytes,at+16)?,round:small(bytes,at+20)?!=0,origin:i64::try_from(word(bytes,at+24)?).ok()?});}
            },
            kind if kind==Kind::Round as u32 || kind==Kind::CopyWords as u32 || kind==Kind::ZeroWords as u32=>{
                let words=if kind==Kind::Round as u32 {small(bytes,16)?} else {u32::try_from(word(bytes,16)?).ok()?};
                if !plan.push(Move {source:word(bytes,0)?,target:word(bytes,8)?,words,round:kind==Kind::Round as u32,origin:0}) {return None;}
            },
            kind if kind==Kind::Copy as u32=>{
                let table=word(bytes,24)?;
                if table==0 {
                    if !plan.push(Move {source:word(bytes,0)?,target:word(bytes,8)?,words:small(bytes,16)?,round:false,origin:0}) {return None;}
                } else {
                    if stage==0 || rows as usize>LIMIT {return None;}
                    let at=usize::try_from(table.checked_sub(stage)?).ok()?;
                    let table=staged.get(at..at.checked_add(rows as usize*24)?)?;
                    for row in table.as_chunks::<24>().0 {
                        if !plan.push(Move {source:word(row,0)?,target:word(row,8)?,words:small(row,16)?,round:false,origin:0}) {return None;}
                    }
                }
            },
            _=>return None,
        }
        (!plan.moves.is_empty()).then_some(plan)
    }
    fn push(&mut self,mut next:Move)->bool {
        if self.moves.len()>=LIMIT || next.words==0 || next.target==0 || !next.target.is_multiple_of(4) || !next.source.is_multiple_of(4) {return false;}
        if next.source!=0 && next.source!=next.target && overlaps(next.source,next.words,next.target,next.words) {return false;}
        let mut origin=None;
        for previous in &self.moves {
            for (new_address,old_address) in [(next.source,previous.target),(next.target,previous.source),(next.target,previous.target)] {
                if new_address==0 || old_address==0 || !overlaps(new_address,next.words,old_address,previous.words) {continue;}
                let offset=previous.origin as i128+(new_address as i128-old_address as i128)/4;
                let Ok(offset)=i64::try_from(offset) else {return false;};
                if origin.is_some_and(|value|value!=offset) {return false;}
                origin=Some(offset);
            }
        }
        let end=self.moves.iter().map(|operation|operation.origin as i128+operation.words as i128).max().unwrap_or(0);
        let Ok(end)=i64::try_from(end) else {return false;};
        next.origin=origin.unwrap_or(end);self.moves.push(next);true
    }
    pub(crate) fn then(mut self,next:Self)->Option<(Command,u32)> {
        for operation in next.moves {if !self.push(operation) {return None;}}
        let low=self.moves.iter().map(|operation|operation.origin).min()?;
        let high=self.moves.iter().map(|operation|operation.origin as i128+operation.words as i128).max()?;
        let words=u32::try_from(high-low as i128).ok()?;
        let mut command=Command::new(Kind::Pointwise);command.word(0,self.moves.len() as u64);
        for (index,operation) in self.moves.iter().enumerate() {
            let at=16+index*32;command.put(at,&operation.source.to_ne_bytes());command.put(at+8,&operation.target.to_ne_bytes());
            command.put(at+16,&operation.words.to_ne_bytes());command.put(at+20,&u32::from(operation.round).to_ne_bytes());
            command.put(at+24,&u64::try_from(operation.origin as i128-low as i128).ok()?.to_ne_bytes());
        }
        Some((command,words.div_ceil(128)))
    }
}
#[cfg(test)] mod tests {
    use super::*;
    fn copy(source:u64,target:u64,words:u64)->Command {let mut command=Command::new(Kind::CopyWords);command.word(0,source);command.word(1,target);command.word(2,words);command}
    fn plan(command:&Command)->Plan {Plan::read(command,&[],0,1).unwrap()}
    #[test] fn shifted_dependencies_have_the_same_thread() {
        let (command,blocks)=plan(&copy(128,1024,200)).then(plan(&copy(1040,4096,100))).unwrap();
        assert_eq!(blocks,2);assert_eq!(word(&command.payload,72),Some(4));
        let joined=plan(&command).then(plan(&copy(4096,8200,100))).unwrap().0;
        assert_eq!(word(&joined.payload,104),Some(4));
    }
    #[test] fn conflicting_aliases_and_partial_in_place_moves_are_rejected() {
        let first=plan(&copy(128,1024,200));
        assert!(first.then(plan(&copy(1040,128,100))).is_none());
        assert!(Plan::read(&copy(128,132,100),&[],0,1).is_none());
    }
    #[test] fn dependency_can_start_before_an_earlier_range() {
        let joined=plan(&copy(128,1040,100)).then(plan(&copy(1024,4096,200))).unwrap().0;
        assert_eq!(word(&joined.payload,40),Some(4));assert_eq!(word(&joined.payload,72),Some(0));
    }
    #[test] fn copy_list_reads_only_retained_staged_rows() {
        let mut command=Command::new(Kind::Copy);command.word(3,4096);let mut table=Vec::new();
        for (source,target) in [(128u64,1024u64),(512,2048)] {table.extend_from_slice(&source.to_ne_bytes());table.extend_from_slice(&target.to_ne_bytes());table.extend_from_slice(&16u32.to_ne_bytes());table.extend_from_slice(&0u32.to_ne_bytes());}
        assert_eq!(Plan::read(&command,&table,4096,2).unwrap().moves.len(),2);
        assert!(Plan::read(&command,&table[..47],4096,2).is_none());
    }
}
