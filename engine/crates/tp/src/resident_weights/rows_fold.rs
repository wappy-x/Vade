//! ROWS FOLD, the host side of `Weights::apply_routed_rows_fold`: where a token's planes lie in the row-major planes of
//! a batch step, and the folded rows with the message that carries them. Everything here is held in place: a decode
//! step makes no host allocation for it (`tests/rows_fold_alloc.rs` counts). It compiles on every host, so the host
//! tests run the words the device path runs.
use super::Rows;
use crate::Result;
use std::ops::Range;
/// Most tokens of one fold: the chats of a batch step.
pub const TOKENS:usize=crate::bounded_decoder::BATCH_ROWS;
/// Where one token lies in the planes of a fold, in words.
#[derive(Debug,Clone,PartialEq,Eq)]
pub struct Planes {
    /// Its picks in the ID plane, and its weights in the weight plane (`m * picks` words each, row-major).
    pub picked:Range<usize>,
    /// This rank's rows of its picks in the projected plane (`m * picks * rows.count` words, one row a (token, pick)).
    pub projected:Range<usize>,
    /// This rank's rows of its shared row in the shared plane, when the fold has one.
    pub own:Option<Range<usize>>,
}
/// The planes of token `token` of `m`. `picks`: picks a token. `rows`: this rank's rows of the fold's output.
/// `shared`: the words of the shared plane when the fold has a shared row: `m * rows.total` (the gathered rows: this
/// rank reads its own part of each) or `m * rows.count` (this rank's rows alone). Err: an extent outside these forms.
pub fn planes(token:usize,m:usize,picks:usize,rows:Rows,shared:Option<usize>)->Result<Planes> {
    let local=rows.count;
    if token>=m || m>TOKENS || picks==0 || local==0 || rows.start.checked_add(local).is_none_or(|end|end>rows.total) {return Err("rows fold token or row extent differs".into());}
    let each=picks.checked_mul(local).ok_or("rows fold extent overflow")?;
    let own=match shared {
        None=>None,
        Some(words) if words==m*local=>Some(token*local..(token+1)*local),
        Some(words) if words==m*rows.total=>{let at=token*rows.total+rows.start;Some(at..at+local)},
        Some(_)=>return Err("rows fold shared row extent differs".into()),
    };
    Ok(Planes {picked:token*picks..(token+1)*picks,projected:token*each..(token+1)*each,own})
}
/// The folded rows of a fold, one a token in token order, held in place.
pub struct Folded<T> {rows:[Option<T>;TOKENS],len:usize}
impl<T> Default for Folded<T> {fn default()->Self {Self {rows:std::array::from_fn(|_|None),len:0}}}
impl<T> Folded<T> {
    /// The next token's folded row.
    pub fn push(&mut self,row:T)->Result<()> {
        let slot=self.rows.get_mut(self.len).ok_or("rows fold holds more tokens than a batch step")?;
        *slot=Some(row);self.len+=1;Ok(())
    }
    pub fn len(&self)->usize {self.len}
    pub fn is_empty(&self)->bool {self.len==0}
    /// The one message of the fold: every folded row in token order, none still to round (the fold made finished words).
    /// The first `len()` parts are the message; the entries behind them repeat a row and are not read.
    pub fn parts(&self)->Result<[(&T,bool);TOKENS]> {
        let first=self.rows[0].as_ref().ok_or("rows fold has no folded row")?;
        let mut parts=[(first,false);TOKENS];
        for (part,row) in parts.iter_mut().zip(&self.rows[..self.len]) {part.0=row.as_ref().ok_or("rows fold row is absent")?;}
        Ok(parts)
    }
}
/// The rows of every part of the message: the same rows a token, in place.
pub fn message_rows(rows:Rows)->[Rows;TOKENS] {[rows;TOKENS]}
