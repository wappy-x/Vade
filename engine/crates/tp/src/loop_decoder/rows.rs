//! Many rows a step: the row list, the limits of a step, and the one launch helper for an arm on many rows.
//!
//! A step takes a list of rows. A row is one token of one chat at its position. The rows of one chat are
//! consecutive positions. Row order: the chats in the order of the step, in each chat the positions rising.
//! Every value of a step is one backend value, row-major: `[rows][words of one row]`.
//! A token step of today is the step with one row, and it keeps its launches.
//!
//! Where rows see each other, and only inside one chat: the key and the value convolution rings, the key and the
//! value planes, the two stream convolution rings. Everything else reads one row alone. No row waits for another
//! row inside a visit: every input of a place where rows meet exists before its launch.
//!
//! The contract of the pieces, one file each:
//! - this file: `Rows`, `Limits`, `rows_operation`.
//! - `mix`: `Weights::product_rows`. `Token` and its four calls take `[rows][width]` values (piece D2).
//! - `experts`: `sum` and `close` take `[rows][..]` values (piece D2).
//! - `state`: `row_rings`, `commit`, `discard` (piece D1).
//! - `attention`: `half_rows` (piece D1).
//! - the driver: `Driver::with_limits`, `Driver::step_rows`, `ChatStep`, `Wanted`, `StepRows`.
//!
//! The exactness rule of every piece: a row of a step of many rows has the words of that row in steps of one row.
//! A row gets the same input words and runs the same unit code; a product keeps its chain.
use super::mix::{need,number};
use crate::{Result,bounded_decoder::Compute};
use spark_gpu_probe::{causal_ops::Shape,prompt_ops};

/// The rows of one chat in a step: the position of its first row, and its count of rows (at least 1).
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct ChatRows {pub first:u64,pub count:usize}
/// The rows of one step: the chats in step order. The rows of chat `i` are the rows
/// `offset(i) .. offset(i) + count(i)` of every `[rows][..]` value of the step.
#[derive(Debug,Clone,PartialEq,Eq)]
pub struct Rows {chats:Vec<ChatRows>,offsets:Vec<usize>,total:usize}
impl Rows {
    pub fn new(chats:Vec<ChatRows>)->Result<Self> {
        need(!chats.is_empty() && chats.iter().all(|chat|chat.count>0),"a step has no chat, or a chat with no row")?;
        let (mut offsets,mut total)=(Vec::with_capacity(chats.len()),0usize);
        for chat in &chats {
            need(chat.first.checked_add(chat.count as u64).is_some(),"the rows of a chat pass the largest position")?;
            offsets.push(total);total=total.checked_add(chat.count).ok_or("loop decoder: the rows of a step overflow")?;
        }
        Ok(Self {chats,offsets,total})
    }
    /// The chats in step order.
    pub fn chats(&self)->&[ChatRows] {&self.chats}
    /// The first row of chat `chat` in the step.
    pub fn offset(&self,chat:usize)->usize {self.offsets[chat]}
    /// The rows of the step, `R`.
    pub fn total(&self)->usize {self.total}
    /// The position of each row, in row order.
    pub fn positions(&self)->impl Iterator<Item=u64>+'_ {self.chats.iter().flat_map(|chat|(0..chat.count as u64).map(move |row|chat.first+row))}
}
/// The limits of a step of one driver. They are sizes of the state and of the launches, so the driver holds them.
/// - `step_rows`: the most rows of one step, over all its chats. A ring of a sliding layer holds
///   `step_rows - 1` rows more than its window, so a row of a step never reads a plane row that a later row of the
///   same step wrote over (ruling ROWS, point 2).
/// - `key_bytes`: the most bytes of the values over the keys (the bias, the scores, the weights) of one launch
///   group. The attention half runs the part over the keys in groups of rows under this limit. A group changes no
///   word: a row never reads a score of another row.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Limits {pub step_rows:usize,pub key_bytes:usize}
impl Limits {
    /// One row a step: the state sizes of a driver that takes no rows.
    pub const ONE:Self=Self {step_rows:1,key_bytes:1<<30};
}
/// The bit of input `input` in the `shared` word of `rows_operation`.
pub const fn shared(input:usize)->u32 {1<<input}
/// One arm for `rows` rows. `rows` is any count from 1: the rows of a step, of one chat, or of a group of them.
/// Each input is `[rows][part]`, or it is given whole to every row when its bit in `shared` is set, or it is absent.
/// The weight is one for all rows. The result is `[rows][words of one launch]`, row-major.
/// Each row has the words of `Compute::operation` on its own parts: the same unit code runs.
/// - With one row this is `operation` with the same shape.
/// - With more rows it is one launch of the rows entry of the arms (`Compute::block_rows`, Inkling's own kernel
///   entry; Ruling R6g) when the backend has it and the counts fit (`block_rows::lengths`). The shared inputs are this
///   bit mask.
/// - Else it is one `operation` a row and a join. Each row gets its own part of every input that is not shared,
///   also of the step words.
pub fn rows_operation<C:Compute>(compute:&C,s:Shape,rows:usize,x:[Option<&C::Value>;4],shared:u32,w:&[f32])->Result<C::Value> {
    need(rows>0,"a launch for rows has no row")?;
    if rows==1 {return compute.operation(s,x,w);}
    let mut parts=[0usize;4];
    for (n,input) in x.iter().enumerate() {
        let Some(value)=input else {continue;};
        if shared>>n&1==1 {continue;}
        let words=compute.words(value);need(words>0 && words.is_multiple_of(rows),"an input of a launch for rows holds no whole count of rows")?;
        parts[n]=words/rows;
    }
    if spark_gpu_probe::block_rows::lengths(s,rows,shared).is_some() && let Some(value)=compute.block_rows(s,rows,shared,x,w)? {return Ok(value);}
    let mut outs=Vec::with_capacity(rows);
    for row in 0..rows {
        let mut own:[Option<C::Value>;4]=[None,None,None,None];
        for n in 0..4 {if let (Some(value),true)=(x[n],parts[n]>0) {own[n]=Some(compute.range(value,row*parts[n],parts[n])?);}}
        let input=|n:usize|if parts[n]>0 {own[n].as_ref()} else {x[n]};
        outs.push(compute.operation(s,[input(0),input(1),input(2),input(3)],w)?);
    }
    compute.join(&outs.iter().collect::<Vec<_>>())
}
/// A float holds every whole number up to this one: the most source rows of a gather.
pub(super) const GATHER_ROWS:usize=1<<24;
/// Rows of `width` words out of `x`, which holds whole rows of that width: row `i` of the result is row `table[i]`
/// of `x`. Bits move and nothing is computed. A run of rows in rising order is a range and needs no launch; any
/// other table is one launch (`prompt_ops::Op::PermuteRows`).
pub fn gather<C:Compute>(compute:&C,x:&C::Value,table:&[usize],width:usize)->Result<C::Value> {
    need(width>0 && !table.is_empty() && compute.words(x).is_multiple_of(width),"a gather source holds no whole rows")?;
    let source=compute.words(x)/width;
    need(source<=GATHER_ROWS && table.iter().all(|&row|row<source),"a gather names a row outside its source")?;
    if table.windows(2).all(|pair|pair[1]==pair[0]+1) {return compute.range(x,table[0]*width,table.len()*width);}
    let mut s=prompt_ops::Shape::new(prompt_ops::Op::PermuteRows,1).ok_or("loop decoder: a gather exceeds the launch format")?;
    s.rows=number(table.len())?;s.cols=number(width)?;s.first=number(source)?;
    need(s.lengths().is_some(),"a gather exceeds the launch format")?;
    let table=compute.upload(&table.iter().map(|&row|row as f32).collect::<Vec<_>>())?;
    compute.prompt(s,[Some(x),Some(&table),None,None,None],&[])
}
/// `gather` with its index table already on the backend: `rows` row numbers as floats, each a row of `x`. The caller
/// made the table (`swap_table`), so its rows are inside the source.
pub fn gather_table<C:Compute>(compute:&C,x:&C::Value,table:&C::Value,rows:usize,width:usize)->Result<C::Value> {
    need(width>0 && compute.words(table)==rows && compute.words(x).is_multiple_of(width) && compute.words(x)/width<=GATHER_ROWS,"a gather table or source holds no whole rows")?;
    let mut s=prompt_ops::Shape::new(prompt_ops::Op::PermuteRows,1).ok_or("loop decoder: a gather exceeds the launch format")?;
    s.rows=number(rows)?;s.cols=number(width)?;s.first=number(compute.words(x)/width)?;
    need(s.lengths().is_some(),"a gather exceeds the launch format")?;
    compute.prompt(s,[Some(x),Some(table),None,None,None],&[])
}
/// The index table of `swap`: row `j * first + i` of the result is row `i * second + j` of the source.
pub fn swap_table(first:usize,second:usize)->Vec<f32> {(0..second).flat_map(|j|(0..first).map(move |i|(i*second+j) as f32)).collect()}
/// `[first][second][width]` to `[second][first][width]`: the same rows of `width` words in the other order.
/// With one row on a side the two orders are one, and no launch runs.
pub fn swap<C:Compute>(compute:&C,x:&C::Value,first:usize,second:usize,width:usize)->Result<C::Value> {
    need(first>0 && second>0 && width>0 && first.checked_mul(second).and_then(|rows|rows.checked_mul(width))==Some(compute.words(x)),"a value differs from the two counts of its rows")?;
    gather(compute,x,&(0..second).flat_map(|j|(0..first).map(move |i|i*second+j)).collect::<Vec<_>>(),width)
}
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{fixture::Fixture,mix::{Weight,Weights,shape}};
    use crate::bounded_decoder::Host;
    use spark_gpu_probe::causal_ops::Op;
    fn bits(x:&[f32])->Vec<u32> {x.iter().map(|v|v.to_bits()).collect()}
    #[test]
    fn the_row_list_gives_each_chat_its_rows() {
        let rows=Rows::new(vec![ChatRows {first:5,count:3},ChatRows {first:0,count:1},ChatRows {first:9,count:2}]).unwrap();
        assert_eq!((rows.total(),rows.offset(0),rows.offset(1),rows.offset(2)),(6,0,3,4));
        assert_eq!(rows.positions().collect::<Vec<_>>(),[5,6,7,0,9,10]);
        assert!(Rows::new(Vec::new()).is_err() && Rows::new(vec![ChatRows {first:0,count:0}]).is_err() && Rows::new(vec![ChatRows {first:u64::MAX,count:1}]).is_err());
    }
    #[test]
    fn an_arm_for_rows_gives_each_row_the_words_of_its_own_launch() {
        let fixture=Fixture::small();let width=fixture.description.hidden as usize;
        let x=crate::loop_decoder::fixture::made("rows test input",3*width,12,-2,2);
        // A weight for all rows.
        let weight=Weights::<Host>::words(&fixture,Weight::LastNorm).unwrap();
        let mut s=shape(Op::RoundedNorm);s.heads=1;s.dim=width as u32;s.taps=1;s.epsilon=1.0e-6;
        let rows=rows_operation(&Host,s,3,[Some(&x),None,None,None],0,weight).unwrap();
        for row in 0..3 {assert_eq!(bits(&rows[row*width..(row+1)*width]),bits(&Host.operation(s,[Some(&x[row*width..(row+1)*width].to_vec()),None,None,None],weight).unwrap()));}
        assert_eq!(bits(&rows_operation(&Host,s,1,[Some(&x[..width].to_vec()),None,None,None],0,weight).unwrap()),bits(&rows[..width]));
        // An input that all rows share: the message of a gate.
        let (gates,message)=(crate::loop_decoder::fixture::made("rows test gates",3*2*width,12,-2,2),x[..width].to_vec());
        let mut s=shape(Op::GateBlend);s.dim=width as u32;
        let rows=rows_operation(&Host,s,3,[Some(&gates),Some(&x),Some(&message),None],shared(2),&[]).unwrap();
        for row in 0..3 {
            let one=Host.operation(s,[Some(&gates[row*2*width..(row+1)*2*width].to_vec()),Some(&x[row*width..(row+1)*width].to_vec()),Some(&message),None],&[]).unwrap();
            assert_eq!(bits(&rows[row*width..(row+1)*width]),bits(&one));
        }
        assert!(rows_operation(&Host,s,0,[Some(&gates),Some(&x),Some(&message),None],shared(2),&[]).is_err());
        assert!(rows_operation(&Host,s,2,[Some(&gates),Some(&x),Some(&message),None],shared(2),&[]).is_err());
    }
    #[test]
    fn a_gather_moves_whole_rows_and_a_swap_changes_their_order() {
        let x=(0..24).map(|i|i as f32).collect::<Vec<_>>();
        // A run of rows is a range; another table is a launch. Both give the rows that the table names.
        assert_eq!(gather(&Host,&x,&[2,3,4],2).unwrap(),[4.0,5.0,6.0,7.0,8.0,9.0]);
        assert_eq!(gather(&Host,&x,&[5,0,0,11],2).unwrap(),[10.0,11.0,0.0,1.0,0.0,1.0,22.0,23.0]);
        assert!(gather(&Host,&x,&[12],2).is_err() && gather(&Host,&x,&[],2).is_err() && gather(&Host,&x,&[0],5).is_err());
        // `[2][3][4]` to `[3][2][4]`, and back.
        let swapped=swap(&Host,&x,2,3,4).unwrap();
        assert_eq!(swapped[..8],[0.0,1.0,2.0,3.0,12.0,13.0,14.0,15.0]);assert_eq!(swap(&Host,&swapped,3,2,4).unwrap(),x);
        assert_eq!(swap(&Host,&x,1,6,4).unwrap(),x);assert_eq!(swap(&Host,&x,6,1,4).unwrap(),x);
        assert!(swap(&Host,&x,2,3,5).is_err());
    }
    #[test]
    fn a_product_for_rows_is_one_product_a_row() {
        let fixture=Fixture::small();let width=fixture.description.hidden as usize;
        let x=crate::loop_decoder::fixture::made("rows test input",3*width,8,-2,2);
        let rows=fixture.product_rows(&Host,Weight::Router(0),&x,3).unwrap();
        let each=rows.len()/3;assert!(each>0 && rows.len()==3*each);
        for row in 0..3 {assert_eq!(bits(&rows[row*each..(row+1)*each]),bits(&Weights::product(&fixture,&Host,Weight::Router(0),&x[row*width..(row+1)*width].to_vec()).unwrap()));}
        assert!(fixture.product_rows(&Host,Weight::Router(0),&x,0).is_err());
    }
}
