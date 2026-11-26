//! Whether a resident join may return a view of its parts in place of a copy.
#![forbid(unsafe_code)]
/// One join part as the view decision reads it: its owner allocation (`None`: the part is
/// not a view of shared storage), the device address of its first word, and its words.
pub(crate) type Part<O>=(Option<O>,u64,usize);
/// Whether a join may return a view of its parts in place of a copy. All of these hold:
/// the caller ordered no copy (`copy`) and no words past the joined ones (`reserve`); no
/// part shares a byte with the copy table (`table`: its address and its words), because
/// the next copying join writes that table, and the copy path refuses such a source, as
/// on rc.9; the parts lie back to back, in join order, in one shared allocation. A part
/// of zero words reads nothing and does not count. `words` is the sum of the part words.
pub(crate) fn is_view<O:PartialEq+Copy>(parts:impl Iterator<Item=Part<O>>+Clone,table:Option<(u64,usize)>,words:usize,reserve:usize,copy:bool)->bool {
    let in_table=|&(_,start,count):&Part<O>|table.is_some_and(|(table_start,table_words)|
        start<table_start+(table_words*4) as u64 && table_start<start+(count*4) as u64);
    let mut live=parts.filter(|part|part.2>0);
    if copy || reserve>words || live.clone().any(|part|in_table(&part)) {return false;}
    let Some((Some(owner),first,_))=live.clone().next() else {return false;};
    let mut next=first;
    live.all(|(other,address,count)|{let same=other==Some(owner) && address==next;next=next.saturating_add((count*4) as u64);same})
}
#[cfg(test)] mod tests {
    use super::{is_view,Part};
    /// Two allocations. Addresses are byte addresses; a word is four bytes.
    const A:Option<u8>=Some(1);const B:Option<u8>=Some(2);
    fn view(parts:&[Part<u8>],table:Option<(u64,usize)>,reserve:usize,copy:bool)->bool {
        is_view(parts.iter().copied(),table,parts.iter().map(|part|part.2).sum(),reserve,copy)
    }
    /// A join that orders no copy and no spare words.
    fn plain(parts:&[Part<u8>],table:Option<(u64,usize)>)->bool {view(parts,table,0,false)}
    #[test] fn back_to_back_parts_of_one_allocation_are_a_view() {
        assert!(plain(&[(A,4096,8)],None),"one part");
        assert!(plain(&[(A,4096,8),(A,4128,4),(A,4144,1)],None));
        // A part of zero words reads nothing: it may sit anywhere, in any allocation.
        assert!(plain(&[(B,0,0),(A,4096,8),(None,77,0),(A,4128,4)],None));
        assert!(!plain(&[(A,4096,8),(A,4132,4)],None),"a gap");
        assert!(!plain(&[(A,4128,4),(A,4096,8)],None),"the wrong order");
        assert!(!plain(&[(A,4096,8),(A,4124,4)],None),"parts that overlap");
        assert!(!plain(&[(A,4096,8),(B,4128,4)],None),"two allocations");
        assert!(!plain(&[(None,4096,8)],None),"a part that owns its storage");
        assert!(!plain(&[(A,4096,8),(None,4128,4)],None),"a later part that owns its storage");
        assert!(!plain(&[],None),"no part");
        assert!(!plain(&[(A,4096,0)],None),"no word");
    }
    #[test] fn a_part_that_shares_a_byte_with_the_copy_table_is_copied() {
        // The table is 16 words at 8192: bytes 8192..8256. This is the guard of 1471a237f.
        // Without it each join of this first group is a view of bytes the next join writes.
        let table=Some((8192,16));
        assert!(!plain(&[(A,8192,1)],table),"the table's first word: the aliased source of the views check");
        assert!(!plain(&[(A,8192,16)],table),"the whole table");
        assert!(!plain(&[(A,8188,2)],table),"the part's last word is the table's first word");
        assert!(!plain(&[(A,8252,2)],table),"the part's first word is the table's last word");
        assert!(!plain(&[(A,8100,100)],table),"a part that holds the whole table");
        assert!(!plain(&[(A,8184,2),(A,8192,1)],table),"the second of two back-to-back parts");
        // A part that only touches the table shares no byte with it.
        assert!(plain(&[(A,8184,2)],table),"it ends where the table starts");
        assert!(plain(&[(A,8256,2)],table),"it starts where the table ends");
        assert!(plain(&[(A,8200,0),(A,4096,8)],table),"a part of zero words inside the table");
        assert!(plain(&[(A,8192,16)],None),"no table yet");
    }
    #[test] fn a_copy_order_or_spare_words_always_copy() {
        let parts=[(A,4096,8),(A,4128,4)];
        assert!(view(&parts,None,0,false));
        assert!(view(&parts,None,12,false),"exactly the joined words");
        assert!(!view(&parts,None,13,false),"one spare word needs storage of its own");
        // The decoder orders a copy when a part is a kept row or a held cache row (29592d186).
        assert!(!view(&parts,None,0,true));
        assert!(!view(&[(A,4096,8)],None,0,true));
    }
}
