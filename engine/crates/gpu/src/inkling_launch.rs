//! The launch ABI of the Inkling image (crate `kernels-inkling`): its recorded-launch kind numbers and the layout of
//! one recorded command. One source for the host and the GPU. The general image reads `resident_launch.rs`, which
//! stays byte for byte private main's; the host test below ties these numbers and this layout to the host's `Kind`
//! and `Command`, so a command the host records is the command this image replays.
//!
//! How a seat adds an entry to the Inkling image: (1) one kind number here and the same number in the host's `Kind`
//! (`resident_launch_host.rs`, a line marked `// INKLING`, and its entry name in `Kind::kernel`); (2) in
//! `kernels-inkling/src/lib.rs` one arm in the `match` of `run` and one `entry` line. Its row in `REGISTERS.txt`.
/// `resident_launch_block_arms`: the 14 Inkling block arms, one launch.
pub const BLOCK_ARMS:u32=89;
/// `resident_launch_plain_packed`: the packed row product of one expert matrix (`packed_product`).
pub const PLAIN_PACKED:u32=88;
/// `resident_launch_plain_packed_group`: the packed row products of up to 32 matrices of one shape in one launch.
pub const PLAIN_PACKED_GROUP:u32=95;
/// `resident_launch_plain_packed_picked`: the grouped packed product whose experts the route picked on the device.
pub const PLAIN_PACKED_PICKED:u32=96;
/// `resident_launch_block_rows`: many rows of one block arm in one launch.
pub const BLOCK_ROWS:u32=91;
/// `resident_launch_plain_staged`: the staged BF16 row product (`staged_product`), one matrix with one input row.
pub const PLAIN_STAGED:u32=90;
/// `resident_launch_plain_staged_rows`: the rows tile of the staged BF16 product, one matrix with many input rows.
pub const PLAIN_STAGED_ROWS:u32=92;
/// `resident_launch_plain_staged_narrow`: the narrow staged BF16 row product (`staged_narrow`), one matrix of few rows
/// with one input row.
pub const PLAIN_STAGED_NARROW:u32=93;
/// `resident_launch_block_scores`: `WindowScores` at rows with the key rows staged a tile at a time (`block_scores`).
pub const BLOCK_SCORES:u32=97;
/// `resident_launch_block_softmax`: `SinkSoftmax` at rows, a block a (row, head) (`block_attention::softmax_rows_stage`).
pub const BLOCK_SOFTMAX:u32=98;
/// One recorded command: the layout of the host's `resident_launch::Command` (host test).
#[repr(C,align(16))]
#[derive(Clone,Copy)]
pub struct Command {pub kind:u32,pub reserved:[u32;3],pub payload:[u8;1024]}
/// The 32-bit words of one Command: the length of the one-command table of a direct launch.
pub const COMMAND_WORDS:usize=core::mem::size_of::<Command>()/4;
#[cfg(test)] mod tests {
    use super::*;
    use crate::resident_launch::{Command as Host,Kind};
    #[test] fn the_inkling_abi_is_the_hosts() {
        assert_eq!((Kind::BlockArms as u32,Kind::BlockRows as u32,Kind::PlainStaged as u32,Kind::PlainStagedRows as u32,Kind::PlainStagedNarrow as u32,Kind::PlainPacked as u32,Kind::PlainPackedGroup as u32,Kind::PlainPackedPicked as u32,Kind::BlockScores as u32,Kind::BlockSoftmax as u32),(BLOCK_ARMS,BLOCK_ROWS,PLAIN_STAGED,PLAIN_STAGED_ROWS,PLAIN_STAGED_NARROW,PLAIN_PACKED,PLAIN_PACKED_GROUP,PLAIN_PACKED_PICKED,BLOCK_SCORES,BLOCK_SOFTMAX));
        assert_eq!((Kind::BlockArms.kernel(),Kind::BlockRows.kernel(),Kind::PlainStaged.kernel(),Kind::PlainStagedRows.kernel(),Kind::PlainStagedNarrow.kernel(),Kind::PlainPacked.kernel(),Kind::PlainPackedGroup.kernel(),Kind::PlainPackedPicked.kernel(),Kind::BlockScores.kernel(),Kind::BlockSoftmax.kernel()),("resident_launch_block_arms","resident_launch_block_rows","resident_launch_plain_staged","resident_launch_plain_staged_rows","resident_launch_plain_staged_narrow","resident_launch_plain_packed","resident_launch_plain_packed_group","resident_launch_plain_packed_picked","resident_launch_block_scores","resident_launch_block_softmax"));
        use core::mem::{align_of,offset_of,size_of};
        assert_eq!((size_of::<Command>(),align_of::<Command>()),(size_of::<Host>(),align_of::<Host>()));
        assert_eq!([offset_of!(Command,kind),offset_of!(Command,reserved),offset_of!(Command,payload)],[offset_of!(Host,kind),offset_of!(Host,reserved),offset_of!(Host,payload)]);
        // No general kind has an Inkling number.
        assert!(Kind::ALL.iter().filter(|k|[BLOCK_ARMS,BLOCK_ROWS,PLAIN_STAGED,PLAIN_STAGED_ROWS,PLAIN_STAGED_NARROW,PLAIN_PACKED,PLAIN_PACKED_GROUP,PLAIN_PACKED_PICKED,BLOCK_SCORES,BLOCK_SOFTMAX].contains(&(**k as u32))).count()==10);
    }
}
