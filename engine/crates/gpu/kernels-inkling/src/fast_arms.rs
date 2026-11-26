//! One-row arms of the token step with their operand reads issued in groups before their chains (Inkling image
//! only). Each word is the word of the shared arm (the contiguous `RelBias` form of `run`):
//! the same products, the same sums in the same order, the same roundings. Only the order of the loads changes: a
//! group of `GROUP` places is read into registers first, so the reads of a group go out together instead of one
//! chain step at a time.
use crate::fixed_math::rn;

/// Places read before their terms are added. A part of the read order, not of the sum order.
const GROUP:usize=16;

/// The contiguous `RelBias` word of `head` and `distance`: the chain over the `width` words of the head's state, in
/// rising place, each state word read as BF16, rounded once. The state words and table words of a group are read
/// first.
#[inline(always)]
pub(crate) fn rel_word(width:usize,extent:usize,head:usize,distance:usize,state:&[f32],table:&[f32])->f32 {
    let row=&state[head*width..(head+1)*width];
    let mut sum=0.0f32;let mut i=0;
    while i<width {
        let (mut values,mut taps)=([0.0f32;GROUP],[0.0f32;GROUP]);
        let mut j=0;
        while j<GROUP {if i+j<width {values[j]=row[i+j];taps[j]=table[(i+j)*extent+distance];}j+=1;}
        j=0;
        while j<GROUP {if i+j<width {sum+=rn(values[j])*taps[j];}j+=1;}
        i+=GROUP;
    }
    rn(sum)
}
