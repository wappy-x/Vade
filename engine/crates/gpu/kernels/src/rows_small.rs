//! Prompt StepRows for a SMALL token count (1 to 16): the KDA recurrence of a draft verify pass. Writes the same bytes as the generic
//! one-thread-per-column code in `prompt_rows` (`prompt_ops::step_column`): the same expressions in the same order, no fused multiply-add added.
//!
//! The fused single-token step (`kda_step`) in a loop over the tokens, with dim 128. One warp owns (head, 32 values), lane = value. Token 0 reads the
//! incoming state; the new state goes to the public `[head][key][value]` layout of the output and later tokens read it back from there (each lane reads
//! and writes only its own value words, in the same order as the serial column), and each token's output word goes to `[tokens][heads][dim]` after the
//! state. Launch: 32-thread blocks, `heads * 4` blocks. (Wrapped Prepare, GatedNorm and Route have lane P's `prompt_prepare`, `prompt_gated_norm` and
//! `prompt_route`, which measured equal or faster than this file's earlier versions of them, so those are not here.)
//!
//! The host launches this entry only for `rows_small::admitted(shape)`; the entry traps on any other shape.
use crate::prompt_ops::{Arguments,Op};

#[no_mangle]
pub extern "ptx-kernel" fn prompt_rows_small(p:Arguments) {
    // SAFETY: The host validates Shape::lengths and passes those exact extents, with same-context contiguous owners that live through stream
    // completion; inputs are immutable and the output is private, initialized (zeroed) earlier on the stream and overlaps no input. The entry traps unless
    // the shape is StepRows with dim 128, heads <= 64 and tokens <= 16, in a 32-thread block, and the block index is in range. A block owns head
    // `block / 4` and values `32 * (block % 4)..+32`: each lane reads its own value word of every state row and writes the same word, plus its own word of
    // each token's output, so every output word has one writer; a lane reads back only words its own thread wrote earlier in this launch. There is no shared
    // memory and no barrier. Every shuffle is reached by all 32 lanes of its warp (lanes never diverge before a shuffle).
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($sum:ident,$value:expr)=>{{let value:f32=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
        macro_rules! shuffle {($value:expr,$lane:expr)=>{{let result:f32;asm!("shfl.sync.idx.b32 {result}, {value}, {lane}, 31, 0xffffffff;",result=out(reg32)result,value=in(reg32)$value,lane=in(reg32)$lane,options(nomem,nostack));result}};}
        let s=p.shape;let (m,lane)=(s.tokens as usize,_thread_idx_x() as usize);
        if m==0 || m>16 {asm!("trap;");}
        const D:usize=128;
        let (h,d)=(s.heads as usize,s.dim as usize);let block=_block_idx_x() as usize;
        if s.op!=Op::StepRows || d!=D || h==0 || h>64 || _block_dim_x()!=32 || block>=h*(D/32) || p.units as usize!=h*D {asm!("trap;");}
        let (head,tile)=(block/(D/32),block%(D/32));let value=tile*32+lane;
        let packed=p.input[0] as *const f32;let incoming=(p.input[1] as *const f32).add(head*D*D);let decays=p.input[2] as *const f32;
        let state_out=(p.output as *mut f32).add(head*D*D);let outputs=(p.output as *mut f32).add(h*D*D);
        let mut token=0;
        while token<m {
            let at=token*h+head;let record=packed.add(at*(4*D+1));let factor_row=decays.add(at*D);
            // Token 0 reads the incoming state; later tokens read the state this lane wrote for the one before.
            let state:*const f32=if token==0 {incoming} else {state_out as *const f32};
            // Lane l holds keys l, l+32, l+64, l+96: decay, update key (record[D..]) and fold key (record[0..]).
            let mut decay=[0.0f32;4];let mut keys=[0.0f32;4];let mut queries=[0.0f32;4];
            let mut j=0;while j<4 {let key=32*j+lane;decay[j]=*factor_row.add(key);keys[j]=*record.add(D+key);queries[j]=*record.add(key);j+=1;}
            let beta=*record.add(4*D);
            let mut prediction=0.0f32;
            macro_rules! predict {($j:expr)=>{{
                let (factors,key_values)=(decay[$j],keys[$j]);let mut batch=0;
                while batch<4 {
                    let mut word=[0.0f32;8];let mut i=0;while i<8 {word[i]=*state.add((32*$j+8*batch+i)*D+value);i+=1;}
                    i=0;while i<8 {
                        let source=(8*batch+i) as u32;let factor=shuffle!(factors,source);let key_value=shuffle!(key_values,source);
                        let decayed=mul!(word[i],factor);add!(prediction,mul!(decayed,key_value));i+=1;
                    }
                    batch+=1;
                }
            }};}
            predict!(0);predict!(1);predict!(2);predict!(3);
            let delta=mul!(*record.add(2*D+value)-prediction,beta);
            let mut sum=0.0f32;
            macro_rules! update {($j:expr)=>{{
                let (factors,key_values,query_values)=(decay[$j],keys[$j],queries[$j]);let mut batch=0;
                while batch<4 {
                    let mut word=[0.0f32;8];let mut i=0;while i<8 {word[i]=*state.add((32*$j+8*batch+i)*D+value);i+=1;}
                    i=0;while i<8 {
                        let source=(8*batch+i) as u32;let factor=shuffle!(factors,source);let key_value=shuffle!(key_values,source);let query=shuffle!(query_values,source);
                        let mut updated=mul!(word[i],factor);add!(updated,mul!(key_value,delta));
                        *state_out.add((32*$j+8*batch+i)*D+value)=updated;add!(sum,mul!(updated,query));i+=1;
                    }
                    batch+=1;
                }
            }};}
            update!(0);update!(1);update!(2);update!(3);
            *outputs.add(token*h*D+head*D+value)=crate::small_math::round(sum);
            token+=1;
        }
    }
}
