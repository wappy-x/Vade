//! Load-time product of a BF16 vocabulary head and the hidden projection, including its bias column.
//! Each lane accumulates one output in ascending hidden index, in FP64, then converts once to FP32.
/// # Safety
/// Launch from Backend::fold_markov with its validated shape and live disjoint output.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn inkling_markov_fold(args:[u64;6]) {
    let [head,hidden_out,output,rows,hidden,columns]=args;
    let mut sum=0.0f64;
    // SAFETY: Backend::fold_markov checks the matrix extents and uses separate contiguous allocations
    // for the two inputs and output, kept live until the result read. The grid has 128 lanes per
    // block. Each in-range lane reads one head row and one projection column and owns output[index].
    unsafe {
        let index=core::arch::nvptx::_block_idx_x() as u64*128+core::arch::nvptx::_thread_idx_x() as u64;
        if index>=rows*columns {return;}
        let row=index/columns;let column=index%columns;
        for k in 0..hidden {
            let bits=*(head as *const u16).add((row*hidden+k) as usize);
            let right=*(hidden_out as *const f32).add((k*columns+column) as usize);
            sum+=(f32::from_bits(u32::from(bits)<<16) as f64)*(right as f64);
        }
        *(output as *mut f32).add(index as usize)=sum as f32;
    }
}
