use crate::resident_launch_abi::{Command,Kind};
#[no_mangle]
pub extern "ptx-kernel" fn resident_launch_pointwise(table:*const Command,command_index:u64) {
    // SAFETY: The recorder retains the aligned command table and every operand
    // until the ordered queue drains. Each descriptor covers checked word spans.
    // Its dependency proof maps every overlapping read or write to the same
    // thread index. No operation has a shifted in-place overlap. Thus sequential
    // operations within one thread preserve every dependency without a grid
    // barrier. The payload contains at most thirty-one initialized descriptors.
    unsafe {
        let command=&*table.add(command_index as usize);if command.kind!=Kind::Pointwise as u32 {core::arch::nvptx::trap();}
        let payload=command.payload.as_ptr();let count=payload.cast::<u64>().read() as usize;let index=crate::index() as u64;
        let mut operation=0usize;
        while operation<count {
            let row=payload.add(16+operation*32);let source=row.cast::<u64>().read();let target=row.add(8).cast::<u64>().read();
            let words=row.add(16).cast::<u32>().read() as u64;let round=row.add(20).cast::<u32>().read()!=0;let origin=row.add(24).cast::<u64>().read();
            if index>=origin && index-origin<words {
                let offset=(index-origin) as usize;let bits=if source==0 {0} else {(source as *const u32).add(offset).read()};
                let value=if round {crate::bf16::rounded(f32::from_bits(bits)).to_bits()} else {bits};
                (target as *mut u32).add(offset).write(value);
            }
            operation+=1;
        }
    }
}
