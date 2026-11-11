macro_rules! scatter_tail {($command:expr)=>{{
    let command=$command;let first=command.reserved[0];
    if first!=0 && core::arch::nvptx::_block_idx_y() as u32>=first {
        let window=&*command.payload.as_ptr().add(crate::bus_shadow_abi::OFFSET).cast::<crate::bus_shadow_abi::Window>();
        let blocks=core::arch::nvptx::_grid_dim_x() as u64;
        let threads=core::arch::nvptx::_block_dim_x() as u64;
        let block=(core::arch::nvptx::_block_idx_y() as u64-u64::from(first))*blocks+core::arch::nvptx::_block_idx_x() as u64;
        let stride=(core::arch::nvptx::_grid_dim_y() as u64-u64::from(first))*blocks*threads;
        let lines=crate::mhc_params::shadow_lines(window.bytes);
        let count=if window.ids==0 {1} else {u64::from(window.picks)};
        let mut line=block*threads+core::arch::nvptx::_thread_idx_x() as u64;
        while line<lines*count {
            let weight=if window.ids==0 {Some(window.weight)} else {
                let id=*(window.ids as *const f32).add((line/lines) as usize);
                crate::device_route_abi::selected(id,window.experts).map(|expert|(*(window.weight as *const crate::expert_ops::Arguments).add(expert as usize)).weight)
            };
            if let Some(weight)=weight {
                let address=weight+(line%lines)*crate::mhc_params::SHADOW_LINE;
                core::arch::asm!("prefetch.global.L2 [{address}];",address=in(reg64)address,options(nostack,readonly));
            }
            line+=stride;
        }
        return;
    }
}};}
pub(crate) use scatter_tail;
/// The same prefetch as a launch of its own (recorder kind ShadowSide): every block issues prefetches for the window in
/// the command's payload. It writes nothing.
macro_rules! side_window {($command:expr)=>{{
    let command=$command;
    let window=&*command.payload.as_ptr().add(crate::bus_shadow_abi::OFFSET).cast::<crate::bus_shadow_abi::Window>();
    let threads=core::arch::nvptx::_block_dim_x() as u64;
    let stride=core::arch::nvptx::_grid_dim_x() as u64*threads;
    let lines=crate::mhc_params::shadow_lines(window.bytes);
    let count=if window.ids==0 {1} else {u64::from(window.picks)};
    let mut line=core::arch::nvptx::_block_idx_x() as u64*threads+core::arch::nvptx::_thread_idx_x() as u64;
    while line<lines*count {
        let weight=if window.ids==0 {Some(window.weight)} else {
            let id=*(window.ids as *const f32).add((line/lines) as usize);
            crate::device_route_abi::selected(id,window.experts).map(|expert|(*(window.weight as *const crate::expert_ops::Arguments).add(expert as usize)).weight)
        };
        if let Some(weight)=weight {
            let address=weight+(line%lines)*crate::mhc_params::SHADOW_LINE;
            core::arch::asm!("prefetch.global.L2 [{address}];",address=in(reg64)address,options(nostack,readonly));
        }
        line+=stride;
    }
}};}
pub(crate) use side_window;
