//! The four stable prompt-routing phases share one stream and fixed segment order.
use super::{Backend, DeviceVector};
use spark_gpu_probe::fp8_abi::GroupedRouteParams;

pub(super) fn launch(
    b: &Backend<'_>, mut grouped: GroupedRouteParams, count: usize,
    scratch: Option<&DeviceVector<'_>>,
) -> Result<(), String> {
    for (mut phase, blocks) in [(0u32, count.div_ceil(256)), (1, 512), (2, 1), (3, 512)] {
        unsafe {
            if let Some(scratch) = scratch {
                let mut scratch = scratch.device_address();
                b.launch_flash("flash_next_compact_route_prefill_packed_ids", blocks as u32, 256,
                    &mut [(&mut grouped as *mut GroupedRouteParams).cast(),
                        (&mut phase as *mut u32).cast(), (&mut scratch as *mut u64).cast()])?;
            } else {
                b.launch_flash("flash_next_compact_route_prefill", blocks as u32, 256,
                    &mut [(&mut grouped as *mut GroupedRouteParams).cast(),
                        (&mut phase as *mut u32).cast()])?;
            }
        }
    }
    Ok(())
}
