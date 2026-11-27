//! Host checks for the arithmetic used by the device epilogue.
use engine_model::{flash::nvfp4::{bf16_to_f32, f32_to_bf16}, swiglu::SwiGlu};
use spark_gpu_probe::{next_math, swiglu};

#[test]
fn capped_kernel_math_matches_independent_host_reference() {
    for bound in [10.0, 3.5, 1.00390625] { for bf16 in [false, true] {
        let c = SwiGlu::new(Some(bound)).unwrap();
        for i in 0..32768u32 {
            let g = (i as f32 - 16384.0) / 256.0;
            let u = ((i.wrapping_mul(1664525).wrapping_add(1013904223) >> 8) as f32 - 8388608.0) / 131072.0;
            let got = swiglu::apply(g, u, bf16, c.limit().unwrap(), next_math::exp);
            assert_eq!(got.to_bits(), c.reference(g, u, bf16).to_bits(), "g={g} u={u} bound={bound} bf16={bf16}");
        }
    } }
}

#[test]
fn uncapped_kernel_math_keeps_qwen_bits() {
    let round = |x| bf16_to_f32(f32_to_bf16(x));
    for i in 0..65536u32 {
        let g = (i as f32 - 32768.0) / 512.0;
        let u = ((i.wrapping_mul(1103515245).wrapping_add(12345) >> 8) as f32 - 8388608.0) / 524288.0;
        for bf16 in [false, true] {
            // This is the parent layer_swiglu expression. The host exponential
            // is identical on each side; the device test supplies parent PTX.
            let old = if bf16 { let g = round(g);
                round(round(g / (1.0 + (-g).exp())) * round(u))
            } else { g / (1.0 + (-g).exp()) * u };
            assert_eq!(swiglu::apply(g, u, bf16, 0.0, f32::exp).to_bits(), old.to_bits());
        }
    }
    for bf16 in [false, true] { for g in [0.0, -0.0] { for u in [-0.0, 0.0, -2.0, 2.0] {
        assert_eq!(swiglu::apply(g, u, bf16, 0.0, f32::exp).to_bits(), (g*u).to_bits());
    } } }
}

#[test]
fn swiglu_abi_retains_parent_field_offsets() {
    use spark_gpu_probe::kernel_params::SwiGlu;
    assert_eq!(std::mem::size_of::<SwiGlu>(), 32);
    assert_eq!(std::mem::offset_of!(SwiGlu, gate), 0);
    assert_eq!(std::mem::offset_of!(SwiGlu, up), 8);
    assert_eq!(std::mem::offset_of!(SwiGlu, width), 16);
    assert_eq!(std::mem::offset_of!(SwiGlu, bf16), 20);
    assert_eq!(std::mem::offset_of!(SwiGlu, limit), 24);
}
