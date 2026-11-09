use engine_model::flash::{nvfp4, refcpu::PleHash};
use spark_gpu_probe::frontend_math::{dequant, rows};
#[test]
fn frontend_matches_cpu_bits() {
    let mut p = [0i64; 35];
    p[..3].copy_from_slice(&[i64::MAX, i64::MIN + 1, -712367821]);
    for h in 0..16 {
        p[3 + h] = 2500011 + h as i64;
        p[19 + h] = h as i64 * 2500030;
    }
    let mut reference = PleHash::new(p[..3].to_vec(), p[3..19].to_vec(), p[19..].to_vec(), 248044);
    for _ in 0..2 {
        reference.reset();
        let mut history = [248044; 2];
        for i in 0..10000 {
            let t = if i % 17 < 2 {
                248044
            } else {
                (i * 7919 % 248320) as u32
            };
            assert_eq!(rows(t, &mut history, &p).as_slice(), reference.advance(t));
        }
    }
    for width in [1, 8] {
        for _row in 0..width {
            let mut host =
                PleHash::new(p[..3].to_vec(), p[3..19].to_vec(), p[19..].to_vec(), 248044);
            let mut device =
                PleHash::new(p[..3].to_vec(), p[3..19].to_vec(), p[19..].to_vec(), 248044);
            for t in [11751, 13, 561, 6511, 314, 9564, 369, 19241, 248044, 13] {
                let mut previous = device.advance_history(t);
                assert_eq!(rows(t, &mut previous, &p).as_slice(), host.advance(t));
            }
        }
    }
    for scale in 0..127 {
        for byte in 0..=255 {
            for global in [0.000173, 1.0, std::f32::consts::PI] {
                let mut expected = [0.0; 16];
                nvfp4::dequantize(&[byte; 8], &[scale], global, &mut expected).unwrap();
                for i in 0..16 {
                    assert_eq!(
                        dequant(byte, i % 2 != 0, scale, global).to_bits(),
                        expected[i].to_bits()
                    );
                }
            }
        }
    }
}
