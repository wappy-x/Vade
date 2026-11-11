use spark_gpu_probe::norm_boundary::{NormShape,WeightType};
#[test]
#[allow(clippy::excessive_precision)] // literal constants stay as written.
fn norm_operand_precision_and_extent_are_checked_before_upload() {
    let bf=NormShape::new(2,1e-5,WeightType::Bf16).unwrap();bf.check_weights(&[1.0,-0.0]).unwrap();
    for bad in [&[][..],&[1.0],&[1.0,1.0,1.0],&[1.0,1.00390625],&[f32::NAN,1.0],&[f32::INFINITY,1.0]] {assert!(bf.check_weights(bad).is_err());}
    let fp=NormShape::new(2,1e-5,WeightType::F32).unwrap();fp.check_weights(&[1.00390625,-0.0]).unwrap();
    assert!(fp.check_weights(&[f32::NAN,0.0]).is_err());
    for (width,epsilon) in [(0,1.0),(usize::MAX,1.0),(1,0.0),(1,-1.0),(1,f32::NAN),(1,f32::INFINITY)] {
        assert!(NormShape::new(width,epsilon,WeightType::Bf16).is_err());
    }
}
#[test]
#[allow(clippy::excessive_precision)] // literal constants stay as written.
fn norm_rounds_before_weight_and_keeps_an_fp32_parameter() {
    let bf=NormShape::new(1,1e-5,WeightType::Bf16).unwrap();
    // A final-only cast returns 1.5078125 for this tie.
    assert_eq!(bf.weight(1.00390625,1.5).to_bits(),1.5f32.to_bits());
    let fp=NormShape::new(1,1e-5,WeightType::F32).unwrap();
    assert_eq!(fp.weight(1.00390625,1.0000009536743164).to_bits(),1.0000009536743164f32.to_bits());
    assert_eq!(fp.weight(-0.0,1.0).to_bits(),(-0.0f32).to_bits());
}
