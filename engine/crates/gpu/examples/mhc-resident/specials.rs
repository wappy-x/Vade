use super::Result;
use std::path::Path;
#[cfg(target_os="linux")]
use super::{Shape,bytes,data,equal,write};

#[cfg(target_os="linux")]
pub(super) fn run(out:&Path)->Result<()> {
    use spark_gpu_probe::{driver::with_backend,layer_ptx,norm_boundary::{NormShape,WeightType}};
    use std::rc::Rc;
    with_backend(0,layer_ptx()?,usize::MAX,|backend| {
        let shape=Shape {rows:2,streams:4,hidden:129,iterations:20,rms_eps:1e-6,eps:1e-6};
        let wide=shape.wide();let hidden=shape.hidden as usize;let count=shape.coefficients();
        for precision in [WeightType::Bf16,WeightType::F32] {
            let norm=NormShape::new(hidden,1e-6,precision)?;
            let weights=vec![0.0;wide*count];let base=vec![0.0;count];let scale=[0.25,0.5,0.125];
            let weight=vec![if precision==WeightType::F32 {1.00013} else {1.0};hidden];
            let ordinary_norm=backend.resident_norm(norm,&weight)?;
            let ordinary=backend.resident_mhc(Shape {rows:1,..shape},&weights,&base,&scale)?;
            let resident=backend.resident_mhc_boundary(Shape {rows:1,..shape},norm,&weights,&base,&scale,&weight)?;
            for case in 0..8 {
                let mut input=data(2*wide,3);
                match case {
                    0=>{input[0]=-0.0;input[wide]=0.0;},
                    1=>input.fill(f32::from_bits(0x7f7f0000)),
                    2=>input[7]=f32::INFINITY,
                    3=>input[wide+7]=f32::NEG_INFINITY,
                    4=>input[7]=f32::from_bits(0x7fc12345),
                    5=>input[wide+7]=f32::from_bits(0xffc54321),
                    6=>input[7]=f32::from_bits(0x7f800001),
                    _=>{input.fill(f32::from_bits(0x7f7f0000));input[hidden..2*hidden].fill(-f32::from_bits(0x7f7f0000));},
                }
                let finite=input.iter().all(|value|value.is_finite());let mut expected=vec![Vec::new();5];
                for row in 0..2 {
                    let source=backend.upload_vector(&input[row*wide..(row+1)*wide])?;
                    let pass=ordinary.begin(backend.round_bf16_lanes(&source)?)?;
                    expected[0].extend(backend.read_vector_values(pass.normalized())?);
                    expected[1].extend(backend.read_vector_values(pass.coefficients())?);
                    expected[2].extend(backend.read_vector_values(pass.collapsed().vector())?);
                    expected[3].extend(backend.read_vector_values(ordinary_norm.apply(pass.collapsed())?.vector())?);
                }
                expected[4]=input.iter().copied().map(spark_gpu_probe::bf16::rounded).collect();
                let source=Rc::new(backend.upload_vector(&input)?);let pass=resident.begin_rows(source,2)?;
                let normalized=pass.normalized();let planes=[pass.mixing_normalized(),pass.coefficients(),pass.collapsed(),&*normalized,pass.residual()];
                for (plane,vector) in planes.into_iter().enumerate() {
                    let got=bytes(&backend.read_vector_values(vector)?);let want=bytes(&expected[plane]);
                    write(&out.join(format!("special-{case}-{precision:?}-{plane}.f32")),&got)?;
                    equal(&format!("special case={case} precision={precision:?} plane={plane}"),&got,&want)?;
                }
                if case==1 && (!finite || !expected[2].iter().any(|value|value.is_infinite()) || !expected[3].iter().any(|value|value.is_nan())) {
                    return Err("finite overflow fixture did not reach nonfinite normalization".into());
                }
                println!("MHC-BOUNDARY SPECIAL PASS case={case} precision={precision:?} rows=2 finite_input={finite}");
            }
        }
        Ok(())
    })?;
    write(&out.join("COMPLETE"),b"MHC-BOUNDARY SPECIAL PASS cases=16\n")
}
#[cfg(not(target_os="linux"))]
pub(super) fn run(_:&Path)->Result<()> {Err("boundary special probe requires Linux and CUDA".into())}
