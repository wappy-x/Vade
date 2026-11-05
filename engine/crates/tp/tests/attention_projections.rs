//! Complete attention matrix rows. These synthetic headers do not prove a decoder.
use engine_model::{plan::{layer::{LayerPlan,Topology,Kind},state::StatePlan},safetensors::Dtype};
use spark_tp::layer_driver::LayerBoundary;
use spark_gpu_probe::{dynamic_mxfp8::Activation,mxfp8_projection::Shape,norm_boundary::WeightType};
#[path="support/attention_checkpoint.rs"] mod checkpoint;
use checkpoint::Checkpoint;
#[test]
fn complete_attention_rows_cover_norm_low_rank_gate_and_value_producers() {
    for norm in [Dtype::Bf16,Dtype::F32] {
        let c=Checkpoint::new(norm);let d=c.admission.description();
        for layer in 0..d.layers.len() {
            let full_plan=LayerPlan::new(d,layer,1,Topology {rank:0,parts:1}).unwrap();let mut kinds=Vec::new();
            for (step,op) in full_plan.steps.iter().enumerate() {
                if op.kind==Kind::Combine {break;}
                if op.kind!=Kind::Project || !c.operands.contains_key(&op.weights[0].name) {continue;}
                let name=&op.weights[0].name;let (w,s)=&c.operands[name];let cols=op.weights[0].shape[1] as usize;
                let raw=(0..cols).map(|i|(i%19) as f32/8.0-1.0).collect::<Vec<_>>();let q=Activation::from_f32(&raw).unwrap();
                let expected=Shape::new(op.weights[0].shape[0] as usize,cols,WeightType::F32).unwrap().reference(w,s,q.decoded()).unwrap();
                for parts in [1,2,4] {
                    let mut accumulated=Vec::new();let mut output=Vec::new();let mut origin=0;
                    for rank in 0..parts {
                        let t=Topology {rank,parts};let state=StatePlan::new(d,t,1,isize::MAX as usize).unwrap();
                        let b=LayerBoundary::new(d,c.admission.headers(),&state,layer,1,t,4096).unwrap();
                        let p=b.prepare_mxfp8_projection_checkpoint(step,&c.admission).unwrap();
                        let replicated=op.output_splits[0]==engine_model::plan::layer::Split::Replicated;
                        assert_eq!(p.row_origin(),if replicated {0} else {origin});origin+=p.shape().rows() as u64;
                        let (activation,acc,out)=p.dynamic_reference(&raw).unwrap();
                        assert_eq!(activation.values(),q.values());assert_eq!(activation.scales(),q.scales());
                        if replicated {assert_eq!(acc,expected.0);assert_eq!(out,expected.1);}
                        accumulated.extend(acc);output.extend(out);assert!(b.require_executable().is_err());
                    }
                    // Replicated matrices are checked once. Split matrices join by row.
                    if op.output_splits[0]==engine_model::plan::layer::Split::Replicated {
                        assert_eq!(&accumulated[..expected.0.len()],expected.0);assert_eq!(&output[..expected.1.len()],expected.1);
                    } else {assert_eq!(accumulated,expected.0);assert_eq!(output,expected.1);}
                }
                let producer=full_plan.steps[..step].iter().find(|s|s.outputs.iter().any(|b|op.inputs.contains(&b.name))).unwrap();
                kinds.push(producer.kind);
            }
            assert!(kinds.contains(&Kind::Norm));
            assert!(kinds.contains(&if layer==0 {Kind::GatedNorm} else {Kind::Values}));
        }
    }
}
