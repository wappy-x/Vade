//! Config-to-operation tests. The saved fixture supplies all checkpoint values.
use engine_model::{host_ops::router::{Router, RouterKind}, plan::ConfigPlan, shapes::Ffn};
const CONFIG: &[u8] = include_bytes!("fixtures/glm/config.json");

#[test]
fn saved_description_and_field_reader_select_the_same_math() {
    let ConfigPlan::Deferred(d) = ConfigPlan::from_config(CONFIG).unwrap() else { panic!() };
    let Ffn::Experts(e) = d.layers.iter().find_map(|l| matches!(l.ffn,Ffn::Experts(_)).then_some(l.ffn)).unwrap() else { panic!() };
    let from_plan = Router::from_description(&d.routing,e).unwrap();
    assert_eq!(from_plan,Router::from_config(CONFIG).unwrap());
    assert_eq!(from_plan.kind(),RouterKind::CorrectedSigmoid);
    let (ids,weights) = from_plan.route(&vec![0.0;from_plan.experts()],&vec![0.0;from_plan.experts()]).unwrap();
    assert_eq!(ids,(0..from_plan.selected()).collect::<Vec<_>>());
    assert_eq!(weights.iter().map(|w| w.to_bits()).collect::<Vec<_>>(),[0x3ea00000;8]);
    println!("saved config route: experts={} selected={} groups={} keep={} scale={} ids={ids:?} weight_bits={:x?}",
        from_plan.experts(),from_plan.selected(),from_plan.groups(),from_plan.selected_groups(),from_plan.scale(),
        weights.iter().map(|w| w.to_bits()).collect::<Vec<_>>());
}

#[test]
fn plan_refuses_unexecutable_router_geometry_or_f32_scale() {
    let source = std::str::from_utf8(CONFIG).unwrap();
    for (old,new) in [("\"n_group\": 1","\"n_group\": 288"),
        ("\"n_group\": 1","\"n_group\": 144"),
        ("\"routed_scaling_factor\": 2.5","\"routed_scaling_factor\": 1e40"),
        ("\"routed_scaling_factor\": 2.5","\"routed_scaling_factor\": 1e-60")] {
        assert!(source.contains(old));
        assert!(ConfigPlan::from_config(source.replace(old,new).as_bytes()).is_err(),"{new}");
    }
}

#[test]
fn softmax_reuses_the_existing_reference_bits() {
    let op = Router::softmax(8,3).unwrap();
    let logits = [0.0,-2.0,2.0,2.0,1.0,0.0,-0.0,-1.0];
    let (ids,weights) = op.route(&logits,&[]).unwrap();
    let old = engine_model::flash::refcpu::route(&logits,3);
    assert_eq!(op.kind(),RouterKind::SoftmaxTopK);
    assert_eq!(ids,old.0);
    assert_eq!(weights.iter().map(|w| w.to_bits()).collect::<Vec<_>>(),old.1.iter().map(|w| w.to_bits()).collect::<Vec<_>>());
    assert!(op.route(&logits,&[0.0;8]).is_err());
    for (e,k) in [(0,1),(1025,1),(4,0),(4,5)] { assert!(Router::softmax(e,k).is_err()); }
}

#[test]
fn source_order_weight_sum_is_stable_across_repeated_rows() {
    let source = br#"{"n_routed_experts":6,"num_experts_per_tok":4,"n_group":3,"topk_group":2,
        "norm_topk_prob":true,"routed_scaling_factor":2.5,"scoring_func":"sigmoid","topk_method":"noaux_tc"}"#;
    let op = Router::from_config(source).unwrap();
    let bias = [1.5,1.25,0.5,0.25,-1.0,-1.0];
    let rows = [[0.0;6],[20.0,-20.0,0.0,0.0,20.0,20.0],[-1000.0;6]];
    let expected: Vec<_> = rows.iter().map(|r| op.route(r,&bias).unwrap()).collect();
    for chunk in 1..=rows.len() {
        for _ in 0..3 {
            let actual: Vec<_> = rows.chunks(chunk).flat_map(|part| part.iter().map(|r| op.route(r,&bias).unwrap())).collect();
            for (a,e) in actual.iter().zip(&expected) {
                assert_eq!(a.0,e.0);
                assert_eq!(a.1.iter().map(|w| w.to_bits()).collect::<Vec<_>>(),e.1.iter().map(|w| w.to_bits()).collect::<Vec<_>>());
            }
        }
    }
}
