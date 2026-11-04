use engine_model::plan::{ConfigPlan, ModelPlan, description::TokenMixer,
    tensor_split::{Ownership, Owner, ProjectionRows}, storage::{WeightStorage, ScaleEncoding}};
use engine_model::shapes::{Ffn, MAX_SPARKS};

#[test]
fn intervals_cover_complete_units_in_rank_order() {
    for parts in 1..=MAX_SPARKS {
        let n = u64::from(parts)*6;
        for (shape,axis,unit) in [(vec![2,n,3],1,2),(vec![n,7],0,3),(vec![4,n],1,1)] {
            let p = Ownership::equal(&shape,axis,unit,parts).unwrap();
            assert_eq!(p.elements(),shape.iter().product());
            assert_eq!(Ownership::checked(&shape,axis,unit,p.owners()).unwrap(),p);
            for rank in 0..parts { assert_eq!(p.owner(rank).unwrap().interval,u64::from(rank)*6..(u64::from(rank)+1)*6); }
            assert!(p.owner(parts).is_err());
        }
    }
}

#[test]
fn bad_bounds_gaps_overlaps_units_and_rank_order_refuse() {
    let good = [Owner {rank:0,interval:0..4},Owner {rank:1,interval:4..8}];
    for (a,b) in [(0..3,4..8),(0..5,4..8),(0..4,4..7),(0..4,4..9),(0..0,0..8),(0..4,8..4),(1..4,4..8),(0..2,2..8)] {
        assert!(Ownership::checked(&[8],0,1,&[Owner {rank:0,interval:a},Owner {rank:1,interval:b}]).is_err());
    }
    let mut wrong = good.clone(); wrong.swap(0,1); assert!(Ownership::checked(&[8],0,1,&wrong).is_err());
    wrong = good.clone(); wrong[1].rank=0; assert!(Ownership::checked(&[8],0,1,&wrong).is_err());
    for (shape,axis,unit,parts) in [(vec![],0,1,1),(vec![0],0,1,1),(vec![8],1,1,2),(vec![8],0,0,2),
        (vec![8],0,3,2),(vec![8],0,1,0),(vec![8],0,1,MAX_SPARKS+1),(vec![8],0,u64::MAX,2),
        (vec![u64::MAX,2],0,1,1),(vec![1;17],0,1,1)] {
        assert!(Ownership::equal(&shape,axis,unit,parts).is_err(),"{shape:?} {axis} {unit} {parts}");
    }
    assert!(Ownership::checked(&[8],0,1,&[]).is_err());
    assert!(Ownership::checked(&[8],0,3,&good).is_err());
}

#[test]
fn projection_keeps_full_k_and_original_fp8_scale_grid() {
    let p = Ownership::equal(&[300,257],0,1,2).unwrap();
    let s = ProjectionRows::new(&p,1,WeightStorage::Fp8Block {block:[128,128]}).unwrap();
    assert_eq!(s.rows(),150..300); assert_eq!(s.shape(),[300,257]);
    let grid = s.scale().unwrap(); assert_eq!((grid.block,grid.shape),([128,128],[3,3]));
    assert_eq!(s.scale_index(0,0).unwrap(),3);
    assert_eq!(s.scale_index(105,256).unwrap(),5);
    assert_eq!(s.scale_index(106,128).unwrap(),7);
    assert!(s.scale_index(150,0).is_err()); assert!(s.scale_index(0,257).is_err());
    assert!(s.direct_scale_rows().is_err());
    assert_eq!(s.input_features(),257);
    let aligned = Ownership::equal(&[512,257],0,1,2).unwrap();
    let s = ProjectionRows::new(&aligned,1,WeightStorage::Fp8Block {block:[128,128]}).unwrap();
    assert_eq!(s.direct_scale_rows().unwrap(),2..4);
    assert!(ProjectionRows::new(&Ownership::equal(&[300,258],1,1,2).unwrap(),0,WeightStorage::Plain).is_err());
    assert!(ProjectionRows::new(&Ownership::equal(&[300],0,1,2).unwrap(),0,WeightStorage::Plain).is_err());
    assert!(ProjectionRows::new(&p,0,WeightStorage::Fp8Block {block:[0,128]}).is_err());
}

#[test]
fn quantized_slices_do_not_rebase_or_slice_compressed_pages() {
    let p = Ownership::equal(&[128,64],0,1,4).unwrap();
    for rank in 0..4 {
        let row = u64::from(rank)*32;
        for storage in [WeightStorage::Mxfp8,WeightStorage::Nvfp4 {encoding:ScaleEncoding::Direct,activation_scale:true},
            WeightStorage::Nvfp4 {encoding:ScaleEncoding::Csf,activation_scale:false},
            WeightStorage::Nvfp4 {encoding:ScaleEncoding::Csf32,activation_scale:true}] {
            let s = ProjectionRows::new(&p,rank,storage).unwrap();
            let columns = if storage == WeightStorage::Mxfp8 {2} else {4};
            assert_eq!(s.scale_index(0,0).unwrap(),row*columns);
            assert_eq!(s.scale_index(31,63).unwrap(),(row+32)*columns-1);
            if s.scale().unwrap().encoding == ScaleEncoding::Direct { assert_eq!(s.direct_scale_rows().unwrap(),row..row+32); }
            else { assert!(s.direct_scale_rows().is_err()); }
        }
    }
    let plain = ProjectionRows::new(&p,0,WeightStorage::Plain).unwrap(); assert!(plain.scale().is_none()); assert!(plain.scale_index(0,0).is_err());
    assert!(ProjectionRows::new(&Ownership::equal(&[128,65],0,1,4).unwrap(),0,
        WeightStorage::Nvfp4 {encoding:ScaleEncoding::Direct,activation_scale:false}).is_err());
}

#[test]
fn saved_description_checks_every_count_without_opening_admission() {
    const CONFIG: &[u8] = include_bytes!("fixtures/glm/config.json");
    let ConfigPlan::Deferred(d) = ConfigPlan::from_config(CONFIG).unwrap() else { panic!() };
    let mut accepted = Vec::new();
    for parts in 1..=MAX_SPARKS {
        let result = (|| -> engine_model::Result<()> {
            for layer in &d.layers {
                let (heads,unit) = match layer.mixer { TokenMixer::Kda(k) => (k.heads,k.head_dim),
                    TokenMixer::LatentAttention(a) => (a.heads,a.value_dim) };
                Ownership::equal(&[heads*unit],0,unit,parts)?;
                if let TokenMixer::LatentAttention(a) = layer.mixer {
                    Ownership::equal(&[a.heads*(a.key_dim+a.value_dim),a.kv_rank],0,a.key_dim+a.value_dim,parts)?;
                }
                let (width,shared) = match layer.ffn { Ffn::Dense {width} => (width,0),Ffn::Experts(e) => (e.width,e.shared_width) };
                for n in [width,d.hidden].into_iter().chain((shared>0).then_some(shared)) { Ownership::equal(&[n],0,1,parts)?; }
            }
            Ok(())
        })();
        if result.is_ok() { accepted.push(parts); }
        println!("TENSOR_TOPOLOGY parts={parts} logical_intervals={}",if result.is_ok() {"PASS"} else {"REFUSED"});
    }
    assert_eq!(accepted,[1,2,4,8]);
    assert!(ConfigPlan::from_config(CONFIG).unwrap().into_model_plan().is_err()); assert!(ModelPlan::from_config(CONFIG).is_err());
}
