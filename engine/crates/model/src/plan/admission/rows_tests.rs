use super::*;

fn fixture(storage:WeightStorage,dtype:Dtype)->(WeightView,BTreeMap<String,Vec<u8>>) {
    let ConfigPlan::Deferred(description)=ConfigPlan::from_config(include_bytes!("../../../tests/fixtures/glm/config.json")).unwrap() else {panic!()};
    let logical=LogicalWeight {name:"test.weight".into(),shape:vec![64,32],storage};
    let mut components=Vec::new();let mut payloads=BTreeMap::new();
    let mut insert=|name:&str,dtype:Dtype,shape:Vec<u64>,data:Vec<u8>| {
        components.push(ComponentView {name:name.into(),dtype,shape,start:0,bytes:data.len() as u64});payloads.insert(name.into(),data);
    };
    let columns=if matches!(storage,WeightStorage::Nvfp4 {..}) {16} else {32};
    let weight=(0..64*columns*dtype.bytes()).map(|index|(index%256) as u8).collect();
    insert("test.weight",dtype,vec![64,columns],weight);
    match storage {
        WeightStorage::Plain=>{},
        WeightStorage::Mxfp8=>insert("test.weight_scale",Dtype::U8,vec![64,1],(0..64).collect()),
        WeightStorage::Fp8Block {block}=>{
            let shape=vec![64u64.div_ceil(block[0]),1,32u64.div_ceil(block[1]),1];
            let scales=(0..shape[0]*shape[2]).flat_map(|index|((index+1) as f32).to_le_bytes()).collect();
            insert("test.weight_scale",Dtype::F32,shape,scales);
        },
        WeightStorage::Nvfp4 {encoding,..}=>{
            if encoding==ScaleEncoding::Csf32 {
                for (suffix,dtype,shape,data) in [
                    ("bases",Dtype::U8,vec![64],vec![32;64]),
                    ("codes",Dtype::U8,vec![64],vec![0x10;64]),
                    ("page_bases",Dtype::U32,vec![1],vec![0;4]),
                    ("offsets",Dtype::U16,vec![3],vec![0,0,0,0,1,0]),
                    ("exceptions",Dtype::U16,vec![1],vec![0x3f,0x7e])] {
                    insert(&format!("test.weight_scale.csf32_{suffix}"),dtype,shape,data);
                }
            } else {insert("test.weight_scale",Dtype::F8E4M3,vec![64,2],vec![0x38;128]);}
            insert("test.weight_scale_2",Dtype::F32,vec![],1.25f32.to_le_bytes().to_vec());
            insert("test.input_scale",Dtype::F32,vec![],0.75f32.to_le_bytes().to_vec());
        },
    }
    (WeightView {logical,binding:description.bind_tensor("lm_head.weight").unwrap(),components},payloads)
}

#[test]
fn row_upload_matches_full_planes_and_reads_only_kept_weight_rows() {
    for (storage,dtype) in [
        (WeightStorage::Plain,Dtype::Bf16),(WeightStorage::Plain,Dtype::F32),
        (WeightStorage::Mxfp8,Dtype::F8E4M3),(WeightStorage::Fp8Block {block:[16,7]},Dtype::F8E4M3),
        (WeightStorage::Nvfp4 {encoding:ScaleEncoding::Direct,activation_scale:true},Dtype::U8),
        (WeightStorage::Nvfp4 {encoding:ScaleEncoding::Csf32,activation_scale:true},Dtype::U8)] {
        let (view,payloads)=fixture(storage,dtype);
        let full=view.upload(|component|Ok(payloads[&component.name].clone())).unwrap();
        for selected in [0..32,32..64,0..64] {
            let mut weight_reads=Vec::new();
            let local=view.upload_rows(selected.clone(),|component,offset,count| {
                if component.name=="test.weight" {weight_reads.push((offset,count));}
                Ok(payloads[&component.name][offset as usize..(offset+count) as usize].to_vec())
            }).unwrap();
            let stride=full.weight.len()/64;
            assert_eq!(weight_reads,vec![(selected.start*stride as u64,(selected.end-selected.start)*stride as u64)]);
            assert_eq!(local.weight,full.weight[selected.start as usize*stride..selected.end as usize*stride]);
            let block=match storage {WeightStorage::Fp8Block {block}=>block[0],_=>1};
            let scale_stride=full.scales.len()/(64/block as usize);
            assert_eq!(local.scales,full.scales[selected.start as usize/block as usize*scale_stride..selected.end as usize/block as usize*scale_stride]);
            assert_eq!(local.shape,vec![selected.end-selected.start,32]);assert_eq!(local.dtype,full.dtype);
            assert_eq!(local.storage,full.storage);assert_eq!(local.global_scale,full.global_scale);assert_eq!(local.input_scale,full.input_scale);
        }
    }
}

#[test]
#[allow(clippy::reversed_empty_ranges)] // empty and reversed ranges are test inputs.
fn invalid_row_ranges_are_rejected_before_reads_and_short_reads_fail() {
    let (view,payloads)=fixture(WeightStorage::Fp8Block {block:[16,7]},Dtype::F8E4M3);
    for selected in [0..0,64..65,32..16,1..32,0..31] {
        assert!(view.upload_rows(selected,|_,_,_|panic!("invalid range caused I/O")).is_err());
    }
    assert!(view.upload_rows(0..32,|component,offset,count|Ok(payloads[&component.name][offset as usize..(offset+count-1) as usize].to_vec())).is_err());
    let mut nonmatrix=view.clone();nonmatrix.logical.shape=vec![64];
    assert!(nonmatrix.upload_rows(0..32,|_,_,_|panic!("nonmatrix caused I/O")).is_err());
}
