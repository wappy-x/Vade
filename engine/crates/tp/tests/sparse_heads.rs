//! Host bits and independent ledger oracles. These tests do not establish device exactness.
use engine_model::plan::{ConfigPlan,TextDescription,description::TokenMixer,layer::Topology};
use spark_gpu_probe::{causal_ops::{self,Op,Shape},prompt_ops::{self,Op as Row,Shape as RowShape}};
use spark_tp::{admission_plan::SessionChargePlan,sparse_heads::{self,Layout}};

fn description()->TextDescription {
    let ConfigPlan::Deferred(description)=ConfigPlan::from_config(include_bytes!("../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!("expected text description")};description
}
fn layout(heads:u64,key:u64,value:u64,rank:u32,parts:u32)->Layout {Layout::new(heads,key,value,Topology {rank,parts}).unwrap()}
fn bits(values:&[f32])->Vec<u32> {values.iter().map(|v|v.to_bits()).collect()}
fn sample(index:usize)->f32 {if index.is_multiple_of(19) {-0.0} else {(index as i32%31-15) as f32/32.0}}
fn shape(op:Op,heads:usize,dim:usize,rows:usize,cols:usize)->Shape {
    Shape {op,heads:heads as u32,dim:dim as u32,taps:0,rows:rows as u32,cols:cols as u32,epsilon:0.0,bound:0.0}
}

#[test]
fn checkpoint_owns_complete_heads_and_predictor_uses_the_same_map() {
    let description=description();
    for parts in [1,2,4,8] {
        for rank in 0..parts {
            let maps=sparse_heads::layouts(&description,Topology {rank,parts}).unwrap();assert_eq!(maps.len(),11);
            for (layer,owned) in maps {
                let TokenMixer::LatentAttention(spec)=description.layers[layer].mixer else {panic!("expected latent layer")};
                assert_eq!((spec.heads,spec.key_dim,spec.value_dim),(64,256,256));
                assert_eq!(owned.heads(),rank as usize*64/parts as usize..(rank+1) as usize*64/parts as usize);
                assert_eq!(owned.projection_rows(),rank as usize*32768/parts as usize..(rank+1) as usize*32768/parts as usize);
                assert_eq!(owned.query_rows(),rank as usize*16384/parts as usize..(rank+1) as usize*16384/parts as usize);
                assert_eq!(owned.attended_rows(),owned.query_rows());
                assert_eq!(owned.row_words()*parts as usize,32768);
            }
        }
    }
    let predictor=description.with_draft_layer().unwrap().unwrap();
    assert_eq!(sparse_heads::layouts(&predictor,Topology {rank:1,parts:2}).unwrap().len(),12);
    for rank in 0..3 {assert_eq!(layout(6,3,2,rank,3).heads(),rank as usize*2..rank as usize*2+2);}
}

#[test]
fn cache_handoff_preserves_raw_words_and_prefix_suffix_boundaries() {
    for (heads,key,value,parts) in [(64,256,256,2),(12,3,5,3),(8,4,8,4),(1,1,1,1)] {
        let width=(heads*(key+value)) as usize;
        let source=(0..7*width).map(|index|[0,0x80000000,0x7fc01234,0x7f800000,0xff800000,index as u32][index%6]).collect::<Vec<_>>();
        for rank in 0..parts {
            let owned=layout(heads,key,value,rank,parts);let local=owned.local_expanded(&source).unwrap();
            let prefix=owned.local_expanded(&source[..3*width]).unwrap();let suffix=owned.local_expanded(&source[3*width..]).unwrap();
            assert_eq!([prefix,suffix].concat(),local);
            for row in 0..7 {for head in owned.heads() {
                let local_head=head-owned.heads().start;let start=row*owned.row_words()+local_head*(key+value) as usize;
                let global=row*width+head*(key+value) as usize;
                assert_eq!(&local[start..start+(key+value) as usize],&source[global..global+(key+value) as usize]);
            }}
        }
    }
    let owned=layout(6,3,5,1,3);let source=(0..1161*48).map(|index|index as u32).collect::<Vec<_>>();
    let all=owned.local_expanded(&source).unwrap();
    for held in [0,1040,1151,1152,1153,1161] {
        let prefix=owned.local_expanded(&source[..held*48]).unwrap();let suffix=owned.local_expanded(&source[held*48..]).unwrap();
        assert_eq!([prefix,suffix].concat(),all,"held={held}");
    }
}

#[test]
fn projection_rows_keep_every_serial_product_and_bf16_bit() {
    for (heads,key,value,parts) in [(64,8,4,2),(6,3,2,3)] {
        let latent=17;let width=(heads*(key+value)) as usize;
        let weight=(0..width*latent).map(|index|sample(index+7)).collect::<Vec<_>>();
        let source=(0..4*latent).map(sample).collect::<Vec<_>>();
        for input in source.chunks_exact(latent) {
            let full=causal_ops::reference(shape(Op::Project,0,0,width,latent),[input,&[],&[],&[]],&weight).unwrap();
            for rank in 0..parts {
                let owned=layout(heads,key,value,rank,parts);let rows=owned.projection_rows();
                let local=causal_ops::reference(shape(Op::Project,0,0,rows.len(),latent),[input,&[],&[],&[]],&weight[rows.start*latent..rows.end*latent]).unwrap();
                assert_eq!(bits(&local),bits(&owned.local_expanded(&full).unwrap()));
                assert_eq!(bits(&local.iter().map(|&v|spark_gpu_probe::bf16::rounded(v)).collect::<Vec<_>>()),bits(&full[rows].iter().map(|&v|spark_gpu_probe::bf16::rounded(v)).collect::<Vec<_>>()));
            }
        }
    }
}

fn attention(heads:u64,key:u64,value:u64,parts:u32,tokens:usize,total:usize,slots:usize) {
    let width=(heads*(key+value)) as usize;let heads=heads as usize;
    let expanded=(0..total*width).map(|index|sample(index+11)).collect::<Vec<_>>();
    let query=(0..tokens*heads*key as usize).map(sample).collect::<Vec<_>>();
    let mut selected=vec![-1.0;tokens*(slots+1)];
    for token in 0..tokens {
        let count=if tokens==1 {slots.min(total)} else {token*slots/(tokens-1)};
        for slot in 0..count {selected[token*(slots+1)+slot]=((total-1-slot+token)%total) as f32;}
        if tokens>1 && count>1 {selected[token*(slots+1)+count-1]=selected[token*(slots+1)];}
        selected[token*(slots+1)+slots]=count as f32;
    }
    let base=RowShape {heads:heads as u32,dim:key as u32,taps:value as u32,rows:total as u32,cols:slots as u32,slots:slots as u32,..RowShape::new(Row::ScoresRows,tokens).unwrap()};
    let scores=prompt_ops::reference(base,[&query,&expanded,&selected,&[],&[]],&[]).unwrap();
    let probabilities=prompt_ops::reference(RowShape {op:Row::SoftmaxRows,..base},[&scores,&selected,&[],&[],&[]],&[]).unwrap();
    let full=prompt_ops::reference(RowShape {op:Row::ValuesRows,..base},[&probabilities,&expanded,&selected,&[],&[]],&[]).unwrap();
    for token in 0..tokens {
        let count=selected[token*(slots+1)+slots] as usize;let mut keys=vec![0.0;slots*heads*key as usize];let mut values=vec![0.0;slots*heads*value as usize];
        for slot in 0..count {let row=selected[token*(slots+1)+slot] as usize;for head in 0..heads {
            let start=row*width+head*(key+value) as usize;
            keys[(slot*heads+head)*key as usize..(slot*heads+head+1)*key as usize].copy_from_slice(&expanded[start..start+key as usize]);
            values[(slot*heads+head)*value as usize..(slot*heads+head+1)*value as usize].copy_from_slice(&expanded[start+key as usize..start+(key+value) as usize]);
        }}
        let live=[count as f32];let scalar=shape(Op::Scores,heads,key as usize,slots,0);
        let query_row=&query[token*heads*key as usize..(token+1)*heads*key as usize];
        let scalar_scores=causal_ops::reference(scalar,[query_row,&keys,&live,&[]],&[]).unwrap();
        assert_eq!(bits(&scalar_scores),bits(&scores[token*heads*slots..(token+1)*heads*slots]));
        let scalar_probabilities=causal_ops::reference(Shape {op:Op::Softmax,..scalar},[&scalar_scores,&live,&[],&[]],&[]).unwrap();
        assert_eq!(bits(&scalar_probabilities),bits(&probabilities[token*heads*slots..(token+1)*heads*slots]));
        let scalar_values=causal_ops::reference(Shape {op:Op::Values,dim:value as u32,..scalar},[&scalar_probabilities,&values,&live,&[]],&[]).unwrap();
        assert_eq!(bits(&scalar_values),bits(&full[token*heads*value as usize..(token+1)*heads*value as usize]));
    }
    let mut packets=Vec::new();
    for rank in 0..parts {
        let owned=layout(heads as u64,key,value,rank,parts);let local=RowShape {heads:owned.local_heads() as u32,..base};
        let local_query=owned.local_queries(&query).unwrap();let local_expanded=owned.local_expanded(&expanded).unwrap();
        let local_scores=prompt_ops::reference(local,[&local_query,&local_expanded,&selected,&[],&[]],&[]).unwrap();
        let local_probabilities=prompt_ops::reference(RowShape {op:Row::SoftmaxRows,..local},[&local_scores,&selected,&[],&[],&[]],&[]).unwrap();
        for token in 0..tokens {
            let expected=(token*heads+owned.heads().start)*slots..(token*heads+owned.heads().end)*slots;
            let actual=token*owned.local_heads()*slots..(token+1)*owned.local_heads()*slots;
            assert_eq!(bits(&local_scores[actual.clone()]),bits(&scores[expected.clone()]));
            assert_eq!(bits(&local_probabilities[actual]),bits(&probabilities[expected]));
        }
        let values=prompt_ops::reference(RowShape {op:Row::ValuesRows,..local},[&local_probabilities,&local_expanded,&selected,&[],&[]],&[]).unwrap();
        packets.push((owned,values));
    }
    let refs=packets.iter().map(|(owned,values)|(*owned,values.as_slice())).collect::<Vec<_>>();
    assert_eq!(bits(&sparse_heads::gather_attended(&refs,tokens).unwrap()),bits(&full));
}

#[test]
fn split_attention_matches_replicated_scores_softmax_and_values_bits() {
    for (heads,key,value,parts,tokens,total,slots) in [
        (64,256,256,2,1,16,16),(64,256,256,2,1,1041,1041),(64,256,256,2,1,2051,2051),
        (64,8,4,2,4,19,11),(12,3,5,3,4,23,7),(8,4,8,4,3,31,19),
    ] {attention(heads,key,value,parts,tokens,total,slots);}
}

#[test]
fn gather_preserves_special_bits_in_token_then_global_head_order() {
    let special:[u32;6]=[0,0x80000000,0x7fc01234,0x7f800000,0xff800000,0x3f800000];
    let packets=(0..3).map(|rank|(layout(6,3,2,rank,3),(0..8).map(|index|special[(index+rank as usize)%6]).collect::<Vec<_>>())).collect::<Vec<_>>();
    let refs=packets.iter().map(|(owned,words)|(*owned,words.as_slice())).collect::<Vec<_>>();
    let got=sparse_heads::gather_attended(&refs,2).unwrap();
    let mut expected=Vec::new();for token in 0..2 {for (_,words) in &packets {expected.extend_from_slice(&words[token*4..token*4+4]);}}
    assert_eq!(got,expected);
}

fn state(tokens:usize)->usize {152_633_344+44*768*tokens+44*128*(tokens/4)}
fn charges(prompt:usize,tokens:usize,expanded_row:usize)->(usize,usize) {
    let held=expanded_row*prompt.min(1152);
    let early=2*state(prompt.clamp(1,1152))+2*held;let late=2*state(tokens)+held;
    let base=early.max(late)+2_228_224*prompt.clamp(1,256)+2_478_080;
    let legacy=2*state(tokens)+2*held+2_478_080;
    (base+8_650_752,legacy.max(base)+8_650_752)
}

#[test]
fn session_ledger_only_reduces_expanded_planes_and_preserves_other_reservations() {
    let description=description();
    for parts in [1,2,4,8] {for rank in 0..parts {
        let ledger=SessionChargePlan::for_sparse_heads(&description,Topology {rank,parts},32768).unwrap();
        assert_eq!(ledger.expanded_bytes(1),720_896/parts as usize);
        assert_eq!(ledger.append_reservation_bytes(),8_650_752);
        for (prompt,tokens) in [(0,8192),(15,8192),(1040,8192),(1152,8192),(1153,8192),(4158,8192),(8192,8192),(16384,16384),(32768,32768)] {
            let (session,request)=charges(prompt,tokens,720_896/parts as usize);
            assert_eq!(ledger.state_bytes(tokens).unwrap(),state(tokens));
            assert_eq!(ledger.session(prompt,tokens).unwrap(),session);
            assert_eq!(ledger.request(prompt,tokens).unwrap(),request);
        }
        assert!(ledger.session(8193,8192).is_err());assert!(ledger.request(1,0).is_err());assert!(ledger.session(1,32769).is_err());
    }}
    let ledger=SessionChargePlan::for_sparse_heads(&description,Topology {rank:0,parts:2},8192).unwrap();
    assert_eq!(ledger.expanded_bytes(1152),415_236_096);
    println!("HEAD_SPLIT_LEDGER row=360448 held1152=415236096 session4158_8192={} request4158_8192={}",ledger.session(4158,8192).unwrap(),ledger.request(4158,8192).unwrap());
    let predictor=description.with_draft_layer().unwrap().unwrap();
    assert_eq!(SessionChargePlan::for_sparse_heads(&predictor,Topology {rank:1,parts:2},8192).unwrap().expanded_bytes(1),393_216);
}

#[test]
fn physical_cache_and_memo_charges_include_page_rounding() {
    let owned=layout(64,256,256,1,2);assert_eq!(owned.row_words(),16384);
    assert_eq!(owned.allocation_bytes(64).unwrap(),2_097_152);
    assert_eq!(owned.allocation_bytes(1152).unwrap(),37_748_736);
    let tiny=layout(6,3,2,1,3);assert_eq!(tiny.allocation_bytes(1).unwrap(),4096);
    assert_eq!(tiny.allocation_bytes(103).unwrap(),4096);assert_eq!(tiny.allocation_bytes(0).unwrap(),0);
    assert!(owned.expanded_words(usize::MAX).is_err());assert!(owned.allocation_bytes(usize::MAX).is_err());
    let mut description=description();for layer in &mut description.layers {if let TokenMixer::LatentAttention(ref mut spec)=layer.mixer {spec.heads=6;spec.key_dim=3;spec.value_dim=5;}}
    let ledger=SessionChargePlan::for_sparse_heads(&description,Topology {rank:1,parts:2},128).unwrap();
    assert_eq!(ledger.expanded_bytes(0),0);assert_eq!(ledger.expanded_bytes(1),45_056);assert_eq!(ledger.expanded_bytes(43),45_056);
}

#[test]
fn malformed_shapes_packets_and_unsupported_complete_unit_splits_are_refused() {
    for (heads,key,value,rank,parts) in [(0,1,1,0,1),(1,0,1,0,1),(1,1,0,0,1),(64,256,256,0,0),(64,256,256,2,2),(64,256,256,0,3),(64,256,256,0,65),(u64::MAX,1,1,0,1),(2,u64::MAX,1,0,1)] {
        assert!(Layout::new(heads,key,value,Topology {rank,parts}).is_err());
    }
    let owned=layout(64,256,256,0,2);assert!(owned.local_expanded(&[0u32;3]).is_err());assert!(owned.local_queries(&[0u32;3]).is_err());
    assert!(sparse_heads::gather_attended::<u32>(&[],1).is_err());
    let left=layout(4,2,2,0,2);let right=layout(4,2,2,1,2);let words=[0u32;4];
    assert!(sparse_heads::gather_attended(&[(left,&words)],1).is_err());
    assert!(sparse_heads::gather_attended(&[(right,&words),(left,&words)],1).is_err());
    assert!(sparse_heads::gather_attended(&[(left,&words),(left,&words)],1).is_err());
    assert!(sparse_heads::gather_attended(&[(left,&words),(right,&words[..3])],1).is_err());
    assert!(sparse_heads::gather_attended(&[(left,&words),(layout(4,3,2,1,2),&words)],1).is_err());
    assert!(sparse_heads::gather_attended(&[(left,&words),(right,&words)],usize::MAX).is_err());
    assert_eq!(sparse_heads::gather_attended::<u32>(&[(left,&[]),(right,&[])],0).unwrap(),Vec::<u32>::new());
}
