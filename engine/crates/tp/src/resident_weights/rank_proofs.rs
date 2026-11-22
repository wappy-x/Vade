//! CPU words for the resident decode paths that the host decoder does not launch.
use super::{Rows,Topology,WeightStorage,batch_layout::Layout};
use crate::matrix_role::{fp8_host_product,nv_host_product,nv_role};
use engine_model::plan::storage::ScaleEncoding;
use spark_gpu_probe::{causal_ops::{self,Shape,Op},expert_ops,tc_twin::Rng};

fn shape(op:Op,rows:usize,cols:usize)->Shape {
    Shape {op,heads:0,dim:0,taps:0,rows:rows as u32,cols:cols as u32,epsilon:0.0,bound:2.5}
}
fn bits(words:&[f32])->Vec<u32> {words.iter().map(|word|word.to_bits()).collect()}
fn sample(i:usize,salt:usize)->f32 {(((i*31+i/7+salt)%139) as f32-69.0)/64.0}
const NV_STORAGE:WeightStorage=WeightStorage::Nvfp4 {encoding:ScaleEncoding::Direct,activation_scale:true};

fn owners(rows:usize,cols:usize,storage:WeightStorage,parts:u32)->Vec<Rows> {
    (0..parts).map(|rank|Rows::new(&[rows as u64,cols as u64],storage,Topology {rank,parts}).unwrap()).collect()
}

/// The same scatter table used by the resident joined put and takes, applied to each rank's payload.
fn gather(rows:&[Rows],packets:&[Vec<f32>])->Vec<f32> {
    let layout=Layout::new(rows).unwrap();
    assert_eq!(layout.chunks,packets.len());
    let mut output=vec![f32::NAN;layout.output_words().unwrap()];
    for (rank,packet) in packets.iter().enumerate() {
        assert_eq!(packet.len(),layout.stride);
        for [src,dst,count] in layout.scatter(rank).unwrap() {
            assert!(output[dst..dst+count].iter().all(|value|value.is_nan()));
            output[dst..dst+count].copy_from_slice(&packet[src..src+count]);
        }
    }
    assert!(output.iter().all(|value|value.is_finite()));
    output
}

fn fp8_outputs(parts:u32,rows:usize,cols:usize,block:[usize;2],codes:&[u8],scales:&[u8],input:&[f32])->Vec<f32> {
    let [br,bc]=block;
    let scale_bytes=cols.div_ceil(bc)*4;
    let owners=owners(rows,cols,WeightStorage::Fp8Block {block:[br as u64,bc as u64]},parts);
    assert!(owners.iter().all(|owner|owner.split));
    let packets=owners.iter().map(|owner| {
        let start=owner.start;let end=start+owner.count;
        fp8_host_product(owner.count,cols,&codes[start*cols..end*cols],
            &scales[start/br*scale_bytes..end.div_ceil(br)*scale_bytes],br,bc,input).unwrap()
    }).collect::<Vec<_>>();
    gather(&owners[..1],&packets)
}

fn fp8_weights(rows:usize,cols:usize,[br,bc]:[usize;2],rng:&mut Rng)->(Vec<u8>,Vec<u8>) {
    let codes=(0..rows*cols).map(|_|(rng.below(127) as u8)^((rng.below(2) as u8)<<7)).collect();
    let scales=(0..rows.div_ceil(br)*cols.div_ceil(bc)).flat_map(|i|(2.0+sample(i,9)).to_le_bytes()).collect();
    (codes,scales)
}

#[test]
fn block_fp8_scale_bands_keep_every_projected_bit() {
    let mut rng=Rng(0x5ca99a55);
    // Eight scale bands put distinct scale codes on both sides of each rank seam.
    for (cols,block) in [(1536,[128,128]),(4096,[128,128]),(16384,[128,128]),(12288,[1,128])] {
        let rows=8*block[0];
        let (codes,scales)=fp8_weights(rows,cols,block,&mut rng);
        let input=(0..cols).map(|i|sample(i,31)).collect::<Vec<_>>();
        let full=fp8_host_product(rows,cols,&codes,&scales,block[0],block[1],&input).unwrap();
        for parts in [2,4] {
            assert_eq!(bits(&fp8_outputs(parts,rows,cols,block,&codes,&scales,&input)),bits(&full),"{parts} ranks, {cols} columns");
        }
    }
}

#[test]
fn shared_expert_block_fp8_step_keeps_gate_up_and_down_words() {
    let (hidden,width)=(4096,512);
    let block=[128,128];
    let mut rng=Rng(0x372);
    let gate=fp8_weights(width,hidden,block,&mut rng);
    let up=fp8_weights(width,hidden,block,&mut rng);
    let down=fp8_weights(hidden/8,width,block,&mut rng);
    let input=(0..hidden).map(|i|sample(i,24)).collect::<Vec<_>>();
    let run=|parts| {
        let gate=fp8_outputs(parts,width,hidden,block,&gate.0,&gate.1,&input);
        let up=fp8_outputs(parts,width,hidden,block,&up.0,&up.1,&input);
        let middle=causal_ops::reference(shape(Op::SwiGlu,width,0),[&gate,&up,&[],&[]],&[]).unwrap();
        let down=fp8_outputs(parts,hidden/8,width,block,&down.0,&down.1,&middle);
        [gate,up,middle,down].map(|row|bits(&row))
    };
    assert_eq!(run(2),run(4));
}

#[test]
fn dense_mxfp8_step_keeps_gate_up_and_down_words() {
    use spark_gpu_probe::{mxfp8_projection::Shape as Mx,next_params::MX_TC,norm_boundary::WeightType};
    let (hidden,width,output)=(4096,12288,32);
    let gate_rows=64;
    let mut rng=Rng(0x608);
    let mut weights=|rows,cols| {
        let codes=(0..rows*cols).map(|_|(0x30+rng.below(16) as u8) | ((rng.below(2) as u8)<<7)).collect::<Vec<_>>();
        let scales=(0..rows*cols/32).map(|_|118+rng.below(8) as u8).collect::<Vec<_>>();
        (codes,scales)
    };
    let gate=weights(gate_rows,hidden);
    let up=weights(gate_rows,hidden);
    let down=weights(output,width);
    let input=(0..hidden).map(|i|sample(i,25)).collect::<Vec<_>>();
    let run=|parts| {
        let project=|weights:&(Vec<u8>,Vec<u8>),rows,cols,input:&[f32]| {
            let owners=owners(rows,cols,WeightStorage::Mxfp8,parts);
            let packets=owners.iter().map(|owner| {
                let (start,end)=(owner.start*cols,(owner.start+owner.count)*cols);
                Mx::new(owner.count,cols,WeightType::F32).unwrap()
                    .reference_mode(MX_TC,&weights.0[start..end],&weights.1[start/32..end/32],input).unwrap().1
            }).collect::<Vec<_>>();
            gather(&owners[..1],&packets)
        };
        let gate=project(&gate,gate_rows,hidden,&input);
        let up=project(&up,gate_rows,hidden,&input);
        let middle=causal_ops::reference(shape(Op::SwiGlu,gate_rows,0),[&gate,&up,&[],&[]],&[]).unwrap().repeat(width/gate_rows);
        let down=project(&down,output,width,&middle);
        [gate,up,middle,down].map(|row|bits(&row))
    };
    assert_eq!(run(2),run(4));
}

struct Nv {codes:Vec<u8>,scales:Vec<u8>,cols:usize,role:u32}
impl Nv {
    fn new(name:&str,rows:usize,cols:usize,rng:&mut Rng)->Self {
        Self {codes:(0..rows*cols/2).map(|_|rng.next_u64() as u8).collect(),
            scales:(0..rows*cols/16).map(|_|0x28+rng.below(32) as u8).collect(),cols,role:nv_role(name)}
    }
    fn project(&self,owner:Rows,input:&[f32])->Vec<f32> {
        let (start,end)=(owner.start*self.cols,(owner.start+owner.count)*self.cols);
        nv_host_product(owner.count,self.cols,&self.codes[start/2..end/2],&self.scales[start/16..end/16],0.75,1.0,self.role,input).unwrap()
    }
}

#[test]
fn routed_decode_keeps_local_middle_and_fold_words() {
    let (rows,width,experts,picks)=(256,256,8,4);
    let mut rng=Rng(0x91);
    let input=(0..rows).map(|i|sample(i,10)).collect::<Vec<_>>();
    let router=(0..experts*rows).map(|i|sample(i,17)).collect::<Vec<_>>();
    let corrections=(0..experts).map(|i|sample(i,2)/8.0).collect::<Vec<_>>();
    let gate=(0..experts).map(|_|Nv::new("gate_proj",width,rows,&mut rng)).collect::<Vec<_>>();
    let up=(0..experts).map(|_|Nv::new("up_proj",width,rows,&mut rng)).collect::<Vec<_>>();
    let down=(0..experts).map(|_|Nv::new("down_proj",rows,width,&mut rng)).collect::<Vec<_>>();
    let shared=Nv::new("down_proj",rows,width,&mut rng);
    let run=|parts| {
        let router_owners=owners(experts,rows,WeightStorage::Plain,parts);
        let logits=gather(&router_owners[..1],&router_owners.iter().map(|owner| {
            let start=owner.start*rows;let end=start+owner.count*rows;
            causal_ops::reference(shape(Op::Project,owner.count,rows),[&input,&[],&[],&[]],&router[start..end]).unwrap()
        }).collect::<Vec<_>>());
        let route=causal_ops::reference(Shape {heads:2,dim:2,taps:1,..shape(Op::Route,experts,picks)},
            [&logits,&[],&[],&[]],&corrections).unwrap();
        let (ids,probability)=(&route[..picks],&route[picks..2*picks]);
        let gate_owners=owners(width,rows,NV_STORAGE,parts);
        let down_owners=owners(rows,width,NV_STORAGE,parts);
        let (mut gates,mut ups,mut middles,mut downs,mut folded,mut shared_rows)=(Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new());
        for &id in ids {
            let id=id as usize;
            let gate_parts=gate_owners.iter().map(|&owner|gate[id].project(owner,&input)).collect::<Vec<_>>();
            let up_parts=gate_owners.iter().map(|&owner|up[id].project(owner,&input)).collect::<Vec<_>>();
            let middle_parts=gate_parts.iter().zip(&up_parts).map(|(gate,up)|
                causal_ops::reference(shape(Op::SwiGlu,gate.len(),0),[gate,up,&[],&[]],&[]).unwrap()).collect::<Vec<_>>();
            let middle=gather(&gate_owners[..1],&middle_parts);
            downs.push(down_owners.iter().map(|&owner|down[id].project(owner,&middle)).collect::<Vec<_>>());
            gates.extend(gather(&gate_owners[..1],&gate_parts));
            ups.extend(gather(&gate_owners[..1],&up_parts));
            middles.extend(middle);
        }
        for (rank,&owner) in down_owners.iter().enumerate() {
            let local=downs.iter().flat_map(|pick|pick[rank].iter().copied()).collect::<Vec<_>>();
            let shared=shared.project(owner,&input);
            let fold=Shape {taps:1,dim:causal_ops::FOLD_ROUNDS,..shape(Op::ExpertFold,picks,owner.count)};
            assert!(causal_ops::fast_admitted(fold,causal_ops::FAST_THREADS));
            folded.push(causal_ops::reference(fold,[ids,probability,&local,&shared],&[]).unwrap());
            shared_rows.push(shared);
        }
        let shared=gather(&down_owners[..1],&shared_rows);
        let all_down=downs.iter().flat_map(|packets|gather(&down_owners[..1],packets)).collect::<Vec<_>>();
        let output=gather(&down_owners[..1],&folded);
        let full=causal_ops::reference(Shape {taps:1,..shape(Op::ExpertFold,picks,rows)},
            [ids,probability,&all_down,&shared],&[]).unwrap();
        assert_eq!(bits(&output),bits(&full),"local fold equals the full output-row fold");
        [logits,route,gates,ups,middles,all_down,shared,output].map(|words|bits(&words))
    };
    assert_eq!(run(4),run(2));
}

#[test]
fn nvfp4_general_row_geometry_keeps_the_packed_projection_contract() {
    // The grouped kernel lists the pair's local widths. The quarter widths
    // use general row geometry and keep the same first/second matrix role.
    for (rows,cols,name) in [(2048,4096,"gate_proj"),(4096,2048,"down_proj")] {
        let pair=owners(rows,cols,NV_STORAGE,2);
        let four=owners(rows,cols,NV_STORAGE,4);
        assert!(expert_ops::nv_packed_shape(pair[0].count as u32,cols as u32,1));
        assert!(!expert_ops::nv_packed_shape(four[0].count as u32,cols as u32,1));
        let mut rng=Rng(0x17);
        let input=(0..cols).map(|i|sample(i,7)).collect::<Vec<_>>();
        let matrix=Nv::new(name,rows,cols,&mut rng);
        let project=|owners:&[Rows]|gather(&owners[..1],&owners.iter().map(|&owner|matrix.project(owner,&input)).collect::<Vec<_>>());
        assert_eq!(bits(&project(&pair)),bits(&project(&four)),"{name}");
    }
}

#[test]
fn recurrent_planar_pair_and_quarter_match_the_general_layout() {
    let (heads,dim)=(64,128);
    let prepared=(0..heads*(4*dim+1)).map(|i|sample(i,3)/8.0).collect::<Vec<_>>();
    let state=(0..heads*dim*dim).map(|i|sample(i+i/(dim*dim),11)).collect::<Vec<_>>();
    let decode=|parts,planar_enabled| {
        let mut states=Vec::new();
        let mut rows=Vec::new();
        for rank in 0..parts {
            let topology=Topology {rank,parts};
            let layout=crate::sparse_heads::Layout::new(heads as u64,dim as u64,dim as u64,topology).unwrap();
            let local=layout.local_heads();
            let range=layout.heads();
            let p=&prepared[range.start*(4*dim+1)..range.end*(4*dim+1)];
            let s=&state[range.start*dim*dim..range.end*dim*dim];
            let planar=Shape {heads:local as u32,dim:dim as u32,..shape(Op::StepPlanar,0,0)};
            assert!(causal_ops::fast_admitted(planar,causal_ops::FAST_THREADS));
            let fast=planar_enabled;
            let op=if fast {planar} else {Shape {op:Op::Step,..planar}};
            let result=causal_ops::reference(op,[p,s,&[],&[]],&[]).unwrap();
            if fast {
                states.extend_from_slice(&result[..local*dim*dim]);
                rows.extend_from_slice(&result[local*dim*dim..]);
            } else {
                for head in result.chunks_exact(dim*dim+dim) {
                    states.extend_from_slice(&head[..dim*dim]);
                    rows.extend_from_slice(&head[dim*dim..]);
                }
            }
        }
        assert_eq!(states.len(),state.len());
        assert_eq!(rows.len(),heads*dim);
        (bits(&states),bits(&rows))
    };
    let pair=decode(2,true);
    assert_eq!(pair,decode(4,true));
    assert_eq!(pair,decode(4,false));
}

#[test]
fn nvfp4_raw_quarters_match_the_served_second_matrix_word() {
    use spark_gpu_probe::{tc_twin as nv,tc_twin_fold as fold};
    for cols in [1024,2048,4096,8192,12288,16384] {
        let quarter=cols/4;
        let rows=(0..8).map(|row|nv::hard_row(cols,(cols+row) as u64)).collect::<Vec<_>>();
        let form=|parts:usize|rows.iter().map(|row| {
            let partials=(0..parts).map(|rank| {
                let owned=crate::rows(4,rank,parts);
                let raw=owned.map(|index| {
                    let range=index*quarter..(index+1)*quarter;
                    let scales=range.start/16..range.end/16;
                    nv::chain(&row.w[range.clone()],&row.a[range],&row.sw[scales.clone()],&row.sa[scales])
                }).collect::<Vec<_>>();
                fold::join(&raw)
            }).collect::<Vec<_>>();
            let joined=fold::reduce(&partials);
            let word=nv::word(joined,0.75,0.5);
            assert_eq!(joined,nv::chain_quarters(&row.w,&row.a,&row.sw,&row.sa));
            (joined,word)
        }).collect::<Vec<_>>();
        assert_eq!(form(2),form(4),"{cols} columns");
    }
}

#[test]
fn dense_latency_split_and_inline_merge_keep_the_same_canonical_word() {
    use spark_gpu_probe::{kernel_params::latency_split,tc_twin_bf16 as bf,tc_twin_fold as fold};
    // The quarter owner crosses the solo row threshold. The CPU twin
    // models the four canonical K ranges used by both kernel schedules.
    let (rows,cols)=(2048,256);
    let input=(0..cols).map(|i|sample(i,7)).collect::<Vec<_>>();
    let weight=(0..rows*cols).map(|i|bf::bf16(sample(i,i/cols))).collect::<Vec<_>>();
    let run=|parts| {
        let ownership=owners(rows,cols,WeightStorage::Plain,parts);
        let split=latency_split(ownership[0].count,false);
        assert_eq!(split,parts==4);
        ownership.into_iter().flat_map(|owner| {
            weight[owner.start*cols..(owner.start+owner.count)*cols].chunks_exact(cols).map(|row| {
                let acc=if split {fold::join(&bf::parts(row,&input))} else {bf::canonical(row,&input)};
                (acc,bf::word(acc))
            })
        }).collect::<Vec<_>>()
    };
    assert_eq!(run(2),run(4));
}

#[test]
fn sparse_absorption_keeps_each_head_and_the_output_projection() {
    use crate::sparse_heads::{Layout as Heads,gather_attended};
    use spark_gpu_probe::tc_twin_bf16::bf16;
    let (heads,key,value,latent,slots)=(8,32,32,32,5);
    let query=(0..heads*key).map(|i|f32::from_bits(u32::from(bf16(sample(i,14)))<<16)).collect::<Vec<_>>();
    let cache=(0..slots*latent).map(|i|f32::from_bits(u32::from(bf16(sample(i,23)))<<16)).collect::<Vec<_>>();
    // Packed and transposed as load-time absorption expects: each head has
    // its own key and value weights, two BF16 patterns per word.
    let packed=(0..heads*(key+value)*latent/2).map(|i| {
        let lo=u32::from(bf16(sample(2*i,37)));
        let hi=u32::from(bf16(sample(2*i+1,37)));
        f32::from_bits(lo|(hi<<16))
    }).collect::<Vec<_>>();
    let output=Nv::new("o_proj",64,heads*value,&mut Rng(0x49));
    let decode=|parts| {
        let mut observed=[Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new()];
        let mut shards=Vec::new();
        for rank in 0..parts {
            let layout=Heads::new(heads as u64,key as u64,value as u64,Topology {rank,parts}).unwrap();
            let rows=layout.projection_rows();
            let weights=&packed[rows.start*latent/2..rows.end*latent/2];
            let s=Shape {heads:layout.local_heads() as u32,dim:key as u32,taps:value as u32,..shape(Op::AbsorbQuery,slots,latent)};
            let local_query=layout.local_queries(&query).unwrap();
            let aq=causal_ops::reference(s,[&local_query,&[],&[],&[]],weights).unwrap();
            let count=[(slots-1) as f32];
            let scores=causal_ops::reference(Shape {op:Op::AbsorbScores,..s},[&aq,&cache,&count,&[]],&[]).unwrap();
            let probability=causal_ops::reference(Shape {op:Op::Softmax,cols:0,taps:0,..s},[&scores,&count,&[],&[]],&[]).unwrap();
            let values=causal_ops::reference(Shape {op:Op::AbsorbValues,..s},[&probability,&cache,&count,&[]],&[]).unwrap();
            let projected=causal_ops::reference(Shape {op:Op::AbsorbProject,..s},[&values,&[],&[],&[]],weights).unwrap();
            for (seen,current) in observed.iter_mut().zip([local_query,aq,scores,probability,values]) {seen.extend(bits(&current));}
            shards.push((layout,projected));
        }
        let attended=gather_attended(&shards.iter().map(|(layout,row)|(*layout,row.as_slice())).collect::<Vec<_>>(),1).unwrap();
        let rows=owners(64,attended.len(),NV_STORAGE,parts);
        let projected=gather(&rows[..1],&rows.iter().map(|&row|output.project(row,&attended)).collect::<Vec<_>>());
        (observed,bits(&attended),bits(&projected))
    };
    assert_eq!(decode(2),decode(4));
}

#[test]
fn replicated_mhc_keeps_both_boundary_outputs() {
    use crate::{bounded_decoder::{Compute,Host,TestCheckpoint},layer_driver::LayerBoundary};
    use engine_model::{plan::{layer::Kind,state::StatePlan},safetensors::Dtype};
    let checkpoint=TestCheckpoint::new_with_head(Dtype::Bf16,true);
    let admission=&checkpoint.admission;
    let d=admission.description();
    let hidden=d.hidden as usize;
    let input=(0..hidden*d.residual.streams as usize).map(|i|sample(i,29)).collect::<Vec<_>>();
    let matrix=(0..hidden*hidden).map(|i|sample(i,2)/16.0).collect::<Vec<_>>();
    let run=|parts| {
        let states=(0..parts).map(|rank|StatePlan::new(d,Topology {rank,parts},8,isize::MAX as usize).unwrap()).collect::<Vec<_>>();
        let plans=states.iter().map(|state|LayerBoundary::new(d,admission.headers(),state,0,2,state.topology(),usize::MAX).unwrap()).collect::<Vec<_>>();
        plans[0].program().steps.iter().enumerate().filter(|(_,step)|step.kind==Kind::MhcPrepare).map(|(index,_)| {
            let packets=plans.iter().map(|plan| {
                let block=plan.prepare_residual_block_checkpoint(index,admission).unwrap();
                Host.begin((0,index),&block,&input).unwrap()
            }).collect::<Vec<_>>();
            let rows=owners(hidden,hidden,WeightStorage::Plain,parts);
            let local=rows.iter().zip(&packets).map(|(owner,(_,normalized))| {
                let weight=&matrix[owner.start*hidden..(owner.start+owner.count)*hidden];
                causal_ops::reference(shape(Op::Project,owner.count,hidden),[normalized,&[],&[],&[]],weight).unwrap()
            }).collect::<Vec<_>>();
            let block=gather(&rows[..1],&local);
            let mut outputs=packets.into_iter().map(|(pass,normalized)| {
                (bits(&normalized),bits(&Host.finish(pass,&block).unwrap()))
            });
            let first=outputs.next().unwrap();
            assert!(outputs.all(|output|output==first));
            first
        }).collect::<Vec<_>>()
    };
    let pair=run(2);
    assert_eq!(pair.len(),2);
    assert_eq!(pair,run(4));
}
