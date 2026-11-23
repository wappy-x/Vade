//! Host tests of the block mix, the messages and the gates, on a made-up checkpoint.
//! The expected words come from a scalar replay that keeps the block list as raw vectors and norms every entry at
//! every mix, as the reference does. The driver norms a finished block once. Both must give the same words.
use super::{*,support::{*,hand}};
use super::super::fixture::Fixture;
use spark_gpu_probe::fixed_math::rn;
use crate::bounded_decoder::Host;

const GATES:[Message;MESSAGES]=[Message::Query,Message::Key,Message::Value,Message::Expert];
/// What one visit gave: the inputs of the attention norms (one, or three), the input of the expert norm, and the
/// length of the block list after the visit.
#[derive(Debug,PartialEq)]
struct Seen {inputs:Vec<Vec<u32>>,expert:Vec<u32>,blocks:usize,messages:bool}
/// For each visit a made-up output of the first convolution and of the last convolution.
fn adds(d:&LoopTextDescription,seed:u64)->Vec<(Vec<f32>,Vec<f32>)> {
    let mut n=Numbers::new(seed);d.visits.iter().map(|_|(n.bf16(d.hidden as usize,1.5),n.bf16(d.hidden as usize,1.5))).collect()
}
fn sum(a:&[f32],b:&[f32])->Vec<f32> {a.iter().zip(b).map(|(x,y)|x+y).collect()}
/// The scalar form of one gate.
fn gate(d:&LoopTextDescription,w:&Made,layer:usize,which:Message,value:&[f32],message:&[f32])->Vec<f32> {
    let mut joined=hand::rounded_norm(value,&[],d.epsilon as f32);joined.extend(hand::rounded_norm(message,&[],d.epsilon as f32));
    let inner=hand::product(w.get(Weight::GateDown(layer,which)),&joined).into_iter().map(rn).collect::<Vec<_>>();
    hand::gate_blend(&hand::product(w.get(Weight::GateForget(layer,which)),&inner),&hand::product(w.get(Weight::GateGain(layer,which)),&inner),value,message)
}
/// One token through the driver's mix calls, with made-up outputs in place of attention and of the experts.
fn walk(c:&Stand,d:&LoopTextDescription,w:&Made,embedding:&[f32],adds:&[(Vec<f32>,Vec<f32>)])->(Vec<Seen>,Vec<u32>) {
    let run=Run::new(c,w,d).unwrap();let mut token=Token::new();let mut stream=embedding.to_vec();let mut seen=Vec::new();
    for (visit,(first,last)) in d.visits.iter().zip(adds) {
        let read=token.enter(&run,visit,stream).unwrap();
        let inputs=match token.attention_inputs(&run,visit,&read).unwrap() {Inputs::Same(one)=>vec![bits(&one)],Inputs::Each(each)=>each.iter().map(|v|bits(v)).collect()};
        let residual=if d.layers[visit.layer as usize].boundary {first.clone()} else {sum(&read,first)};
        let expert=bits(&token.expert_input(&run,visit,&read,&residual).unwrap());
        seen.push(Seen {inputs,expert,blocks:token.blocks(),messages:token.has_messages()});
        stream=sum(&residual,last).into_iter().map(rn).collect();
    }
    (seen,bits(&token.end_mix(&run,&stream).unwrap()))
}
/// The same token as a scalar replay of the visit order: the saved start, the restart of each later pass, the
/// messages from the block outputs of the pass before, the boundary rule, and the two mixes of each visit.
fn replay(d:&LoopTextDescription,w:&Made,embedding:&[f32],adds:&[(Vec<f32>,Vec<f32>)])->(Vec<Seen>,Vec<u32>) {
    let width=d.hidden as usize;let start=d.core.start as usize;
    let entries=|list:&[Vec<f32>],last:&[f32]|->Vec<Vec<f32>> {list.iter().cloned().chain([last.to_vec()]).collect()};
    let mix=|query:&[f32],all:&[Vec<f32>]|hand::mix(query,&all.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let (mut stream,mut list,mut saved,mut messages)=(embedding.to_vec(),Vec::<Vec<f32>>::new(),None,None::<Vec<Vec<f32>>>);
    let mut seen=Vec::new();
    for (visit,(first,last)) in d.visits.iter().zip(adds) {
        let layer=visit.layer as usize;let spec=&d.layers[layer];
        if layer==start {
            if visit.messages {
                let (first_stream,first_list):(Vec<f32>,Vec<Vec<f32>>)=saved.clone().unwrap();
                let from=entries(&list[first_list.len()+1..],&stream);
                messages=Some(w.get(Weight::Messages).chunks_exact(width).map(|query|mix(query,&from)).collect());
                stream=first_stream;list=first_list;
            } else {saved=Some((stream.clone(),list.clone()));messages=None;}
        }
        if !visit.messages {messages=None;}
        let keep=stream.clone();
        let mut inputs=if spec.attention_queries==0 {vec![keep.clone()]} else {
            w.get(Weight::MixAttention(layer)).chunks_exact(width).map(|query|mix(query,&entries(&list,&keep))).collect()};
        if let Some(all)=&messages {
            inputs=(0..ATTENTION_INPUTS).map(|i|gate(d,w,layer,GATES[i],&inputs[if inputs.len()==1 {0} else {i}],&all[(layer-start)*MESSAGES+i])).collect();
        }
        let residual=if spec.boundary {list.push(keep.clone());first.clone()} else {sum(&keep,first)};
        let mut expert=mix(w.get(Weight::MixExperts(layer)),&entries(&list,&residual));
        if let Some(all)=&messages {expert=gate(d,w,layer,Message::Expert,&expert,&all[(layer-start)*MESSAGES+3]);}
        seen.push(Seen {inputs:inputs.iter().map(|v|bits(v)).collect(),expert:bits(&expert),blocks:list.len(),messages:messages.is_some()});
        stream=sum(&residual,last).into_iter().map(rn).collect();
    }
    (seen,bits(&mix(w.get(Weight::MixOutput),&entries(&list,&stream))))
}
fn check_walk(d:&LoopTextDescription,seed:u64)->(Stand,Vec<Seen>) {check_with(d,&Made::new(d,seed),seed)}
fn check_with(d:&LoopTextDescription,w:&Made,seed:u64)->(Stand,Vec<Seen>) {
    let c=Stand::new();let adds=adds(d,seed+100);let embedding=Numbers::new(seed+200).bf16(d.hidden as usize,2.0);
    let (seen,end)=walk(&c,d,w,&embedding,&adds);let (expected,expected_end)=replay(d,w,&embedding,&adds);
    assert_eq!(seen.len(),d.visits.len());
    for (index,(got,want)) in seen.iter().zip(&expected).enumerate() {assert_eq!(got,want,"visit {index} (layer {})",d.visits[index].layer);}
    assert_eq!(end,expected_end,"end mix");
    (c,seen)
}

#[test]
fn a_token_walk_equals_the_scalar_replay() {
    let d=small();let (c,seen)=check_walk(&d,1);
    // Layers 0, 1 once; layers 2, 3 in three passes; layers 4, 5 once.
    assert_eq!(d.visits.iter().map(|v|v.layer).collect::<Vec<_>>(),[0,1,2,3,2,3,2,3,4,5]);
    // The block list after each visit: it grows at the first layer of a block and a later pass restarts it.
    assert_eq!(seen.iter().map(|s|s.blocks).collect::<Vec<_>>(),[1,1,2,2,2,2,2,2,3,3]);
    assert_eq!(seen.iter().map(|s|s.messages).collect::<Vec<_>>(),[false,false,false,false,true,true,true,true,false,false]);
    // One input serves query, key and value where the layer has one query or none and reads no message.
    assert_eq!(seen.iter().map(|s|s.inputs.len()).collect::<Vec<_>>(),[1,3,1,3,3,3,3,3,1,3]);
    // Another seed is another set of numbers, with the same structure.
    for seed in [2,3] {check_walk(&d,seed);}
    // What one token costs here. A later pass makes all its messages in one blend. No value comes to the host.
    let (mixes,passes)=(9+10+1,2);
    assert_eq!(c.count(Op::MixBlend),mixes+passes);assert_eq!(c.count(Op::RowDots),mixes+passes);
    // One plain norm a mix, one for the embedding that layer 0 puts on the list, one a later pass for its last stream.
    assert_eq!(c.count(Op::PlainNorm),mixes+1+passes);
    assert_eq!(c.count(Op::GateBlend),4*4);
    // Rounded norms of a visit with messages: the value once for one input (or once each for three), and four messages.
    // A later pass norms all its messages in one launch; a visit with three inputs norms them in one launch.
    assert_eq!(c.count(Op::RoundedNorm),passes*(1+(1+1)+(1+1)));
    // The driver never calls the norm of the backend: on a device that norm rounds its input first.
    assert_eq!(c.norms.get(),0);
    assert_eq!(c.reads.get(),0);
    // Every launch of a block arm ran on the arm and had the bits of its hand form.
    assert!(c.standing.borrow().is_empty() && c.compared.get()>0);
}
#[test]
fn every_shape_of_loop_equals_the_scalar_replay() {
    // One pass: no message, no restart, no gate.
    let (c,seen)=check_walk(&description(6,2,2,4,1,&[0,2,4]),4);
    assert!(seen.iter().all(|s|!s.messages));assert_eq!((c.count(Op::GateBlend),c.count(Op::RoundedNorm),c.norms.get()),(0,0,0));
    // A loop over two blocks: a message blends two block outputs, as the released checkpoint does.
    let d=description(8,2,2,6,2,&[0,1,2,4,6]);let (_,seen)=check_walk(&d,5);
    assert_eq!(seen.iter().map(|s|s.blocks).collect::<Vec<_>>(),[1,1,2,2,3,3,2,2,3,3,4,4]);
    // A loop over five blocks: a message blends five vectors in one dot launch.
    let d=description(12,2,2,12,2,&[0,2,3,5,7,8,10]);let (c,seen)=check_walk(&d,6);
    assert_eq!(seen.last().unwrap().blocks,6);assert_eq!(c.count(Op::RowDots),c.count(Op::MixBlend));
    // Blocks of three layers and a loop that ends with the layers.
    check_walk(&description(6,3,3,6,3,&[0,1,3,4]),7);
    // The loop is the whole tail and has one layer a block.
    check_walk(&description(4,1,1,4,2,&[0,1,2,3]),8);
}
/// The two made-up checkpoints of the fixture file, with their stored weights. Each visit list comes from its config
/// text. The wide one is above 128 values a row, so the two sum trees of a norm give different words there.
#[test]
fn the_visit_list_of_each_fixture_walks_as_the_scalar_replay() {
    let small=Fixture::small();let d=&small.description;let (c,seen)=check_with(d,&Made::stored(&small),31);
    assert_eq!(d.visits.iter().map(|v|v.layer).collect::<Vec<_>>(),[0,1,2,3,2,3]);
    assert_eq!(seen.iter().map(|s|s.blocks).collect::<Vec<_>>(),[1,1,2,2,2,2]);
    assert_eq!(seen.iter().map(|s|s.messages).collect::<Vec<_>>(),[false,false,false,false,true,true]);
    assert_eq!(c.count(Op::GateBlend),2*MESSAGES);
    let wide=Fixture::wide();let d=&wide.description;let (c,seen)=check_with(d,&Made::stored(&wide),32);
    assert_eq!(d.visits.iter().map(|v|v.layer).collect::<Vec<_>>(),[0,1,2,3,4,5,2,3,4,5,2,3,4,5,6,7]);
    // Each later pass starts again from the list that the first pass met, and layers 6 and 7 read all four blocks.
    assert_eq!(seen.iter().map(|s|s.blocks).collect::<Vec<_>>(),[1,1,2,2,3,3,2,2,3,3,2,2,3,3,4,4]);
    assert_eq!(seen.iter().filter(|s|s.messages).count(),8);assert_eq!(c.count(Op::GateBlend),8*MESSAGES);
    assert!(c.standing.borrow().is_empty());
}
#[test]
fn a_long_stack_runs_in_parts_with_the_same_words() {
    let d=small();let w=Made::new(&d,9);let width=d.hidden as usize;let mut n=Numbers::new(10);
    for count in [1usize,2,4,5,6,8,9,13] {for queries in [1usize,3,7] {
        let c=Stand::new();let run=Run::new(&c,&w,&d).unwrap();
        let rows=(0..count).map(|_|hand::plain_norm(&n.words(width,3.0),MIX_NORM_EPSILON)).collect::<Vec<_>>();
        let query=n.words(queries*width,2.0);
        let got=run.blend(&query,&rows.iter().collect::<Vec<_>>(),1).unwrap();
        let slices=rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let want=query.chunks_exact(width).flat_map(|q|hand::blend(&slices.iter().map(|row|hand::dot(q,row)*mix_scale(width)).collect::<Vec<_>>(),&slices)).collect::<Vec<_>>();
        assert_eq!(bits(&got),bits(&want),"count {count} queries {queries}");
        assert_eq!(c.count(Op::RowDots),1);assert_eq!(c.count(Op::MixBlend),1);
    }}
}
#[test]
fn the_weights_of_a_mix_add_up_to_one_and_follow_the_query() {
    // Two rows that are the two unit directions, and a query that points at the second: the blend leans to row two.
    let d=small();let w=Made::new(&d,11);let c=Stand::new();let run=Run::new(&c,&w,&d).unwrap();let width=d.hidden as usize;
    let (mut a,mut b)=(vec![0.0f32;width],vec![0.0f32;width]);a[0]=1.0;b[1]=1.0;
    let mut query=vec![0.0f32;width];query[1]=4.0*(width as f32).sqrt();
    let out=run.blend(&query,&[&a,&b],1).unwrap();
    // Scores 0 and 4: the weights are 1 / (1 + e^4) and e^4 / (1 + e^4).
    let (first,second)=(1.0/(1.0+4.0f64.exp()),4.0f64.exp()/(1.0+4.0f64.exp()));
    assert!((f64::from(out[0])-first).abs()<1e-6 && (f64::from(out[1])-second).abs()<1e-6,"{} {}",out[0],out[1]);
    assert!(out[2..].iter().all(|&v|v==0.0));
    // A zero query gives equal weights.
    let even=run.blend(&vec![0.0;width],&[&a,&b],1).unwrap();assert_eq!((even[0],even[1]),(0.5,0.5));
}
#[test]
fn a_gate_with_no_outer_weight_returns_the_value() {
    let d=small();let mut w=Made::new(&d,12);let width=d.hidden as usize;let mut n=Numbers::new(13);
    for part in [Weight::GateForget(2,Message::Key),Weight::GateGain(2,Message::Key)] {w.words.get_mut(&part).unwrap().fill(0.0);}
    let c=Stand::new();let run=Run::new(&c,&w,&d).unwrap();
    let (value,message)=(n.words(width,2.0).iter().map(|v|v+3.0).collect::<Vec<_>>(),n.words(width,2.0));
    let (normed,message_normed)=(run.gate_norm(&value).unwrap(),run.gate_norm(&message).unwrap());
    assert_eq!(bits(&run.gate(2,Message::Key,&value,&normed,(&message,&message_normed)).unwrap()),bits(&value));
    // With its weights the same gate moves the value, and it equals the scalar form.
    let moved=run.gate(2,Message::Query,&value,&normed,(&message,&message_normed)).unwrap();
    // Many rows in one norm launch give the words of one launch a row.
    let both=run.gate_norm(&[value.clone(),message.clone()].concat()).unwrap();
    assert_eq!(bits(&both),[bits(&normed),bits(&message_normed)].concat());
    assert_ne!(bits(&moved),bits(&value));assert_eq!(bits(&moved),bits(&gate(&d,&w,2,Message::Query,&value,&message)));
}
#[test]
fn the_norms_of_a_gate_take_the_config_epsilon_and_a_mix_takes_its_own() {
    let d=small();let w=Made::new(&d,14);let c=Stand::new();let run=Run::new(&c,&w,&d).unwrap();let width=d.hidden as usize;
    assert_eq!(d.epsilon,1e-5);
    // Small values, so the epsilon decides the scale.
    let x=Numbers::new(15).words(width,0.004);
    let scale=|epsilon:f32|1.0/(hand::tree(&x)/width as f32+epsilon).sqrt();
    // A gate norm is rounded once. Under 128 values the norm kernel's tree is the 32-lane tree.
    assert_eq!(bits(&run.gate_norm(&x).unwrap()),bits(&x.iter().map(|&v|rn(v*scale(1e-5))).collect::<Vec<_>>()));
    assert_eq!(bits(&run.plain_norm(&x,1).unwrap()),bits(&x.iter().map(|&v|v*scale(MIX_NORM_EPSILON)).collect::<Vec<_>>()));
    assert_ne!(scale(1e-5).to_bits(),scale(MIX_NORM_EPSILON).to_bits());
    // Two rows in one launch are the two single rows.
    let y=Numbers::new(16).words(width,2.0);let both=run.plain_norm(&[x.clone(),y.clone()].concat(),2).unwrap();
    assert_eq!(bits(&both),bits(&[run.plain_norm(&x,1).unwrap(),run.plain_norm(&y,1).unwrap()].concat()));
}
#[test]
fn the_score_scale_is_the_64_bit_value_rounded_once() {
    assert_eq!(mix_scale(2304).to_bits(),(1.0f32/48.0).to_bits());assert_eq!(mix_scale(4),0.5);
    for width in [1usize,2,3,32,40,1000,4096,65_536] {assert_eq!(mix_scale(width).to_bits(),((width as f64).powf(-0.5) as f32).to_bits(),"width {width}");}
}
#[test]
fn a_wrong_extent_or_order_is_refused() {
    let d=small();let w=Made::new(&d,17);let c=Stand::new();let run=Run::new(&c,&w,&d).unwrap();let width=d.hidden as usize;
    let stream=vec![1.0f32;width];let short=vec![1.0f32;width-1];
    let mut token=Token::new();
    assert!(token.enter(&run,&d.visits[0],short.clone()).unwrap_err().contains("no whole count of rows of the width"));
    assert!(token.attention_inputs(&run,&d.visits[1],&short).is_err());
    assert!(token.expert_input(&run,&d.visits[0],&stream,&short).is_err());
    assert!(token.end_mix(&run,&short).is_err());
    // A later pass with no first pass before it.
    let later=d.visits.iter().find(|v|v.messages).unwrap();
    assert!(Token::new().enter(&run,later,stream.clone()).unwrap_err().contains("no saved start"));
    // A visit that reads messages in a pass that made none.
    let second=d.visits.iter().find(|v|v.messages && v.layer!=d.core.start).unwrap();
    assert!(Token::new().enter(&run,second,stream.clone()).unwrap_err().contains("differ on the messages"));
    assert!(Token::<Vec<f32>>::new().attention_inputs(&run,later,&stream).unwrap_err().contains("no pass made"));
    // Message queries of another loop length.
    let mut cut=Made::new(&d,17);cut.words.get_mut(&Weight::Messages).unwrap().truncate(width);
    let cut_run=Run::new(&c,&cut,&d).unwrap();let mut token=Token::new();let mut at=stream.clone();
    let mut refused=None;
    for visit in &d.visits {
        at=match token.enter(&cut_run,visit,at) {Ok(v)=>v,Err(e)=>{refused=Some(e);break;}};
        token.attention_inputs(&cut_run,visit,&at).unwrap();token.expert_input(&cut_run,visit,&at,&stream).unwrap();
    }
    assert!(refused.unwrap().contains("message queries differ"));
    // Two attention queries are no form of this family.
    let mut odd=d.clone();odd.layers[1].attention_queries=2;
    let mut two=Made::new(&d,17);two.words.get_mut(&Weight::MixAttention(1)).unwrap().truncate(2*width);
    let odd_run=Run::new(&c,&two,&odd).unwrap();
    assert!(Token::new().attention_inputs(&odd_run,&odd.visits[1],&stream).unwrap_err().contains("one query or three"));
    // A mix over raw vectors, and a visit outside the layers.
    let mut raw=d.clone();raw.residual.normalize=false;assert!(Run::new(&c,&w,&raw).is_err());
    let mut outside=d.visits[0];outside.layer=d.layers.len() as u64;
    assert!(Token::new().attention_inputs(&run,&outside,&stream).unwrap_err().contains("outside the description"));
}

#[test]
fn the_dots_of_a_step_run_in_parts_of_whole_rows_with_the_same_words() {
    // A step with more vectors than one launch takes: the rows go in parts. Here the limit of a launch is made
    // small, and each result must have the bits of the one launch for all rows.
    let d=small();let w=Made::new(&d,11);let width=d.hidden as usize;let mut n=Numbers::new(12);
    let c=Stand::new();let run=Run::new(&c,&w,&d).unwrap();
    for (count,kept,asked) in [(7usize,3usize,2usize),(5,1,1),(12,4,3),(2,5,7)] {
        let stack=n.words(count*kept*width,3.0);let query=n.words(asked*width,2.0);
        let whole=run.row_dots(&query,&stack,count,kept,prompt_ops::MAX_ROWS).unwrap();assert_eq!(whole.len(),count*asked*kept);
        // Each dot against the hand form: row of the step, then query, then vector.
        let want=(0..count).flat_map(|row|{let (query,stack)=(&query,&stack);(0..asked).flat_map(move |q|(0..kept).map(move |v|hand::dot(&query[q*width..(q+1)*width],&stack[(row*kept+v)*width..(row*kept+v+1)*width])))}).collect::<Vec<_>>();
        assert_eq!(bits(&whole),bits(&want),"count {count} kept {kept} asked {asked}");
        for most in [kept,kept+1,2*kept,3*kept+1,count*kept-1,count*kept] {
            assert_eq!(bits(&run.row_dots(&query,&stack,count,kept,most).unwrap()),bits(&whole),"count {count} kept {kept} asked {asked}, {most} vectors a launch");
        }
        assert!(run.row_dots(&query,&stack,count,kept,kept-1).is_err());
    }
}
#[test]
fn the_dots_of_every_mix_shape_are_the_chain_of_a_plain_product() {
    // `Compute::dot_rows` against the plain product chain (`Op::Project`: the one-row form that the device fold entry is
    // swept against): every dot of every count of stack rows from 1 to 16, for 1, 3 and 48 query rows, bit for bit.
    // The device runs the fold entry for one query row and 2 to 16 stack rows; its words must be these.
    let d=small();let width=d.hidden as usize;let mut n=Numbers::new(21);
    for queries in [1usize,3,48] {for count in 1..=16usize {
        let (query,stack)=(n.words(queries*width,2.0),n.words(count*width,3.0));
        let got=Host.dot_rows(&query,&stack,count,width).unwrap();assert_eq!(got.len(),queries*count);
        for q in 0..queries {
            let mut s=shape(Op::Project);s.rows=count as u32;s.cols=width as u32;
            let plain=Host.operation(s,[Some(&query[q*width..(q+1)*width].to_vec()),None,None,None],&stack).unwrap();
            assert_eq!(bits(&got[q*count..(q+1)*count]),bits(&plain),"queries {queries} count {count} query {q}");
        }
    }}
}
#[test]
fn a_mix_of_a_step_on_the_list_form_has_the_words_of_one_row_at_a_time() {
    // The host rows entry takes the list form (`block_rows::LIST`): the block list of a step as it is made, no swap.
    // Each row of the step must have the bits of the mix of that row alone, for several counts of rows, vectors
    // and queries, and the two arms must run on the list form (no swap: `PermuteRows` would be a prompt launch).
    let d=small();let w=Made::new(&d,31);let width=d.hidden as usize;let mut n=Numbers::new(32);
    let run=Run::new(&Host,&w,&d).unwrap();
    for (count,kept,asked) in [(2usize,1usize,1usize),(3,2,1),(5,4,3),(7,3,2),(12,6,4)] {
        let list=(0..kept).map(|_|n.words(count*width,3.0)).collect::<Vec<_>>();let query=n.words(asked*width,2.0);
        let joined=list.concat();
        let listed=run.blend_listed(&query,&joined,count,kept).unwrap().expect("the host has the list form");
        assert_eq!(bits(&run.blend(&query,&list.iter().collect::<Vec<_>>(),count).unwrap()),bits(&listed),"count {count} kept {kept} asked {asked}");
        for row in 0..count {
            let own=list.iter().map(|entry|entry[row*width..(row+1)*width].to_vec()).collect::<Vec<_>>();
            let alone=run.blend(&query,&own.iter().collect::<Vec<_>>(),1).unwrap();
            assert_eq!(bits(&listed[row*asked*width..(row+1)*asked*width]),bits(&alone),"count {count} kept {kept} asked {asked} row {row}");
        }
    }
}
