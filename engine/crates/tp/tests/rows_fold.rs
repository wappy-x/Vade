//! ROWS FOLD: in a row pass whose picks stay on the device, the routed down step waits for its fold, and the compute gets
//! both as one stage with every input of the fold (`Compute::rows_routed_fold`): the device folds each token's rows on its
//! rank and exchanges the folded rows. A compute that refuses the stage gets the down step and the fold as they were, so
//! every chat keeps the bits of its own plain step.
use spark_tp::bounded_decoder::{Compute,Decoder,Host,State,BATCH_ROWS};
use engine_model::{plan::{admission::CheckpointAdmission,layer::{Kind,Step}},safetensors::Dtype};
use spark_gpu_probe::causal_ops::{Op,Shape};
#[allow(dead_code,reason="the shared fixture has more forms than this test uses")]
#[path="support/attention_checkpoint.rs"] mod fixture;
type V=Vec<f32>;
/// The host compute as a device with its exchange attached: a batch step of 2 to 8 chats keeps its picks on the device
/// (`routes_rows`) and folds its rows (`folds_rows`). The routed projections themselves are refused (the default), so the
/// row pass makes the middle on the host path, in (token, pick) order. Each fold stage is checked and counted. `answer`
/// true answers it with zeros, false refuses it (the caller keeps the two steps).
#[derive(Default)]
struct Folding {with_shared:std::cell::Cell<usize>,without_shared:std::cell::Cell<usize>,rows:std::cell::Cell<usize>,answer:bool,
    /// The shared rows stay on their rank (`rows_project_local`): the compute answers with `LOCAL` words a token, and counts.
    local:bool,kept:std::cell::Cell<usize>,
    /// The compute does not fold (`folds_rows` false: a batch step in the grouped expert form, a verify pass, a prompt group).
    closed:bool,
    /// Times the row pass asked `folds_rows`: the one flag a routed layer pays.
    flags:std::cell::Cell<usize>,
    /// The fold launches the device plans for the stages of this compute, on one rank of a pair at the model's shape
    /// (`rows_fold::one_launch`, the rule `apply_routed_rows_fold` reads). The plan, not a device run.
    launches:std::cell::Cell<usize>}
/// The model's fold on one rank of a pair: 8 picks, 2,048 of the 4,096 rows, with the shared row kept on its rank.
fn model_fold_launches(m:usize)->usize {
    use spark_tp::resident_weights::rows_fold;
    let fold=Shape {op:Op::ExpertFold,rows:8,cols:2048,taps:1,heads:0,dim:0,epsilon:0.0,bound:0.0};
    rows_fold::launches(rows_fold::one_launch(fold,m,Some(m*2048),2048).is_some(),m)
}
/// The mark of a shared row kept on its rank. A host compute has no rows of its own, so the words are a mark, not rows.
const LOCAL:f32=7.0;
impl Compute for Folding {
    type Value=V;type Pass=<Host as Compute>::Pass;
    fn mhc_row_launches(&self)->u64 {0}
    fn rows_greedy(&self,_:&Self::Value,_:usize)->spark_tp::Result<Option<Vec<u32>>> {Ok(None)}
    fn routes_rows(&self,m:usize)->bool {(2..=BATCH_ROWS).contains(&m)}
    fn latent_absorption_chats(&self,name:&str,s:spark_gpu_probe::prompt_ops::Shape,g:&[spark_tp::bounded_decoder::ChatGather<'_,Self::Value>],stride:usize,
        q:&Self::Value,n:&Self::Value)->spark_tp::Result<Option<Self::Value>> {Host.latent_absorption_chats(name,s,g,stride,q,n)}
    fn folds_rows(&self,m:usize)->bool {self.flags.set(self.flags.get()+1);!self.closed && self.routes_rows(m)}
    fn rows_routed_fold(&self,down:&Step,ids:&V,input:&V,weights:&V,shared:Option<&V>,fold:Shape,m:usize)->spark_tp::Result<Option<V>> {
        // What the device form needs: the one down step, every token's picks and weights, one middle row a (token, pick),
        // one gathered shared row a token, and the fold shape of one token's whole output.
        let picks=fold.rows as usize;
        assert_eq!((down.kind,fold.op),(Kind::ExpertDown,Op::ExpertFold));assert!(picks>0 && m>0);
        assert_eq!((ids.len(),weights.len()),(m*picks,m*picks));
        assert_eq!(input.len(),m*picks*down.weights[0].shape[1] as usize,"the middle is one row a (token, pick)");
        assert_eq!(fold.cols,down.weights[0].shape[0] as u32,"the fold makes one whole output row a token");
        assert_eq!(fold.taps!=0,shared.is_some());
        if let Some(shared)=shared {
            if self.local {assert!(shared.len()==3*m && shared.iter().all(|&word|word==LOCAL),"the fold must get the rows this rank kept");}
            else {assert_eq!(shared.len(),m*fold.cols as usize,"the shared row is one gathered row a token");}
        }
        let count=if shared.is_some() {&self.with_shared} else {&self.without_shared};count.set(count.get()+1);self.rows.set(self.rows.get()+m);
        self.launches.set(self.launches.get()+model_fold_launches(m));
        Ok(self.answer.then(||vec![0.0;m*fold.cols as usize]))
    }
    fn rows_project_local(&self,_:&CheckpointAdmission,_:&str,x:&V,m:usize,_:bool)->spark_tp::Result<Option<V>> {
        if !self.local {return Ok(None);}
        assert!(m>0 && !x.is_empty() && x.len().is_multiple_of(m));self.kept.set(self.kept.get()+1);Ok(Some(vec![LOCAL;3*m]))
    }
    fn select_known(&self,ids:&V,count:&V,row:&[f32])->spark_tp::Result<()> {Host.select_known(ids,count,row)}
    fn prompt_end(&self)->spark_tp::Result<()> {Ok(())} fn prompt_seeds(&self)->bool {false}
    fn prepare_beside(&self,c:&CheckpointAdmission,_:usize)->spark_tp::Result<()> {self.prepare(c)}
    fn upload(&self,v:&[f32])->spark_tp::Result<V> {Host.upload(v)}
    fn read(&self,v:&V)->spark_tp::Result<V> {Host.read(v)}
    fn words(&self,v:&V)->usize {v.len()}
    fn range(&self,v:&V,start:usize,len:usize)->spark_tp::Result<V> {Host.range(v,start,len)}
    fn join(&self,v:&[&V])->spark_tp::Result<V> {Host.join(v)}
    fn zeros(&self,n:usize)->spark_tp::Result<V> {Host.zeros(n)}
    fn operation(&self,s:Shape,x:[Option<&V>;4],w:&[f32])->spark_tp::Result<V> {Host.operation(s,x,w)}
    fn norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&V,w:&[f32])->spark_tp::Result<V> {Host.norm(name,s,x,w)}
    fn begin(&self,key:(usize,usize),p:&spark_tp::layer_operands::ResidualBlockOperands,x:&V)->spark_tp::Result<(Self::Pass,V)> {Host.begin(key,p,x)}
    fn finish(&self,p:Self::Pass,x:&V)->spark_tp::Result<V> {Host.finish(p,x)}
    fn mx(&self,p:&spark_tp::mxfp8_projection::Mxfp8Projection<'_>,x:&V)->spark_tp::Result<V> {Host.mx(p,x)}
    fn round(&self,x:&V)->spark_tp::Result<V> {Host.round(x)}
    fn mean(&self,x:&V,s:u32,h:u32)->spark_tp::Result<V> {Host.mean(x,s,h)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&V>;5],w:&[f32])->spark_tp::Result<V> {Host.prompt(s,x,w)}
    fn rows_norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&V,m:usize,w:&[f32])->spark_tp::Result<V> {Host.rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&spark_tp::layer_operands::ResidualBlockOperands,x:&V,m:usize)->spark_tp::Result<(Self::Pass,V)> {Host.rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:Self::Pass,x:&V,m:usize)->spark_tp::Result<V> {Host.rows_finish(p,x,m)}
    fn has_latent_absorption(&self,_:&str)->bool {false}
    fn gather_absorbed(&self,s:Shape,x:[Option<&V>;4],reserve:usize,expected:f32)->spark_tp::Result<V> {Host.gather_absorbed(s,x,reserve,expected)}
    fn latent_absorption(&self,name:&str,s:Shape,q:&V,l:&V,n:&V,slots:usize)->spark_tp::Result<V> {Host.latent_absorption(name,s,q,l,n,slots)}
    fn latent_absorption_rows(&self,name:&str,s:spark_gpu_probe::prompt_ops::Shape,q:&V,l:&V,n:&V)->spark_tp::Result<V> {Host.latent_absorption_rows(name,s,q,l,n)}
}
/// Each chat's state after its own prompt (chats at different positions), read by the plain step of `decoder`.
fn states<C:Compute<Value=V>>(decoder:&Decoder<'_,C>,prompts:&[&[u32]])->Vec<State<V>> {
    prompts.iter().map(|prompt| {
        let mut state=decoder.zero_state().unwrap();
        for &token in *prompt {state=decoder.step(token,&state,|_,_|Ok(())).unwrap().2;}
        state
    }).collect()
}
#[test]
fn a_batch_step_folds_its_routed_down_rows_at_the_fold_step_or_keeps_both_steps() {
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    let routed=c.admission.description().layers.iter().filter(|layer|matches!(layer.ffn,engine_model::shapes::Ffn::Experts(_))).count();assert!(routed>0);
    // Three chats at three positions. Two chats ride the row forms too (`ROUTED_CHATS` is 1): the cohorts below start at two.
    let prompts:[&[u32];3]=[&[1],&[7,2],&[3,4,5]];let tokens=[2u32,5,1];
    let plain=Decoder::new(&c.admission,Host,8).unwrap();
    let expected=states(&plain,&prompts).iter().zip(tokens).map(|(state,token)|bits(&plain.step(token,state,|_,_|Ok(())).unwrap().0)).collect::<Vec<_>>();
    let batch=|compute:&Folding| {
        let decoder=Decoder::new(&c.admission,compute,8).unwrap();let held=states(&decoder,&prompts);
        // The prompts ran as plain steps: no row pass, so no fold stage yet.
        assert_eq!((compute.with_shared.get(),compute.without_shared.get()),(0,0));
        let rows=tokens.iter().zip(&held).map(|(&token,state)|(token,state)).collect::<Vec<_>>();
        decoder.step_batch(&rows).unwrap().into_iter().map(|(logits,_,_)|bits(&logits)).collect::<Vec<_>>()
    };
    for local in [false,true] {
        // The stage refused: the down step runs at the fold step on the old path, and every chat keeps the bits of its own
        // step. With the shared rows kept on their rank, the refused fold's shared row is made again with its exchange.
        let refused=Folding {local,..Folding::default()};
        assert_eq!(batch(&refused),expected,"shared rows kept local: {local}");
        assert_eq!(refused.without_shared.get(),0,"a fold stage was asked without its shared row");
        assert_eq!((refused.with_shared.get(),refused.rows.get()),(routed,3*routed),"every routed down step of the batch step must reach its fold as one stage of all rows");
        assert_eq!(refused.kept.get(),if local {routed} else {0},"one shared projection a routed layer stays on its rank");
        assert_eq!(refused.flags.get(),routed,"the row pass asks the compute once a routed layer whether it folds");
        // The stage answered: its output is the layer's expert output for every chat (the fold step is not run again).
        let answered=Folding {answer:true,local,..Folding::default()};
        let actual=batch(&answered);
        assert_eq!((answered.with_shared.get(),answered.kept.get(),answered.flags.get()),(routed,if local {routed} else {0},routed));
        for (chat,(got,want)) in actual.iter().zip(&expected).enumerate() {assert_ne!(got,want,"chat {chat}: the folded output was not used");}
    }
    // Cohorts of 2 to 8 chats at eight positions, the chats rotating through the slots: every routed layer is still ONE
    // stage of all rows, each chat keeps the bits of its own step, and the stage's launch plan at the model's shape is
    // ONE fold launch a routed layer on a rank, for any row count (it was one a row). A cohort of two is what a served
    // pair of different chats runs on a device: its picks stay there and its down rows fold before the exchange. With
    // the routed span run once a chat (`ROUTED_CHATS` 2) no fold stage is asked at two rows and this loop fails there.
    let prompts:[&[u32];8]=[&[1],&[7,2],&[3,4,5],&[6],&[2,2,7,1],&[5,3],&[4],&[1,6,6]];let tokens=[2u32,5,1,7,3,6,4,2];
    let alone=states(&plain,&prompts).iter().zip(tokens).map(|(state,token)|bits(&plain.step(token,state,|_,_|Ok(())).unwrap().0)).collect::<Vec<_>>();
    for n in 2..=8usize {
        let compute=Folding {local:true,..Folding::default()};
        let decoder=Decoder::new(&c.admission,&compute,8).unwrap();let held=states(&decoder,&prompts);
        let chats=(0..n).map(|slot|(slot*3+n)%8).collect::<Vec<_>>();
        let mut seen=chats.clone();seen.sort_unstable();seen.dedup();assert_eq!(seen.len(),n,"the cohort names a chat once");
        let rows=chats.iter().map(|&chat|(tokens[chat],&held[chat])).collect::<Vec<_>>();
        let actual=decoder.step_batch(&rows).unwrap();
        for (&chat,(logits,_,_)) in chats.iter().zip(&actual) {assert_eq!(bits(logits),alone[chat],"rows={n} chat={chat}");}
        assert_eq!((compute.with_shared.get(),compute.without_shared.get(),compute.rows.get()),(routed,0,n*routed),"rows={n}");
        assert_eq!(compute.launches.get(),routed,"rows={n}: one fold launch a routed layer on a rank");
    }
    assert_eq!((1..=8).map(model_fold_launches).collect::<Vec<_>>(),[1;8],"1 chat keeps its one launch; 2 to 8 chats share one");
    // A compute that does not fold: the pass pays its one flag a routed layer and nothing else of the fold runs.
    let closed=Folding {closed:true,local:true,answer:true,..Folding::default()};
    assert_eq!(batch(&closed),expected,"a pass that does not fold keeps every chat's bits");
    assert_eq!((closed.flags.get(),closed.kept.get(),closed.with_shared.get(),closed.without_shared.get()),(routed,0,0,0),
        "a pass that does not fold asks one flag a routed layer: no shared rows kept, no fold stage asked");
}
/// ROWS FOLD twin for several tokens (no device): each rank folds its own rows of every token and the folded rows are
/// gathered, against the fold of the gathered rows (main's row pass: the wrapped fold). 3 and 8 tokens, rows that differ
/// by token and pick, picks whose slot order is not their ID order, with and without the shared row, the shared plane
/// gathered or this rank's own. The token ranges are the device path's own (`rows_fold::planes`, which
/// `apply_routed_rows_fold` reads); the folds are the host references of the two kernels. What this does NOT run: the
/// device function, the fold kernel and the mailbox gather themselves. Those need a GPU: the check tool on a pair
/// compares every row's logits with the chat alone at 1 to 8 rows.
#[test]
fn rows_folded_on_each_rank_then_gathered_equal_the_gathered_rows_folded() {
    use spark_gpu_probe::{bf16::rounded,causal_ops::reference as fold_one,prompt_ops::{reference as fold_rows,Shape as RowShape}};
    use spark_tp::resident_weights::{Rows,rows_fold};
    let (picks,hidden,experts)=(8usize,96usize,288usize);
    // Finished BF16 words, as the routed column-tile projection stores them: the fold's cast then changes none.
    let word=|seed:usize|rounded(((seed.wrapping_mul(2654435761))%2001) as f32/250.0-4.0);
    for m in [3usize,8] {for ranks in [2usize,3] {for taps in [0u32,1] {
        let ids=(0..m*picks).map(|at|((at/picks*29+at%picks*101+11)%experts) as f32).collect::<Vec<_>>();
        let weights=(0..m*picks).map(|at|word(at*7+3)/8.0).collect::<Vec<_>>();
        let down=(0..m*picks*hidden).map(|at|word(at+17)).collect::<Vec<_>>();
        let shared=if taps!=0 {(0..m*hidden).map(|at|word(at*5+3)).collect::<Vec<_>>()} else {Vec::new()};
        assert!(down.iter().chain(&shared).all(|&v|rounded(v).to_bits()==v.to_bits()));
        // A token's slot order is not its ID order, and no two tokens hold the same rows.
        for token in 0..m {let row=&ids[token*picks..(token+1)*picks];assert!(row.windows(2).any(|pair|pair[0]>pair[1]));
            assert!((0..picks).all(|pick|!row[..pick].contains(&row[pick])));}
        assert!((1..m).all(|token|down[..picks*hidden]!=down[token*picks*hidden..(token+1)*picks*hidden]));
        let whole=Shape {op:Op::ExpertFold,rows:picks as u32,cols:hidden as u32,taps,heads:0,dim:0,epsilon:0.0,bound:0.0};
        // Gather, then fold: the wrapped fold of the gathered rows, every token.
        let expected=fold_rows(RowShape::wrapped(whole,m).unwrap(),[&ids,&weights,&down,&shared,&[]],&[]).unwrap();
        assert_eq!(expected.len(),m*hidden);
        // Fold, then gather: each rank folds its own rows of each token; the gather puts them at their place in the token's row.
        let count=hidden/ranks;
        for own_plane in [false,true] {
            if taps==0 && own_plane {continue;}
            let mut got=vec![f32::NAN;m*hidden];
            for rank in 0..ranks {
                let rows=Rows {start:rank*count,count,total:hidden,split:true};
                let projected=(0..m*picks).flat_map(|row|down[row*hidden+rows.start..row*hidden+rows.start+count].iter().copied()).collect::<Vec<_>>();
                let plane=if taps==0 {Vec::new()} else if own_plane {(0..m).flat_map(|token|shared[token*hidden+rows.start..token*hidden+rows.start+count].iter().copied()).collect()} else {shared.clone()};
                for token in 0..m {
                    let at=rows_fold::planes(token,m,picks,rows,(taps!=0).then_some(plane.len())).unwrap();
                    let own=at.own.map_or(&plane[..0],|own|&plane[own]);
                    let part=fold_one(Shape {cols:count as u32,..whole},[&ids[at.picked.clone()],&weights[at.picked],&projected[at.projected],own],&[]).unwrap();
                    got[token*hidden+rows.start..token*hidden+rows.start+count].copy_from_slice(&part);
                }
            }
            assert!(expected.iter().map(|v|v.to_bits()).eq(got.iter().map(|v|v.to_bits())),"m={m} ranks={ranks} taps={taps} shared plane of this rank alone={own_plane}");
        }
        // The rows differ by token in the result too: a fold of the wrong token's rows cannot pass.
        assert!((1..m).all(|token|expected[..hidden]!=expected[token*hidden..(token+1)*hidden]));
    }}}
    // Extents outside the two shared forms, a token past the rows, more tokens than a batch step: refused.
    let rows=Rows {start:0,count:48,total:96,split:true};
    assert!(rows_fold::planes(0,3,8,rows,Some(3*96+1)).is_err());assert!(rows_fold::planes(3,3,8,rows,None).is_err());
    assert!(rows_fold::planes(0,rows_fold::TOKENS+1,8,rows,None).is_err());assert!(rows_fold::planes(0,3,0,rows,None).is_err());
    assert_eq!(rows_fold::planes(2,3,8,Rows {start:48,..rows},Some(3*96)).unwrap(),rows_fold::Planes {picked:16..24,projected:2*8*48..3*8*48,own:Some(2*96+48..3*96)});
}
/// FOLD ROWS ONE LAUNCH twin (no device): a run of batch steps whose cohort changes (1 to 8 chats; chats join, leave and
/// move between slots), at the model's shape on a pair (8 picks of 288, 4,096 rows, 2,048 a rank, the shared row kept on
/// its rank). Each rank folds as the device path does: the ONE row-fold launch where `rows_fold::one_launch` gives a
/// shape (its host reference, `prompt_ops::reference`), else the launch a token (`causal_ops::reference` with the round
/// mark, over `rows_fold::planes`). Every chat's gathered row must have the bits of the chat ALONE: the plain step's fold
/// of its own whole rows with the round mark (`fold_then_gather`). A chat's planes depend on the chat and the step only,
/// so a word read from another chat's planes, or from the wrong slot, cannot pass. The fold launches of each step are
/// counted on each rank. What this does NOT run: the device function, the kernel and the gather themselves (a GPU: the
/// list sweep `kernel-bench --reading-probe .. route-fold` and the check tool's rows against the chat alone).
#[test]
fn one_fold_launch_of_a_changing_cohort_gives_each_chat_the_bits_of_the_chat_alone() {
    use spark_gpu_probe::{bf16::rounded,causal_ops::{reference as fold_one,FOLD_ROUNDS},prompt_ops::reference as fold_rows};
    use spark_tp::resident_weights::{Rows,rows_fold};
    let (picks,hidden,experts,ranks,chats)=(8usize,4096usize,288usize,2usize,8usize);let count=hidden/ranks;
    // Finished BF16 words, as the routed column-tile projection stores them.
    let word=|seed:usize|rounded(((seed.wrapping_mul(2654435761))%4001) as f32/500.0-4.0);
    // A chat's planes at a step: its picks (slot order is not ID order, no pick twice), weights, whole rows, shared row.
    let planes_of=|chat:usize,step:usize| {
        let salt=chat*7919+step*104729;
        let ids=(0..picks).map(|pick|((salt*31+pick*101+11)%experts) as f32).collect::<Vec<_>>();
        assert!((0..picks).all(|pick|!ids[..pick].contains(&ids[pick])) && ids.windows(2).any(|pair|pair[0]>pair[1]));
        let weights=(0..picks).map(|pick|word(salt+pick*7+3)/8.0).collect::<Vec<_>>();
        let down=(0..picks*hidden).map(|at|word(salt*13+at+17)).collect::<Vec<_>>();
        let shared=(0..hidden).map(|at|word(salt*5+at*3+1)).collect::<Vec<_>>();
        (ids,weights,down,shared)
    };
    let whole=Shape {op:Op::ExpertFold,rows:picks as u32,cols:hidden as u32,taps:1,heads:0,dim:0,epsilon:0.0,bound:0.0};
    let local=Shape {cols:count as u32,..whole};
    let mut widths=std::collections::BTreeSet::new();let mut total=[0usize;2];
    // Widths 1 to 8, back down, and up again; the cohort rotates so a chat leaves, returns and changes its slot.
    for (step,width) in (1..=chats).chain((1..chats).rev()).chain([3,8,5,2]).enumerate() {
        let cohort=(0..width).map(|slot|(slot*5+step*3)%chats).collect::<Vec<_>>();
        let mut seen=cohort.clone();seen.sort_unstable();seen.dedup();assert_eq!(seen.len(),width);
        let held=cohort.iter().map(|&chat|planes_of(chat,step)).collect::<Vec<_>>();
        // The row-major planes of the batch step: one row a (token, pick), in slot order.
        let ids=held.iter().flat_map(|p|p.0.iter().copied()).collect::<Vec<_>>();
        let weights=held.iter().flat_map(|p|p.1.iter().copied()).collect::<Vec<_>>();
        let mut got=vec![f32::NAN;width*hidden];let mut launches=Vec::with_capacity(ranks);
        for rank in 0..ranks {
            let rows=Rows {start:rank*count,count,total:hidden,split:true};
            // This rank's rows of every (token, pick), and of every token's shared row (`rows_project_local`).
            let projected=held.iter().flat_map(|p|(0..picks).flat_map(move |pick|p.2[pick*hidden+rows.start..pick*hidden+rows.start+count].iter().copied())).collect::<Vec<_>>();
            let own=held.iter().flat_map(|p|p.3[rows.start..rows.start+count].iter().copied()).collect::<Vec<_>>();
            let one=rows_fold::one_launch(Shape {dim:64,..local},width,Some(own.len()),count);
            assert_eq!(one.is_some(),width>=2,"width={width}: 2 to 8 chats of the model's shape take the one launch");
            launches.push(rows_fold::launches(one.is_some(),width));
            let folded=match one {
                Some(shape)=>{assert_eq!((shape.tokens as usize,shape.inner.cols as usize,shape.inner.dim),(width,count,0));
                    fold_rows(shape,[&ids,&weights,&projected,&own,&[]],&[]).unwrap()},
                None=>(0..width).flat_map(|token| {
                    let at=rows_fold::planes(token,width,picks,rows,Some(own.len())).unwrap();
                    fold_one(Shape {dim:FOLD_ROUNDS,..local},[&ids[at.picked.clone()],&weights[at.picked],&projected[at.projected],&own[at.own.unwrap()]],&[]).unwrap()
                }).collect(),
            };
            assert_eq!(folded.len(),width*count);
            // The gather: this rank's folded row of token t at its rows of token t's whole row.
            for token in 0..width {got[token*hidden+rows.start..token*hidden+rows.start+count].copy_from_slice(&folded[token*count..(token+1)*count]);}
        }
        assert_eq!(launches,[1,1],"step={step} width={width}: one fold launch a rank");
        for (sum,made) in total.iter_mut().zip(&launches) {*sum+=made;}
        for (slot,(&chat,planes)) in cohort.iter().zip(&held).enumerate() {
            let alone=fold_one(Shape {dim:FOLD_ROUNDS,..whole},[&planes.0,&planes.1,&planes.2,&planes.3],&[]).unwrap();
            assert!(alone.iter().map(|v|v.to_bits()).eq(got[slot*hidden..(slot+1)*hidden].iter().map(|v|v.to_bits())),"step={step} width={width} slot={slot} chat={chat}");
        }
        widths.insert(width);
    }
    assert_eq!(widths.into_iter().collect::<Vec<_>>(),(1..=chats).collect::<Vec<_>>());assert_eq!(total[0],total[1]);
    // The one launch on a rank of a pair and of four. The forms that keep a launch a token: the gathered shared plane, a rank count or a fold the sweep does not hold.
    let own=|m:usize|Some(m*count);
    for m in 2..=chats {
        assert!(rows_fold::one_launch(local,m,own(m),count).is_some());
        assert!(rows_fold::one_launch(local,m,Some(m*hidden),count).is_none(),"the gathered shared plane holds a token's rows apart");
        assert!(rows_fold::one_launch(Shape {taps:0,..local},m,None,count).is_none() && rows_fold::one_launch(local,m,None,count).is_none());
        assert!(rows_fold::one_launch(Shape {cols:1024,..local},m,Some(m*1024),1024).is_some(),"a rank of four takes the one launch: its 1,024 rows are swept");
        assert!(rows_fold::one_launch(Shape {cols:512,..local},m,Some(m*512),512).is_none(),"a rank of eight is not swept");
        assert!(rows_fold::one_launch(Shape {rows:7,..local},m,own(m),count).is_none() && rows_fold::one_launch(local,m,own(m),count+1).is_none());
        assert!(rows_fold::one_launch(Shape {op:Op::Route,..local},m,own(m),count).is_none());
        assert_eq!((rows_fold::launches(true,m),rows_fold::launches(false,m)),(1,m));
    }
    assert!(rows_fold::one_launch(local,1,own(1),count).is_none() && rows_fold::one_launch(local,rows_fold::TOKENS+1,own(rows_fold::TOKENS+1),count).is_none());
    // No row reads another row: with every other chat's planes replaced (other IDs, NaN rows), a chat's folded words stay.
    let width=8;let held=(0..width).map(|chat|planes_of(chat,99)).collect::<Vec<_>>();
    let build=|poison:Option<usize>| {
        let rows=Rows {start:count,count,total:hidden,split:true};let at=|chat:usize|poison.is_some_and(|keep|keep!=chat);
        let ids=held.iter().enumerate().flat_map(|(chat,p)|p.0.iter().map(move |&id|if at(chat) {(experts-1) as f32-id} else {id})).collect::<Vec<_>>();
        let weights=held.iter().enumerate().flat_map(|(chat,p)|p.1.iter().map(move |&w|if at(chat) {-w} else {w})).collect::<Vec<_>>();
        let projected=held.iter().enumerate().flat_map(|(chat,p)|(0..picks).flat_map(move |pick|p.2[pick*hidden+rows.start..pick*hidden+rows.start+count].iter().map(move |&v|if at(chat) {f32::NAN} else {v}))).collect::<Vec<_>>();
        let own=held.iter().enumerate().flat_map(|(chat,p)|p.3[rows.start..rows.start+count].iter().map(move |&v|if at(chat) {f32::NAN} else {v})).collect::<Vec<_>>();
        fold_rows(rows_fold::one_launch(local,width,Some(own.len()),count).unwrap(),[&ids,&weights,&projected,&own,&[]],&[]).unwrap()
    };
    let clean=build(None);
    for keep in [0usize,3,7] {let poisoned=build(Some(keep));
        assert!(clean[keep*count..(keep+1)*count].iter().map(|v|v.to_bits()).eq(poisoned[keep*count..(keep+1)*count].iter().map(|v|v.to_bits())),"chat {keep} read another chat's planes");
        assert!((0..width).filter(|&chat|chat!=keep).all(|chat|poisoned[chat*count..(chat+1)*count].iter().all(|v|v.is_nan())),"the poison did not reach its own rows");}
    // Why the premise matters (negative control): on words that are NOT finished BF16 the launch a token (it casts each
    // row word first) and the row fold (it reads the word as it lies) differ. ROUTE ROWS makes finished words only.
    let raw=(0..picks*count).map(|at|((at*37+5)%1013) as f32/977.0-0.5).collect::<Vec<_>>();
    assert!(raw.iter().any(|&v|rounded(v).to_bits()!=v.to_bits()));
    let (ids,weights,_,shared)=planes_of(0,0);
    let cast=fold_one(Shape {dim:FOLD_ROUNDS,..local},[&ids,&weights,&raw,&shared[..count]],&[]).unwrap();
    let lies=fold_one(local,[&ids,&weights,&raw,&shared[..count]],&[]).unwrap();
    assert!(cast.iter().zip(&lies).any(|(a,b)|a.to_bits()!=b.to_bits()),"the control has no teeth");
}
