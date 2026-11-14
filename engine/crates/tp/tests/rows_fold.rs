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
/// The host compute as a device with its exchange attached: a batch step of 3 to 8 chats keeps its picks on the device
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
    flags:std::cell::Cell<usize>}
/// The mark of a shared row kept on its rank. A host compute has no rows of its own, so the words are a mark, not rows.
const LOCAL:f32=7.0;
impl Compute for Folding {
    type Value=V;type Pass=<Host as Compute>::Pass;
    fn mhc_row_launches(&self)->u64 {0}
    fn rows_greedy(&self,_:&Self::Value,_:usize)->spark_tp::Result<Option<Vec<u32>>> {Ok(None)}
    fn routes_rows(&self,m:usize)->bool {(3..=BATCH_ROWS).contains(&m)}
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
    // Three chats at three positions: the fewest that ride the row forms (two chats alone take the plain step's path).
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
