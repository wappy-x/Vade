//! CHAT ROWS on the host: for five or more chats a latent layer's absorbed attention runs as ONE row group
//! (`Compute::latent_absorption_chats`), each chat on its own rows of one latent value. The host compute has no absorbed
//! weights, so no host test reached that path. This compute answers the three absorbed calls with ONE stand-in function of
//! a row's own query and its own selected latent rows, laid out as the device lays them out (chat i's Gather at row
//! i x stride, a selection table of row IDs and a count a chat). The batch step and the chat alone then agree bit for bit
//! only if `Decoder::chat_rows` hands every chat its own query, its own latent rows, its own count and its own place in
//! the output. What this does NOT run: the absorbed kernels and their arithmetic (a GPU: `latent-absorb chats` and the
//! check tool's rows against the chat alone).
use spark_tp::bounded_decoder::{ChatGather,Compute,Decoder,Host,chat_attention_layers};
use engine_model::{plan::{admission::CheckpointAdmission,description::TokenMixer},safetensors::Dtype};
use spark_gpu_probe::{causal_ops::Shape,prompt_ops::Shape as RowShape};
use std::cell::Cell;
#[allow(dead_code,reason="the shared fixture has more forms than this test uses")]
#[path="support/attention_checkpoint.rs"] mod fixture;
type V=Vec<f32>;
/// The stand-in for one row's absorbed attention: every word of the row's query and of its `live` latent rows enters
/// every output word, in one fixed order. Finite for finite inputs. No live row: zeros, as the decode step gives.
fn attend(heads:usize,key:usize,value:usize,width:usize,query:&[f32],rows:&[f32],live:usize)->V {
    assert_eq!(query.len(),heads*key,"a row's query is its own heads x key words");assert!(rows.len()>=live*width);
    if live==0 {return vec![0.0;heads*value];}
    (0..heads*value).map(|at| {
        let (head,feature)=(at/value,at%value);let mut sum=0.0f32;
        for row in 0..live {for word in 0..width {sum+=query[head*key+(row+word+feature)%key]*rows[row*width+word];}}
        spark_gpu_probe::bf16::rounded((sum/(live*width) as f32).clamp(-4.0,4.0))
    }).collect()
}
/// A row group as the device reads it: token t's `selected` row holds `slots` row IDs of the latent value, then its count.
fn attend_rows(s:RowShape,query:&[f32],latent:&[f32],selected:&[f32])->V {
    let (tokens,heads,key,value,width,slots)=(s.tokens as usize,s.heads as usize,s.dim as usize,s.taps as usize,s.cols as usize,s.slots as usize);
    assert_eq!((query.len(),selected.len()),(tokens*heads*key,tokens*(slots+1)));
    (0..tokens).flat_map(|token| {
        let table=&selected[token*(slots+1)..(token+1)*(slots+1)];let live=table[slots] as usize;assert!(live<=slots);
        let rows=table[..live].iter().flat_map(|&id|latent[id as usize*width..(id as usize+1)*width].iter().copied()).collect::<V>();
        attend(heads,key,value,width,&query[token*heads*key..(token+1)*heads*key],&rows,live)
    }).collect()
}
/// The host compute with the absorbed path open. Counts the attention launches of each kind.
#[derive(Default)]
struct Absorbing {
    /// Absorbed attentions of one chat alone (`latent_absorption`), row groups of several chats, and the chats in them.
    alone:Cell<usize>,groups:Cell<usize>,grouped_chats:Cell<usize>,
    /// The largest count of selected rows a row group saw: the test must not pass on empty selections alone.
    most:Cell<usize>,
    /// A chat's attended rows land one slot late in the group's output: the fault the test must catch.
    shifted:bool}
impl Compute for Absorbing {
    type Value=V;type Pass=<Host as Compute>::Pass;
    fn mhc_row_launches(&self)->u64 {0}
    fn rows_greedy(&self,_:&V,_:usize)->spark_tp::Result<Option<Vec<u32>>> {Ok(None)}
    fn has_latent_absorption(&self,_:&str)->bool {true}
    fn gather_absorbed(&self,s:Shape,x:[Option<&V>;4],reserve:usize,expected:f32)->spark_tp::Result<V> {Host.gather_absorbed(s,x,reserve,expected)}
    fn latent_absorption(&self,_:&str,s:Shape,q:&V,l:&V,_:&V,_:usize)->spark_tp::Result<V> {
        self.alone.set(self.alone.get()+1);
        Ok(attend(s.heads as usize,s.dim as usize,s.taps as usize,s.cols as usize,q,l,s.rows as usize))
    }
    fn latent_absorption_rows(&self,_:&str,s:RowShape,q:&V,l:&V,n:&V)->spark_tp::Result<V> {Ok(attend_rows(s,q,l,n))}
    fn latent_absorption_chats(&self,_:&str,s:RowShape,gathers:&[ChatGather<'_,V>],stride:usize,q:&V,selected:&V)->spark_tp::Result<Option<V>> {
        let (chats,width)=(gathers.len(),s.cols as usize);
        assert_eq!((s.tokens as usize,s.rows as usize,s.first as usize,s.slots as usize),(chats,chats*stride,stride,stride),"the group's shape names every chat's rows");
        // The device form: chat i's Gather writes its rows at row i x stride of one latent value. A word no Gather wrote is a NaN.
        let mut whole=vec![f32::NAN;chats*stride*width];
        for (chat,(shape,x,expected)) in gathers.iter().enumerate() {
            assert!(shape.dim as usize==width && shape.cols as usize<=stride);
            let rows=Host.gather_absorbed(*shape,*x,stride*width,*expected)?;assert!(rows.len()<=stride*width);
            whole[chat*stride*width..chat*stride*width+rows.len()].copy_from_slice(&rows);
            // The table names this chat's rows and its count, and no other chat's.
            let table=&selected[chat*(stride+1)..(chat+1)*(stride+1)];
            assert_eq!(table[stride].to_bits(),expected.to_bits(),"chat {chat}: the table holds the count of its own context");
            assert!(table[..stride].iter().enumerate().all(|(slot,&id)|id==(chat*stride+slot) as f32),"chat {chat}: the table names its own rows");
            self.most.set(self.most.get().max(*expected as usize));
        }
        self.groups.set(self.groups.get()+1);self.grouped_chats.set(self.grouped_chats.get()+chats);
        let mut out=attend_rows(s,q,&whole,selected);
        if self.shifted {let each=out.len()/chats;out.rotate_right(each);}
        Ok(Some(out))
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
    fn prompt(&self,s:RowShape,x:[Option<&V>;5],w:&[f32])->spark_tp::Result<V> {Host.prompt(s,x,w)}
    fn rows_norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&V,m:usize,w:&[f32])->spark_tp::Result<V> {Host.rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&spark_tp::layer_operands::ResidualBlockOperands,x:&V,m:usize)->spark_tp::Result<(Self::Pass,V)> {Host.rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:Self::Pass,x:&V,m:usize)->spark_tp::Result<V> {Host.rows_finish(p,x,m)}
}
/// A run of batch steps whose cohort changes: 1 to 8 chats, down again and up again, the chats rotating so that a chat
/// leaves, returns, and stands in another slot at another position. Every chat keeps the logit bits, the ID and the next
/// state of its own plain step. The attention launches of each step are counted: five or more chats take ONE row group
/// a latent layer (one exchange); fewer chats keep one attention a chat.
#[test]
fn a_changing_cohort_shares_one_attention_group_a_layer_and_each_chat_keeps_its_own_bits() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let latent=c.admission.description().layers.iter().filter(|layer|matches!(layer.mixer,TokenMixer::LatentAttention(_))).count();assert!(latent>0);
    let (grouping,single)=(Absorbing::default(),Absorbing::default());
    let batch=Decoder::new(&c.admission,&grouping,80).unwrap();let scalar=Decoder::new(&c.admission,&single,80).unwrap();
    let mut actual=(0..8).map(|_|batch.zero_state().unwrap()).collect::<Vec<_>>();
    let mut expected=(0..8).map(|_|scalar.zero_state().unwrap()).collect::<Vec<_>>();let mut tokens=[1u32,2,3,4,5,6,7,1];
    let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    let (mut grouped_steps,mut widths)=(0usize,std::collections::BTreeSet::new());let counted=chat_attention_layers();
    for (step,width) in (1..=8).chain((1..8).rev()).chain(1..=8).enumerate() {
        let chats=(0..width).map(|i|(i+step)%8).rev().collect::<Vec<_>>();
        let rows=chats.iter().map(|&chat|(tokens[chat],&actual[chat])).collect::<Vec<_>>();
        for count in [&grouping.alone,&grouping.groups,&grouping.grouped_chats] {count.set(0);}
        let outputs=batch.step_batch(&rows).unwrap();
        // The launch plan of the step: one row group a latent layer from five chats on (five kernels and one exchange a
        // layer, whatever the chat count), else one attention a chat (five kernels and one exchange a chat).
        let plan=(grouping.groups.get(),grouping.grouped_chats.get(),grouping.alone.get());
        assert_eq!(plan,if width>=5 {(latent,width*latent,0)} else {(0,0,width*latent)},"step={step} width={width}");
        if width>=5 {grouped_steps+=1;}widths.insert(width);
        for (&chat,(logits,id,next)) in chats.iter().zip(outputs) {
            let (want,want_id,want_next)=scalar.step(tokens[chat],&expected[chat],|_,_|Ok(())).unwrap();
            assert_eq!(id,want_id,"step={step} width={width} chat={chat}");assert_eq!(bits(&logits),bits(&want),"step={step} width={width} chat={chat}");
            actual[chat]=next;expected[chat]=want_next;tokens[chat]=id;
        }
    }
    assert_eq!(widths.into_iter().collect::<Vec<_>>(),(1..=8).collect::<Vec<_>>());
    assert!(grouping.most.get()>1,"the groups selected {} rows at most: the run must attend to real rows",grouping.most.get());
    assert_eq!(chat_attention_layers()-counted,(grouped_steps*latent) as u64,"the served counter counts one a grouped layer");
    // The chats stood at different positions inside one group (a chat that sat out is behind the others).
    assert!(actual.iter().map(|state|state.position).collect::<std::collections::BTreeSet<_>>().len()>1);
    // The test has teeth: a group that hands each chat the next chat's attended rows does not give the chats' own bits.
    let wrong=Absorbing {shifted:true,..Absorbing::default()};
    let decoder=Decoder::new(&c.admission,&wrong,80).unwrap();
    let rows=(0..6).map(|chat|(tokens[chat],&expected[chat])).collect::<Vec<_>>();
    let shifted=decoder.step_batch(&rows).unwrap();assert_eq!(wrong.groups.get(),latent);
    let own=(0..6).map(|chat|bits(&scalar.step(tokens[chat],&expected[chat],|_,_|Ok(())).unwrap().0)).collect::<Vec<_>>();
    assert!(shifted.iter().zip(&own).any(|((logits,_,_),want)|&bits(logits)!=want),"the control has no teeth");
}
