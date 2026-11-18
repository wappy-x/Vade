//! Forward borrowed compute calls through the same implementation.
use super::*;
impl<C:Compute> Compute for &C {
    fn work<T>(&self,run:impl FnOnce()->Result<T>)->Result<T> {(**self).work(run)}
    fn resident_weights_ready(&self)->bool {(**self).resident_weights_ready()}
    fn memo_summary(&self,c:&CheckpointAdmission)->Option<crate::projection_memo::Summary> {(**self).memo_summary(c)}
    fn freeze_memo(&self,p:crate::projection_memo::Plan)->Result<crate::projection_memo::Plan> {(**self).freeze_memo(p)}
    fn capture_counts(&self)->[u64;4] {(**self).capture_counts()}
    fn fatal_error(&self)->Option<String> {(**self).fatal_error()}
    fn memory_bytes(&self)->Option<(usize,usize,usize)> {(**self).memory_bytes()}
    fn vector_allocations(&self)->Option<(usize,usize)> {(**self).vector_allocations()}
    fn vector_pool_counts(&self)->Option<(usize,usize,usize)> {(**self).vector_pool_counts()}
    type Value=C::Value;type Pass=C::Pass;
    fn state_allocations(&self,v:&C::Value,out:&mut Vec<(usize,usize)>)->bool {(**self).state_allocations(v,out)}
    fn plan_cost(&self,name:&str,ns:u128) {(**self).plan_cost(name,ns)}
    fn serving_trace(&self)->bool {(**self).serving_trace()}
    fn profile_tick_begin(&self,tick:u64) {(**self).profile_tick_begin(tick)}
    fn profile_tick_end(&self) {(**self).profile_tick_end()}
    fn trace_start(&self) {(**self).trace_start()}
    fn cost_report(&self,phase:&str) {(**self).cost_report(phase)}
    fn mhc_row_launches(&self)->u64 {(**self).mhc_row_launches()}
    fn projects_complete_rows(&self)->bool {(**self).projects_complete_rows()}
    fn rows_greedy(&self,logits:&C::Value,rows:usize)->Result<Option<Vec<u32>>> {(**self).rows_greedy(logits,rows)}
    fn greedy_projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value)->Result<u32> {(**self).greedy_projection(c,name,x)}
    fn greedy_projection_mass(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value)->Result<(u32,Option<f32>)> {(**self).greedy_projection_mass(c,name,x)}
    fn sparse_topology(&self)->Topology {(**self).sparse_topology()}
    fn owned_expansion(&self,name:&str,input:&C::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<Option<C::Value>> {(**self).owned_expansion(name,input,tokens,layout)}
    fn owned_expansion_rows(&self,name:&str,input:&C::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<Option<C::Value>> {(**self).owned_expansion_rows(name,input,tokens,layout)}
    fn gather_attended(&self,input:&C::Value,tokens:usize,layout:crate::sparse_heads::Layout)->Result<C::Value> {(**self).gather_attended(input,tokens,layout)}
    fn embedding(&self,token:u32,streams:usize)->Result<Option<C::Value>> {(**self).embedding(token,streams)}
    fn indirect_attention(&self,shape:Shape,parts:&[&C::Value],query:&C::Value)->Result<Option<C::Value>> {(**self).indirect_attention(shape,parts,query)}
    fn has_latent_absorption(&self,name:&str)->bool {(**self).has_latent_absorption(name)}
    fn latent_absorption(&self,name:&str,shape:Shape,query:&C::Value,latent:&C::Value,count:&C::Value,slots:usize)->Result<C::Value> {(**self).latent_absorption(name,shape,query,latent,count,slots)}
    fn latent_absorption_rows(&self,name:&str,shape:spark_gpu_probe::prompt_ops::Shape,query:&C::Value,latent:&C::Value,selected:&C::Value)->Result<C::Value> {(**self).latent_absorption_rows(name,shape,query,latent,selected)}
    fn latent_absorption_chats(&self,name:&str,shape:spark_gpu_probe::prompt_ops::Shape,gathers:&[crate::bounded_decoder::ChatGather<'_,C::Value>],stride:usize,
        query:&C::Value,selected:&C::Value)->Result<Option<C::Value>> {(**self).latent_absorption_chats(name,shape,gathers,stride,query,selected)}
    fn planar_kda(&self,shape:Shape)->bool {(**self).planar_kda(shape)}
    fn prepare(&self,c:&engine_model::plan::admission::CheckpointAdmission)->Result<()> {(**self).prepare(c)}
    fn prepare_beside(&self,c:&engine_model::plan::admission::CheckpointAdmission,live:usize)->Result<()> {(**self).prepare_beside(c,live)}
    fn prepare_names(&self,c:&engine_model::plan::admission::CheckpointAdmission,names:&[String])->Result<()> {(**self).prepare_names(c,names)}
    fn projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value,round:bool)->Result<Option<C::Value>> {(**self).projection(c,name,x,round)}
    fn projection_rows(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&[&C::Value],round:bool)->Result<Option<Vec<C::Value>>> {(**self).projection_rows(c,name,x,round)}
    fn projection_batch(&self,c:&CheckpointAdmission,input:&[(&str,&C::Value)],round:bool)->Result<Option<Vec<C::Value>>> {(**self).projection_batch(c,input,round)}
    fn projection_stage(&self,c:&CheckpointAdmission,jobs:&[crate::projection_stage::Job<'_,C::Value>])->Result<Option<Vec<C::Value>>> {(**self).projection_stage(c,jobs)}
    fn routed_experts(&self,steps:&[engine_model::plan::layer::Step],ids:&C::Value,input:&C::Value)->Result<Option<Vec<C::Value>>> {(**self).routed_experts(steps,ids,input)}
    fn routed_fold(&self,steps:&[engine_model::plan::layer::Step],ids:&C::Value,input:&C::Value,weights:&C::Value,shared:Option<&C::Value>,
        fold:spark_gpu_probe::causal_ops::Shape)->Result<Option<C::Value>> {(**self).routed_fold(steps,ids,input,weights,shared,fold)}
    fn folds_routed(&self)->bool {(**self).folds_routed()}
    fn projection_origin(&self,name:&str,x:&C::Value,round:bool)->Result<Option<C::Value>> {(**self).projection_origin(name,x,round)}
    fn projection_origins(&self,name:&str,rows:&C::Value,width:usize,slots:&[usize],round:bool)->Result<Vec<Option<C::Value>>> {(**self).projection_origins(name,rows,width,slots,round)}
    fn prepare_origins(&self,name:&str,rows:&C::Value,width:usize,round:bool)->Result<()> {(**self).prepare_origins(name,rows,width,round)}
    fn remember_projections(&self,name:&str,round:bool,kept:&[(&C::Value,&C::Value)])->Result<()> {(**self).remember_projections(name,round,kept)}
    fn repeated_projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,x:&C::Value,round:bool)->Result<Option<C::Value>> {(**self).repeated_projection(c,name,x,round)}
    fn upload(&self,v:&[f32])->Result<C::Value> {(**self).upload(v)}
    fn upload_known(&self,v:&[f32])->Result<C::Value> {(**self).upload_known(v)}
    fn read(&self,v:&C::Value)->Result<Vec<f32>> {(**self).read(v)}
    fn read_pair(&self,first:&C::Value,second:&C::Value)->Result<(Vec<f32>,Vec<f32>)> {(**self).read_pair(first,second)}
    fn select_known(&self,ids:&C::Value,count:&C::Value,row:&[f32])->Result<()> {(**self).select_known(ids,count,row)}
    fn select_known_rows(&self,value:&C::Value,slots:usize,rows:&[f32])->Result<bool> {(**self).select_known_rows(value,slots,rows)}
    fn range(&self,v:&C::Value,s:usize,n:usize)->Result<C::Value> {(**self).range(v,s,n)}
    fn words(&self,v:&C::Value)->usize {(**self).words(v)}
    fn join(&self,v:&[&C::Value])->Result<C::Value> {(**self).join(v)}
    fn cache_append(&self,old:&C::Value,new:&C::Value,rows:usize,width:usize,room:usize)->Result<C::Value> {(**self).cache_append(old,new,rows,width,room)}
    fn join_reserved(&self,v:&[&C::Value],reserve:usize)->Result<C::Value> {(**self).join_reserved(v,reserve)}
    fn zeros(&self,n:usize)->Result<C::Value> {(**self).zeros(n)}
    fn gather_reserved(&self,s:Shape,x:[Option<&C::Value>;4],reserve:usize)->Result<C::Value> {(**self).gather_reserved(s,x,reserve)}
    fn gather_absorbed(&self,s:Shape,x:[Option<&C::Value>;4],reserve:usize,expected:f32)->Result<C::Value> {(**self).gather_absorbed(s,x,reserve,expected)}
    fn operation(&self,s:Shape,x:[Option<&C::Value>;4],w:&[f32])->Result<C::Value> {(**self).operation(s,x,w)}
    fn operation_prepared(&self,s:Shape,x:[Option<&C::Value>;4],w:&std::rc::Rc<crate::bounded_decoder::StaticParameters>)->Result<C::Value> {(**self).operation_prepared(s,x,w)}
    fn norm(&self,name:&str,s:NormShape,x:&C::Value,w:&[f32])->Result<C::Value> {(**self).norm(name,s,x,w)}
    fn begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&C::Value)->Result<(C::Pass,C::Value)> {(**self).begin(key,p,x)}
    fn begin_next(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&C::Value,next:&[&str])->Result<(C::Pass,C::Value)> {(**self).begin_next(key,p,x,next)}
    fn finish(&self,p:C::Pass,x:&C::Value)->Result<C::Value> {(**self).finish(p,x)}
    fn mx(&self,p:&Mxfp8Projection<'_>,x:&C::Value)->Result<C::Value> {(**self).mx(p,x)}
    fn round(&self,x:&C::Value)->Result<C::Value> {(**self).round(x)}
    fn bf16_sum(&self,x:&C::Value,y:&C::Value)->Result<C::Value> {(**self).bf16_sum(x,y)}
    fn mean(&self,x:&C::Value,s:u32,h:u32)->Result<C::Value> {(**self).mean(x,s,h)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&C::Value>;5],w:&[f32])->Result<C::Value> {(**self).prompt(s,x,w)}
    fn rows_recurrent(&self,s:spark_gpu_probe::recurrent_rows::Shape,x:[&C::Value;7],gate:&C::Value,w:[&[f32];3],keep:bool)
        ->Result<Option<crate::bounded_decoder::RecurrentRows<C::Value>>> {(**self).rows_recurrent(s,x,gate,w,keep)}
    fn rows_embedding(&self,t:&[u32],n:usize)->Result<Option<C::Value>> {(**self).rows_embedding(t,n)}
    fn rows_project(&self,c:&CheckpointAdmission,name:&str,x:&C::Value,m:usize,round:bool)->Result<Option<C::Value>> {(**self).rows_project(c,name,x,m,round)}
    fn rows_stage(&self,c:&CheckpointAdmission,jobs:&[(&str,&C::Value,bool)],m:usize)->Result<Option<Vec<C::Value>>> {(**self).rows_stage(c,jobs,m)}
    fn rows_stage_open(&self,c:&CheckpointAdmission,jobs:&[(&str,&C::Value,bool)],m:usize)->Result<Option<Vec<Option<C::Value>>>> {(**self).rows_stage_open(c,jobs,m)}
    fn rows_stage_close(&self)->Result<Vec<(usize,C::Value)>> {(**self).rows_stage_close()}
    fn batch_rows(&self,chats:usize,state_bytes:usize)->Result<()> {(**self).batch_rows(chats,state_bytes)}
    fn rows_experts(&self,a:&[(&str,usize)],b:Option<(&[(&str,usize)],f32)>,x:&C::Value,p:&[f32])->Result<Option<C::Value>> {(**self).rows_experts(a,b,x,p)}
    fn rows_experts_gathered(&self,a:&[(&str,usize)],b:Option<(&[(&str,usize)],f32)>,x:&C::Value,o:&[f32],p:&[f32])->Result<Option<C::Value>> {(**self).rows_experts_gathered(a,b,x,o,p)}
    fn rows_experts_open(&self,a:&[(&str,usize)],b:(&[(&str,usize)],f32),x:&C::Value,p:&[f32])->Result<bool> {(**self).rows_experts_open(a,b,x,p)}
    fn rows_experts_close(&self,failed:bool)->Result<C::Value> {(**self).rows_experts_close(failed)}
    fn rows_experts_open_gathered(&self,a:&[(&str,usize)],b:(&[(&str,usize)],f32),x:&C::Value,o:&[f32],p:&[f32])->Result<bool> {(**self).rows_experts_open_gathered(a,b,x,o,p)}
    fn rows_routed(&self,steps:&[engine_model::plan::layer::Step],bound:Option<f32>,ids:&C::Value,x:&C::Value,m:usize)->Result<Option<C::Value>> {(**self).rows_routed(steps,bound,ids,x,m)}
    fn routes_rows(&self,m:usize)->bool {(**self).routes_rows(m)}
    fn rows_routed_fold(&self,down:&engine_model::plan::layer::Step,ids:&C::Value,input:&C::Value,weights:&C::Value,shared:Option<&C::Value>,fold:Shape,m:usize)->Result<Option<C::Value>> {(**self).rows_routed_fold(down,ids,input,weights,shared,fold,m)}
    fn folds_rows(&self,m:usize)->bool {(**self).folds_rows(m)}
    fn folds_groups(&self,m:usize)->bool {(**self).folds_groups(m)}
    fn rows_experts_fold(&self,first:&[(&str,usize)],source:&C::Value,places:&[f32],ids:&C::Value,weights:&C::Value,shared:Option<&C::Value>,fold:Shape,m:usize)->Result<Option<C::Value>> {(**self).rows_experts_fold(first,source,places,ids,weights,shared,fold,m)}
    fn rows_project_local(&self,c:&CheckpointAdmission,name:&str,x:&C::Value,m:usize,round:bool)->Result<Option<C::Value>> {(**self).rows_project_local(c,name,x,m,round)}
    #[allow(clippy::type_complexity)] // the trait's tuple a set.
    fn rows_experts_gathered_wide(&self,sets:&[(&[(&str,usize)],&[(&str,usize)],f32,&C::Value,&[f32],&[f32])])->Result<Option<Vec<C::Value>>> {(**self).rows_experts_gathered_wide(sets)}
    #[allow(clippy::type_complexity)] // the trait's tuple a set.
    fn rows_experts_gathered_wide_for(&self,sets:&[(&[(&str,usize)],&[(&str,usize)],f32,&C::Value,&[f32],&[f32])],down:&[&[(&str,usize)]],reader:(Shape,Option<&str>))->Result<Option<Vec<C::Value>>> {(**self).rows_experts_gathered_wide_for(sets,down,reader)}
    #[allow(clippy::type_complexity)] // the trait's tuple a set.
    fn rows_experts_fold_wide(&self,sets:&[(&[(&str,usize)],&C::Value,&[f32],&C::Value,&C::Value,Option<&C::Value>,Shape,usize)])->Result<Option<Vec<C::Value>>> {(**self).rows_experts_fold_wide(sets)}
    #[allow(clippy::type_complexity)]
    fn rows_experts_device_wide(&self,steps:&[engine_model::plan::layer::Step],bound:f32,sets:&[(usize,&C::Value,&C::Value)],reader:(Shape,Option<&str>))->Result<Option<Vec<C::Value>>> {(**self).rows_experts_device_wide(steps,bound,sets,reader)}
    #[allow(clippy::type_complexity)]
    fn rows_experts_fold_device_wide(&self,down:&engine_model::plan::layer::Step,sets:&[(usize,&C::Value,&C::Value,&C::Value,Option<&C::Value>)],fold:Shape)->Result<Option<Vec<C::Value>>> {(**self).rows_experts_fold_device_wide(down,sets,fold)}
    fn rows_project_owned(&self,c:&CheckpointAdmission,name:&str,x:&C::Value,m:usize,round:bool,rows:std::ops::Range<usize>)->Result<Option<C::Value>> {(**self).rows_project_owned(c,name,x,m,round,rows)}
    fn note_batch_experts(&self,picks:usize,distinct:usize) {(**self).note_batch_experts(picks,distinct)}
    fn rows_verify(&self,on:bool) {(**self).rows_verify(on)}
    fn warm_routed_rows(&self,steps:&[engine_model::plan::layer::Step],picks:usize,max_rows:usize)->Result<()> {(**self).warm_routed_rows(steps,picks,max_rows)}
    fn rows_release(&self)->Result<()> {(**self).rows_release()}
    fn rows_keep(&self)->Result<()> {(**self).rows_keep()}
    fn rows_return_lease(&self)->Result<()> {(**self).rows_return_lease()}
    fn rows_phase_end(&self)->Result<()> {(**self).rows_phase_end()}
    fn rows_unrecorded(&self,p:bool)->Result<bool> {(**self).rows_unrecorded(p)}
    fn prompt_end(&self)->Result<()> {(**self).prompt_end()}
    fn prompt_seeds(&self)->bool {(**self).prompt_seeds()}
    fn rows_charge(&self,bytes:usize) {(**self).rows_charge(bytes)}
    fn rows_follow(&self,follows:bool) {(**self).rows_follow(follows)}
    fn rows_settle(&self)->Result<()> {(**self).rows_settle()}
    fn decode_selection_reads(&self,reads:bool) {(**self).decode_selection_reads(reads)}
    fn rows_peak_begin(&self)->usize {(**self).rows_peak_begin()}
    fn rows_peak_end(&self,whole:usize)->usize {(**self).rows_peak_end(whole)}
    fn rows_reserve(&self,words:usize)->Result<C::Value> {(**self).rows_reserve(words)}
    fn rows_fill(&self,target:&mut C::Value,offset:usize,part:&C::Value,expected:usize)->Result<()> {(**self).rows_fill(target,offset,part,expected)}
    fn rows_append(&self,held:Option<&C::Value>,rows:usize,new:&C::Value,width:usize)->Result<C::Value> {(**self).rows_append(held,rows,new,width)}
    fn rows_extend(&self,held:Option<&C::Value>,rows:usize,new:&C::Value,width:usize,room:usize)->Result<C::Value> {(**self).rows_extend(held,rows,new,width,room)}
    fn rows_norm(&self,name:&str,s:NormShape,x:&C::Value,m:usize,w:&[f32])->Result<C::Value> {(**self).rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&C::Value,m:usize)->Result<(C::Pass,C::Value)> {(**self).rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:C::Pass,x:&C::Value,m:usize)->Result<C::Value> {(**self).rows_finish(p,x,m)}
}
