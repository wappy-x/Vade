//! One chat's layer program and kept verifier rows.
use super::*;

impl<'f,C:Compute> Decoder<'f,C> {
    pub(in crate::bounded_decoder) fn verify_open_rows(&self,tokens:&[u32],state:&State<C::Value>,first:usize,
        solo_verify:bool,mut picked:Option<&mut Vec<u32>>)->Result<(Vec<Vec<f32>>,State<C::Value>,OpenRows<C::Value>)> {
        let (logits,next,open)=self.rows_pass(tokens,state,HeadRows::From(first),2,solo_verify,picked.as_deref_mut())?;
        if let Some(ids)=picked {need(first<tokens.len() && ids.len()==tokens.len()-first,"prompt greedy row count differs")?;}
        Ok((logits,next,open.ok_or("prompt group kept no rows")?))
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn layer_rows(&self,layer:usize,first:u64,m:usize,residual:C::Value,state:&BTreeMap<String,C::Value>,expanded:Option<&Expanded<C::Value>>,keep:bool,solo_verify:bool,cut:Option<(usize,&str)>)->Result<(C::Value,BTreeMap<String,C::Value>,Option<C::Value>,Option<KeptRows<C::Value>>)> {
        let b=&self.compute;let last=first+m as u64;
        let mut chat=self.rows_chat(layer,first,m,state,expanded,keep)?;
        chat.solo_verify=solo_verify;
        let planning=std::time::Instant::now();
        // A lone verifier has no nested schedule lookup while this layer runs.
        // Other passes release the schedule before executing their owned plan.
        let (mut loan,owned);
        let plan=if solo_verify {
            loan=self.schedules.borrow_mut();
            scheduled_layer(&mut loan,layer,last)?
        } else {
            owned=scheduled_layer(&mut self.schedules.borrow_mut(),layer,last)?.clone();
            &owned
        };
        b.plan_cost("planning",planning.elapsed().as_nanos());
        let mut values=BTreeMap::new();values.insert("residual".to_owned(),residual);
        // GROUP FOLD: a prompt group (no kept rows) may fold a routed layer's down rows before their exchange.
        let mut passes=BTreeMap::new();let mut fold=if keep {RowsFold::default()} else {RowsFold::of_group()};let mut skip=0;
        for index in 0..cut.map_or(plan.steps.len(),|cut|cut.0) {
            if skip>0 {skip-=1;continue;}
            if let Some(done)=self.rows_stateless(layer,index,plan,m,&mut values,&mut passes,&mut fold)? {skip=done;continue;}
            skip=self.rows_history(layer,index,plan,&mut chat,&mut values)?;
        }
        need(skip==0,"prompt layer cut is inside a stage")?;
        let output=values.remove(cut.map_or("output",|cut|cut.1)).ok_or("prompt layer has no output")?;
        let (next,cache,kept)=self.rows_chat_end(layer,chat,&mut values)?;
        if layer==0 && keep {spark_gpu_probe::prompt_launches::record(if solo_verify {"verify_solo_plan"} else {"verify_owned_plan"});}
        Ok((output,next,cache,kept))
    }
}

fn scheduled_layer(schedules:&mut [LayerSchedule],layer:usize,last:u64)->Result<&LayerPlan> {
    schedules.get_mut(layer).ok_or("prompt layer is outside the recipe")?.at(last).map_err(|e|e.to_string())
}
