//! Raw greedy choice policy shared by compact and full-logit verification.
use super::*;
pub const UNMAPPED_GREEDY:&str="greedy winner is not in the tokenizer";
impl Speculator {
    /// Let the caller process the verified prefix and its text stops before failing.
    pub fn check_greedy_error(&self)->Result<()> {if self.greedy_error {Err(UNMAPPED_GREEDY.into())} else {Ok(())}}
    pub(super) fn checked_greedy(&mut self, token:u32, unmasked:bool, stop:&impl Fn(u32)->bool,
        allowed:&impl Fn(u32)->bool)->Result<u32> {
        // Plain decode never asks for a choice after EOS or the output limit.
        if unmasked && self.tokens().len()<self.config.max_new && !self.history.last().is_some_and(|&t|stop(t)) && !allowed(token) {
            self.greedy_error=true;
        }
        Ok(token)
    }
    /// ID-only verification with the same validity check as exact logits.
    pub fn finish_greedy_filtered(&mut self, target:&mut (impl Target+?Sized), plan:Proposal,
        output:&[u32], stop:impl Fn(u32)->bool, allowed:impl Fn(u32)->bool)->Result<&[u32]> {
        if !plain_greedy(self.options) {return Err("greedy result for sampled draft".into());}
        if output.len()!=plan.input.len() {return Err("invalid prepared draft row count".into());}
        let target_ms=plan.elapsed_ms();let learn=Instant::now();
        for i in (0..output.len()).rev() {
            if let Some(next)=self.recycling.successors.get_mut(plan.input[i] as usize) {*next=output[i];}
        }
        let learn_ms=learn.elapsed().as_secs_f64()*1000.0;let unmasked=target.reject_unmapped_greedy();
        self.finish_choices(target,plan,output,target_ms,learn_ms,&stop,|s,i|s.checked_greedy(output[i],unmasked,&stop,&allowed))
    }
}
