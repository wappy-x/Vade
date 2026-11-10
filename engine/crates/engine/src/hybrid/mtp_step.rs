//! A child of the target decode module observes its current hidden row without copying a layer loop.
impl<'m, 'a, 'b> super::super::Decode<'m, 'a, 'b> {
    pub(crate) fn step_with_mtp(&mut self, session: &mut crate::hybrid::Session<'m, 'a, 'b>, token: u32, logits: bool)
        -> crate::hybrid::Result<Vec<Vec<f32>>> {
        let entry=session.position;
        let output=self.step(&mut [super::super::Row { session:&mut *session,token,logits }])?;
        if !session.valid { return Ok(output); }
        let committed=session.position;
        session.position=entry; session.valid=false;
        let observed=self.hidden.slice(0..self.model.config.hidden as usize)
            .and_then(|hidden| session.mtp_observe_hidden(&[token],&hidden));
        session.position=committed; session.valid=observed.is_ok(); observed?;
        Ok(output)
    }
}
