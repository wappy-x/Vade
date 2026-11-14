//! Request sequencing for a checked multi-stream text description.
//! No checkpoint factory is supplied here. The decoder, final norm, state,
//! and transport owners must be bound before the public server can use this.
use crate::{Engine, Session, Tokenizer};
use engine_model::{plan::{TextDescription,description::DecoderDescription,text_edges::TextEdges},safetensors::Tensor};
use std::collections::BTreeMap;

/// A factory owns one immutable checkpoint and rank. create establishes the
/// session owners. A bounded probe may load weights during a token step.
/// A failure must drain unfinished work and leave committed state unchanged.
/// Public service admission must also check its physical resource contract.
pub trait Factory {
    type Program<'a>: TokenProgram where Self:'a;
    fn create(&self, description: &TextDescription, edges: &TextEdges, context: usize) -> Result<Self::Program<'_>,String>;
}
/// One ordered token transaction. Each method owns or borrows live operands.
/// commit completes all compute and transport before publishing new state.
/// abort drains users on every error path and retires uncertain device state.
/// Drop must also drain unfinished work. Production panic aborts are not recoverable.
pub trait TokenProgram {
    type Residual;
    type Hidden;
    fn embed(&mut self, token: u32, position: usize) -> Result<Self::Residual,String>;
    fn layer(&mut self, index: usize, operation: &DecoderDescription, residual: Self::Residual) -> Result<Self::Residual,String>;
    fn mean(&mut self, residual: Self::Residual) -> Result<Self::Hidden,String>;
    fn norm(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden,String>;
    fn head(&mut self, hidden: Self::Hidden) -> Result<Vec<f32>,String>;
    fn commit(&mut self) -> Result<(),String>;
    fn abort(&mut self) -> Result<(),String>;
}
pub struct TextEngine<F> { factory:F, description:TextDescription, edges:TextEdges, context:usize }
impl<F:Factory> TextEngine<F> {
    /// Read service metadata from one admitted checkpoint. The factory must use
    /// that checkpoint and supply the complete decoder and physical state.
    pub fn from_checkpoint<T:Tokenizer>(factory:F,checkpoint:&engine_model::plan::admission::CheckpointAdmission,
        tokenizer:&T,context:usize) -> Result<Self,String> {
        let bound = crate::text_checkpoint::CheckpointTextEdges::new(checkpoint)?;
        Self::with_edges(factory,checkpoint.description().clone(),bound.edges().clone(),tokenizer,context)
    }
    /// Bind source headers and the complete tokenizer EOS set before a session.
    /// The supplied Factory is an execution implementation, not a readiness flag.
    /// Header success alone must not make a deferred public checkpoint executable.
    pub fn new<T:Tokenizer>(factory:F, description:TextDescription, headers:&BTreeMap<String,Tensor>,
        tokenizer:&T, context:usize) -> Result<Self,String> {
        let edges = description.text_edges(headers).map_err(|e| e.to_string())?;
        Self::with_edges(factory,description,edges,tokenizer,context)
    }
    fn with_edges<T:Tokenizer>(factory:F,description:TextDescription,edges:TextEdges,tokenizer:&T,context:usize) -> Result<Self,String> {
        if context == 0 || context as u64 > description.max_positions || description.layers.is_empty()
            || usize::try_from(edges.vocab).is_err() || usize::try_from(edges.hidden).is_err() {
            return Err("invalid text program context or dimensions".into());
        }
        let stops = tokenizer.checkpoint_eos_token_ids()?.ok_or("text program requires checkpoint EOS metadata")?;
        if stops.len() != edges.eos_tokens.len() || edges.eos_tokens.iter().any(|id| !stops.contains(id) || !tokenizer.has_token(*id)) {
            return Err("text program and tokenizer EOS sets differ".into());
        }
        Ok(Self {factory,description,edges,context})
    }
    pub fn edges(&self) -> &TextEdges {&self.edges}
}
impl<F:Factory> Engine for TextEngine<F> {
    type Session<'a> = TextSession<'a,F::Program<'a>> where Self:'a;
    fn session(&self) -> Result<Self::Session<'_>,String> {
        let program = self.factory.create(&self.description,&self.edges,self.context)?;
        Ok(TextSession {program,description:&self.description,context:self.context,position:0,failed:false})
    }
    fn context_limit(&self) -> usize {self.context}
    fn vocab_size(&self) -> usize {self.edges.vocab as usize}
    fn eos_token_ids(&self) -> &[u32] {&self.edges.eos_tokens}
}
pub struct TextSession<'a,P> {program:P,description:&'a TextDescription,context:usize,position:usize,failed:bool}
impl<P:TokenProgram> TextSession<'_,P> {
    fn token(&mut self, token:u32) -> Result<Vec<f32>,String> {
        let mut residual = self.program.embed(token,self.position)?;
        for (index,operation) in self.description.layers.iter().enumerate() {
            residual = self.program.layer(index,operation,residual)?;
        }
        let hidden = self.program.mean(residual)?;
        let hidden = self.program.norm(hidden)?;
        let logits = self.program.head(hidden)?;
        if logits.len() as u64 != self.description.vocab || logits.iter().any(|x| !x.is_finite()) {
            return Err("text program returned invalid full logits".into());
        }
        self.program.commit()?;
        self.position += 1; Ok(logits)
    }
}
impl<P:TokenProgram> Session for TextSession<'_,P> {
    fn step(&mut self, token:u32) -> Result<Vec<f32>,String> {
        if self.failed {return Err("text session previously failed".into());}
        if u64::from(token) >= self.description.vocab || self.position >= self.context {
            return Err("text request exceeds vocabulary or context".into());
        }
        // Mark failed before work starts. An unwinding producer cannot make the
        // session reusable. Its owner must drain work when the session drops.
        self.failed = true;
        match self.token(token) {
            Ok(logits) => {self.failed = false; Ok(logits)}
            Err(error) => match self.program.abort() {
                Ok(()) => Err(error), Err(cleanup) => Err(format!("{error}; text abort failed: {cleanup}")),
            },
        }
    }
    fn can_fork(&self) -> bool {false}
    fn can_fork_prefix(&self, _position:usize) -> bool {false}
    fn truncate(&mut self, position:usize) -> Result<(),String> {
        if self.failed || position != self.position {return Err("text program has no qualified rollback binding".into());}
        Ok(())
    }
}
