//! Header binding outside the decoder stack. No final norm backend is selected.
use super::{TextDescription, glm_index::{TensorRole,TensorStorage}, storage::WeightStorage};
use crate::{Result,invalid,safetensors::{Dtype,Tensor}};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredTensor { pub name: String, pub tensor: Tensor }
#[derive(Debug, Clone, PartialEq)]
pub struct TextEdges {
    pub hidden: u64, pub vocab: u64, pub streams: u64, pub epsilon: f64,
    pub eos_tokens: Vec<u32>, pub embedding: StoredTensor, pub norm: StoredTensor,
    pub head: StoredTensor, pub head_scales: StoredTensor,
}
impl TextDescription {
    /// Use the schema's tensor roles, recipes and actual headers. The final norm
    /// retains its stored dtype. This does not infer its loaded runtime dtype.
    pub fn text_edges(&self, headers: &BTreeMap<String,Tensor>) -> Result<TextEdges> {
        if self.tied_embeddings || self.hidden == 0 || !self.hidden.is_multiple_of(32) || self.vocab == 0
            || self.residual.streams == 0 || !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(invalid("unsupported text edge geometry or tied storage"));
        }
        let mut embedding = None; let mut norm = None; let mut head = None; let mut scales = None;
        for (name,tensor) in headers {
            let binding = self.bind_tensor(name)?;
            let (slot,shape,dtype,recipe) = match (binding.operation,binding.component) {
                (TensorRole::Embedding,TensorStorage::Weight) => (&mut embedding,vec![self.vocab,self.hidden],Dtype::Bf16,WeightStorage::Plain),
                (TensorRole::Head,TensorStorage::Weight) => (&mut head,vec![self.vocab,self.hidden],Dtype::F8E4M3,WeightStorage::Mxfp8),
                (TensorRole::Head,TensorStorage::BlockScale) => (&mut scales,vec![self.vocab,self.hidden/32],Dtype::U8,WeightStorage::Mxfp8),
                (TensorRole::FinalNorm,TensorStorage::Weight) => {
                    if !matches!(tensor.dtype,Dtype::Bf16|Dtype::F32) {return Err(invalid("unsupported stored final norm dtype"));}
                    (&mut norm,vec![self.hidden],tensor.dtype,WeightStorage::Plain)
                }
                (TensorRole::Embedding|TensorRole::Head|TensorRole::FinalNorm,_) => return Err(invalid("unexpected text edge component")),
                _ => continue,
            };
            let bytes = shape.iter().try_fold(dtype.bytes(),|n,&d| n.checked_mul(d)).ok_or_else(|| invalid("text tensor size overflow"))?;
            if tensor.shape != shape || tensor.dtype != dtype || binding.storage != recipe
                || tensor.end.checked_sub(tensor.start) != Some(bytes) || slot.is_some() {
                return Err(invalid(format!("invalid text edge tensor {name}")));
            }
            *slot = Some(StoredTensor {name:name.clone(),tensor:tensor.clone()});
        }
        let mut eos_tokens = Vec::new();
        for &id in &self.eos_tokens {
            let id = u32::try_from(id).map_err(|_| invalid("EOS ID exceeds token ABI"))?;
            if u64::from(id) >= self.vocab || eos_tokens.contains(&id) {return Err(invalid("invalid text EOS set"));}
            eos_tokens.push(id);
        }
        if eos_tokens.is_empty() {return Err(invalid("empty text EOS set"));}
        let missing = || invalid("missing text embedding, final norm, head, or scales");
        Ok(TextEdges {hidden:self.hidden,vocab:self.vocab,streams:self.residual.streams,epsilon:self.epsilon,eos_tokens,
            embedding:embedding.ok_or_else(missing)?,norm:norm.ok_or_else(missing)?,head:head.ok_or_else(missing)?,head_scales:scales.ok_or_else(missing)?})
    }
}
