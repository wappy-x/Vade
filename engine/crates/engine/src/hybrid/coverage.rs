//! Every reconciled tensor has an explicit text-forward use or refusal.
use engine_model::shapes::{TensorEntry,Place,Part,Store};
pub fn use_of(t: &TensorEntry) -> Result<&'static str,String> {
    match (t.role.place,t.role.part,t.role.store,t.shape.len()) {
        (Place::Unused,_,_,_)=>Ok("vision excluded"),
        (_,Part::Expert {id:Some(_),..},Store::Weight,2)=>Ok("routed descriptor"),
        (_,Part::Query|Part::Key|Part::Value|Part::Output|Part::LinearQkv|Part::LinearZ|Part::LinearOut|Part::Shared(_)|Part::Expert {id:Some(_),..},Store::BlockScale,2)=>Ok("paired BF16 multiplier"),
        (Place::Model,Part::Embedding,Store::Weight,2)=>Ok("BF16 embedding"),
        (_,Part::Query|Part::Key|Part::Value|Part::Output|Part::LinearQkv|Part::LinearZ|Part::LinearOut|Part::Shared(_)|Part::Head,Store::Weight,2)=>Ok("fixed projection"),
        (_,Part::Whole|Part::LinearHeads,Store::Weight,2)=>Ok("canonical BF16 projection"),
        (_,Part::Whole|Part::LinearHeads|Part::LinearNorm|Part::QueryNorm|Part::KeyNorm,Store::Weight,1)|(_,Part::LinearConv,Store::Weight,3)=>Ok("norm or recurrent parameter"),
        _=>Err(format!("{}: role is not consumed by the typed hybrid forward",t.name)),
    }
}
