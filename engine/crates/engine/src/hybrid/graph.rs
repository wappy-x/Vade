//! Exact stream capture: device inputs change, kernel arguments stay fixed.
use super::*;
pub(super) type Graphs<'a> = Vec<(Vec<u64>, Vec<std::rc::Weak<()>>, Option<DecodeGraph<'a>>)>;
pub(super) fn replay<'a>(b: &Backend<'a>, graphs: &mut Graphs<'a>, key: Vec<u64>,
    lives: Vec<std::rc::Weak<()>>, layers: usize, rows: usize, mut run: impl FnMut() -> Result<()>) -> Result<()> {
    if !b.graph_capture_ready() { return run(); }
    graphs.retain(|(_, lives, _)| lives.iter().all(|life| life.upgrade().is_some()));
    let index = if let Some(i) = graphs.iter().position(|(k, l, _)| *k == key && l.len() == lives.len()
        && l.iter().zip(&lives).all(|(a,b)| std::rc::Weak::ptr_eq(a,b))) { i } else {
        if graphs.len() == 4 { graphs.remove(0); }
        let graph = unsafe { b.try_capture_batch(layers, rows, &mut run)? };
        graphs.push((key, lives, graph)); graphs.len()-1
    };
    match &graphs[index].2 { Some(graph) => unsafe { b.replay_decode(graph) }, None => run() }
}
