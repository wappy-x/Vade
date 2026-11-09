//! Typed failures for logical splitting and memory budgets.
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitError {
    Ranks { ranks: u32 },
    Budgets { ranks: u32, given: usize },
    Unsupported { what: &'static str },
    Shape { what: &'static str },
    Overflow { what: &'static str },
    NoFit { rank: u32, needed: u64, available: u64 },
}
impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ranks { ranks } => write!(f, "invalid Spark count: {ranks}"),
            Self::Budgets { ranks, given } => write!(f, "{ranks} ranks need {ranks} budgets, got {given}"),
            Self::Unsupported { what } => write!(f, "unsupported split: {what}"),
            Self::Shape { what } => write!(f, "invalid split shape: {what}"),
            Self::Overflow { what } => write!(f, "split size overflow: {what}"),
            Self::NoFit { rank, needed, available } =>
                f.write_str(&engine_base::admission::refusal(*needed as u128, *available as usize, &format!("weights/rank {rank} {}; session geometry, CONTEXT and OUTPUT not supplied to splitter", engine_base::admission::bytes(*needed as u128)), "the caller's per-rank split budget", engine_base::admission::SERVE)),
        }
    }
}
impl std::error::Error for SplitError {}
