//! A small ledger. Amounts are whole cents.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Credit,
    Debit,
}

#[derive(Debug, Clone, Copy)]
pub struct Entry {
    pub kind: Kind,
    pub cents: u64,
}

/// The balance in cents: credits add to it and debits subtract from it.
pub fn balance(entries: &[Entry]) -> i64 {
    entries
        .iter()
        .map(|e| match e.kind {
            Kind::Credit => e.cents as i64,
            Kind::Debit => e.cents as i64,
        })
        .sum()
}

/// The entries of one kind, in their original order.
pub fn of_kind(entries: &[Entry], kind: Kind) -> Vec<Entry> {
    entries.iter().copied().filter(|e| e.kind == kind).collect()
}
