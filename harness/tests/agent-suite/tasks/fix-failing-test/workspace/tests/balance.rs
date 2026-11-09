use ledger::{balance, of_kind, Entry, Kind};

fn credit(cents: u64) -> Entry {
    Entry { kind: Kind::Credit, cents }
}

fn debit(cents: u64) -> Entry {
    Entry { kind: Kind::Debit, cents }
}

#[test]
fn empty_ledger_is_zero() {
    assert_eq!(balance(&[]), 0);
}

#[test]
fn credits_add() {
    assert_eq!(balance(&[credit(250), credit(100)]), 350);
}

#[test]
fn debits_subtract() {
    assert_eq!(balance(&[credit(1000), debit(300), debit(50)]), 650);
}

#[test]
fn balance_can_go_negative() {
    assert_eq!(balance(&[debit(75)]), -75);
}

#[test]
fn filter_keeps_order() {
    let all = [credit(1), debit(2), credit(3)];
    let cents: Vec<u64> = of_kind(&all, Kind::Credit).iter().map(|e| e.cents).collect();
    assert_eq!(cents, vec![1, 3]);
}
