// Hidden tests for follow-agents-md. The runner copies this file in after
// the agent finishes; the agent never sees it.
use tally::order::median;

#[test]
fn median_values() {
    assert_eq!(median(&[]), None);
    assert_eq!(median(&[5]), Some(5));
    assert_eq!(median(&[3, 1, 2]), Some(2));
    assert_eq!(median(&[4, 1, 3, 2]), Some(2));
    assert_eq!(median(&[-5, 10]), Some(-5));
}
