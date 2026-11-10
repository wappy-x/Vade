use throttle::{nested::Inner, outer_limit};

#[test]
fn outer_stays_ten() {
    assert_eq!(outer_limit(3), 32);
}

#[test]
fn inner_burst_is_twenty_five() {
    assert_eq!(Inner::burst_limit(3), 77);
}
