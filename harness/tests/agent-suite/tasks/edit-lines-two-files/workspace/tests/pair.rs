#[test]
fn handshake_uses_v4() {
    assert!(pair_schema::handshake_ok());
}
