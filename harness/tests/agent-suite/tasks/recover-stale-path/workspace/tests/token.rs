use token_auth::Token;

#[test]
fn expired_at_exact_boundary() {
    let t = Token { subject: "build-bot", expires_at: 100 };
    assert!(!t.is_expired(99));
    assert!(t.is_expired(100));
    assert!(t.is_expired(101));
}
