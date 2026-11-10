/// An access token for one subject.
pub struct Token {
    pub subject: &'static str,
    pub expires_at: u64,
}

impl Token {
    /// True when the token is no longer valid at `now`.
    pub fn is_expired(&self, now: u64) -> bool {
        now > self.expires_at
    }
}
