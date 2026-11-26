//! Optional initial short-prompt admission grace, parsed once at startup.
use std::{sync::OnceLock, time::Duration};
static ENABLED: OnceLock<Result<bool, String>> = OnceLock::new();
fn parse(value: Result<String, std::env::VarError>) -> Result<bool, String> {
    match value.as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => Ok(false),
        Ok("1") => Ok(true),
        _ => Err("VADE_INTAKE_GRACE must be unset, 0 or 1".into()),
    }
}
/// Invalid values stay cached as startup errors, including when a service restarts.
pub fn load() -> Result<bool, String> {
    ENABLED.get_or_init(|| parse(std::env::var("VADE_INTAKE_GRACE"))).clone()
}
pub(crate) fn minimum_prompt_age(enabled: bool) -> Duration {
    Duration::from_millis(if enabled { 10 } else { 2 })
}
#[cfg(test)]
mod tests {
    use super::{minimum_prompt_age, parse};
    use std::time::Duration;
    #[test]
    fn intake_grace_is_strict_and_preserves_default_age() {
        assert_eq!(parse(Err(std::env::VarError::NotPresent)), Ok(false));
        assert_eq!(parse(Ok("0".into())), Ok(false));
        assert_eq!(parse(Ok("1".into())), Ok(true));
        for value in ["", "true", "yes", "2", " 1", "1 "] {
            assert_eq!(parse(Ok(value.into())).unwrap_err(), "VADE_INTAKE_GRACE must be unset, 0 or 1");
        }
        assert!(parse(Err(std::env::VarError::NotUnicode("bad".into()))).is_err());
        assert_eq!(minimum_prompt_age(false), Duration::from_millis(2));
        assert_eq!(minimum_prompt_age(true), Duration::from_millis(10));
    }
}
