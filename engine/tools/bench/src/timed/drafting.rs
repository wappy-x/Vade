//! Keep requested drafting separate from verbatim, potentially stale server metadata.
use super::value::{Value, a, encode, n, o, s};
use std::collections::BTreeSet;

pub const SERVER_DEPTH_LABEL: &str = "server-reported (known to read 0 on the FP4-Local path until the health fix lands)";

fn depth(value: &Value) -> Option<u64> {
    value.uint().or_else(|| value.str().and_then(|v| v.parse().ok()))
}

fn enabled(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(v) => Some(*v),
        Value::String(v) if matches!(v.as_str(), "off" | "disabled" | "false") => Some(false),
        Value::String(v) if matches!(v.as_str(), "on" | "enabled" | "true") => Some(true),
        _ => depth(value).map(|v| v != 0),
    }
}

fn agree<T: PartialEq>(previous: &mut Option<T>, current: T, field: &str) -> Result<(), String> {
    if previous.as_ref().is_some_and(|v| v != &current) {
        return Err(format!("/health reports contradictory drafting settings at {field}"));
    }
    *previous = Some(current);
    Ok(())
}

pub fn metadata(health: &Value, requested: Option<&str>) -> Result<Value, String> {
    let mut server_depth = None;
    let mut reported_depth = Value::Null;
    let mut depth_field = Value::Null;
    for field in ["mtp_depth", "effective_mtp_depth", "mtp.depth", "vade.mtp_depth",
        "vade.numerics.mtp_depth", "numerics.mtp_depth"] {
        let value = health.path(field);
        if value != &Value::Null {
            let current = depth(value).ok_or_else(|| format!("invalid /health {field}"))?;
            agree(&mut server_depth, current, field)?;
            if reported_depth == Value::Null {
                reported_depth = value.clone();
                depth_field = s(field);
            }
        }
    }
    let mut server_enabled = server_depth.map(|v| v != 0);
    for field in ["mtp", "mtp_enabled", "mtp.enabled", "vade.mtp_enabled"] {
        let value = health.path(field);
        if value != &Value::Null && !matches!(value, Value::Object(_)) {
            let current = enabled(value).ok_or_else(|| format!("invalid /health {field}"))?;
            agree(&mut server_enabled, current, field)?;
            // A numeric legacy `mtp` field denotes depth; a boolean does not.
            if field == "mtp" {
                if let Some(current) = depth(value) {
                    agree(&mut server_depth, current, field)?;
                    if reported_depth == Value::Null {
                        reported_depth = value.clone();
                        depth_field = s(field);
                    }
                }
            }
        }
    }
    let mut server_mode = None;
    for field in ["draft", "draft_mode", "effective_draft", "vade.draft", "vade.draft_mode"] {
        let value = health.path(field);
        if value != &Value::Null {
            let mode = value.str().filter(|v| matches!(*v, "plain" | "strict" | "auto"))
                .ok_or_else(|| format!("invalid /health {field}"))?;
            agree(&mut server_mode, mode, field)?;
            if mode != "auto" {
                agree(&mut server_enabled, mode != "plain", field)?;
            }
        }
    }
    let drafting = match requested {
        Some("plain") => "off",
        Some("strict") => "strict-requested",
        None => "server-default",
        Some(_) => return Err("--draft must be plain or strict".into()),
    };
    Ok(o([
        ("request_mode", s(requested.unwrap_or("server"))),
        ("drafting", s(drafting)),
        ("server_draft_mode", server_mode.map(s).unwrap_or(Value::Null)),
        ("server_mtp_depth", reported_depth),
        ("server_mtp_depth_field", depth_field),
        ("server_mtp_depth_label", s(SERVER_DEPTH_LABEL)),
        ("source", s("requested draft policy; server fields copied from /health, not proof of effective drafting")),
    ]))
}

/// `vade.numerics` is immutable server metadata attached to every SSE echo.
/// Its resident/configured depth is not the request's effective `mtp_depth`.
pub fn validate_echo(value: &Value, requested: Option<&str>) -> Result<(), String> {
    let Some(requested) = requested else { return Ok(()) };
    let Value::Object(items) = value else { return Ok(()) };
    for (key, value) in items {
        if matches!(key.as_str(), "numerics" | "build" | "checkpoint") {
            continue;
        }
        if matches!(value, Value::Object(_)) {
            validate_echo(value, Some(requested))?;
            continue;
        }
        if value == &Value::Null { continue; }
        let contradiction = match key.as_str() {
            "draft" | "draft_mode" | "effective_draft" => {
                if requested == "plain" { value.str() != Some("plain") }
                else { value.str() != Some("strict") }
            }
            "mtp_depth" | "effective_mtp_depth" => {
                if requested == "plain" { depth(value) != Some(0) }
                else { !depth(value).is_some_and(|v| v > 0) }
            }
            "mtp" | "mtp_enabled" => {
                if requested == "plain" { enabled(value) != Some(false) }
                else { enabled(value) != Some(true) }
            }
            "draft_used" | "model_draft_used" if requested == "plain" => enabled(value) != Some(false),
            _ => false,
        };
        if contradiction {
            return Err(format!("drafting echo disagrees with --draft {requested}: {key}={}", encode(value)));
        }
    }
    Ok(())
}

/// Retain response policy reports separately from requested policy and health.
/// These echoes are not independent proof of effective drafting or token use.
pub fn observations(records: &[Value]) -> Value {
    fn visit(value: &Value, modes: &mut BTreeSet<String>, depths: &mut BTreeSet<u64>) {
        if let Value::Object(items) = value {
            for (key, value) in items {
                match key.as_str() {
                    "numerics" | "build" | "checkpoint" => continue,
                    "draft" | "draft_mode" | "effective_draft" => {
                        if let Some(mode) = value.str() { modes.insert(mode.into()); }
                    }
                    "mtp_depth" | "effective_mtp_depth" => {
                        if let Some(depth) = depth(value) { depths.insert(depth); }
                    }
                    _ => visit(value, modes, depths),
                }
            }
        }
    }
    let mut modes = BTreeSet::new();
    let mut depths = BTreeSet::new();
    let mut echo_requests = 0;
    let mut modes_unreported = 0;
    let mut depths_unreported = 0;
    for record in records {
        let mut request_modes = BTreeSet::new();
        let mut request_depths = BTreeSet::new();
        visit(record.get("sampling_echo"), &mut request_modes, &mut request_depths);
        echo_requests += usize::from(!request_modes.is_empty() || !request_depths.is_empty());
        modes_unreported += usize::from(request_modes.is_empty());
        depths_unreported += usize::from(request_depths.is_empty());
        modes.extend(request_modes);
        depths.extend(request_depths);
    }
    o([
        ("observed_draft_modes", a(modes.into_iter().map(s).collect())),
        ("observed_mtp_depths", a(depths.into_iter().map(n).collect())),
        ("echo_requests", n(echo_requests)),
        ("unreported_requests", n(records.len() - echo_requests)),
        ("mode_unreported_requests", n(modes_unreported)),
        ("depth_unreported_requests", n(depths_unreported)),
        ("source", s("server-reported per-request sampling echoes; excludes immutable server numerics; not independent proof of effective drafting")),
    ])
}

/// An OFF label is permitted only when every request explicitly asked plain.
/// Health and response metadata are retained as reports, not drafting truth.
pub fn validate_requests(decoding: &Value, records: &[Value]) -> Result<(), String> {
    let expected = decoding.get("request_mode").str();
    for (index, record) in records.iter().enumerate() {
        let requested = record.get("requested_draft_mode").str();
        if requested != expected || (decoding.get("drafting").str() == Some("off") && requested != Some("plain")) {
            return Err(format!("request {} drafting provenance disagrees with receipt header", index + 1));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::value::{b, parse};

    fn health(depth: usize) -> Value {
        o([("vade", o([("numerics", o([("mtp_depth", n(depth))]))]))])
    }

    #[test]
    fn requested_policy_is_separate_from_verbatim_health_depth() {
        for reported in [n(0), n(3), s("3")] {
            let server = o([("vade", o([("numerics", o([("mtp_depth", reported.clone())]))]))]);
            for (mode, label) in [(None, "server-default"), (Some("plain"), "off"), (Some("strict"), "strict-requested")] {
                let actual = metadata(&server, mode).unwrap();
                assert_eq!(actual.get("request_mode"), &s(mode.unwrap_or("server")));
                assert_eq!(actual.get("drafting"), &s(label));
                assert_eq!(actual.get("server_mtp_depth"), &reported);
                assert_eq!(actual.get("server_mtp_depth_field"), &s("vade.numerics.mtp_depth"));
                assert_eq!(actual.get("server_mtp_depth_label"), &s(SERVER_DEPTH_LABEL));
                assert_eq!(actual.get("server_draft_mode"), &Value::Null);
                assert_eq!(actual.get("mtp_depth"), &Value::Null);
            }
        }
    }

    #[test]
    fn missing_or_zero_depth_never_implies_off_or_blocks_strict() {
        for server in [o([]), health(0)] {
            assert_eq!(metadata(&server, None).unwrap().get("drafting"), &s("server-default"));
            assert_eq!(metadata(&server, Some("strict")).unwrap().get("drafting"), &s("strict-requested"));
        }
        assert_eq!(metadata(&o([]), None).unwrap().get("server_mtp_depth"), &Value::Null);
        assert!(metadata(&health(3), Some("invalid")).is_err());
    }

    #[test]
    fn conflicting_health_and_bad_depth_cannot_label_a_receipt() {
        let mut server = health(3);
        server.put("mtp_enabled", b(false));
        assert!(metadata(&server, None).is_err());
        assert!(metadata(&o([("mtp_depth", n(-1))]), None).is_err());
        let explicit = o([("draft", s("strict")), ("mtp_depth", n(3))]);
        assert_eq!(metadata(&explicit, None).unwrap().get("server_draft_mode"), &s("strict"));
    }

    #[test]
    fn plain_echo_may_retain_server_depth_in_immutable_numerics() {
        let echo = parse(br#"{"vade":{"draft":"plain","mtp_depth":0,"draft_used":false,"model_draft_used":false,"numerics":{"mtp_depth":3}}}"#).unwrap();
        validate_echo(&echo, Some("plain")).unwrap();
        assert!(validate_echo(&echo, Some("strict")).is_err());
        let strict = parse(br#"{"vade":{"draft":"strict","mtp_depth":3,"draft_used":true}}"#).unwrap();
        validate_echo(&strict, Some("strict")).unwrap();
        validate_echo(&strict, None).unwrap();
        assert!(validate_echo(&strict, Some("plain")).is_err());
        assert!(validate_echo(&o([("model_draft_used", b(true))]), Some("plain")).is_err());
        for value in [s("unknown"), Value::Bool(true), n(-1)] {
            assert!(validate_echo(&o([("mtp_depth", value)]), Some("strict")).is_err());
        }
        assert!(validate_echo(&o([("draft", s("unknown"))]), Some("strict")).is_err());
    }

    #[test]
    fn observations_show_mixed_effective_modes_and_missing_echoes() {
        let records = parse(br#"[
            {"sampling_echo":{"vade":{"draft":"strict","mtp_depth":3,"numerics":{"mtp_depth":5}}}},
            {"sampling_echo":{"vade":{"draft":"plain","mtp_depth":0,"numerics":{"mtp_depth":5}}}},
            {"sampling_echo":{"vade":{"draft":"strict","mtp_depth":3}}},
            {"sampling_echo":{"vade":{"numerics":{"mtp_depth":5}}}},
            {"sampling_echo":{"sampling":{"draft":"auto"}}}
        ]"#).unwrap();
        let actual = observations(records.arr());
        assert_eq!(actual.get("observed_draft_modes"), &a(vec![s("auto"), s("plain"), s("strict")]));
        assert_eq!(actual.get("observed_mtp_depths"), &a(vec![n(0), n(3)]));
        assert_eq!(actual.get("echo_requests"), &n(4));
        assert_eq!(actual.get("unreported_requests"), &n(1));
        assert_eq!(actual.get("mode_unreported_requests"), &n(1));
        assert_eq!(actual.get("depth_unreported_requests"), &n(2));
    }

    #[test]
    fn off_requires_every_request_to_have_asked_plain() {
        let plain = metadata(&health(0), Some("plain")).unwrap();
        let request = |mode| o([("requested_draft_mode", s(mode))]);
        validate_requests(&plain, &[request("plain"), request("plain")]).unwrap();
        assert!(validate_requests(&plain, &[request("plain"), request("server")]).is_err());
        assert!(validate_requests(&plain, &[request("plain"), request("strict")]).is_err());
        assert!(validate_requests(&plain, &[o([])]).is_err());
        validate_requests(&metadata(&health(0), None).unwrap(), &[request("server")]).unwrap();
    }
}
