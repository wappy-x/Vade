use super::*;

const LOCAL: &str = "127.0.0.1:8787";

fn head(first: &str, fields: &[(&str, &str)]) -> Head {
    Head { first: first.into(), fields: fields.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect() }
}
fn post(fields: &[(&str, &str)]) -> Head {
    head("POST /v1/chat/completions HTTP/1.1", fields)
}
fn check(access: &Access, head: &Head) -> Result<Option<String>, Refusal> {
    access.check(head, LOCAL.parse().unwrap())
}
fn status(access: &Access, head: &Head) -> u16 {
    match check(access, head) {
        Ok(_) => 200,
        Err(Refusal::Status(code, _)) => code,
        Err(Refusal::Preflight(_)) => 204,
    }
}

#[test]
fn loopback_hosts_with_the_listener_port_pass_and_others_get_421() {
    let access = Access::default();
    for host in ["127.0.0.1:8787", "localhost:8787", "LOCALHOST:8787", "[::1]:8787", "localhost", "127.0.0.1", "[::1]"] {
        assert_eq!(status(&access, &post(&[("host", host)])), 200, "{host}");
    }
    for host in ["evil.example", "evil.example:8787", "localhost:8788", "127.0.0.1:80", "127.0.0.2:8787", "localhost.:8787", "127.0.0.1.nip.io:8787", "[::1]:1", "localhost:08787"] {
        assert_eq!(status(&access, &post(&[("host", host)])), 421, "{host}");
    }
    // The listener's own loopback address passes even when it is not 127.0.0.1.
    assert!(access.check(&post(&[("host", "127.0.0.2:9")]), "127.0.0.2:9".parse().unwrap()).is_ok());
    assert!(access.check(&post(&[("host", "[::1]:9")]), "[::1]:9".parse().unwrap()).is_ok());
}

#[test]
fn missing_or_duplicate_host_is_a_framing_error() {
    let access = Access::default();
    assert_eq!(status(&access, &post(&[])), 400);
    assert_eq!(status(&access, &post(&[("host", "localhost:8787"), ("host", "localhost:8787")])), 400);
    // Even HTTP/1.0 must name the listener; there is no anonymous fallback.
    assert_eq!(status(&access, &head("POST /v1/chat/completions HTTP/1.0", &[])), 400);
}

#[test]
fn allow_host_admits_exact_values_with_or_without_the_port() {
    let mut access = Access::default();
    access.allow_host("Vade.LAN").unwrap();
    access.allow_host("box.example:9000").unwrap();
    for host in ["vade.lan", "vade.lan:8787", "VADE.lan:8787", "box.example:9000"] {
        assert_eq!(status(&access, &post(&[("host", host)])), 200, "{host}");
    }
    for host in ["vade.lan:1", "box.example", "box.example:8787", "other.lan:8787"] {
        assert_eq!(status(&access, &post(&[("host", host)])), 421, "{host}");
    }
    for bad in ["", "a b", "a/b", "a@b", "x\r\ny"] {
        assert!(Access::default().allow_host(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn any_origin_is_refused_with_403_unless_allow_listed() {
    let access = Access::default();
    for origin in ["https://evil.example", "null", "http://localhost:8787", "http://127.0.0.1:8787"] {
        assert_eq!(status(&access, &post(&[("host", LOCAL), ("origin", origin)])), 403, "{origin}");
    }
    // Host is checked first: a rebinding page gets 421 regardless of its Origin.
    assert_eq!(status(&access, &post(&[("host", "evil.example:8787"), ("origin", "http://evil.example:8787")])), 421);
    assert_eq!(status(&access, &post(&[("host", LOCAL), ("origin", "null"), ("origin", "null")])), 400);
}

#[test]
fn allow_origin_admits_the_exact_origin_and_names_it_for_cors() {
    let mut access = Access::default();
    access.allow_origin("https://App.Example:3000").unwrap();
    access.allow_origin("null").unwrap();
    assert_eq!(check(&access, &post(&[("host", LOCAL), ("origin", "https://app.example:3000")])), Ok(Some("https://app.example:3000".into())));
    assert_eq!(check(&access, &post(&[("host", LOCAL), ("origin", "null")])), Ok(Some("null".into())));
    for origin in ["https://app.example", "http://app.example:3000", "https://app.example:3000.evil", "https://evil.example"] {
        assert_eq!(status(&access, &post(&[("host", LOCAL), ("origin", origin)])), 403, "{origin}");
    }
    assert_eq!(cors(Some("null")), "Access-Control-Allow-Origin: null\r\nVary: Origin\r\n");
    assert_eq!(cors(None), "");
    for bad in ["", "*", "evil.example", "https://", "https://a/b", "https://a/", "1http://a", "https://a b"] {
        assert!(Access::default().allow_origin(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn preflight_from_an_allowed_origin_is_answered_before_the_token() {
    let mut access = Access::default();
    access.allow_origin("https://app.example").unwrap();
    access.require_token("secret").unwrap();
    let preflight = head("OPTIONS /v1/chat/completions HTTP/1.1", &[
        ("host", LOCAL),
        ("origin", "https://app.example"),
        ("access-control-request-method", "POST"),
        ("access-control-request-headers", "authorization, content-type"),
    ]);
    let Err(Refusal::Preflight(lines)) = check(&access, &preflight) else { panic!("preflight expected") };
    assert!(lines.starts_with("Access-Control-Allow-Origin: https://app.example\r\nVary: Origin\r\n"));
    assert!(lines.contains("Access-Control-Allow-Methods: POST\r\n"));
    assert!(lines.contains("Access-Control-Allow-Headers: authorization, content-type\r\n"));
    // Header injection in the request list falls back to a fixed list.
    let hostile = head("OPTIONS / HTTP/1.1", &[("host", LOCAL), ("origin", "https://app.example"), ("access-control-request-method", "POST\r\nX: y"), ("access-control-request-headers", "a\r\nb")]);
    let Err(Refusal::Preflight(lines)) = check(&access, &hostile) else { panic!("preflight expected") };
    assert!(!lines.contains("X: y") && lines.contains("Allow-Methods: GET, POST\r\n") && lines.contains("Allow-Headers: authorization, content-type\r\n"));
    // A preflight from a foreign Origin is refused like any other request.
    let foreign = head("OPTIONS / HTTP/1.1", &[("host", LOCAL), ("origin", "https://evil.example"), ("access-control-request-method", "POST")]);
    assert_eq!(status(&access, &foreign), 403);
    let mut out = Vec::new();
    write_preflight(&mut out, "X-Vade-Gateway: v1\r\n", &cors(Some("null"))).unwrap();
    assert!(out.starts_with(b"HTTP/1.1 204 No Content\r\nX-Vade-Gateway: v1\r\nAccess-Control-Allow-Origin: null\r\n"));
}

#[test]
fn token_requires_one_exact_bearer_credential() {
    let mut access = Access::default();
    assert!(!access.consumes_authorization());
    access.require_token("s3cret-Token_1").unwrap();
    assert!(access.consumes_authorization());
    assert_eq!(status(&access, &post(&[("host", LOCAL), ("authorization", "Bearer s3cret-Token_1")])), 200);
    assert_eq!(status(&access, &post(&[("host", LOCAL), ("authorization", "bearer s3cret-Token_1")])), 200);
    for value in ["", "Bearer", "Bearer ", "Bearer s3cret-Token_", "Bearer s3cret-Token_12", "Basic s3cret-Token_1", "s3cret-Token_1"] {
        assert_eq!(status(&access, &post(&[("host", LOCAL), ("authorization", value)])), 401, "{value:?}");
    }
    assert_eq!(status(&access, &post(&[("host", LOCAL)])), 401);
    let twice = post(&[("host", LOCAL), ("authorization", "Bearer s3cret-Token_1"), ("authorization", "Bearer s3cret-Token_1")]);
    assert_eq!(status(&access, &twice), 401);
    // Host and Origin refusals come first and never reveal whether a token is set.
    assert_eq!(status(&access, &post(&[("host", "evil.example")])), 421);
    assert_eq!(status(&access, &post(&[("host", LOCAL), ("origin", "https://evil.example")])), 403);
    for bad in ["", "a b", "tab\there", "é", &"x".repeat(1025)] {
        assert!(Access::default().require_token(bad).is_err());
    }
    // Without a token, Authorization is the upstream's business.
    assert_eq!(status(&Access::default(), &post(&[("host", LOCAL), ("authorization", "Bearer anything")])), 200);
}

#[test]
fn token_environment_fallback_is_overridden_by_the_flag() {
    let mut flag = Access::default();
    flag.require_token("from-flag").unwrap();
    // The flag wins without consulting the environment.
    flag.token_from_env().unwrap();
    assert_eq!(status(&flag, &post(&[("host", LOCAL), ("authorization", "Bearer from-flag")])), 200);
    if std::env::var_os(TOKEN_ENV).is_none() {
        let mut none = Access::default();
        none.token_from_env().unwrap();
        assert!(!none.consumes_authorization());
    }
}

#[test]
fn token_compare_has_fixed_work_and_both_gateway_helps_name_the_safe_source() {
    // The compare runs over the configured token for every length, like the engine key check.
    assert!(same(b"secret-token", b"secret-token"));
    for wrong in [&b""[..], b"secret-toke", b"secret-tokenx", b"secret-tokeN", b"\0ecret-token"] {
        assert!(!same(b"secret-token", wrong), "{wrong:?}");
    }
    assert!(!same(b"", b""), "an empty token never matches");
    // --token is visible to every local user in the process list; help says so.
    assert!(super::super::BROWSER_GUARD_HELP.contains("Prefer VADE_GATEWAY_TOKEN: other local users can read a --token value in the process list."));
}
