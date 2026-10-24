//! Exact gateway defaults, inserted without reserializing caller fields.
use crate::{json, transport::Head};
use std::borrow::Cow;

// One source of policy shared with engine-model, without a Cargo dependency.
#[path = "../../../engine/crates/model/src/sampling_policy.rs"]
mod policy;
pub(super) const DEFAULTS_JSON: &str = policy::GREEDY_DEFAULTS_JSON;
pub(super) const SAMPLING_FIELDS: &str = policy::FIELDS;

#[cfg(test)]
pub(super) const EXPLICIT_SAMPLING_EXAMPLES: &[(&str, &str)] = &[
    ("temperature", "0.125"),
    ("seed", "42"),
    ("top_p", "9.00e-1"),
    ("top_k", "10"),
    ("min_p", "0.05"),
    ("typical_p", "0.9"),
    ("frequency_penalty", "0.25"),
    ("presence_penalty", "0.25"),
    ("repetition_penalty", "1.1"),
    ("logit_bias", r#"{"42":-1.20e0}"#),
    ("n", "2"),
    ("best_of", "3"),
    ("sampling", "{}"),
    ("generation_config", "{}"),
    (r"temperat\u0075re", "0.75"),
];

pub(super) fn capabilities_header(method: &str, path: &str) -> String {
    if method == "GET" && path.split('?').next() == Some("/v1/capabilities") {
        format!(
            "X-Vade-Sampling-Defaults: {DEFAULTS_JSON}\r\nX-Vade-Sampling-Defaults-When: no sampling fields present; otherwise unchanged (engine defaults)\r\nX-Vade-Sampling-Fields: {SAMPLING_FIELDS}\r\n"
        )
    } else {
        String::new()
    }
}

pub(super) fn apply<'a>(method: &str, path: &str, head: &Head, body: &'a [u8]) -> Cow<'a, [u8]> {
    apply_to(method, path, head, body, Cow::Borrowed(body)).0
}

// Decide from client bytes, then append to the effective, policy-edited object.
// Never parse that enlarged body against the client-side JSON size limit.
// Return whether the complete set of defaults was inserted for policy telemetry.
pub(super) fn apply_to<'a>(
    method: &str,
    path: &str,
    head: &Head,
    original: &[u8],
    body: Cow<'a, [u8]>,
) -> (Cow<'a, [u8]>, bool) {
    if method != "POST"
        || !matches!(
            path.split('?').next().unwrap_or(path),
            "/v1/completions"
                | "/completions"
                | "/v1/chat/completions"
                | "/chat/completions"
                | "/v1/responses"
                | "/responses"
                | "/v1/messages"
                | "/messages"
        )
        || head
            .get("content-encoding")
            .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
    {
        return (body, false);
    }
    let Some(value) = std::str::from_utf8(original)
        .ok()
        .and_then(|s| json::view(s).ok())
        .filter(|v| v.fields().is_some())
    else {
        return (body, false);
    };
    // Presence, including null and escaped field names, suppresses all defaults.
    if value
        .fields()
        .unwrap()
        .any(|(name, _)| policy::is_field(name.as_ref()))
    {
        return (body, false);
    }
    // Policy edits only replace values or add fields to this validated object.
    let end = body.iter().rposition(|b| !b.is_ascii_whitespace()).unwrap();
    debug_assert_eq!(body[end], b'}');
    let comma = body[..end].iter().rfind(|b| !b.is_ascii_whitespace()) != Some(&b'{');
    let extra = format!(
        "{}{}",
        if comma { "," } else { "" },
        &DEFAULTS_JSON[1..DEFAULTS_JSON.len() - 1]
    );
    let mut patched = Vec::with_capacity(body.len() + extra.len());
    patched.extend_from_slice(&body[..end]);
    patched.extend_from_slice(extra.as_bytes());
    patched.extend_from_slice(&body[end..]);
    (Cow::Owned(patched), true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head() -> Head {
        Head {
            first: String::new(),
            fields: Vec::new(),
        }
    }

    #[test]
    fn unsampled_objects_get_complete_neutral_defaults_without_reserializing() {
        for path in [
            "/v1/completions",
            "/completions",
            "/v1/chat/completions",
            "/chat/completions",
            "/v1/responses?x=1",
            "/responses",
            "/v1/messages",
            "/messages",
        ] {
            for original in [
                "{}",
                " { }\n",
                r#"{ "model":"mock", "extension":1.20e0, "input":{"seed":42} }"#,
            ] {
                let patched = apply("POST", path, &head(), original.as_bytes());
                let value = json::view(std::str::from_utf8(&patched).unwrap()).unwrap();
                for (name, expected) in
                    [("temperature", "0"), ("seed", "20260905"), ("top_p", "1.0"),
                     ("top_k", "0"), ("repetition_penalty", "1.0"),
                     ("presence_penalty", "0.0"), ("frequency_penalty", "0.0")]
                {
                    assert_eq!(value.get(name).unwrap().raw(), expected);
                }
                let end = original.rfind('}').unwrap();
                assert!(patched.starts_with(&original.as_bytes()[..end]));
                assert!(patched.ends_with(&original.as_bytes()[end..]));
                assert_eq!(
                    apply("POST", path, &head(), &patched).as_ref(),
                    patched.as_ref()
                );
            }
        }
    }

    #[test]
    fn any_explicit_sampling_field_suppresses_all_defaults_byte_for_byte() {
        for path in [
            "/v1/completions",
            "/completions",
            "/v1/chat/completions",
            "/chat/completions",
            "/v1/responses?x=1",
            "/responses",
            "/v1/messages",
            "/messages",
        ] {
            for &(field, _) in EXPLICIT_SAMPLING_EXAMPLES {
                for explicit in ["null", "0", "-1", "1.20e0", "{}"] {
                    let original = format!(" {{ \"{field}\":{explicit},\"extension\":1.00 }}\n");
                    let patched = apply("POST", path, &head(), original.as_bytes());
                    assert_eq!(patched.as_ref(), original.as_bytes(), "{path}: {field}");
                    assert!(matches!(patched, Cow::Borrowed(_)));
                }
            }
        }
    }

    #[test]
    fn capabilities_explain_the_presence_rule_and_every_known_sampling_field() {
        let header = capabilities_header("GET", "/v1/capabilities?x=1");
        assert!(header.contains(&format!("X-Vade-Sampling-Defaults: {DEFAULTS_JSON}\r\n")));
        assert!(header.contains("X-Vade-Sampling-Defaults-When: no sampling fields present; otherwise unchanged (engine defaults)\r\n"));
        assert!(header.contains(&format!("X-Vade-Sampling-Fields: {SAMPLING_FIELDS}\r\n")));
        assert!(capabilities_header("POST", "/v1/responses").is_empty());
    }

    #[test]
    fn non_generation_and_opaque_bodies_are_unchanged() {
        for (method, path, body) in [
            ("POST", "/v1/admission", b"{}".as_slice()),
            ("POST", "/v1/messages/count_tokens", b"{}".as_slice()),
            ("POST", "/custom", b"{}".as_slice()),
            ("GET", "/v1/capabilities", b"".as_slice()),
            ("POST", "/v1/responses", b"\xffopaque".as_slice()),
        ] {
            assert_eq!(apply(method, path, &head(), body).as_ref(), body);
        }
        let compressed = Head {
            fields: vec![("content-encoding".into(), "gzip".into())],
            ..head()
        };
        assert_eq!(
            apply("POST", "/v1/responses", &compressed, b"{}").as_ref(),
            b"{}"
        );
    }
}
