//! Opt-in scheduling hints and lossless output bounds. Never edits input messages.
use super::*;
use crate::json::{self, View};
use std::borrow::Cow;

#[derive(Clone, Copy, Default)]
pub(super) struct Flags {
    pub stable: bool,
    pub affinity: bool,
    pub clamp: bool,
    pub greedy: bool,
    pub no_sticky: bool,
}
#[derive(Clone)]
struct Limits {
    context: u64,
    input: u64,
    output: u64,
    model: String,
}
pub(super) struct Runtime {
    origins: Vec<Origin>,
    limits: Vec<Option<Limits>>,
    flags: Flags,
    router: Router,
}
pub(super) struct Prepared<'a> {
    pub index: usize,
    pub lease: Option<Lease>,
    pub body: Cow<'a, [u8]>,
    pub report: String,
}
impl Runtime {
    pub(super) fn new(urls: &[String], flags: Flags) -> io::Result<Self> {
        let origins = urls
            .iter()
            .map(|url| Origin::parse(url))
            .collect::<io::Result<Vec<_>>>()?;
        if origins.is_empty() || origins.len() > 64 {
            return Err(transport::invalid("configure 1..64 upstreams"));
        }
        let mut limits = Vec::new();
        for origin in &origins {
            limits.push(if flags.clamp {
                let (status, body) = fetch(origin, "GET", "/v1/capabilities", b"", None)?;
                if status != 200 {
                    return Err(transport::invalid("capabilities request failed"));
                }
                let v =
                    view(&body).ok_or_else(|| transport::invalid("invalid capabilities JSON"))?;
                if text(v, "object").as_deref() != Some("engine.capabilities")
                    || text(v, "contract_version").as_deref() != Some("vade.engine.v1")
                    || text(v, "admission").as_deref() != Some("/v1/admission")
                    || text(v, "admission_kind").as_deref() != Some("context_fit")
                {
                    return Err(transport::invalid(
                        "clamping requires the engine admission contract",
                    ));
                }
                let context = positive(v.get("context_limit"))?;
                let input = positive(v.get("input_limit"))?;
                let output = positive(v.get("output_limit"))?;
                if input >= context || output > context {
                    return Err(transport::invalid("inconsistent capability limits"));
                }
                Some(Limits {
                    context,
                    input,
                    output,
                    model: text(v, "model")
                        .ok_or_else(|| transport::invalid("capabilities model missing"))?,
                })
            } else {
                None
            });
        }
        Ok(Self {
            router: Router::new(origins.len(), !flags.no_sticky),
            origins,
            limits,
            flags,
        })
    }
    #[cfg(test)]
    pub(super) fn single(origin: Origin) -> Self {
        Self {
            origins: vec![origin],
            limits: vec![None],
            flags: Flags::default(),
            router: Router::new(1, true),
        }
    }
    pub(super) fn origin(&self, index: usize) -> &Origin {
        &self.origins[index]
    }
    pub(super) fn origins(&self) -> &[Origin] {
        &self.origins
    }
    /// Reserve an upstream outside `excluded`: rendezvous affinity when enabled,
    /// otherwise the fewest in-flight requests with conversation stickiness.
    pub(super) fn route(&self, path: &str, head: &Head, body: &[u8], excluded: &[usize]) -> Option<Lease> {
        let pinned = self.flags.affinity.then(|| {
            let parsed = view(body);
            let request = if path.split('?').next() == Some("/v1/admission") { parsed.and_then(|v| v.get("request")) } else { parsed };
            affinity(&key(request, head), &self.origins)
        });
        let conversation = if self.origins.len() > 1 { super::route::conversation_key(body) } else { None };
        self.router.acquire(pinned, conversation, excluded)
    }
    pub(super) fn prepare<'a>(
        &self,
        method: &str,
        path: &str,
        head: &Head,
        body: &'a [u8],
    ) -> Prepared<'a> {
        self.prepare_with(
            method,
            path,
            head,
            body,
            |index, endpoint, request, output, limits| {
                self.admit(index, endpoint, head, request, output, limits)
            },
        )
    }
    fn prepare_with<'a>(
        &self,
        method: &str,
        path: &str,
        head: &Head,
        body: &'a [u8],
        mut admit: impl FnMut(usize, &str, &[u8], u64, &Limits) -> io::Result<u64>,
    ) -> Prepared<'a> {
        let lease = self.route(path, head, body, &[]);
        let index = lease.as_ref().map_or(0, |lease| lease.index);
        let (patched, mut report) = self.prepare_policy(index, method, path, head, body, &mut admit);
        // Finalize telemetry after sampling, keeping routing and admission on client bytes.
        let (body, defaults_added) =
            super::super::sampling::apply_to(method, path, head, body, patched);
        report.sampling_defaults = defaults_added;
        Prepared {
            index: report.index,
            lease,
            body,
            report: report.json(),
        }
    }
    fn prepare_policy<'a>(
        &self,
        index: usize,
        method: &str,
        path: &str,
        head: &Head,
        body: &'a [u8],
        admit: &mut impl FnMut(usize, &str, &[u8], u64, &Limits) -> io::Result<u64>,
    ) -> (Cow<'a, [u8]>, Report) {
        if !self.flags.stable && !self.flags.affinity && !self.flags.greedy && !self.flags.clamp {
            return (Cow::Borrowed(body), Report::new(index));
        }
        let parsed = view(body);
        let key = if self.flags.affinity || self.flags.stable {
            key(
                parsed.and_then(|v| {
                    if path == "/v1/admission" {
                        v.get("request")
                    } else {
                        Some(v)
                    }
                }),
                head,
            )
        } else {
            String::new()
        };
        let generation = method == "POST"
            && matches!(
                path.split('?').next().unwrap_or(path),
                "/v1/chat/completions" | "/v1/responses"
            );
        let mut report = Report::new(index);
        let Some(value) = parsed.filter(|v| {
            generation
                && v.fields().is_some()
                && head
                    .get("content-encoding")
                    .is_none_or(|v| v.eq_ignore_ascii_case("identity"))
        }) else {
            return (Cow::Borrowed(body), report);
        };
        if !self.flags.stable && !self.flags.greedy && !self.flags.clamp {
            return (Cow::Borrowed(body), report);
        }
        let mut patched = Edits {
            body,
            value,
            fields: Vec::new(),
        };
        if self.flags.stable {
            if value.get("prompt_cache_key").is_none() {
                patched.set("prompt_cache_key", json::quote(&key));
                report.key = true;
            }
            if value.get("cache").is_none() {
                patched.set("cache", "true".into());
                report.cache = true;
            }
        }
        if let Some(limits) = self.limits[index]
            .as_ref()
            .filter(|l| text(value, "model").as_deref() == Some(&l.model))
            && Self::clamp(index, path, &mut patched, limits, &mut report, admit).is_err()
        {
            report.error = Some("admission_unavailable");
        }
        if report.oversized.is_some() {
            patched.fields.clear();
            report.clamps.clear();
            report.key = false;
            report.cache = false;
        }
        (patched.finish(), report)
    }
    fn clamp(
        index: usize,
        path: &str,
        body: &mut Edits<'_>,
        limits: &Limits,
        report: &mut Report,
        admit: &mut impl FnMut(usize, &str, &[u8], u64, &Limits) -> io::Result<u64>,
    ) -> io::Result<()> {
        let endpoint = path.split('?').next().unwrap_or(path);
        let preferred = if endpoint == "/v1/responses" {
            "max_output_tokens"
        } else {
            "max_tokens"
        };
        clamp_fields(body, limits.output, preferred, "output_limit", report)?;
        let reserved = reservation(body, preferred)?;
        let count = admit(index, endpoint, &body.finish(), reserved, limits)?;
        if count > limits.input {
            report.clamps.clear();
            report.key = false;
            report.cache = false;
            report.oversized = Some((count, limits.input));
            return Ok(());
        }
        let available = limits
            .context
            .checked_sub(count)
            .filter(|n| *n > 0)
            .ok_or_else(|| transport::invalid("no output room"))?;
        if reserved > available {
            clamp_fields(body, available, preferred, "context_reservation", report)?;
            let final_count = admit(
                index,
                endpoint,
                &body.finish(),
                reservation(body, preferred)?,
                limits,
            )?;
            if final_count != count {
                return Err(transport::invalid(
                    "admission count changed after output clamp",
                ));
            }
        }
        Ok(())
    }
    fn admit(
        &self,
        index: usize,
        path: &str,
        head: &Head,
        body: &[u8],
        output: u64,
        limits: &Limits,
    ) -> io::Result<u64> {
        let request =
            std::str::from_utf8(body).map_err(|_| transport::invalid("invalid request JSON"))?;
        let envelope = format!(
            "{{\"contract_version\":\"vade.engine.v1\",\"endpoint\":{},\"output_tokens\":{output},\"request\":{request}}}",
            json::quote(path)
        );
        let (status, bytes) = fetch(
            &self.origins[index],
            "POST",
            "/v1/admission",
            envelope.as_bytes(),
            Some(head),
        )?;
        let v = view(&bytes)
            .filter(|_| status == 200)
            .ok_or_else(|| transport::invalid("admission request failed"))?;
        if text(v, "object").as_deref() != Some("engine.admission")
            || text(v, "contract_version").as_deref() != Some("vade.engine.v1")
            || text(v, "admission_kind").as_deref() != Some("context_fit")
            || text(v, "model").as_deref() != Some(&limits.model)
            || integer(v.get("output_tokens")) != Some(output)
            || integer(v.get("context_limit")) != Some(limits.context)
            || integer(v.get("input_limit")) != Some(limits.input)
            || integer(v.get("output_limit")) != Some(limits.output)
            || v.get("reserves_resources")
                .is_none_or(|v| v.raw() != "false")
        {
            return Err(transport::invalid("admission contract mismatch"));
        }
        let prompt = integer(v.get("prompt_tokens"))
            .ok_or_else(|| transport::invalid("admission prompt count missing"))?;
        let fits = prompt <= limits.input
            && output <= limits.output
            && prompt
                .checked_add(output)
                .is_some_and(|n| n <= limits.context);
        if v.get("admitted")
            .is_none_or(|v| v.raw() != if fits { "true" } else { "false" })
        {
            return Err(transport::invalid("inconsistent admission decision"));
        }
        Ok(prompt)
    }
}
fn view(bytes: &[u8]) -> Option<View<'_>> {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| json::view(s).ok())
}
fn text(v: View<'_>, key: &str) -> Option<String> {
    v.get(key).and_then(View::as_str).map(|s| s.into_owned())
}
/// A plain unsigned JSON integer (digits only, no sign, fraction or exponent).
pub(super) fn integer(v: Option<View<'_>>) -> Option<u64> {
    v.filter(|v| !v.raw().is_empty() && v.raw().bytes().all(|b| b.is_ascii_digit()))?
        .raw()
        .parse()
        .ok()
}
fn positive(v: Option<View<'_>>) -> io::Result<u64> {
    integer(v)
        .filter(|n| *n > 0)
        .ok_or_else(|| transport::invalid("positive token limit required"))
}
fn reservation(body: &Edits<'_>, preferred: &str) -> io::Result<u64> {
    [
        preferred,
        "max_completion_tokens",
        "max_tokens",
        "max_output_tokens",
    ]
    .iter()
    .find_map(|key| body.get(key))
    .map(|v| positive(Some(v)))
    .unwrap_or_else(|| Err(transport::invalid("missing reservation")))
}
fn clamp_fields(
    body: &mut Edits<'_>,
    limit: u64,
    preferred: &'static str,
    reason: &str,
    report: &mut Report,
) -> io::Result<()> {
    let mut found = false;
    for field in ["max_tokens", "max_output_tokens", "max_completion_tokens"] {
        if let Some(old) = body.get(field) {
            let n = positive(Some(old))?;
            found = true;
            if n > limit {
                report.clamps.push(format!(
                    "{{\"field\":{},\"from\":{n},\"to\":{limit},\"reason\":{}}}",
                    json::quote(field),
                    json::quote(reason)
                ));
                body.set(field, limit.to_string());
            }
        }
    }
    if !found {
        report.clamps.push(format!(
            "{{\"field\":{},\"from\":null,\"to\":{limit},\"reason\":{}}}",
            json::quote(preferred),
            json::quote(reason)
        ));
        body.set(preferred, limit.to_string());
    }
    Ok(())
}
// Keep decisions on validated client spans plus explicit policy replacements.
// Rendering never reparses a decorated body against the client input limit.
struct Edits<'a> {
    body: &'a [u8],
    value: View<'a>,
    fields: Vec<(&'static str, String)>,
}
impl<'a> Edits<'a> {
    fn get(&self, field: &str) -> Option<View<'_>> {
        self.fields
            .iter()
            .find(|(name, _)| *name == field)
            .and_then(|(_, replacement)| json::view(replacement).ok())
            .or_else(|| self.value.get(field))
    }
    fn set(&mut self, field: &'static str, replacement: String) {
        if let Some((_, old)) = self.fields.iter_mut().find(|(name, _)| *name == field) {
            *old = replacement;
        } else {
            self.fields.push((field, replacement));
        }
    }
    fn finish(&self) -> Cow<'a, [u8]> {
        if self.fields.is_empty() {
            return Cow::Borrowed(self.body);
        }
        let base = self.body.as_ptr() as usize;
        let mut spans = Vec::new();
        let mut extra = String::new();
        let mut comma = self.value.fields().unwrap().next().is_some();
        for (field, replacement) in &self.fields {
            if let Some(old) = self.value.get(field) {
                let start = old.raw().as_ptr() as usize - base;
                spans.push((start, start + old.raw().len(), replacement.as_str()));
            } else {
                if comma {
                    extra.push(',');
                }
                extra.push_str(&format!("{}:{replacement}", json::quote(field)));
                comma = true;
            }
        }
        let end = self.value.raw().as_ptr() as usize - base + self.value.raw().len() - 1;
        spans.push((end, end, &extra));
        spans.sort_unstable_by_key(|(start, _, _)| std::cmp::Reverse(*start));
        let mut out = self.body.to_vec();
        for (start, end, replacement) in spans {
            out.splice(start..end, replacement.bytes());
        }
        Cow::Owned(out)
    }
}
// FNV-1a is a non-secret scheduling hash. Engine prefix validation owns correctness.
pub(super) fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, b| {
        (h ^ *b as u64).wrapping_mul(0x100000001b3)
    })
}
fn key(v: Option<View<'_>>, head: &Head) -> String {
    if let Some(key) = v.and_then(|v| text(v, "prompt_cache_key")) {
        return key;
    }
    for header in [
        "session_id",
        "x-session-id",
        "x-codex-session-id",
        "session-id",
    ] {
        if let Some(value) = head.get(header) {
            return format!("vade-{:016x}", hash(value.as_bytes()));
        }
    }
    let mut prefix = Vec::new();
    if let Some(v) = v {
        for field in ["model", "instructions", "tools"] {
            if let Some(value) = v.get(field) {
                prefix.extend_from_slice(field.as_bytes());
                prefix.push(0);
                prefix.extend_from_slice(value.raw().as_bytes());
            }
        }
        if let Some(messages) = v.get("messages").or_else(|| v.get("input")) {
            if let Some(array) = messages.array() {
                for message in array {
                    let role = text(message, "role");
                    if matches!(role.as_deref(), Some("system" | "developer" | "user")) {
                        prefix.push(0);
                        prefix.extend_from_slice(message.raw().as_bytes());
                    }
                    if role.as_deref() == Some("user") {
                        break;
                    }
                }
            } else {
                prefix.extend_from_slice(messages.raw().as_bytes());
            }
        }
    }
    format!("vade-{:016x}", hash(&prefix))
}
fn affinity(key: &str, origins: &[Origin]) -> usize {
    origins
        .iter()
        .enumerate()
        .max_by_key(|(_, origin)| {
            hash(format!("{key}\0{}{}", origin.endpoint.authority, origin.prefix).as_bytes())
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}
struct Report {
    index: usize,
    key: bool,
    cache: bool,
    sampling_defaults: bool,
    clamps: Vec<String>,
    oversized: Option<(u64, u64)>,
    error: Option<&'static str>,
}
impl Report {
    fn new(index: usize) -> Self {
        Self {
            index,
            key: false,
            cache: false,
            sampling_defaults: false,
            clamps: Vec::new(),
            oversized: None,
            error: None,
        }
    }
    fn json(&self) -> String {
        format!(
            "{{\"upstream\":{},\"cache_key_added\":{},\"cache_default_added\":{},\"greedy_added\":{},\"sampling_defaults_added\":{},\"clamps\":[{}],\"oversized_input\":{},\"error\":{}}}",
            self.index,
            self.key,
            self.cache,
            self.sampling_defaults,
            if self.sampling_defaults {
                super::super::sampling::DEFAULTS_JSON
            } else {
                "null"
            },
            self.clamps.join(","),
            self.oversized.map_or_else(
                || "null".into(),
                |(tokens, limit)| format!("{{\"tokens\":{tokens},\"input_limit\":{limit}}}")
            ),
            self.error.map_or_else(|| "null".into(), json::quote)
        )
    }
}
fn fetch(
    origin: &Origin,
    method: &str,
    path: &str,
    body: &[u8],
    forwarded: Option<&Head>,
) -> io::Result<(u16, Vec<u8>)> {
    let mut socket = origin.connect()?;
    let parsed = Head {
        first: format!("{method} {path} HTTP/1.1"),
        fields: forwarded.map_or_else(
            || vec![("content-type".into(), "application/json".into())],
            |head| head.fields.clone(),
        ),
    };
    let lines = std::iter::once(parsed.first.as_bytes().to_vec())
        .chain(
            parsed
                .fields
                .iter()
                .map(|(key, value)| format!("{key}: {value}").into_bytes()),
        )
        .collect();
    let head = WireHead { parsed, lines };
    head.forward_upstream(
        &mut socket,
        &format!("{method} {} HTTP/1.1", origin.target(path)),
        &format!(
            "Host: {}\r\nContent-Length: {}\r\nConnection: close\r\n",
            origin.endpoint.authority,
            body.len()
        ),
    )?;
    socket.write_all(body)?;
    socket.flush()?;
    let mut reader = BufReader::new(socket);
    let head = loop {
        let head = WireHead::read(&mut reader)?;
        if status(&head.parsed)? >= 200 {
            break head;
        }
    };
    let code = status(&head.parsed)?;
    let mut body = BodyReader::new(reader, &head.body_head(), false, 64 * 1024)?;
    let mut bytes = Vec::new();
    body.read_to_end(&mut bytes)?;
    Ok((code, bytes))
}

#[cfg(test)]
#[path = "invisible_policy_tests.rs"]
mod tests;
