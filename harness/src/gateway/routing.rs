//! Cache locality is a scheduling hint, never a cache-hit or security claim.
//! Original std-only router; see docs/gateway-routing.md for contract and limits.
use crate::{
    contract::Capabilities,
    json,
    transport::{self, ConnectionPool, Endpoint, Head, invalid},
};
use std::{
    borrow::Cow,
    collections::{HashMap, hash_map::RandomState},
    hash::{BuildHasher, Hash, Hasher},
    io::{self, Read},
    sync::{Arc, Mutex, MutexGuard, atomic::AtomicBool},
    time::{Duration, Instant},
};

pub const MAX_ENDPOINTS: usize = 64;
const MAX_METADATA: usize = 64 * 1024;
const PREFIX_BYTES: usize = 16 * 1024;

#[derive(Clone, Debug)]
pub struct RoutingConfig {
    pub affinity_capacity: usize,
    pub affinity_ttl: Duration,
    pub endpoint_concurrency: usize,
    pub capability_ttl: Duration,
    pub probe_interval: Duration,
    pub failure_cooldown: Duration,
}
impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            affinity_capacity: 4096,
            affinity_ttl: Duration::from_secs(15 * 60),
            endpoint_concurrency: 8,
            capability_ttl: Duration::from_secs(30),
            probe_interval: Duration::from_secs(10),
            failure_cooldown: Duration::from_secs(5),
        }
    }
}
impl RoutingConfig {
    fn validate(&self) -> io::Result<()> {
        if !(1..=65_536).contains(&self.affinity_capacity)
            || !(1..=16).contains(&self.endpoint_concurrency)
            || self.affinity_ttl.is_zero()
            || self.affinity_ttl > Duration::from_secs(86_400)
            || self.probe_interval.is_zero()
            || self.probe_interval > self.capability_ttl
            || self.capability_ttl > Duration::from_secs(300)
            || self.failure_cooldown.is_zero()
            || self.failure_cooldown > Duration::from_secs(300)
        {
            return Err(invalid("gateway routing limits are invalid"));
        }
        Ok(())
    }
}

/// Endpoint identifiers are zero-based configuration indices. No URLs, model
/// names, credentials, prompt text or affinity hashes are included in metrics.
#[derive(Clone, Debug, Default)]
pub struct EndpointStats {
    pub endpoint: usize,
    pub group: Option<usize>,
    pub healthy: bool,
    pub cooling_down: bool,
    pub in_flight: usize,
    pub requests: u64,
    pub failures: u64,
    pub overloads: u64,
    pub probe_failures: u64,
    pub affinity_hits: u64,
    pub fallbacks: u64,
    pub usage_responses: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
}
impl EndpointStats {
    pub fn cached_ratio(&self) -> Option<f64> {
        (self.prompt_tokens > 0).then(|| self.cached_tokens as f64 / self.prompt_tokens as f64)
    }
}
#[derive(Clone, Debug)]
pub struct RoutingStats {
    pub affinity_entries: usize,
    pub affinity_capacity: usize,
    pub endpoints: Vec<EndpointStats>,
}
impl RoutingStats {
    pub fn to_json(&self) -> String {
        let endpoints = self.endpoints.iter().map(|s| {
            format!(
                "{{\"endpoint\":{},\"group\":{},\"healthy\":{},\"cooling_down\":{},\"in_flight\":{},\"requests\":{},\"failures\":{},\"overloads\":{},\"probe_failures\":{},\"affinity_hits\":{},\"fallbacks\":{},\"usage_responses\":{},\"prompt_tokens\":{},\"cached_tokens\":{},\"cached_tokens_ratio\":{}}}",
                s.endpoint, s.group.map_or("null".into(), |v| v.to_string()), s.healthy,
                s.cooling_down, s.in_flight, s.requests, s.failures, s.overloads,
                s.probe_failures, s.affinity_hits, s.fallbacks, s.usage_responses,
                s.prompt_tokens, s.cached_tokens,
                s.cached_ratio().map_or("null".into(), |v| format!("{v:.8}"))
            )
        }).collect::<Vec<_>>().join(",");
        format!(
            "{{\"object\":\"vade.gateway.stats\",\"affinity_entries\":{},\"affinity_capacity\":{},\"endpoints\":[{}]}}",
            self.affinity_entries, self.affinity_capacity, endpoints
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Signature {
    // Parsing accepts only the known contract version; model plus all known
    // fingerprints, limits and contract flags must match. Instance cache domain
    // differs between replicas and is tracked separately for invalidation.
    caps: Capabilities,
    responses: bool,
    completions: bool,
    messages: bool,
    count_tokens: bool,
}
impl Signature {
    fn differences(&self, other: &Self) -> Vec<&'static str> {
        let mut fields = Vec::new();
        macro_rules! compare {
            ($a:expr, $b:expr, $prefix:literal, $($field:ident),+) => {
                $(if $a.$field != $b.$field { fields.push(concat!($prefix, stringify!($field))); })+
            };
        }
        compare!(self.caps, other.caps, "", model, build_hash, context_limit, input_limit,
            output_limit, tools, thinking, default_thinking, cache_scope, prefix_cache,
            cache_granularity_tokens, prompt_cache_key, exact_admission, kv_checkpoints);
        if let (Some(a), Some(b)) = (&self.caps.identity, &other.caps.identity) {
            if a.weights_fingerprint != b.weights_fingerprint { fields.push("identity.weights.fingerprint"); }
            if a.weights_fingerprint_kind != b.weights_fingerprint_kind { fields.push("identity.weights.fingerprint_kind"); }
            compare!(a, b, "identity.", tokenizer_fingerprint, template_fingerprint);
        }
        if self.responses != other.responses || self.completions != other.completions
            || self.messages != other.messages || self.count_tokens != other.count_tokens
            || self.caps.chat_completions != other.caps.chat_completions {
            fields.push("endpoints");
        }
        fields
    }

    fn parse(bytes: &str) -> io::Result<(Self, Option<String>)> {
        let mut caps = Capabilities::parse(bytes)?
            .ok_or_else(|| invalid("cluster requires recognized capabilities"))?;
        if caps.identity.is_none() {
            return Err(invalid("cluster requires complete model fingerprints"));
        }
        let endpoints = json::view(bytes)?.get("endpoints");
        let advertises = |path| {
            endpoints
                .and_then(json::View::array)
                .is_some_and(|mut items| items.any(|v| v.as_str().is_some_and(|v| v == path)))
        };
        let responses = advertises("/v1/responses");
        let completions = advertises("/v1/completions");
        let messages = advertises("/v1/messages");
        let count_tokens = advertises("/v1/messages/count_tokens");
        if !caps.chat_completions && !responses && !completions && !messages {
            return Err(invalid(
                "cluster endpoint advertises no supported generation route",
            ));
        }
        let domain = caps.cache_domain.take();
        Ok((
            Self {
                caps,
                responses,
                completions,
                messages,
                count_tokens,
            },
            domain,
        ))
    }
}
struct Node {
    probed: bool,
    group: Option<usize>,
    domain: Option<String>,
    epoch: u64,
    health_serial: u64,
    healthy_until: Option<Instant>,
    cooldown_until: Option<Instant>,
    stats: EndpointStats,
}
impl Node {
    fn healthy(&self, now: Instant) -> bool {
        self.healthy_until.is_some_and(|at| now < at)
    }
    fn available(&self, now: Instant, limit: usize) -> bool {
        self.healthy(now)
            && self.cooldown_until.is_none_or(|at| now >= at)
            && self.stats.in_flight < limit
    }
}
#[derive(Clone, Copy)]
struct Affinity {
    endpoint: usize,
    epoch: u64,
    used: Instant,
}
struct State {
    nodes: Vec<Node>,
    groups: Vec<Signature>,
    affinity: HashMap<(usize, u64), Affinity>,
    cursor: usize,
}
struct Inner {
    endpoints: Vec<Endpoint>,
    config: RoutingConfig,
    hash: RandomState,
    state: Mutex<State>,
}

/// Construction performs no I/O. `gateway::run_cluster[_until]` owns periodic
/// capability discovery; the stats handle may be cloned and read by an owner.
#[derive(Clone)]
pub struct Cluster(Arc<Inner>);
impl Cluster {
    pub fn new(upstreams: &[String], config: RoutingConfig) -> io::Result<Self> {
        config.validate()?;
        if !(1..=MAX_ENDPOINTS).contains(&upstreams.len()) {
            return Err(invalid("cluster requires between 1 and 64 endpoints"));
        }
        let mut endpoints: Vec<Endpoint> = Vec::new();
        for url in upstreams {
            let endpoint = Endpoint::parse(url)?.gateway_route("/chat/completions", None)?;
            let path = endpoint.path.split('?').next().unwrap_or("");
            if endpoint.path.contains('?')
                || ![
                    "/chat/completions",
                    "/responses",
                    "/messages",
                    "/messages/count_tokens",
                    "/admission",
                    "/capabilities",
                ]
                .iter()
                .any(|suffix| path.ends_with(suffix))
            {
                return Err(invalid(
                    "cluster requires known API suffixes without query strings",
                ));
            }
            let probe = endpoint.gateway_route("/capabilities", None)?;
            if endpoints.iter().any(|e| {
                e.host == endpoint.host
                    && e.port == endpoint.port
                    && e.gateway_route("/capabilities", None)
                        .is_ok_and(|p| p.path == probe.path)
            }) {
                return Err(invalid("duplicate cluster endpoint"));
            }
            endpoints.push(endpoint);
        }
        let nodes = (0..endpoints.len())
            .map(|endpoint| Node {
                probed: false,
                group: None,
                domain: None,
                epoch: 0,
                health_serial: 0,
                healthy_until: None,
                cooldown_until: None,
                stats: EndpointStats {
                    endpoint,
                    ..EndpointStats::default()
                },
            })
            .collect();
        Ok(Self(Arc::new(Inner {
            endpoints,
            config,
            hash: RandomState::new(),
            state: Mutex::new(State {
                nodes,
                groups: Vec::new(),
                affinity: HashMap::new(),
                cursor: 0,
            }),
        })))
    }
    fn state(&self) -> MutexGuard<'_, State> {
        self.0.state.lock().unwrap_or_else(|e| e.into_inner())
    }
    #[cfg(all(test, any(target_os = "macos", target_os = "linux")))]
    pub(super) fn replace_endpoint_for_test(&mut self, index: usize, endpoint: Endpoint) {
        Arc::get_mut(&mut self.0).unwrap().endpoints[index] = endpoint;
    }
    pub fn stats(&self) -> RoutingStats {
        let now = Instant::now();
        let mut state = self.state();
        state
            .affinity
            .retain(|_, a| now.saturating_duration_since(a.used) < self.0.config.affinity_ttl);
        RoutingStats {
            affinity_entries: state.affinity.len(),
            affinity_capacity: self.0.config.affinity_capacity,
            endpoints: state
                .nodes
                .iter()
                .map(|node| EndpointStats {
                    group: node.group,
                    healthy: node.healthy(now),
                    cooling_down: node.cooldown_until.is_some_and(|at| now < at),
                    ..node.stats.clone()
                })
                .collect(),
        }
    }
    pub(super) fn len(&self) -> usize {
        self.0.endpoints.len()
    }
    pub(super) fn probe_interval(&self) -> Duration {
        self.0.config.probe_interval
    }
    pub(super) fn refresh(&self, index: usize, cancel: &AtomicBool) {
        let serial = self.state().nodes[index].health_serial;
        let result = (|| {
            let endpoint = self.0.endpoints[index].gateway_route("/capabilities", None)?;
            // Probes use the configured upstream credential, without client requests or affinity keys.
            let mut response = transport::request_method_pooled_deadline(
                "GET",
                &endpoint,
                &[],
                None,
                Some(cancel),
                &ConnectionPool::default(),
                Some(Instant::now() + Duration::from_secs(5)),
            )?;
            if response.status != 200 {
                return Err(invalid("capability probe refused"));
            }
            let mut bytes = Vec::new();
            response
                .body
                .by_ref()
                .take((MAX_METADATA + 1) as u64)
                .read_to_end(&mut bytes)?;
            if bytes.len() > MAX_METADATA {
                return Err(invalid("capability probe too large"));
            }
            Signature::parse(
                std::str::from_utf8(&bytes).map_err(|_| invalid("capabilities not UTF-8"))?,
            )
        })();
        self.apply_probe(index, serial, result, Instant::now());
    }
    fn apply_probe(
        &self,
        index: usize,
        serial: u64,
        result: io::Result<(Signature, Option<String>)>,
        now: Instant,
    ) {
        let mut state = self.state();
        let old_group = state.nodes[index].group;
        let result = result.and_then(|(signature, domain)| {
            let group = if let Some(group) = old_group {
                if state.groups[group] != signature {
                    return Err(invalid(
                        "endpoint compatibility changed; restart gateway to reconfigure",
                    ));
                }
                group
            } else if let Some(group) = state.groups.iter().position(|s| *s == signature) {
                group
            } else {
                let group = state.groups.len();
                state.groups.push(signature);
                group
            };
            Ok((group, domain))
        });
        let node = &mut state.nodes[index];
        node.probed = true;
        match result {
            Ok((group, domain)) => {
                if node.domain != domain {
                    node.epoch = node.epoch.wrapping_add(1);
                }
                node.group = Some(group);
                node.domain = domain;
                // A successful probe cannot undo a later dispatch failure.
                if node.health_serial == serial {
                    node.healthy_until = Some(now + self.0.config.capability_ttl);
                }
            }
            Err(_) => {
                node.healthy_until = None;
                node.epoch = node.epoch.wrapping_add(1);
                node.stats.probe_failures = node.stats.probe_failures.saturating_add(1);
            }
        }
    }

    pub(super) fn select(
        &self,
        path: &str,
        query: Option<&str>,
        body: &[u8],
        head: &Head,
    ) -> Result<Lease, RouteError> {
        self.select_excluding(path, query, body, head, None)
    }

    pub(super) fn select_excluding(
        &self,
        path: &str,
        query: Option<&str>,
        body: &[u8],
        head: &Head,
        excluded: Option<usize>,
    ) -> Result<Lease, RouteError> {
        let value = std::str::from_utf8(body)
            .ok()
            .and_then(|s| json::view(s).ok());
        let request = if path == "/v1/admission" {
            value.and_then(|v| v.get("request"))
        } else {
            value
        };
        let model = request
            .and_then(|v| v.get("model"))
            .and_then(json::View::as_str);
        let route = if path == "/v1/admission" {
            value
                .and_then(|v| v.get("endpoint"))
                .and_then(json::View::as_str)
                .map_or("chat", |v| api_route(&v))
        } else {
            api_route(path)
        };
        let key = request.map(|v| self.key(v, route, query, head));
        let now = Instant::now();
        let mut state = self.state();
        if state.nodes.iter().any(|node| !node.probed) {
            return Err(RouteError(503, "initial capability discovery is pending".into()));
        }
        // All discovered groups participate, even unhealthy ones. Never choose
        // a different fingerprint merely because the original group is down.
        // Messages model names can be client aliases. Prefer an exact loaded ID;
        // an alias still requires one unambiguous compatibility group below.
        let messages_alias = route == "messages"
            && model.as_ref().is_some_and(|model| {
                !state
                    .groups
                    .iter()
                    .any(|group| model.as_ref() == group.caps.model)
            });
        let mut groups = state
            .groups
            .iter()
            .enumerate()
            .filter(|(_, group)| {
                model
                    .as_ref()
                    .is_none_or(|model| messages_alias || model.as_ref() == group.caps.model)
            })
            .map(|(i, _)| i);
        let group = groups.next().ok_or(RouteError(
            503,
            "no verified compatible endpoint is available".into(),
        ))?;
        if let Some(second) = groups.next() {
            let groups = std::iter::once(group).chain(std::iter::once(second)).chain(groups);
            let mut fields = Vec::new();
            let mut descriptions = Vec::new();
            for id in groups {
                fields.extend(state.groups[group].differences(&state.groups[id]));
                let endpoints = state.nodes.iter().enumerate()
                    .filter(|(_, node)| node.group == Some(id))
                    .map(|(index, _)| {
                        let endpoint = &self.0.endpoints[index];
                        let path = endpoint.path.split('?').next().unwrap_or("");
                        format!("endpoint {index} (http://{}{path})", endpoint.authority)
                    }).collect::<Vec<_>>().join(", ");
                descriptions.push(format!("group {id}: [{endpoints}]"));
            }
            fields.sort_unstable();
            fields.dedup();
            return Err(RouteError(409, format!(
                "Model replicas are incompatible; fields differ: {}. {}. Use matching model artifacts and serving settings, then restart the gateway; or configure a separate gateway for each group.",
                fields.join(", "), descriptions.join("; ")
            ).into()));
        }
        let signature = &state.groups[group];
        if (path.ends_with("/responses") && !signature.responses)
            || (path.ends_with("/messages") && !signature.messages)
            || (path.ends_with("/messages/count_tokens") && !signature.count_tokens)
            || (path.ends_with("/chat/completions") && !signature.caps.chat_completions)
            || (path != "/v1/admission" && route == "completions" && !signature.completions)
            || (path == "/v1/admission" && !signature.caps.exact_admission)
        {
            return Err(RouteError(
                503,
                "compatible endpoints do not advertise this API route".into(),
            ));
        }
        state
            .affinity
            .retain(|_, a| now.saturating_duration_since(a.used) < self.0.config.affinity_ttl);
        let affinity_key = key.map(|k| (group, k));
        let previous = affinity_key.and_then(|k| state.affinity.get(&k)).copied();
        let available = |index: usize| {
            let node = &state.nodes[index];
            Some(index) != excluded
                && node.group == Some(group)
                && node.available(now, self.0.config.endpoint_concurrency)
        };
        let affine =
            previous.filter(|a| available(a.endpoint) && state.nodes[a.endpoint].epoch == a.epoch);
        let endpoint = affine
            .map(|a| a.endpoint)
            .or_else(|| {
                (0..state.nodes.len())
                    .map(|n| (state.cursor + n) % state.nodes.len())
                    .filter(|&i| available(i))
                    .min_by_key(|&i| state.nodes[i].stats.in_flight)
            })
            .ok_or(RouteError(
                503,
                "all compatible endpoints are unhealthy or saturated".into(),
            ))?;
        state.cursor = (endpoint + 1) % state.nodes.len();
        let node = &mut state.nodes[endpoint];
        node.stats.in_flight += 1;
        if affine.is_some() {
            node.stats.affinity_hits = node.stats.affinity_hits.saturating_add(1);
        }
        if previous.is_some() && affine.is_none() {
            node.stats.fallbacks = node.stats.fallbacks.saturating_add(1);
        }
        let epoch = node.epoch;
        if let Some(key) = affinity_key {
            if !state.affinity.contains_key(&key)
                && state.affinity.len() >= self.0.config.affinity_capacity
            {
                if let Some(oldest) = state
                    .affinity
                    .iter()
                    .min_by_key(|(_, a)| a.used)
                    .map(|(k, _)| *k)
                {
                    state.affinity.remove(&oldest);
                }
            }
            state.affinity.insert(
                key,
                Affinity {
                    endpoint,
                    epoch,
                    used: now,
                },
            );
        }
        Ok(Lease {
            cluster: self.clone(),
            index: endpoint,
            started: false,
            completed: false,
        })
    }

    fn key(&self, request: json::View<'_>, route: &str, query: Option<&str>, head: &Head) -> u64 {
        let mut hash = self.0.hash.build_hasher();
        // Process-local randomized std hash, deliberately not an identity,
        // credential store, authorization check or cryptographic construction.
        for name in [
            "authorization",
            "x-api-key",
            "openai-organization",
            "openai-project",
        ] {
            name.hash(&mut hash);
            // Include all occurrences and the same hop-header policy as relay.
            // The engine still owns authorization; this only separates hints.
            for (_, value) in head.fields.iter().filter(|(key, _)| key == name) {
                if !transport::hop_header(name, head) {
                    value.hash(&mut hash);
                }
            }
        }
        query.hash(&mut hash);
        route.hash(&mut hash);
        if let Some(key) = request.get("prompt_cache_key").and_then(json::View::as_str) {
            "client".hash(&mut hash);
            key.hash(&mut hash);
        } else {
            "prefix".hash(&mut hash);
            for name in ["system", "instructions", "tools"] {
                name.hash(&mut hash);
                if let Some(value) = request.get(name) {
                    prefix(value.raw(), &mut hash);
                }
            }
            // Stable across appended history: system/developer messages and the
            // first ordinary message, each using leading wire bytes unchanged.
            if let Some(messages) = request.get("messages").or_else(|| request.get("input")) {
                if let Some(items) = messages.array() {
                    for item in items.take(17) {
                        prefix(item.raw(), &mut hash);
                        let role = item.get("role").and_then(json::View::as_str);
                        if !role.is_some_and(|r| r == "system" || r == "developer") {
                            break;
                        }
                    }
                } else {
                    prefix(messages.raw(), &mut hash);
                }
            }
            // Text Completions carry their whole context in `prompt`; its
            // leading bytes play the role of the first message above.
            if route == "completions" {
                if let Some(prompt) = request.get("prompt") {
                    "prompt".hash(&mut hash);
                    prefix(prompt.raw(), &mut hash);
                }
            }
        }
        hash.finish()
    }
}
/// Affinity namespace of a known API path; text Completions are distinct from chat.
fn api_route(path: &str) -> &'static str {
    if path.ends_with("/responses") {
        "responses"
    } else if path.ends_with("/messages") || path.ends_with("/messages/count_tokens") {
        "messages"
    } else if path.ends_with("/completions") && !path.ends_with("/chat/completions") {
        "completions"
    } else {
        "chat"
    }
}
fn prefix(text: &str, hash: &mut impl Hasher) {
    text.as_bytes()[..text.len().min(PREFIX_BYTES)].hash(hash);
}

pub(super) struct RouteError(pub u16, pub Cow<'static, str>);
pub(super) struct Lease {
    cluster: Cluster,
    index: usize,
    started: bool,
    completed: bool,
}
impl Lease {
    pub(super) fn index(&self) -> usize {
        self.index
    }
    pub(super) fn endpoint(&self) -> &Endpoint {
        &self.cluster.0.endpoints[self.index]
    }
    pub(super) fn start(&mut self) {
        self.started = true;
        let mut state = self.cluster.state();
        let stats = &mut state.nodes[self.index].stats;
        stats.requests = stats.requests.saturating_add(1);
    }
    pub(super) fn status(&mut self, status: u16, retry_after: Option<&str>) {
        if status == 429 || status >= 500 {
            let cooldown = retry_after
                .and_then(|s| s.parse::<u64>().ok())
                .map(|s| Duration::from_secs(s.clamp(1, 300)))
                .unwrap_or(self.cluster.0.config.failure_cooldown);
            let mut state = self.cluster.state();
            let node = &mut state.nodes[self.index];
            node.health_serial = node.health_serial.wrapping_add(1);
            let until = Instant::now() + cooldown;
            node.cooldown_until = Some(node.cooldown_until.map_or(until, |old| old.max(until)));
            if matches!(status, 429 | 503 | 529) {
                node.stats.overloads = node.stats.overloads.saturating_add(1);
            }
        }
    }
    pub(super) fn finish(&mut self, status: u16, usage: Option<(u64, u64)>) {
        self.completed = true;
        let mut state = self.cluster.state();
        let stats = &mut state.nodes[self.index].stats;
        if status >= 400 {
            stats.failures = stats.failures.saturating_add(1);
        }
        if (200..300).contains(&status) {
            if let Some((prompt, cached)) = usage {
                // On overflow leave all cumulative token metrics unchanged.
                if let Some((p, c)) = stats
                    .prompt_tokens
                    .checked_add(prompt)
                    .zip(stats.cached_tokens.checked_add(cached))
                {
                    stats.prompt_tokens = p;
                    stats.cached_tokens = c;
                    stats.usage_responses = stats.usage_responses.saturating_add(1);
                }
            }
        }
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        let mut state = self.cluster.state();
        let node = &mut state.nodes[self.index];
        node.stats.in_flight -= 1;
        if self.started && !self.completed {
            node.stats.failures = node.stats.failures.saturating_add(1);
            node.health_serial = node.health_serial.wrapping_add(1);
            node.healthy_until = None;
            node.cooldown_until = Some(Instant::now() + self.cluster.0.config.failure_cooldown);
        }
    }
}

#[cfg(test)]
#[path = "routing_tests.rs"]
mod tests;
