//! Reuse of a running engine: rebuild the request from today's config and
//! build manifest, then compare it with the saved session so an unchanged
//! `vade up` or `vade chat` attaches instead of restarting.
use super::*;

impl<R: Runner> App<R> {
    pub(super) fn requested_session(
        &self,
        options: &UpOptions,
        existing: Option<&Session>,
        rdma: bool,
    ) -> Result<Session, String> {
        self.requested_session_with(options, existing, rdma, |name| std::env::var(name))
    }
    /// `read`: this shell's 2-Spark switches (plan::two_spark_switches), part of the launch.
    pub(super) fn requested_session_with(
        &self,
        options: &UpOptions,
        existing: Option<&Session>,
        rdma: bool,
        read: impl Fn(&str) -> Result<String, std::env::VarError>,
    ) -> Result<Session, String> {
        // Bare up keeps the recorded launch options, but always rebuilds the
        // request from today's configuration and activated build manifest.
        let mut options = match existing.filter(|_| options.supplied_options.is_empty()) {
            Some(session) => {
                let mut recorded = recorded_options(session);
                if options.mtp_explicit { recorded.mtp = options.mtp; recorded.mtp_explicit = true; }
                recorded
            }
            None => options.clone(),
        };
        let manifest = self.manifest()?;
        if !manifest.supports_output && !options.output_explicit {
            options.output = options.context.min(8192);
        }
        if !manifest.supports_output && options.output != options.context.min(8192) {
            return Err(format!(
                "This engine build has a fixed output limit. Use --output {}, or install source that supports FLASH_SERVE_OUTPUT_TOKENS.",
                options.context.min(8192)
            ));
        }
        let token = match existing {
            Some(session) => session.token.clone(),
            None if self.dry => "0".repeat(64),
            None => text::random_hex(32)?,
        };
        let mut session = Session::new(&self.config, &options, &manifest, token)?;
        // Capture before resolving ~/ paths and fabric names: a saved session
        // contains expanded launch paths, while the next request starts with
        // the configured strings. Hash the complete request, excluding only
        // ownership and progress fields, so unchanged launches stay local.
        let mut fields = text::parse(&session.encode())?;
        for key in ["token", "phase", "admitted_chats"] {
            fields.remove(key);
        }
        for (index, rank) in session.ranks.iter().enumerate() {
            fields.remove(&format!("rank.{index}.attempted"));
            for (peer, address) in &rank.machine.fabric {
                fields.insert(format!("rank.{index}.fabric.{peer}"), address.clone());
            }
        }
        for (key, value) in [
            (
                "local_state_dir",
                self.config.state_dir.to_string_lossy().into_owned(),
            ),
            (
                "local_stage_dir",
                self.config.stage_dir.to_string_lossy().into_owned(),
            ),
            (
                "build_manifest",
                self.config.build_manifest.to_string_lossy().into_owned(),
            ),
            ("model_repo", self.config.model_repo.clone()),
            ("model_revision", self.config.model_revision.clone()),
            (
                "supports_fp4_chain",
                manifest.supports_fp4_chain.to_string(),
            ),
            (
                "harness_sha256",
                manifest.harness_sha256.unwrap_or_default(),
            ),
            ("rdma", (session.chained() && rdma).to_string()),
        ] {
            fields.insert(key.into(), value);
        }
        // Two Sparks: the path and level switches the ranks start with, so a changed one is a
        // different launch, never the running server.
        if session.two_sparks() {
            for (name, value) in plan::two_spark_switches(read, session.chained() && rdma)? {
                fields.insert(format!("env.{name}"), value.unwrap_or_else(|| "<unset>".into()));
            }
        }
        session.launch_fingerprint = Some(crate::sha256::sha(text::encode(&fields).as_bytes()));
        Ok(session)
    }
}

fn recorded_options(session: &Session) -> UpOptions {
    UpOptions {
        supplied_options: Default::default(),
        sparks: None,
        hosts: session
            .ranks
            .iter()
            .map(|rank| rank.machine.name.clone())
            .collect(),
        replicas: session.replicas,
        mtp: session.mtp,
        mtp_explicit: session.mtp_explicit,
        fp4_local: session.fp4_local,
        context: session.context,
        output: session.output,
        output_explicit: true,
        chats: session.chats,
        fast_prompts: session.fast_prompts,
        memory_bytes: session.memory_bytes,
        cache_bytes: session.cache_bytes,
    }
}

pub(super) fn same_launch(a: &Session, b: &Session) -> bool {
    // Older records remain readable for down/recovery, but cannot prove that
    // the requested model paths, build and transport match the live process.
    a.launch_fingerprint
        .as_ref()
        .is_some_and(|value| b.launch_fingerprint.as_ref() == Some(value))
}
