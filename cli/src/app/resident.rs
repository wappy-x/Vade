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
        if existing.is_some_and(|s|s.diagnostic_unqualified_text) && !options.diagnostic_unqualified_text {
            return Err("This engine is unqualified diagnostics. Repeat --diagnostic-unqualified-text explicitly, or run vade down before a normal start.".into());
        }
        // Bare up keeps the recorded launch options, but always rebuilds the
        // request from today's configuration and activated build manifest.
        let mut options = match existing.filter(|session| options.supplied_options.is_empty() && !session.diagnostic_unqualified_text) {
            Some(session) => {
                let mut recorded = recorded_options(session);
                if options.mtp_explicit { recorded.mtp = options.mtp; recorded.mtp_explicit = true; }
                recorded
            }
            None => {
                let mut options = options.clone();
                // A chain or a pair spreads the weights, so it has room for a long
                // context. One Spark, as each plain replica is, keeps the short one.
                if options.fp4_local && !options.supplied_options.contains("--context") {
                    let sparks = if options.replicas { options.replica_sparks.max(1) } else { self.config.selected(&options)?.len() };
                    // A running engine keeps its size, which may be the fallback.
                    if let Some(session) = existing { options.context = session.context; }
                    else if sparks > 1 { options.context = crate::cli::FP4_LONG_CONTEXT; }
                }
                options
            }
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
        // Two FP4-Local Sparks run the tensor split, which needs RDMA. Refuse TCP here, before a
        // start or an ownership write. A diagnostic start may already have asked the checkpoint plan.
        // FP8 pairs keep their ring and may use TCP.
        if session.two_sparks() && !rdma {
            return Err("Two FP4-Local Sparks run the tensor split, which needs RDMA. Unset SEQPAR_RDMA, then run vade up.".into());
        }
        // decode_fuse serves one FP4-Local Spark; refuse it here, before any remote call.
        if session.decode_fuse && (session.group_len() > 1 || !session.fp4_local) {
            return Err("--decode-fuse on runs on one FP4-Local Spark. Leave it off for more Sparks or FP8.".into());
        }
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
            for (peer, address) in &rank.machine.fabric_aux {
                fields.insert(format!("rank.{index}.fabric_aux.{peer}"), address.clone());
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
        session.launch_fingerprint = Some(crate::sha256::sha(text::encode(&fields).as_bytes()));
        Ok(session)
    }

    /// Ask one Spark which text service its checkpoint declares. An older helper
    /// has no such operation and is read as the two-Spark wire, so a current-family
    /// start keeps working until `vade install`.
    pub(super) fn remote_text_service(&mut self, machine: &Machine, worker: &str) -> Result<Option<crate::text_service::TextService>, String> {
        let mut exec = runner::ssh(
            &machine.ssh,
            format!("exec {} __worker", remote_path(worker)),
            &format!("{} text service", machine.name),
        );
        exec.stdin = text::encode(&Fields::from([
            ("op".into(), "text-service".into()),
            ("model_path".into(), machine.model_path.clone()),
        ])).into_bytes();
        exec.timeout = std::time::Duration::from_secs(90);
        let output = self.runner.run(&exec).map_err(|error| format!("Transport: {}: {error}", machine.name))?;
        // The real dry runner prints the command and returns success with empty
        // output. That is not a report and not an old helper.
        if self.dry && output.success && output.stdout.trim().is_empty() {
            return Ok(None);
        }
        if !output.success && output.stdout.trim().is_empty() {
            return Err(format!("Transport: {} did not reply to the text-service check. Nothing started.", machine.name));
        }
        let fields = text::parse(&output.stdout).map_err(|_| "The text-service reply was not a record. Nothing started.".to_string())?;
        if let Some(error) = fields.get("error") {
            if error.contains("Unknown machine operation. Reinstall vade.") {
                return Ok(Some(crate::text_service::TextService::pair()));
            }
            return Err(format!("{error} Nothing started."));
        }
        if !output.success {
            return Err("The text-service check failed. Nothing started.".into());
        }
        if !fields.contains_key("text_sparks") {
            return Ok(Some(crate::text_service::TextService::pair()));
        }
        crate::text_service::TextService::from_report(&fields).map(Some)
    }
}

pub(super) fn recorded_options(session: &Session) -> UpOptions {
    UpOptions {
        // The context of a running engine stays, also after a restart.
        supplied_options: ["--context".to_owned()].into(),
        sparks: None,
        listed: false,
        links: Vec::new(),
        rest: Vec::new(),
        hosts: session
            .ranks
            .iter()
            .map(|rank| rank.machine.name.clone())
            .collect(),
        replicas: session.replicas,
        replica_sparks: session.replica_sparks,
        mtp: session.mtp,
        // A depth 0 that came from a fall (no drafting layer, or too little memory) is nobody's choice: a restart
        // tries the default again.
        mtp_explicit: session.mtp_explicit && session.draft_fall.is_none(),
        fp4_local: session.fp4_local,
        context: session.context,
        output: session.output,
        output_explicit: true,
        chats: session.chats,
        fast_prompts: session.fast_prompts,
        memory_bytes: session.memory_bytes,
        cache_bytes: session.cache_bytes,
        decode_fuse: Some(session.decode_fuse),
        diagnostic_unqualified_text: session.diagnostic_unqualified_text,
        checkpoint_service: None,
        text_width_unread: false,
        draft_fall: session.draft_fall,
    }
}

pub(super) fn same_launch(a: &Session, b: &Session) -> bool {
    // Older records remain readable for down/recovery, but cannot prove that
    // the requested model paths, build and transport match the live process.
    a.launch_fingerprint
        .as_ref()
        .is_some_and(|value| b.launch_fingerprint.as_ref() == Some(value))
}
