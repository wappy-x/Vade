use super::{App, args::Mode};
use std::io;
use vade_harness::{
    agent::{CONTINUATION_PROMPT, Limits, MAX_AUTO_CONTINUES, RunBudget, TOOL_CONTINUATION_PROMPT},
    contract::PromptBoundary,
};

pub(crate) struct Answers {
    seed: Option<u64>,
    pub(crate) user: Option<String>,
    pub(crate) direct_turns: usize,
    pub(crate) retrying: bool,
    pub(crate) available: bool,
    pub(crate) cut_off: bool,
    pub(crate) auto_continues: usize,
    pub(crate) continuing: bool,
    pub(crate) budget: Option<RunBudget>,
}
impl Answers {
    pub(crate) fn new(options: vade_harness::contract::GenerationOptions) -> Self {
        Self {
            seed: options.seed,
            user: None,
            direct_turns: 0,
            retrying: false,
            available: false,
            cut_off: false,
            auto_continues: 0,
            continuing: false,
            budget: None,
        }
    }
    pub(crate) fn request_seed(&self) -> Option<u64> {
        self.seed
    }
    pub(crate) fn begin(&mut self, user: &str, direct_turns: usize) {
        if !std::mem::take(&mut self.continuing) {
            self.auto_continues = 0;
            self.budget = None;
        }
        self.user = Some(user.into());
        self.direct_turns = direct_turns;
        self.available = false;
        self.cut_off = false;
    }
}
impl App {
    pub(crate) fn answer_generation(&self) -> vade_harness::contract::GenerationOptions {
        let mut generation = self.config.generation;
        // A bookkeeping seed must not turn an unspecified sampler into an
        // explicit request and bypass the Engine's no-field greedy policy.
        generation.seed = self.answers.request_seed();
        generation
    }
    pub(crate) fn reset_answers(&mut self) {
        self.answers = Answers::new(self.config.generation);
    }
    pub(crate) fn resumed_answers(&mut self) {
        self.reset_answers();
        if self.config.generation.seed.is_none() {
            if let Some(seed) = self
                .journal
                .as_ref()
                .and_then(|j| j.lock().ok()?.generation_seed())
            {
                // Existing journals retain their recorded sampling choice.
                // New default sessions do not persist an implicit seed.
                self.answers.seed = Some(seed);
            }
        }
    }
    pub(crate) fn retry_answer(&mut self) -> io::Result<()> {
        if self.pending.is_some() {
            return self.notice("Wait for cleanup before /retry.");
        }
        if !matches!(self.config.mode, Mode::Direct | Mode::Agent) || !self.answers.available {
            return self.notice("No completed answer to retry in this conversation.");
        }
        let Some(user) = self.answers.user.clone() else {
            return Ok(());
        };
        if self.config.mode == Mode::Agent {
            let Some(c) = &self.native else {
                return self.notice("History unavailable; use /new.");
            };
            // Preserve all completed tools and their receipts. Only the final
            // answer is replaced, and raw journal records remain on disk.
            let replacement = self
                .journal
                .as_ref()
                .unwrap()
                .lock()
                .map_err(|_| io::Error::other("journal lock failed"))?
                .retry_answer_with_optional_seed(
                    c,
                    self.answers.request_seed(),
                    Limits::default(),
                )?;
            self.replace_native(replacement, PromptBoundary::Compaction)?;
            self.answers.retrying = true;
        } else {
            if self.session.pruned_turns > 0
                || self.session.turns() != self.answers.direct_turns + 1
            {
                return self.notice("Last turn is no longer available to retry; use /new.");
            }
            self.session.undo();
            self.direct_prefix_blocked = false;
            if let Some(client) = self.http.as_mut() {
                client.reset_context();
            }
        }
        let sampling = if let Some(seed) = self.answers.request_seed() {
            format!("unchanged seed {seed}; use a different --seed for different sampling")
        } else if self.config.generation.temperature.is_some() {
            "unchanged sampling settings (no seed supplied)".into()
        } else {
            "unchanged upstream defaults (a greedy answer may repeat)".into()
        };
        self.notice(&format!("Retrying the last answer with {sampling}. Completed tools are retained; this run cannot execute tools."))?;
        self.start(user)
    }
    pub(crate) fn continue_answer(&mut self) -> io::Result<()> {
        if self.pending.is_some() {
            return self.notice("Wait for cleanup before /continue.");
        }
        if !self.answers.cut_off {
            return self.notice("The last reply was not cut off; no continuation sent.");
        }
        // Native agent sessions have tools; direct chat continues prose.
        let prompt = if self.config.mode == Mode::Agent {
            TOOL_CONTINUATION_PROMPT
        } else {
            CONTINUATION_PROMPT
        };
        self.start(prompt.into())
    }
    pub(crate) fn auto_continue_direct(&mut self) -> io::Result<()> {
        if !self.config.auto_continue || !self.answers.cut_off {
            return Ok(());
        }
        let reason = if self.direct_prefix_blocked
            || self.session.turns() >= vade_harness::session::MAX_HISTORY_TURNS
        {
            Some("history limit reached; use /new before continuing")
        } else if self.answers.auto_continues >= MAX_AUTO_CONTINUES {
            Some("32 automatic continuations reached; use /continue for a fresh budget")
        } else if self.answers.budget.as_ref().is_some_and(RunBudget::expired) {
            Some("active-work time limit reached; use /continue for a fresh budget")
        } else {
            None
        };
        if let Some(reason) = reason {
            self.failed = true;
            self.last_success = false;
            self.queue_paused = true;
            return self.run_notice(&format!("Automatic continuation stopped: {reason}."));
        }
        self.answers.auto_continues += 1;
        self.answers.continuing = true;
        self.run_notice(&format!(
            "Continuing automatically ({}/{MAX_AUTO_CONTINUES}).",
            self.answers.auto_continues
        ))?;
        let result = self.start(CONTINUATION_PROMPT.into());
        if result.is_err() || self.pending.is_none() {
            self.answers.continuing = false;
        }
        result
    }
}
