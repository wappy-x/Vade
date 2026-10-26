# C8 cohort draft selection

Local world-one Flash-Next executables enable the C8 cohort controller and
healthy graph-capture fallback by default. Their unset settings match:

```sh
VADE_MTP_C8=1 VADE_POOL_GRAPH_FALLBACK=1 VADE_MTP_COHORT_ADAPT=1 flash-serve ...
```

To restore the previous automatic policy and disable both experiment controls:

```sh
VADE_MTP_C8=0 VADE_POOL_GRAPH_FALLBACK=0 VADE_MTP_COHORT_ADAPT=0 flash-serve ...
```

An explicit `VADE_MTP_BATCH` retains its previous table and disables the implicit
cohort default. `VADE_MTP_C8=0` also disables that implicit cohort default.
The two- and four-stage layer pipelines enable guarded Auto8 through
`VADE_W2_PP_C8` (unset or `1`); there the wide tick below is the default and the
cohort controller is opt-in (`VADE_MTP_COHORT=off` brings back the controller and
the 16-row tick while keeping the guard). Set `VADE_W2_PP_C8=0` on every Spark to
disable the guard itself and restore the previous defaults on either route: Auto
batching, no cohort controller, no wide tick and no graph fallback. Explicit existing switch
values retain their overrides. An explicit cohort `1` requires Auto8 and either
world one or an enabled two- or four-stage pipeline. Native ranks and other
pipeline sizes retain their previous defaults and reject explicit cohort activation.
Explicit request draft/depth controls retain their meaning. The C8 and graph
switches can be controlled independently; world one uses all three zeros for a full rollback.

`VADE_MTP_COHORT=wide` (opt-in on one Spark; the default on a two- or four-stage
pipeline with its guard on, where `off` restores the controller's 16-row tick) widens the cohort tick: with `VADE_VERIFY_ROWS` unset, five to eight chats verify up
to 32 rows in one pass, so eight chats draft depth 3 (four rows each) instead of the
controller's one proposal. Four chats or fewer stay on the 16-row tile. The cost
controller becomes opt-in under `wide` (`VADE_MTP_COHORT_ADAPT=1` turns it back on);
`VADE_VERIFY_ROWS=16` keeps the 16-row tick. Rows above 16 use each row kernel's
M 17-32 path, whose bytes per chat equal the chat's bytes alone. Any other value of
`VADE_MTP_COHORT` is a startup error, and so is `wide` on native ranks or on a
pipeline with `VADE_W2_PP_C8=0` (one Spark accepts it with that switch). On a pipeline, give every Spark the same
value (or leave it unset on all): the resolved tick joins the pipeline identity.

```sh
VADE_MTP_COHORT=wide flash-serve ...
```

For a complete eligible greedy cohort of six to eight decoders, the controller
compares plain execution with one speculative proposal per chat. It measures the
whole cycle, including preparation, target verification, rollback and publication,
against the useful output count. Each path gets two initial observations; its
first observation is excluded from the steady cost estimate. Normal path changes
have an eight-tick minimum dwell, including initial/probe intervals. A path needs a two
percent cost advantage to change the preference. `VADE_C8_GUARD_ROBUST` is parsed once
at startup (unset or `1` on, `0` the two-percent path above, anything else a startup
error). On, draft stays selected unless plain is at least eight percent cheaper per
token, and a draft-preferred cohort does not probe plain until 256 samples. `0` keeps
the two-percent latch and the 64-sample probe, including the timer that starts before
`prepare_cohort_path`. The parsed switch joins the pipeline identity. It changes only which
path runs; verified target tokens stay the greedy ids either way. Only a changed membership resets measured costs. Fairness-only ticks retain the selected path without learning or reclaiming
pooled storage. Prompt/socket/terminal ineligibility suspends learning and holds
plain execution for at least eight ticks after the latest unsafe tick. Safety/eligibility can force plain immediately;
drafting cannot resume until that plain interval expires. Ineligible cohorts and short terminal
tails use ordinary plain execution.

If any initial canonical target ID lies outside the actual prefix-limited draft
head, the whole membership remains plain until it changes. Existing speculators
expose their retained canonical next token after the separate pending field is
consumed, so a membership change checks every live decoder. This conservative
hint uses the loaded head shape and target IDs. It can forgo later draft gains;
it does not change target verification. Other unproductive probes back off to
2,048 cycles if no proposal has survived, or 64 cycles after a survivor.

Switching to drafts synchronizes and releases unused pooled decode execution
storage on every rank. The front marks the next pipeline request for reclamation;
each worker completes prior readers, releases its pool and relays the mark
downstream before executing that request. The mark also propagates when the front's pool is already empty.
Reusable rollback arenas stay allocated because recurrent-state views
can retain their underlying owner. The memory ceiling and admission margin are
unchanged. The graph fallback switch permits a healthy capture refusal to run
the same queued kernels eagerly; other errors keep their existing error path.
A refused capture backs off for 64 pooled steps, doubling subsequent refusals
up to 4,096 steps for each workspace/arm. Replacing a workspace resets that
capture epoch. Invalid switch values fail before local or pipeline/rank startup;
the cohort controller rejects unsupported startup routes.

`cohort_cost` diagnostics include the cohort size, selected path, whole-cycle
cost, committed/proposed/accepted counts and the head-coverage hint. Validate
exact IDs, output bytes and terminal usage across alone/cohort execution, mixed
lengths/depths, staggered arrivals and both draft settings when changing this policy in a deployment.

The pipeline handshake compares the already-parsed route switch and effective
batch, cohort, graph-fallback and robust-guard settings on every link. Unset and
explicit defaults agree; a different parsed setting refuses startup with local
and peer values before weights load. Reclamation rides the existing ordered
request stream without an additional round trip.
Rank zero owns proposal generation, sampling, acceptance and the cost timer.
Every stage verifies target rows, and the timer includes the synchronous remote
execution and link exchange. Each worker applies a rejected prefix on the next
request for that session, so its rollback cost enters the following cycle.
Pipeline target verification keeps its existing eager execution path; healthy
local capture fallback remains off by default on the pipeline.
Explicit `0` may allocate a startup environment string. The disabled decode path adds no clocks, target work or heap allocations.

The executable parses the experiment controls and batch/C8 inputs once, after
checkpoint route selection and before GPU loading or peer connections,
and passes an immutable `Experiments` value through service and engine
construction, including the resolved batch mode. Embedded callers opt in through `FlashNext::with_experiments`;
ordinary constructors use disabled defaults. Pipeline identity code does not
reparse local experiment settings. During a retained draft path, filtered
single-row plans use session execution rather than creating pooled decode
storage; fairness skips cannot grow that pool.
