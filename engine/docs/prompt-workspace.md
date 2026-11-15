# Borrowed prompt workspace

A permanent larger prompt workspace reduces the session pool. This can delay
admission and replace shared decode steps with scalar steps. Keep startup
workspace and admission capacity fixed.

A lone prompt can instead borrow unreserved session pool bytes for one
scheduler prefill call. Fund all live and retired reservations, retained cache
owners and request holds first. Both ranks offer a group from that remainder
and agree on the smaller offer before running it. Return the scratch after
committing the new state and dropping the old state, before the scheduler
accepts another request. Drain the stream and trim actual spare allocations
on success and failure. Loading no predictor preserves the existing path.

This design requires host cleanup and admission regressions, identical device
token IDs, and balanced device measurements before it can be enabled.

## Host validation

The touched serving and tensor-split crates pass 890 tests with zero failures
and seven ignored device tests. The lease regressions cover actual host
scratch deallocation, rank offers, exact remainder thresholds, arrivals,
eight sessions, cancellation, injected errors and an 8192-token cached resume.
The real-socket scheduler arrival regression passes 20 consecutive runs.
Disabling expansion makes the threshold regression fail at 256 versus 512
rows. Host results do not establish device token identity or speed.

The checkpoint, context and startup group fix the two larger-group byte deltas.
Compute them once after the memo bound is locked. Runtime offers read that
plan; they must not rebuild layer recipes on a request's clock.
