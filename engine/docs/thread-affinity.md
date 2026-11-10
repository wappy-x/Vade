# Topology-derived host thread placement

The bounded text service and resident pair check separate the host mHC
workers from decode, the wire and CUDA workers. The wire thread also runs
the mailbox doorbell. It is not a separate thread.

Startup reads the allowed CPU mask, maximum online CPU frequencies and L3
shared_cpu_list metadata. It places the conductor's worker count in one
fast cache domain. Decode can move among all fast cores in another L3
domain. The wire and CUDA workers can move within the initial allowed mask,
excluding host-worker cores and cores reserved by co-located ranks.
No fixed CPU number, box name,
checkpoint name or context-length threshold selects that map.

Far-domain discovery retains observed NIC interrupt counts and CPU order.
The complete selected fast-domain mask is used, not one preferred core.
Background masks can include slower cores and the unused part of the host
cache domain. The scheduler can use these instead of competing with decode
for the far domain. This is not an IRQ or scheduler policy change.

Ranks exchange their local kernel identity and reservations. Co-located
ranks cannot reuse cores, including background masks. Initial allowed masks
are exchanged with the role domains. Required roles are reserved first;
then shared spare CPUs go to the first eligible rank. If disjoint domains
do not fit, the whole plan
falls back with a reason. This includes small allowed masks and two local
ranks when their combined reservations exceed the fast-core budget.
Existing mHC placement remains in use on fallback. External allowed masks
are never widened beyond the rank's initial set.

The conductor inherits only its host-worker mask. The wire pins itself
after it starts. The parent restores its initial mask before placing
decode. The text service places decode only after the weights are
resident, so load readers spawned by the parent keep the initial mask.
The resident check places decode before its jobs. Managed startup ignores
`TP_CPU` and logs the reason, so children cannot inherit its singleton
mask. Other transport callers retain their existing explicit `TP_CPU`
behavior.

CUDA workers are enumerated under `/proc/self/task` while the owned CUDA
context is live. The existing affinity boundary checks each target's Tgid
against the current process before it changes the thread's mask. Each
decode placement first gives every CUDA worker the rank's background mask.
The resident check refreshes again after prefill, outside decode timing. Failed
affinity operations are logged. A sleeping CUDA worker can retain an old
last-CPU sample outside its new mask until it wakes. The kernel's allowed
mask, not that historical CPU value, identifies its current placement.

`TP AFFINITY` records contain the actual selected roles and CUDA thread
IDs. The code does not change CPU clocks, IRQ routing, idle settings or
host policies. It does not change model arithmetic, rounding, worker count,
message order or doorbell linger. Host tests cover domain separation, IRQ
preference, CPU renumbering, co-located fallback and owned-thread affinity.
A device test checks live CUDA-worker placement. Loopback exactness still
does not establish physical-pair speed; that needs the device screen.
