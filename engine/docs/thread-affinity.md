# Topology-derived host thread placement

The bounded text service and resident pair check separate the host mHC
workers from decode, the wire and CUDA workers. The wire thread also runs
the mailbox doorbell. It is not a separate thread.

Startup reads the allowed CPU mask, the online CPU list, maximum CPU
frequencies and L3 shared_cpu_list metadata. It places the conductor's
worker count in one fast cache domain (the near domain). Decode can move
among all fast cores of a disjoint L3 domain (the far domain, the larger L3
when there is a choice). The wire and CUDA workers can move among all
allowed CPUs of that far domain, so the doorbell shares decode's L3 and never
runs in the near domain, including its spare fast cores and its slow cores.
No fixed CPU number, box name, checkpoint name or context-length threshold
selects that map. Offline CPUs are left out of every mask.

Why the far domain: on the GB10 reference boxes, served loads whose wire
thread ran on a spare near fast core (CPU 8 or 9, beside the mHC workers)
were 0.5 to 1.0 ms a step slower than loads whose wire ran in the far domain
(pair three, train-9m-served C2 against C1 and C3). Under load the scheduler
keeps the busy wire on idle fast cores; the CUDA workers use little CPU and
can sit on the far domain's slower cores. NIC interrupt counts are not read:
they do not change any mask.

Ranks exchange their local kernel identity and reservations. Co-located
ranks cannot reuse cores, including background masks. Required roles are
reserved first; then spare far-domain CPUs go to the first eligible rank.
Decode needs at least one far fast core; at least one CUDA worker must be
visible at startup. If disjoint domains do not fit, the whole plan falls
back with a reason. This includes small allowed masks, a CPU with no maximum
frequency, and two local ranks when their combined reservations exceed the
fast-core budget. Existing mHC placement remains in use on fallback.
External allowed masks are never widened beyond the rank's initial set.

The conductor inherits only its host-worker mask. The wire places itself
after it starts; if the kernel refuses its mask, it falls back to the rank's
initial mask and logs the mask it keeps, so it never stays on the mHC cores.
The parent restores its initial mask before placing decode. The text service
places decode only after the weights are resident, so load readers spawned by
the parent keep the initial mask. The resident check places decode before its
jobs. Managed startup ignores `TP_CPU` and logs the reason, so children
cannot inherit its singleton mask. Other transport callers retain their
existing explicit `TP_CPU` behavior.

Threads the text service starts after decode is placed inherit the decode
mask: the scheduler's request-preparation thread and the prompt-preparation
worker. Both call prefer_fast_cores themselves, so any mask that includes the
far fast cores leads them back there. In served loads they showed about 21
and 1 voluntary context switches over a whole benchmark, and decode's wait for
a CPU stayed at the level of the plain decode-mask run; they are left as they
are.

CUDA workers are enumerated under `/proc/self/task` while the owned CUDA
context is live. The existing affinity boundary checks each target's Tgid
against the current process before it changes the thread's mask. Each
decode placement moves every CUDA worker whose mask differs from the far
domain mask, including workers started after an earlier placement. The
resident check refreshes again after prefill, outside decode timing. The text
service does not refresh after it starts serving; no served load has shown a
CUDA worker created after that point. Failed affinity operations are logged.
A sleeping CUDA worker can retain an old last-CPU sample outside its new mask
until it wakes. The kernel's allowed mask, not that historical CPU value,
identifies its current placement.

`TP AFFINITY` records contain the actual selected roles and CUDA thread
IDs. The code does not change CPU clocks, IRQ routing, idle settings or
host policies. It does not change model arithmetic, rounding, worker count,
message order or doorbell linger. Host tests cover domain separation on
fixture sysfs trees (reference layout, a missing CPU frequency, an offline
CPU, a container cpuset), CPU renumbering, co-located fallback, a refused
affinity call, the wire fallback, a late CUDA worker, `TP_CPU` set in a child
process and owned-thread affinity. A device test checks live CUDA-worker
placement. Loopback exactness still does not establish physical-pair speed;
that needs the device screen.
