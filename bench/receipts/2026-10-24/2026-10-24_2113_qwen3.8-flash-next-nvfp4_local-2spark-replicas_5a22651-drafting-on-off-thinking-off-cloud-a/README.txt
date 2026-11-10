Vade v0.1.0 (5a226519a2a44a0ff7ea9be43118b8b9ec0d8f9e): vade up --replicas 2 on two rented GB10 VMs
2026-10-24 21:00-21:13 CT. E1-J3 release test lane. Informative measurement, no gate. Not a full RigMark card.

WHAT RAN
- Boxes: vm-a + vm-b = two rented GB10 VMs (containers, one NVIDIA GB10 each, 116 GiB RAM, driver 580.173.02).
- Code: public GitHub wappy-x/vade main = v0.1.0 5a22651 only. Model: nvidia/Qwen3.8-Flash-Next-FP4-Local
  @ b9ce18a6887e18c089fdea8509a33e63429a3300, sha256sum -c of the committed manifest 33/33 OK on both VMs.
- INSTALL.md "Route B: from a controller" + "Replicas": controller = vm-a itself (Linux aarch64), fresh clone,
  cli built with cargo, fp4-local.conf with the documented edits (two machine sections, no fabric entries,
  local_port 18181 because the controller box also serves on 18080), SEQPAR_RDMA=0, then
  vade install, vade up --replicas 2 (defaults: context 32768, output 8192, drafting engine default).
- vade status: "independent one-Spark replicas behind the local gateway", "Concurrent chats: 8 per replica,
  16 across 2 replicas (engine-admitted at this context)", Health ready. vade doctor rc 0 (Harness doctor: passed).

INSTALL TIMING (wall, controller, model check excluded)
- Rust step 1 (rustup + 1.99.0 + nightly-2026-10-01) on both VMs in parallel: vm-a 14.4 s, vm-b 14.7 s.
- Controller: clone 1.9 s, cli build 7.3 s, vade install 291.5 s, vade up --replicas 2 68.1 s.
  Clone to "ready on 2 machines as independent replicas behind the gateway": 368.8 s.

EXACTNESS
- vade verify through the gateway: IDENTICAL: 20/20 runs, 1 machine, 128 tokens.
- vade verify --url on each replica tunnel directly (run in parallel): replica A IDENTICAL: 20/20 runs, 1 machine,
  128 tokens; replica B IDENTICAL: 20/20 runs, 1 machine, 128 tokens (vade-verify-replica-a/b.json).
- C8 rows vs the single-server C8 rows of the E1-J1 one-Spark cards on the same two VMs (same request bodies):
  drafting OFF 24/24 identical vs vm-a card and 24/24 vs vm-b card (2026-10-24_1946_*-cloud-a-vma / -vmb);
  drafting ON 24/24 identical vs vm-a card and 24/24 vs vm-b card (2026-10-24_1954_*-cloud-a-vma / -vmb).
- Within the replicas run: OFF vs ON C8 24/24 identical, C16 48/48 identical.
- Every response: build 5a22651, world 1 / topology local, cache off, thinking off. Both replicas served:
  C8 12/12 per replica in both modes; C16 OFF 24/24, C16 ON 25/23 (split by server_start_unix_ms).

THROUGHPUT (aggregate end-to-end tok/s through the gateway, median of 3 rounds; rounds in brackets)
  drafting OFF ("draft":"plain"):  C8 141.901 [113.548 145.804 141.901]   C16 216.539 [190.869 216.539 239.207]
  drafting ON (as served):         C8 239.396 [232.195 239.396 247.995]   C16 213.058 [213.058 180.409 252.945]
  Per-stream decode median: OFF C8 18.899, C16 14.837; ON C8 34.195, C16 17.538 tok/s.
  TTFT median: OFF C8 0.851 s, C16 1.679 s; ON C8 0.777 s, C16 1.782 s.
  ON C16: 26/48 streams reported mtp_depth 0 (served plain under load), 22/48 depth 3. All 72 ON streams draft_used.
  Single-server reference (E1-J1 one-Spark cards, same VMs, same C8 bodies): C8 OFF 73.268 (vm-a) / 70.945 (vm-b),
  C8 ON 97.456 (vm-a) / 102.799 (vm-b). Replicas C8: OFF 141.901, ON 239.396.

WORKLOAD
- RigMark concurrency rows exactly as the card: code workload, 3 rounds, 256 tokens, seed 20260905+stream,
  temperature 0, top_p 1, top_k 0, penalties neutral, cache false, enable_thinking false; OFF adds "draft":"plain",
  ON omits draft. Levels 8 and 16 (the card itself runs 1,2,4,8). Client = rep.py (sha256 6685b3ef...) using the
  E1 lane RigMark copy rigmark/bench.py (sha256 ea249e4e82c1195a1acd3526840d3e29727f60fe62702d73f9a86f68b359557d),
  run on vm-a against the gateway http://127.0.0.1:18181. repcmp.py made replicas-numbers.json.

PUBLIC vade bench (no flags, drafting as served) - run on each VM's own one-Spark server earlier in this job
(bundle route, ./vade launch), receipts vade-bench-vm-a-bundle.json / vade-bench-vm-b-bundle.json:
  vm-a: C1 34.956 C2 70.162 C4 128.245 C8 110.425 C16 124.848 tok/s
  vm-b: C1 32.773 C2 56.751 C4 116.435 C8 99.780 C16 125.304 tok/s

FILES
  replicas-numbers.json                      summary + identity results (repcmp.py)
  replicas-c8-c16-drafting-off.json / -on.json   every request body + response (token ids, vade provenance)
  vade-verify-replica-a.json / -b.json        per-replica vade verify receipts
  health-gateway/replica-a/replica-b-start/end.json
  vade-bench-vm-a-bundle.json / vm-b          public vade bench receipts (one-Spark bundle servers)
  rep.py, repcmp.py                           measurement + comparison scripts
