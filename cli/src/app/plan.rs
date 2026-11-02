//! Resolve launch paths, fabric addresses, artifacts and the worker environment.
use super::*;

/// The refusal of a start whose ranks hold their PTX in different folders.
pub(super) const PTX_FOLDERS: &str = "expanded absolute PTX paths differ";

impl<R: Runner> App<R> {
    pub(super) fn resolve_rank_paths(&mut self, session: &mut Session) -> Result<(), String> {
        let mut shared_paths = None;
        for index in 0..session.ranks.len() {
            let rank = &session.ranks[index];
            let reply = self.worker(
                session,
                rank,
                "paths",
                Fields::from([
                    ("ptx_path".into(), rank.artifact.ptx.clone()),
                    ("model_path".into(), rank.machine.model_path.clone()),
                ]),
            )?;
            if self.dry {
                continue;
            }
            let resolved = |key| {
                reply
                    .get(key)
                    .filter(|path| {
                        Path::new(path).is_absolute()
                            && crate::state::validate_remote_path(path).is_ok()
                    })
                    .cloned()
                    .ok_or_else(|| format!(
                        "{}: the helper did not return valid absolute PTX and model paths. Nothing started; run vade install.",
                        rank.machine.name
                    ))
            };
            let ptx = resolved("ptx_path")?;
            let model = resolved("model_path")?;
            if let Some((first_name, first_ptx, first_model)) = &shared_paths {
                for (label, first, current) in
                    [("PTX", first_ptx, &ptx), ("model", first_model, &model)]
                {
                    if first != current {
                        return Err(format!(
                            "{} and {first_name}: expanded absolute {label} paths differ. Nothing started. Use identical expanded absolute PTX and model paths on every selected rank; check remote stage_dir and model_path, including ~/ expansion, then rerun vade install if staging changed.",
                            rank.machine.name
                        ));
                    }
                }
            } else {
                shared_paths = Some((rank.machine.name.clone(), ptx.clone(), model.clone()));
            }
            // Persist and launch the exact strings checked above, even if a
            // later SSH session has a different HOME. run_hash hashes argv.
            session.ranks[index].artifact.ptx = ptx;
            session.ranks[index].machine.model_path = model;
        }
        if self.dry {
            crate::output::line(format_args!("# expanded absolute PTX/model path equality is unchecked until live vade up"))?;
        }
        Ok(())
    }

    pub(super) fn resolve_fabric(&mut self, session: &mut Session) -> Result<(), String> {
        let mut addresses = BTreeMap::new();
        for rank in &session.ranks {
            for address in [&rank.listen] {
                if address.parse::<std::net::SocketAddr>().is_ok()
                    || addresses.contains_key(address)
                {
                    continue;
                }
                // The host's resolver is authoritative for its fabric names.
                let host = address.rsplit_once(':').ok_or("Invalid fabric address.")?.0;
                let exec = runner::ssh(
                    &rank.machine.ssh,
                    format!("getent ahostsv4 {}", quote(host)),
                    "fabric address resolution",
                );
                let output = self.checked(exec, "Cannot resolve fabric hostname. Set a numeric fabric address or fix its existing DNS entry.")?;
                if self.dry {
                    continue;
                }
                let ip = output.stdout.lines().filter_map(|line| line.split_whitespace().next())
                    .find_map(|value| value.parse::<std::net::Ipv4Addr>().ok())
                    .ok_or("Fabric hostname did not resolve to an IPv4 address. Set a numeric address in config.")?;
                let port = address.rsplit_once(':').unwrap().1;
                addresses.insert(address.clone(), format!("{ip}:{port}"));
            }
        }
        for rank in &mut session.ranks {
            for address in [&mut rank.listen, &mut rank.next] {
                if let Some(numeric) = addresses.get(address) {
                    *address = numeric.clone();
                }
                if !self.dry && address.parse::<std::net::SocketAddr>().is_err() {
                    return Err(
                        "Fabric address did not resolve to a numeric socket address.".into(),
                    );
                }
            }
        }
        Ok(())
    }

    pub(super) fn manifest(&self) -> Result<Manifest, String> {
        if self.dry && !self.config.build_manifest.exists() {
            crate::output::line(format_args!(
                "# no build manifest yet; artifact hashes below are placeholders until vade install"
            ))?;
            let artifacts = self
                .config
                .machines
                .iter()
                .map(|m| {
                    let dir = format!(
                        "{}/builds/{}-{}",
                        m.stage_dir,
                        "0".repeat(40),
                        "0".repeat(40)
                    );
                    (
                        m.name.clone(),
                        Artifact {
                            worker: format!("{dir}/vade"),
                            binary: format!("{dir}/flash-serve"),
                            ptx: format!("{dir}/flash.ptx"),
                            binary_sha256: "0".repeat(64),
                            ptx_sha256: "0".repeat(64),
                            worker_sha256: "0".repeat(64),
                        },
                    )
                })
                .collect();
            return Ok(Manifest {
                engine_sha: "0".repeat(40),
                cli_sha: "0".repeat(40),
                harness_sha256: None,
                supports_output: true,
                supports_fp4_chain: true,
                artifacts,
            });
        }
        Manifest::load(&self.config.build_manifest)
    }

    pub(super) fn environment(&self, session: &Session) -> Result<Fields, String> {
        let mut fields = Fields::new();
        let rdma = if session.chained() && rdma_enabled()? { "1" } else { "0" };
        // The engine chooses the path itself: two FP4-Local Sparks run the tensor split over RDMA
        // (requested_session refuses TCP); chains and FP8 pairs use RDMA or TCP (SEQPAR_RDMA below).
        for (name, value) in [
            (
                "SEQPAR_CONTROL_SECRET",
                if self.dry {
                    "<generated-secret>".into()
                } else {
                    text::random_hex(32)?
                },
            ),
            ("SEQPAR_RDMA", rdma.into()),
            ("TP_TOPOLOGY", session.topology().into()),
            (
                "TP_RUN_ID",
                format!(
                    "vade-{}",
                    if self.dry {
                        "<new-run-id>".into()
                    } else {
                        text::random_hex(12)?
                    }
                ),
            ),
            ("SPARK_PROFILE", "0".into()),
            ("FLASH_BF16_EXPERTS", "0".into()),
            ("SPARK_KV_BF16", "0".into()),
            ("SPARK_ADAPTIVE_PREFILL", "0".into()),
            ("TP_WATCH_STDIN", "0".into()),
            ("SEQPAR_TRACE_TRANSFERS", "0".into()),
            ("SEQPAR_SERIAL_COMPUTE", "0".into()),
            ("TP_RDMA", rdma.into()),
            ("TP_RDMA_SMALL", rdma.into()),
            ("SEQPAR_OVERLAP", "1".into()),
            ("FLASH_SERVE_OUTPUT_TOKENS", session.output.to_string()),
        ] {
            fields.insert(format!("env.{name}"), value);
        }
        for (name, value) in [
            ("SEQPAR_MEMORY_BYTES", session.memory_bytes),
            ("FLASH_SERVE_KV_BYTES", session.cache_bytes),
        ] {
            if let Some(bytes) = value {
                fields.insert(format!("env.{name}"), bytes.to_string());
            } else {
                fields.insert(format!("env_unset.{name}"), "1".into());
            }
        }
        if let Some(chats) = session.chats {
            fields.insert("env.FLASH_SERVE_MAX_ACTIVE".into(), chats.to_string());
        } else {
            fields.insert("env_unset.FLASH_SERVE_MAX_ACTIVE".into(), "1".into());
        }
        for name in [
            "MTP",
            "FLASH_BF16_MMA",
            "TP_DEADLINE_UNIX",
            "FLASH_DUMP_LOGITS",
            "TP_CPU",
            "TP_AUX_LISTEN",
            "TP_AUX_NEXT",
            "FLASH_NVFP4_MANIFEST",
            "FLASH_SERVE_GENERATION_TIMEOUT_MS",
        ] {
            fields.insert(format!("env_unset.{name}"), "1".into());
        }
        if session.fp4_local {
            // Keep the checkpoint receipt in each rank's state folder, never in
            // the checkpoint. A chain starts every rank from no receipt, so
            // different start histories cannot split its content identity.
            fields.insert("model_receipt".into(), if session.chained() { "pipeline" } else { "private" }.into());
        }
        Ok(fields)
    }
}
