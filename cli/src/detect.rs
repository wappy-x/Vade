//! Read the machine we are on; OEM names and advertised capacities are not requirements.
use std::fs;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const GIB: u64 = 1024 * 1024 * 1024;
// Unified RAM holds weights, context, work buffers and the operating system.
// The launcher only refuses memory that cannot hold the checkpoint plus 8 GiB
// for OS/workspace; the engine's own admission decides the context it funds.
const WORKING_RESERVE: u64 = 8 * GIB;
const MEMORY_RESERVE: u64 = WORKING_RESERVE;
const MODEL_DISK_RESERVE: u64 = GIB;

/// The engine's startup ceiling: MemAvailable less its 10% reserve, at least
/// 2 GiB (flash-serve bootstrap). Reported only; the engine enforces it.
pub(crate) fn engine_ceiling(available: u64) -> u64 {
    available.saturating_sub((available / 10).max(2 * GIB))
}

/// The memory quantities every preflight prints, from /proc/meminfo.
pub(crate) fn memory_summary(total: u64, available: u64) -> String {
    format!(
        "{:.1} GiB available of {:.1} GiB (page cache counts as available); the engine may use up to {:.1} GiB",
        gib(available),
        gib(total),
        gib(engine_ceiling(available)),
    )
}

/// Require the checkpoint and the same working room as a local launch.
pub(crate) fn check_memory(total: u64, available: u64, model_bytes: u64) -> Result<(), String> {
    let required_memory = model_bytes.saturating_add(MEMORY_RESERVE);
    if total < required_memory || available < required_memory {
        return Err(format!(
            "Not enough memory: {}. The checkpoint and working room need {:.1} GiB before any context. Dropping the page cache does not help. Stop other programs that hold memory (see free -g and nvidia-smi), then run vade doctor and retry vade up.",
            memory_summary(total, available),
            gib(required_memory),
        ));
    }
    Ok(())
}

/// What `vade detect` reads about this Spark: names, memory, free disk on the
/// install and model volumes, and the NVIDIA driver and GPU.
#[derive(Clone, Debug)]
pub struct Identity {
    pub vendor: String,
    pub product: String,
    pub os: String,
    pub architecture: String,
    pub memory_total: u64,
    pub memory_available: u64,
    pub install_free: u64,
    pub models_free: u64,
    pub volumes_shared: bool,
    pub driver: String,
    /// The driver's supported CUDA version reported by nvidia-smi, not a toolkit version.
    pub cuda: String,
    pub gpu: String,
}

impl Identity {
    /// The report `vade detect` and `vade doctor` print.
    pub fn display(&self, models: &Path) -> String {
        format!(
            "Machine: {} / {}\nOS: {} ({})\nMemory: {}\nDisk free: {:.1} GiB install, {:.1} GiB models{}\nGPU: {}\nDriver: {}   CUDA supported: {}\nModels: {}\n",
            plain(&self.vendor),
            plain(&self.product),
            plain(&self.os),
            plain(&self.architecture),
            memory_summary(self.memory_total, self.memory_available),
            gib(self.install_free),
            gib(self.models_free),
            if self.volumes_shared {
                " (same volume)"
            } else {
                ""
            },
            plain(&self.gpu),
            plain(&self.driver),
            plain(&self.cuda),
            crate::text::quoted(&models.to_string_lossy()),
        )
    }

    /// The model is already present: do not require space for another copy.
    pub fn check_resources(&self, model_bytes: u64, install_required: u64) -> Result<(), String> {
        check_memory(self.memory_total, self.memory_available, model_bytes)?;
        let install_need = install_required.saturating_add(if self.volumes_shared {
            MODEL_DISK_RESERVE
        } else {
            0
        });
        if self.install_free < install_need {
            return Err(format!(
                "Not enough install disk space: {:.1} GiB free; {:.1} GiB needed{}. Free space on the install volume and run vade doctor before vade install.",
                gib(self.install_free),
                gib(install_need),
                if self.volumes_shared {
                    " including model-volume working room"
                } else {
                    ""
                },
            ));
        }
        if self.models_free < MODEL_DISK_RESERVE {
            return Err(format!(
                "Not enough model disk space: {:.1} GiB free; {:.1} GiB needed for working room. Free disk space and run vade doctor before vade launch.",
                gib(self.models_free),
                gib(MODEL_DISK_RESERVE),
            ));
        }
        Ok(())
    }
}

/// Read this machine, with free space measured for `install` and `models`.
pub fn detect(install: &Path, models: &Path) -> Result<Identity, String> {
    detect_at(Path::new("/"), install, models, command)
}

fn detect_at(
    root: &Path,
    install: &Path,
    models: &Path,
    mut run: impl FnMut(&str, &[String]) -> Result<String, String>,
) -> Result<Identity, String> {
    let meminfo = fs::read_to_string(root.join("proc/meminfo")).map_err(
        |_| "Cannot read Linux memory information from /proc/meminfo; run Vade on the Spark.",
    )?;
    let (memory_total, memory_available) = memory(&meminfo)?;
    let architecture = run("uname", &["-m".into()])
        .map_err(|_| "Cannot read the machine architecture with uname.")?;
    let architecture = plain(&architecture);
    if architecture != "aarch64" {
        return Err(
            "Vade's Spark launcher needs Linux aarch64; run this command on the Spark.".into(),
        );
    }
    let vendor = optional_line(&root.join("sys/class/dmi/id/sys_vendor"));
    let product = optional_line(&root.join("sys/class/dmi/id/product_name"));
    let os = fs::read_to_string(root.join("etc/os-release"))
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("PRETTY_NAME=")
                    .map(|s| plain(s.trim_matches(['\"', '\''])))
            })
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Linux (version unavailable)".into());
    let (install_free, install_dev) = disk(install, "install", &mut run)?;
    let (models_free, models_dev) = disk(models, "model", &mut run)?;
    let query = run(
        "nvidia-smi",
        &[
            "--query-gpu=name,driver_version".into(),
            "--format=csv,noheader,nounits".into(),
        ],
    )
    .map_err(
        |_| "Cannot read the NVIDIA GPU and driver; nvidia-smi must work before Vade can launch.",
    )?;
    let (gpu, driver) = gpu_driver(&query)?;
    if !gpu.split(',').all(|name| {
        name.split(|c: char| !c.is_ascii_alphanumeric())
            .any(|word| word.eq_ignore_ascii_case("GB10"))
    }) {
        return Err(
            "The Spark launcher needs an NVIDIA GB10 GPU; nvidia-smi reported another GPU.".into(),
        );
    }
    let header = run("nvidia-smi", &[]).map_err(
        |_| "Cannot read CUDA support from nvidia-smi; check the installed NVIDIA driver.",
    )?;
    let cuda = cuda_version(&header)?;
    // The shipped GPU build uses CUDA Driver API 13000 (dependency-manifest.txt).
    // Require that supported build baseline before attempting to load its PTX.
    if cuda.split('.').next().and_then(|n| n.parse::<u32>().ok()).is_none_or(|major| major < 13) {
        return Err(format!("NVIDIA driver {driver} supports CUDA {cuda}; this Vade build requires driver CUDA support 13.0 or newer. Run nvidia-smi to check the driver, then vade doctor before vade launch."));
    }
    Ok(Identity {
        vendor,
        product,
        os,
        architecture,
        memory_total,
        memory_available,
        install_free,
        models_free,
        volumes_shared: install_dev == models_dev,
        driver,
        cuda,
        gpu,
    })
}

fn memory(text: &str) -> Result<(u64, u64), String> {
    let field = |name: &str| {
        let mut matches = text.lines().filter_map(|line| line.strip_prefix(name));
        let value = matches.next()?;
        if matches.next().is_some() {
            return None;
        }
        let mut words = value.split_whitespace();
        let kib = words.next()?.parse::<u64>().ok()?;
        if words.next()? != "kB" || words.next().is_some() {
            return None;
        }
        kib.checked_mul(1024)
    };
    match (field("MemTotal:"), field("MemAvailable:")) {
        (Some(total), Some(available)) if total > 0 && available <= total => Ok((total, available)),
        _ => Err("Cannot read MemTotal and MemAvailable from /proc/meminfo; memory admission is unavailable.".into()),
    }
}

fn disk(
    path: &Path,
    label: &str,
    run: &mut impl FnMut(&str, &[String]) -> Result<String, String>,
) -> Result<(u64, u64), String> {
    let probe = existing_ancestor(path).map_err(|_| {
        format!("Cannot inspect the {label} volume; check its folder and permissions.")
    })?;
    let device = fs::metadata(&probe)
        .map_err(|_| {
            format!("Cannot inspect the {label} volume; check its folder and permissions.")
        })?
        .dev();
    let output = run("df", &["-Pk".into(), probe.to_string_lossy().into_owned()])
        .map_err(|_| format!("Cannot read free space on the {label} volume with df."))?;
    let free = df_available(&output)
        .ok_or_else(|| format!("Cannot read free space on the {label} volume from df."))?;
    Ok((free, device))
}

fn existing_ancestor(path: &Path) -> std::io::Result<PathBuf> {
    let mut path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    loop {
        match fs::metadata(&path) {
            Ok(_) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if !path.pop() {
                    return Err(error);
                }
            }
            Err(error) => return Err(error),
        }
    }
}

fn df_available(text: &str) -> Option<u64> {
    // A mount path may contain spaces. Locate the capacity column, then take
    // the available 1-KiB blocks immediately before it, not the path columns.
    let line = text.lines().skip(1).find(|line| !line.trim().is_empty())?;
    let words: Vec<_> = line.split_whitespace().collect();
    let capacity = words.iter().position(|word| {
        word.strip_suffix('%')
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()))
    })?;
    if capacity < 4 {
        return None;
    }
    words[capacity - 1].parse::<u64>().ok()?.checked_mul(1024)
}

fn gpu_driver(text: &str) -> Result<(String, String), String> {
    let mut names = Vec::new();
    let mut drivers = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let (name, driver) = line
            .rsplit_once(',')
            .ok_or("Cannot read the GPU name and driver version from nvidia-smi.")?;
        let name = plain(name);
        let driver = driver.trim();
        if name.is_empty() || !version(driver) {
            return Err("Cannot read the GPU name and driver version from nvidia-smi.".into());
        }
        if !names.contains(&name) {
            names.push(name);
        }
        if !drivers.contains(&driver) {
            drivers.push(driver);
        }
    }
    if names.is_empty() {
        return Err("No NVIDIA GPU was reported by nvidia-smi.".into());
    }
    Ok((names.join(", "), drivers.join(", ")))
}

fn cuda_version(text: &str) -> Result<String, String> {
    let value = text
        .split_once("CUDA Version:")
        .and_then(|(_, tail)| tail.split_whitespace().next())
        .filter(|value| version(value));
    value.map(str::to_owned).ok_or_else(|| {
        "Cannot read CUDA support from nvidia-smi; check the installed NVIDIA driver.".into()
    })
}

fn version(value: &str) -> bool {
    let parts: Vec<_> = value.split('.').collect();
    parts.len() >= 2
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|c| c.is_ascii_digit()))
}

fn optional_line(path: &Path) -> String {
    fs::read_to_string(path)
        .ok()
        .map(|value| plain(&value))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unavailable".into())
}

fn plain(value: &str) -> String {
    let bounded: String = value.chars().take(256).collect();
    crate::text::terminal_text(&bounded)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn gib(value: u64) -> f64 {
    value as f64 / GIB as f64
}

fn command(program: &str, args: &[String]) -> Result<String, String> {
    let error = || format!("Cannot complete {program} detection.");
    let mut child = Command::new(program)
        .args(args)
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()
        .map_err(|_| error())?;
    let mut stdout = child.stdout.take().ok_or_else(error)?;
    let (send, receive) = std::sync::mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout
            .by_ref()
            .take(64 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes);
        let _ = send.send(result);
    });
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if start.elapsed() < Duration::from_secs(5) => {
                std::thread::sleep(Duration::from_millis(10))
            }
            _ => break Err(error()),
        }
    };
    if status.is_err() {
        // The leader has not been reaped, so this process-group ID still
        // belongs to this probe. Never signal a PID after successful wait.
        unsafe extern "C" {
            fn kill(pid: i32, signal: i32) -> i32;
        }
        unsafe {
            kill(-(child.id() as i32), 9);
        }
        let _ = child.wait();
        return Err(error());
    }
    let bytes = receive
        .recv_timeout(Duration::from_secs(5).saturating_sub(start.elapsed()))
        .map_err(|_| error())?
        .map_err(|_| error())?;
    if !status?.success() || bytes.len() > 64 * 1024 {
        return Err(error());
    }
    String::from_utf8(bytes).map_err(|_| error())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new(vendor: &str, product: &str, total_gib: u64) -> Self {
            let root = std::env::temp_dir().join(format!(
                "vade-detect-{}",
                crate::text::random_hex(12).unwrap()
            ));
            for dir in ["sys/class/dmi/id", "proc", "etc", "home/models"] {
                fs::create_dir_all(root.join(dir)).unwrap();
            }
            fs::write(
                root.join("sys/class/dmi/id/sys_vendor"),
                format!("{vendor}\n"),
            )
            .unwrap();
            fs::write(
                root.join("sys/class/dmi/id/product_name"),
                format!("{product}\n"),
            )
            .unwrap();
            fs::write(
                root.join("proc/meminfo"),
                format!(
                    "MemTotal: {} kB\nMemAvailable: {} kB\nSwapFree: 999999999 kB\n",
                    total_gib * GIB / 1024,
                    (total_gib - 4) * GIB / 1024,
                ),
            )
            .unwrap();
            fs::write(
                root.join("etc/os-release"),
                "PRETTY_NAME=\"DGX OS fixture / Ubuntu 24.04\"\n",
            )
            .unwrap();
            Self(root)
        }
        fn identity(&self) -> Identity {
            self.detect(fake_command).unwrap()
        }
        fn detect(
            &self,
            run: impl FnMut(&str, &[String]) -> Result<String, String>,
        ) -> Result<Identity, String> {
            detect_at(
                &self.0,
                &self.0.join("home/.vade/stage"),
                &self.0.join("home/models"),
                run,
            )
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fake_command(program: &str, args: &[String]) -> Result<String, String> {
        match program {
            "uname" => { assert_eq!(args, ["-m"]); Ok("aarch64\n".into()) }
            "df" => {
                assert_eq!(args[0], "-Pk");
                assert!(Path::new(&args[1]).exists());
                Ok("Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/test 1000000000 600000000 400000000 60% /a path with spaces\n".into())
            }
            "nvidia-smi" if args.is_empty() => Ok("| NVIDIA-SMI 580.178.04 Driver Version: 580.178.04 CUDA Version: 13.0 |\n| Memory-Usage: Not Supported |\n".into()),
            "nvidia-smi" => {
                assert_eq!(args, ["--query-gpu=name,driver_version", "--format=csv,noheader,nounits"]);
                Ok("NVIDIA GB10, 580.178.04\n".into())
            }
            _ => panic!("unexpected command {program}"),
        }
    }

    // Only NVIDIA and GIGABYTE strings are reported observations. The other
    // values are illustrative fixtures, never a brand allowlist or claim of DMI spelling.
    macro_rules! brand {
        ($name:ident, $vendor:literal, $product:literal, $ram:literal) => {
            #[test]
            fn $name() {
                let fixture = Fixture::new($vendor, $product, $ram);
                let identity = fixture.identity();
                assert_eq!(identity.vendor, $vendor);
                assert_eq!(identity.product, $product);
                assert_eq!(identity.memory_total, $ram * GIB);
                assert_eq!(identity.memory_available, ($ram - 4) * GIB);
                assert_eq!(identity.gpu, "NVIDIA GB10");
                assert_eq!(identity.driver, "580.178.04");
                assert_eq!(identity.cuda, "13.0");
                assert_eq!(identity.architecture, "aarch64");
                assert!(identity.volumes_shared);
                assert!(
                    identity
                        .display(&fixture.0.join("home/models"))
                        .contains($product)
                );
                assert!(
                    !fixture.0.join("home/.vade").exists(),
                    "detect must not create installation folders"
                );
            }
        };
    }
    brand!(nvidia_founders, "NVIDIA", "NVIDIA_DGX_Spark", 122);
    brand!(dell_pro_max, "Dell Inc.", "Pro Max GB10 fixture", 60);
    brand!(
        lenovo_thinkstation,
        "LENOVO",
        "ThinkStation PGX fixture",
        120
    );
    brand!(
        msi_edgexpert,
        "Micro-Star International",
        "EdgeXpert fixture",
        60
    );
    brand!(
        asus_ascent,
        "ASUSTeK COMPUTER INC.",
        "Ascent GX10 fixture",
        120
    );
    brand!(hp_zgx, "HP", "ZGX Nano fixture", 60);
    brand!(acer_veriton, "Acer", "Veriton GN100 fixture", 120);
    brand!(gigabyte_atom, "GIGABYTE", "AI TOP ATOM", 122);

    #[test]
    fn admission_uses_available_ram_and_counts_shared_disk_once() {
        let fixture = Fixture::new("unknown", "future OEM", 122);
        let mut identity = fixture.identity();
        identity.memory_available = 56 * GIB;
        assert!(
            identity
                .check_resources(96 * GIB, 4 * GIB)
                .unwrap_err()
                .contains("Not enough memory")
        );
        identity.memory_available = 108 * GIB;
        identity.install_free = 4 * GIB;
        assert!(
            identity
                .check_resources(96 * GIB, 4 * GIB)
                .unwrap_err()
                .contains("install disk space")
        );
        identity.install_free = 5 * GIB;
        identity.models_free = 5 * GIB;
        assert!(identity.check_resources(96 * GIB, 4 * GIB).is_ok());
        identity.volumes_shared = false;
        identity.install_free = 4 * GIB;
        assert!(identity.check_resources(96 * GIB, 4 * GIB).is_ok());
        identity.models_free = GIB - 1;
        assert!(
            identity
                .check_resources(96 * GIB, 4 * GIB)
                .unwrap_err()
                .contains("model disk space")
        );
    }

    #[test]
    fn memory_text_names_mem_available_required_and_ceiling() {
        let fixture = Fixture::new("unknown", "future OEM", 122);
        let mut identity = fixture.identity();
        identity.memory_available = 20 * GIB;
        let error = identity.check_resources(96 * GIB, 0).unwrap_err();
        for text in ["20.0 GiB available of 122.0 GiB", "need 104.0 GiB", "may use up to 18.0 GiB", ". Stop other programs"] {
            assert!(error.contains(text), "{error}");
        }
        assert!(identity.display(Path::new("/m")).contains("Memory: 20.0 GiB available of 122.0 GiB (page cache counts as available); the engine may use up to 18.0 GiB"));
        assert_eq!(engine_ceiling(110 * GIB), 99 * GIB);
        assert_eq!(engine_ceiling(GIB), 0);
    }

    #[test]
    fn parse_failures_do_not_guess_ram_disk_or_cuda() {
        for value in [
            "MemTotal: 1 kB\n",
            "MemTotal: 1 kB\nMemAvailable: 2 kB\n",
            "MemTotal: -1 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 18446744073709551615 kB\nMemAvailable: 0 kB\n",
        ] {
            assert!(memory(value).is_err());
        }
        assert_eq!(
            df_available(
                "Filesystem blocks used avail cap path\n/dev/x 20 10 10 50% /volume with spaces\n"
            ),
            Some(10 * 1024)
        );
        assert_eq!(
            df_available("Filesystem blocks used avail cap path\n/dev/x 20 10 -1 50% /\n"),
            None
        );
        assert!(cuda_version("CUDA Version: N/A").is_err());
        assert!(gpu_driver("NVIDIA GB10, N/A").is_err());
        assert!(gpu_driver("").is_err());
        assert_eq!(cuda_version("CUDA Version: 13.1 |"), Ok("13.1".into()));
    }

    #[test]
    fn unknown_dmi_and_driver_versions_are_observations_not_rejections() {
        let fixture = Fixture::new("unknown OEM", "brand new box", 60);
        fs::remove_file(fixture.0.join("sys/class/dmi/id/sys_vendor")).unwrap();
        let identity = fixture.identity();
        assert_eq!(identity.vendor, "unavailable");
        assert_eq!(identity.product, "brand new box");
        assert!(identity.check_resources(16 * GIB, GIB).is_ok());
        assert_eq!(
            gpu_driver("NVIDIA GB10, 595.91.07\n").unwrap().1,
            "595.91.07"
        );
        assert!(!plain("OEM\u{1b}[31m\nspoof\u{202e}").contains('\u{1b}'));
    }

    #[test]
    fn wrong_architecture_gpu_and_broken_driver_fail_in_one_line() {
        let fixture = Fixture::new("OEM", "fixture", 120);
        for failure in ["architecture", "gpu", "driver", "cuda", "df"] {
            let error = fixture
                .detect(|program, args| match (failure, program, args.is_empty()) {
                    ("architecture", "uname", _) => Ok("x86_64\n".into()),
                    ("gpu", "nvidia-smi", false) => Ok("NVIDIA B200, 580.178.04\n".into()),
                    ("driver", "nvidia-smi", _) | ("df", "df", _) => {
                        Err("private child details\nnot for display".into())
                    }
                    ("cuda", "nvidia-smi", true) => Ok("CUDA Version: N/A\n".into()),
                    _ => fake_command(program, args),
                })
                .unwrap_err();
            assert!(!error.contains('\n'), "{failure}: {error}");
            assert!(!error.contains("private child details"));
            assert!(!fixture.0.join("home/.vade").exists());
        }
    }

    #[test]
    fn disk_capacity_and_mount_path_do_not_override_measured_free_space() {
        let fixture = Fixture::new("OEM", "fixture", 120);
        for capacity_gib in [1024, 2048, 4096] {
            let identity = fixture.detect(|program, args| {
                if program == "df" {
                    Ok(format!("Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/spark {} {} {} 99% /models with spaces\n",
                        capacity_gib * GIB / 1024, (capacity_gib - 2) * GIB / 1024, 2 * GIB / 1024))
                } else { fake_command(program, args) }
            }).unwrap();
            assert_eq!(identity.install_free, 2 * GIB);
            assert_eq!(identity.models_free, 2 * GIB);
            assert!(
                identity
                    .check_resources(16 * GIB, 4 * GIB)
                    .unwrap_err()
                    .contains("install disk space")
            );
        }
    }
}
