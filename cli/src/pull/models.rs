//! `vade models`, and the support check a pull records: which Vade route can
//! serve a folder, or why none can yet.
use std::{collections::BTreeMap, fs, path::{Path, PathBuf}};
use crate::evals::json::{self, Value};

/// The Vade route that can serve a model folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    /// Qwen3.8 Flash-Next FP4-Local: `vade launch` and `vade up`.
    Fp4Local,
    /// Qwen3.8 Flash-Next FP8: `vade up --fp8` from a controller.
    Fp8,
    /// The pinned Qwen3-8B BF16 catalog: `vade run qwen3-8b`.
    Qwen3Bf16,
    /// A pull started here and did not finish.
    Incomplete,
    /// No route yet; `reason` says why.
    Unsupported,
}

/// The support state of one folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Support { pub route: Route, pub reason: String }

impl Support {
    pub(crate) fn state(&self) -> &'static str {
        match self.route {
            Route::Fp4Local | Route::Fp8 | Route::Qwen3Bf16 => "Supported",
            Route::Incomplete => "Incomplete",
            Route::Unsupported => "Not supported yet",
        }
    }
    pub(crate) fn text(&self) -> String {
        match self.route {
            Route::Fp4Local => "Qwen3.8 Flash-Next FP4-Local; serve it with vade launch or vade up".into(),
            Route::Fp8 => "Qwen3.8 Flash-Next FP8; serve it from a controller with vade up --fp8".into(),
            Route::Qwen3Bf16 => "the pinned Qwen3-8B BF16 files; serve them with vade run qwen3-8b".into(),
            Route::Incomplete => "a pull did not finish; run the same vade pull again".into(),
            Route::Unsupported => self.reason.clone(),
        }
    }
    /// The step that serves `folder`, and a note that explains it. `saved` is
    /// the checkpoint `vade launch` serves first, ahead of VADE_MODELS, when one
    /// is saved; `home` holds the new Vade home the step then suggests.
    pub(crate) fn next(&self, folder: &Path, saved: Option<&Path>, home: &Path) -> Option<(String, Option<String>)> {
        let quote = |path: &Path| crate::runner::quote(&path.to_string_lossy());
        let same = saved.is_some_and(|saved| fs::canonicalize(saved).is_ok_and(|saved| fs::canonicalize(folder).is_ok_and(|folder| folder == saved)));
        let new_home = || {
            // The org and name, and a number when that home is already in use.
            let part = |path: Option<&Path>| path.and_then(Path::file_name).map(|name| name.to_string_lossy()
                .chars().filter(|c| c.is_ascii_alphanumeric() || "._-".contains(*c)).collect::<String>()).unwrap_or_default();
            let base = format!(".vade-{}-{}", part(folder.parent()), part(Some(folder)));
            (1..).map(|n| home.join(if n == 1 { base.clone() } else { format!("{base}-{n}") }))
                .find(|path| fs::symlink_metadata(path).is_err()).unwrap_or_else(|| home.join(&base))
        };
        let shown = quote(folder);
        match self.route {
            Route::Fp4Local => Some(match saved {
                _ if same => ("vade launch".into(), None),
                None => (format!("VADE_MODELS={shown} vade launch"), None),
                Some(saved) => (format!("VADE_HOME={} VADE_MODELS={shown} vade launch", quote(&new_home())),
                    Some(format!("vade launch serves its saved checkpoint {} first, so this step uses a new, unused Vade home.", quote(saved)))),
            }),
            Route::Fp8 => Some((format!("set model_path = {shown} in the controller config, then run vade up --fp8"), None)),
            Route::Qwen3Bf16 => Some((format!("vade run qwen3-8b --weights-source {shown}"), None)),
            Route::Unsupported => Some(("docs/adding-a-model.md lists what a new model needs".into(), None)),
            Route::Incomplete => None,
        }
    }
    pub(crate) fn json(&self) -> Value {
        let (state, route) = match self.route {
            Route::Fp4Local => ("supported", "fp4-local"),
            Route::Fp8 => ("supported", "fp8"),
            Route::Qwen3Bf16 => ("supported", "qwen3-8b"),
            Route::Incomplete => ("incomplete", "none"),
            Route::Unsupported => ("not supported yet", "none"),
        };
        json::object(vec![("state", state.into()), ("route", route.into()), ("reason", if self.reason.is_empty() { Value::Null } else { self.reason.as_str().into() })])
    }
}

/// Check `folder` with the same metadata checks `vade launch` runs, then the
/// pinned Qwen3-8B catalog. `hashes` are SHA-256 values a pull just checked;
/// without them the folder's manifest supplies them.
pub(crate) fn support(folder: &Path, hashes: Option<&BTreeMap<String, String>>) -> Support {
    let reason = match crate::launch::serving_format(folder) {
        Ok(true) => return Support { route: Route::Fp4Local, reason: String::new() },
        Ok(false) => return Support { route: Route::Fp8, reason: String::new() },
        Err(reason) => reason,
    };
    // Each catalog file must be in the folder with its pinned SHA-256 as a pull
    // recorded it; names alone are not enough.
    let recorded = hashes.cloned().or_else(|| manifest(folder).map(|manifest| recorded(&manifest))).unwrap_or_default();
    let catalog = crate::catalog::MODEL_FILES.iter().all(|(name, sha)| recorded.get(*name).is_some_and(|hash| hash == sha)
        && fs::symlink_metadata(folder.join(name)).is_ok_and(|meta| meta.is_file()));
    if catalog { return Support { route: Route::Qwen3Bf16, reason: String::new() }; }
    Support { route: Route::Unsupported, reason: crate::text::terminal_text(&reason.replace(['\r', '\n'], " ")).into_owned() }
}

pub(super) fn manifest(folder: &Path) -> Option<Value> {
    let bytes = super::read_small(&folder.join(super::MANIFEST), 16 << 20, false).ok()?;
    json::parse(std::str::from_utf8(&bytes).ok()?).ok()
}

fn recorded(manifest: &Value) -> BTreeMap<String, String> {
    manifest["files"].as_array().into_iter().flatten()
        .filter_map(|file| Some((file["path"].as_str()?.to_owned(), file["sha256"].as_str()?.to_owned()))).collect()
}

/// Run `vade models [FOLDER] [--json]`.
pub fn run(args: &[String]) -> Result<(), String> {
    let mut json = false;
    let mut root = None;
    for arg in args {
        match arg.as_str() {
            "--json" if !json => json = true,
            value if !value.starts_with('-') && root.is_none() => root = Some(PathBuf::from(value)),
            value => return Err(format!("Unexpected argument {} for vade models. Run vade models --help.", crate::text::argument(value))),
        }
    }
    // From here, with --json, a failure is the schema object with error and next.
    if json { crate::checks::set_json(Some(crate::checks::Answer { schema: "vade.models.v1", mode: "models", dry: false })); }
    let home = std::env::var_os("HOME").filter(|home| !home.is_empty()).map(PathBuf::from).ok_or("Cannot find your home folder. Set HOME.")?;
    let root = root.unwrap_or_else(|| home.join("models"));
    let rows = list(&root)?;
    let shown = super::display(&root, &home);
    if json {
        let models = rows.iter().map(|row| json::object(vec![
            ("name", row.name.as_str().into()), ("path", row.path.to_string_lossy().into_owned().into()),
            ("repo", row.repo.clone().map_or(Value::Null, Value::from)), ("commit", row.commit.clone().map_or(Value::Null, Value::from)),
            ("bytes", Value::Int(row.bytes as i64)), ("support", row.support.json()),
        ])).collect();
        let value = json::object(vec![("schema", "vade.models.v1".into()), ("root", root.to_string_lossy().into_owned().into()), ("models", Value::Array(models))]);
        return crate::checks::print_json(&value);
    }
    if rows.is_empty() {
        return crate::output::line(format_args!("No models in {shown} yet. Download one with vade pull <org>/<name>; see vade pull --help."));
    }
    crate::output::text(format_args!("{}", table(&rows, &shown)))
}

/// The `vade models` table: two lines for each folder.
fn table(rows: &[Row], shown: &str) -> String {
    let width = rows.iter().map(|row| row.name.chars().count()).max().unwrap_or(0).max(5);
    let source = |row: &Row| row.repo.as_ref().map_or("-".into(), |repo| format!("{repo}@{}", row.commit.as_deref().map_or("?", |c| &c[..c.len().min(12)])));
    let swidth = rows.iter().map(|row| source(row).chars().count()).max().unwrap_or(0).max(6);
    let mut out = format!("Models in {shown}:\n  {:width$}  {:>10}  {:swidth$}  STATE\n", "MODEL", "SIZE", "SOURCE");
    for row in rows {
        out += &format!("  {:width$}  {:>10}  {:swidth$}  {}\n", crate::text::terminal_text(&row.name), super::bytes(row.bytes), source(row), row.support.state());
        out += &format!("  {:width$}  {:>10}  {:swidth$}  {}\n", "", "", "", row.support.text());
    }
    out
}

/// One local model folder.
struct Row { name: String, path: PathBuf, repo: Option<String>, commit: Option<String>, bytes: u64, support: Support }

/// Every model folder under `root`, three levels deep at most.
fn list(root: &Path) -> Result<Vec<Row>, String> {
    if !root.exists() { return Ok(Vec::new()); }
    let mut found = Vec::new();
    let mut pending = vec![(root.to_path_buf(), 0usize)];
    let mut seen = 0usize;
    while let Some((folder, depth)) = pending.pop() {
        if is_model(&folder) { found.push(folder); continue; }
        if depth == 3 { continue; }
        let entries = fs::read_dir(&folder).map_err(|error| crate::text::file_error("Cannot read the model folder", &folder, &error))?;
        for entry in entries.flatten() {
            seen += 1;
            if seen > 10_000 { return Err("The model folder is too large to list; name a smaller folder.".into()); }
            if entry.file_name().to_string_lossy().starts_with('.') { continue; }
            let Ok(kind) = entry.file_type() else { continue };
            // Follow a link only to a model folder, as launch does.
            if kind.is_dir() { pending.push((entry.path(), depth + 1)); }
            else if kind.is_symlink() && fs::canonicalize(entry.path()).is_ok_and(|target| target.is_dir() && is_model(&target)) { found.push(entry.path()); }
        }
    }
    found.sort();
    Ok(found.into_iter().map(|path| {
        let manifest = manifest(&path);
        let text = |key: &str| manifest.as_ref().and_then(|m| m[key].as_str().map(str::to_owned));
        let state = path.join(super::STATE);
        let unfinished = fs::read_dir(&state).into_iter().flatten().flatten().any(|entry| entry.file_name().to_string_lossy().ends_with(".part"))
            || fs::symlink_metadata(state.join(super::UPDATING)).is_ok() || (manifest.is_none() && state.is_dir());
        let support = if unfinished { Support { route: Route::Incomplete, reason: String::new() } } else { support(&path, None) };
        let name = path.strip_prefix(root).ok().filter(|rest| !rest.as_os_str().is_empty()).unwrap_or(&path).display().to_string();
        // The manifest is only a record: show a source only when it is well formed.
        let repo = text("repo").filter(|repo| super::hub::repo(repo).is_ok());
        let commit = text("commit").filter(|commit| crate::text::is_lower_hex(commit, 40));
        Row { name, repo, commit, bytes: size(&path), support, path }
    }).collect())
}

fn is_model(folder: &Path) -> bool {
    folder.join("config.json").is_file() || folder.join(super::MANIFEST).is_file() || folder.join(super::STATE).is_dir()
}

/// Bytes in regular files under `folder`, without links or pull state.
fn size(folder: &Path) -> u64 {
    let mut total = 0u64;
    let mut pending = vec![fs::canonicalize(folder).unwrap_or_else(|_| folder.to_path_buf())];
    let mut seen = 0usize;
    while let Some(folder) = pending.pop() {
        for entry in fs::read_dir(&folder).into_iter().flatten().flatten() {
            seen += 1;
            if seen > 100_000 { return total; }
            if entry.file_name() == super::STATE { continue; }
            match entry.metadata() {
                Ok(meta) if meta.is_dir() => pending.push(entry.path()),
                Ok(meta) if meta.is_file() => total = total.saturating_add(meta.len()),
                _ => {}
            }
        }
    }
    total
}

/// (name, bytes, repository, commit, support) for each folder `vade models`
/// lists, and the text table it prints.
#[cfg(test)]
pub(super) fn tests_list(root: &Path) -> (Vec<(String, u64, Option<String>, Option<String>, Support)>, String) {
    let rows = list(root).unwrap();
    let text = table(&rows, &root.to_string_lossy());
    (rows.into_iter().map(|row| (row.name, row.bytes, row.repo, row.commit, row.support)).collect(), text)
}
