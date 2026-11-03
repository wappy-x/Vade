//! `vade pull` downloads a model from the Hugging Face Hub, and `vade models`
//! lists the models on this machine.
//!
//! A pull lists the repository at one commit, selects the files Vade needs,
//! downloads each one through curl with resume and retries, and checks its size
//! and hash: the SHA-256 the Hub keeps for large files, else the Git blob ID.
//! It then writes `.vade-model.json` in the folder and a `<folder>.sha256`
//! checksum list beside it, which the FP4-Local routes of `vade launch` and
//! `vade up` use to check every byte before they serve the model (see
//! docs/pull-models.md).
mod fetch;
mod hub;
mod models;
mod record;

pub use models::run as models;
pub(crate) use record::{Verified, save, snapshot, verified};
pub(crate) use record::RECORD;

use std::{collections::BTreeMap, fs, io::IsTerminal, os::unix::fs::MetadataExt, path::{Component, Path, PathBuf}, time::{Duration, Instant}};
use crate::evals::json::{self, Value};
use crate::unix_file::HeldLock;
use fetch::{Failure, Transport};
use hub::Remote;

/// The pull folder's private state: partial files, the lock and the list of
/// files already checked.
pub(crate) const STATE: &str = ".vade-pull";
/// In the pull state from the first file a pull moves into place until its
/// records are written: the folder may mix two commits until then.
const UPDATING: &str = "updating";
/// The manifest file in every pulled folder.
pub(crate) const MANIFEST: &str = ".vade-model.json";
const INTERRUPTED: &str = "Interrupted. Run the same vade pull again to continue.";

/// One `vade pull` request.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Request {
    repo: String,
    revision: String,
    include: Vec<String>,
    to: Option<PathBuf>,
}

fn parse(args: &[String]) -> Result<Request, String> {
    let usage = "Run vade pull --help.";
    let (mut spec, mut include, mut to) = (None, Vec::new(), None);
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        match arg {
            "--include" | "--to" => {
                let value = args.get(index + 1).filter(|value| !value.is_empty() && !value.starts_with("--"))
                    .ok_or_else(|| format!("Give {arg} a value. {usage}"))?;
                if arg == "--include" {
                    if value.chars().any(char::is_control) || value.len() > 1024 { return Err(format!("Give --include a plain pattern. {usage}")); }
                    include.push(value.clone());
                } else if to.replace(PathBuf::from(value)).is_some() {
                    return Err(format!("Use --to only once. {usage}"));
                }
                index += 2;
            }
            _ if arg.starts_with('-') => return Err(format!("Unknown option {} for vade pull. {usage}", crate::text::argument(arg))),
            _ if spec.is_some() => return Err(format!("Give one model. {usage}")),
            _ => { spec = Some(arg.to_owned()); index += 1; }
        }
    }
    let spec = spec.ok_or_else(|| format!("Name the model to pull, such as vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local. {usage}"))?;
    let (repo, revision) = spec.split_once('@').unwrap_or((&spec, "main"));
    hub::repo(repo)?;
    hub::revision(revision)?;
    Ok(Request { repo: repo.into(), revision: revision.into(), include, to })
}

/// Everything a pull reads from the environment, so tests can set it directly.
struct Setup {
    endpoint: String,
    token: Option<String>,
    home: PathBuf,
    /// The Vade home (`VADE_HOME`, else `~/.vade`), for the saved checkpoint.
    root: PathBuf,
    program: String,
    backoff: Duration,
    /// Redraw one progress line in place when standard error is a terminal.
    live: bool,
}

impl Setup {
    fn from_environment() -> Result<Self, String> {
        let endpoint = match std::env::var("HF_ENDPOINT") {
            Ok(value) if !value.is_empty() => hub::endpoint(&value)?,
            _ => hub::DEFAULT_ENDPOINT.into(),
        };
        let home = std::env::var_os("HOME").filter(|home| !home.is_empty()).map(PathBuf::from).ok_or("Cannot find your home folder. Set HOME.")?;
        let root = crate::launch::root()?;
        Ok(Self { endpoint, token: token(&root)?, home, root, program: "curl".into(), backoff: Duration::from_secs(2), live: std::io::stderr().is_terminal() })
    }
}

/// HF_TOKEN, else the token file in the Vade home. The value is never printed.
fn token(root: &Path) -> Result<Option<String>, String> {
    match std::env::var("HF_TOKEN") {
        Ok(value) if !value.trim().is_empty() => token_text(&value, "HF_TOKEN"),
        _ => token_file(&root.join("hf-token")),
    }
}

/// The token in `path`, which may be a link to the real file, but must be a
/// regular file of at most 4 KiB.
fn token_file(path: &Path) -> Result<Option<String>, String> {
    match read_small(path, 4096, true) {
        Ok(bytes) => token_text(&String::from_utf8(bytes).map_err(|_| "The Hugging Face token file is not text.")?, "the token file"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::FileTooLarge => Err("The Hugging Face token file is too large. Keep only the token in it.".into()),
        Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => Err("The Hugging Face token file is not a regular file. Save the token in a plain file.".into()),
        Err(_) => Err("Cannot read the Hugging Face token file. Check its permissions.".into()),
    }
}

fn token_text(value: &str, source: &str) -> Result<Option<String>, String> {
    let value = value.trim();
    if value.is_empty() { return Ok(None); }
    if value.len() > 512 || !value.bytes().all(|b| b.is_ascii_graphic() && b != b'"' && b != b'\\') {
        return Err(format!("The Hugging Face token in {source} has characters a token cannot have. Copy it again from your Hugging Face settings."));
    }
    Ok(Some(value.into()))
}

/// Read at most `limit` bytes of a regular file. The open never blocks on a
/// FIFO, and it follows a symlink only when `follow` is set. A file that is
/// not regular fails with `InvalidInput`; a longer one with `FileTooLarge`.
pub(super) fn read_small(path: &Path, limit: u64, follow: bool) -> std::io::Result<Vec<u8>> {
    let bytes = read_head(path, limit + 1, follow)?;
    if bytes.len() as u64 > limit { return Err(std::io::ErrorKind::FileTooLarge.into()); }
    Ok(bytes)
}

/// The first `limit` bytes of a regular file, opened as [`read_small`] does.
fn read_head(path: &Path, limit: u64, follow: bool) -> std::io::Result<Vec<u8>> {
    use std::{io::Read, os::unix::fs::OpenOptionsExt};
    use crate::unix_flags::{O_CLOEXEC, O_NOFOLLOW, O_NONBLOCK};
    let file = fs::OpenOptions::new().read(true).custom_flags(O_NONBLOCK | O_CLOEXEC | if follow { 0 } else { O_NOFOLLOW }).open(path)?;
    if !file.metadata()?.is_file() { return Err(std::io::ErrorKind::InvalidInput.into()); }
    let mut bytes = Vec::new();
    file.take(limit).read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// Run `vade pull` with these arguments (after the command name).
pub fn run(args: &[String], dry: bool) -> Result<(), String> {
    let request = parse(args)?;
    let _interruptions = crate::runner::Interruptions::install();
    pull(&request, &Setup::from_environment()?, dry, &mut |line| crate::output::line(format_args!("{line}")))
}

fn pull(request: &Request, setup: &Setup, dry: bool, say: &mut dyn FnMut(&str) -> Result<(), String>) -> Result<(), String> {
    let began = Instant::now();
    let folder = target(request, setup)?;
    let shown = display(&folder, &setup.home);
    // A dry run writes nothing in the model folder; its scratch is temporary.
    let (folder, scratch) = if dry {
        let temporary = fs::canonicalize(std::env::temp_dir()).map_err(|_| "Cannot open the system temporary folder. Check TMPDIR.")?;
        (folder, temporary.join(format!("vade-pull-{}", crate::text::random_hex(8)?)))
    } else {
        fs::create_dir_all(&folder).map_err(|error| crate::text::file_error("Cannot create the model folder", &folder, &error))?;
        let folder = fs::canonicalize(&folder).map_err(|_| "Cannot resolve the model folder.")?;
        let scratch = folder.join(STATE);
        (folder, scratch)
    };
    crate::text::private_dir(&scratch)?;
    let result = (|| {
        let _lock = (!dry).then(|| lock(&scratch)).transpose()?;
        let transport = Transport { endpoint: setup.endpoint.clone(), token: setup.token.clone(), scratch: scratch.clone(),
            program: setup.program.clone(), backoff: setup.backoff };
        let commit = hub::commit(&api(&transport, request, &hub::revision_url(&setup.endpoint, &request.repo, &request.revision), "revision")?.1)?;
        if crate::text::is_lower_hex(&request.revision, 40) && request.revision != commit {
            return Err("The Hub answered with another commit than the one requested. Retry, or check HF_ENDPOINT.".into());
        }
        let files = list(&transport, request, &commit)?;
        let chosen = hub::select(&files, &request.include);
        if chosen.is_empty() {
            return Err(format!("No file in {} matches the selection. {}", request.repo,
                if request.include.is_empty() { "Name the files with --include." } else { "Check the --include patterns; * matches any text, / included." }));
        }
        let total: u64 = chosen.iter().map(|file| file.size).sum();
        say(&format!("Pulling {} at {} ({}): {} files, {} into {shown}", request.repo, &commit[..12], request.revision, chosen.len(), bytes(total)))?;
        if dry {
            for file in &chosen { say(&format!("  {:>10}  {}", bytes(file.size), file.path))?; }
            say("Dry run: nothing was downloaded.")?;
            return Ok(());
        }
        // Decide what the records keep before any file changes.
        let selective = !request.include.is_empty();
        if selective && fs::symlink_metadata(scratch.join(UPDATING)).is_ok() {
            return Err("An earlier pull of this folder stopped while it replaced files. Run vade pull without --include to finish it first.".into());
        }
        let kept = kept(&folder, request, &commit, &chosen);
        covered(&folder, &chosen, &kept, selective)?;
        let (hashes, resumed) = download(&transport, request, &commit, &folder, &scratch, &chosen, setup.live)?;
        stopped()?;
        let mut records: Vec<Record> = chosen.iter().map(|file| Record { path: file.path.clone(), size: file.size, sha256: hashes[&file.path].clone(),
            how: if file.sha256.is_some() { "sha256" } else { "git blob id" } }).chain(kept).collect();
        records.sort_by(|a, b| a.path.cmp(&b.path));
        let recorded: BTreeMap<String, String> = records.iter().map(|record| (record.path.clone(), record.sha256.clone())).collect();
        let support = models::support(&folder, Some(&recorded));
        let card = chosen.iter().any(|file| file.path == "README.md");
        write_manifest(request, setup, &folder, &commit, &records, card, &support)?;
        checksums(&folder, &records)?;
        record::after_pull(&folder, &scratch, &commit, &records);
        match fs::remove_file(scratch.join(UPDATING)) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(crate::text::file_error("Cannot clear the update mark", &scratch, &error)),
            _ => {}
        }
        clean(&scratch);
        say(&format!("{}: {}", support.state(), support.text()))?;
        let saved = crate::launch::saved_model(&setup.root);
        if let Some((next, note)) = support.next(&folder, saved.as_deref(), &setup.home) {
            say(&format!("Next: {next}"))?;
            if let Some(note) = note { say(&format!("Note: {note}"))?; }
        }
        say(&format!("Verified {} files, {:.3} GiB in {:.1} s, {resumed} resumed, in {shown}", chosen.len(), gib(total), began.elapsed().as_secs_f64()))?;
        Ok(())
    })();
    if dry { let _ = fs::remove_dir_all(&scratch); }
    result
}

/// The model folder: `--to`, else `~/models/<org>/<name>`.
fn target(request: &Request, setup: &Setup) -> Result<PathBuf, String> {
    let path = match &request.to {
        None => return Ok(setup.home.join("models").join(&request.repo)),
        Some(path) => path,
    };
    let text = path.to_str().filter(|text| !text.chars().any(char::is_control)).ok_or("Give --to a folder path in plain text.")?;
    let path = if text == "~" { setup.home.clone() } else if let Some(rest) = text.strip_prefix("~/") { setup.home.join(rest) } else { path.clone() };
    let path = std::path::absolute(&path).map_err(|_| "Cannot resolve the --to folder. Change to an existing folder and retry.")?;
    if path.parent().is_none() || path.components().any(|part| part == Component::ParentDir) {
        return Err("Give --to a model folder, not the file system root or a path with '..'.".into());
    }
    Ok(path)
}

/// Hold the pull lock for this folder. A pull that dies loses it with its process.
fn lock(scratch: &Path) -> Result<HeldLock, String> {
    let file = fs::File::options().read(true).write(true).create(true).truncate(false).open(scratch.join("lock"))
        .map_err(|_| "Cannot create the pull lock in the model folder.")?;
    HeldLock::try_take(file).ok().flatten().ok_or_else(|| "Another vade pull is writing this model folder. Wait for it to finish, or stop it, then run vade pull again.".into())
}

/// GET one API reply as (headers, body), mapping refusals to the step the
/// user can take.
fn api(transport: &Transport, request: &Request, url: &str, what: &str) -> Result<(String, String), String> {
    let (body, headers) = (transport.scratch.join("reply.json"), transport.scratch.join("reply.headers"));
    let fetched = transport.get(url, &body, &headers);
    // Leave a path that is not a regular file for the owner to inspect.
    if let Err(failure @ Failure::NotRegular(_)) = fetched { return Err(refusal(failure, request, what, transport.token.is_some())); }
    if let Err(Failure::TooLarge) = fetched {
        let _ = (fs::remove_file(&body), fs::remove_file(&headers));
        return Err(format!("The Hub sent a {what} reply over the size limit (64 MiB, headers 1 MiB). Check HF_ENDPOINT, then run vade pull again."));
    }
    let reply = fetched.map_err(|failure| refusal(failure, request, what, transport.token.is_some())).and_then(|()| {
        let text = read_small(&body, fetch::API_BODY, false).ok().and_then(|bytes| String::from_utf8(bytes).ok()).ok_or("Cannot read the Hub reply.")?;
        let headers = read_small(&headers, fetch::API_HEADERS, false).map(|bytes| String::from_utf8_lossy(&bytes).into_owned()).unwrap_or_default();
        Ok((headers, text))
    });
    let _ = (fs::remove_file(&body), fs::remove_file(&headers));
    reply
}

fn refusal(failure: Failure, request: &Request, what: &str, token: bool) -> String {
    let repo = &request.repo;
    let access = if token { "The token was sent; check that it can read this model." }
        else { "For a gated or private model, accept its terms on the model page, then set HF_TOKEN or save the token in ~/.vade/hf-token." };
    match failure {
        // The Hub answers 401, not 404, for a model ID that does not exist.
        Failure::Http(401 | 403) => format!("The Hub refused access to {repo}. Check the model ID: the Hub also refuses an ID that does not exist. {access}"),
        Failure::Http(404) if what == "revision" => format!("The Hub has no model {repo} at revision {}. Check both names. {access}", request.revision),
        Failure::Http(404) => format!("The Hub has no {what} for {repo}. Check the name and revision."),
        Failure::Http(code) => format!("The Hub answered HTTP {code} for the {what} of {repo}. Run vade pull again later."),
        Failure::Missing => "vade pull needs curl on PATH. Install curl, then run vade pull again.".into(),
        Failure::Interrupted => INTERRUPTED.into(),
        Failure::TooLarge => format!("The Hub sent more bytes for the {what} of {repo} than its listing gives. Run vade pull again; if this repeats, check HF_ENDPOINT."),
        Failure::NotRegular(path) => format!("{} in the pull state is not a regular file. Remove it, then run vade pull again.",
            crate::text::terminal_text(&path.to_string_lossy())),
        Failure::Curl(code, message) => format!("The {what} request for {repo} failed (curl exit {code}{}). Check the network, then run vade pull again.",
            if message.is_empty() { String::new() } else { format!(": {message}") }),
    }
}

/// Stop with the resume hint once Ctrl-C arrived.
fn stopped() -> Result<(), String> {
    if crate::runner::interrupted() { Err(INTERRUPTED.into()) } else { Ok(()) }
}

/// The whole recursive listing, page by page.
fn list(transport: &Transport, request: &Request, commit: &str) -> Result<Vec<Remote>, String> {
    let mut url = hub::tree_url(&transport.endpoint, &request.repo, commit);
    let mut files = Vec::new();
    for _ in 0..1000 {
        let (headers, body) = api(transport, request, &url, "file listing")?;
        files.extend(hub::tree_page(&body)?);
        if files.len() > hub::MAX_FILES { return Err(format!("The Hub file listing names more than {} files.", hub::MAX_FILES)); }
        match hub::next_page(&headers, &transport.endpoint)? {
            Some(next) => url = next,
            None => {
                let mut names = std::collections::BTreeSet::new();
                if files.iter().any(|file| !names.insert(file.path.clone())) { return Err("The Hub file listing names one file twice.".into()); }
                return Ok(files);
            }
        }
    }
    Err("The Hub file listing has too many pages.".into())
}

/// One checked file, remembered so a rerun need not hash it again.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Checked { size: u64, inode: u64, modified: i64, nanos: i64, changed: i64, changed_nanos: i64, expected: String, sha256: String }

fn identity(meta: &fs::Metadata) -> (u64, u64, i64, i64, i64, i64) { (meta.len(), meta.ino(), meta.mtime(), meta.mtime_nsec(), meta.ctime(), meta.ctime_nsec()) }

fn read_checked(scratch: &Path) -> BTreeMap<String, Checked> {
    parse_checked(&read_small(&scratch.join("checked"), 16 << 20, false).map(|bytes| String::from_utf8_lossy(&bytes).into_owned()).unwrap_or_default())
}

/// One row per file. A line that is not a row, such as the first line of the
/// record beside the model, gives nothing.
fn parse_checked(text: &str) -> BTreeMap<String, Checked> {
    text.lines().filter_map(|line| {
        // A row from an older pull has no change time, so its file is hashed again.
        let mut part = line.splitn(9, '\t');
        let mut next = || part.next();
        let (size, inode, modified, nanos, changed, changed_nanos, expected, sha256, path) = (next()?, next()?, next()?, next()?, next()?, next()?, next()?, next()?, next()?);
        Some((path.to_owned(), Checked { size: size.parse().ok()?, inode: inode.parse().ok()?, modified: modified.parse().ok()?,
            nanos: nanos.parse().ok()?, changed: changed.parse().ok()?, changed_nanos: changed_nanos.parse().ok()?, expected: expected.into(), sha256: sha256.into() }))
    }).collect()
}

/// The record a finished pull leaves beside these files of `folder`.
#[cfg(test)]
pub(crate) fn record_for_test(folder: &Path, names: &[&str]) {
    let rows: Vec<(String, String)> = names.iter().map(|name| (name.to_string(), crate::sha256::digest_reader(fs::File::open(folder.join(name)).unwrap()).unwrap())).collect();
    record::save(folder, "vade pull test", snapshot(folder, &rows));
}

fn checked_text(checked: &BTreeMap<String, Checked>) -> String {
    checked.iter().map(|(path, c)| format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{path}\n", c.size, c.inode, c.modified, c.nanos, c.changed, c.changed_nanos, c.expected, c.sha256)).collect()
}

fn write_checked(scratch: &Path, checked: &BTreeMap<String, Checked>) -> Result<(), String> {
    crate::text::atomic_write(&scratch.join("checked"), &checked_text(checked))
}

/// The SHA-256 of `path` if it matches `file`, else why not.
fn verify(path: &Path, file: &Remote) -> Result<String, String> {
    let meta = fs::symlink_metadata(path).map_err(|_| format!("{} is missing.", file.path))?;
    if !meta.is_file() { return Err(format!("{} is not a regular file.", file.path)); }
    if meta.len() != file.size { return Err(format!("{} has {} bytes; the Hub lists {}.", file.path, meta.len(), file.size)); }
    let (sha256, git) = fetch::digests(path, file.size, file.sha256.is_none()).map_err(|_| format!("Cannot read {} as a regular file of {} bytes.", file.path, file.size))?;
    match (&file.sha256, git) {
        (Some(expected), _) if *expected != sha256 => Err(format!("{} has SHA-256 {sha256}; the Hub lists {expected}.", file.path)),
        (None, Some(git)) if git != file.git => Err(format!("{} has Git ID {git}; the Hub lists {}.", file.path, file.git)),
        _ => Ok(sha256),
    }
}

/// Download and check every chosen file. Return the hashes and resume count.
fn download(transport: &Transport, request: &Request, commit: &str, folder: &Path, scratch: &Path, chosen: &[Remote], live: bool)
    -> Result<(BTreeMap<String, String>, usize), String> {
    let mut checked = read_checked(scratch);
    let mut hashes = BTreeMap::new();
    let mut resumed = 0;
    let total: u64 = chosen.iter().map(|file| file.size).sum();
    let mut done = 0;
    let mut marked = false;
    for (index, file) in chosen.iter().enumerate() {
        stopped()?;
        let path = folder.join(&file.path);
        no_links(folder, &path)?;
        let expected = file.sha256.clone().unwrap_or_else(|| file.git.clone());
        let label = format!("[{}/{}] {}", index + 1, chosen.len(), file.path);
        let mut progress = Progress::new(live, total, done);
        // A file this pull already checked, unchanged since, is not hashed again.
        let known = fs::symlink_metadata(&path).ok().filter(fs::Metadata::is_file).and_then(|meta| {
            let c = checked.get(&file.path)?;
            ((c.size, c.inode, c.modified, c.nanos, c.changed, c.changed_nanos) == identity(&meta) && c.expected == expected && c.size == file.size).then(|| c.sha256.clone())
        });
        let present = known.map(Ok).unwrap_or_else(|| verify(&path, file));
        // A check that Ctrl-C cut short is no reason to download the file.
        stopped()?;
        let sha256 = match present {
            Ok(sha256) => { progress.show(&label, file.size, file.size, 0, Duration::ZERO, Some("present")); sha256 }
            Err(_) => {
                if fs::symlink_metadata(&path).is_ok_and(|meta| meta.is_file()) {
                    if !marked { crate::text::atomic_write(&scratch.join(UPDATING), &format!("{commit}\n"))?; marked = true; }
                    fs::remove_file(&path).map_err(|error| crate::text::file_error("Cannot remove an old model file", &path, &error))?;
                }
                let part = scratch.join(format!("{}.part", &crate::sha256::sha(file.path.as_bytes())[..24]));
                let url = hub::file_url(&transport.endpoint, &request.repo, commit, &file.path);
                let began = Instant::now();
                let (sha256, continued) = pull_file(transport, request, file, &url, &part, &label, &mut progress, began)?;
                resumed += usize::from(continued);
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).map_err(|error| crate::text::file_error("Cannot create a model subfolder", parent, &error))?;
                }
                no_links(folder, &path)?;
                if !marked { crate::text::atomic_write(&scratch.join(UPDATING), &format!("{commit}\n"))?; marked = true; }
                fs::rename(&part, &path).map_err(|error| crate::text::file_error("Cannot move a downloaded file into place", &path, &error))?;
                progress.show(&label, file.size, file.size, progress.received, began.elapsed(), Some("done"));
                sha256
            }
        };
        clear_bad(scratch, file)?;
        if let Ok(meta) = fs::symlink_metadata(&path) {
            let (size, inode, modified, nanos, changed, changed_nanos) = identity(&meta);
            checked.insert(file.path.clone(), Checked { size, inode, modified, nanos, changed, changed_nanos, expected, sha256: sha256.clone() });
            write_checked(scratch, &checked)?;
        }
        hashes.insert(file.path.clone(), sha256);
        done += file.size;
    }
    Ok((hashes, resumed))
}

/// Remove this file's failed copies, including names from older pulls.
fn clear_bad(scratch: &Path, file: &Remote) -> Result<(), String> {
    let base = scratch.join("bad").join(&file.path);
    no_links(scratch, &base)?;
    let parent = base.parent().ok_or("Cannot name a bad file folder.")?;
    let entries = match fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(crate::text::file_error("Cannot read the bad file folder", parent, &error)),
    };
    let prefix = format!("{}.bad-", base.file_name().ok_or("Cannot name a bad file.")?.to_string_lossy());
    for entry in entries {
        let entry = entry.map_err(|error| crate::text::file_error("Cannot read the bad file folder", parent, &error))?;
        let name = entry.file_name();
        let old = name.to_str().and_then(|name| name.strip_prefix(&prefix)).is_some_and(|time| !time.is_empty() && time.bytes().all(|byte| byte.is_ascii_digit()));
        if entry.path() == base || old {
            fs::remove_file(entry.path()).map_err(|error| crate::text::file_error("Cannot remove a bad copy", &entry.path(), &error))?;
        }
    }
    Ok(())
}

/// Keep only the latest failed download for this file path.
fn quarantine(path: &Path, scratch: &Path, file: &Remote) -> Result<(), String> {
    let base = scratch.join("bad").join(&file.path);
    no_links(scratch, &base)?;
    if let Some(parent) = base.parent() {
        fs::create_dir_all(parent).map_err(|error| crate::text::file_error("Cannot create the bad file folder", parent, &error))?;
    }
    clear_bad(scratch, file)?;
    fs::rename(path, &base).map_err(|error| crate::text::file_error("Cannot keep the bad file", path, &error))
}

/// Check each completed response. Retry a bad copy once from zero.
fn pull_file(transport: &Transport, request: &Request, file: &Remote, url: &str, part: &Path, label: &str, progress: &mut Progress, began: Instant)
    -> Result<(String, bool), String> {
    let mut received = 0;
    let mut retried = false;
    loop {
        let fetched = transport.download(url, part, file.size, &mut |event| match event {
            fetch::Event::Progress { have, received: now } => {
                progress.received = received + now;
                progress.show(label, have, file.size, progress.received, began.elapsed(), None);
            }
            fetch::Event::Restart => progress.notice(&format!("{}: The server did not accept Range. Pull starts the file again.", file.path)),
        });
        let complete = fetched.is_ok();
        let continued = match fetched {
            Ok(continued) => continued,
            Err(Failure::TooLarge) if fs::symlink_metadata(part).is_ok_and(|meta| meta.is_file()) => false,
            Err(failure) => return Err(refusal(failure, request, &format!("file {}", file.path), transport.token.is_some())),
        };
        let verified = verify(part, file);
        // Keep the partial file when Ctrl-C stops the check.
        stopped()?;
        match verified {
            Ok(sha256) if complete => return Ok((sha256, continued)),
            result => {
                let error = result.err().unwrap_or_else(|| format!("{} exceeded the Hub size limit.", file.path));
                quarantine(part, &transport.scratch, file)?;
                if retried { return Err(format!("{error} Pull kept the latest bad copy in {STATE}/bad. Check the source, then run vade pull again.")); }
                progress.notice(&format!("{}: The size or hash check failed. Pull kept the bad copy in {STATE}/bad and will try again.", file.path));
            }
        }
        received = progress.received;
        retried = true;
    }
}

/// Refuse a symlink anywhere between the model folder and `path`, so a link
/// cannot send a download outside the folder.
fn no_links(folder: &Path, path: &Path) -> Result<(), String> {
    let relative = path.strip_prefix(folder).map_err(|_| "A model file escapes the model folder.")?;
    let mut current = folder.to_path_buf();
    for part in relative.components() {
        current.push(part);
        if fs::symlink_metadata(&current).is_ok_and(|meta| meta.file_type().is_symlink()) {
            return Err(format!("{} in the model folder is a symlink. Remove it, then run vade pull again.", relative.display()));
        }
    }
    Ok(())
}

/// Remove partial files and curl scratch; keep the lock and the checked list.
fn clean(scratch: &Path) {
    for entry in fs::read_dir(scratch).into_iter().flatten().flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().ends_with(".part") { let _ = fs::remove_file(entry.path()); }
    }
}

/// Replace `path` with `text` through a temporary file in the same folder.
fn publish(path: &Path, text: &str) -> Result<(), String> {
    let temporary = path.with_file_name(format!(".vade-write-{}", crate::text::random_hex(6)?));
    let result = fs::write(&temporary, text).and_then(|()| fs::File::open(&temporary)?.sync_all()).and_then(|()| fs::rename(&temporary, path));
    if result.is_err() { let _ = fs::remove_file(&temporary); }
    result.map_err(|error| crate::text::file_error("Cannot write", path, &error))
}

/// One file in the records a pull writes: the manifest and the checksum list.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Record { path: String, size: u64, sha256: String, how: &'static str }

/// Files that an earlier pull of the same repository and commit recorded and
/// that this selection leaves out, while each is still in the folder at its
/// recorded size. The new records keep them, so a pull of a few files does
/// not drop the others from the checksum list.
fn kept(folder: &Path, request: &Request, commit: &str, chosen: &[Remote]) -> Vec<Record> {
    let Some(manifest) = models::manifest(folder) else { return Vec::new() };
    if manifest["repo"].as_str() != Some(request.repo.as_str()) || manifest["commit"].as_str() != Some(commit) { return Vec::new(); }
    let names: std::collections::BTreeSet<&str> = chosen.iter().map(|file| file.path.as_str()).collect();
    let rows: BTreeMap<String, Record> = manifest["files"].as_array().into_iter().flatten().filter_map(|row| {
        let path = row["path"].as_str().filter(|path| hub::safe_path(path) && !hub::hidden(path) && !names.contains(path))?;
        let size = row["bytes"].as_i64().and_then(|size| u64::try_from(size).ok())?;
        let sha256 = row["sha256"].as_str().filter(|sha| crate::text::is_lower_hex(sha, 64))?;
        let how = match row["checked"].as_str()? { "sha256" => "sha256", "git blob id" => "git blob id", _ => return None };
        let file = folder.join(path);
        (no_links(folder, &file).is_ok() && fs::symlink_metadata(&file).is_ok_and(|meta| meta.is_file() && meta.len() == size)).then_some(())?;
        Some((path.to_owned(), Record { path: path.into(), size, sha256: sha256.into(), how }))
    }).collect();
    rows.into_values().collect()
}

/// `<folder>.sha256` beside the folder.
fn list_path(folder: &Path) -> Option<PathBuf> {
    let (name, parent) = (folder.file_name()?, folder.parent()?);
    Some(parent.join(format!("{}.sha256", name.to_string_lossy())))
}

/// Refuse, before any file changes, a pull whose records would leave out a
/// file that the checksum list beside the folder names and that is still in
/// the folder: launch reads that list, and an engine needs every shard in it.
fn covered(folder: &Path, chosen: &[Remote], kept: &[Record], selective: bool) -> Result<(), String> {
    let names: std::collections::BTreeSet<&str> = chosen.iter().map(|file| file.path.as_str()).chain(kept.iter().map(|record| record.path.as_str())).collect();
    // A selection over a folder that holds other model files, with no record
    // of them, would publish a checksum list without them.
    if selective && let Some(path) = present(folder)?.into_iter().find(|path| !names.contains(path.as_str())) {
        return Err(format!("This folder holds {}, which this pull would leave out of its records, and launch checks the folder against them. \
            Pull without --include, or use --to with a new folder.", crate::text::terminal_text(&path)));
    }
    let Some(list) = list_path(folder) else { return Ok(()) };
    let text = match read_small(&list, 16 << 20, false) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(_) => return Err(format!("Cannot read the checksum list {}. It must be a regular file of at most 16 MiB.", crate::text::terminal_text(&list.to_string_lossy()))),
    };
    for line in text.lines() {
        let Some((_, name)) = line.split_once(char::is_whitespace) else { continue };
        let name = name.trim_start().trim_start_matches('*');
        let name = name.strip_prefix("./").unwrap_or(name);
        if hub::safe_path(name) && !names.contains(name) && fs::symlink_metadata(folder.join(name)).is_ok_and(|meta| meta.is_file()) {
            return Err(format!("The checksum list beside this folder names {}, which this pull leaves out, and launch checks the folder against that list. \
                Pull without --include, or use --to with a new folder.", crate::text::terminal_text(name)));
        }
    }
    Ok(())
}

/// The regular files under `folder` that a default pull takes, as relative
/// paths, without links or paths with a dot part.
fn present(folder: &Path) -> Result<Vec<String>, String> {
    let (mut found, mut pending, mut seen) = (Vec::new(), vec![(folder.to_path_buf(), String::new())], 0usize);
    while let Some((dir, prefix)) = pending.pop() {
        for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
            seen += 1;
            if seen > hub::MAX_FILES { return Err("The model folder holds too many files to check. Use --to with a new folder.".into()); }
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') { continue; }
            let path = if prefix.is_empty() { name } else { format!("{prefix}/{name}") };
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => pending.push((entry.path(), path)),
                Ok(kind) if kind.is_file() && hub::wanted(&path) => found.push(path),
                _ => {}
            }
        }
    }
    found.sort();
    Ok(found)
}

fn write_manifest(request: &Request, setup: &Setup, folder: &Path, commit: &str, records: &[Record], card: bool,
    support: &models::Support) -> Result<(), String> {
    // Only a model card this pull just checked; the front matter is at the top.
    let license = card.then(|| read_head(&folder.join("README.md"), 64 << 10, false).ok()).flatten()
        .and_then(|bytes| hub::license(&String::from_utf8_lossy(&bytes)));
    let rows: Vec<String> = records.iter().map(|record| json::packed(&json::object(vec![
        ("path", record.path.as_str().into()), ("bytes", Value::Int(record.size as i64)),
        ("sha256", record.sha256.as_str().into()), ("checked", record.how.into()),
    ]))).collect();
    let field = |name: &str, value: Value| format!("  \"{name}\": {},\n", json::packed(&value));
    let mut text = String::from("{\n");
    text += &field("schema", "vade.model.v1".into());
    text += &field("repo", request.repo.as_str().into());
    text += &field("revision", request.revision.as_str().into());
    text += &field("commit", commit.into());
    text += &field("source", setup.endpoint.as_str().into());
    text += &field("pulled_at", utc(std::time::SystemTime::now()).into());
    text += &field("license", license.map_or(Value::Null, Value::from));
    text += &field("bytes", Value::Int(records.iter().map(|record| record.size as i64).sum()));
    text += &field("support", support.json());
    text += &format!("  \"files\": [\n    {}\n  ]\n}}\n", rows.join(",\n    "));
    publish(&folder.join(MANIFEST), &text)
}

/// `<folder>.sha256` beside the folder, in sha256sum form with `./` names: the
/// checksum list the FP4-Local routes of `vade launch`, `vade up` and
/// `vade install` check before they serve. `records` are sorted by path.
fn checksums(folder: &Path, records: &[Record]) -> Result<(), String> {
    let Some(list) = list_path(folder) else { return Ok(()) };
    publish(&list, &records.iter().map(|record| format!("{}  ./{}\n", record.sha256, record.path)).collect::<String>())
}

/// Redraw terminal progress in place. Limit log updates, but always finish a file.
struct Progress { live: bool, total: u64, done: u64, last: Option<Instant>, received: u64 }
impl Progress {
    fn new(live: bool, total: u64, done: u64) -> Self { Self { live, total, done, last: None, received: 0 } }
    fn line(&mut self, label: &str, have: u64, size: u64, received: u64, elapsed: Duration, finish: Option<&str>, now: Instant) -> Option<String> {
        let gap = if self.live { Duration::from_millis(250) } else { Duration::from_secs(5) };
        if finish.is_none() && (self.last.is_some_and(|last| now.duration_since(last) < gap) || elapsed < gap) { return None; }
        self.last = Some(now);
        let status = if finish == Some("present") { "present".into() }
            else { format!("{:.1} MB/s{}", received as f64 / elapsed.as_secs_f64().max(0.001) / 1e6, if finish.is_some() { " done" } else { "" }) };
        Some(format!("{label} {:.3} / {:.3} GiB {status} total {:.3} / {:.3} GiB", gib(have), gib(size), gib(self.done + have), gib(self.total)))
    }
    fn render(&self, line: &str, finish: bool) -> String {
        format!("{}{}{}", if self.live { "\r\x1b[2K" } else { "" }, crate::text::terminal_text(line), if !self.live || finish { "\n" } else { "" })
    }
    fn show(&mut self, label: &str, have: u64, size: u64, received: u64, elapsed: Duration, finish: Option<&str>) {
        if let Some(line) = self.line(label, have, size, received, elapsed, finish, Instant::now()) {
            crate::output::stderr(format_args!("{}", self.render(&line, finish.is_some())));
        }
    }
    fn notice(&self, line: &str) { crate::output::stderr(format_args!("{}", self.render(line, true))); }
}

fn gib(value: u64) -> f64 { value as f64 / (1u64 << 30) as f64 }

/// Decimal units, as the Hub shows them.
fn bytes(value: u64) -> String {
    match value {
        0..1_000 => format!("{value} B"),
        1_000..1_000_000 => format!("{:.1} kB", value as f64 / 1e3),
        1_000_000..1_000_000_000 => format!("{:.1} MB", value as f64 / 1e6),
        _ => format!("{:.2} GB", value as f64 / 1e9),
    }
}

/// `path` with the home folder shown as `~`.
fn display(path: &Path, home: &Path) -> String {
    let text = match path.strip_prefix(home) { Ok(rest) => format!("~/{}", rest.display()), Err(_) => path.display().to_string() };
    crate::text::terminal_text(&text).into_owned()
}

/// UTC time as `YYYY-MM-DDTHH:MM:SSZ`.
fn utc(time: std::time::SystemTime) -> String {
    let seconds = time.duration_since(std::time::UNIX_EPOCH).map_or(0, |since| since.as_secs()) as i64;
    let (days, rest) = (seconds.div_euclid(86_400), seconds.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (proleptic Gregorian calendar).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rest / 3600, rest % 3600 / 60, rest % 60)
}

#[cfg(test)]
mod tests;
