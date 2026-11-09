//! Complete recurrent-state comparisons, preserving the original capture format.
use super::json::{self, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

const BOUND: f64 = 1e-3;
const HEAD_ELEMENTS: usize = 128 * 128;
const HEAD_BYTES: usize = HEAD_ELEMENTS * 4;
const LAYERS: [usize; 36] = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33,
    34, 36, 37, 38, 40, 41, 42, 44, 45, 46,
];
type Ranks = BTreeMap<usize, Value>;

const USAGE: &str = "usage: vade test gdn-state [-h] [--identity]\n                           [--legacy-prefill-group LEGACY_PREFILL_GROUP]\n                           fp32 candidate [split]";
const DESCRIPTION: &str = "Compare complete Flash GDN dumps: FP32 first, split or FP32 control second.\nWith a third directory: FP32 reference, perturbed FP32, split. Each layer's\nmedian/max head-normalized errors and relative Frobenius must be <= 2x its\nperturbed-FP32 floor. Prints raw head rows and a per-layer table. Zero floors\npass only with zero split error; undefined infinite floors cannot pass. PASS\nrequires max|a-b| / max|b| <= 1e-3 for each of 36 layers x 48 heads.\nElementwise relative diagnostics use a floor of 1e-3 * max|b| for that head.\nAn all-zero reference head passes only if the candidate is also all zero. FP32\ncontrols additionally require max_abs_error == 0. Standard library only.\n--identity compares raw state bytes (including signed zero), for two\nunperturbed arms using the same arithmetic. --legacy-prefill-group supplies\nthe externally verified group for old manifests (the historical split capture used 512).\nPrompt bytes and world must match; checkpoint identity must be verified from\nthe capture receipts because v1 manifests do not record it.";

struct Args {
    fp32: PathBuf,
    candidate: PathBuf,
    split: Option<PathBuf>,
    identity: bool,
    legacy: Option<Value>,
}

/// `vade test gdn-state`: compare two recurrent-state captures.
pub fn run(args: &[String]) -> Result<i32, String> {
    let args = match parse_args(args) {
        Ok(Some(args)) => args,
        Ok(None) => {
            crate::public_output(&format!(
                "{USAGE}\n\n{DESCRIPTION}\n\npositional arguments:\n  fp32                  session directory from FLASH_BF16_MMA unset\n  candidate             split session directory, or FP32 control\n  split                 enables floor mode: reference, perturbed FP32, split\n\noptions:\n  -h, --help            show this help message and exit\n  --identity            compare exact state bytes for two unperturbed arms\n  --legacy-prefill-group LEGACY_PREFILL_GROUP\n                        verified prefill group for manifests missing that\n                        field\n"
            ));
            return Ok(0);
        }
        Err(error) => {
            crate::output::stderr_line(format_args!("{USAGE}\nvade test gdn-state: error: {error}"));
            return Ok(2);
        }
    };
    match compare(&args) {
        Ok(code) => Ok(code),
        Err(error) => {
            crate::output::stderr_line(format_args!("FAIL invalid dump: {error}"));
            Ok(1)
        }
    }
}

fn parse_args(args: &[String]) -> Result<Option<Args>, String> {
    let mut dirs = Vec::new();
    let mut identity = false;
    let mut legacy = None;
    let mut unknown = Vec::new();
    let mut index = 0;
    let mut positional = false;
    while index < args.len() {
        let arg = &args[index];
        if positional || !arg.starts_with('-') || arg == "-" || negative_number(arg) {
            dirs.push(PathBuf::from(arg));
        } else if arg == "--" {
            positional = true;
        } else if arg == "-h" || (arg.len() >= 3 && "--help".starts_with(arg)) {
            return Ok(None);
        } else if arg
            .split('=')
            .next()
            .is_some_and(|s| s.len() >= 3 && "--identity".starts_with(s))
        {
            if let Some((_, explicit)) = arg.split_once('=') {
                return Err(format!(
                    "argument --identity: ignored explicit argument {}",
                    repr(explicit)
                ));
            }
            identity = true;
        } else if arg
            .split('=')
            .next()
            .is_some_and(|s| s.len() >= 3 && "--legacy-prefill-group".starts_with(s))
        {
            let value = if let Some((_, value)) = arg.split_once('=') {
                value.to_owned()
            } else {
                index += 1;
                args.get(index)
                    .filter(|s| !s.starts_with('-') || negative_number(s))
                    .cloned()
                    .ok_or_else(|| {
                        "argument --legacy-prefill-group: expected one argument".to_owned()
                    })?
            };
            legacy = Some(python_int(&value).ok_or_else(|| {
                format!(
                    "argument --legacy-prefill-group: invalid int value: {}",
                    repr(&value)
                )
            })?);
        } else {
            unknown.push(arg.clone());
        }
        index += 1;
    }
    if dirs.len() < 2 {
        return Err(format!(
            "the following arguments are required: {}",
            if dirs.is_empty() {
                "fp32, candidate"
            } else {
                "candidate"
            }
        ));
    }
    if dirs.len() > 3 {
        unknown.extend(dirs[3..].iter().map(|s| s.display().to_string()));
    }
    if !unknown.is_empty() {
        return Err(format!("unrecognized arguments: {}", unknown.join(" ")));
    }
    if legacy.as_ref().is_some_and(|v| !positive_multiple_64(v)) {
        return Err("--legacy-prefill-group must be a positive multiple of 64".into());
    }
    if identity && dirs.len() == 3 {
        return Err("--identity takes exactly two directories".into());
    }
    Ok(Some(Args {
        fp32: dirs[0].clone(),
        candidate: dirs[1].clone(),
        split: dirs.get(2).cloned(),
        identity,
        legacy,
    }))
}

fn negative_number(value: &str) -> bool {
    let Some(value) = value.strip_prefix('-') else {
        return false;
    };
    if value.is_empty() {
        return false;
    }
    if let Some((whole, fraction)) = value.split_once('.') {
        whole.bytes().all(|c| c.is_ascii_digit())
            && !fraction.is_empty()
            && fraction.bytes().all(|c| c.is_ascii_digit())
    } else {
        value.bytes().all(|c| c.is_ascii_digit())
    }
}
fn python_int(value: &str) -> Option<Value> {
    let canonical = json::integer_input(value).ok()?;
    Some(
        canonical
            .parse::<i64>()
            .map_or_else(|_| Value::BigInt(canonical.clone()), Value::Int),
    )
}

fn integer_text(value: &Value) -> Option<String> {
    match value {
        Value::Int(v) => Some(v.to_string()),
        Value::BigInt(v) => Some(v.clone()),
        _ => None,
    }
}
fn nonnegative_integer(value: &Value) -> bool {
    match value {
        Value::Int(v) => *v >= 0,
        Value::BigInt(v) => !v.starts_with('-'),
        _ => false,
    }
}
fn positive_integer(value: &Value) -> bool {
    match value {
        Value::Int(v) => *v > 0,
        Value::BigInt(v) => !v.starts_with('-') && v != "0",
        _ => false,
    }
}
fn positive_multiple_64(value: &Value) -> bool {
    positive_integer(value)
        && integer_text(value)
            .unwrap()
            .bytes()
            .fold(0_u32, |n, digit| (n * 10 + u32::from(digit - b'0')) % 64)
            == 0
}

fn repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::from(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch == quote => {
                out.push('\\');
                out.push(ch);
            }
            ch => out.push(ch),
        }
    }
    out.push(quote);
    out
}

fn io_error(path: &Path, error: std::io::Error) -> String {
    if let Some(errno) = error.raw_os_error() {
        let text = error.to_string();
        let message = text.split(" (os error ").next().unwrap_or(&text);
        format!(
            "[Errno {errno}] {message}: {}",
            repr(&path.display().to_string())
        )
    } else {
        error.to_string()
    }
}
fn open(path: &Path) -> Result<File, String> {
    File::open(path).map_err(|e| io_error(path, e))
}
fn bytes(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|e| io_error(path, e))
}
fn size(path: &Path) -> Result<u64, String> {
    fs::metadata(path)
        .map(|m| m.len())
        .map_err(|e| io_error(path, e))
}
fn get<'a>(doc: &'a Value, key: &str) -> Option<&'a Value> {
    if let Value::Object(fields) = doc {
        fields
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    } else {
        None
    }
}
fn int(doc: &Value, key: &str) -> usize {
    match get(doc, key) {
        Some(Value::Int(v)) => *v as usize,
        Some(Value::BigInt(v)) => v.parse().unwrap_or(usize::MAX),
        _ => unreachable!(),
    }
}

fn boolean(doc: &Value, key: &str) -> bool {
    matches!(get(doc, key), Some(Value::Bool(true)))
}
fn is_string(doc: &Value, key: &str, expected: &str) -> bool {
    matches!(get(doc, key), Some(Value::Str(s)) if s == expected)
}
fn equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, value)| b.iter().any(|(other, v)| key == other && equal(value, v)))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| equal(a, b))
        }
        (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => {
            integer_float_equal(&a.to_string(), *b)
        }
        (Value::BigInt(a), Value::Float(b)) | (Value::Float(b), Value::BigInt(a)) => {
            integer_float_equal(a, *b)
        }
        (Value::Float(a), Value::Float(b)) => a == b || (a.is_nan() && b.is_nan()),
        (Value::Bool(a), Value::Int(b)) | (Value::Int(b), Value::Bool(a)) => i64::from(*a) == *b,
        (Value::Bool(a), Value::Float(b)) | (Value::Float(b), Value::Bool(a)) => {
            f64::from(u8::from(*a)) == *b
        }
        _ => a == b,
    }
}
fn integer_float_equal(integer: &str, float: f64) -> bool {
    float.is_finite()
        && float.fract() == 0.0
        && integer
            == if float == 0.0 {
                "0".into()
            } else {
                format!("{float:.0}")
            }
}
fn optional_equal(a: Option<&Value>, b: Option<&Value>) -> bool {
    match (a, b) {
        (None | Some(Value::Null), None | Some(Value::Null)) => true,
        (Some(a), Some(b)) => equal(a, b),
        _ => false,
    }
}
fn perturbation(doc: &Value) -> Result<Option<&Value>, String> {
    let Some(probe) = get(doc, "layer0_perturbation").filter(|v| !matches!(v, Value::Null)) else {
        return Ok(None);
    };
    if !matches!(probe, Value::Object(fields) if fields.len() == 4 && fields.iter().all(|(k, _)| ["ulp", "every", "scope", "index"].contains(&k.as_str())))
        || !matches!(get(probe, "ulp"), Some(Value::Int(-1 | 1)))
        || !matches!(get(probe, "every"), Some(v) if positive_integer(v))
        || !is_string(probe, "scope", "prefill")
        || !is_string(probe, "index", "absolute_token,residual_channel")
    {
        return Err("invalid layer-0 perturbation metadata".into());
    }
    if boolean(doc, "bf16_mma") {
        return Err("layer-0 perturbation requires the FP32 path".into());
    }
    Ok(Some(probe))
}
fn entries(directory: &Path) -> Result<Vec<PathBuf>, String> {
    // Path.glob returns no files for a missing or non-directory path.
    match fs::read_dir(directory) {
        Ok(entries) => entries
            .map(|entry| entry.map(|e| e.path()).map_err(|e| io_error(directory, e)))
            .collect(),
        Err(_) => Ok(Vec::new()),
    }
}
fn glob_state(name: &str) -> bool {
    name.strip_prefix("layer-")
        .and_then(|s| s.strip_suffix(".f32"))
        .is_some_and(|s| s.contains("-rank-"))
}
fn load_dump(directory: &Path, legacy: Option<&Value>) -> Result<(Ranks, Vec<u8>), String> {
    let paths = entries(directory)?;
    let mut parsed_manifests = BTreeMap::new();
    for path in &paths {
        let name = path.file_name().unwrap().to_string_lossy();
        if !name.starts_with("rank-") || !name.ends_with(".json") {
            continue;
        }
        let raw = fs::read_to_string(path).map_err(|e| io_error(path, e))?;
        let mut doc = json::parse(&raw)?;
        if !matches!(doc, Value::Object(_)) {
            return Err(format!("{name}: manifest must be an object"));
        }
        for key in [
            "position",
            "rank",
            "world",
            "total_heads",
            "head_start",
            "head_end",
            "key_dim",
            "value_dim",
        ] {
            if !matches!(get(&doc, key), Some(v) if nonnegative_integer(v)) {
                return Err(format!("{name}: invalid {key}"));
            }
        }
        let rank = integer_text(get(&doc, "rank").unwrap()).unwrap();
        if name
            != format!(
                "rank-{}.json",
                integer_text(get(&doc, "rank").unwrap()).unwrap()
            )
            || parsed_manifests.contains_key(&rank)
        {
            return Err("duplicate or noncanonical rank manifest".into());
        }
        if !is_string(&doc, "format", "vade-gdn-state-v1")
            || !is_string(&doc, "dtype", "f32-le")
            || int(&doc, "total_heads") != 48
            || int(&doc, "key_dim") != 128
            || int(&doc, "value_dim") != 128
            || !matches!(get(&doc, "layers"), Some(Value::Array(v)) if v.len() == LAYERS.len() && v.iter().zip(LAYERS).all(|(v,l)| matches!(v, Value::Int(n) if *n == l as i64)))
            || !matches!(get(&doc, "bf16_mma"), Some(Value::Bool(_)))
            || !matches!(get(&doc, "expert_precision"), Some(Value::Str(_)))
        {
            return Err(format!(
                "rank {rank}: wrong format, dimensions, layers or BF16 MMA path"
            ));
        }
        perturbation(&doc)?;
        if get(&doc, "prefill_group").is_none() {
            if let (Some(group), Value::Object(fields)) = (legacy, &mut doc) {
                fields.push(("prefill_group".into(), group.clone()));
            }
        }
        if get(&doc, "prefill_group").is_some()
            && !matches!(get(&doc, "prefill_group"), Some(v) if positive_multiple_64(v))
        {
            return Err(format!("rank {rank}: invalid prefill group"));
        }
        parsed_manifests.insert(rank, doc);
    }
    if parsed_manifests.is_empty() {
        return Err("no completed rank manifests; pass a session-NNNNN directory".into());
    }
    let first = parsed_manifests
        .get("0")
        .ok_or("dump must include every rank from 0 through world-1")?;
    let world = int(first, "world");
    if !(1..=128).contains(&world)
        || parsed_manifests.len() != world
        || !(0..world).all(|r| parsed_manifests.contains_key(&r.to_string()))
    {
        return Err("dump must include every rank from 0 through world-1".into());
    }
    // Only convert ranks after the original 1..128 world/rank-set rule proves
    // they fit. Keeping decimal keys before this check preserves huge invalid
    // rank diagnostics and prevents distinct big integers from colliding.
    let manifests: Ranks = parsed_manifests
        .into_iter()
        .map(|(rank, doc)| (rank.parse().unwrap(), doc))
        .collect();
    let first = &manifests[&0];
    let mut expected = BTreeSet::new();
    let mut covered = 0;
    let mut prompt = None;
    for (&rank, doc) in &manifests {
        for key in ["position", "world", "expert_precision", "bf16_mma"] {
            if !optional_equal(get(doc, key), get(first, key)) {
                return Err(format!("rank {rank}: inconsistent {key}"));
            }
        }
        if !optional_equal(perturbation(doc)?, perturbation(first)?)
            || !optional_equal(get(doc, "prefill_group"), get(first, "prefill_group"))
        {
            return Err(format!("rank {rank}: inconsistent probe or prefill group"));
        }
        let end = int(doc, "head_end");
        if int(doc, "head_start") != covered || !(covered..=48).contains(&end) {
            return Err(format!("rank {rank}: head ranges overlap or leave a gap"));
        }
        covered = end;
        let path = directory.join(format!("prompt-rank-{rank}.u32"));
        if size(&path)? as u128 != ((int(doc, "position") as u128 + 1) * 4) {
            return Err(format!("rank {rank}: truncated or oversized prompt prefix"));
        }
        let ids = bytes(&path)?;
        if prompt.as_ref().is_some_and(|previous| previous != &ids) {
            return Err(format!("rank {rank}: prompt prefix differs between ranks"));
        }
        prompt = Some(ids);
        for layer in LAYERS {
            let name = format!("layer-{layer:03}-rank-{rank}.f32");
            if size(&directory.join(&name))? != ((end - int(doc, "head_start")) * HEAD_BYTES) as u64
            {
                return Err(format!(
                    "layer {layer} rank {rank}: truncated or oversized recurrent state"
                ));
            }
            expected.insert(name);
        }
    }
    let actual: BTreeSet<_> = paths
        .iter()
        .filter_map(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| glob_state(n))
        .collect();
    if covered != 48 || actual != expected {
        return Err("missing or extra recurrent heads/layer files".into());
    }
    Ok((manifests, prompt.unwrap()))
}
fn matching_dump(
    reference: &Ranks,
    ids: &[u8],
    directory: &Path,
    legacy: Option<&Value>,
) -> Result<Ranks, String> {
    let (candidate, candidate_ids) = load_dump(directory, legacy)?;
    if ids != candidate_ids || reference.keys().ne(candidate.keys()) {
        return Err("input prefixes or rank sets differ between dumps".into());
    }
    for (&rank, doc) in reference {
        let common = |doc: &Value| match doc {
            Value::Object(fields) => Value::Object(
                fields
                    .iter()
                    .filter(|(k, _)| !["bf16_mma", "layer0_perturbation"].contains(&k.as_str()))
                    .cloned()
                    .collect(),
            ),
            _ => unreachable!(),
        };
        if !equal(&common(doc), &common(&candidate[&rank])) {
            return Err(format!(
                "rank {rank}: dump metadata differs between paths; legacy manifests need --legacy-prefill-group with the verified capture group"
            ));
        }
    }
    Ok(candidate)
}

#[derive(Default)]
struct Stats {
    elements: usize,
    max_ref_abs: f64,
    max_head_normalized_error: f64,
    max_abs_error: f64,
    max_relative_error: f64,
    sum_relative_error: f64,
    above_abs: usize,
    above_rel: usize,
}
impl Stats {
    fn add(&mut self, other: &Self) {
        self.elements += other.elements;
        self.max_ref_abs = self.max_ref_abs.max(other.max_ref_abs);
        self.max_head_normalized_error = self
            .max_head_normalized_error
            .max(other.max_head_normalized_error);
        self.max_abs_error = self.max_abs_error.max(other.max_abs_error);
        self.max_relative_error = self.max_relative_error.max(other.max_relative_error);
        self.sum_relative_error += other.sum_relative_error;
        self.above_abs += other.above_abs;
        self.above_rel += other.above_rel;
    }
}
fn scientific(value: f64) -> String {
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-inf"
        } else {
            "inf"
        }
        .into();
    }
    if value.is_nan() {
        return "nan".into();
    }
    let raw = format!("{value:.9e}");
    let (mantissa, exponent) = raw.split_once('e').unwrap();
    format!("{mantissa}e{:+03}", exponent.parse::<i32>().unwrap())
}
fn show(label: &str, stats: &Stats) {
    crate::output::report(format_args!(
        "{label} elements={} state_bytes={} max_ref_abs={} max_head_normalized_error={} max_abs_error={} max_relative_error={} mean_relative_error={} frac_abs_gt_1e-3={} frac_rel_gt_1e-3={}",
        stats.elements,
        stats.elements * 4,
        scientific(stats.max_ref_abs),
        scientific(stats.max_head_normalized_error),
        scientific(stats.max_abs_error),
        scientific(stats.max_relative_error),
        scientific(stats.sum_relative_error / stats.elements as f64),
        scientific(stats.above_abs as f64 / stats.elements as f64),
        scientific(stats.above_rel as f64 / stats.elements as f64)
    ), true);
}
fn normalized(error: f64, scale: f64) -> f64 {
    if scale != 0.0 {
        error / scale
    } else if error != 0.0 {
        f64::INFINITY
    } else {
        0.0
    }
}
fn floor_ratio(error: f64, floor: f64) -> f64 {
    if !floor.is_finite() || !error.is_finite() {
        f64::INFINITY
    } else {
        normalized(error, floor)
    }
}
fn read_block(file: &mut File, path: &Path) -> Result<Vec<u8>, String> {
    let mut block = vec![0; HEAD_BYTES];
    let mut read = 0;
    while read < block.len() {
        match file.read(&mut block[read..]) {
            Ok(0) => break,
            Ok(n) => read += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(io_error(path, e)),
        }
    }
    block.truncate(read);
    Ok(block)
}
fn float_values(block: &[u8]) -> impl Iterator<Item = f64> + '_ {
    block
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b) as f64)
}
fn trailing(file: &mut File, path: &Path) -> Result<bool, String> {
    let mut byte = [0];
    file.read(&mut byte)
        .map(|n| n != 0)
        .map_err(|e| io_error(path, e))
}
fn identical_files(reference: &Path, candidate: &Path) -> Result<(bool, usize), String> {
    let mut ref_file = open(reference)?;
    let mut cand_file = open(candidate)?;
    let mut offset = 0;
    loop {
        let b = read_block(&mut ref_file, reference)?;
        let a = read_block(&mut cand_file, candidate)?;
        if b != a {
            return Ok((
                false,
                offset
                    + b.iter()
                        .zip(&a)
                        .position(|(b, a)| b != a)
                        .unwrap_or(b.len().min(a.len())),
            ));
        }
        if b.is_empty() {
            return Ok((true, offset));
        }
        offset += b.len();
    }
}
fn identity(args: &Args, reference: &Ranks, ids: &[u8]) -> Result<i32, String> {
    let candidate = matching_dump(reference, ids, &args.candidate, args.legacy.as_ref())?;
    if boolean(&reference[&0], "bf16_mma") != boolean(&candidate[&0], "bf16_mma")
        || perturbation(&reference[&0])?.is_some()
        || perturbation(&candidate[&0])?.is_some()
    {
        return Err("identity mode requires two unperturbed dumps with the same arithmetic".into());
    }
    let mma = if boolean(&reference[&0], "bf16_mma") {
        "True"
    } else {
        "False"
    };
    crate::output::report(format_args!(
        "comparison=bit-identity position={} world={} bf16_mma={mma} prompt_identical=true checkpoint_identity=not_recorded",
        int(&reference[&0], "position"),
        reference.len()
    ), true);
    let mut passed = true;
    let mut state_bytes = 0;
    for layer in LAYERS {
        for (&rank, doc) in reference {
            let name = format!("layer-{layer:03}-rank-{rank}.f32");
            let (equal, offset) =
                identical_files(&args.fp32.join(&name), &args.candidate.join(&name))?;
            let size = (int(doc, "head_end") - int(doc, "head_start")) * HEAD_BYTES;
            if equal && offset != size {
                return Err(format!(
                    "layer {layer} rank {rank}: state size changed during identity comparison"
                ));
            }
            state_bytes += size;
            passed &= equal;
            crate::output::report(format_args!(
                "identity layer={layer:03} rank={rank} bit_identical={equal} state_bytes={size} first_different_byte={}",
                if equal {
                    "none".into()
                } else {
                    offset.to_string()
                }
            ), true);
        }
    }
    crate::output::report(format_args!(
        "{} bit_identical={passed} state_bytes_per_dump={state_bytes} input_state_bytes={}",
        if passed { "PASS" } else { "FAIL" },
        2 * state_bytes
    ), true);
    Ok(i32::from(!passed))
}
fn pair_stats(b: &[u8], a: &[u8], layer: usize, head: usize) -> Result<Stats, String> {
    let mut stats = Stats {
        elements: HEAD_ELEMENTS,
        ..Stats::default()
    };
    for (bv, av) in float_values(b).zip(float_values(a)) {
        if !av.is_finite() || !bv.is_finite() {
            return Err(format!("layer {layer} head {head}: nonfinite state"));
        }
        stats.max_ref_abs = stats.max_ref_abs.max(bv.abs());
        let absolute = (av - bv).abs();
        stats.max_abs_error = stats.max_abs_error.max(absolute);
        stats.above_abs += usize::from(absolute > BOUND);
    }
    stats.max_head_normalized_error = normalized(stats.max_abs_error, stats.max_ref_abs);
    let floor = BOUND * stats.max_ref_abs;
    for (bv, av) in float_values(b).zip(float_values(a)) {
        let relative = normalized((av - bv).abs(), bv.abs().max(floor));
        stats.max_relative_error = stats.max_relative_error.max(relative);
        stats.sum_relative_error += relative;
        stats.above_rel += usize::from(relative > BOUND);
    }
    Ok(stats)
}
fn two_arm(args: &Args, reference: &Ranks, ids: &[u8]) -> Result<i32, String> {
    let candidate = matching_dump(reference, ids, &args.candidate, args.legacy.as_ref())?;
    if boolean(&reference[&0], "bf16_mma")
        || perturbation(&reference[&0])?.is_some()
        || perturbation(&candidate[&0])?.is_some()
    {
        return Err(
            "two-arm mode requires an unperturbed FP32 reference and an unperturbed candidate"
                .into(),
        );
    }
    let control = !boolean(&candidate[&0], "bf16_mma");
    crate::output::report(format_args!(
        "position={} layers={} heads=48 world={} dtype=f32-le axes=value_head,key,value comparison={} relative_floor=1e-3*head_max_ref_abs",
        int(&reference[&0], "position"),
        LAYERS.len(),
        reference.len(),
        if control { "fp32-control" } else { "split" }
    ), true);
    let mut overall = Stats::default();
    let (mut worst_layer, mut worst_head) = (LAYERS[0], 0);
    for layer in LAYERS {
        let mut stats = Stats::default();
        let mut layer_worst_head = 0;
        for (&rank, doc) in reference {
            let name = format!("layer-{layer:03}-rank-{rank}.f32");
            let rp = args.fp32.join(&name);
            let cp = args.candidate.join(&name);
            let (mut r, mut c) = (open(&rp)?, open(&cp)?);
            for head in int(doc, "head_start")..int(doc, "head_end") {
                let (b, a) = (read_block(&mut r, &rp)?, read_block(&mut c, &cp)?);
                if a.len() != HEAD_BYTES || b.len() != HEAD_BYTES {
                    return Err(format!(
                        "layer {layer} head {head}: state size changed during comparison"
                    ));
                }
                let head_stats = pair_stats(&b, &a, layer, head)?;
                show(
                    &format!(
                        "layer={layer:03} head={head:02} rank={rank} relative_floor={}",
                        scientific(BOUND * head_stats.max_ref_abs)
                    ),
                    &head_stats,
                );
                if head_stats.max_head_normalized_error > stats.max_head_normalized_error {
                    layer_worst_head = head;
                }
                stats.add(&head_stats);
            }
            if trailing(&mut r, &rp)? || trailing(&mut c, &cp)? {
                return Err(format!(
                    "layer {layer} rank {rank}: state grew during comparison"
                ));
            }
        }
        if stats.elements != 48 * HEAD_ELEMENTS {
            return Err(format!("layer {layer}: incomplete state coverage"));
        }
        show(
            &format!("layer={layer:03} worst_head={layer_worst_head:02}"),
            &stats,
        );
        if stats.max_head_normalized_error > overall.max_head_normalized_error {
            worst_layer = layer;
            worst_head = layer_worst_head;
        }
        overall.add(&stats);
    }
    show(
        &format!("overall worst_layer={worst_layer:03} worst_head={worst_head:02}"),
        &overall,
    );
    let passed =
        overall.max_head_normalized_error <= BOUND && (!control || overall.max_abs_error == 0.0);
    crate::output::report(format_args!(
        "{} metric=max_head_normalized_error bound={}; every (layer,head) max|a-b|/max|b| <= bound; fp32_control_exact_zero={}",
        if passed { "PASS" } else { "FAIL" },
        scientific(BOUND),
        if control { "required" } else { "n/a" }
    ), true);
    Ok(i32::from(!passed))
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
}
fn floor(args: &Args, reference: &Ranks, ids: &[u8]) -> Result<i32, String> {
    let perturbed = matching_dump(reference, ids, &args.candidate, args.legacy.as_ref())?;
    let split_path = args.split.as_ref().unwrap();
    let split = matching_dump(reference, ids, split_path, args.legacy.as_ref())?;
    if boolean(&reference[&0], "bf16_mma")
        || perturbation(&reference[&0])?.is_some()
        || boolean(&perturbed[&0], "bf16_mma")
        || perturbation(&perturbed[&0])?.is_none()
        || !boolean(&split[&0], "bf16_mma")
        || perturbation(&split[&0])?.is_some()
    {
        return Err("floor mode requires unperturbed FP32, tagged perturbed FP32, unperturbed split in that order".into());
    }
    let probe = perturbation(&perturbed[&0])?.unwrap();
    let ulp = match get(probe, "ulp") {
        Some(Value::Int(v)) => v,
        _ => unreachable!(),
    };
    crate::output::report(format_args!(
        "position={} layers={} heads=48 world={} dtype=f32-le axes=value_head,key,value comparison=amplification-floor bound=2 ulp={ulp} every={} input_state_bytes={}",
        int(&reference[&0], "position"),
        LAYERS.len(),
        reference.len(),
        integer_text(get(probe, "every").unwrap()).unwrap(),
        3 * LAYERS.len() * 48 * HEAD_BYTES
    ), true);
    let mut table = Vec::new();
    for layer in LAYERS {
        let (mut floor_norm, mut split_norm) = (Vec::new(), Vec::new());
        let (mut ref_energy, mut floor_energy, mut split_energy) = (0.0, 0.0, 0.0);
        for (&rank, doc) in reference {
            let name = format!("layer-{layer:03}-rank-{rank}.f32");
            let paths = [
                args.fp32.join(&name),
                args.candidate.join(&name),
                split_path.join(&name),
            ];
            let mut files = [open(&paths[0])?, open(&paths[1])?, open(&paths[2])?];
            for head in int(doc, "head_start")..int(doc, "head_end") {
                let blocks = [
                    read_block(&mut files[0], &paths[0])?,
                    read_block(&mut files[1], &paths[1])?,
                    read_block(&mut files[2], &paths[2])?,
                ];
                if blocks.iter().any(|block| block.len() != HEAD_BYTES) {
                    return Err(format!(
                        "layer {layer} head {head}: state size changed during comparison"
                    ));
                }
                let (mut scale, mut df, mut ds, mut re, mut fe, mut se) =
                    (0.0_f64, 0.0_f64, 0.0_f64, 0.0, 0.0, 0.0);
                for ((b, p), s) in float_values(&blocks[0])
                    .zip(float_values(&blocks[1]))
                    .zip(float_values(&blocks[2]))
                {
                    if !b.is_finite() || !p.is_finite() || !s.is_finite() {
                        return Err(format!("layer {layer} head {head}: nonfinite state"));
                    }
                    let (ef, es) = ((p - b).abs(), (s - b).abs());
                    scale = scale.max(b.abs());
                    df = df.max(ef);
                    ds = ds.max(es);
                    re += b * b;
                    fe += ef * ef;
                    se += es * es;
                }
                let (fnorm, snorm) = (normalized(df, scale), normalized(ds, scale));
                floor_norm.push(fnorm);
                split_norm.push(snorm);
                ref_energy += re;
                floor_energy += fe;
                split_energy += se;
                crate::output::report(format_args!(
                    "raw layer={layer:03} head={head:02} rank={rank} state_bytes={HEAD_BYTES} max_ref_abs={} floor_max_abs={} split_max_abs={} floor_normalized={} split_normalized={} ratio={} ref_energy={} floor_energy={} split_energy={}",
                    scientific(scale),
                    scientific(df),
                    scientific(ds),
                    scientific(fnorm),
                    scientific(snorm),
                    scientific(floor_ratio(snorm, fnorm)),
                    scientific(re),
                    scientific(fe),
                    scientific(se)
                ), true);
            }
            for index in 0..3 {
                if trailing(&mut files[index], &paths[index])? {
                    return Err(format!(
                        "layer {layer} rank {rank}: state grew during comparison"
                    ));
                }
            }
        }
        if floor_norm.len() != 48 {
            return Err(format!("layer {layer}: incomplete state coverage"));
        }
        let (fm, sm) = (median(&mut floor_norm), median(&mut split_norm));
        let (fx, sx) = (floor_norm[47], split_norm[47]);
        let (ff, sf) = (
            normalized(floor_energy, ref_energy).sqrt(),
            normalized(split_energy, ref_energy).sqrt(),
        );
        let ratios = [
            floor_ratio(sm, fm),
            floor_ratio(sx, fx),
            floor_ratio(sf, ff),
        ];
        let passed = ratios.iter().all(|r| *r <= 2.0);
        table.push((
            layer,
            [fm, fx, ff, sm, sx, sf, ratios[0], ratios[1], ratios[2]],
            passed,
        ));
    }
    crate::output::report(format_args!(
        "layer | floor_head_median | floor_head_max | floor_fro | split_head_median | split_head_max | split_fro | median_ratio | max_ratio | fro_ratio | state_bytes | verdict"
    ), true);
    for (layer, values, passed) in &table {
        crate::output::report(format_args!(
            "{layer:03} | {} | {} | {}",
            values
                .iter()
                .map(|v| scientific(*v))
                .collect::<Vec<_>>()
                .join(" | "),
            48 * HEAD_BYTES,
            if *passed { "PASS" } else { "FAIL" }
        ), true);
    }
    let passed = table.iter().all(|(_, _, passed)| *passed);
    crate::output::report(format_args!(
        "{} metric=per_layer_head_median,head_max,relative_frobenius bound=2*floor; every layer must pass all three metrics; raw head ratios are diagnostics",
        if passed { "PASS" } else { "FAIL" }
    ), true);
    Ok(i32::from(!passed))
}
fn compare(args: &Args) -> Result<i32, String> {
    let (reference, ids) = load_dump(&args.fp32, args.legacy.as_ref())?;
    if args.identity {
        identity(args, &reference, &ids)
    } else if args.split.is_some() {
        floor(args, &reference, &ids)
    } else {
        two_arm(args, &reference, &ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_preserves_signed_zero() {
        let root =
            std::env::temp_dir().join(format!("vade-gdn-{}", crate::text::random_hex(12).unwrap()));
        fs::create_dir(&root).unwrap();
        let reference = root.join("reference");
        let candidate = root.join("candidate");
        fs::write(&reference, [0, 0, 0, 0]).unwrap();
        fs::write(&candidate, [0, 0, 0, 128]).unwrap();
        let same = identical_files(&reference, &reference);
        let changed = identical_files(&reference, &candidate);
        fs::remove_file(reference).unwrap();
        fs::remove_file(candidate).unwrap();
        fs::remove_dir(root).unwrap();
        assert_eq!(same.unwrap(), (true, 4));
        assert_eq!(changed.unwrap(), (false, 3));
    }
}
