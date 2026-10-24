//! Frozen long-context scoring and retrieval comparison.
use super::json::{self, Value};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

const LENGTHS: [i64; 2] = [32768, 131072];
const DEPTHS: [i64; 5] = [10, 25, 50, 75, 90];
const BUCKETS: [(usize, usize); 6] = [
    (0, 4096),
    (4096, 8192),
    (8192, 16384),
    (16384, 32768),
    (32768, 65536),
    (65536, 131072),
];

fn io_error(error: std::io::Error) -> String {
    use std::io::ErrorKind;
    match error.kind() {
        ErrorKind::NotFound => "FileNotFoundError",
        ErrorKind::PermissionDenied => "PermissionError",
        ErrorKind::AlreadyExists => "FileExistsError",
        ErrorKind::IsADirectory => "IsADirectoryError",
        ErrorKind::NotADirectory => "NotADirectoryError",
        _ => "OSError",
    }
    .to_owned()
}

fn obj(items: Vec<(&str, Value)>) -> Value {
    Value::Object(items.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn string(value: &Value, key: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("missing or invalid {key}"))
}
fn integer(value: &Value, key: &str) -> Result<i64, String> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("missing or invalid {key}"))
}
fn array<'a>(value: &'a Value, key: &str) -> Result<&'a [Value], String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("missing or invalid {key}"))
}
fn nullable_position(position: Option<usize>) -> Value {
    position
        .map(|n| Value::Int(n as i64))
        .unwrap_or(Value::Null)
}
fn matches_string(value: &Value, key: &str, expected: &str) -> bool {
    value.get(key).and_then(Value::as_str) == Some(expected)
}
fn matches_integer(value: &Value, key: &str, expected: i64) -> bool {
    value.get(key).is_some_and(|v| match v {
        Value::Bool(n) => i64::from(*n) == expected,
        _ => v.as_f64() == Some(expected as f64),
    })
}
fn is_null(value: Option<&Value>) -> bool {
    value.is_none_or(|v| matches!(v, Value::Null))
}

pub fn bucket(position: usize) -> Result<usize, String> {
    BUCKETS
        .iter()
        .position(|&(lo, hi)| lo <= position && position < hi)
        .ok_or_else(|| "position outside 0..128k".to_owned())
}

#[derive(Clone, Eq, PartialEq)]
pub enum TokenId {
    Small(i64),
    Big(String),
}
impl TokenId {
    fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Int(n) if *n >= 0 => Some(Self::Small(*n)),
            Value::BigInt(n) if !n.starts_with('-') => Some(Self::Big(n.clone())),
            _ => None,
        }
    }
    fn index(&self) -> Option<usize> {
        match self {
            Self::Small(n) => usize::try_from(*n).ok(),
            Self::Big(n) => n.parse().ok(),
        }
    }
}
impl Ord for TokenId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => a.cmp(b),
            (Self::Small(_), Self::Big(_)) => std::cmp::Ordering::Less,
            (Self::Big(_), Self::Small(_)) => std::cmp::Ordering::Greater,
            (Self::Big(a), Self::Big(b)) => a.len().cmp(&b.len()).then_with(|| a.cmp(b)),
        }
    }
}
impl PartialOrd for TokenId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl std::fmt::Display for TokenId {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Small(n) => n.fmt(out),
            Self::Big(n) => n.fmt(out),
        }
    }
}

fn decimal_parts(value: &str) -> (bool, &str) {
    let negative = value.starts_with('-');
    let digits = value.trim_start_matches('-').trim_start_matches('0');
    if digits.is_empty() {
        (false, "0")
    } else {
        (negative, digits)
    }
}
fn magnitude_compare(a: &str, b: &str) -> std::cmp::Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}
fn integer_compare(a: &str, b: &str) -> std::cmp::Ordering {
    let (a_negative, a) = decimal_parts(a);
    let (b_negative, b) = decimal_parts(b);
    if a_negative != b_negative {
        return b_negative.cmp(&a_negative);
    }
    let order = magnitude_compare(a, b);
    if a_negative { order.reverse() } else { order }
}
fn decimal_add(a: &str, b: &str) -> String {
    let a: Vec<_> = a.bytes().rev().collect();
    let b: Vec<_> = b.bytes().rev().collect();
    let mut digits = Vec::with_capacity(a.len().max(b.len()) + 1);
    let mut carry = 0;
    for i in 0..a.len().max(b.len()) {
        let value = a.get(i).map_or(0, |n| n - b'0') + b.get(i).map_or(0, |n| n - b'0') + carry;
        digits.push(value % 10 + b'0');
        carry = value / 10;
    }
    if carry != 0 {
        digits.push(carry + b'0');
    }
    digits.reverse();
    String::from_utf8(digits).unwrap()
}
fn decimal_subtract(a: &str, b: &str) -> String {
    let b: Vec<_> = b.bytes().rev().collect();
    let mut digits = Vec::with_capacity(a.len());
    let mut borrow = 0;
    for (i, n) in a.bytes().rev().enumerate() {
        let value = (n - b'0') as i16 - b.get(i).map_or(0, |n| (n - b'0') as i16) - borrow;
        borrow = i16::from(value < 0);
        digits.push((value + borrow * 10) as u8 + b'0');
    }
    while digits.len() > 1 && digits.last() == Some(&b'0') {
        digits.pop();
    }
    digits.reverse();
    String::from_utf8(digits).unwrap()
}
fn integer_difference(a: &str, b: &str) -> String {
    let (a_negative, a) = decimal_parts(a);
    let (b_negative, b) = decimal_parts(b);
    if a_negative != b_negative {
        return decimal_add(a, b);
    }
    match magnitude_compare(a, b) {
        std::cmp::Ordering::Less => decimal_subtract(b, a),
        _ => decimal_subtract(a, b),
    }
}
fn integer_float(value: &str) -> Result<f64, String> {
    let number = value
        .parse::<f64>()
        .map_err(|_| "OverflowError".to_owned())?;
    if number.is_finite() {
        Ok(number)
    } else {
        Err("OverflowError".to_owned())
    }
}

#[derive(Clone)]
pub enum Logit {
    Int(i64),
    Big(String),
    Float(f64),
}
impl Logit {
    fn integer_text(&self) -> Option<String> {
        match self {
            Self::Int(n) => Some(n.to_string()),
            Self::Big(n) => Some(n.clone()),
            Self::Float(_) => None,
        }
    }
    fn as_float(&self) -> Result<f64, String> {
        match self {
            Self::Int(n) => Ok(*n as f64),
            Self::Big(n) => integer_float(n),
            Self::Float(n) => Ok(*n),
        }
    }
    fn compare(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a.cmp(b),
            (Self::Float(a), Self::Float(b)) => a.partial_cmp(b).unwrap(),
            (_, Self::Float(b)) => {
                let whole = b.trunc();
                let order = integer_compare(&self.integer_text().unwrap(), &format!("{whole:.0}"));
                if order != Ordering::Equal {
                    return order;
                }
                if *b > whole {
                    Ordering::Less
                } else if *b < whole {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            }
            (Self::Float(_), _) => other.compare(self).reverse(),
            _ => integer_compare(
                &self.integer_text().unwrap(),
                &other.integer_text().unwrap(),
            ),
        }
    }
    fn difference(&self, other: &Self) -> Result<f64, String> {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => Ok((*a as i128 - *b as i128).unsigned_abs() as f64),
            (Self::Float(_), _) | (_, Self::Float(_)) => {
                Ok((self.as_float()? - other.as_float()?).abs())
            }
            _ => integer_float(&integer_difference(
                &self.integer_text().unwrap(),
                &other.integer_text().unwrap(),
            )),
        }
    }
}

pub struct ScoreRow {
    pub position: usize,
    pub input_id: u32,
    pub top1_id: TokenId,
    pub topk: Vec<(TokenId, Logit)>,
    pub scores: Vec<(TokenId, Logit)>,
    pub full_logits: Option<Vec<f64>>,
}

fn pairs(items: &Value, k: usize) -> Result<Vec<(TokenId, Logit)>, String> {
    let Some(items) = items.as_array().filter(|items| items.len() == k) else {
        return Err("wrong top-k width".to_owned());
    };
    let mut result = Vec::with_capacity(k);
    for item in items {
        let Some(pair) = item.as_array().filter(|pair| pair.len() == 2) else {
            return Err("invalid ID/logit pair".to_owned());
        };
        let token = TokenId::from_value(&pair[0])
            .ok_or_else(|| "invalid, duplicate or nonfinite ID/logit".to_owned())?;
        if result.iter().any(|(id, _)| id == &token) {
            return Err("invalid, duplicate or nonfinite ID/logit".to_owned());
        }
        let logit = match &pair[1] {
            Value::Int(n) => Logit::Int(*n),
            Value::BigInt(n) => {
                integer_float(n)?;
                Logit::Big(n.clone())
            }
            Value::Float(n) if n.is_finite() => Logit::Float(*n),
            _ => return Err("invalid, duplicate or nonfinite ID/logit".to_owned()),
        };
        result.push((token, logit));
    }
    Ok(result)
}

enum ScoreInput {
    Ndjson(BufReader<File>),
    Native {
        logits: BufReader<File>,
        inputs: BufReader<File>,
        vocab: usize,
    },
}

pub struct ScoreRows {
    input: ScoreInput,
    position: usize,
    length: usize,
    k: usize,
    ids_hash: super::Sha256,
    expected_hash: Option<Value>,
    finished: bool,
}

fn read_record(stream: &mut BufReader<File>) -> Result<Value, String> {
    let mut line = Vec::new();
    loop {
        let buffer = stream.fill_buf().map_err(io_error)?;
        if buffer.is_empty() {
            break;
        }
        if let Some(end) = buffer.iter().position(|&b| b == b'\n' || b == b'\r') {
            let carriage = buffer[end] == b'\r';
            line.extend_from_slice(&buffer[..end]);
            stream.consume(end + 1);
            if carriage && stream.fill_buf().map_err(io_error)?.first() == Some(&b'\n') {
                stream.consume(1);
            }
            return json::parse(std::str::from_utf8(&line).map_err(|_| "UnicodeDecodeError")?);
        }
        let size = buffer.len();
        line.extend_from_slice(buffer);
        stream.consume(size);
    }
    if line.is_empty() {
        return Err("StopIteration".to_owned());
    }
    json::parse(std::str::from_utf8(&line).map_err(|_| "UnicodeDecodeError")?)
}

pub fn score_meta(directory: &Path, case: &Value) -> Result<Value, String> {
    let id = string(case, "id")?;
    let path = directory.join(format!("{id}.score.jsonl"));
    if path.exists() {
        read_record(&mut BufReader::new(File::open(path).map_err(io_error)?))
    } else {
        json::read(&directory.join(id).join("metadata.json"))
    }
}

pub fn score_rows(directory: &Path, case: &Value, k: usize) -> Result<ScoreRows, String> {
    let id = string(case, "id")?;
    let length = integer(case, "length")?;
    if length < 0 {
        return Err("invalid scoring length".to_owned());
    }
    let path = directory.join(format!("{id}.score.jsonl"));
    let input = if path.exists() {
        let mut stream = BufReader::new(File::open(path).map_err(io_error)?);
        let meta = read_record(&mut stream)?;
        if !matches_string(&meta, "object", "vade.score.v1")
            || !matches_string(&meta, "path", "prefill_all")
            || !matches_integer(&meta, "prompt_tokens", length)
            || !matches_integer(&meta, "top_k", k as i64)
            || meta.get("prompt_sha256") != case.get("prompt_sha256")
        {
            return Err("scoring metadata differs from manifest".to_owned());
        }
        ScoreInput::Ndjson(stream)
    } else {
        let native = directory.join(id);
        let meta = json::read(&native.join("metadata.json"))?;
        let vocab = meta
            .get("vocab_size")
            .and_then(Value::as_i64)
            .filter(|&n| n >= k as i64 && n <= 1_000_000);
        if !matches_string(&meta, "path", "prefill_all")
            || !matches_integer(&meta, "prompt_tokens", length)
            || meta.get("prompt_sha256") != case.get("prompt_sha256")
            || vocab.is_none()
        {
            return Err("invalid native dump metadata".to_owned());
        }
        ScoreInput::Native {
            logits: BufReader::new(File::open(native.join("logits.f32")).map_err(io_error)?),
            inputs: BufReader::new(File::open(native.join("input_ids.u32")).map_err(io_error)?),
            vocab: vocab.unwrap() as usize,
        }
    };
    Ok(ScoreRows {
        input,
        position: 0,
        length: length as usize,
        k,
        ids_hash: super::Sha256::new(),
        expected_hash: case.get("input_ids_sha256").cloned(),
        finished: false,
    })
}

fn verified_hash(hash: &mut super::Sha256, expected: &Option<Value>) -> Result<bool, String> {
    let actual = std::mem::replace(hash, super::Sha256::new()).finalize();
    let expected = expected.as_ref().ok_or("KeyError")?;
    Ok(expected.as_str() == Some(actual.as_str()))
}

impl ScoreRows {
    fn read_row(&mut self) -> Result<ScoreRow, String> {
        match &mut self.input {
            ScoreInput::Ndjson(stream) => {
                let row = read_record(stream)?;
                if !matches_integer(&row, "position", self.position as i64)
                    || !matches!(row.get("input_id"), Some(Value::Int(_) | Value::BigInt(_)))
                {
                    return Err("missing, reordered or invalid scoring row".to_owned());
                }
                let input_id = match row.get("input_id") {
                    Some(Value::Int(n)) => u32::try_from(*n).map_err(|_| "error".to_owned())?,
                    _ => return Err("error".to_owned()),
                };
                let topk = pairs(row.get("topk").ok_or("KeyError")?, self.k)?;
                let scores = pairs(row.get("scores").ok_or("KeyError")?, self.k)?;
                let top1_id = row
                    .get("top1_id")
                    .and_then(TokenId::from_value)
                    .ok_or_else(|| "top-1 ID is inconsistent with top-k".to_owned())?;
                let best = topk
                    .iter()
                    .max_by(|a, b| a.1.compare(&b.1).then_with(|| b.0.cmp(&a.0)))
                    .ok_or_else(|| "top-1 ID is inconsistent with top-k".to_owned())?;
                if best.0 != top1_id {
                    return Err("top-1 ID is inconsistent with top-k".to_owned());
                }
                self.ids_hash.update(&input_id.to_le_bytes());
                Ok(ScoreRow {
                    position: self.position,
                    input_id,
                    top1_id,
                    topk,
                    scores,
                    full_logits: None,
                })
            }
            ScoreInput::Native {
                logits,
                inputs,
                vocab,
            } => {
                let mut raw = vec![0; *vocab * 4];
                let mut token = [0; 4];
                if logits.read_exact(&mut raw).is_err() || inputs.read_exact(&mut token).is_err() {
                    return Err("truncated native dump".to_owned());
                }
                let values: Vec<f64> = raw
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()) as f64)
                    .collect();
                if values.iter().any(|v| !v.is_finite()) {
                    return Err("nonfinite native logits".to_owned());
                }
                self.ids_hash.update(&token);
                let mut topk: Vec<(i64, f64)> = Vec::with_capacity(self.k + 1);
                for (token, &value) in values.iter().enumerate() {
                    let at = topk.partition_point(|&(id, logit)| {
                        logit > value || logit == value && id < token as i64
                    });
                    if at < self.k {
                        topk.insert(at, (token as i64, value));
                        topk.truncate(self.k);
                    }
                }
                Ok(ScoreRow {
                    position: self.position,
                    input_id: u32::from_le_bytes(token),
                    top1_id: TokenId::Small(topk[0].0),
                    topk: topk
                        .into_iter()
                        .map(|(id, value)| (TokenId::Small(id), Logit::Float(value)))
                        .collect(),
                    scores: Vec::new(),
                    full_logits: Some(values),
                })
            }
        }
    }
    fn finish(&mut self) -> Result<(), String> {
        match &mut self.input {
            ScoreInput::Ndjson(stream) => {
                let end = read_record(stream)?;
                let mut rest = String::new();
                stream.read_to_string(&mut rest).map_err(io_error)?;
                if !matches_string(&end, "object", "vade.score.end")
                    || !matches_integer(&end, "rows", self.length as i64)
                    || !rest.trim_matches(py_whitespace).is_empty()
                {
                    return Err("missing terminal record or excess scoring rows".to_owned());
                }
                if !verified_hash(&mut self.ids_hash, &self.expected_hash)? {
                    return Err("input token IDs differ from frozen manifest".to_owned());
                }
            }
            ScoreInput::Native { logits, inputs, .. } => {
                let mut extra = [0; 1];
                if logits.read(&mut extra).map_err(io_error)? != 0
                    || inputs.read(&mut extra).map_err(io_error)? != 0
                    || !verified_hash(&mut self.ids_hash, &self.expected_hash)?
                {
                    return Err("native dump size or input IDs differ".to_owned());
                }
            }
        }
        Ok(())
    }
}

impl Iterator for ScoreRows {
    type Item = Result<ScoreRow, String>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        if self.position == self.length {
            self.finished = true;
            return match self.finish() {
                Ok(()) => None,
                Err(error) => Some(Err(error)),
            };
        }
        let row = self.read_row();
        self.position += 1;
        if row.is_err() {
            self.finished = true;
        }
        Some(row)
    }
}

// Keep the partials and final half-even correction used by Python's math.fsum.
fn fsum(values: impl IntoIterator<Item = f64>) -> Result<f64, String> {
    let mut partials: Vec<f64> = Vec::new();
    let mut special_sum = 0.0;
    let mut infinity_sum = 0.0;
    for mut x in values {
        let original = x;
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let yr = hi - x;
            let lo = y - yr;
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        if !x.is_finite() {
            if original.is_finite() {
                return Err("OverflowError".to_owned());
            }
            if original.is_infinite() {
                infinity_sum += original;
            }
            special_sum += original;
            partials.clear();
            continue;
        }
        if x != 0.0 {
            partials.push(x);
        }
    }
    if special_sum != 0.0 {
        if infinity_sum.is_nan() {
            return Err("-inf + inf in fsum".to_owned());
        }
        return Ok(special_sum);
    }
    let Some(mut hi) = partials.pop() else {
        return Ok(0.0);
    };
    let mut lo = 0.0;
    while let Some(y) = partials.pop() {
        let x = hi;
        hi = x + y;
        let yr = hi - x;
        lo = y - yr;
        if lo != 0.0 {
            break;
        }
    }
    if partials
        .last()
        .is_some_and(|&p| lo < 0.0 && p < 0.0 || lo > 0.0 && p > 0.0)
    {
        let y = lo * 2.0;
        let x = hi + y;
        if y == x - hi {
            hi = x;
        }
    }
    Ok(hi)
}

#[derive(Clone, Default)]
struct Bucket {
    positions: i64,
    disagreements: i64,
    logit_cells: i64,
    absolute_logit_difference: f64,
    first_divergence: Option<usize>,
}
impl Bucket {
    fn value(&self, i: usize, finalize: bool) -> Value {
        let mut fields = vec![
            (
                "bucket",
                Value::Str(format!("{}-{}k", BUCKETS[i].0 / 1024, BUCKETS[i].1 / 1024)),
            ),
            ("positions", Value::Int(self.positions)),
            ("disagreements", Value::Int(self.disagreements)),
            ("logit_cells", Value::Int(self.logit_cells)),
            (
                "absolute_logit_difference",
                Value::Float(self.absolute_logit_difference),
            ),
            ("first_divergence", nullable_position(self.first_divergence)),
        ];
        if finalize {
            fields.push((
                "agreement_pct",
                if self.positions != 0 {
                    Value::Float(100.0 * (1.0 - self.disagreements as f64 / self.positions as f64))
                } else {
                    Value::Null
                },
            ));
            fields.push((
                "mean_abs_logit_diff",
                if self.logit_cells != 0 {
                    Value::Float(self.absolute_logit_difference / self.logit_cells as f64)
                } else {
                    Value::Null
                },
            ));
        }
        obj(fields)
    }
    fn metric(&self, metric: &str) -> f64 {
        if metric == "mean_abs_logit_diff" {
            self.absolute_logit_difference / self.logit_cells as f64
        } else {
            self.disagreements as f64 / self.positions as f64
        }
    }
}
fn bucket_values(stats: &[Bucket; 6], finalize: bool) -> Value {
    Value::Array(
        stats
            .iter()
            .enumerate()
            .map(|(i, s)| s.value(i, finalize))
            .collect(),
    )
}
struct Comparison {
    id: String,
    length: i64,
    first_divergence: Option<usize>,
    buckets: [Bucket; 6],
}
impl Comparison {
    fn value(&self, finalize: bool) -> Value {
        obj(vec![
            ("id", Value::Str(self.id.clone())),
            ("length", Value::Int(self.length)),
            ("first_divergence", nullable_position(self.first_divergence)),
            ("buckets", bucket_values(&self.buckets, finalize)),
        ])
    }
}

fn compare_document(
    a_dir: &Path,
    b_dir: &Path,
    case: &Value,
    k: usize,
    floor: bool,
) -> Result<Comparison, String> {
    let a_meta = score_meta(a_dir, case)?;
    let b_meta = score_meta(b_dir, case)?;
    let same_world = [2, 4].into_iter().any(|world|
        matches_integer(&a_meta, "world", world) && matches_integer(&b_meta, "world", world));
    if !same_world
        || !is_null(a_meta.get("perturbation"))
    {
        return Err("scoring requires the same world (2 or 4) and unperturbed A".to_owned());
    }
    let probe = b_meta.get("perturbation");
    if floor {
        if !probe.is_some_and(|p| {
            matches!(p, Value::Object(_))
                && (matches_integer(p, "ulp", -1) || matches_integer(p, "ulp", 1))
                && matches_integer(p, "every", 1)
                && matches_string(p, "scope", "prefill")
        }) {
            return Err("C lacks the explicit layer-zero ULP floor provenance".to_owned());
        }
    } else if !is_null(probe) {
        return Err("B must be unperturbed".to_owned());
    }
    let mut a_rows = score_rows(a_dir, case, k)?;
    let mut b_rows: Option<ScoreRows> = None;
    let mut stats: [Bucket; 6] = std::array::from_fn(|_| Bucket::default());
    let mut first = None;
    loop {
        let a = a_rows.next().transpose()?;
        if b_rows.is_none() {
            b_rows = Some(score_rows(b_dir, case, k)?);
        }
        let b = b_rows.as_mut().unwrap().next().transpose()?;
        let (a, b) = match (a, b) {
            (None, None) => break,
            (Some(a), Some(b)) => (a, b),
            _ => {
                return Err(
                    "teacher-forced arms have different position/token sequences".to_owned(),
                );
            }
        };
        if a.position != b.position || a.input_id != b.input_id {
            return Err("teacher-forced arms have different position/token sequences".to_owned());
        }
        let s = &mut stats[bucket(a.position)?];
        let different = a.top1_id != b.top1_id;
        s.positions += 1;
        s.disagreements += i64::from(different);
        if different {
            first.get_or_insert(a.position);
            s.first_divergence.get_or_insert(a.position);
        }
        let scores = if let Some(full) = &b.full_logits {
            a.topk
                .iter()
                .map(|(token, _)| {
                    token
                        .index()
                        .and_then(|index| full.get(index))
                        .copied()
                        .map(|v| (token.clone(), Logit::Float(v)))
                        .ok_or_else(|| "native arms have different vocabulary sizes".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            b.scores
        };
        if scores.len() != a.topk.len()
            || scores
                .iter()
                .any(|(token, _)| !a.topk.iter().any(|(id, _)| id == token))
        {
            return Err(
                "candidate logits are missing A's top-k IDs; rank-wise differences are invalid"
                    .to_owned(),
            );
        }
        let differences = a
            .topk
            .iter()
            .map(|(token, value)| {
                value.difference(&scores.iter().find(|(id, _)| id == token).unwrap().1)
            })
            .collect::<Result<Vec<_>, _>>()?;
        s.absolute_logit_difference += fsum(differences)?;
        s.logit_cells += k as i64;
    }
    Ok(Comparison {
        id: string(case, "id")?,
        length: integer(case, "length")?,
        first_divergence: first,
        buckets: stats,
    })
}

fn aggregate(documents: &[Comparison], length: i64) -> [Bucket; 6] {
    let mut result: [Bucket; 6] = std::array::from_fn(|_| Bucket::default());
    for document in documents.iter().filter(|doc| doc.length == length) {
        for (total, item) in result.iter_mut().zip(&document.buckets) {
            total.positions += item.positions;
            total.disagreements += item.disagreements;
            total.logit_cells += item.logit_cells;
            total.absolute_logit_difference += item.absolute_logit_difference;
            if let Some(pos) = item.first_divergence {
                if total.first_divergence.is_none_or(|old| pos < old) {
                    total.first_divergence = Some(pos);
                }
            }
        }
    }
    result
}

// Python strip/splitlines includes these control characters as well as Unicode whitespace.
fn py_whitespace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}
fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((offset, c)) = chars.next() {
        if matches!(
            c,
            '\n' | '\r'
                | '\u{b}'
                | '\u{c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        ) {
            lines.push(&text[start..offset]);
            start = offset + c.len_utf8();
            if c == '\r' && chars.peek().is_some_and(|&(_, next)| next == '\n') {
                let (at, next) = chars.next().unwrap();
                start = at + next.len_utf8();
            }
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

pub fn retrieval(directory: &Path, cases: &[Value]) -> Result<Value, String> {
    let mut rows = Vec::new();
    let mut groups: Vec<(String, i64, i64)> = Vec::new();
    for case in cases {
        if matches_string(case, "kind", "teacher") {
            continue;
        }
        let id = string(case, "id")?;
        let reply = json::read(&directory.join(format!("{id}.reply.json")))?;
        let Some(text) = reply
            .get("text")
            .and_then(Value::as_str)
            .filter(|_| reply.get("prompt_sha256") == case.get("prompt_sha256"))
        else {
            return Err("retrieval reply does not match frozen prompt".to_owned());
        };
        let stripped = text.trim_matches(py_whitespace);
        let lines = splitlines(stripped);
        let query = array(case, "query")?;
        let exact = case.get("expected").and_then(Value::as_str) == Some(stripped);
        let mut keys = Vec::new();
        for (i, record) in query.iter().enumerate() {
            let correct = lines.len() == query.len()
                && record.get("value").and_then(Value::as_str) == lines.get(i).copied();
            let depth = integer(record, "depth")?;
            let group = format!(
                "{}-{}-depth{depth}",
                string(case, "kind")?,
                integer(case, "length")?
            );
            let index = groups
                .iter()
                .position(|(key, _, _)| key == &group)
                .unwrap_or_else(|| {
                    groups.push((group, 0, 0));
                    groups.len() - 1
                });
            groups[index].1 += i64::from(correct);
            groups[index].2 += 1;
            keys.push(obj(vec![
                ("depth", Value::Int(depth)),
                (
                    "position",
                    record.get("position").unwrap_or(&Value::Null).clone(),
                ),
                ("correct", Value::Bool(correct)),
            ]));
        }
        rows.push(obj(vec![
            ("id", Value::Str(id)),
            ("exact_match", Value::Bool(exact)),
            ("keys", Value::Array(keys)),
        ]));
    }
    let groups = Value::Object(
        groups
            .into_iter()
            .map(|(group, correct, total)| {
                (
                    group,
                    obj(vec![
                        ("correct", Value::Int(correct)),
                        ("total", Value::Int(total)),
                        (
                            "exact_match_pct",
                            Value::Float(100.0 * correct as f64 / total as f64),
                        ),
                    ]),
                )
            })
            .collect(),
    );
    Ok(obj(vec![
        ("cases", Value::Array(rows)),
        ("per_depth", groups),
    ]))
}

fn decision(
    split: &[[Bucket; 6]; 2],
    floor: Option<&[[Bucket; 6]; 2]>,
    a: &Value,
    b: &Value,
    factor: f64,
) -> Result<(&'static str, Vec<String>), String> {
    let mut reasons = Vec::new();
    for (length_index, &length) in LENGTHS.iter().enumerate() {
        let rows = &split[length_index];
        for (i, row) in rows.iter().enumerate() {
            if row.positions == 0 {
                continue;
            }
            for metric in ["mean_abs_logit_diff", "disagreement_rate"] {
                let value = row.metric(metric);
                let origin = rows[0].metric(metric);
                let label = format!("{}-{}k", BUCKETS[i].0 / 1024, BUCKETS[i].1 / 1024);
                if let Some(floor) = floor {
                    if value > factor * floor[length_index][i].metric(metric) {
                        reasons.push(format!(
                            "{length} {label} {metric} exceeds {factor:.2}x floor"
                        ));
                    }
                }
                if value > factor * origin {
                    reasons.push(format!(
                        "{length} {label} {metric} exceeds {factor:.2}x first bucket (not flat)"
                    ));
                }
            }
        }
    }
    let a_cases = array(a, "cases")?;
    for case in array(b, "cases")? {
        let baseline = a_cases
            .iter()
            .find(|a| a.get("id") == case.get("id"))
            .ok_or_else(|| "missing reference retrieval case".to_owned())?;
        if baseline.get("exact_match") == Some(&Value::Bool(true))
            && case.get("exact_match") != Some(&Value::Bool(true))
            || array(baseline, "keys")?
                .iter()
                .zip(array(case, "keys")?)
                .any(|(a, b)| {
                    a.get("correct") == Some(&Value::Bool(true))
                        && b.get("correct") != Some(&Value::Bool(true))
                })
        {
            reasons.push(format!("{} loses reference retrieval", string(case, "id")?));
        }
    }
    if !reasons.is_empty() {
        return Ok(("FAIL", reasons));
    }
    if floor.is_none() {
        return Ok((
            "PARTIAL",
            vec!["perturbed-FP32 floor is missing".to_owned()],
        ));
    }
    if !a_cases
        .iter()
        .all(|case| case.get("exact_match") == Some(&Value::Bool(true)))
    {
        return Ok((
            "PARTIAL",
            vec!["reference retrieval has misses; owner review required".to_owned()],
        ));
    }
    Ok(("PASS", Vec::new()))
}

pub fn compare(args: &super::LongArgs, manifest: &Value) -> Result<i32, String> {
    let cases = array(manifest, "cases")?;
    let ids = cases
        .iter()
        .map(|case| string(case, "id"))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if ids.len() != cases.len() {
        return Err("duplicate frozen case IDs".to_owned());
    }
    let teacher: Vec<&Value> = cases
        .iter()
        .filter(|case| matches_string(case, "kind", "teacher"))
        .collect();
    for length in LENGTHS {
        let docs: Vec<_> = teacher
            .iter()
            .copied()
            .filter(|case| matches_integer(case, "length", length))
            .collect();
        let documents = docs
            .iter()
            .map(|case| integer(case, "document"))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let hashes = docs
            .iter()
            .map(|case| string(case, "prompt_sha256"))
            .collect::<Result<BTreeSet<_>, _>>()?;
        if !(4..=8).contains(&docs.len())
            || documents.len() != docs.len()
            || hashes.len() != docs.len()
        {
            return Err("suite requires 4..8 distinct scoring documents at each length".to_owned());
        }
    }
    let mut coverage = Vec::new();
    let k = integer(manifest, "keys_per_depth")?;
    for case in cases
        .iter()
        .filter(|case| !matches_string(case, "kind", "teacher"))
    {
        let query = array(case, "query")?;
        let mut depths = query
            .iter()
            .map(|record| integer(record, "depth"))
            .collect::<Result<Vec<_>, _>>()?;
        let unique: BTreeSet<i64> = depths.iter().copied().collect();
        let kind = string(case, "kind")?;
        let depth = if kind == "needle" && query.len() as i64 == k && unique.len() == 1 {
            Some(depths[0])
        } else if kind == "multi" {
            depths.sort_unstable();
            let mut expected: Vec<i64> = (0..k).flat_map(|_| DEPTHS).collect();
            expected.sort_unstable();
            if depths != expected {
                return Err("invalid retrieval query coverage".to_owned());
            }
            None
        } else {
            return Err("invalid retrieval query coverage".to_owned());
        };
        coverage.push((
            kind,
            integer(case, "length")?,
            integer(case, "document")?,
            depth,
        ));
    }
    let mut expected = BTreeSet::new();
    for length in LENGTHS {
        for doc in 0..4 {
            for depth in DEPTHS {
                expected.insert(("needle".to_owned(), length, doc, Some(depth)));
            }
            expected.insert(("multi".to_owned(), length, doc, None));
        }
    }
    if coverage.len() != expected.len() || coverage.into_iter().collect::<BTreeSet<_>>() != expected
    {
        return Err("suite requires four retrieval documents, five depths and multi-key cases at both lengths".to_owned());
    }
    let split_docs = teacher
        .iter()
        .map(|case| compare_document(&args.a_dumps, &args.b_dumps, case, args.top_k, false))
        .collect::<Result<Vec<_>, _>>()?;
    let floor_docs = args
        .c_dumps
        .as_ref()
        .map(|c| {
            teacher
                .iter()
                .map(|case| compare_document(&args.a_dumps, c, case, args.top_k, true))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let split = LENGTHS.map(|length| aggregate(&split_docs, length));
    let floor = floor_docs
        .as_ref()
        .map(|docs| LENGTHS.map(|length| aggregate(docs, length)));
    let a = retrieval(&args.a_dumps, cases)?;
    let b = retrieval(&args.b_dumps, cases)?;
    let (verdict, reasons) = decision(&split, floor.as_ref(), &a, &b, args.floor_factor)?;
    let by_length = |stats: &[[Bucket; 6]; 2]| {
        Value::Object(
            LENGTHS
                .iter()
                .enumerate()
                .map(|(i, length)| (length.to_string(), bucket_values(&stats[i], true)))
                .collect(),
        )
    };
    let result = obj(vec![
        ("status", Value::Str(verdict.to_owned())),
        (
            "reasons",
            Value::Array(reasons.into_iter().map(Value::Str).collect()),
        ),
        (
            "checkpoint_sha256",
            manifest.get("checkpoint_sha256").unwrap_or(&Value::Null).clone(),
        ),
        (
            "metric",
            Value::Str("raw logits on A's top-k token IDs; position = input token index, predicts position+1".to_owned()),
        ),
        ("split_vs_fp32", by_length(&split)),
        (
            "perturbed_fp32_vs_fp32",
            floor.as_ref().map(by_length).unwrap_or(Value::Null),
        ),
        (
            "documents_split",
            Value::Array(split_docs.iter().map(|doc| doc.value(true)).collect()),
        ),
        (
            "documents_floor",
            floor_docs.as_ref().map(|docs| Value::Array(docs.iter().map(|doc| doc.value(true)).collect())).unwrap_or(Value::Null),
        ),
        ("retrieval", obj(vec![("A", a), ("B", b)])),
    ]);
    if let Some(path) = &args.report {
        json::write_json(path, &result)?;
    }
    json::validate(&result)?;
    crate::output::report(format_args!("{}", json::packed(&result)), true);
    Ok(i32::from(verdict == "FAIL"))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn native_fixture(vocab: usize) -> (crate::test_gate::TempDir, Value, Vec<u8>, Vec<u8>) {
        let dir = crate::test_gate::TempDir::new("vade-native-capture-test").unwrap();
        let inputs: Vec<u8> = [7u32, 11].into_iter().flat_map(u32::to_le_bytes).collect();
        let logits: Vec<u8> = (0..2 * vocab).flat_map(|i| ((i % vocab) as f32).to_le_bytes()).collect();
        let case = json::parse(&format!(r#"{{"id":"sample","length":2,"prompt_sha256":"frozen","input_ids_sha256":"{}"}}"#, super::super::sha(&inputs))).unwrap();
        let path = dir.path.join("sample");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("metadata.json"), format!(r#"{{"path":"prefill_all","prompt_tokens":2,"prompt_sha256":"frozen","vocab_size":{vocab}}}"#)).unwrap();
        std::fs::write(path.join("logits.f32"), &logits).unwrap();
        std::fs::write(path.join("input_ids.u32"), &inputs).unwrap();
        (dir, case, logits, inputs)
    }
    #[test]
    fn native_capture_row_bytes_follow_declared_vocab() {
        for vocab in [3, 129] {
            let (dir, case, logits, _) = native_fixture(vocab);
            assert_eq!(logits.len(), 2 * vocab * 4);
            let rows = score_rows(&dir.path, &case, 2).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
            assert_eq!(rows.len(), 2);
            for (position, row) in rows.iter().enumerate() {
                assert_eq!(row.position, position);
                assert_eq!(row.input_id, [7, 11][position]);
                assert_eq!(row.top1_id.to_string(), (vocab - 1).to_string());
                assert_eq!(row.full_logits.as_ref().unwrap(), &(0..vocab).map(|i| i as f64).collect::<Vec<_>>());
            }
        }
    }
    #[test]
    fn native_capture_refuses_truncated_missing_or_excess_bytes() {
        let (dir, case, logits, inputs) = native_fixture(3);
        let path = dir.path.join("sample");
        for (name, original) in [("logits.f32", &logits), ("input_ids.u32", &inputs)] {
            std::fs::write(path.join(name), &original[..original.len() - 1]).unwrap();
            let result = score_rows(&dir.path, &case, 2).unwrap().collect::<Result<Vec<_>, _>>();
            assert_eq!(result.err().as_deref(), Some("truncated native dump"));
            let mut extra = original.clone();
            extra.push(0);
            std::fs::write(path.join(name), extra).unwrap();
            let result = score_rows(&dir.path, &case, 2).unwrap().collect::<Result<Vec<_>, _>>();
            assert_eq!(result.err().as_deref(), Some("native dump size or input IDs differ"));
            std::fs::remove_file(path.join(name)).unwrap();
            assert_eq!(score_rows(&dir.path, &case, 2).err().as_deref(), Some("FileNotFoundError"));
            std::fs::write(path.join(name), original).unwrap();
        }
        std::fs::remove_file(path.join("metadata.json")).unwrap();
        assert_eq!(score_rows(&dir.path, &case, 2).err().as_deref(), Some("FileNotFoundError"));
    }
    #[test]
    fn native_capture_refuses_nonfinite_logits_and_changed_input_ids() {
        let (dir, case, logits, mut inputs) = native_fixture(3);
        let path = dir.path.join("sample");
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = logits.clone();
            bad[..4].copy_from_slice(&value.to_le_bytes());
            std::fs::write(path.join("logits.f32"), bad).unwrap();
            let result = score_rows(&dir.path, &case, 2).unwrap().collect::<Result<Vec<_>, _>>();
            assert_eq!(result.err().as_deref(), Some("nonfinite native logits"));
        }
        std::fs::write(path.join("logits.f32"), logits).unwrap();
        inputs[0] ^= 1;
        std::fs::write(path.join("input_ids.u32"), inputs).unwrap();
        let result = score_rows(&dir.path, &case, 2).unwrap().collect::<Result<Vec<_>, _>>();
        assert_eq!(result.err().as_deref(), Some("native dump size or input IDs differ"));
    }
    #[test]
    fn bucket_boundaries_and_summation() {
        for (i, (lo, hi)) in BUCKETS.iter().copied().enumerate() {
            assert_eq!(bucket(lo), Ok(i));
            assert_eq!(bucket(hi - 1), Ok(i));
        }
        assert!(bucket(131072).is_err());
        assert_eq!(fsum([1e100, 1.0, 1e-100]).unwrap(), 1e100);
        assert_eq!(
            fsum([1.0, 2f64.powi(-53), 2f64.powi(-54)]).unwrap(),
            f64::from_bits(1.0f64.to_bits() + 1)
        );
        assert_eq!(splitlines("a\r\nb\u{85}c\n"), vec!["a", "b", "c"]);
        assert_eq!(
            Logit::Int(9_007_199_254_740_993)
                .difference(&Logit::Int(9_007_199_254_740_992))
                .unwrap(),
            1.0
        );
        assert_eq!(
            Logit::Int(9_007_199_254_740_993).compare(&Logit::Float(9_007_199_254_740_992.0)),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            Logit::Float(-0.0).compare(&Logit::Float(0.0)),
            std::cmp::Ordering::Equal
        );
        let big = "18446744073709551616";
        let next = "18446744073709551617";
        assert_eq!(integer_difference(next, big), "1");
        let a = Logit::Big(big.into());
        let b = Logit::Big(next.into());
        assert_eq!(b.difference(&a).unwrap(), 1.0);
        assert!(
            b.compare(&Logit::Float(18_446_744_073_709_551_616.0))
                .is_gt()
        );
        assert_eq!(TokenId::Big(big.into()).to_string(), big);
        assert!(TokenId::Big(big.into()) > TokenId::Small(i64::MAX));
    }
}
