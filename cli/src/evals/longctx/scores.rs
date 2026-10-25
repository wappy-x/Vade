//! Stream validated score rows from NDJSON or native binary captures.
use super::*;
use crate::sha256::Sha256;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

/// One scored position of a capture: the input token, the top-1 and top-k
/// pairs, the scored pairs and, for native dumps, every logit.
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

/// Streams a case's score rows from NDJSON or a native dump, checking the input
/// IDs against the frozen hash when the stream ends.
pub struct ScoreRows {
    input: ScoreInput,
    position: usize,
    length: usize,
    k: usize,
    ids_hash: Sha256,
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

/// A case's capture metadata: the first NDJSON record, or the native dump's
/// `metadata.json`.
pub fn score_meta(directory: &Path, case: &Value) -> Result<Value, String> {
    let id = string(case, "id")?;
    let path = directory.join(format!("{id}.score.jsonl"));
    if path.exists() {
        read_record(&mut BufReader::new(File::open(path).map_err(io_error)?))
    } else {
        json::read(&directory.join(id).join("metadata.json"))
    }
}

/// Open a case's score rows with top-k width `k`.
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
        ids_hash: Sha256::new(),
        expected_hash: case.get("input_ids_sha256").cloned(),
        finished: false,
    })
}

fn verified_hash(hash: &mut Sha256, expected: &Option<Value>) -> Result<bool, String> {
    let actual = std::mem::replace(hash, Sha256::new()).finalize();
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
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&v| f32::from_le_bytes(v) as f64)
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

#[cfg(test)]
mod tests;
