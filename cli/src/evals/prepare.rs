//! Frozen long-context inputs. Integer seeding and draws follow Python's
//! `random.Random` MT19937 sequence so existing prompt archives remain reusable.
use super::json::{Value, io_error, object, packed, parse, write_json};
use super::{LongArgs, sha};
use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const LENGTHS: [usize; 2] = [32768, 131072];
const DEPTHS: [usize; 5] = [10, 25, 50, 75, 90];
use super::MODEL;

fn string(value: impl Into<String>) -> Value {
    Value::Str(value.into())
}
fn integer(value: usize) -> Value {
    Value::Int(value as i64)
}
fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, String> {
    value.get(name).ok_or_else(|| "KeyError".into())
}
fn text(value: &Value) -> Result<&str, String> {
    value.as_str().ok_or_else(|| "TypeError".into())
}
fn array(value: &Value) -> Result<&[Value], String> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| "TypeError".into())
}

/// CPython initializes integer seeds using their absolute-value little-endian
/// 32-bit limbs, then init_by_array; getrandbits returns the first draw low.
struct Random {
    state: [u32; 624],
    index: usize,
}
impl Random {
    fn new(seed: &str) -> Self {
        let keys = seed_words(seed);
        let mut result = Self {
            state: [0; 624],
            index: 624,
        };
        result.state[0] = 19650218;
        for i in 1..624 {
            let x = result.state[i - 1];
            result.state[i] = 1812433253u32
                .wrapping_mul(x ^ (x >> 30))
                .wrapping_add(i as u32);
        }
        let (mut i, mut j) = (1, 0);
        for _ in 0..624.max(keys.len()) {
            let x = result.state[i - 1];
            result.state[i] = (result.state[i] ^ (x ^ (x >> 30)).wrapping_mul(1664525))
                .wrapping_add(keys[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                result.state[0] = result.state[623];
                i = 1;
            }
            if j >= keys.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            let x = result.state[i - 1];
            result.state[i] =
                (result.state[i] ^ (x ^ (x >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                result.state[0] = result.state[623];
                i = 1;
            }
        }
        result.state[0] = 0x80000000;
        result
    }
    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^ (y >> 18)
    }
    fn bits(&mut self, count: u32) -> u64 {
        if count <= 32 {
            (self.next() >> (32 - count)) as u64
        } else {
            let low = self.next() as u64;
            low | ((self.next() >> (64 - count)) as u64) << 32
        }
    }
    fn below(&mut self, upper: usize) -> usize {
        let count = usize::BITS - upper.leading_zeros();
        loop {
            let result = self.bits(count) as usize;
            if result < upper {
                return result;
            }
        }
    }
    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.below(i + 1);
            values.swap(i, j);
        }
    }
}

fn seed_words(seed: &str) -> Vec<u32> {
    let mut words = vec![0u32];
    for digit in seed.trim_start_matches('-').bytes() {
        let mut carry = (digit - b'0') as u64;
        for word in &mut words {
            let n = *word as u64 * 10 + carry;
            *word = n as u32;
            carry = n >> 32;
        }
        if carry != 0 {
            words.push(carry as u32);
        }
    }
    words
}
fn seed_decimal(mut words: Vec<u32>) -> String {
    let mut digits = Vec::new();
    while words.iter().any(|w| *w != 0) {
        let mut carry = 0u64;
        for word in words.iter_mut().rev() {
            let n = (carry << 32) | *word as u64;
            *word = (n / 10) as u32;
            carry = n % 10;
        }
        digits.push((b'0' + carry as u8) as char);
    }
    if digits.is_empty() {
        "0".into()
    } else {
        digits.into_iter().rev().collect()
    }
}
fn shifted_seed(seed: &str, offset: usize) -> String {
    let mut words = seed_words(seed);
    let negative = seed.starts_with('-');
    if negative && words.len() == 1 && words[0] as usize <= offset {
        return (offset - words[0] as usize).to_string();
    }
    let mut carry = offset as u64;
    for word in &mut words {
        if negative {
            let current = *word as u64;
            *word = current.wrapping_sub(carry) as u32;
            carry = u64::from(current < carry);
        } else {
            let current = *word as u64 + carry;
            *word = current as u32;
            carry = current >> 32;
        }
    }
    if carry != 0 && !negative {
        words.push(carry as u32);
    }
    let decimal = seed_decimal(words);
    if negative {
        format!("-{decimal}")
    } else {
        decimal
    }
}

struct Tokenizer {
    process: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}
impl Tokenizer {
    fn new(command: &[String]) -> Result<Self, String> {
        let program = command.first().ok_or("CPU tokenizer stopped")?;
        let mut process = Command::new(program)
            .args(&command[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(io_error)?;
        let input = process.stdin.take().ok_or("CPU tokenizer stopped")?;
        let output = BufReader::new(process.stdout.take().ok_or("CPU tokenizer stopped")?);
        Ok(Self {
            process,
            input: Some(input),
            output,
        })
    }
    fn call(&mut self, body: Value) -> Result<Value, String> {
        let input = self.input.as_mut().ok_or("CPU tokenizer stopped")?;
        input
            .write_all(packed(&body).as_bytes())
            .and_then(|_| input.write_all(b"\n"))
            .and_then(|_| input.flush())
            .map_err(io_error)?;
        let mut line = String::new();
        if self.output.read_line(&mut line).map_err(io_error)? == 0 {
            return Err("CPU tokenizer stopped".into());
        }
        let result = parse(&line)?;
        if result.get("error").is_some() {
            return Err("CPU tokenizer rejected input".into());
        }
        Ok(result)
    }
    fn encode(&mut self, input: &str) -> Result<Vec<Value>, String> {
        let result = self.call(object(vec![("text", string(input))]))?;
        Ok(array(field(&result, "ids")?)?.to_vec())
    }
    fn decode(&mut self, ids: &[Value]) -> Result<String, String> {
        let result = self.call(object(vec![
            ("ids", Value::Array(ids.to_vec())),
            ("skip_special", Value::Bool(false)),
        ]))?;
        Ok(text(field(&result, "decoded")?)?.into())
    }
    fn frame(&mut self) -> Result<(String, String), String> {
        let marker = "LONGCTX_CONTENT_MARKER";
        let result = self.call(object(vec![
            (
                "messages",
                Value::Array(vec![object(vec![
                    ("role", string("user")),
                    ("content", string(marker)),
                ])]),
            ),
            // GLM templates have no thinking-off form; frame them at low effort.
            if crate::card::is_glm_model(MODEL) {
                ("reasoning_effort", string("low"))
            } else {
                ("enable_thinking", Value::Bool(false))
            },
        ]))?;
        let framed = text(field(&result, "text")?)?;
        if framed.matches(marker).count() != 1 {
            return Err("tokenizer template did not preserve marker".into());
        }
        let (before, after) = framed.split_once(marker).unwrap();
        Ok((before.into(), after.into()))
    }
    fn close(&mut self) -> Result<(), String> {
        self.input.take();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if self.process.try_wait().map_err(io_error)?.is_some() {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                return Err("CPU tokenizer stopped".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}

fn body(prompt: String, maximum: &Value) -> Value {
    body_for(MODEL, prompt, maximum)
}

/// One frozen request body for `model`'s family. GLM servers refuse a
/// thinking-off ask, so a GLM body carries the low-effort field instead and
/// never either thinking-off field. Other families keep today's exact body.
fn body_for(model: &str, prompt: String, maximum: &Value) -> Value {
    let thinking: Vec<(&str, Value)> = if crate::card::is_glm_model(model) {
        vec![("reasoning_effort", string("low")), ("mtp_depth", Value::Int(0))]
    } else {
        vec![
            ("enable_thinking", Value::Bool(false)),
            (
                "chat_template_kwargs",
                object(vec![("enable_thinking", Value::Bool(false))]),
            ),
        ]
    };
    let mut fields = vec![
        ("model", string(model)),
        ("prompt", string(prompt)),
        ("max_tokens", maximum.clone()),
        ("stream", Value::Bool(false)),
        ("cache", Value::Bool(false)),
        ("draft", string("plain")),
    ];
    fields.extend(thinking);
    fields.extend([
        ("temperature", Value::Int(0)),
        ("top_p", Value::Int(1)),
        ("top_k", Value::Int(0)),
        ("repetition_penalty", Value::Int(1)),
        ("presence_penalty", Value::Int(0)),
        ("frequency_penalty", Value::Int(0)),
    ]);
    object(fields)
}

// Python's str.split recognizes ASCII information separators in addition to
// Unicode White_Space; char::is_whitespace alone omits U+001C..U+001F.
fn whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
fn filler(source: &str, seed: &str, target: usize) -> Result<String, String> {
    let mut random = Random::new(seed);
    let words = source
        .split(whitespace)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if words.is_empty() {
        return Err("frozen source is empty".into());
    }
    let pieces = words
        .chunks(48)
        .map(|w| w.join(" ") + "\n")
        .collect::<Vec<_>>();
    let (mut output, mut size) = (String::new(), 0);
    while size < target * 6 {
        let piece = &pieces[random.below(pieces.len())];
        output.push_str(piece);
        size += piece.chars().count();
    }
    Ok(output)
}
fn fit(codec: &mut Tokenizer, input: &str, tokens: usize) -> Result<String, String> {
    let ids = codec.encode(input)?;
    if ids.len() < tokens {
        return Err("insufficient filler for token budget".into());
    }
    for trim in 0..9 {
        // A negative Python slice retains all but abs(end) items.
        let end = if trim <= tokens {
            tokens - trim
        } else {
            ids.len().saturating_sub(trim - tokens)
        };
        let result = codec.decode(&ids[..end.min(ids.len())])?;
        if !result.contains('\u{fffd}') {
            if codec.encode(&result)?.len() == tokens {
                return Ok(result);
            }
            if trim != 0 {
                let padding = codec.encode(&" x".repeat(64))?;
                let extra = fill_between(codec, &result, "", tokens, &padding)?;
                return Ok(result + &extra);
            }
        }
    }
    Err("cannot find a complete Unicode boundary for token budget".into())
}
fn fill_between(
    codec: &mut Tokenizer,
    before: &str,
    after: &str,
    target: usize,
    filler_ids: &[Value],
) -> Result<String, String> {
    let estimate = target as i64 - codec.encode(&(before.to_owned() + after))?.len() as i64;
    for adjustment in [0, 1, -1, 2, -2, 3, -3, 4, -4, 8, -8, 16, -16] {
        let n = estimate + adjustment;
        if n >= 0 && n as usize <= filler_ids.len() {
            let result = codec.decode(&filler_ids[..n as usize])?;
            if codec.encode(&(before.to_owned() + &result + after))?.len() == target {
                return Ok(result);
            }
        }
    }
    Err("cannot fit exact token budget at a text boundary".into())
}

fn needles(
    codec: &mut Tokenizer,
    source: &str,
    seed: &str,
    target: usize,
    document: usize,
    keys_per_depth: usize,
    maximum: &Value,
) -> Result<Vec<Value>, String> {
    let mut random = Random::new(seed);
    let (prefix, suffix) = codec.frame()?;
    let pad = codec.encode(&filler(source, seed, target)?)?;
    let mut archive = prefix + "Read the archive and retrieve only the requested values.\n";
    let mut records = Vec::new();
    for depth in DEPTHS {
        archive += &fill_between(codec, &archive, "", target * depth / 100, &pad)?;
        for _ in 0..keys_per_depth {
            let key = format!("key_{:016x}", random.bits(64));
            let value = format!("value_{:016x}", random.bits(64));
            archive.push('\n');
            let position = codec.encode(&archive)?.len();
            records.push(object(vec![
                ("depth", integer(depth)),
                ("position", integer(position)),
                ("key", string(&key)),
                ("value", string(&value)),
            ]));
            archive += &format!("Archive entry: {key} = {value}.\n");
        }
    }
    let mut groups = DEPTHS
        .iter()
        .map(|&depth| {
            (
                "needle",
                Some(depth),
                records
                    .iter()
                    .filter(|r| r.get("depth").and_then(Value::as_i64) == Some(depth as i64))
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let mut shuffled = records.clone();
    random.shuffle(&mut shuffled);
    groups.push(("multi", None, shuffled));
    let mut cases = Vec::new();
    for (variant, depth, query) in groups {
        let question = "\nQuestion: Return the values for these keys in this exact order, one value per line. No explanations or key names.\n".to_owned()
            + &query.iter().map(|r| text(field(r, "key")?).map(str::to_owned)).collect::<Result<Vec<_>, _>>()?.join("\n") + "\n";
        let tail = fill_between(codec, &archive, &(question.clone() + &suffix), target, &pad)?;
        let prompt = archive.clone() + &tail + &question + &suffix;
        let actual = codec.encode(&prompt)?.len();
        if actual != target
            || records.iter().any(|r| {
                let position = r.get("position").and_then(Value::as_i64).unwrap_or(0) as f64;
                let depth = r.get("depth").and_then(Value::as_i64).unwrap_or(0) as f64;
                (position - target as f64 * depth / 100.0).abs()
                    > (16 + 128 * keys_per_depth) as f64
            })
        {
            return Err("needle token budget or measured depth differs".into());
        }
        let name = format!(
            "{variant}-{target}-d{document}-{}",
            depth.map_or("all".into(), |d| d.to_string())
        );
        let expected = query
            .iter()
            .map(|r| text(field(r, "value")?).map(str::to_owned))
            .collect::<Result<Vec<_>, _>>()?
            .join("\n");
        cases.push(object(vec![
            ("id", string(name)),
            ("kind", string(variant)),
            ("length", integer(target)),
            ("document", integer(document)),
            ("body", body(prompt, maximum)),
            ("records", Value::Array(records.clone())),
            ("query", Value::Array(query)),
            ("expected", string(expected)),
        ]));
    }
    Ok(cases)
}

fn frozen_sources(
    root: &Path,
    instruction_root: &Path,
    documents: usize,
    codec: &mut Tokenizer,
) -> Result<Vec<(String, Value)>, String> {
    let mut paths = vec![root.join("pp448.body"), root.join("code.body")];
    let mut instructions = Vec::new();
    if instruction_root.is_file() {
        let mut seen = HashSet::new();
        let data = fs::read_to_string(instruction_root).map_err(io_error)?;
        // TextIO's default newline handling translates both CRLF and lone CR.
        let data = data.replace("\r\n", "\n").replace('\r', "\n");
        for line in data.split_inclusive('\n') {
            let record = parse(line)?;
            let mut obj = match record.get("body").and_then(Value::as_str) {
                Some(body) => parse(body)?,
                None => record,
            };
            if let Some(prompt) = obj
                .get("doc")
                .and_then(|v| v.get("prompt"))
                .and_then(Value::as_str)
            {
                obj = object(vec![("prompt", string(prompt))]);
            }
            let content = packed(
                obj.get("messages")
                    .or_else(|| obj.get("prompt"))
                    .unwrap_or(&Value::Null),
            );
            if !seen.insert(content) {
                continue;
            }
            instructions.push((
                obj,
                object(vec![
                    (
                        "name",
                        string(
                            instruction_root
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy(),
                        ),
                    ),
                    ("sha256", string(sha(line.as_bytes()))),
                ]),
            ));
            if instructions.len() == documents - 2 {
                break;
            }
        }
    } else {
        let mut found = if instruction_root.exists() {
            fs::read_dir(instruction_root)
                .map_err(io_error)?
                .filter_map(|entry| entry.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name().is_some_and(|name| {
                        let name = name.to_string_lossy();
                        name.starts_with("instruction-ifeval-") && name.ends_with(".json")
                    })
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        found.sort();
        paths.extend(found.into_iter().take(documents - 2));
    }
    if paths.len() + instructions.len() < documents {
        return Err("need frozen prose/code bodies and at least two IFEval inputs".into());
    }
    let mut inputs = Vec::new();
    for path in paths {
        let bytes = fs::read(&path).map_err(io_error)?;
        let data = std::str::from_utf8(&bytes).map_err(|_| "UnicodeDecodeError")?;
        inputs.push((
            parse(data)?,
            object(vec![
                (
                    "name",
                    string(path.file_name().unwrap_or_default().to_string_lossy()),
                ),
                ("sha256", string(sha(&bytes))),
            ]),
        ));
    }
    inputs.extend(instructions);
    let mut sources = Vec::new();
    for (obj, origin) in inputs {
        let mut source = match obj.get("prompt").and_then(Value::as_str) {
            Some(prompt) => prompt.to_owned(),
            None => obj
                .get("messages")
                .and_then(Value::as_array)
                .map(|messages| {
                    messages
                        .iter()
                        .filter_map(|m| m.get("content").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                })
                .unwrap_or_default(),
        };
        if source.is_empty()
            && let Some(ids) = obj.get("prompt_ids") {
            source = codec.decode(array(ids)?)?;
        }
        if source.is_empty() {
            return Err("frozen input has no text".into());
        }
        sources.push((source, origin));
    }
    Ok(sources)
}

/// `vade test longctx prepare`: build the frozen inputs and return the manifest.
pub fn run(args: &LongArgs) -> Result<Value, String> {
    if !(1..=8).contains(&args.keys_per_depth) || args.max_tokens < 64 * args.keys_per_depth {
        return Err("K must be 1..8 with at least 64*K output tokens".into());
    }
    if args.output.exists() {
        return Err("FileExistsError".into());
    }
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent).map_err(io_error)?;
    }
    fs::create_dir(&args.output).map_err(io_error)?;
    let mut codec = Tokenizer::new(&args.tokenizer_command)?;
    let result = prepare(args, &mut codec);
    let closed = codec.close();
    closed.and(result)
}
fn prepare(args: &LongArgs, codec: &mut Tokenizer) -> Result<Value, String> {
    let sources = frozen_sources(
        &args.frozen_bodies,
        &args.ifeval_inputs,
        args.documents,
        codec,
    )?;
    let mut cases = Vec::new();
    for length in LENGTHS {
        for (document, (source, origin)) in sources.iter().enumerate() {
            let seed = shifted_seed(&args.seed, document * 1009 + length);
            let prompt = fit(codec, &filler(source, &seed, length)?, length)?;
            cases.push(object(vec![
                ("id", string(format!("teacher-{length}-d{document}"))),
                ("kind", string("teacher")),
                ("length", integer(length)),
                ("document", integer(document)),
                ("body", body(prompt, &Value::Int(0))),
                ("origin", origin.clone()),
            ]));
        }
        for (document, (source, _)) in sources.iter().take(4).enumerate() {
            let seed = shifted_seed(&args.seed, document * 1009 + length);
            cases.extend(needles(
                codec,
                source,
                &seed,
                length,
                document,
                args.keys_per_depth,
                &args.max_tokens_value,
            )?);
        }
    }
    let mut entries = Vec::new();
    for case in cases {
        let Value::Object(mut fields) = case else {
            return Err("TypeError".into());
        };
        let at = fields
            .iter()
            .position(|(k, _)| k == "body")
            .ok_or("KeyError")?;
        let request = fields.remove(at).1;
        let prompt = text(field(&request, "prompt")?)?;
        let prompt_ids = codec.encode(prompt)?;
        let length = fields
            .iter()
            .find(|(k, _)| k == "length")
            .and_then(|(_, v)| v.as_i64())
            .ok_or("KeyError")? as usize;
        if prompt_ids.len() != length {
            return Err(
                "engine special-token encoding changes the requested context length".into(),
            );
        }
        let id = fields
            .iter()
            .find(|(k, _)| k == "id")
            .and_then(|(_, v)| v.as_str())
            .ok_or("KeyError")?;
        let filename = format!("{id}.body.json");
        let mut bytes = Vec::with_capacity(prompt_ids.len() * 4);
        for token in prompt_ids {
            let id = token
                .as_i64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or("error")?;
            bytes.extend(id.to_le_bytes());
        }
        fields.extend([
            ("body_file".into(), string(&filename)),
            ("prompt_sha256".into(), string(sha(prompt.as_bytes()))),
            ("input_ids_sha256".into(), string(sha(&bytes))),
        ]);
        write_json(&args.output.join(filename), &request)?;
        entries.push(Value::Object(fields));
    }
    let seed = args
        .seed
        .parse::<i64>()
        .map_or_else(|_| Value::BigInt(args.seed.clone()), Value::Int);
    let manifest = object(vec![
        ("schema", string("vade.longctx.v1")),
        ("seed", seed),
        (
            "lengths",
            Value::Array(LENGTHS.into_iter().map(integer).collect()),
        ),
        ("max_tokens", args.max_tokens_value.clone()),
        ("documents", integer(args.documents)),
        ("keys_per_depth", integer(args.keys_per_depth)),
        ("checkpoint_sha256", string(&args.checkpoint_hash)),
        ("cases", Value::Array(entries)),
    ]);
    write_json(&args.output.join("manifest.json"), &manifest)?;
    Ok(object(vec![
        ("status", string("READY")),
        ("teacher_documents", integer(args.documents * 2)),
        ("retrieval_cases", integer(48)),
        ("manifest", string("manifest.json")),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn seeded_archive_draws() {
        let mut random = Random::new("42");
        assert_eq!(random.bits(64), 0x1c80317fa3b1799d);
        assert_eq!(random.bits(64), 0xbdd640fb06671ad1);
        let mut order = (0..10).collect::<Vec<_>>();
        Random::new("42").shuffle(&mut order);
        assert_eq!(order, [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]);
        assert_eq!(
            filler("α β γ\u{1c}δ", "42", 2).unwrap(),
            "α β γ δ\nα β γ δ\n"
        );
        assert_eq!(shifted_seed("-4294967296", 131072), "-4294836224");
        assert_eq!(shifted_seed("-1", 32768), "32767");
        assert_eq!(
            Random::new("12345678901234567890123456789012345678901234567890").bits(64),
            0xf4b4b002e4c6c05f
        );
    }
}

#[cfg(test)]
mod greedy_request_tests {
    use super::*;
    #[test]
    fn greedy_prepared_requests_and_capture_options_omit_seed() {
        let request = body("fixture".into(), &Value::Int(8));
        assert_eq!(request["model"].as_str(), Some("qwen3.8-flash-next-fp8"));
        assert_eq!(request["temperature"], Value::Int(0));
        assert!(request.get("seed").is_none());
        // Qwen keeps today's exact thinking-off fields.
        assert_eq!(packed(&request), r#"{"model":"qwen3.8-flash-next-fp8","prompt":"fixture","max_tokens":8,"stream":false,"cache":false,"draft":"plain","enable_thinking":false,"chat_template_kwargs":{"enable_thinking":false},"temperature":0,"top_p":1,"top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0}"#);
        // A GLM body never asks thinking off; it carries low effort instead.
        let glm = body_for("glm-5.3-flash-fp4-local", "fixture".into(), &Value::Int(8));
        assert_eq!(packed(&glm), r#"{"model":"glm-5.3-flash-fp4-local","prompt":"fixture","max_tokens":8,"stream":false,"cache":false,"draft":"plain","reasoning_effort":"low","mtp_depth":0,"temperature":0,"top_p":1,"top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0}"#);
        let options = super::super::sampling();
        assert!(options.iter().any(|(key, value)| *key == "temperature" && *value == Value::Int(0)));
        assert!(options.iter().all(|(key, _)| *key != "seed"));
    }
}
