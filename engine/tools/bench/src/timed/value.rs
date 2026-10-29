//! Ordered JSON for frozen request hashes: updating a key keeps its position.
use std::fmt::Write;

#[path = "../../../export/src/sha256.rs"]
mod sha256;
pub use sha256::digest;

// Default decimal conversion bound of the retained Python replay interpreter.
pub const PYTHON_INT_DIGITS: usize = 4300;

#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        fn numbers(left: &str, right: &str) -> bool {
            let li = !left.contains(['.', 'e', 'E']);
            let ri = !right.contains(['.', 'e', 'E']);
            match (li, ri) {
                (true, true) => left == right,
                (false, false) => left
                    .parse::<f64>()
                    .ok()
                    .zip(right.parse::<f64>().ok())
                    .is_some_and(|(l, r)| l == r),
                _ => {
                    let (integer, float) = if li { (left, right) } else { (right, left) };
                    float.parse::<f64>().is_ok_and(|f| {
                        f.is_finite()
                            && f.fract() == 0.0
                            && ((f == 0.0 && integer == "0") || integer == format!("{f:.0}"))
                    })
                }
            }
        }
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(l), Self::Bool(r)) => l == r,
            (Self::Number(l), Self::Number(r)) => numbers(l, r),
            (Self::Bool(v), Self::Number(n)) | (Self::Number(n), Self::Bool(v)) => {
                numbers(if *v { "1" } else { "0" }, n)
            }
            (Self::String(l), Self::String(r)) => l == r,
            (Self::Array(l), Self::Array(r)) => l == r,
            (Self::Object(l), Self::Object(r)) => {
                l.len() == r.len()
                    && l.iter().all(|(key, value)| {
                        r.iter().any(|(other_key, other_value)| {
                            key == other_key && value == other_value
                        })
                    })
            }
            _ => false,
        }
    }
}
static NULL: Value = Value::Null;
impl Value {
    pub fn get(&self, key: &str) -> &Value {
        if let Self::Object(items) = self {
            items
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .unwrap_or(&NULL)
        } else {
            &NULL
        }
    }
    pub fn put(&mut self, key: &str, value: Value) {
        if !matches!(self, Self::Object(_)) {
            *self = Self::Object(Vec::new());
        }
        let Self::Object(items) = self else {
            unreachable!()
        };
        if let Some((_, v)) = items.iter_mut().find(|(k, _)| k == key) {
            *v = value;
        } else {
            items.push((key.to_owned(), value));
        }
    }
    pub fn del(&mut self, key: &str) {
        if let Self::Object(items) = self {
            items.retain(|(k, _)| k != key);
        }
    }
    pub fn arr(&self) -> &[Value] {
        if let Self::Array(items) = self {
            items
        } else {
            &[]
        }
    }
    pub fn str(&self) -> Option<&str> {
        if let Self::String(text) = self {
            Some(text)
        } else {
            None
        }
    }
    pub fn num(&self) -> Option<f64> {
        if let Self::Number(text) = self {
            text.parse().ok()
        } else {
            None
        }
    }
    /// Python's nonnegative int shape, without imposing a machine-word limit.
    pub fn nonnegative_integer(&self) -> Option<&str> {
        if let Self::Number(text) = self
            && !text.contains(['.', 'e', 'E', '-'])
        {
            return Some(text);
        }
        None
    }
    /// Strict JSON integer, so bools and integral floats are not token counts.
    pub fn uint(&self) -> Option<u64> {
        self.nonnegative_integer()?.parse().ok()
    }
    pub fn integer_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let left = self.nonnegative_integer()?;
        let right = other.nonnegative_integer()?;
        Some(left.len().cmp(&right.len()).then_with(|| left.cmp(right)))
    }
    /// Python truthiness is also used for optional error objects and deltas.
    pub fn yes(&self) -> bool {
        match self {
            Self::Null => false,
            Self::Bool(v) => *v,
            Self::Number(v) => v.parse::<f64>().is_ok_and(|n| n != 0.0),
            Self::String(v) => !v.is_empty(),
            Self::Array(v) => !v.is_empty(),
            Self::Object(v) => !v.is_empty(),
        }
    }
    pub fn path(&self, path: &str) -> &Value {
        path.split('.').fold(self, |v, key| v.get(key))
    }
}
pub fn o<const N: usize>(items: [(&str, Value); N]) -> Value {
    let mut value = Value::Object(Vec::new());
    for (key, item) in items {
        value.put(key, item);
    }
    value
}
pub fn a(items: Vec<Value>) -> Value {
    Value::Array(items)
}
pub fn s(text: impl Into<String>) -> Value {
    Value::String(text.into())
}
pub fn n(number: impl ToString) -> Value {
    let text = number.to_string();
    // A missing rate is explicit JSON null; nonfinite clocks/rates must never
    // turn an otherwise reviewable receipt into invalid JSON.
    let unsigned = text.trim_start_matches(['+', '-']);
    if ["nan", "inf", "infinity"]
        .iter()
        .any(|v| unsigned.eq_ignore_ascii_case(v))
    {
        Value::Null
    } else {
        Value::Number(text)
    }
}
pub fn b(value: bool) -> Value {
    Value::Bool(value)
}

pub fn encode(value: &Value) -> String {
    fn quoted(out: &mut String, text: &str) {
        out.push('"');
        for ch in text.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\0'..='\u{1f}' => {
                    write!(out, "\\u{:04x}", ch as u32).unwrap();
                }
                _ => out.push(ch),
            }
        }
        out.push('"');
    }
    fn append(out: &mut String, value: &Value) {
        match value {
            Value::Null => out.push_str("null"),
            Value::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
            Value::Number(v) => out.push_str(v),
            Value::String(v) => quoted(out, v),
            Value::Array(items) => {
                out.push('[');
                for (i, value) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    append(out, value);
                }
                out.push(']');
            }
            Value::Object(items) => {
                out.push('{');
                for (i, (key, value)) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    quoted(out, key);
                    out.push(':');
                    append(out, value);
                }
                out.push('}');
            }
        }
    }
    let mut out = String::new();
    append(&mut out, value);
    out
}

// Match Python json.loads/json.dumps float spelling, while retaining integer type.
fn float_spelling(raw: &str) -> Result<String, String> {
    let value = raw.parse::<f64>().map_err(|e| e.to_string())?;
    if !value.is_finite() {
        return Err("non-finite JSON number cannot appear in a benchmark receipt".into());
    }
    if value == 0.0 {
        return Ok(if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .into());
    }
    let magnitude = value.abs();
    let mut rendered = magnitude.to_string();
    // Rust's shortest formatter rounds exact decimal midpoints upward; Python
    // chooses an even last digit. Detect the midpoint using the exact binary
    // significand, without introducing another floating-point rounding.
    if let Some((_, frac)) = rendered.split_once('.') {
        let bits = magnitude.to_bits();
        let encoded_exponent = ((bits >> 52) & 0x7ff) as i32;
        let significand =
            (bits & ((1u64 << 52) - 1)) | if encoded_exponent == 0 { 0 } else { 1u64 << 52 };
        let binary_exponent = if encoded_exponent == 0 {
            -1074
        } else {
            encoded_exponent - 1075
        };
        let last = *rendered.as_bytes().last().unwrap();
        if binary_exponent + frac.len() as i32 + significand.trailing_zeros() as i32 == -1
            && (last - b'0') % 2 == 1
        {
            rendered.pop();
            rendered.push((last - 1) as char);
        }
    }
    let (whole, frac) = rendered.split_once('.').unwrap_or((&rendered, ""));
    let digits = format!("{whole}{frac}");
    let first = digits.find(|c| c != '0').unwrap();
    let exponent = whole.len() as i32 - first as i32 - 1;
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if !(-4..16).contains(&exponent) {
        let significant = digits[first..].trim_end_matches('0');
        let mantissa = if significant.len() == 1 {
            significant.to_owned()
        } else {
            format!("{}.{}", &significant[..1], &significant[1..])
        };
        Ok(format!(
            "{sign}{mantissa}e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        ))
    } else {
        Ok(format!(
            "{sign}{rendered}{}",
            if rendered.contains('.') { "" } else { ".0" }
        ))
    }
}

pub fn parse(bytes: &[u8]) -> Result<Value, String> {
    struct Parser<'a> {
        bytes: &'a [u8],
        at: usize,
    }
    impl Parser<'_> {
        fn space(&mut self) {
            while self
                .bytes
                .get(self.at)
                .is_some_and(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
            {
                self.at += 1;
            }
        }
        fn error(&self, message: &str) -> String {
            format!("{message} at byte {}", self.at)
        }
        fn literal(&mut self, text: &[u8], value: Value) -> Result<Value, String> {
            if self.bytes.get(self.at..self.at + text.len()) != Some(text) {
                return Err(self.error("invalid JSON literal"));
            }
            self.at += text.len();
            Ok(value)
        }
        fn hex(&mut self) -> Result<u16, String> {
            let bytes = self
                .bytes
                .get(self.at..self.at + 4)
                .ok_or_else(|| self.error("truncated Unicode escape"))?;
            let text =
                std::str::from_utf8(bytes).map_err(|_| self.error("invalid Unicode escape"))?;
            let value =
                u16::from_str_radix(text, 16).map_err(|_| self.error("invalid Unicode escape"))?;
            self.at += 4;
            Ok(value)
        }
        fn string(&mut self) -> Result<String, String> {
            if self.bytes.get(self.at) != Some(&b'"') {
                return Err(self.error("expected JSON string"));
            }
            self.at += 1;
            let mut out = Vec::new();
            loop {
                let byte = *self
                    .bytes
                    .get(self.at)
                    .ok_or_else(|| self.error("unterminated JSON string"))?;
                self.at += 1;
                match byte {
                    b'"' => {
                        return String::from_utf8(out)
                            .map_err(|_| self.error("invalid UTF-8 string"));
                    }
                    b'\\' => {
                        let escaped = *self
                            .bytes
                            .get(self.at)
                            .ok_or_else(|| self.error("truncated JSON escape"))?;
                        self.at += 1;
                        match escaped {
                            b'"' | b'\\' | b'/' => out.push(escaped),
                            b'b' => out.push(8),
                            b'f' => out.push(12),
                            b'n' => out.push(b'\n'),
                            b'r' => out.push(b'\r'),
                            b't' => out.push(b'\t'),
                            b'u' => {
                                let first = self.hex()?;
                                let scalar = if (0xd800..=0xdbff).contains(&first) {
                                    if self.bytes.get(self.at..self.at + 2) != Some(b"\\u") {
                                        return Err(self.error("unpaired Unicode surrogate"));
                                    }
                                    self.at += 2;
                                    let second = self.hex()?;
                                    if !(0xdc00..=0xdfff).contains(&second) {
                                        return Err(self.error("unpaired Unicode surrogate"));
                                    }
                                    0x10000 + ((first as u32 - 0xd800) << 10) + second as u32
                                        - 0xdc00
                                } else {
                                    first as u32
                                };
                                let ch = char::from_u32(scalar)
                                    .ok_or_else(|| self.error("invalid Unicode scalar"))?;
                                let mut buf = [0; 4];
                                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                            }
                            _ => return Err(self.error("invalid JSON escape")),
                        }
                    }
                    0..=31 => return Err(self.error("unescaped JSON control character")),
                    _ => out.push(byte),
                }
            }
        }
        fn value(&mut self, depth: usize) -> Result<Value, String> {
            if depth > 512 {
                return Err(self.error("JSON nesting exceeds 512"));
            }
            self.space();
            match self.bytes.get(self.at).copied() {
                Some(b'n') => self.literal(b"null", Value::Null),
                Some(b't') => self.literal(b"true", b(true)),
                Some(b'f') => self.literal(b"false", b(false)),
                Some(b'N' | b'I') => {
                    Err(self.error("non-finite JSON number cannot appear in a benchmark receipt"))
                }
                Some(b'"') => self.string().map(s),
                Some(b'[') => {
                    self.at += 1;
                    self.space();
                    let mut items = Vec::new();
                    if self.bytes.get(self.at) == Some(&b']') {
                        self.at += 1;
                        return Ok(a(items));
                    }
                    loop {
                        items.push(self.value(depth + 1)?);
                        self.space();
                        match self.bytes.get(self.at) {
                            Some(b',') => self.at += 1,
                            Some(b']') => {
                                self.at += 1;
                                return Ok(a(items));
                            }
                            _ => return Err(self.error("expected array comma or closing bracket")),
                        }
                    }
                }
                Some(b'{') => {
                    self.at += 1;
                    self.space();
                    let mut object = o([]);
                    if self.bytes.get(self.at) == Some(&b'}') {
                        self.at += 1;
                        return Ok(object);
                    }
                    loop {
                        self.space();
                        let key = self.string()?;
                        self.space();
                        if self.bytes.get(self.at) != Some(&b':') {
                            return Err(self.error("expected object colon"));
                        }
                        self.at += 1;
                        object.put(&key, self.value(depth + 1)?);
                        self.space();
                        match self.bytes.get(self.at) {
                            Some(b',') => self.at += 1,
                            Some(b'}') => {
                                self.at += 1;
                                return Ok(object);
                            }
                            _ => return Err(self.error("expected object comma or closing brace")),
                        }
                    }
                }
                Some(b'-' | b'0'..=b'9') => {
                    let start = self.at;
                    if self.bytes[self.at] == b'-' {
                        self.at += 1;
                        if self.bytes.get(self.at..self.at + 8) == Some(b"Infinity") {
                            return Err(self.error(
                                "non-finite JSON number cannot appear in a benchmark receipt",
                            ));
                        }
                    }
                    match self.bytes.get(self.at) {
                        Some(b'0') => self.at += 1,
                        Some(b'1'..=b'9') => {
                            while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
                                self.at += 1;
                            }
                        }
                        _ => return Err(self.error("invalid JSON number")),
                    }
                    let mut float = false;
                    if self.bytes.get(self.at) == Some(&b'.') {
                        float = true;
                        self.at += 1;
                        let before = self.at;
                        while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
                            self.at += 1;
                        }
                        if before == self.at {
                            return Err(self.error("invalid JSON fraction"));
                        }
                    }
                    if self
                        .bytes
                        .get(self.at)
                        .is_some_and(|b| matches!(b, b'e' | b'E'))
                    {
                        float = true;
                        self.at += 1;
                        if self
                            .bytes
                            .get(self.at)
                            .is_some_and(|b| matches!(b, b'+' | b'-'))
                        {
                            self.at += 1;
                        }
                        let before = self.at;
                        while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
                            self.at += 1;
                        }
                        if before == self.at {
                            return Err(self.error("invalid JSON exponent"));
                        }
                    }
                    let raw = std::str::from_utf8(&self.bytes[start..self.at]).unwrap();
                    if !float && raw.trim_start_matches('-').len() > PYTHON_INT_DIGITS {
                        return Err(self.error("JSON integer exceeds Python's decimal digit limit"));
                    }
                    Ok(n(if float {
                        float_spelling(raw)?
                    } else if raw == "-0" {
                        "0".into()
                    } else {
                        raw.to_owned()
                    }))
                }
                _ => Err(self.error("expected JSON value")),
            }
        }
    }
    let mut parser = Parser { bytes, at: 0 };
    let value = parser.value(0)?;
    parser.space();
    if parser.at != bytes.len() {
        return Err(parser.error("trailing JSON data"));
    }
    Ok(value)
}
