//! Ordered JSON for preserving evaluation receipt bytes.
use std::path::Path;
/// The longest integer accepted, matching Python's default string-conversion
/// limit.
pub const MAX_INTEGER_DIGITS: usize = 4300;

/// A JSON value. Objects keep their key order, and integers too large for `i64`
/// keep their exact digits. Serialization uses the receipt format's spelling;
/// parsing does not preserve input whitespace, escapes or repeated object keys.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    BigInt(String),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}
impl Value {
    /// An object's field.
    pub fn get(&self, key: &str) -> Option<&Self> {
        if let Self::Object(fields) = self {
            fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        } else {
            None
        }
    }
    /// The string, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        if let Self::Str(v) = self {
            Some(v)
        } else {
            None
        }
    }
    /// The integer, if this is one that fits in `i64`.
    pub fn as_i64(&self) -> Option<i64> {
        if let Self::Int(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    /// Any number as `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::BigInt(v) => v.parse().ok(),
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }
    /// The elements, if this is an array.
    pub fn as_array(&self) -> Option<&Vec<Self>> {
        if let Self::Array(v) = self {
            Some(v)
        } else {
            None
        }
    }
    /// Set an object field in place, or append it; does nothing on non-objects.
    pub fn insert(&mut self, key: impl Into<String>, value: Self) {
        let key = key.into();
        if let Self::Object(fields) = self {
            if let Some((_, old)) = fields.iter_mut().find(|(k, _)| k == &key) {
                *old = value;
            } else {
                fields.push((key, value));
            }
        }
    }
}
impl std::ops::Index<&str> for Value {
    type Output = Value;
    fn index(&self, key: &str) -> &Value {
        self.get(key).unwrap_or(&Value::Null)
    }
}
impl std::ops::Index<usize> for Value {
    type Output = Value;
    fn index(&self, key: usize) -> &Value {
        self.as_array()
            .and_then(|a| a.get(key))
            .unwrap_or(&Value::Null)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Self::Str(v.into())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}
impl From<usize> for Value {
    fn from(v: usize) -> Self {
        Self::Int(v as i64)
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
/// An object with these fields in this order.
pub fn object(fields: Vec<(&str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
/// Python `==` on decoded JSON: object key order is ignored, and numbers
/// (including booleans) compare by value.
pub fn equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, v)| b.iter().any(|(k, w)| key == k && equal(v, w)))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(v, w)| equal(v, w))
        }
        _ => scalar_equal(a, b),
    }
}
fn scalar_equal(a: &Value, b: &Value) -> bool {
    let integer = |v: &Value| match v {
        Value::Int(n) => Some(n.to_string()),
        Value::BigInt(n) => Some(n.clone()),
        Value::Bool(n) => Some(i64::from(*n).to_string()),
        _ => None,
    };
    match (integer(a), integer(b)) {
        (Some(a), Some(b)) => a == b,
        (Some(integer), None) | (None, Some(integer)) => {
            let real = if matches!(a, Value::Float(_)) { a } else { b };
            real.as_f64().is_some_and(|real| {
                real.is_finite()
                    && real.fract() == 0.
                    && integer
                        == if real == 0. {
                            "0".into()
                        } else {
                            format!("{real:.0}")
                        }
            })
        }
        (None, None) => a == b,
    }
}
// Decimal blocks accepted by the reference interpreter's Unicode 16 parser.
fn decimal_digit(c: char) -> Option<u32> {
    if c.is_ascii_digit() {
        return Some(c as u32 - '0' as u32);
    }
    const ZEROES: &[u32] = &[
        0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66,
        0xde6, 0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90,
        0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0,
        0xff10, 0x104a0, 0x10d30, 0x10d40, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450,
        0x114d0, 0x11650, 0x116c0, 0x116d0, 0x116da, 0x11730, 0x118e0, 0x11950, 0x11bf0, 0x11c50,
        0x11d50, 0x11da0, 0x11f50, 0x16130, 0x16a60, 0x16ac0, 0x16b50, 0x16d70, 0x1ccf0, 0x1d7ce,
        0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e5f1, 0x1e950, 0x1fbf0,
    ];
    ZEROES.iter().find_map(|&zero| {
        let code = c as u32;
        if (zero..zero + 10).contains(&code) {
            Some(code - zero)
        } else {
            None
        }
    })
}
/// Python `int(str)` for a command-line value: optional sign, Unicode decimal
/// digits and single underscores between digits. Returns canonical digits.
pub fn integer_input(value: &str) -> Result<String, String> {
    let value = value.trim();
    let (negative, value) = if let Some(value) = value.strip_prefix('-') {
        (true, value)
    } else {
        (false, value.strip_prefix('+').unwrap_or(value))
    };
    let mut digits = String::new();
    let mut previous_digit = false;
    for c in value.chars() {
        if c == '_' {
            if !previous_digit {
                return Err("invalid int value".into());
            }
            previous_digit = false;
        } else if let Some(digit) = decimal_digit(c) {
            digits.push((b'0' + digit as u8) as char);
            previous_digit = true;
        } else {
            return Err("invalid int value".into());
        }
    }
    if !previous_digit || digits.len() > MAX_INTEGER_DIGITS {
        return Err("invalid int value".into());
    }
    let digits = digits.trim_start_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    Ok(format!(
        "{}{digits}",
        if negative && digits != "0" { "-" } else { "" }
    ))
}
/// Python `float(str)` for a command-line value, including underscores between
/// digits and Unicode decimal digits.
pub fn float_input(value: &str) -> Result<f64, String> {
    let chars: Vec<_> = value.trim().chars().collect();
    let mut normalized = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if c == '_' {
            if i == 0
                || i + 1 == chars.len()
                || decimal_digit(chars[i - 1]).is_none()
                || decimal_digit(chars[i + 1]).is_none()
            {
                return Err("invalid float value".into());
            }
        } else if let Some(digit) = decimal_digit(c) {
            normalized.push((b'0' + digit as u8) as char);
        } else {
            normalized.push(c);
        }
    }
    normalized.parse().map_err(|_| "invalid float value".into())
}
/// A number or boolean as `f64`.
pub fn numeric(value: &Value) -> Option<f64> {
    match value {
        Value::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
        _ => value.as_f64(),
    }
}
/// Python truthiness of a decoded value.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(v) => *v,
        Value::Str(v) => !v.is_empty(),
        Value::BigInt(v) => v != "0",
        Value::Int(v) => *v != 0,
        Value::Float(v) => *v != 0.,
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
    }
}
/// The Python exception name for an I/O error, as receipts record it.
pub fn io_error(error: std::io::Error) -> String {
    match error.kind() {
        std::io::ErrorKind::NotFound => "FileNotFoundError",
        std::io::ErrorKind::AlreadyExists => "FileExistsError",
        std::io::ErrorKind::PermissionDenied => "PermissionError",
        std::io::ErrorKind::IsADirectory => "IsADirectoryError",
        std::io::ErrorKind::NotADirectory => "NotADirectoryError",
        std::io::ErrorKind::ConnectionRefused => "ConnectionRefusedError",
        std::io::ErrorKind::ConnectionReset => "ConnectionResetError",
        std::io::ErrorKind::ConnectionAborted => "ConnectionAbortedError",
        std::io::ErrorKind::BrokenPipe => "BrokenPipeError",
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => "TimeoutError",
        std::io::ErrorKind::UnexpectedEof => "IncompleteRead",
        _ => "OSError",
    }
    .into()
}
/// Read and parse a JSON file.
pub fn read(path: &Path) -> Result<Value, String> {
    parse(&std::fs::read_to_string(path).map_err(io_error)?)
}
/// Create a new file (never overwrite) holding one packed JSON line.
pub fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    use std::io::Write;
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(io_error)?;
    validate(value)?;
    writeln!(out, "{}", packed(value)).map_err(io_error)
}
/// Refuse NaN and infinities, which JSON cannot represent.
pub fn validate(value: &Value) -> Result<(), String> {
    match value {
        Value::Float(v) if !v.is_finite() => Err(format!(
            "Out of range float values are not JSON compliant: {}",
            if v.is_nan() {
                "nan"
            } else if v.is_sign_negative() {
                "-inf"
            } else {
                "inf"
            }
        )),
        Value::Array(items) => {
            for item in items {
                validate(item)?;
            }
            Ok(())
        }
        Value::Object(items) => {
            for (_, item) in items {
                validate(item)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
/// Python's `repr(float)`: shortest round-trip digits, exponent form below 1e-4
/// and from 1e16.
pub fn float(v: f64) -> String {
    // Rust and Python use shortest-roundtrip decimal formatting. Python switches
    // to exponent notation below 1e-4 or at 1e16 and pads the signed exponent.
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let raw = half_even_decimal(v, format!("{v:?}"));
    let (mantissa, power) = if let Some((m, e)) = raw.split_once('e') {
        (m.to_string(), e.parse::<i32>().unwrap())
    } else {
        let sign = if raw.starts_with('-') { "-" } else { "" };
        let unsigned = raw.trim_start_matches('-');
        let point = unsigned.find('.').unwrap_or(unsigned.len()) as i32;
        let digits: String = unsigned.chars().filter(|c| *c != '.').collect();
        let first = digits.find(|c: char| c != '0').unwrap_or(0);
        let significant = digits[first..].trim_end_matches('0');
        let m = if significant.len() > 1 {
            format!("{sign}{}.{}", &significant[..1], &significant[1..])
        } else {
            format!("{sign}{significant}")
        };
        (m, point - first as i32 - 1)
    };
    if !(-4..16).contains(&power) {
        format!("{mantissa}e{power:+03}")
    } else if raw.contains('e') {
        let sign = if mantissa.starts_with('-') { "-" } else { "" };
        let digits = mantissa.trim_start_matches('-').replace('.', "");
        let point = power + 1;
        if point <= 0 {
            format!("{sign}0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!(
                "{sign}{digits}{}.0",
                "0".repeat(point as usize - digits.len())
            )
        } else {
            format!(
                "{sign}{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        }
    } else {
        raw
    }
}
fn half_even_decimal(value: f64, raw: String) -> String {
    // At exact decimal midpoints Rust's shortest formatting can choose an odd
    // final digit. Python chooses the even neighbor. Compare the exact binary
    // rational with that midpoint without rounding a large scaled float.
    let (mantissa, exponent) = raw.split_once('e').map_or((raw.as_str(), 0), |(m, e)| {
        (m, e.parse::<i32>().unwrap_or(0))
    });
    let mantissa = mantissa.trim_start_matches('-');
    let places = mantissa.split_once('.').map_or(0, |(_, s)| s.len() as i32);
    let digits = mantissa.replace('.', "");
    let Ok(decimal) = digits.parse::<u128>() else {
        return raw;
    };
    if decimal % 2 == 0 {
        return raw;
    }
    let scale = exponent - places;
    let bits = value.abs().to_bits();
    let encoded = ((bits >> 52) & 0x7ff) as i32;
    let significand = (bits & ((1u64 << 52) - 1)) | if encoded == 0 { 0 } else { 1u64 << 52 };
    let binary = if encoded == 0 {
        -1074
    } else {
        encoded - 1023 - 52
    };
    if significand == 0 || binary + 1 - scale + significand.trailing_zeros() as i32 != 0 {
        return raw;
    }
    let odd = (significand >> significand.trailing_zeros()) as u128;
    let midpoint = 2 * decimal - 1;
    let equal = if scale < 0 {
        let mut lhs = Some(odd);
        for _ in 0..-scale {
            lhs = lhs.and_then(|n| n.checked_mul(5));
            if lhs.is_none() {
                break;
            }
        }
        lhs == Some(midpoint)
    } else {
        let mut rhs = Some(midpoint);
        for _ in 0..scale {
            rhs = rhs.and_then(|n| n.checked_mul(5));
            if rhs.is_none() {
                break;
            }
        }
        rhs == Some(odd)
    };
    if !equal {
        return raw;
    }
    let mut bytes = raw.into_bytes();
    let end = bytes.iter().position(|c| *c == b'e').unwrap_or(bytes.len());
    if let Some(index) = (0..end).rev().find(|&i| bytes[i].is_ascii_digit()) {
        bytes[index] -= 1;
    }
    String::from_utf8(bytes).unwrap()
}
/// Compact JSON with no spaces and non-ASCII text kept as is.
pub fn packed(value: &Value) -> String {
    let mut out = String::new();
    encode(value, &mut out);
    out
}
fn quoted(value: &str, out: &mut String) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if c < ' ' => {
                use std::fmt::Write;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
fn encode(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
        Value::Int(v) => out.push_str(&v.to_string()),
        Value::BigInt(v) => out.push_str(v),
        Value::Float(v) => out.push_str(&float(*v)),
        Value::Str(v) => quoted(v, out),
        Value::Array(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                encode(v, out);
            }
            out.push(']');
        }
        Value::Object(items) => {
            out.push('{');
            for (i, (k, v)) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                quoted(k, out);
                out.push(':');
                encode(v, out);
            }
            out.push('}');
        }
    }
}
/// Parse one JSON document with Python's `json.loads` rules and error text;
/// a repeated key keeps the last value in the first key's position.
pub fn parse(input: &str) -> Result<Value, String> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    let result = parser.value(0).and_then(|v| {
        parser.space();
        if parser.pos != parser.bytes.len() {
            Err("Extra data".into())
        } else {
            Ok(v)
        }
    });
    result.map_err(|message: String| {
        if message.starts_with("Exceeds the limit (") {
            return message;
        }
        let prefix = String::from_utf8_lossy(&parser.bytes[..parser.pos]);
        let chars = prefix.chars().count();
        let line = prefix.chars().filter(|c| *c == '\n').count() + 1;
        let column = prefix.rsplit('\n').next().unwrap_or("").chars().count() + 1;
        format!("{message}: line {line} column {column} (char {chars})")
    })
}
struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl Parser<'_> {
    fn space(&mut self) {
        while self
            .bytes
            .get(self.pos)
            .is_some_and(|c| b" \t\r\n".contains(c))
        {
            self.pos += 1;
        }
    }
    fn take(&mut self, c: u8) -> bool {
        if self.bytes.get(self.pos) == Some(&c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > 256 {
            return Err("JSON nesting limit".into());
        }
        self.space();
        match self.bytes.get(self.pos).copied() {
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b'{') => {
                self.pos += 1;
                self.space();
                let mut v = Value::Object(Vec::new());
                if self.take(b'}') {
                    return Ok(v);
                }
                loop {
                    let k = self.string()?;
                    self.space();
                    if !self.take(b':') {
                        return Err("Expecting ':' delimiter".into());
                    }
                    let item = self.value(depth + 1)?;
                    v.insert(k, item);
                    self.space();
                    if self.take(b'}') {
                        return Ok(v);
                    }
                    if !self.take(b',') {
                        return Err("Expecting ',' delimiter".into());
                    }
                    self.space();
                }
            }
            Some(b'[') => {
                self.pos += 1;
                self.space();
                let mut v = Vec::new();
                if self.take(b']') {
                    return Ok(Value::Array(v));
                }
                loop {
                    v.push(self.value(depth + 1)?);
                    self.space();
                    if self.take(b']') {
                        return Ok(Value::Array(v));
                    }
                    if !self.take(b',') {
                        return Err("Expecting ',' delimiter".into());
                    }
                }
            }
            Some(b't') => self.literal(b"true", Value::Bool(true)),
            Some(b'f') => self.literal(b"false", Value::Bool(false)),
            Some(b'n') => self.literal(b"null", Value::Null),
            Some(b'N') => self.literal(b"NaN", Value::Float(f64::NAN)),
            Some(b'I') => self.literal(b"Infinity", Value::Float(f64::INFINITY)),
            Some(b'-') if self.bytes.get(self.pos..self.pos + 9) == Some(b"-Infinity") => {
                self.literal(b"-Infinity", Value::Float(f64::NEG_INFINITY))
            }
            Some(b'-' | b'0'..=b'9') => {
                let start = self.pos;
                self.take(b'-');
                if !self.take(b'0') {
                    let before = self.pos;
                    while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                        self.pos += 1;
                    }
                    if before == self.pos {
                        return Err("Expecting value".into());
                    }
                }
                let mut real = false;
                if self.take(b'.') {
                    real = true;
                    self.digits()?;
                }
                if self.take(b'e') || self.take(b'E') {
                    real = true;
                    if !self.take(b'+') {
                        self.take(b'-');
                    }
                    self.digits()?;
                }
                let s = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap();
                if !real {
                    let digits = s.trim_start_matches('-').len();
                    if digits > MAX_INTEGER_DIGITS {
                        return Err(format!(
                            "Exceeds the limit ({MAX_INTEGER_DIGITS} digits) for integer string conversion: value has {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
                        ));
                    }
                    if let Ok(n) = s.parse() {
                        return Ok(Value::Int(n));
                    }
                    return Ok(Value::BigInt(s.into()));
                }
                Ok(Value::Float(s.parse().map_err(|_| "invalid number")?))
            }
            _ => Err("Expecting value".into()),
        }
    }
    fn literal(&mut self, s: &[u8], v: Value) -> Result<Value, String> {
        if self.bytes.get(self.pos..self.pos + s.len()) != Some(s) {
            return Err("Expecting value".into());
        }
        self.pos += s.len();
        Ok(v)
    }
    fn digits(&mut self) -> Result<(), String> {
        let start = self.pos;
        while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
            self.pos += 1;
        }
        if self.pos == start {
            Err("invalid number".into())
        } else {
            Ok(())
        }
    }
    fn string(&mut self) -> Result<String, String> {
        if !self.take(b'"') {
            return Err("Expecting property name enclosed in double quotes".into());
        }
        let mut out = Vec::new();
        loop {
            let b = *self.bytes.get(self.pos).ok_or("Unterminated string")?;
            self.pos += 1;
            match b {
                b'"' => return String::from_utf8(out).map_err(|_| "invalid UTF-8".into()),
                0..=31 => return Err("Invalid control character".into()),
                b'\\' => {
                    let c = *self.bytes.get(self.pos).ok_or("Invalid escape")?;
                    self.pos += 1;
                    match c {
                        b'"' | b'\\' | b'/' => out.push(c),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'n' => out.push(10),
                        b'r' => out.push(13),
                        b't' => out.push(9),
                        b'u' => {
                            let first = self.unit()?;
                            let scalar = if (0xd800..=0xdbff).contains(&first) {
                                if !self.take(b'\\') || !self.take(b'u') {
                                    return Err("invalid Unicode surrogate".into());
                                }
                                let second = self.unit()?;
                                if !(0xdc00..=0xdfff).contains(&second) {
                                    return Err("invalid Unicode surrogate".into());
                                }
                                0x10000 + ((first as u32 - 0xd800) << 10) + (second as u32 - 0xdc00)
                            } else {
                                first as u32
                            };
                            let mut buf = [0; 4];
                            out.extend_from_slice(
                                char::from_u32(scalar)
                                    .ok_or("invalid Unicode scalar")?
                                    .encode_utf8(&mut buf)
                                    .as_bytes(),
                            );
                        }
                        _ => return Err("Invalid escape".into()),
                    }
                }
                _ => out.push(b),
            }
        }
    }
    fn unit(&mut self) -> Result<u16, String> {
        let s = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or("Invalid Unicode escape")?;
        let text = std::str::from_utf8(s).map_err(|_| "Invalid Unicode escape")?;
        let n = u16::from_str_radix(text, 16).map_err(|_| "Invalid Unicode escape")?;
        self.pos += 4;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn receipt_order_unicode_and_decimal_boundaries() {
        let source = r#"{"z":1,"a":"\uD83D\uDE00\n","z":2,"big":18446744073709551616}"#;
        assert_eq!(
            packed(&parse(source).unwrap()),
            "{\"z\":2,\"a\":\"😀\\n\",\"big\":18446744073709551616}"
        );
        // The exact binary value, whose shortest round-trip text is shorter.
        #[allow(clippy::excessive_precision)]
        for (v, text) in [
            (-887777373534812.25, "-887777373534812.2"),
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (-0.0, "-0.0"),
        ] {
            assert_eq!(float(v), text);
        }
        assert!(truthy(&Value::Str("0".into())));
        assert!(!equal(
            &Value::BigInt("18446744073709551617".into()),
            &Value::Float(18446744073709551616.0)
        ));
        assert!(equal(&Value::Bool(false), &Value::Float(-0.0)));
        assert_eq!(
            parse("{").unwrap_err(),
            "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
        );
    }
}
