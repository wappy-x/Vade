//! Small strict JSON reader for bounded model metadata. Numbers remain decimal
//! strings so tensor dimensions and offsets never round through floating point.
use crate::{Result, invalid};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

impl Value {
    pub fn object(&self) -> Result<&BTreeMap<String, Value>> {
        match self {
            Self::Object(v) => Ok(v),
            _ => Err(invalid("expected JSON object")),
        }
    }
    pub fn array(&self) -> Result<&[Value]> {
        match self {
            Self::Array(v) => Ok(v),
            _ => Err(invalid("expected JSON array")),
        }
    }
    pub fn string(&self) -> Result<&str> {
        match self {
            Self::String(v) => Ok(v),
            _ => Err(invalid("expected JSON string")),
        }
    }
    pub fn boolean(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            _ => Err(invalid("expected JSON boolean")),
        }
    }
    pub fn u64(&self) -> Result<u64> {
        match self {
            Self::Number(v) if v.bytes().all(|b| b.is_ascii_digit()) => v
                .parse()
                .map_err(|_| invalid("unsigned integer outside u64 range")),
            _ => Err(invalid("expected nonnegative JSON integer")),
        }
    }
    pub fn f64(&self) -> Result<f64> {
        let Self::Number(v) = self else {
            return Err(invalid("expected JSON number"));
        };
        let n: f64 = v
            .parse()
            .map_err(|_| invalid("invalid floating point number"))?;
        if !n.is_finite() {
            return Err(invalid("non-finite floating point number"));
        }
        Ok(n)
    }
}

pub fn field<'a>(object: &'a BTreeMap<String, Value>, name: &str) -> Result<&'a Value> {
    object
        .get(name)
        .ok_or_else(|| invalid(format!("missing field {name}")))
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub bytes: usize,
    pub depth: usize,
    pub nodes: usize,
    pub string_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            bytes: 16 * 1024 * 1024,
            depth: 32,
            nodes: 200_000,
            string_bytes: 1024 * 1024,
        }
    }
}

pub fn parse(input: &[u8], limits: Limits) -> Result<Value> {
    if input.len() > limits.bytes {
        return Err(invalid("JSON byte limit exceeded"));
    }
    let text = std::str::from_utf8(input).map_err(|_| invalid("JSON is not UTF-8"))?;
    let mut p = Parser {
        input,
        text,
        pos: 0,
        nodes: 0,
        limits,
    };
    let result = p.value(0)?;
    p.whitespace();
    if p.pos != input.len() {
        return Err(p.error("trailing JSON data"));
    }
    Ok(result)
}

struct Parser<'a> {
    input: &'a [u8],
    text: &'a str,
    pos: usize,
    nodes: usize,
    limits: Limits,
}
impl Parser<'_> {
    fn error(&self, message: &str) -> crate::Error {
        invalid(format!("{message} at byte {}", self.pos))
    }
    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }
    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }
    fn consume(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn require(&mut self, byte: u8) -> Result<()> {
        if self.consume(byte) {
            Ok(())
        } else {
            Err(self.error("unexpected JSON character"))
        }
    }
    fn literal(&mut self, bytes: &[u8], value: Value) -> Result<Value> {
        if self.input.get(self.pos..self.pos + bytes.len()) != Some(bytes) {
            return Err(self.error("invalid JSON literal"));
        }
        self.pos += bytes.len();
        Ok(value)
    }
    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > self.limits.depth {
            return Err(self.error("JSON depth limit exceeded"));
        }
        if self.nodes >= self.limits.nodes {
            return Err(self.error("JSON node limit exceeded"));
        }
        self.nodes += 1;
        self.whitespace();
        match self.peek() {
            Some(b'n') => self.literal(b"null", Value::Null),
            Some(b't') => self.literal(b"true", Value::Bool(true)),
            Some(b'f') => self.literal(b"false", Value::Bool(false)),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(b'[') => {
                self.pos += 1;
                self.whitespace();
                let mut out = Vec::new();
                if self.consume(b']') {
                    return Ok(Value::Array(out));
                }
                loop {
                    out.push(self.value(depth + 1)?);
                    self.whitespace();
                    if self.consume(b']') {
                        break;
                    }
                    self.require(b',')?;
                }
                Ok(Value::Array(out))
            }
            Some(b'{') => {
                self.pos += 1;
                self.whitespace();
                let mut out = BTreeMap::new();
                if self.consume(b'}') {
                    return Ok(Value::Object(out));
                }
                loop {
                    self.whitespace();
                    let key = self.string()?;
                    self.whitespace();
                    self.require(b':')?;
                    let value = self.value(depth + 1)?;
                    if out.insert(key, value).is_some() {
                        return Err(self.error("duplicate JSON object key"));
                    }
                    self.whitespace();
                    if self.consume(b'}') {
                        break;
                    }
                    self.require(b',')?;
                }
                Ok(Value::Object(out))
            }
            _ => Err(self.error("expected JSON value")),
        }
    }
    fn number(&mut self) -> Result<Value> {
        let start = self.pos;
        self.consume(b'-');
        if !self.consume(b'0') {
            if !matches!(self.peek(), Some(b'1'..=b'9')) {
                return Err(self.error("invalid JSON number"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if self.consume(b'.') {
            let digits = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == digits {
                return Err(self.error("missing decimal digits"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let digits = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == digits {
                return Err(self.error("missing exponent digits"));
            }
        }
        Ok(Value::Number(
            std::str::from_utf8(&self.input[start..self.pos])
                .unwrap()
                .to_owned(),
        ))
    }
    fn hex4(&mut self) -> Result<u32> {
        let mut value = 0;
        for _ in 0..4 {
            let digit = match self.peek() {
                Some(b'0'..=b'9') => self.input[self.pos] - b'0',
                Some(b'a'..=b'f') => self.input[self.pos] - b'a' + 10,
                Some(b'A'..=b'F') => self.input[self.pos] - b'A' + 10,
                _ => return Err(self.error("invalid Unicode escape")),
            };
            self.pos += 1;
            value = value * 16 + u32::from(digit);
        }
        Ok(value)
    }
    fn string(&mut self) -> Result<String> {
        self.require(b'"')?;
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return Err(self.error("unterminated JSON string")),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(0..=31) => return Err(self.error("unescaped control character")),
                Some(b'\\') => {
                    self.pos += 1;
                    let ch = match self.peek() {
                        Some(b'"') => '"',
                        Some(b'\\') => '\\',
                        Some(b'/') => '/',
                        Some(b'b') => '\u{8}',
                        Some(b'f') => '\u{c}',
                        Some(b'n') => '\n',
                        Some(b'r') => '\r',
                        Some(b't') => '\t',
                        Some(b'u') => {
                            self.pos += 1;
                            let first = self.hex4()?;
                            let scalar = if (0xd800..=0xdbff).contains(&first) {
                                self.require(b'\\')?;
                                self.require(b'u')?;
                                let second = self.hex4()?;
                                if !(0xdc00..=0xdfff).contains(&second) {
                                    return Err(self.error("invalid low surrogate"));
                                }
                                0x10000 + (first - 0xd800) * 1024 + second - 0xdc00
                            } else {
                                first
                            };
                            let ch = char::from_u32(scalar)
                                .ok_or_else(|| self.error("invalid Unicode scalar"))?;
                            out.push(ch);
                            if out.len() > self.limits.string_bytes {
                                return Err(self.error("JSON string limit exceeded"));
                            }
                            continue;
                        }
                        _ => return Err(self.error("invalid JSON escape")),
                    };
                    self.pos += 1;
                    out.push(ch);
                }
                Some(_) => {
                    // The whole input was checked as UTF-8; current position is a character boundary.
                    let ch = self.text[self.pos..].chars().next().unwrap();
                    self.pos += ch.len_utf8();
                    out.push(ch);
                }
            }
            if out.len() > self.limits.string_bytes {
                return Err(self.error("JSON string limit exceeded"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_integer_and_unicode() {
        let v = parse(
            br#"{"n":18446744073709551615,"s":"\uD83D\uDE80"}"#,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(v.object().unwrap()["n"].u64().unwrap(), u64::MAX);
        assert_eq!(v.object().unwrap()["s"].string().unwrap(), "🚀");
    }
    #[test]
    fn malformed_inputs() {
        for s in [
            "",
            "[",
            "{",
            "[1,]",
            "{\"a\":1,}",
            "{\"a\":1,\"a\":2}",
            "01",
            "1.",
            "1e",
            "+1",
            "-.2",
            "true false",
            "\"\\uD800\"",
            "\"\\uDC00\"",
            "\"\\uD800\\u0041\"",
            "\"\\x\"",
            "\"\n\"",
        ] {
            assert!(
                parse(s.as_bytes(), Limits::default()).is_err(),
                "accepted {s}"
            );
        }
        assert!(parse(&[b'"', 0xff, b'"'], Limits::default()).is_err());
    }
    #[test]
    fn bounds() {
        assert!(
            parse(
                b"[[[]]]",
                Limits {
                    depth: 1,
                    ..Limits::default()
                }
            )
            .is_err()
        );
        assert!(
            parse(
                b"[1,2]",
                Limits {
                    nodes: 2,
                    ..Limits::default()
                }
            )
            .is_err()
        );
        assert!(
            parse(
                b"\"abcd\"",
                Limits {
                    string_bytes: 3,
                    ..Limits::default()
                }
            )
            .is_err()
        );
        assert!(
            parse(
                b"null",
                Limits {
                    bytes: 3,
                    ..Limits::default()
                }
            )
            .is_err()
        );
    }
    #[test]
    fn rejects_lossy_integer_access() {
        for s in ["-1", "1.0", "1e2", "18446744073709551616"] {
            assert!(
                parse(s.as_bytes(), Limits::default())
                    .unwrap()
                    .u64()
                    .is_err()
            );
        }
        assert!(parse(b"1e9999", Limits::default()).unwrap().f64().is_err());
    }
}
