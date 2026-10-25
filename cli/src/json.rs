//! A small, bounded JSON reader for engine health replies: validate the
//! complete document before using its top-level fields. Evaluation receipts,
//! which must keep their key order and bytes, use `evals::json` instead.
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
enum Value {
    String(String),
    Number(f64),
    Object(BTreeMap<String, Value>),
    Array(Vec<Value>),
    Other,
}
struct Parser<'a> {
    bytes: &'a [u8],
    offset: usize,
}

/// Whether `input` is one complete JSON object.
pub fn valid(input: &str) -> bool {
    fields(input).is_some()
}

/// Whether the nested key `path` exists; array elements are named by index.
pub fn path_exists(input: &str, path: &[&str]) -> bool {
    path_value(input, path).is_some()
}

/// A top-level string field.
pub fn string(input: &str, key: &str) -> Option<String> {
    match fields(input)?.get(key) {
        Some(Value::String(value)) => Some(value.clone()),
        _ => None,
    }
}

/// A top-level number field.
pub fn number(input: &str, key: &str) -> Option<f64> {
    match fields(input)?.get(key) {
        Some(Value::Number(value)) => Some(*value),
        _ => None,
    }
}

/// A string at the nested key `path`.
pub fn path_string(input: &str, path: &[&str]) -> Option<String> {
    if let Value::String(value) = path_value(input, path)? {
        Some(value)
    } else {
        None
    }
}

/// A number at the nested key `path`.
pub fn path_number(input: &str, path: &[&str]) -> Option<f64> {
    if let Value::Number(value) = path_value(input, path)? {
        Some(value)
    } else {
        None
    }
}

fn path_value(input: &str, path: &[&str]) -> Option<Value> {
    let mut value = Value::Object(fields(input)?);
    for key in path {
        value = match value {
            Value::Object(mut fields) => fields.remove(*key)?,
            Value::Array(items) => items.get(key.parse::<usize>().ok()?)?.clone(),
            _ => return None,
        };
    }
    Some(value)
}

fn fields(input: &str) -> Option<BTreeMap<String, Value>> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        offset: 0,
    };
    parser.whitespace();
    let fields = parser.object(0)?;
    parser.whitespace();
    if parser.offset != parser.bytes.len() {
        return None;
    }
    Some(fields)
}

impl Parser<'_> {
    fn whitespace(&mut self) {
        while self
            .bytes
            .get(self.offset)
            .is_some_and(|b| b" \r\n\t".contains(b))
        {
            self.offset += 1;
        }
    }
    fn take(&mut self, byte: u8) -> bool {
        if self.bytes.get(self.offset) == Some(&byte) {
            self.offset += 1;
            true
        } else {
            false
        }
    }
    fn object(&mut self, depth: usize) -> Option<BTreeMap<String, Value>> {
        if depth > 32 || !self.take(b'{') {
            return None;
        }
        self.whitespace();
        let mut fields = BTreeMap::new();
        if self.take(b'}') {
            return Some(fields);
        }
        loop {
            let key = self.quoted()?;
            self.whitespace();
            if !self.take(b':') {
                return None;
            }
            self.whitespace();
            let value = self.value(depth + 1)?;
            if fields.insert(key, value).is_some() {
                return None;
            }
            self.whitespace();
            if self.take(b'}') {
                return Some(fields);
            }
            if !self.take(b',') {
                return None;
            }
            self.whitespace();
        }
    }
    fn value(&mut self, depth: usize) -> Option<Value> {
        if depth > 32 {
            return None;
        }
        match self.bytes.get(self.offset).copied()? {
            b'"' => self.quoted().map(Value::String),
            b'{' => self.object(depth).map(Value::Object),
            b'[' => {
                self.offset += 1;
                self.whitespace();
                let mut items = Vec::new();
                if self.take(b']') {
                    return Some(Value::Array(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.whitespace();
                    if self.take(b']') {
                        return Some(Value::Array(items));
                    }
                    if !self.take(b',') {
                        return None;
                    }
                    self.whitespace();
                }
            }
            b't' | b'f' | b'n' => {
                let word: &[u8] = match self.bytes[self.offset] {
                    b't' => b"true",
                    b'f' => b"false",
                    _ => b"null",
                };
                if self.bytes.get(self.offset..self.offset + word.len())? != word {
                    return None;
                }
                self.offset += word.len();
                Some(Value::Other)
            }
            b'-' | b'0'..=b'9' => self.number().map(Value::Number),
            _ => None,
        }
    }
    fn number(&mut self) -> Option<f64> {
        let start = self.offset;
        self.take(b'-');
        if !self.take(b'0') {
            let b = self.bytes.get(self.offset)?;
            if !(b'1'..=b'9').contains(b) {
                return None;
            }
            self.digits()?;
        }
        if self.take(b'.') {
            self.digits()?;
        }
        if self.take(b'e') || self.take(b'E') {
            if !self.take(b'+') {
                self.take(b'-');
            }
            self.digits()?;
        }
        std::str::from_utf8(&self.bytes[start..self.offset])
            .ok()?
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())
    }
    fn digits(&mut self) -> Option<()> {
        let start = self.offset;
        while self.bytes.get(self.offset).is_some_and(u8::is_ascii_digit) {
            self.offset += 1;
        }
        if self.offset > start { Some(()) } else { None }
    }
    fn quoted(&mut self) -> Option<String> {
        if !self.take(b'"') {
            return None;
        }
        let mut value = Vec::new();
        loop {
            let byte = *self.bytes.get(self.offset)?;
            self.offset += 1;
            match byte {
                b'"' => return String::from_utf8(value).ok(),
                0..=31 => return None,
                b'\\' => {
                    let escaped = *self.bytes.get(self.offset)?;
                    self.offset += 1;
                    match escaped {
                        b'"' | b'\\' | b'/' => value.push(escaped),
                        b'b' => value.push(8),
                        b'f' => value.push(12),
                        b'n' => value.push(10),
                        b'r' => value.push(13),
                        b't' => value.push(9),
                        b'u' => {
                            let first = self.code_unit()?;
                            let scalar = if (0xd800..=0xdbff).contains(&first) {
                                if !self.take(b'\\') || !self.take(b'u') {
                                    return None;
                                }
                                let second = self.code_unit()?;
                                if !(0xdc00..=0xdfff).contains(&second) {
                                    return None;
                                }
                                0x10000
                                    + ((u32::from(first) - 0xd800) << 10)
                                    + (u32::from(second) - 0xdc00)
                            } else {
                                u32::from(first)
                            };
                            let mut buffer = [0; 4];
                            value.extend_from_slice(
                                char::from_u32(scalar)?.encode_utf8(&mut buffer).as_bytes(),
                            );
                        }
                        _ => return None,
                    }
                }
                _ => value.push(byte),
            }
        }
    }
    fn code_unit(&mut self) -> Option<u16> {
        let bytes = self.bytes.get(self.offset..self.offset + 4)?;
        let text = std::str::from_utf8(bytes).ok()?;
        if !bytes.iter().all(u8::is_ascii_hexdigit) {
            return None;
        }
        let result = u16::from_str_radix(text, 16).ok()?;
        self.offset += 4;
        Some(result)
    }
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;
    #[test]
    fn validates_complete_top_level_json() {
        assert_eq!(string(r#"{"status":"re\u0061dy","extra":[1,{},true],"model":"model"}"#, "status").as_deref(), Some("ready"));
        for input in [
            r#"{"nested":{"status":"ready"}}"#, r#"{"status":"ready","status":"busy"}"#,
            r#"{"status":"ready"} trailing"#, r#"{"status":"ready",}"#, r#"{"status":"ready","n":01}"#,
        ] { assert!(string(input, "status").is_none()); }
        assert_eq!(number(r#"{"tok_s":12.5}"#, "tok_s"), Some(12.5));
        assert_eq!(number(r#"{"tok_s":-12.5}"#, "tok_s"), Some(-12.5));
        assert_eq!(number(r#"{"tok_s":1.25e1}"#, "tok_s"), Some(12.5));
        for input in [
            r#"{"tok_s":"12.5"}"#, r#"{"tok_s":1e999}"#, r#"{"tok_s":NaN}"#, r#"{"tok_s":Infinity}"#,
            r#"{"tok_s":01}"#, r#"{"tok_s":1.}"#, r#"{"nested":{"tok_s":12.5}}"#, r#"{"tok_s":12.5} trailing"#,
        ] { assert!(number(input, "tok_s").is_none()); }
    }
}
