//! First-party, bounded JSON parsing. Numbers retain their original spelling.
//! Implements the JSON grammar described by RFC 8259; no external parser code.
use std::{borrow::Cow, io};

pub const MAX_INPUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_DEPTH: usize = 64;
const MAX_VALUES: usize = 100_000;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, name: &str) -> Option<&Value> {
        match self {
            Self::Object(fields) => fields.iter().find(|(key, _)| key == name).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub fn parse(input: &str) -> io::Result<Value> {
    if input.len() > MAX_INPUT_BYTES {
        return Err(invalid("JSON exceeds size limit"));
    }
    let mut parser = Parser {
        input: input.as_bytes(),
        pos: 0,
        values: 0,
    };
    let value = parser.value(0)?;
    parser.space();
    if parser.pos != input.len() {
        return Err(invalid("trailing JSON data"));
    }
    Ok(value)
}

pub fn parse_bytes(input: &[u8]) -> io::Result<Value> {
    let input = std::str::from_utf8(input).map_err(|_| invalid("JSON is not UTF-8"))?;
    parse(input)
}

/// A fully validated JSON span. Lookups borrow wire bytes, with no value tree
/// or copied plain text. Escaped strings need temporary decoded buffers.
#[derive(Clone, Copy, Debug)]
pub struct View<'a>(&'a str);

pub fn view(input: &str) -> io::Result<View<'_>> {
    if input.len() > MAX_INPUT_BYTES {
        return Err(invalid("JSON exceeds size limit"));
    }
    let mut parser = Parser {
        input: input.as_bytes(),
        pos: 0,
        values: 0,
    };
    parser.skip_value(0, true)?;
    parser.space();
    if parser.pos != input.len() {
        return Err(invalid("trailing JSON data"));
    }
    Ok(View(input.trim_matches([' ', '\r', '\n', '\t'])))
}
impl<'a> View<'a> {
    pub fn raw(self) -> &'a str {
        self.0
    }
    pub fn as_str(self) -> Option<Cow<'a, str>> {
        if !self.0.starts_with('"') {
            return None;
        }
        Parser {
            input: self.0.as_bytes(),
            pos: 0,
            values: 0,
        }
        .borrowed_string()
        .ok()
    }
    pub fn fields(self) -> Option<Fields<'a>> {
        self.0.starts_with('{').then(|| {
            Fields(Parser {
                input: self.0.as_bytes(),
                pos: 1,
                values: 0,
            })
        })
    }
    pub fn get(self, key: &str) -> Option<Self> {
        self.fields()?
            .find_map(|(name, value)| (name == key).then_some(value))
    }
    pub fn array(self) -> Option<Elements<'a>> {
        self.0.starts_with('[').then(|| {
            Elements(Parser {
                input: self.0.as_bytes(),
                pos: 1,
                values: 0,
            })
        })
    }
}
pub struct Fields<'a>(Parser<'a>);
impl<'a> Iterator for Fields<'a> {
    type Item = (Cow<'a, str>, View<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.space();
        if self.0.input.get(self.0.pos) == Some(&b'}') {
            return None;
        }
        let name = self.0.borrowed_string().ok()?;
        self.0.space();
        if !self.0.take(b':') {
            return None;
        }
        let value = span(&mut self.0)?;
        self.0.space();
        self.0.take(b',');
        Some((name, value))
    }
}
pub struct Elements<'a>(Parser<'a>);
impl<'a> Iterator for Elements<'a> {
    type Item = View<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.space();
        if self.0.input.get(self.0.pos) == Some(&b']') {
            return None;
        }
        let value = span(&mut self.0)?;
        self.0.space();
        self.0.take(b',');
        Some(value)
    }
}
fn span<'a>(parser: &mut Parser<'a>) -> Option<View<'a>> {
    parser.space();
    let start = parser.pos;
    parser.skip_value(0, false).ok()?;
    Some(View(
        std::str::from_utf8(&parser.input[start..parser.pos]).ok()?,
    ))
}

/// Escape a string exactly, including all control characters.
pub fn quote(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 2);
    write_string(input, &mut out);
    out
}

pub fn stringify(value: &Value) -> io::Result<String> {
    let mut values = 0;
    let len = serialized_len(value, 0, &mut values)?;
    let mut out = String::with_capacity(len);
    write_value(value, &mut out);
    Ok(out)
}

/// Exact escaped byte count, checked before allocating a wire request.
pub(crate) fn quoted_len(input: &str) -> io::Result<usize> {
    if input.len() > MAX_INPUT_BYTES {
        return Err(invalid("JSON string exceeds size limit"));
    }
    let mut size = 2;
    for byte in input.bytes() {
        size += match byte {
            b'"' | b'\\' | b'\n' | b'\r' | b'\t' | 8 | 12 => 2,
            0..=31 => 6,
            _ => 1,
        };
        if size > MAX_INPUT_BYTES {
            return Err(invalid("escaped JSON string exceeds size limit"));
        }
    }
    Ok(size)
}

fn serialized_len(value: &Value, depth: usize, values: &mut usize) -> io::Result<usize> {
    *values += 1;
    if depth > MAX_DEPTH || *values > MAX_VALUES {
        return Err(invalid("JSON complexity limit exceeded"));
    }
    let mut size = match value {
        Value::Null => 4,
        Value::Bool(true) => 4,
        Value::Bool(false) => 5,
        Value::String(text) => quoted_len(text)?,
        Value::Number(number) => {
            if number.len() > MAX_INPUT_BYTES {
                return Err(invalid("JSON number exceeds size limit"));
            }
            let mut parser = Parser {
                input: number.as_bytes(),
                pos: 0,
                values: 0,
            };
            parser.number()?;
            if parser.pos != number.len() {
                return Err(invalid("invalid JSON number"));
            }
            number.len()
        }
        Value::Array(_) | Value::Object(_) => 2,
    };
    match value {
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                size += usize::from(i > 0) + serialized_len(item, depth + 1, values)?;
                if size > MAX_INPUT_BYTES {
                    return Err(invalid("serialized JSON exceeds size limit"));
                }
            }
        }
        Value::Object(fields) => {
            let mut names = std::collections::HashSet::new();
            for (i, (key, item)) in fields.iter().enumerate() {
                if !names.insert(key.as_str()) {
                    return Err(invalid("duplicate JSON object key"));
                }
                size += usize::from(i > 0)
                    + quoted_len(key)?
                    + 1
                    + serialized_len(item, depth + 1, values)?;
                if size > MAX_INPUT_BYTES {
                    return Err(invalid("serialized JSON exceeds size limit"));
                }
            }
        }
        _ => {}
    }
    Ok(size)
}

pub(crate) fn write_string(input: &str, out: &mut String) {
    use std::fmt::Write;
    out.push('"');
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ch if ch < '\u{20}' => {
                let _ = write!(out, "\\u{:04x}", ch as u32);
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
}

fn write_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => out.push_str(value),
        Value::String(value) => write_string(value, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(item, out);
            }
            out.push(']');
        }
        Value::Object(fields) => {
            out.push('{');
            for (i, (key, value)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                write_value(value, out);
            }
            out.push('}');
        }
    }
}

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
    values: usize,
}

impl<'a> Parser<'a> {
    fn space(&mut self) {
        while self
            .input
            .get(self.pos)
            .is_some_and(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.pos += 1;
        }
    }

    fn take(&mut self, byte: u8) -> bool {
        if self.input.get(self.pos) == Some(&byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> io::Result<Value> {
        self.values += 1;
        if depth > MAX_DEPTH || self.values > MAX_VALUES {
            return Err(invalid("JSON complexity limit exceeded"));
        }
        self.space();
        match self.input.get(self.pos).copied() {
            Some(b'n') => {
                self.literal(b"null")?;
                Ok(Value::Null)
            }
            Some(b't') => {
                self.literal(b"true")?;
                Ok(Value::Bool(true))
            }
            Some(b'f') => {
                self.literal(b"false")?;
                Ok(Value::Bool(false))
            }
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b'-' | b'0'..=b'9') => Ok(Value::Number(self.number()?.to_owned())),
            Some(b'[') => {
                self.pos += 1;
                self.space();
                let mut items = Vec::new();
                if self.take(b']') {
                    return Ok(Value::Array(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.space();
                    if self.take(b']') {
                        break;
                    }
                    if !self.take(b',') {
                        return Err(invalid("expected JSON array separator"));
                    }
                }
                Ok(Value::Array(items))
            }
            Some(b'{') => {
                self.pos += 1;
                self.space();
                let mut fields = Vec::new();
                let mut names = std::collections::HashSet::new();
                if self.take(b'}') {
                    return Ok(Value::Object(fields));
                }
                loop {
                    self.space();
                    let key = self.string()?;
                    if !names.insert(key.clone()) {
                        return Err(invalid("duplicate JSON object key"));
                    }
                    self.space();
                    if !self.take(b':') {
                        return Err(invalid("expected JSON object colon"));
                    }
                    fields.push((key, self.value(depth + 1)?));
                    self.space();
                    if self.take(b'}') {
                        break;
                    }
                    if !self.take(b',') {
                        return Err(invalid("expected JSON object separator"));
                    }
                }
                Ok(Value::Object(fields))
            }
            _ => Err(invalid("expected JSON value")),
        }
    }

    fn literal(&mut self, value: &[u8]) -> io::Result<()> {
        if self.input.get(self.pos..self.pos + value.len()) != Some(value) {
            return Err(invalid("invalid JSON literal"));
        }
        self.pos += value.len();
        Ok(())
    }

    fn number(&mut self) -> io::Result<&'a str> {
        let start = self.pos;
        self.take(b'-');
        if !self.take(b'0') {
            if !self
                .input
                .get(self.pos)
                .is_some_and(|b| matches!(b, b'1'..=b'9'))
            {
                return Err(invalid("invalid JSON number"));
            }
            self.digits();
        }
        if self.take(b'.') {
            let before = self.pos;
            self.digits();
            if before == self.pos {
                return Err(invalid("invalid JSON fraction"));
            }
        }
        if self.take(b'e') || self.take(b'E') {
            if !self.take(b'+') {
                self.take(b'-');
            }
            let before = self.pos;
            self.digits();
            if before == self.pos {
                return Err(invalid("invalid JSON exponent"));
            }
        }
        Ok(std::str::from_utf8(&self.input[start..self.pos]).unwrap())
    }

    fn digits(&mut self) {
        while self.input.get(self.pos).is_some_and(u8::is_ascii_digit) {
            self.pos += 1;
        }
    }

    fn borrowed_string(&mut self) -> io::Result<Cow<'a, str>> {
        if self.input.get(self.pos) != Some(&b'"') {
            return Err(invalid("expected JSON string"));
        }
        let start = self.pos + 1;
        for (offset, &byte) in self.input[start..].iter().enumerate() {
            match byte {
                b'"' => {
                    self.pos = start + offset + 1;
                    return Ok(Cow::Borrowed(
                        std::str::from_utf8(&self.input[start..start + offset])
                            .map_err(|_| invalid("invalid string UTF-8"))?,
                    ));
                }
                b'\\' => return self.string().map(Cow::Owned),
                0..=31 => return Err(invalid("unescaped JSON control character")),
                _ => {}
            }
        }
        Err(invalid("unterminated JSON string"))
    }

    // Validate without building a value tree. Unescaped keys/text and numbers
    // borrow input. Only duplicate-key sets and decoded escapes need allocation.
    fn skip_value(&mut self, depth: usize, validate_keys: bool) -> io::Result<()> {
        self.values += 1;
        if depth > MAX_DEPTH || self.values > MAX_VALUES {
            return Err(invalid("JSON complexity limit exceeded"));
        }
        self.space();
        match self.input.get(self.pos).copied() {
            Some(b'n') => self.literal(b"null"),
            Some(b't') => self.literal(b"true"),
            Some(b'f') => self.literal(b"false"),
            Some(b'"') => self.borrowed_string().map(|_| ()),
            Some(b'-' | b'0'..=b'9') => self.number().map(|_| ()),
            Some(b'[' | b'{') => {
                let object = self.input[self.pos] == b'{';
                let end = if object { b'}' } else { b']' };
                self.pos += 1;
                self.space();
                if self.take(end) {
                    return Ok(());
                }
                let mut names = std::collections::HashSet::new();
                loop {
                    self.space();
                    if object {
                        let name = self.borrowed_string()?;
                        if validate_keys && !names.insert(name) {
                            return Err(invalid("duplicate JSON object key"));
                        }
                        self.space();
                        if !self.take(b':') {
                            return Err(invalid("expected JSON object colon"));
                        }
                    }
                    self.skip_value(depth + 1, validate_keys)?;
                    self.space();
                    if self.take(end) {
                        return Ok(());
                    }
                    if !self.take(b',') {
                        return Err(invalid("expected JSON separator"));
                    }
                }
            }
            _ => Err(invalid("expected JSON value")),
        }
    }

    fn string(&mut self) -> io::Result<String> {
        if !self.take(b'"') {
            return Err(invalid("expected JSON string"));
        }
        let mut out = String::new();
        let mut start = self.pos;
        loop {
            let byte = *self
                .input
                .get(self.pos)
                .ok_or_else(|| invalid("unterminated JSON string"))?;
            if byte == b'"' || byte == b'\\' {
                out.push_str(
                    std::str::from_utf8(&self.input[start..self.pos])
                        .map_err(|_| invalid("invalid string UTF-8"))?,
                );
                self.pos += 1;
                if byte == b'"' {
                    return Ok(out);
                }
                let escape = *self
                    .input
                    .get(self.pos)
                    .ok_or_else(|| invalid("unterminated JSON escape"))?;
                self.pos += 1;
                match escape {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{08}'),
                    b'f' => out.push('\u{0c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let first = self.hex4()?;
                        let scalar = if (0xd800..=0xdbff).contains(&first) {
                            if !self.take(b'\\') || !self.take(b'u') {
                                return Err(invalid("unpaired JSON surrogate"));
                            }
                            let second = self.hex4()?;
                            if !(0xdc00..=0xdfff).contains(&second) {
                                return Err(invalid("unpaired JSON surrogate"));
                            }
                            0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
                        } else {
                            first
                        };
                        out.push(
                            char::from_u32(scalar)
                                .ok_or_else(|| invalid("invalid JSON Unicode scalar"))?,
                        );
                    }
                    _ => return Err(invalid("invalid JSON escape")),
                }
                start = self.pos;
            } else {
                if byte < 0x20 {
                    return Err(invalid("unescaped JSON control character"));
                }
                self.pos += 1;
            }
        }
    }

    fn hex4(&mut self) -> io::Result<u32> {
        let mut value = 0;
        for _ in 0..4 {
            let byte = *self
                .input
                .get(self.pos)
                .ok_or_else(|| invalid("incomplete JSON Unicode escape"))?;
            value = (value << 4)
                | match byte {
                    b'0'..=b'9' => (byte - b'0') as u32,
                    b'a'..=b'f' => (byte - b'a' + 10) as u32,
                    b'A'..=b'F' => (byte - b'A' + 10) as u32,
                    _ => return Err(invalid("invalid JSON Unicode escape")),
                };
            self.pos += 1;
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_exact_strings_and_numbers() {
        let value = parse(r#"{"x":"\uD83D\uDE80\n\u0000é","n":-0.123e+999,"a":[true,false,null]}"#)
            .unwrap();
        assert_eq!(value.get("x").unwrap().as_str(), Some("🚀\n\0é"));
        assert_eq!(parse(&stringify(&value).unwrap()).unwrap(), value);
        assert!(stringify(&value).unwrap().contains("-0.123e+999"));
    }

    #[test]
    fn rejects_malformed_and_ambiguous_json() {
        for input in [
            "",
            "[1,]",
            "{\"a\":1,}",
            "01",
            "+1",
            "1.",
            "1e",
            "NaN",
            "true false",
            "\"\n\"",
            r#""\uD800""#,
            r#""\uDC00""#,
            r#"{"a":1,"a":2}"#,
        ] {
            assert!(parse(input).is_err(), "accepted {input:?}");
        }
        assert!(parse_bytes(&[b'"', 0xff, b'"']).is_err());
        assert!(parse(&format!("{}0{}", "[".repeat(66), "]".repeat(66))).is_err());
        assert!(parse(&" ".repeat(MAX_INPUT_BYTES + 1)).is_err());
    }

    #[test]
    fn serializer_bounds_constructed_values_and_escaping() {
        assert!(stringify(&Value::Number("NaN".into())).is_err());
        assert!(
            stringify(&Value::Object(vec![
                ("x".into(), Value::Null),
                ("x".into(), Value::Null)
            ]))
            .is_err()
        );
        assert!(stringify(&Value::String("\0".repeat(MAX_INPUT_BYTES / 6 + 1))).is_err());
        let mut deep = Value::Null;
        for _ in 0..66 {
            deep = Value::Array(vec![deep]);
        }
        assert!(stringify(&deep).is_err());
        let text = "é🦀\n\0\"\\";
        assert_eq!(quoted_len(text).unwrap(), quote(text).len());
    }
    #[test]
    fn borrowed_views_match_owned_validation_and_preserve_text() {
        fn compare(view: View<'_>, value: &Value) {
            match value {
                Value::Null => assert_eq!(view.raw(), "null"),
                Value::Bool(v) => assert_eq!(view.raw(), if *v { "true" } else { "false" }),
                Value::Number(v) => assert_eq!(view.raw(), v),
                Value::String(v) => assert_eq!(view.as_str().unwrap(), v.as_str()),
                Value::Array(values) => {
                    let views: Vec<_> = view.array().unwrap().collect();
                    assert_eq!(views.len(), values.len());
                    for (view, value) in views.into_iter().zip(values) {
                        compare(view, value);
                    }
                }
                Value::Object(values) => {
                    let views: Vec<_> = view.fields().unwrap().collect();
                    assert_eq!(views.len(), values.len());
                    for ((name, v), (key, value)) in views.into_iter().zip(values) {
                        assert_eq!(name, key.as_str());
                        compare(v, value);
                        compare(view.get(key).unwrap(), value);
                    }
                }
            }
        }
        let fixtures = [
            r#" { "choices":[{"index":0,"delta":{"content":"é🦀","reasoning":null},"finish_reason":null}], "n":-1.200e+03 } "#,
            r#"{"escaped\u006bey":"\ud83e\udd80\n\"\\","arr":[[],{},true,false,null,0]}"#,
            r#"{"x":1,"\u0078":2}"#,
            r#"{"x":[1,]}"#,
            "[01]",
            "{} x",
            r#""\ud800""#,
            r#"{"x":true,"x":false}"#,
            "\"bad\u{0000}control\"",
            "[1e]",
            "null",
        ];
        for input in fixtures {
            let owned = parse(input);
            let borrowed = view(input);
            assert_eq!(owned.is_ok(), borrowed.is_ok(), "{input:?}");
            if let (Ok(value), Ok(v)) = (owned, borrowed) {
                compare(v, &value);
            }
        }
        let text = "\"zero copy 🦀\"";
        let borrowed = view(text).unwrap().as_str().unwrap();
        assert!(matches!(borrowed, Cow::Borrowed(_)));
        assert_eq!(borrowed.as_ptr(), text[1..].as_ptr());
        assert!(matches!(
            view(r#""decode\n""#).unwrap().as_str().unwrap(),
            Cow::Owned(_)
        ));
        // Deterministic malformed-input corpus exercises both validators. This
        // is a local grammar differential check, not a fuzzing completeness claim.
        let alphabet = b"{}[],:\"\\012-+eE \n\tntfalsrue";
        let mut state = 0x123456789abcdefu64;
        for length in 0..80 {
            for _ in 0..64 {
                let input: String = (0..length)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        alphabet[state as usize % alphabet.len()] as char
                    })
                    .collect();
                assert_eq!(parse(&input).is_ok(), view(&input).is_ok(), "{input:?}");
            }
        }
        assert!(
            view(&format!(
                "{}0{}",
                "[".repeat(MAX_DEPTH + 2),
                "]".repeat(MAX_DEPTH + 2)
            ))
            .is_err()
        );
    }
}
