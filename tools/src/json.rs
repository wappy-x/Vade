//! Small JSON reader and writer for host tools and examples.
use std::collections::BTreeMap;
#[derive(Clone, Debug, PartialEq)]
pub enum Value { Null, Bool(bool), Number(String), String(String), Array(Vec<Value>), Object(BTreeMap<String, Value>) }
impl Value {
    pub fn get(&self, key: &str) -> &Self {
        match self { Self::Object(o) => o.get(key).unwrap_or(&Self::Null),
            Self::Array(a) => key.parse::<usize>().ok().and_then(|i| a.get(i)).unwrap_or(&Self::Null), _ => &Self::Null }
    }
    pub fn text(&self) -> Result<&str, String> {
        match self { Self::String(s) | Self::Number(s) => Ok(s), _ => Err("expected JSON string or number".into()) }
    }
    pub fn array(&self) -> Result<&[Self], String> { if let Self::Array(a) = self { Ok(a) } else { Err("expected JSON array".into()) } }
    pub fn encode(&self) -> String {
        match self { Self::Null => "null".into(), Self::Bool(b) => b.to_string(), Self::Number(n) => n.clone(),
            Self::String(s) => quote(s), Self::Array(a) => format!("[{}]", a.iter().map(Self::encode).collect::<Vec<_>>().join(",")),
            Self::Object(o) => format!("{{{}}}", o.iter().map(|(k,v)| format!("{}:{}",quote(k),v.encode())).collect::<Vec<_>>().join(",")) }
    }
}
pub fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() { match c { '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"), '\n' => out.push_str("\\n"),
        '\r' => out.push_str("\\r"), '\t' => out.push_str("\\t"), c if c < ' ' => out.push_str(&format!("\\u{:04x}",c as u32)), c => out.push(c) } }
    out.push('"'); out
}
pub fn parse(text: &str) -> Result<Value, String> {
    let mut p = Parser { bytes: text.as_bytes(), i: 0 }; let v = p.value(0)?;
    p.space(); if p.i != p.bytes.len() { return Err("trailing JSON data".into()); } Ok(v)
}
struct Parser<'a> { bytes: &'a [u8], i: usize }
impl Parser<'_> {
    fn space(&mut self) { while self.bytes.get(self.i).is_some_and(|c| matches!(c,b' '|b'\t'|b'\r'|b'\n')) { self.i += 1; } }
    fn take(&mut self, c: u8) -> bool { self.space(); if self.bytes.get(self.i) == Some(&c) { self.i += 1; true } else { false } }
    fn hex(&mut self) -> Result<u32, String> {
        let end = self.i + 4; let s = self.bytes.get(self.i..end).ok_or("short JSON escape")?;
        let n = u32::from_str_radix(std::str::from_utf8(s).map_err(|_| "invalid JSON escape")?,16).map_err(|_| "invalid JSON escape")?;
        self.i = end; Ok(n)
    }
    fn string(&mut self) -> Result<String, String> {
        if !self.take(b'"') { return Err("expected JSON string".into()); }
        let mut out = Vec::new();
        loop {
            let c = *self.bytes.get(self.i).ok_or("unclosed JSON string")?; self.i += 1;
            match c { b'"' => return String::from_utf8(out).map_err(|_| "invalid UTF-8".into()),
                b'\\' => { let c = *self.bytes.get(self.i).ok_or("short JSON escape")?; self.i += 1;
                    match c { b'"'|b'\\'|b'/' => out.push(c), b'n' => out.push(b'\n'), b'r' => out.push(b'\r'), b't' => out.push(b'\t'),
                        b'b' => out.push(8), b'f' => out.push(12), b'u' => {
                            let mut n = self.hex()?;
                            if (0xd800..=0xdbff).contains(&n) {
                                if self.bytes.get(self.i..self.i+2) != Some(b"\\u") { return Err("missing low surrogate".into()); }
                                self.i += 2; let low = self.hex()?;
                                if !(0xdc00..=0xdfff).contains(&low) { return Err("invalid low surrogate".into()); }
                                n = 0x10000 + ((n-0xd800)<<10) + low-0xdc00;
                            }
                            let ch = char::from_u32(n).ok_or("invalid Unicode scalar")?;
                            out.extend_from_slice(ch.encode_utf8(&mut [0;4]).as_bytes());
                        }, _ => return Err("invalid JSON escape".into()) }
                }, 0..=31 => return Err("control byte in JSON string".into()), _ => out.push(c) }
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > 128 { return Err("JSON nesting limit".into()); } self.space();
        match self.bytes.get(self.i).copied().ok_or("missing JSON value")? {
            b'"' => Ok(Value::String(self.string()?)),
            b'[' => { self.i += 1; let mut a = Vec::new(); if self.take(b']') { return Ok(Value::Array(a)); }
                loop { a.push(self.value(depth+1)?); if self.take(b']') { break; } if !self.take(b',') { return Err("expected JSON comma".into()); } }
                Ok(Value::Array(a)) },
            b'{' => { self.i += 1; let mut o = BTreeMap::new(); if self.take(b'}') { return Ok(Value::Object(o)); }
                loop { let k = self.string()?; if !self.take(b':') { return Err("expected JSON colon".into()); }
                    if o.insert(k,self.value(depth+1)?).is_some() { return Err("duplicate JSON key".into()); }
                    if self.take(b'}') { break; } if !self.take(b',') { return Err("expected JSON comma".into()); } }
                Ok(Value::Object(o)) },
            b'n'|b't'|b'f' => { for (s,v) in [("null",Value::Null),("true",Value::Bool(true)),("false",Value::Bool(false))] {
                if self.bytes[self.i..].starts_with(s.as_bytes()) { self.i += s.len(); return Ok(v); } } Err("invalid JSON literal".into()) },
            b'-'|b'0'..=b'9' => {
                let start = self.i; if self.bytes[self.i] == b'-' { self.i += 1; }
                if self.bytes.get(self.i) == Some(&b'0') { self.i += 1; } else { self.digits()?; }
                if self.bytes.get(self.i) == Some(&b'.') { self.i += 1; self.digits()?; }
                if self.bytes.get(self.i).is_some_and(|c| matches!(c,b'e'|b'E')) {
                    self.i += 1; if self.bytes.get(self.i).is_some_and(|c| matches!(c,b'+'|b'-')) { self.i += 1; } self.digits()?;
                }
                Ok(Value::Number(std::str::from_utf8(&self.bytes[start..self.i]).unwrap().into()))
            }, _ => Err("invalid JSON value".into()) }
    }
    fn digits(&mut self) -> Result<(), String> { let start = self.i;
        while self.bytes.get(self.i).is_some_and(u8::is_ascii_digit) { self.i += 1; }
        if start == self.i { Err("missing JSON digit".into()) } else { Ok(()) }
    }
}
