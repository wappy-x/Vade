//! Template-only JSON view. Jinja dictsort is stable on the original member
//! order when lowercase keys compare equal; the metadata BTreeMap loses it.
use crate::{Result, err};
use engine_model::json;

#[derive(Clone, PartialEq)]
pub(super) enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(Object),
}
#[derive(Clone, PartialEq)]
pub(super) struct Object(Vec<(String, Value)>);
impl Object {
    pub(super) fn get(&self, key: &str) -> Option<&Value> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub(super) fn is_empty(&self) -> bool { self.0.is_empty() }
    pub(super) fn sorted(&self) -> Vec<(&String, &Value)> {
        let mut members: Vec<_> = self.0.iter().map(|(k, v)| (k, v)).collect();
        members.sort_by_cached_key(|(key, _)| key.to_lowercase());
        members
    }
}
impl Value {
    pub(super) fn object(&self) -> Result<&Object> {
        if let Self::Object(o) = self { Ok(o) } else { Err(err("expected JSON object")) }
    }
    pub(super) fn string(&self) -> Result<&str> {
        if let Self::String(s) = self { Ok(s) } else { Err(err("expected JSON string")) }
    }
}

pub(super) fn parse(bytes: &[u8], limits: json::Limits) -> Result<Value> {
    // The shared reader enforces syntax, duplicate keys and all resource bounds.
    // Only retain a different object representation for this template.
    let validated = json::parse(bytes, limits)?;
    ordered(&validated, bytes, &mut 0)
}
fn whitespace(bytes: &[u8], pos: &mut usize) {
    while bytes.get(*pos).is_some_and(u8::is_ascii_whitespace) { *pos += 1; }
}
fn string_end(bytes: &[u8], pos: &mut usize) {
    *pos += 1;
    while *pos < bytes.len() {
        match bytes[*pos] {
            b'\\' => *pos += 2,
            b'"' => { *pos += 1; break; },
            _ => *pos += 1,
        }
    }
}
fn ordered(value: &json::Value, bytes: &[u8], pos: &mut usize) -> Result<Value> {
    whitespace(bytes, pos);
    Ok(match value {
        json::Value::Object(object) => {
            *pos += 1;
            let mut members = Vec::with_capacity(object.len());
            for i in 0..object.len() {
                whitespace(bytes, pos);
                if i > 0 { *pos += 1; whitespace(bytes, pos); }
                let start = *pos;
                string_end(bytes, pos);
                let key = json::parse(&bytes[start..*pos], json::Limits { string_bytes: bytes.len(), ..Default::default() })?.string()?.to_owned();
                whitespace(bytes, pos);
                *pos += 1;
                let child = ordered(&object[&key], bytes, pos)?;
                members.push((key, child));
            }
            whitespace(bytes, pos);
            *pos += 1;
            Value::Object(Object(members))
        },
        json::Value::Array(array) => {
            *pos += 1;
            let mut values = Vec::with_capacity(array.len());
            for (i, value) in array.iter().enumerate() {
                whitespace(bytes, pos);
                if i > 0 { *pos += 1; }
                values.push(ordered(value, bytes, pos)?);
            }
            whitespace(bytes, pos);
            *pos += 1;
            Value::Array(values)
        },
        json::Value::String(s) => { string_end(bytes, pos); Value::String(s.clone()) },
        scalar => {
            while bytes.get(*pos).is_some_and(|b| !b.is_ascii_whitespace() && !b",]}".contains(b)) { *pos += 1; }
            match scalar {
                json::Value::Null => Value::Null,
                json::Value::Bool(b) => Value::Bool(*b),
                json::Value::Number(s) => Value::Number(s.clone()),
                _ => unreachable!(),
            }
        },
    })
}
