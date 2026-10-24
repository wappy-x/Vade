//! Typed failures while reading, validating and reconciling a model plan.
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    Read { file: &'static str, message: String },
    Missing { field: &'static str },
    WrongType { field: &'static str, expected: &'static str },
    Unsupported { field: &'static str, value: String },
    Invalid { field: &'static str, reason: &'static str },
    Overflow { what: &'static str },
    MissingTensor { name: String },
    UnexpectedTensor { name: String },
    TensorMismatch { name: String, expected: Vec<u64>, actual: Vec<u64> },
}
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { file, message } => write!(f, "{file}: {message}"),
            Self::Missing { field } => write!(f, "missing field {field}"),
            Self::WrongType { field, expected } => write!(f, "{field}: expected {expected}"),
            Self::Unsupported { field, value } => write!(f, "unsupported {field}: {value}"),
            Self::Invalid { field, reason } => write!(f, "invalid {field}: {reason}"),
            Self::Overflow { what } => write!(f, "overflow computing {what}"),
            Self::MissingTensor { name } => write!(f, "missing tensor {name}"),
            Self::UnexpectedTensor { name } => write!(f, "unexpected tensor {name}"),
            Self::TensorMismatch { name, expected, actual } =>
                write!(f, "tensor {name}: expected shape {expected:?}, actual {actual:?}"),
        }
    }
}
impl std::error::Error for PlanError {}
