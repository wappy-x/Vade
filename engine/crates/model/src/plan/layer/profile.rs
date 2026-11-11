//! Operand profile checks for the scalar reference. No runtime profile is admitted.
//! Storage bytes do not establish effective parameter values after a runtime load.
use std::fmt;

/// A source declares the arithmetic meaning of its decoded FP32 values.
/// These variants describe requests. They do not select an execution path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Stored weights decoded to FP32. No checkpoint activation casts or quantizers.
    LogicalF32V1,
    /// Conditional BF16 requirements with unqualified parameter loads and backends.
    ConditionalBf16,
    /// Checkpoint runtime values. Their effective parameter loads are not bound.
    CheckpointRuntime,
}
impl Profile {
    pub fn name(self) -> &'static str {
        match self { Self::LogicalF32V1=>"logical-f32-v1",Self::ConditionalBf16=>"conditional-bf16",
            Self::CheckpointRuntime=>"checkpoint-runtime" }
    }
    /// Check the first open producer boundary before any weight read or compute.
    /// No caller flag or stored dtype can qualify an effective runtime load.
    pub fn require_scalar(self) -> Result<(),UnsupportedProfile> {
        match self {
            Self::LogicalF32V1=>Ok(()),
            Self::ConditionalBf16|Self::CheckpointRuntime=>Err(UnsupportedProfile {
                requested:self,boundary:Boundary::EffectiveParameterLoad,
            }),
        }
    }
}
/// Effective values after parameter load remain unqualified.
/// The layer's first fn.float() consumes these values, not the stored header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary { EffectiveParameterLoad }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedProfile { pub requested: Profile, pub boundary: Boundary }
impl fmt::Display for UnsupportedProfile {
    fn fmt(&self,f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f,"{}: effective runtime parameter load is unbound; the reference requires logical-f32-v1",
            self.requested.name())
    }
}
impl std::error::Error for UnsupportedProfile {}
