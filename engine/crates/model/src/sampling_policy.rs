//! Shared field-presence policy for direct serving and the dependency-free gateway.
//! Extra controls preserve the existing checkpoint base; listing a key here does
//! not add support for interpreting its value.
pub const FIELDS: &str = "temperature,top_p,top_k,min_p,typical_p,seed,frequency_penalty,presence_penalty,repetition_penalty,logit_bias,n,best_of,sampling,generation_config";
pub const GREEDY_DEFAULTS_JSON: &str = r#"{"temperature":0,"seed":20260905,"top_p":1.0,"top_k":0,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0}"#;
pub fn is_field(name: &str) -> bool { FIELDS.split(',').any(|field| name == field) }
