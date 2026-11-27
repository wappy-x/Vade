/// OpenAI's Model object requires `created`. The model has no publication time
/// here, so it is the Unix second at which this process first listed the model.
pub(super) fn created() -> u64 {
    static CREATED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CREATED.get_or_init(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default().as_secs())
}

#[cfg(test)]
#[test]
fn model_listing_reports_effective_limits() {
    use super::{Limits, model_response};
    use engine_model::json::{self, Value};
    for (configured, engine, output) in [(32768, 65536, 16384), (65536, 32768, 65536)] {
        let limits = Limits { context_tokens: configured, output_tokens: output, ..Limits::default() };
        for state in ["ready", "loading"] {
            let body = model_response("/v1/models", state, "model", 0, 0, 0, None, &limits, engine, None, None, false).unwrap();
            let value = json::parse(body.as_bytes(), Default::default()).unwrap();
            let entry = value.object().unwrap()["data"].array().unwrap()[0].object().unwrap();
            for key in ["context_length", "max_model_len"] {
                assert_eq!(entry[key], Value::Number(configured.min(engine).to_string()));
            }
            assert_eq!(entry["max_output_tokens"], Value::Number(limits.output_limit(engine).to_string()));
            assert_eq!(entry["input_modalities"], Value::Array(vec![Value::String("text".into())]));
            // openai-python's Model type requires created (Unix seconds).
            assert!(matches!(&entry["created"], Value::Number(n) if n.parse::<u64>().is_ok_and(|t| t > 1_700_000_000)), "{body}");
        }
        for state in ["failed", "poisoned"] {
            assert!(model_response("/v1/models", state, "model", 0, 0, 0, None, &limits, engine, None, None, false).unwrap().contains("\"data\":[]"));
        }
    }
}
