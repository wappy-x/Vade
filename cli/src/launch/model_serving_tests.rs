//! Mandatory serving metadata: discovery must refuse what flash-serve
//! refuses at startup, before any choice is saved or installed.
use super::tests::Fixture;
use super::*;

fn config_with_template(template: &str) -> String {
    format!(
        r#"{{"chat_template":{},"add_bos_token":false,"add_prefix_space":false,"clean_up_tokenization_spaces":false,"split_special_tokens":false,"eos_token":"<|im_end|>","pad_token":"<|endoftext|>"}}"#,
        test_json_string(template)
    )
}

#[test]
fn missing_or_invalid_generation_config_refuses_admission() {
    let f = Fixture::new();
    let path = f.model("checkpoint");
    assert_eq!(inspect(&path).unwrap().context_limit, 8192);
    let generation = path.join("generation_config.json");
    fs::remove_file(&generation).unwrap();
    assert!(
        inspect(&path)
            .unwrap_err()
            .contains("generation_config.json")
    );
    for invalid in [
        "{",
        "[]",
        "\"text\"",
        r#"{"temperature":"hot"}"#,
        r#"{"temperature":-1,"do_sample":true}"#,
        r#"{"top_p":0}"#,
        r#"{"top_p":1.5}"#,
        r#"{"repetition_penalty":0}"#,
        r#"{"repetition_penalty":-1}"#,
        r#"{"top_k":-1}"#,
        r#"{"top_k":1.5}"#,
        r#"{"top_k":"20"}"#,
        r#"{"do_sample":"yes"}"#,
        r#"{"eos_token_id":[]}"#,
        r#"{"eos_token_id":248320}"#,
        r#"{"eos_token_id":[248044,248320]}"#,
        r#"{"eos_token_id":"248044"}"#,
        r#"{"eos_token_id":-1}"#,
        r#"{"eos_token_id":4294967296}"#,
    ] {
        fs::write(&generation, invalid).unwrap();
        let error = inspect(&path).unwrap_err();
        assert!(error.contains("generation_config.json"), "{invalid}: {error}");
        assert!(!error.contains('\n'), "{invalid}: {error}");
    }
    let file = File::create(&generation).unwrap();
    file.set_len(1024 * 1024 + 1).unwrap();
    drop(file);
    assert!(
        inspect(&path)
            .unwrap_err()
            .contains("generation_config.json")
    );
    for valid in [
        "{}",
        // Without do_sample the engine zeroes the temperature before
        // validating, exactly like Generation::open.
        r#"{"temperature":-1}"#,
        r#"{"temperature":-1,"do_sample":false}"#,
        r#"{"do_sample":true,"temperature":0.7,"top_k":20,"top_p":0.9,"repetition_penalty":1.1}"#,
        r#"{"eos_token_id":[248044,248045]}"#,
        r#"{"eos_token_id":248319}"#,
    ] {
        fs::write(&generation, valid).unwrap();
        assert_eq!(inspect(&path).unwrap().context_limit, 8192, "{valid}");
    }
}

#[test]
fn missing_or_invalid_tokenizer_metadata_refuses_admission() {
    let f = Fixture::new();
    let path = f.model("checkpoint");
    let config = path.join("tokenizer_config.json");
    fs::remove_file(&config).unwrap();
    assert!(
        inspect(&path)
            .unwrap_err()
            .contains("tokenizer_config.json")
    );
    for invalid in ["{", "[]", "{}", r#"{"chat_template":42}"#] {
        fs::write(&config, invalid).unwrap();
        let error = inspect(&path).unwrap_err();
        assert!(error.contains("tokenizer_config.json"), "{invalid}: {error}");
    }
    // A chat template the engine does not recognize fails its tokenizer
    // configuration validation at startup, so discovery refuses it too.
    fs::write(&config, config_with_template("hello")).unwrap();
    let error = inspect(&path).unwrap_err();
    assert!(error.contains("tokenizer"), "{error}");
    fs::write(&config, test_tokenizer_config()).unwrap();
    assert_eq!(inspect(&path).unwrap().context_limit, 8192);

    let tokenizer = path.join("tokenizer.json");
    fs::remove_file(&tokenizer).unwrap();
    assert!(inspect(&path).unwrap_err().contains("tokenizer.json"));
    for invalid in ["", "{}", "{"] {
        fs::write(&tokenizer, invalid).unwrap();
        let error = inspect(&path).unwrap_err();
        assert!(error.contains("tokenizer"), "{invalid:?}: {error}");
    }
    fs::write(&tokenizer, test_tokenizer_json()).unwrap();
    assert_eq!(inspect(&path).unwrap().context_limit, 8192);
}

#[test]
fn chat_template_jinja_must_match_the_embedded_template() {
    let f = Fixture::new();
    let path = f.model("checkpoint");
    let jinja = path.join("chat_template.jinja");
    fs::write(&jinja, test_chat_template()).unwrap();
    assert_eq!(inspect(&path).unwrap().context_limit, 8192);
    // Trailing CR/LF differences are normalized away by the engine.
    fs::write(&jinja, format!("{}\r\n\r\n", test_chat_template())).unwrap();
    assert_eq!(inspect(&path).unwrap().context_limit, 8192);
    fs::write(&jinja, "a different template").unwrap();
    assert!(
        inspect(&path)
            .unwrap_err()
            .contains("chat_template.jinja")
    );
    fs::write(&jinja, [0xff, 0xfe]).unwrap();
    assert!(
        inspect(&path)
            .unwrap_err()
            .contains("chat_template.jinja")
    );
    fs::remove_file(&jinja).unwrap();
    assert_eq!(inspect(&path).unwrap().context_limit, 8192);
}

#[test]
fn incomplete_checkpoints_are_never_returned_by_discovery() {
    let f = Fixture::new();
    let complete = f.model("elsewhere/complete");
    let incomplete = f.model("models/incomplete");
    fs::remove_file(incomplete.join("generation_config.json")).unwrap();
    let no_prompt = |_: &str| panic!("unexpected prompt");
    // A previously saved incomplete choice is displaced by VADE_MODELS.
    assert_eq!(
        discover_with(&f.0, Some(&incomplete), Some(complete.as_os_str()), no_prompt)
            .unwrap()
            .path,
        complete
    );
    // The same saved choice recovers through the prompt, whose reason
    // names the missing mandatory file.
    assert_eq!(
        discover_with(&f.0, Some(&incomplete), None, |reason| {
            assert!(reason.contains("generation_config.json"), "{reason}");
            Ok(complete.clone())
        })
        .unwrap()
        .path,
        complete
    );
    // An incomplete checkpoint in ~/models is skipped for VADE_MODELS.
    assert_eq!(
        discover_with(&f.0, None, Some(complete.as_os_str()), no_prompt)
            .unwrap()
            .path,
        complete
    );
    // A prompt naming the incomplete checkpoint refuses it.
    assert!(
        discover_with(&f.0, None, None, |_| Ok(incomplete.clone()))
            .unwrap_err()
            .contains("generation_config.json")
    );
    // Recovery also applies when the tokenizer configuration is missing.
    fs::write(
        incomplete.join("generation_config.json"),
        test_generation_config(),
    )
    .unwrap();
    fs::remove_file(incomplete.join("tokenizer_config.json")).unwrap();
    assert_eq!(
        discover_with(&f.0, Some(&incomplete), Some(complete.as_os_str()), no_prompt)
            .unwrap()
            .path,
        complete
    );
    assert!(
        discover_with(&f.0, None, None, |_| Ok(incomplete.clone()))
            .unwrap_err()
            .contains("tokenizer_config.json")
    );
}
