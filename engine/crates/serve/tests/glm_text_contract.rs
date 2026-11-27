//! Source-backed host vectors. These tests do not run a model or a GPU.
//! See fixtures/glm-text/README.md for sources, derivation, and open proof.
use spark_serve::{Decoder, Tokenizer, tokenizer::ModelTokenizer};
use std::path::PathBuf;
#[path = "../../../tools/export/src/sha256.rs"]
mod sha256;
#[path = "support/glm_tokenizer_evidence.rs"]
mod evidence;

fn artifact_root() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../.local/text-contract") }
fn checkpoint() -> ModelTokenizer {
    let root = artifact_root();
    let file = std::fs::File::open(root.join("tokenizer.json")).expect("save the complete public tokenizer in .local/text-contract");
    assert_eq!(sha256::digest_reader(file).unwrap(), "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d");
    for (name, hash) in [
        ("generation_config.json", "230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f"),
        ("tokenizer_config.json", "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"),
        ("chat_template.jinja", "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5"),
    ] { assert_eq!(sha256::digest(&std::fs::read(root.join(name)).unwrap()), hash, "{name}"); }
    ModelTokenizer::from_directory(root).unwrap()
}
fn chat(body: &str) -> spark_serve::protocol::ChatInput {
    let request = spark_serve::protocol::parse(body.as_bytes(), true, &Default::default()).unwrap();
    let spark_serve::protocol::Prompt::Chat(input) = request.prompt else { panic!() }; input
}

#[test]
#[ignore = "needs the pinned public artifacts in .local/text-contract"]
fn plain_text_ids_and_streamed_bytes_match_source_vectors() {
    let model = checkpoint();
    // Whole pre-tokens use model.vocab directly because ignore_merges=true.
    // The other IDs follow the fixed merge ranks, checked outside this codec.
    let fixture=engine_model::json::parse(include_bytes!("fixtures/glm-text/source-vectors.json"),Default::default()).unwrap();
    for vector in fixture.object().unwrap()["plain"].array().unwrap() {
        let fields=vector.object().unwrap();let text=fields["text"].string().unwrap();
        let ids=fields["ids"].array().unwrap().iter().map(|v|u32::try_from(v.u64().unwrap()).unwrap()).collect::<Vec<_>>();
        assert_eq!(Tokenizer::encode(&model, text).unwrap(), ids, "{text:?}");
        assert_eq!(model.decode(&ids, false).unwrap().as_bytes(), text.as_bytes());
        assert_eq!(ids.iter().flat_map(|&id|model.token_bytes(id).unwrap()).copied().collect::<Vec<_>>(),text.as_bytes());
        let mut decoder = model.decoder();
        let mut decoded = String::new();
        for &id in &ids { decoded.push_str(&Decoder::push(&mut decoder, id).unwrap()); }
        decoded.push_str(&Decoder::finish(&mut decoder).unwrap());
        assert_eq!(decoded.as_bytes(), text.as_bytes(), "{ids:?}");
    }
}

#[test]
#[ignore = "needs the pinned public artifacts in .local/text-contract"]
fn special_tokens_and_complete_eos_set_match_source() {
    let model = checkpoint();
    assert_eq!(model.checkpoint_eos_token_ids().unwrap(), Some([154820,154827,154829].as_slice()));
    let bytes = std::fs::read(artifact_root().join("tokenizer.json")).unwrap();
    let source = engine_model::json::parse(&bytes, engine_model::json::Limits { bytes:32<<20, nodes:2_000_000, string_bytes:1<<20, depth:24 }).unwrap();
    for added in source.object().unwrap()["added_tokens"].array().unwrap() {
        let fields = added.object().unwrap();
        let id = fields["id"].u64().unwrap() as u32;
        let text = fields["content"].string().unwrap();
        let special = fields["special"].boolean().unwrap();
        assert_eq!(engine_tokenizer::Tokenizer::encode(&model, text).unwrap(), [id]);
        assert_eq!(model.is_special(id), special);
        assert_eq!(model.decode(&[id], false).unwrap(), text);
        assert_eq!(model.decode(&[id], true).unwrap(), if special { "" } else { text });
    }
    // added_tokens supplies both the IDs and special flags. Think and tool
    // markers have special=false, so skip_special must keep their bytes.
    for (text, id, special) in [
        ("<|endoftext|>",154820,true), ("[gMASK]",154822,true), ("<sop>",154824,true),
        ("<|system|>",154826,true), ("<|user|>",154827,true), ("<|assistant|>",154828,true),
        ("<|observation|>",154829,true), ("<think>",154841,false), ("</think>",154842,false),
        ("<tool_call>",154843,false), ("</tool_call>",154844,false),
        ("<tool_response>",154845,false), ("</tool_response>",154846,false),
        ("<arg_key>",154847,false), ("</arg_key>",154848,false),
        ("<arg_value>",154849,false), ("</arg_value>",154850,false),
    ] {
        assert_eq!(engine_tokenizer::Tokenizer::encode(&model, text).unwrap(), [id]);
        assert_eq!(model.is_special(id), special);
        assert_eq!(model.decode(&[id], false).unwrap(), text);
        assert_eq!(model.decode(&[id], true).unwrap(), if special { "" } else { text });
        assert!(!Tokenizer::encode(&model, text).unwrap().contains(&id), "plain request must protect caller controls");
    }
}

struct FixedToken(u32);
impl spark_serve::Session for FixedToken {
    fn step(&mut self, _: u32) -> Result<Vec<f32>, String> {
        let mut row = vec![0.0; 154880]; row[self.0 as usize] = 1.0; Ok(row)
    }
    fn truncate(&mut self, _: usize) -> Result<(), String> { Ok(()) }
}
impl spark_serve::Engine for FixedToken {
    type Session<'a> = FixedToken;
    fn session(&self) -> Result<FixedToken, String> { Ok(FixedToken(self.0)) }
    fn context_limit(&self) -> usize { 8192 }
    fn vocab_size(&self) -> usize { 154880 }
    fn eos_token_ids(&self) -> &[u32] { &[154820,154827,154829] }
}
#[test]
fn each_checkpoint_eos_stops_the_shared_host_generator() {
    use spark_serve::generation::{self, Event, FinishReason};
    for id in [154820,154827,154829] {
        let mut events = Vec::new();
        let usage = generation::generate(&FixedToken(id), &[3346], 3, &Default::default(), || Ok(()), |e| { events.push(e); Ok(()) }).unwrap();
        assert_eq!(usage.completion_tokens, 1);
        assert!(!events.iter().any(|e| matches!(e, Event::Token(_))));
        assert_eq!(events.last(), Some(&Event::Finished(FinishReason::Stop, usage)));
    }
}

#[test]
fn text_checks_do_not_open_executable_model_admission() {
    let config = include_bytes!("../../model/tests/fixtures/glm/config.json");
    let plan = engine_model::plan::ConfigPlan::from_config(config).unwrap();
    assert!(matches!(&plan, engine_model::plan::ConfigPlan::Deferred(_)));
    assert!(plan.into_model_plan().is_err());
    assert!(engine_model::plan::ModelPlan::from_config(config).is_err());
}

#[test]
#[ignore = "needs the pinned public artifacts in .local/text-contract"]
fn chat_bytes_and_ids_match_source_vectors() {
    let model = checkpoint();
    let prefix = [154822,154824,154826,25062,287,29905,371,25,7487];
    for (body, tail, bytes) in [
        (r#"{"messages":[{"role":"user","content":""}]}"#,
            vec![154827,154828,154841], "<|user|><|assistant|><think>"),
        (r#"{"messages":[{"role":"user","content":"雪"},{"role":"assistant","content":" answer ","reasoning_content":"é"}]}"#,
            vec![154827,99785,154828,154841,963,154842,9214,154828,154841],
            "<|user|>雪<|assistant|><think>é</think>answer<|assistant|><think>"),
        (r#"{"messages":[{"role":"user","content":"go"},{"role":"developer","content":""},{"role":"system","content":""}]}"#,
            vec![154827,3346,154826,154826,154828,154841], "<|user|>go<|system|><|system|><|assistant|><think>"),
    ] {
        let input = chat(body);
        let expected = format!("[gMASK]<sop><|system|>Reasoning Effort: Max{bytes}");
        assert_eq!(model.format_chat_input(&input, 8192).unwrap().as_bytes(), expected.as_bytes());
        assert_eq!(model.encode_chat(&input, 8192).unwrap(), [prefix.as_slice(),tail.as_slice()].concat());
    }
}

#[test]
#[ignore = "needs the pinned public artifacts in .local/text-contract"]
fn chat_decoder_preserves_the_pinned_output_grammar() {
    use spark_serve::chat_output::{Parser,Event};
    let model = checkpoint(); assert!(model.glm_chat());
    let prompt = model.encode_chat(&chat(r#"{"messages":[{"role":"user","content":"go"}]}"#),8192).unwrap();
    // The pinned template always opens <think>; it has no thinking-off form, so the request asks for none.
    assert!(model.chat_thinking_open(&prompt));
    let mut decoder = model.chat_decoder();
    let mut parser = Parser::new(Default::default(),vec!["f".into()]).unwrap()
        .glm(model.chat_thinking_open(&prompt),Default::default());
    // Vocabulary entries and control IDs are recorded by the source vector test.
    let ids = engine_tokenizer::Tokenizer::encode(&model,"雪</think>ok<tool_call>f<arg_key>x</arg_key><arg_value>é</arg_value></tool_call>").unwrap();
    let mut events = Vec::new();
    for id in ids {events.extend(parser.push(Decoder::push(&mut decoder,id).unwrap().as_bytes()).unwrap());}
    events.extend(parser.push(Decoder::finish(&mut decoder).unwrap().as_bytes()).unwrap()); events.extend(parser.finish().unwrap());
    let reasoning:String = events.iter().filter_map(|e|if let Event::Reasoning(s)=e{Some(s.as_str())}else{None}).collect();
    let content:String = events.iter().filter_map(|e|if let Event::Content(s)=e{Some(s.as_str())}else{None}).collect();
    assert_eq!(reasoning,"雪"); assert_eq!(content,"ok");
    assert_eq!(events.last(),Some(&Event::ToolCall{name:"f".into(),arguments:"{\"x\":\"é\"}".into()}));
}
