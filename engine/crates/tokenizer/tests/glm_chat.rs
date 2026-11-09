//! Real checkpoint template and a vocabulary slice with unchanged token IDs.
use engine_tokenizer::{Tokenizer, chat::{self, Message, Options, ReasoningEffort, Role}};
use std::{path::{Path, PathBuf}, sync::atomic::{AtomicUsize, Ordering}};
const TEMPLATE: &str = include_str!("../templates/glm.jinja");
const HAND: &str = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Say hello.<|assistant|><think>";
// Jinja whitespace control joins all literals. Vocabulary pieces, in order:
// [gMASK], <sop>, <|system|>, Reason, ing, ' Eff', ort, :, ' Max',
// <|user|>, Say, ' hello', ., <|assistant|>, <think>.
const IDS: &[u32] = &[154822,154824,154826,25062,287,29905,371,25,7487,154827,45494,23745,13,154828,154841];
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!("vade-glm-chat-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("tokenizer.json"),include_bytes!("glm-tokenizer-slice.json")).unwrap();
        std::fs::write(path.join("chat_template.jinja"),TEMPLATE).unwrap();
        std::fs::write(path.join("tokenizer_config.json"),"{}").unwrap();
        Self(path)
    }
    fn load(&self) -> Tokenizer { Tokenizer::from_directory(&self.0).unwrap() }
}
impl Drop for Fixture { fn drop(&mut self) { std::fs::remove_dir_all(&self.0).unwrap(); } }
fn check(path: &Path) {
    let t = Tokenizer::from_directory(path).unwrap();
    let messages = [Message::new(Role::User,"Say hello.")];
    for enable_thinking in [true,false] {
        let options = Options { enable_thinking, ..t.chat_options() };
        assert_eq!(t.format_chat(&messages,&options).unwrap(),HAND);
        assert_eq!(chat::encode(&t,&messages,&options).unwrap(),IDS);
        assert_eq!(t.encode(HAND).unwrap(),IDS);
    }
    let pieces = ["[gMASK]","<sop>","<|system|>","Reason","ing"," Eff","ort",":"," Max","<|user|>","Say"," hello",".","<|assistant|>","<think>"];
    for (&id,piece) in IDS.iter().zip(pieces) { assert_eq!(t.token_bytes(id).unwrap(),piece.as_bytes()); }
}
#[test]
fn real_template_plain_turn_has_exact_checkpoint_ids() { check(&Fixture::new().0); }
#[test]
fn template_and_controls_are_both_required() {
    let f = Fixture::new();
    let text = include_str!("glm-tokenizer-slice.json").replace("<|assistant|>","<|other|>");
    std::fs::write(f.0.join("tokenizer.json"),text).unwrap();
    assert!(f.load().format_chat(&[Message::new(Role::User,"Say hello.")],&Options::default()).is_err());
    std::fs::write(f.0.join("tokenizer.json"),include_bytes!("glm-tokenizer-slice.json")).unwrap();
    std::fs::write(f.0.join("chat_template.jinja"),format!("{TEMPLATE}\n")).unwrap();
    assert!(Tokenizer::from_directory(&f.0).is_err());
    std::fs::write(f.0.join("chat_template.jinja"),TEMPLATE).unwrap();
    std::fs::write(f.0.join("tokenizer_config.json"),r#"{"chat_template":"different"}"#).unwrap();
    assert!(Tokenizer::from_directory(&f.0).is_err());
}
#[test]
fn plain_turns_preserve_whitespace_and_template_options() {
    let f = Fixture::new(); let t = f.load();
    let messages = [Message::new(Role::System," rules "),Message::new(Role::User," Say hello.\n")];
    let mut o = Options { reasoning_effort: ReasoningEffort::Low, add_generation_prompt: false, ..Default::default() };
    assert_eq!(t.format_chat(&messages,&o).unwrap(),"[gMASK]<sop><|system|>Reasoning Effort: Low<|system|> rules <|user|> Say hello.\n");
    o.reasoning_effort = ReasoningEffort::Medium;
    assert!(t.format_chat(&messages,&o).unwrap().contains("Reasoning Effort: Max"));
    o.max_bytes = 1; assert!(t.format_chat(&messages,&o).is_err());
    o = Options::default(); o.tools.push("{}".into()); assert!(t.format_chat(&messages,&o).is_err());
    assert!(t.format_chat(&[Message::new(Role::Assistant,"hello")],&Options::default()).is_err());
}
