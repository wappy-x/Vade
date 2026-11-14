//! Synthetic artifact checks. No checkpoint vocabulary is required.
use engine_tokenizer::{Limits, Tokenizer};

const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const MERGES: [(&str, &str); 10] = [("h","e"),("l","l"),("he","ll"),("Ġ","w"),("1","2"),("12","3"),("4","5"),("45","6"),("123","4"),("Ã","©")];
#[derive(Clone, Copy)]
enum Profile { Qwen, Phi, Llama, Gemma, ByteLevel }
fn quote(text: &str) -> String { engine_tokenizer::chat::quote(text) }
fn alphabet() -> Vec<char> {
    let mut extra = 256;
    (0..256u32).map(|b| {
        if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) { char::from_u32(b).unwrap() }
        else { extra += 1; char::from_u32(extra - 1).unwrap() }
    }).collect()
}
fn bytelevel(prefix: bool, trim: bool, regex: bool) -> String {
    format!(r#"{{"type":"ByteLevel","add_prefix_space":{prefix},"trim_offsets":{trim},"use_regex":{regex}}}"#)
}
fn artifact(profile: Profile) -> String {
    let gemma = matches!(profile, Profile::Gemma);
    let mut vocab = if gemma {
        let mut entries = (0..256).map(|b| format!(r#""<0x{b:02X}>":{}"#, 1000+b)).collect::<Vec<_>>();
        entries.extend((33..=126u32).map(|b| format!("{}:{}", quote(&char::from_u32(b).unwrap().to_string()), 2000+b)));
        entries.push(r#""▁":2032"#.into());
        entries
    } else {
        alphabet().iter().enumerate().map(|(id,c)| format!("{}:{id}",quote(&c.to_string()))).collect()
    };
    let merges = MERGES.iter().enumerate().filter_map(|(i, &(a,b))| {
        if gemma && i == 9 { return None; }
        let a = if gemma && a == "Ġ" { "▁" } else { a };
        vocab.push(format!("{}:{}",quote(&format!("{a}{b}")),if gemma {2300+i} else {256+i}));
        Some(format!("[{},{}]",quote(a),quote(b)))
    }).collect::<Vec<_>>().join(",");
    if !gemma {
        vocab.push(r#""solo":512"#.into());
        let alphabet = alphabet();
        for (i,text) in ["١٢٣","١٢٣٤","Ⅳ²٣","Ⅳ²٣𝟜"].iter().enumerate() {
            let spelling: String = text.bytes().map(|b| alphabet[b as usize]).collect();
            vocab.push(format!("{}:{}",quote(&spelling),513+i));
        }
    }
    let controls: &[(&str,u32)] = match profile {
        Profile::Qwen => &[("<|endoftext|>",151643),("<|im_start|>",151644),("<|im_end|>",151645)],
        Profile::Phi => &[("<|endoftext|>",100257),("<|im_start|>",100264),("<|im_end|>",100265),("<|im_sep|>",100266)],
        Profile::Llama => &[("<|begin_of_text|>",128000),("<|end_of_text|>",128001),("<|start_header_id|>",128006),("<|end_header_id|>",128007),("<|eom_id|>",128008),("<|eot_id|>",128009)],
        Profile::Gemma => &[("<pad>",0),("<eos>",1),("<bos>",2),("<unk>",3),("<|turn>",105),("<turn|>",106)],
        Profile::ByteLevel => &[("<|endoftext|>",154820),("[gMASK]",154822),("<sop>",154824),("<|user|>",154827),("<|observation|>",154829),("<think>",154841)],
    };
    let strip = matches!(profile, Profile::Phi);
    let added = controls.iter().map(|(text,id)| format!(r#"{{"id":{id},"content":{},"single_word":false,"lstrip":{strip},"rstrip":{strip},"normalized":false,"special":{}}}"#,quote(text),*text != "<think>")).collect::<Vec<_>>().join(",");
    let normalizer = match profile { Profile::Qwen => r#"{"type":"NFC"}"#, Profile::Gemma => r#"{"type":"Replace","pattern":{"String":" "},"content":"▁"}"#, _ => "null" };
    let pattern = if matches!(profile, Profile::Qwen) { PATTERN.to_owned() } else { PATTERN.replace(r"\p{N}|",r"\p{N}{1,3}|") };
    let pre = if gemma { r#"{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false}"#.to_owned() } else {
        format!(r#"{{"type":"Sequence","pretokenizers":[{{"type":"Split","pattern":{{"Regex":{}}},"behavior":{},"invert":{strip}}},{}]}}"#,quote(&pattern),quote(if strip {"Removed"} else {"Isolated"}),bytelevel(false,!matches!(profile,Profile::Qwen),false))
    };
    let template = r#"{"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}"#;
    let post = match profile {
        Profile::Qwen => bytelevel(false,false,false), Profile::Phi => "null".into(),
        Profile::Llama => format!(r#"{{"type":"Sequence","processors":[{},{template}]}}"#,bytelevel(true,true,true)),
        Profile::Gemma => template.into(), Profile::ByteLevel => bytelevel(true,false,true),
    };
    let decoder = if gemma { r#"{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]}"#.to_owned() } else { bytelevel(!matches!(profile,Profile::Qwen),!matches!(profile,Profile::Qwen),!matches!(profile,Profile::Qwen)) };
    let ignore = matches!(profile,Profile::Llama|Profile::ByteLevel);
    let unk = if gemma {quote("<unk>")} else {"null".into()};
    format!(r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{added}],"normalizer":{normalizer},"pre_tokenizer":{pre},"post_processor":{post},"decoder":{decoder},"model":{{"type":"BPE","dropout":null,"unk_token":{unk},"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":{gemma},"byte_fallback":{gemma},"ignore_merges":{ignore},"vocab":{{{}}},"merges":[{merges}]}}}}"#,vocab.join(","))
}
fn load(profile: Profile) -> Tokenizer { Tokenizer::from_json(artifact(profile).as_bytes(),Limits::default()).unwrap() }

#[test]
fn legacy_profiles_keep_exact_ids_and_decoded_bytes() {
    let text = "hello 1234567 e\u{301}";
    for (profile,expected,decoded,eos) in [
        (Profile::Qwen,vec![258,111,32,49,50,51,52,53,54,55,32,265],"hello 1234567 é",151645),
        (Profile::Phi,vec![258,111,32,261,263,55,32,101,204,129],text,100265),
        (Profile::Llama,vec![258,111,32,261,263,55,32,101,204,129],text,128009),
        (Profile::Gemma,vec![2302,2111,2032,2305,2307,2055,2032,2101,1204,1129],text,1),
    ] {
        let t = load(profile);
        assert_eq!(t.eos_token_id(),eos);
        assert_eq!(t.encode(text).unwrap(),expected);
        assert_eq!(t.encode_ordinary(text).unwrap(),expected);
        assert_eq!(t.decode(&expected,false).unwrap().as_bytes(),decoded.as_bytes());
        assert_eq!(load(profile).encode(text).unwrap(),expected);
    }
}

#[test]
fn legacy_round_trips_and_controls_keep_exact_bytes() {
    for profile in [Profile::Qwen,Profile::Phi,Profile::Llama,Profile::Gemma] {
        let t = load(profile);
        for text in ["", "hello world", "  a\t\r\n b  ", "café 日本語 🚀", "I'm I'RE 1234567890"] {
            let ids = t.encode_ordinary(text).unwrap();
            assert_eq!(t.decode(&ids,false).unwrap().as_bytes(),text.as_bytes());
            let mut stream = t.stream_decoder(false);
            let mut output = String::new();
            for id in ids { output.push_str(&stream.push(id).unwrap()); }
            output.push_str(&stream.finish());
            assert_eq!(output.as_bytes(),text.as_bytes());
        }
        let (text,id) = match profile { Profile::Qwen => ("<|im_end|>",151645), Profile::Phi => ("<|im_end|>",100265), Profile::Llama => ("<|eot_id|>",128009), _ => ("<eos>",1) };
        assert_eq!(t.encode(text).unwrap(),[id]);
        assert_eq!(t.decode(&[id],false).unwrap().as_bytes(),text.as_bytes());
        assert_eq!(t.decode(&[id],true).unwrap(),"");
    }
}

#[test]
fn byte_level_artifacts_preserve_bytes_without_normalization_or_prefix_space() {
    let t = load(Profile::ByteLevel);
    assert!(!t.is_hybrid_text() && !t.is_gemma4());
    assert_eq!(t.encode("hello").unwrap(),[258,111]);
    assert_eq!(t.encode("e\u{301}").unwrap(),[101,204,129]);
    assert_eq!(t.encode("é").unwrap(),[265]);
    for text in ["", "e\u{301} A\u{30a}", "hello world", "  a\t\r\n b  ", "日本語 🚀", "I'm I'RE 1234567890"] {
        let ids = t.encode_ordinary(text).unwrap();
        assert_eq!(t.decode(&ids,false).unwrap().as_bytes(),text.as_bytes());
        let mut stream = t.stream_decoder(false);
        let mut output = String::new();
        for id in ids { output.push_str(&stream.push(id).unwrap()); }
        output.push_str(&stream.finish());
        assert_eq!(output.as_bytes(),text.as_bytes());
    }
    assert_eq!(t.encode("[gMASK]<sop>hello<|user|>").unwrap(),[154822,154824,258,111,154827]);
    assert!(t.encode_ordinary("[gMASK]<sop>").unwrap().iter().all(|&id| !t.is_special(id)));
    assert_eq!(t.decode(&[154822,154841,154827],true).unwrap(),"<think>");
    // Control names and IDs do not select the byte-level codec.
    let other = artifact(Profile::ByteLevel).replace("<|endoftext|>","[stop]").replace("154820","600");
    let other = Tokenizer::from_json(other.as_bytes(),Limits::default()).unwrap();
    assert_eq!(other.encode("[stop]hello").unwrap(),[600,258,111]);
}

#[test]
fn digit_groups_follow_the_pattern_and_do_not_cross_pre_tokens() {
    let t = load(Profile::ByteLevel);
    assert_eq!(t.encode("1234567890").unwrap(),[261,263,55,56,57,48]);
    assert_eq!(t.encode("1234").unwrap(),[261,52],"the vocabulary has 1234, but the split ends at 123");
    for (text,expected) in [("1",vec![49]),("12",vec![260]),("123",vec![261]),("١٢٣٤",vec![513,217,164]),("Ⅳ²٣𝟜",vec![515,240,157,159,156])] {
        assert_eq!(t.encode(text).unwrap(),expected);
        assert_eq!(t.decode(&expected,false).unwrap(),text);
    }
    let single = artifact(Profile::ByteLevel).replace(r"\\p{N}{1,3}|",r"\\p{N}|");
    let single = Tokenizer::from_json(single.as_bytes(),Limits::default()).unwrap();
    assert_eq!(single.encode("1234").unwrap(),[49,50,51,52]);
    assert!(!t.shares_token_ids(&single));
    let grouped_qwen = artifact(Profile::Qwen).replace(r"\\p{N}|",r"\\p{N}{1,3}|");
    let grouped_qwen = Tokenizer::from_json(grouped_qwen.as_bytes(),Limits::default()).unwrap();
    assert_eq!(grouped_qwen.encode("1234").unwrap(),[261,52]);
    assert!(!load(Profile::Qwen).shares_token_ids(&grouped_qwen));
}

#[test]
fn legacy_digit_patterns_keep_three_digit_groups() {
    let grouped = PATTERN.replace(r"\p{N}|",r"\p{N}{1,3}|");
    let hybrid = PATTERN.replace(r"\p{L}+",r"[\p{L}\p{M}]+").replace(r"[^\s\p{L}\p{N}]",r"[^\s\p{L}\p{M}\p{N}]");
    for profile in [Profile::Phi,Profile::Llama] {
        // The prior loader accepted each pattern and always grouped three digits.
        for pattern in [PATTERN,grouped.as_str(),hybrid.as_str()] {
            let json = artifact(profile).replace(&quote(&grouped),&quote(pattern));
            assert!(json.contains(&quote(pattern)),"fixture lost {pattern}");
            let t = Tokenizer::from_json(json.as_bytes(),Limits::default()).unwrap();
            for (text,expected) in [("1234",vec![261,52]),("1234567",vec![261,263,55])] {
                assert_eq!(t.encode(text).unwrap(),expected,"pattern {pattern}");
                assert_eq!(t.encode_ordinary(text).unwrap(),expected);
                assert_eq!(t.decode(&expected,false).unwrap(),text);
            }
            if pattern == PATTERN { assert!(t.shares_token_ids(&load(profile))); }
        }
    }
}

#[test]
fn ignore_merges_uses_model_vocabulary_and_obeys_token_limits() {
    let mut t = load(Profile::ByteLevel);
    assert_eq!(t.encode("solo").unwrap(),[512],"a vocabulary entry needs no merge rule");
    let ranked = artifact(Profile::ByteLevel).replace(r#""ignore_merges":true"#,r#""ignore_merges":false"#);
    let ranked = Tokenizer::from_json(ranked.as_bytes(),Limits::default()).unwrap();
    assert_eq!(ranked.encode("solo").unwrap(),[115,111,108,111]);
    assert!(!t.shares_token_ids(&ranked));
    assert!(t.shares_token_ids(&load(Profile::ByteLevel)));
    t.set_limits(Limits { max_input_bytes:1024,max_tokens:1 }).unwrap();
    assert_eq!(t.encode("solo").unwrap(),[512]);
    assert!(t.encode("solo solo").unwrap_err().0.contains("token count limit"));
    assert!(t.encode("1234").unwrap_err().0.contains("token count limit"));
}

#[test]
fn byte_level_artifacts_reject_unsupported_text_changes() {
    let good = artifact(Profile::ByteLevel);
    for (from,to) in [
        (r#""normalizer":null"#,r#""normalizer":{"type":"NFKC"}"#),
        (r"\\p{N}{1,3}|",r"\\p{N}{1,4}|"),
        (r#""behavior":"Isolated""#,r#""behavior":"Removed""#),
        (r#""invert":false"#,r#""invert":true"#),
        (r#""add_prefix_space":false"#,r#""add_prefix_space":true"#),
        (r#""byte_fallback":false"#,r#""byte_fallback":true"#),
        (r#""ignore_merges":true"#,r#""ignore_merges":"true""#),
        (r#""lstrip":false"#,r#""lstrip":true"#),
    ] {
        assert!(good.contains(from),"fixture lost {from}");
        assert!(Tokenizer::from_json(good.replacen(from,to,1).as_bytes(),Limits::default()).is_err(),"accepted {to}");
    }
}

#[test]
fn malformed_byte_mappings_fail_with_precise_reasons() {
    let good=artifact(Profile::ByteLevel);
    for (from,to,reason) in [
        (r#""!":33"#,r#""!":34"#,"duplicate or oversized token ID"),
        (r#""!":33"#,r#""!":33,"!":33"#,"duplicate JSON object key"),
        (r#""!":33"#,r#""!":1000000"#,"duplicate or oversized token ID"),
        (r#""!":33"#,r#""!":4294967296"#,"token ID overflow"),
        (r#""!":33"#,r#""!":-1"#,"expected nonnegative JSON integer"),
        (r#""!":33"#,r#""!":1.5"#,"expected nonnegative JSON integer"),
        (r#""!":33,"#,r#""#,"missing byte token"),
        (r#""id":154822"#,r#""id":154820"#,"duplicate or oversized added token ID"),
        (r#""id":154822"#,r#""id":33"#,"duplicate or oversized added token ID"),
        (r#""id":154822"#,r#""id":1000000"#,"duplicate or oversized added token ID"),
        (r#""id":154822"#,r#""id":4294967296"#,"added token ID overflow"),
        (r#""content":"[gMASK]""#,r#""content":"<sop>""#,"invalid added token text"),
        (r#""merges":["#,r#""merges":[["missing","e"],"#,"unknown left merge token"),
        (r#""merges":["#,r#""merges":[["h","missing"],"#,"unknown right merge token"),
        (r#""merges":["#,r#""merges":[["h","h"],"#,"unknown merged token"),
        (r#""merges":["#,r#""merges":[["h","e"],"#,"duplicate merge"),
    ] {
        assert_eq!(good.matches(from).count(),1,"ambiguous mutation: {from}");
        let bad=good.replacen(from,to,1);
        let error=match Tokenizer::from_json(bad.as_bytes(),Limits::default()) {
            Ok(_)=>panic!("accepted {from} -> {to}"),Err(error)=>error.0,
        };
        assert!(error.contains(reason),"{from} -> {to}: {error}");
    }
}

struct Directory(std::path::PathBuf);
impl Directory {
    fn new(profile: Profile, generation: Option<&str>) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let serial=NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("vade-tokenizer-fixture-{}-{stamp}-{serial}",std::process::id()));
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("tokenizer.json"),artifact(profile)).unwrap();
        if let Some(text) = generation { std::fs::write(path.join("generation_config.json"),text).unwrap(); }
        if matches!(profile,Profile::Qwen) {
            let config = format!(r#"{{"chat_template":{},"add_bos_token":false,"add_prefix_space":false,"clean_up_tokenization_spaces":false,"split_special_tokens":false,"eos_token":"<|im_end|>","pad_token":"<|endoftext|>"}}"#,quote(include_str!("../templates/qwen3.jinja")));
            std::fs::write(path.join("tokenizer_config.json"),config).unwrap();
        }
        Self(path)
    }
}
impl Drop for Directory { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

#[test]
fn directory_reads_eos_lists_and_single_ids_from_generation_config() {
    for (config,expected) in [
        (Some(r#"{"eos_token_id":[154820,154827,154829],"temperature":1.0}"#),vec![154820,154827,154829]),
        (Some(r#"{"eos_token_id":154827}"#),vec![154827]),
        (Some(r#"{"temperature":1.0}"#),vec![]),
        (None,vec![]),
    ] {
        let dir = Directory::new(Profile::ByteLevel,config);
        let t = Tokenizer::from_directory(&dir.0).unwrap();
        assert_eq!(t.eos_token_ids().unwrap(),expected);
        assert_eq!(t.eos_token_id(),expected.first().copied().unwrap_or(u32::MAX));
        assert_eq!(t.encode("hello").unwrap(),[258,111]);
        let file_only = Tokenizer::from_file(dir.0.join("tokenizer.json")).unwrap();
        assert!(file_only.eos_token_ids().unwrap().is_empty(),"the file loader has no companion settings");
    }
    for config in [None,Some(r#"{"temperature":1.0}"#)] {
        let dir = Directory::new(Profile::Qwen,config);
        assert_eq!(Tokenizer::from_directory(&dir.0).unwrap().eos_token_ids().unwrap(),[151645]);
    }
    let dir = Directory::new(Profile::Qwen,Some(r#"{"eos_token_id":[151643,151645]}"#));
    let t = Tokenizer::from_directory(&dir.0).unwrap();
    assert_eq!(t.eos_token_ids().unwrap(),[151643,151645]);
    assert_eq!(t.eos_token_id(),151645,"existing single-ID callers retain the chat EOS");
    assert_eq!(t.encode("hello").unwrap(),[258,111]);
}

#[test]
fn invalid_eos_settings_fail_only_when_the_list_is_read() {
    for profile in [Profile::ByteLevel,Profile::Qwen] {
        for config in [r#"{"eos_token_id":[]}"#,r#"{"eos_token_id":null}"#,r#"{"eos_token_id":-1}"#,r#"{"eos_token_id":1.5}"#,r#"{"eos_token_id":"154820"}"#,r#"{"eos_token_id":4294967296}"#,r#"{"eos_token_id":248044}"#,r#"{"eos_token_id":[154820,999999]}"#,r#"{"eos_token_id":[]} trailing"#,"[]"] {
            let dir = Directory::new(profile,Some(config));
            let t = Tokenizer::from_directory(&dir.0).unwrap();
            assert_eq!(t.encode("hello").unwrap(),[258,111]);
            assert!(t.eos_token_ids().is_err(),"accepted {config}");
            assert_eq!(t.eos_token_id(),if matches!(profile,Profile::Qwen) {151645} else {u32::MAX});
        }
    }
}

#[test]
fn generation_eos_file_is_read_once_on_first_list_access() {
    let dir = Directory::new(Profile::Qwen,Some("invalid JSON"));
    let t = Tokenizer::from_directory(&dir.0).unwrap();
    assert_eq!(t.eos_token_id(),151645,"legacy chat callers do not read the file");
    std::fs::write(dir.0.join("generation_config.json"),r#"{"eos_token_id":[151643,151645]}"#).unwrap();
    assert_eq!(t.eos_token_ids().unwrap(),[151643,151645]);
    std::fs::write(dir.0.join("generation_config.json"),"invalid JSON").unwrap();
    assert_eq!(t.eos_token_ids().unwrap(),[151643,151645]);
    let invalid = Tokenizer::from_directory(&dir.0).unwrap();
    assert!(invalid.eos_token_ids().is_err());
    std::fs::remove_file(dir.0.join("generation_config.json")).unwrap();
    assert!(invalid.eos_token_ids().is_err(),"errors are cached too");
}

#[test]
fn byte_level_chat_is_rejected() {
    let dir = Directory::new(Profile::ByteLevel,Some(r#"{"eos_token_id":[154820,154827,154829]}"#));
    let t = Tokenizer::from_directory(&dir.0).unwrap();
    let messages = [engine_tokenizer::chat::Message::new(engine_tokenizer::chat::Role::User,"hello")];
    let options = t.chat_options();
    assert!(t.format_chat(&messages,&options).unwrap_err().0.contains("chat template is unsupported"));
    assert!(engine_tokenizer::chat::encode(&t,&messages,&options).unwrap_err().0.contains("chat template is unsupported"));
}
