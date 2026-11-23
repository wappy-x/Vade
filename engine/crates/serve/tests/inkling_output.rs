//! The reply state machine on token IDs and its parser, at every chunk cut.
//! The replies are token IDs that the maker's reference code gave. The expected
//! reasoning and answer are cut out of the reference's own text of the same reply.
//! See fixtures/inkling-chat/README.md.
use spark_serve::chat_output::{ChannelIds, Channels, Event, Limits, Parser};
use std::collections::HashMap;

const VOCABULARY: &str = include_str!("fixtures/inkling-chat/vocabulary.tsv");
const REPLIES: &str = include_str!("fixtures/inkling-chat/reference-replies.jsonl");
const TOOL_REPLIES: &str = include_str!("fixtures/inkling-chat/reference-tool-replies.jsonl");
/// IDs for the hand-made cases. The vocabulary file has no bytes for them.
const PAD: u32 = 199_999;
const TOOL_ERROR: u32 = 200_022;
const CALL: u32 = 200_049;
const TOOL_TEXT: u32 = 200_057;
const BYTE_TOKENS: u32 = 1_000_000;
const TOOLS: [&str; 2] = ["get_weather", "save_note"];

struct Fixture { bytes: HashMap<u32, Vec<u8>>, control: HashMap<String, u32>, ids: ChannelIds }
fn fixture() -> Fixture {
    let (mut bytes, mut control) = (HashMap::new(), HashMap::new());
    for line in VOCABULARY.lines() {
        let mut fields = line.split('\t');
        let (id, hex): (u32, &str) = (fields.next().unwrap().parse().unwrap(), fields.next().unwrap());
        let token: Vec<u8> = (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect();
        if fields.next() == Some("control") { control.insert(String::from_utf8(token).unwrap(), id); } else { bytes.insert(id, token); }
    }
    // One stand-in token for each byte, so that a hand-made reply can hold any text. `word` takes
    // a token of the reference files first. These IDs are above the vocabulary of the checkpoint.
    for byte in 0..=255u8 { bytes.insert(BYTE_TOKENS + u32::from(byte), vec![byte]); }
    let id = |spelling: &str| control[spelling];
    let ids = ChannelIds { message: id("<|message_model|>"), text: vec![id("<|content_text|>"), TOOL_ERROR, TOOL_TEXT], reasoning: id("<|content_thinking|>"),
        call: CALL, end_message: id("<|end_message|>"), stop: id("<|content_model_end_sampling|>") };
    Fixture { bytes, control, ids }
}
impl Fixture {
    /// The first ordinary tokens whose bytes are exactly `text`: enough to write hand-made replies.
    fn word(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text.as_bytes();
        while !rest.is_empty() {
            let (id, token) = self.bytes.iter().filter(|(_, token)| rest.starts_with(token)).max_by_key(|(id, token)| (token.len(), std::cmp::Reverse(**id)))
                .unwrap_or_else(|| panic!("no token for {:?}", String::from_utf8_lossy(rest)));
            out.push(*id);
            rest = &rest[token.len()..];
        }
        out
    }
    /// The marked text of each token, as the reply decoder gives it to the parser.
    /// The stand-in decoder shows no control token and holds incomplete UTF-8, as the tokenizer's decoder does.
    fn decode(&self, tokens: &[u32]) -> Vec<String> {
        let mut channels = Channels::new(self.ids.clone());
        let mut pending: Vec<u8> = Vec::new();
        let mut decode = |token: Option<u32>| -> Result<String, String> {
            match token {
                Some(token) => match self.bytes.get(&token) {
                    Some(token) => pending.extend_from_slice(token),
                    None => return Ok(String::new()),
                },
                None => return Ok(String::from_utf8_lossy(&std::mem::take(&mut pending)).into_owned()),
            }
            let valid = match std::str::from_utf8(&pending) { Ok(_) => pending.len(), Err(error) => error.valid_up_to() };
            Ok(String::from_utf8(pending.drain(..valid).collect()).unwrap())
        };
        let mut out: Vec<String> = tokens.iter().map(|&token| channels.push(token, &mut decode).unwrap()).collect();
        let tail = decode(None).unwrap();
        out.push(channels.finish(tail));
        out
    }
}
/// Feed chunks, finish, and join the text deltas of each kind.
fn events(chunks: &[&[u8]]) -> Vec<Event> { events_with(&[], Limits::default(), chunks).unwrap() }
fn events_with(tools: &[&str], limits: Limits, chunks: &[&[u8]]) -> Result<Vec<Event>, String> {
    let mut parser = Parser::new(limits, tools.iter().map(|name| name.to_string()).collect())?.token_channels();
    let mut events = Vec::new();
    for chunk in chunks { events.extend(parser.push(chunk)?); }
    events.extend(parser.finish()?);
    let mut joined: Vec<Event> = Vec::new();
    for event in events {
        match (joined.last_mut(), &event) {
            (Some(Event::Content(a)), Event::Content(b)) | (Some(Event::Reasoning(a)), Event::Reasoning(b)) => a.push_str(b),
            _ => joined.push(event),
        }
    }
    Ok(joined)
}
/// The three events of one complete call.
fn call(name: &str, arguments: &str) -> [Event; 3] {
    [Event::ToolStart { name: name.into() }, Event::ToolArguments(arguments.into()), Event::ToolCall { name: name.into(), arguments: arguments.into() }]
}
fn expected(reasoning: &str, content: &str) -> Vec<Event> {
    let mut out = Vec::new();
    if !reasoning.is_empty() { out.push(Event::Reasoning(reasoning.into())); }
    if !content.is_empty() { out.push(Event::Content(content.into())); }
    out
}
/// Every cut of the marked text in two, the text token by token, and the text byte by byte.
fn every_cut(marked: &[String], expected: &[Event], name: &str) -> usize { every_cut_with(&[], marked, expected, name) }
fn every_cut_with(tools: &[&str], marked: &[String], expected: &[Event], name: &str) -> usize {
    let whole = marked.concat();
    let bytes = whole.as_bytes();
    let events = |chunks: &[&[u8]]| events_with(tools, Limits::default(), chunks).unwrap();
    for cut in 0..=bytes.len() { assert_eq!(events(&[&bytes[..cut], &bytes[cut..]]), expected, "{name} cut={cut}"); }
    assert_eq!(events(&marked.iter().map(|text| text.as_bytes()).collect::<Vec<_>>()), expected, "{name} by token");
    assert_eq!(events(&bytes.chunks(1).collect::<Vec<_>>()), expected, "{name} by byte");
    bytes.len() + 3
}
#[test]
fn reference_replies_at_every_chunk_cut() {
    let fixture = fixture();
    let (mut replies, mut cuts) = (0, 0);
    for line in REPLIES.lines().filter(|line| !line.trim().is_empty()) {
        let root = engine_model::json::parse(line.as_bytes(), engine_model::json::Limits { bytes: 1 << 20, string_bytes: 1 << 20, ..Default::default() }).unwrap();
        let reply = root.object().unwrap();
        let name = reply["name"].string().unwrap();
        let ids: Vec<u32> = reply["ids"].array().unwrap().iter().map(|id| id.u64().unwrap() as u32).collect();
        let expected = expected(reply["reasoning"].string().unwrap(), reply["content"].string().unwrap());
        let marked = fixture.decode(&ids);
        cuts += every_cut(&marked, &expected, name);
        // No control spelling is shown.
        for event in &expected { let (Event::Content(text) | Event::Reasoning(text)) = event else { panic!() }; assert!(!text.contains("<|"), "{name}"); }
        replies += 1;
    }
    println!("inkling_reply_goldens replies={replies} chunkings={cuts}");
    assert_eq!(replies, 7);
}
#[test]
fn reference_tool_replies_at_every_chunk_cut() {
    // Replies that the model wrote for requests with tools (the maker's reference code, greedy). The expected
    // events were made from the same token IDs by a second writing of the rules: see the fixture README.
    let fixture = fixture();
    let (mut replies, mut calls, mut with_calls, mut cut_in_a_call, mut cuts) = (0, 0, 0, 0, 0);
    for line in TOOL_REPLIES.lines().filter(|line| !line.trim().is_empty()) {
        let root = engine_model::json::parse(line.as_bytes(), engine_model::json::Limits { bytes: 1 << 20, string_bytes: 1 << 20, ..Default::default() }).unwrap();
        let reply = root.object().unwrap();
        let name = reply["name"].string().unwrap();
        let ids: Vec<u32> = reply["ids"].array().unwrap().iter().map(|id| id.u64().unwrap() as u32).collect();
        let tools: Vec<&str> = reply["tools"].array().unwrap().iter().map(|tool| tool.string().unwrap()).collect();
        let mut expected = Vec::new();
        for event in reply["events"].array().unwrap() {
            let fields: Vec<&str> = event.array().unwrap().iter().map(|field| field.string().unwrap()).collect();
            match fields[..] {
                ["reasoning", text] => expected.push(Event::Reasoning(text.into())),
                ["content", text] => expected.push(Event::Content(text.into())),
                ["call", name, arguments] => { expected.extend(call(name, arguments)); calls += 1; }
                _ => panic!("{name}: {fields:?}"),
            }
        }
        let marked = fixture.decode(&ids);
        cuts += every_cut_with(&tools, &marked, &expected, name);
        // No control spelling is shown, in a text or in the arguments of a call.
        for event in &expected {
            let (Event::Content(text) | Event::Reasoning(text) | Event::ToolArguments(text) | Event::ToolStart { name: text } | Event::ToolCall { arguments: text, .. }) = event;
            assert!(!text.contains("<|"), "{name}");
        }
        // A reply that the length limit cut inside a call has the call token and no call: its body is answer text.
        let has_call_token = ids.contains(&CALL);
        let has_call = expected.iter().any(|event| matches!(event, Event::ToolCall { .. }));
        if has_call { with_calls += 1; } else if has_call_token { cut_in_a_call += 1; assert_eq!(reply["finish"].string().unwrap(), "length", "{name}"); }
        // With no declared tool, the same reply gives no call: each call body is answer text.
        let plain = events(&[marked.concat().as_bytes()]);
        assert!(plain.iter().all(|event| matches!(event, Event::Content(_) | Event::Reasoning(_))), "{name}");
        assert_eq!(plain.iter().any(|event| matches!(event, Event::Content(text) if text.contains("\"args\""))), has_call_token, "{name}");
        replies += 1;
    }
    println!("inkling_tool_reply_goldens replies={replies} with_calls={with_calls} calls={calls} cut_in_a_call={cut_in_a_call} chunkings={cuts}");
    assert_eq!((replies, with_calls, calls, cut_in_a_call), (18, 8, 10, 2));
}
#[test]
fn stop_is_the_end_of_the_turn_and_the_end_of_a_message_is_not() {
    let f = fixture();
    let id = |spelling: &str| f.control[spelling];
    let (model, text, thinking, end, stop) = (id("<|message_model|>"), id("<|content_text|>"), id("<|content_thinking|>"), id("<|end_message|>"), id("<|content_model_end_sampling|>"));
    let reply = |parts: &[&[u32]]| f.decode(&parts.concat());
    let hello = f.word("Hello");
    let user = f.word(" user");
    // Two answer messages join with nothing between them: the end of a message does not stop the reply.
    let marked = reply(&[&[model, text], &hello, &[end, model, text], &user, &[end, stop]]);
    every_cut(&marked, &expected("", "Hello user"), "two messages");
    // Reasoning in two messages joins the same way, and the first message can come with no message token.
    // The reference library's own parsing test has this form (tests/utils/test_chat_parsing.py, the multi-block case).
    let marked = reply(&[&[thinking], &hello, &[end, model, thinking], &user, &[end, model, text], &hello, &[end, stop]]);
    every_cut(&marked, &expected("Hello user", "Hello"), "reasoning in two messages");
    // Nothing after the end of the turn is shown, and text outside a message is not shown.
    let marked = reply(&[&hello, &[model, text], &hello, &[end], &user, &[stop], &user, &[PAD]]);
    every_cut(&marked, &expected("", "Hello"), "outside text");
    // The end of the turn closes an open body. A new message after it is read again (a request can ignore EOS).
    let marked = reply(&[&[model, thinking], &hello, &[stop, model, text], &user, &[end]]);
    every_cut(&marked, &expected("Hello", " user"), "stop inside a body");
    // The name after the message token is not shown. A content token with no message token opens a body.
    let marked = reply(&[&[model], &user, &[text], &hello, &[end, thinking], &user, &[end, stop]]);
    assert_eq!(events(&[marked.concat().as_bytes()]), [Event::Content("Hello".into()), Event::Reasoning(" user".into())]);
    // Only the end of the message closes a body: a message or content token inside it adds no text and no cut.
    let marked = reply(&[&[model, thinking], &hello, &[model, text, PAD], &user, &[end, stop]]);
    every_cut(&marked, &expected("Hello user", ""), "controls inside a body");
    // The padding token is never shown and is no stop.
    let marked = reply(&[&[PAD, model, PAD, text, PAD], &hello, &[PAD], &user, &[PAD, end, PAD, stop, PAD]]);
    every_cut(&marked, &expected("", "Hello user"), "padding");
    // A reply that a length limit cut has no end token. Its text is kept.
    every_cut(&reply(&[&[model, thinking], &hello]), &expected("Hello", ""), "cut in reasoning");
    every_cut(&reply(&[&[model, thinking], &hello, &[end, model, text], &user]), &expected("Hello", " user"), "cut in the answer");
    every_cut(&reply(&[&[model, thinking], &hello, &[end, model]]), &expected("Hello", ""), "cut in a header");
    assert!(events(&[reply(&[]).concat().as_bytes()]).is_empty());
    // The other answer kinds of the reply grammar are answer text.
    for kind in [TOOL_ERROR, TOOL_TEXT] {
        every_cut(&reply(&[&[model, kind], &hello, &[end, stop]]), &expected("", "Hello"), "answer kind");
    }
    // A request with no declared tool has no call to make: a tool call body is answer text.
    every_cut(&reply(&[&[model], &user, &[CALL], &hello, &[end, stop]]), &expected("", "Hello"), "tool call body");
}
#[test]
fn tool_calls_at_every_chunk_cut() {
    let f = fixture();
    let id = |spelling: &str| f.control[spelling];
    let (model, text, thinking, end, stop) = (id("<|message_model|>"), id("<|content_text|>"), id("<|content_thinking|>"), id("<|end_message|>"), id("<|content_model_end_sampling|>"));
    let reply = |parts: &[&[u32]]| f.decode(&parts.concat());
    let w = |text: &str| f.word(text);
    let (mut cases, mut cuts) = (0, 0);
    let mut check = |name: &str, marked: Vec<String>, expected: Vec<Event>| { cuts += every_cut_with(&TOOLS, &marked, &expected, name); cases += 1; };
    let weather = r#"{"name":"get_weather","args":{"city":"Tokyo","units":"C"}}"#;
    let arguments = r#"{"city":"Tokyo","units":"C"}"#;
    // The reply of the reference library's own parsing test (tests/utils/test_chat_parsing.py, the multi-block
    // case): reasoning in two messages, an answer, one call. The library gives the same reasoning, answer and
    // name there, and the arguments as the object that this text holds.
    let mut all = expected("Consider the weather. Tokyo, probably.", "Checking the weather now.");
    all.extend(call("get_weather", arguments));
    check("the form of the reference library", reply(&[&[thinking], &w("Consider the weather."), &[end, model, thinking], &w(" Tokyo, probably."),
        &[end, model, text], &w("Checking the weather now."), &[end, model], &w("get_weather"), &[CALL], &w(weather), &[end, stop]]), all);
    // One call and nothing else. The end of the turn also ends its message.
    check("one call", reply(&[&[model], &w("get_weather"), &[CALL], &w(weather), &[end, stop]]), call("get_weather", arguments).to_vec());
    check("a call that the end of the turn closes", reply(&[&[model], &w("get_weather"), &[CALL], &w(weather), &[stop]]), call("get_weather", arguments).to_vec());
    // Two calls in one turn, in the order of the reply.
    let note = r#"{"name":"save_note","args":{}}"#;
    check("two calls", reply(&[&[model], &w("get_weather"), &[CALL], &w(weather), &[end, model], &w("save_note"), &[CALL], &w(note), &[end, stop]]),
        [call("get_weather", arguments), call("save_note", "{}")].concat());
    // Answer text before and after a call keeps its place.
    let mut around = vec![Event::Content("before".into())];
    around.extend(call("save_note", "{}"));
    around.push(Event::Content("after".into()));
    check("text around a call", reply(&[&[model, text], &w("before"), &[end, model], &w("save_note"), &[CALL], &w(note), &[end, model, text], &w("after"), &[end, stop]]), around);
    // The arguments are the text that the model wrote: spaces, member order and escapes stay.
    let spaced = "{ \"args\" :\n { \"b\" : 1 , \"a\" : [ 1 , { \"k\" : \"}]\\\"\" } ] } ,\t\"name\" : \"save_note\" }\n";
    check("spaces and member order", reply(&[&[model], &w("save_note"), &[CALL], &w(spaced), &[end, stop]]),
        call("save_note", "{ \"b\" : 1 , \"a\" : [ 1 , { \"k\" : \"}]\\\"\" } ] }").to_vec());
    // Text in an argument: quotes, braces, non-ASCII, the spelling of a control token as ordinary text, and
    // the noncharacter that the marks use.
    let odd_arguments = "{\"text\":\"Grüße \\\"q\\\" \\\\ {a} [b], 雪 \\u00e9 <|end_message|><|content_model_end_sampling|> \u{fdd0}\u{fdd4} done\"}";
    let odd = format!("{{\"name\":\"save_note\",\"args\":{odd_arguments}}}");
    check("odd text in an argument", reply(&[&[model], &w("save_note"), &[CALL], &w(&odd), &[end, stop]]), call("save_note", odd_arguments).to_vec());
    // Other members of the object are not read. A member name can be written with an escape.
    check("other members", reply(&[&[model], &w("get_weather"), &[CALL], &w(r#"{"id":7,"args":{"n":[1,2]},"note":{"args":1},"name":"get_weather"}"#), &[end, stop]]),
        call("get_weather", r#"{"n":[1,2]}"#).to_vec());
    check("an escaped member name", reply(&[&[model, CALL], &w("{\"n\\u0061me\":\"save_note\",\"\\u0061rgs\":{\"x\":null}}"), &[end, stop]]), call("save_note", r#"{"x":null}"#).to_vec());
    // The name comes from the object, as in the reply grammar. The name in the message header is not read:
    // a message with no header, with no message token, or with another name in its header gives the same call.
    for (name, header) in [("no name in the header", vec![model]), ("no message token", vec![]), ("another name in the header", [&[model][..], &w("save_note")].concat())] {
        check(name, reply(&[&header, &[CALL], &w(weather), &[end, stop]]), call("get_weather", arguments).to_vec());
    }
    // A control token inside the body adds no text and does not end the body.
    let half = weather.len() / 2;
    check("controls inside a call", reply(&[&[model, CALL], &w(&weather[..half]), &[PAD, model, text, thinking, CALL], &w(&weather[half..]), &[end, stop]]), call("get_weather", arguments).to_vec());
    println!("inkling_tool_call_goldens replies={cases} chunkings={cuts}");
    assert_eq!(cases, 13);
}
#[test]
fn a_body_that_is_no_call_of_a_declared_tool_is_answer_text() {
    let f = fixture();
    let id = |spelling: &str| f.control[spelling];
    let (model, text, end, stop) = (id("<|message_model|>"), id("<|content_text|>"), id("<|end_message|>"), id("<|content_model_end_sampling|>"));
    let reply = |parts: &[&[u32]]| f.decode(&parts.concat());
    let w = |text: &str| f.word(text);
    let weather = r#"{"name":"get_weather","args":{"city":"Tokyo"}}"#;
    let (mut cases, mut cuts) = (0, 0);
    // Each body ends with the end of its message, and is shown as it is.
    for body in [
        "get_weather(city=Tokyo)", r#"{"name":"get_weather","args":{"city":"To"#, r#"{"name":"get_weather","args":{"city":"Tokyo"}"#, "{", "null", "7", r#""get_weather""#,
        // not a declared tool
        r#"{"name":"delete_all","args":{}}"#, r#"{"name":"Get_Weather","args":{}}"#, r#"{"name":"","args":{}}"#,
        // no name, or a name that is no text
        r#"{"args":{}}"#, r#"{"name":7,"args":{}}"#, r#"{"name":null,"args":{}}"#, r#"{"name":["get_weather"],"args":{}}"#,
        // no arguments, or arguments that are no object
        r#"{"name":"get_weather"}"#, r#"{"name":"get_weather","args":null}"#, r#"{"name":"get_weather","args":"{}"}"#, r#"{"name":"get_weather","args":[]}"#,
        r#"{"name":"get_weather","args":1}"#, r#"{"name":"get_weather","arguments":{}}"#,
        // one object only, with each member once
        r#"{"name":"get_weather","args":{}}{"name":"save_note","args":{}}"#, r#"[{"name":"get_weather","args":{}}]"#, r#"{"name":"get_weather","args":{}} and text"#,
        r#"{"name":"get_weather","name":"save_note","args":{}}"#, r#"{"name":"get_weather","args":{},"args":{}}"#, r#"{"name":"get_weather","args":{"a":1,"a":2}}"#,
        r#"{"name":"get_weather","args":{"a":01}}"#, r#"{"name":"get_weather","args":{'a':1}}"#, r#"{name:"get_weather",args:{}}"#,
    ] {
        cuts += every_cut_with(&TOOLS, &reply(&[&[model], &w("get_weather"), &[CALL], &w(body), &[end, stop]]), &expected("", body), body);
        cases += 1;
    }
    // A body that is only spaces is shown too. An empty body shows nothing.
    cuts += every_cut_with(&TOOLS, &reply(&[&[model, CALL], &w(" \n"), &[end, stop]]), &expected("", " \n"), "spaces");
    cuts += every_cut_with(&TOOLS, &reply(&[&[model, CALL, end, stop]]), &[], "empty body");
    // A length limit or a stop string cut the reply inside a call body: no end token. The body is answer
    // text, also when its JSON is whole. Only the end of its message makes a call.
    for body in [&weather[..20], weather] {
        cuts += every_cut_with(&TOOLS, &reply(&[&[model, text], &w("I look it up."), &[end, model], &w("get_weather"), &[CALL], &w(body)]),
            &expected("", &format!("I look it up.{body}")), "cut in a call body");
    }
    // A request with no declared tool: the body of a whole call is answer text.
    cuts += every_cut(&reply(&[&[model], &w("get_weather"), &[CALL], &w(weather), &[end, stop]]), &expected("", weather), "no declared tool");
    // A broken body does not stop the next call from being read.
    let mut both = expected("", "{");
    both.extend(call("get_weather", r#"{"city":"Tokyo"}"#));
    cuts += every_cut_with(&TOOLS, &reply(&[&[model, CALL], &w("{"), &[end, model, CALL], &w(weather), &[end, stop]]), &both, "a call after a broken body");
    println!("inkling_tool_call_bad_input bodies={} chunkings={cuts}", cases + 6);
}
#[test]
fn tool_call_limits_and_the_single_call_policy() {
    let f = fixture();
    let id = |spelling: &str| f.control[spelling];
    let (model, text, end, stop) = (id("<|message_model|>"), id("<|content_text|>"), id("<|end_message|>"), id("<|content_model_end_sampling|>"));
    let weather = r#"{"name":"get_weather","args":{"city":"Tokyo"}}"#;
    let one = [&[model, CALL][..], &f.word(weather), &[end]].concat();
    let marked = |tokens: &[u32]| f.decode(tokens).concat();
    // A call body over the byte limit of one call fails the request, at the byte that passes the limit.
    let small = |bytes: usize| Limits { tool_call_bytes: bytes, ..Limits::default() };
    assert_eq!(events_with(&TOOLS, small(weather.len()), &[marked(&one).as_bytes()]).unwrap(), call("get_weather", r#"{"city":"Tokyo"}"#));
    assert_eq!(events_with(&TOOLS, small(weather.len() - 1), &[marked(&one).as_bytes()]).unwrap_err(), "tool call byte limit exceeded");
    assert_eq!(events_with(&TOOLS, small(8), &[marked(&[&[model, CALL][..], &f.word("0123456789")].concat()).as_bytes()]).unwrap_err(), "tool call byte limit exceeded");
    // With no declared tool the body is answer text, and the limit of one call does not apply.
    assert_eq!(events_with(&[], small(8), &[marked(&one).as_bytes()]).unwrap(), expected("", weather));
    // More calls than the limit of one reply fails the request.
    let three = marked(&[&one[..], &one, &one, &[stop]].concat());
    let count = |calls: usize| Limits { tool_calls: calls, ..Limits::default() };
    assert_eq!(events_with(&TOOLS, count(3), &[three.as_bytes()]).unwrap().len(), 9);
    assert_eq!(events_with(&TOOLS, count(2), &[three.as_bytes()]).unwrap_err(), "tool call count limit exceeded");
    // A body that is no call does not count.
    let broken = marked(&[&[model, CALL][..], &f.word("{"), &[end], &one, &[stop]].concat());
    assert_eq!(events_with(&TOOLS, count(1), &[broken.as_bytes()]).unwrap().len(), 4);
    // A request that allows one call: the parser stops at the end of the first call, at every cut.
    let reply = marked(&[&[model, text][..], &f.word("I look it up."), &[end], &one, &one, &[model, text], &f.word("after"), &[end, stop]].concat());
    let mut first = expected("", "I look it up.");
    first.extend(call("get_weather", r#"{"city":"Tokyo"}"#));
    for cut in 0..=reply.len() {
        let mut parser = Parser::new(Limits::default(), TOOLS.map(String::from).to_vec()).unwrap().tool_policy("", false, true).unwrap().token_channels();
        let mut events = parser.push(&reply.as_bytes()[..cut]).unwrap();
        events.extend(parser.push(&reply.as_bytes()[cut..]).unwrap());
        assert!(parser.stopped(), "cut={cut}");
        events.extend(parser.finish().unwrap());
        let mut joined: Vec<Event> = Vec::new();
        for event in events { match (joined.last_mut(), &event) { (Some(Event::Content(a)), Event::Content(b)) => a.push_str(b), _ => joined.push(event) } }
        assert_eq!(joined, first, "cut={cut}");
    }
}
#[test]
fn bytes_of_the_reply_are_kept_and_broken_input_is_an_error() {
    let f = fixture();
    let id = |spelling: &str| f.control[spelling];
    let (model, text, end) = (id("<|message_model|>"), id("<|content_text|>"), id("<|end_message|>"));
    // A character in two tokens is one character. A body that ends inside a character gives the replacement character.
    let dash: Vec<u32> = f.bytes.iter().filter(|(_, token)| token.as_slice() == "—".as_bytes()).map(|(id, _)| *id).collect();
    assert!(!dash.is_empty(), "the vocabulary file holds the dash of a reference reply");
    let mut channels = Channels::new(f.ids.clone());
    let mut halves = ["\u{fdd0}é".as_bytes()[..4].to_vec(), "\u{fdd0}é".as_bytes()[4..].to_vec(), vec![0xe9, 0x9b]].into_iter();
    let mut pending = Vec::new();
    let mut decode = |token: Option<u32>| -> Result<String, String> {
        if token.is_some() { pending.extend(halves.next().unwrap()); }
        let valid = match (token, std::str::from_utf8(&pending)) { (None, _) | (_, Ok(_)) => pending.len(), (_, Err(error)) => error.valid_up_to() };
        Ok(String::from_utf8_lossy(&pending.drain(..valid).collect::<Vec<_>>()).into_owned())
    };
    let mut marked = String::new();
    for token in [model, text, 1, 2, 3, end] { marked.push_str(&channels.push(token, &mut decode).unwrap()); }
    // The noncharacter of the marks is text when the model writes it.
    let expected = expected("", "\u{fdd0}é\u{fffd}");
    for cut in 0..=marked.len() { assert_eq!(events(&[&marked.as_bytes()[..cut], &marked.as_bytes()[cut..]]), expected, "cut={cut}"); }
    // A decoder error fails the push. A mark that the machine never writes fails the parser, with no panic.
    let mut channels = Channels::new(f.ids.clone());
    channels.push(text, |_| Ok(String::new())).unwrap();
    assert_eq!(channels.push(1, |_| Err("decode failed".into())).unwrap_err(), "decode failed");
    let mut parser = Parser::new(Limits::default(), Vec::new()).unwrap().token_channels();
    assert!(parser.push("\u{fdd0}x".as_bytes()).is_err());
    assert!(parser.push(b"x").is_err() && parser.finish().is_err());
    // Text with no mark is in no message: nothing is shown. A mark cut in two at the end shows nothing.
    assert!(events(&[b"plain text"]).is_empty());
    assert_eq!(events(&["\u{fdd0}\u{fdd1}a\u{fdd0}".as_bytes()]), [Event::Content("a".into())]);
    assert!(Parser::new(Limits::default(), Vec::new()).unwrap().token_channels().push(&[0xff]).is_err());
    // Every prefix of a marked reply is a prefix of its events: a stop string or a cancel can end it at any byte.
    let hello = f.word("Hello—Hello");
    let marked = f.decode(&[&[model, id("<|content_thinking|>")][..], &hello, &[end, model, text], &hello, &[end]].concat()).concat();
    for cut in (0..=marked.len()).filter(|&cut| marked.is_char_boundary(cut)) {
        let shown: String = events(&[&marked.as_bytes()[..cut]]).iter().map(|event| match event { Event::Content(text) | Event::Reasoning(text) => text.as_str(), _ => "" }).collect();
        assert!("Hello—HelloHello—Hello".starts_with(&shown), "cut={cut}: {shown:?}");
    }
}
