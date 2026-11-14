//! Published vLLM prompt bytes, tested without executing its implementation.
//! These are renderer checks. They are not checkpoint token-ID or HTTP proofs.
use super::*;
#[allow(clippy::duplicate_mod)] // the shared file is included on purpose.
#[path = "../../../../tools/export/src/sha256.rs"] mod sha256;

const FIXTURE:&str = include_str!("../../tests/fixtures/glm-upstream/vllm-test-cases.json");

fn truth(value:&Value) -> bool {
    match value {Value::Null=>false,Value::Bool(v)=>*v,Value::String(v)=>!v.is_empty(),
        Value::Array(v)=>!v.is_empty(),Value::Object(v)=>!v.is_empty(),Value::Number(v)=>v.parse::<f64>().unwrap()!=0.0}
}
fn messages(source:&str) -> Vec<Message> {
    raw_array(source).unwrap().iter().map(|raw| {
        let value=object(raw,1024*1024).unwrap(); let fields=value.object().unwrap();
        let role=match fields["role"].string().unwrap() {"user"=>Role::User,"assistant"=>Role::Assistant,
            "system"=>Role::System,"developer"=>Role::Developer,"tool"=>Role::Tool,_=>panic!("unsupported fixture role")};
        // The typed renderer accepts strings. Null and missing source content
        // become the same empty string emitted by the upstream template.
        let content=fields.get("content").filter(|v|**v!=Value::Null).map(|v|v.string().unwrap()).unwrap_or("");
        let mut message=Message::new(role,content);
        message.reasoning_content=fields.get("reasoning_content").and_then(|v|v.string().ok()).map(str::to_owned);
        message.tool_call_id=fields.get("tool_call_id").filter(|v|v.string().is_ok_and(|s|!s.is_empty()))
            .or_else(||fields.get("id")).and_then(|v|v.string().ok()).map(str::to_owned);
        if let Some(calls)=raw_member(raw,"tool_calls").unwrap() {
            for call in raw_array(calls).unwrap() {
                let call_value=object(&call,1024*1024).unwrap(); let call_fields=call_value.object().unwrap();
                let function=raw_member(&call,"function").unwrap().unwrap();
                let function_value=object(function,1024*1024).unwrap(); let function_fields=function_value.object().unwrap();
                let arguments=&function_fields["arguments"];
                message.tool_calls.push(ToolCall {name:function_fields["name"].string().unwrap().into(),
                    arguments:arguments.string().map(str::to_owned).unwrap_or_else(|_|raw_member(function,"arguments").unwrap().unwrap().into()),
                    id:call_fields.get("id").and_then(|v|v.string().ok()).map(str::to_owned)});
            }
        }
        message
    }).collect()
}

#[test]
fn published_vllm_text_prompt_bytes_match_typed_renderer() {
    assert_eq!(sha256::digest(FIXTURE.as_bytes()),"8660be3b335be039a57af9e180403be0baee8cd1455f151d7297a20285f39497");
    let cases=raw_array(raw_member(FIXTURE,"cases").unwrap().unwrap()).unwrap();
    let (mut compared,mut refused,mut excluded)=(0,0,0);
    for case in &cases {
        let value=object(case,1024*1024).unwrap(); let fields=value.object().unwrap();
        let name=fields["name"].string().unwrap();
        // These source cases use inputs outside the typed, text-only API.
        let reason=if name.starts_with("continue_") {Some("assistant continuation is not exposed")}
            else {match name {
                "system_developer_custom"=>Some("custom roles have no typed representation"),
                "content_parts"=>Some("multimodal input has no text-only representation"),
                "bundled"|"references"|"bundled_references"|"reference_without_tools"|"bad_tool_content"=>Some("structured tool output has no string representation"),
                _=>None,
            }};
        if let Some(reason)=reason {println!("EXCLUDED {name}: {reason}");excluded+=1;continue;}
        let kwargs=fields["kwargs"].object().unwrap();
        let options=glm::Options {reasoning_effort:kwargs.get("reasoning_effort").and_then(|v|v.string().ok()).unwrap_or("max").into(),
            clear_thinking:kwargs.get("clear_thinking").is_some_and(truth),
            add_generation_prompt:fields["add_generation_prompt"].boolean().unwrap(),
            tools:raw_array(raw_member(case,"tools").unwrap().unwrap()).unwrap(),..Default::default()};
        let actual=glm::format(&messages(raw_member(case,"messages").unwrap().unwrap()),&options);
        if fields.contains_key("error") {
            assert_eq!(fields["error"].string().unwrap(),"ValueError","{name}");
            let error=actual.unwrap_err();
            let reason=match name {"bad_arguments_invalid"=>"expected JSON value",
                "bad_arguments_None"|"bad_arguments_[]"|"bad_arguments_5"|"bad_tool_definition"=>"expected JSON object",
                _=>panic!("unclassified refusal {name}")};
            assert!(error.contains(reason),"{name}: {error}"); refused+=1;
        }
        else {assert_eq!(actual.unwrap(),fields["expected"].string().unwrap(),"{name}");compared+=1;}
    }
    println!("UPSTREAM GLM PROMPTS: compared={compared} refused={refused} excluded={excluded} total={}",cases.len());
    assert_eq!((compared,refused,excluded,cases.len()),(45,6,22,73));
}
