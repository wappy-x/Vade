mod common;
use common::*;
use std::{fs,process::Command};
use vade_tools::{json,rigmark_dashboard};

const TEMPLATE: &str = r#"<style></style>
  <section class="panel card">
    <div class="card-head">
      <div>
        <h2>Over time</h2>
const RAW = /*__DATA__*/[];
const GENERATED = /*__GENERATED__*/"";
const DATA = RAW.map(r => {
  const o = { ...r };
});
const COLS = [
];
    <div class="t-row"><span>Engine</span>
  drawLegend(); drawTime();
"#;
const PAYLOAD: &str = r#"[{"sparks":2,"prose_tok_s":54.153,"ttft_8k_s":-0.00,"c8_tok_s":4.10e+2,"boxes":"m0+m1","status":"PASS"},{"sparks":8,"prose_tok_s":null,"status":"FAIL"}]"#;
const SETS: &str = r#"{"sets":{"2A":["m0","m1"],"2B":["m2","m3"],"4H":["m4","m5","m6","m7"],"4B":["m0","m1","m2","m3"]},"cards":{"day/run/card.json":["m1","m0"]}}"#;
fn page() -> String { TEMPLATE.replace("/*__DATA__*/[]",PAYLOAD).replace("/*__GENERATED__*/\"\"","\"Nov 2, 15:00 CST\"") }

#[test] fn keeps_card_data_and_build_label_byte_for_byte() {
    let (template,page,count) = rigmark_dashboard::render(TEMPLATE,&page(),SETS).unwrap();
    assert_eq!(count,2);
    assert!(page.contains(&format!("const RAW = {PAYLOAD};\n")));
    assert!(page.contains("const GENERATED = \"Nov 2, 15:00 CST\";\n"));
    assert!(template.contains("const RAW = /*__DATA__*/[];"));
    assert!(template.contains("const GENERATED = /*__GENERATED__*/\"\";"));
    assert!(page.contains("[1, 2, 4, 8].map(count"));
    assert!(page.contains("label: \"Set\""));
    assert!(page.contains("drawSparkColumns(); drawLegend(); drawTime();"));
    assert!(page.contains("members.length === names.length"));
    assert!(page.contains("r.host === \"rented\""));
    assert!(page.contains("${esc(r.set_name)}"));
    assert!(page.contains("const CARD_MEMBERS = {\"day/run/card.json\":[\"m1\",\"m0\"]};"));
    let (again,rebuilt,_) = rigmark_dashboard::render(&template,&page,SETS).unwrap();
    assert_eq!(again,template); assert_eq!(rebuilt,page);
}

#[test] fn refreshes_saved_sets_without_adding_columns_or_changing_cards() {
    let (template,page,_) = rigmark_dashboard::render(TEMPLATE,&page(),SETS).unwrap();
    let changed = SETS.replace("\"2A\"","\"Pair\"");
    let (updated,rebuilt,_) = rigmark_dashboard::render(&template,&page,&changed).unwrap();
    assert_eq!(updated.matches("id=\"spark-columns\"").count(),1);
    assert!(updated.contains("\"Pair\":[\"m0\",\"m1\"]"));
    assert!(!updated.contains("\"2A\""));
    assert!(rebuilt.contains(&format!("const RAW = {PAYLOAD};\n")));
    assert!(rigmark_dashboard::render(&template.replace("const CARD_SETS = ","missing = "),&page,SETS).is_err());
}

#[test] fn rejects_ambiguous_sets_and_escapes_script_text() {
    for invalid in ["null","{}",r#"{"sets":{}}"#,r#"{"sets":{"2A":[]}}"#,r#"{"sets":{"2A":["m0","M0"]}}"#,
        r#"{"sets":{"2A":["m0","m1"],"2B":["M1","M0"]}}"#,r#"{"sets":{"2A":[1]}}"#,
        r#"{"sets":{"2A":["m0+m1"]}}"#,r#"{"sets":{"": ["m0"]}}"#,
        r#"{"sets":{"2A":["m0","m1"]},"cards":[]}"#,
        r#"{"sets":{"2A":["m0","m1"]},"cards":{"day/card":["m0","M0"]}}"#] {
        assert!(rigmark_dashboard::render(TEMPLATE,&page(),invalid).is_err(),"{invalid}");
    }
    let name = "</script><script>throw 1</script>";
    let sets = format!("{{\"sets\":{{{}:[\"m0\",\"m1\"]}}}}",json::quote(name));
    let (_,page,_) = rigmark_dashboard::render(TEMPLATE,&page(),&sets).unwrap();
    assert!(!page.contains(name)); assert!(page.contains("\\u003c/script\\u003e"));
}

#[test] fn invalid_input_never_writes_and_existing_output_is_safe() {
    let t = Temp::new("dashboard");
    let template = t.0.join("template.html"); let built = t.0.join("built.html"); let sets = t.0.join("sets.json"); let out = t.0.join("preview");
    write(&template,TEMPLATE); write(&built,page()); write(&sets,SETS);
    let invoke = || Command::new(env!("CARGO_BIN_EXE_vade-tools")).arg("rigmark-dashboard")
        .args([&template,&built,&sets,&out]).output().unwrap();
    let original = fs::read(&built).unwrap();
    for anchor in ["</style>","const DATA = RAW.map(r => {","const COLS = [","/*__DATA__*/[]"] {
        write(&template,TEMPLATE.replace(anchor,"missing"));
        let result = invoke(); assert!(!result.status.success()); assert!(!out.exists());
    }
    write(&template,TEMPLATE);
    for broken in [page().replace("const RAW = ","const missing = "),page().replace(PAYLOAD,"null"),page().replace("\"Nov 2, 15:00 CST\"","42")] {
        write(&built,broken); assert!(!invoke().status.success()); assert!(!out.exists());
    }
    write(&built,&original);
    let result = invoke(); assert!(result.status.success(),"{}",diagnostic(&result));
    assert_eq!(fs::read(&built).unwrap(),original);
    let preview = fs::read(out.join("RigMark Dashboard.html")).unwrap();
    assert!(!invoke().status.success()); assert_eq!(fs::read(out.join("RigMark Dashboard.html")).unwrap(),preview);
}
