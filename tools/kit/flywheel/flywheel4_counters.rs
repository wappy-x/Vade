use super::*;
const RECORDS:[&str;5]=["COUNTS","PROJECTION_DISPATCH","RECORDED_KINDS","EXCHANGE_STAGE","DEFER_GATHER"];
fn variable(record:&str,key:&str)->bool {
    match record {
        "COUNTS"=>matches!(key,"resident_launch_copy"|"resident_launch_copy_bytes"|"resident_launch_copy_words"|"resident_launch_copy_words_bytes"|"resident_launch_zero_words"|"resident_launch_zero_words_bytes"|"resident_merged_zero_launches"|"resident_merged_zero_bytes"),
        "RECORDED_KINDS"=>matches!(key,"resident_launch_copy"|"resident_launch_copy_words"|"resident_launch_zero_words"|"zero_fill_launches"|"zero_fill_rows"|"zero_fill_words"|"resident_launch_put"|"resident_launch_put_wide"|"resident_launch_causal"|"resident_launch_causal_fast"),
        "DEFER_GATHER"=>key!="decode_steps" && key!="scope",
        _=>false,
    }
}
#[derive(Clone,Debug)] pub(super) struct Rule {cut:String,input:usize,repeat:usize,record:String,key:String,kind:String,low:String,high:String}
fn number(value:&str)->Option<u64> {(!value.is_empty() && value.bytes().all(|byte|byte.is_ascii_digit())).then(||value.parse().ok()).flatten()}
fn encode(rules:&[Rule])->String {
    let mut result=String::from("cut\tinput\trun\trecord\tfield\trule\tlow\thigh\n");
    for rule in rules {result.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",rule.cut,rule.input,rule.repeat,rule.record,rule.key,rule.kind,rule.low,rule.high));}result
}
pub(super) fn load(path:&Path)->Result<Vec<Rule>> {
    let content=text(path)?;let mut rows=content.lines();require(rows.next()==Some("cut\tinput\trun\trecord\tfield\trule\tlow\thigh"),"counter declaration header differs")?;
    let mut rules=Vec::new();let mut keys=std::collections::BTreeSet::new();
    for line in rows {let row=line.split('\t').collect::<Vec<_>>();require(row.len()==8,"malformed counter declaration")?;
        let input=row[1].parse::<usize>().map_err(|_|"invalid declaration input")?;let repeat=row[2].parse::<usize>().map_err(|_|"invalid declaration run")?;
        require(CUTS.iter().any(|cut|cut.id==row[0]) && [15,127].contains(&input) && (1..=3).contains(&repeat) && RECORDS.contains(&row[3]),"unknown counter declaration")?;
        require(keys.insert((row[0].to_owned(),input,repeat,row[3].to_owned(),row[4].to_owned())),"duplicate counter declaration")?;
        require(["head","nonincreasing","defer-change","report-only","integer-range","literal"].contains(&row[5]),"unknown counter rule")?;
        if row[5]=="report-only" {require(row[3]=="COUNTS" && ["peak_bytes_after_prompt","peak_bytes_after_decode"].contains(&row[4]),"only peak memory may be report-only")?;}
        if row[5]=="nonincreasing" {require(row[0]=="defer" && (row[3]=="COUNTS" && row[4].starts_with("live_bytes_") || row[3]=="EXCHANGE_STAGE" && ["accepted","original_messages","local_words"].contains(&row[4]) || row[3]=="PROJECTION_DISPATCH" && row[4]=="causal_fast"),"undeclared decreasing field")?;}
        if row[5]=="defer-change" {require(row[0]=="defer" && variable(row[3],row[4]),"undeclared varying field")?;}
        if row[5]=="integer-range" {require(number(row[6]).zip(number(row[7])).is_some_and(|(low,high)|low<=high),"invalid integer bounds")?;}
        rules.push(Rule {cut:row[0].into(),input,repeat,record:row[3].into(),key:row[4].into(),kind:row[5].into(),low:row[6].into(),high:row[7].into()});
    }require(!rules.is_empty(),"counter declarations are empty")?;Ok(rules)
}
pub(super) fn declare(config:&Config)->Result<()> {
    let mut rules=Vec::new();
    for cut in CUTS {for job in config.jobs() {for repeat in 1..=3 {for record in RECORDS {
        let mut reference=if record=="DEFER_GATHER" {[("decode_steps",(job.output-1).to_string()),("fold_accepted","0".into()),("fold_fallback","0".into()),("fold_removed_local_words","0".into()),("middle_accepted","0".into()),("middle_fallback","0".into()),("middle_removed_local_words","0".into()),("scope","commands_completed_by_final_sync".into())].into_iter().map(|(key,value)|(key.to_owned(),value)).collect()} else {fields(&config.root.join("reference").join(&config.boxes[0]).join(job.name).join(format!("run{repeat}/{record}")))?};
        if record=="RECORDED_KINDS" {for name in ["resident_launch_mhc_rms","resident_launch_mhc_dots","resident_launch_mhc_finish","resident_launch_exchange"] {reference.insert(name.into(),"0".into());}}
        for key in reference.keys() {
            let kind=if record=="COUNTS" && key.starts_with("peak_bytes_") {"report-only"}
                else if cut.id=="defer" && (record=="COUNTS" && key.starts_with("live_bytes_") || record=="EXCHANGE_STAGE" && ["accepted","original_messages","local_words"].contains(&key.as_str()) || record=="PROJECTION_DISPATCH" && key=="causal_fast") {"nonincreasing"}
                else if cut.id=="defer" && variable(record,key) {"defer-change"}
                else {"head"};
            rules.push(Rule {cut:cut.id.into(),input:job.input,repeat,record:record.into(),key:key.clone(),kind:kind.into(),low:"0".into(),high:"0".into()});
        }
    }}}}save(config.root.join("COUNTER-DECLARATIONS.tsv"),encode(&rules))?;
    let run2=RUN2.as_str();save(config.root.join("COUNTER-SOURCE.txt"),format!("Frozen before run 4. Source: lane-build-tickets-2026-11-07.md DEFER-GATHER and source 15a38b3dafc5612cfa244c2f619f8278dd13c650. Run-two memory evidence: {run2}/DIFFERENCES.txt. conv/head remain strict except peak high-water marks. DEFER changes only named copy and zero-fill counts/bytes, narrow/wide put shares with total puts conserved, and six admission-work counters. These explicitly named values may change in either direction and must remain unsigned integers. EXCHANGE_STAGE accepted/original_messages/local_words and live bytes cannot increase. Stage payload savings must cover the declared removed words. Captures/replays/waits, take/scatter counts, weight projection dispatch, and every other field stay equal. Admission must enable the cut and accept at least one operation; removed words and accepted counts agree, every work count is a whole number of decode steps. Head live bytes restore exactly. No post-run widening.\n"))?;Ok(())
}
pub(super) fn counters(rules:&[Rule],variant:&Variant,job:&Job,repeat:usize,run:&Path,head:&Path)->Result<Vec<String>> {
    let selected=rules.iter().filter(|rule|rule.cut==variant.cut.id && rule.input==job.input && rule.repeat==repeat).collect::<Vec<_>>();
    require(!selected.is_empty(),"counter declaration is missing")?;let mut differences=Vec::new();
    for record in RECORDS {
        let fields=fields(&run.join(record))?;let baseline=super::fields(&head.join(record))?;
        let declared=selected.iter().filter(|rule|rule.record==record).collect::<Vec<_>>();
        require(fields.len()==declared.len() && declared.iter().all(|rule|fields.contains_key(&rule.key)),format!("{record}: undeclared or missing counter"))?;
        for rule in declared {let value=&fields[&rule.key];let expected=baseline.get(&rule.key).ok_or("head counter missing")?;
            let matches=match rule.kind.as_str() {
                "head"=>value==expected,"literal"=>value==&rule.low,"report-only"=>number(value).is_some(),
                "integer-range"=>number(value).is_some_and(|value|value>=number(&rule.low).unwrap() && value<=number(&rule.high).unwrap()),
                "nonincreasing"=>number(value).zip(number(expected)).is_some_and(|(value,head)|value<=head),"defer-change"=>number(value).is_some(),_=>false,
            };if !matches {differences.push(format!("{record}.{}={} head={} rule={}",rule.key,value,expected,rule.kind));}
        }
        if variant.cut.id=="defer" && record=="RECORDED_KINDS" {
            let puts=integer(&fields,"resident_launch_put")?+integer(&fields,"resident_launch_put_wide")?;
            let before=integer(&baseline,"resident_launch_put")?+integer(&baseline,"resident_launch_put_wide")?;
            if before!=puts {differences.push(format!("put conservation: {before} != {puts}"));}
            let fast=integer(&fields,"resident_launch_causal_fast")?;let generic=integer(&fields,"resident_launch_causal")?;
            let head_fast=integer(&baseline,"resident_launch_causal_fast")?;let head_generic=integer(&baseline,"resident_launch_causal")?;
            let work=super::fields(&run.join("DEFER_GATHER"))?;let accepted=integer(&work,"fold_accepted")?.checked_add(integer(&work,"middle_accepted")?).ok_or("accepted count overflow")?;
            let projection=super::fields(&run.join("PROJECTION_DISPATCH"))?;let head_projection=super::fields(&head.join("PROJECTION_DISPATCH"))?;
            if fast.checked_add(generic)!=head_fast.checked_add(head_generic) || head_fast.checked_sub(fast).is_none_or(|changed|changed>accepted || Some(changed)!=integer(&head_projection,"causal_fast").ok().and_then(|head|head.checked_sub(integer(&projection,"causal_fast").ok()?))) {differences.push("checked causal mix does not conserve calls or match dispatch".into());}
        }
        if variant.cut.id=="defer" && record=="EXCHANGE_STAGE" {
            let work=super::fields(&run.join("DEFER_GATHER"))?;let removed=integer(&work,"fold_removed_local_words")?.checked_add(integer(&work,"middle_removed_local_words")?).ok_or("removed word count overflow")?;
            if integer(&baseline,"local_words")?.checked_sub(integer(&fields,"local_words")?).is_none_or(|saved|saved<removed) {differences.push("stage words do not cover declared payload savings".into());}
        }
    }Ok(differences)
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn only_peaks_may_be_report_only() {
        let root=std::env::temp_dir().join(format!("flywheel4-rules-{}",std::process::id()));fs::create_dir(&root).unwrap();let path=root.join("rules");
        let prefix="cut\tinput\trun\trecord\tfield\trule\tlow\thigh\n";
        save(&path,format!("{prefix}head\t15\t1\tCOUNTS\tgraph_captures\treport-only\t0\t0\n")).unwrap();assert!(load(&path).is_err());
        save(&path,format!("{prefix}head\t15\t1\tCOUNTS\tpeak_bytes_after_prompt\treport-only\t0\t0\n")).unwrap();assert!(load(&path).is_ok());
        save(&path,format!("{prefix}conv\t15\t1\tCOUNTS\tlive_bytes_after_prompt\tnonincreasing\t0\t0\n")).unwrap();assert!(load(&path).is_err());fs::remove_dir_all(root).unwrap();
    }
}
