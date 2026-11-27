//! Compare complete timed token streams from two RigMark card receipts.
use std::{collections::BTreeMap,env,fs,process::ExitCode};
use vade_tools::json::{self,Value};

type Streams = BTreeMap<Vec<String>,Vec<u32>>;

fn read(path: &str) -> Result<Value,String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    json::parse(&text).map_err(|error| format!("{path}: {error}"))
}

fn integer(value: &Value) -> Result<usize,String> {
    match value {
        Value::Number(number) => number.parse().map_err(|_| "invalid integer".into()),
        _ => Err("expected integer".into()),
    }
}

fn runs(card: &Value, draft: bool) -> Result<BTreeMap<Vec<String>,usize>,String> {
    let mut expected = BTreeMap::new();
    let mut add = |path: &[&str], rows: &[Value], count: usize, index: usize| -> Result<(),String> {
        if rows.len() != count { return Err(format!("{}: expected {count} streams",path.join("/"))); }
        for (i,row) in rows.iter().enumerate() {
            let mut key = path.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
            key.insert(index,i.to_string());
            let place = key.join("/");
            if row.get("stream_done_marker") != &Value::Bool(true) {
                return Err(format!("{place}: stream incomplete"));
            }
            let count = integer(row.get("completion_tokens"))?;
            let omitted_eos = match row.get("finish_reason").text()? {
                "stop" => 1,
                "length" => 0,
                _ => return Err(format!("{place}: unknown finish reason")),
            };
            let length = count.checked_sub(omitted_eos).filter(|n| *n > 0).ok_or(format!("{place}: empty output"))?;
            expected.insert(key,length);
        }
        Ok(())
    };
    for name in ["code","prose","structured"] {
        let row = card.get("decode").get(name);
        if integer(row.get("completion_gate").get("passed"))? != 5 {
            return Err(format!("{name}: completion gate did not pass"));
        }
        add(&["decode",name,"runs"],row.get("runs").array()?,5,3)?;
    }
    if !draft {
        for mode in ["cold","warm_replay"] {
            add(&["prefill","8192",mode,"runs"],card.get("prefill").get("8192").get(mode).get("runs").array()?,3,4)?;
        }
        for size in ["1","2","4","8"] {
            let rounds = card.get("concurrency").get(size).get("rounds").array()?;
            if rounds.len() != 3 { return Err(format!("{size} chats: expected 3 rounds")); }
            for (i,round) in rounds.iter().enumerate() {
                add(&["concurrency",size,"rounds",&i.to_string(),"streams"],round.get("streams").array()?,size.parse().unwrap(),5)?;
            }
        }
    }
    Ok(expected)
}

fn streams(card: &Value, receipt: &Value, draft: bool) -> Result<Streams,String> {
    let expected = runs(card,draft)?;
    let mut ids = BTreeMap::new();
    for entry in receipt.array()? {
        let [path,tokens] = entry.array()? else {return Err("expected a stream path and its IDs".into());};
        let location = path.array()?.iter().map(|step| step.text().map(str::to_owned)).collect::<Result<Vec<_>,_>>()?;
        let name = location.join("/");
        let length = expected.get(&location).ok_or(format!("{name}: unexpected stream"))?;
        let tokens = tokens.array()?.iter().map(|token| {
            let value = integer(token)?;
            u32::try_from(value).map_err(|_| format!("{name}: ID outside u32"))
        }).collect::<Result<Vec<_>,_>>()?;
        if tokens.len() != *length { return Err(format!("{name}: captured {} IDs, expected {length}",tokens.len())); }
        if ids.insert(location,tokens).is_some() { return Err(format!("{name}: duplicate stream")); }
    }
    if let Some(missing) = expected.keys().find(|key| !ids.contains_key(*key)) {
        return Err(format!("{}: missing stream",missing.join("/")));
    }
    Ok(ids)
}

fn compare(left: &Streams, right: &Streams) -> Result<usize,String> {
    let mut count = 0;
    for (key,after) in right {
        let name = key.join("/");
        let before = left.get(key).ok_or(format!("{name}: no reference stream"))?;
        if let Some(at) = before.iter().zip(after).position(|(a,b)| a != b) {
            return Err(format!("{name}: token {} differs",at+1));
        }
        if before.len() != after.len() {
            return Err(format!("{name}: reference has {} IDs and candidate has {}",before.len(),after.len()));
        }
        count += after.len();
    }
    Ok(count)
}

fn settings(card: &Value, draft: bool, toggle: bool) -> Result<Value,String> {
    let Value::Object(mut fields) = card.get("settings").clone() else { return Err("missing card settings".into()); };
    let Some(Value::Object(body)) = fields.get_mut("extra_body") else { return Err("missing request settings".into()); };
    let (mode,depth) = if draft { ("strict",2) } else { ("plain",0) };
    if body.get("draft") != Some(&Value::String(mode.into())) || body.get("mtp_depth").map(integer).transpose()? != Some(depth)
        || body.get("cache") != Some(&Value::Bool(false)) {
        return Err("wrong card decoding mode or cache setting".into());
    }
    if toggle {
        body.remove("draft");
        body.remove("mtp_depth");
        fields.remove("concurrency");
        fields.remove("prefill_depths");
    }
    Ok(Value::Object(fields))
}

fn check(mode: &str, comparison: &str, base: &Value, base_receipt: &Value, next: &Value, next_receipt: &Value) -> Result<String,String> {
    let (base_draft,next_draft) = match mode {
        "plain" => (false,false),
        "draft" => (true,true),
        "toggle" => (false,true),
        _ => return Err("comparison mode must be plain, draft or toggle".into()),
    };
    for card in [base,next] {
        if card.get("run").get("comparison_id") != &Value::String(comparison.to_owned()) {
            return Err("wrong RigMark comparison ID".into());
        }
    }
    let reference = base.get("run").get("appliance").get("model_revision");
    let model = base.get("run").get("model");
    if reference.text()?.is_empty() || next.get("run").get("appliance").get("model_revision") != reference
        || model.text()?.is_empty() || next.get("run").get("model") != model {
        return Err("different model or checkpoint".into());
    }
    let toggle = mode == "toggle";
    if settings(base,base_draft,toggle)? != settings(next,next_draft,toggle)? {
        return Err("card request settings differ".into());
    }
    let base = streams(base,base_receipt,base_draft)?;
    let next = streams(next,next_receipt,next_draft)?;
    let count = compare(&base,&next)?;
    Ok(format!("CARD IDS PASS mode={mode} streams={} tokens={count}",next.len()))
}

fn expect_width(card: &Value, width: &str) -> Result<(),String> {
    let ranks = width.parse::<usize>().ok().filter(|n| *n > 0).ok_or("invalid expected rank count")?;
    let hardware = card.get("run").get("appliance").get("hardware").text()?;
    if !hardware.starts_with(&format!("{ranks} NVIDIA DGX Spark")) {
        return Err(format!("card hardware does not describe {ranks} Sparks"));
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let result = (|| {
        let [mode,comparison,base_width,next_width,base,base_ids,next,next_ids] = args.as_slice() else {
            return Err("usage: check-card-ids plain|draft|toggle COMPARISON BASE_WIDTH NEXT_WIDTH BASE.json BASE.ids.json NEXT.json NEXT.ids.json".into());
        };
        let (base,next) = (read(base)?,read(next)?);
        expect_width(&base,base_width)?;
        expect_width(&next,next_width)?;
        check(mode,comparison,&base,&read(base_ids)?,&next,&read(next_ids)?)
    })();
    match result {
        Ok(line) => { println!("{line}"); ExitCode::SUCCESS }
        Err(error) => { eprintln!("CARD IDS FAIL {error}"); ExitCode::FAILURE }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(count: usize) -> (Value,Value) {
        let run = r#"{"stream_done_marker":true,"completion_tokens":3,"finish_reason":"stop"}"#;
        let runs = [run;5].join(",");
        let row = format!(r#"{{"completion_gate":{{"passed":5}},"runs":[{runs}]}}"#);
        let card = format!(r#"{{"run":{{"comparison_id":"test","model":"m","appliance":{{"model_revision":"sha"}}}},"settings":{{"extra_body":{{"draft":"strict","mtp_depth":2,"cache":false}}}},"decode":{{"code":{row},"prose":{row},"structured":{row}}}}}"#);
        let mut ids = ["code","prose","structured"].into_iter().flat_map(|name|
            (0..5).map(move |i| format!(r#"[["decode","{name}","runs",{i}],[4,9]]"#))
        ).collect::<Vec<_>>();
        ids.truncate(count);
        (json::parse(&card).unwrap(),json::parse(&format!("[{}]",ids.join(","))).unwrap())
    }

    #[test]
    fn a_complete_receipt_compares_every_id_and_rejects_a_changed_word() {
        let (card,ids) = receipt(15);
        assert!(check("draft","test",&card,&ids,&card,&ids).unwrap().ends_with("streams=15 tokens=30"));
        let different = json::parse(&ids.encode().replacen("[4,9]","[4,10]",1)).unwrap();
        assert!(check("draft","test",&card,&ids,&card,&different).unwrap_err().contains("code/runs/0: token 2 differs"));
        let different = json::parse(&card.encode().replacen(r#""mtp_depth":2"#,r#""mtp_depth":3"#,1)).unwrap();
        assert!(check("draft","test",&card,&ids,&different,&ids).unwrap_err().contains("wrong card decoding mode"));
    }

    #[test]
    fn missing_duplicate_short_and_unfinished_streams_fail() {
        let (card,ids) = receipt(15);
        let (_,missing) = receipt(14);
        let duplicate = json::parse(&ids.encode().replacen(r#""code","runs",1"#,r#""code","runs",0"#,1)).unwrap();
        let short = json::parse(&ids.encode().replacen("[4,9]","[4]",1)).unwrap();
        let unfinished = json::parse(&card.encode().replacen(r#""stream_done_marker":true"#,r#""stream_done_marker":false"#,1)).unwrap();
        for (value,receipt,expected) in [
            (&card,&missing,"missing stream"),
            (&card,&duplicate,"duplicate stream"),
            (&card,&short,"captured 1 IDs"),
            (&unfinished,&ids,"stream incomplete"),
        ] {
            assert!(check("draft","test",&card,&ids,value,receipt).unwrap_err().contains(expected));
        }
    }
}
