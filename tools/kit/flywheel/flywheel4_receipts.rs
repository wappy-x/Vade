use super::*;
#[path="flywheel4_counters.rs"] mod policy;
pub(super) fn declare(config:&Config)->Result<()> {policy::declare(config)}
fn require(condition:bool,message:impl Into<String>)->Result<()> {if condition {Ok(())} else {Err(message.into())}}
fn bytes(path:&Path)->Result<Vec<u8>> {fs::read(path).map_err(|error|format!("{}: {error}",path.display()))}
fn marker(path:&Path,wanted:&str)->Result<()> {require(text(path)?.trim()==wanted,format!("{}: completion marker differs",path.display()))}
fn fields(path:&Path)->Result<BTreeMap<String,String>> {
    let content=text(path)?;let mut result=BTreeMap::new();
    for word in content.split_whitespace() {let (key,value)=word.split_once('=').ok_or_else(||format!("{}: malformed field",path.display()))?;
        require(!key.is_empty() && !value.is_empty() && result.insert(key.into(),value.into()).is_none(),format!("{}: duplicate or empty field {key}",path.display()))?;}
    require(!result.is_empty(),format!("{}: no fields",path.display()))?;Ok(result)
}
fn integer(fields:&BTreeMap<String,String>,key:&str)->Result<u64> {
    let value=fields.get(key).ok_or_else(||format!("missing integer {key}"))?;
    require(!value.is_empty() && value.bytes().all(|byte|byte.is_ascii_digit()),format!("invalid integer {key}={value}"))?;
    value.parse().map_err(|_|format!("integer overflow {key}"))
}
fn positive(fields:&BTreeMap<String,String>,key:&str)->Result<f64> {
    let value=fields.get(key).ok_or_else(||format!("missing metric {key}"))?;
    let number=value.parse::<f64>().map_err(|_|format!("invalid metric {key}={value}"))?;
    require(number.is_finite() && number>0.0,format!("invalid positive metric {key}={value}"))?;Ok(number)
}
fn ids(path:&Path,expected:usize)->Result<Vec<u32>> {
    let content=text(path)?;let result=content.lines().map(|line| {if line.is_empty() || !line.bytes().all(|byte|byte.is_ascii_digit()) {return Err(format!("{}: invalid token ID",path.display()));}line.parse::<u32>().map_err(|_|"token ID overflow".into())}).collect::<Result<Vec<_>>>()?;
    require(result.len()==expected,format!("{}: expected {expected} IDs, found {}",path.display(),result.len()))?;Ok(result)
}
fn inventory(path:&Path,job:&Job)->Result<()> {
    marker(&path.join("COMPLETE"),"RESIDENT DECODE RUN PASS")?;ids(&path.join("INPUT_IDS"),job.input)?;ids(&path.join("OUTPUT_IDS"),job.output)?;
    let mut actual=std::collections::BTreeSet::new();
    for entry in fs::read_dir(path).map_err(|error|error.to_string())? {let entry=entry.map_err(|error|error.to_string())?;let name=entry.file_name().to_string_lossy().into_owned();
        if name.ends_with("-logits.f32") {actual.insert(name);}}
    let wanted=(0..job.output).map(|token|format!("{token}-logits.f32")).collect::<std::collections::BTreeSet<_>>();
    require(actual==wanted,format!("{}: logit inventory differs",path.display()))?;
    require(!bytes(&path.join("PROMPT_LOGITS"))?.is_empty(),"empty prompt logits")?;bytes(&path.join("TEXT"))?;
    for token in 0..job.output {let name=path.join(format!("{token}-logits.f32"));let size=fs::metadata(&name).map_err(|error|error.to_string())?.len();require(size>0 && size%4==0,format!("{}: invalid logits size",name.display()))?;}
    let timing=fields(&path.join("TIMING"))?;let counts=fields(&path.join("COUNTS"))?;let prompt=fields(&path.join("PREFILL_TIMING"))?;let projection=fields(&path.join("PROJECTION_DISPATCH"))?;
    for record in [&timing,&counts,&projection] {require(integer(record,"decode_steps")?==job.output as u64-1,"decode step count differs")?;}
    for key in ["graph_replays","graph_captures","stream_waits","host_launches"] {integer(&counts,key)?;}
    for (key,value) in &counts {require(value.bytes().all(|byte|byte.is_ascii_digit()),format!("noninteger counter {key}"))?;integer(&counts,key)?;}
    for key in ["project_f32_ordered","project_bf16_ordered"] {integer(&projection,key)?;}
    require(integer(&prompt,"prompt_tokens")?==job.input as u64,"prompt token count differs")?;
    positive(&prompt,"prompt_seconds")?;let seconds=positive(&timing,"decode_seconds")?;let rate=positive(&timing,"decode_tokens_per_second")?;
    require((rate*seconds-(job.output-1) as f64).abs()<0.0001,"decode rate and elapsed seconds disagree")?;Ok(())
}
fn equal(reference:&Path,current:&Path,job:&Job)->Result<()> {
    for name in ["INPUT_IDS","OUTPUT_IDS","PROMPT_LOGITS","TEXT"].into_iter().map(str::to_owned).chain((0..job.output).map(|token|format!("{token}-logits.f32"))) {
        require(bytes(&reference.join(&name))?==bytes(&current.join(&name))?,format!("{} differs from {}",current.join(&name).display(),reference.join(&name).display()))?;
    }Ok(())
}
pub(super) fn reference(config:&Config)->Result<()> {
    for box_name in &config.boxes {
        let root=config.root.join("reference").join(box_name);marker(&root.join("BUILD-REVISION"),REFERENCE_BASE)?;marker(&root.join("glm.exit"),"0")?;
        let log=text(root.join("glm-run.log"))?;require(log.lines().filter(|line|line.starts_with("RESIDENT READY ")).count()==1,"standard head did not load exactly once")?;
        for job in config.jobs() {let directory=root.join(job.name);marker(&directory.join("COMPLETE"),"RESIDENT DECODE SUITE PASS")?;
            for repeat in 1..=3 {let run=directory.join(format!("run{repeat}"));inventory(&run,job)?;
                equal(&config.root.join("golden").join(job.name).join("run1"),&run,job)?;}
        }
    }Ok(())
}
fn median(mut values:Vec<f64>)->Result<f64> {
    require(!values.is_empty() && values.iter().all(|value|value.is_finite() && *value>0.0),"invalid median inputs")?;
    values.sort_by(f64::total_cmp);let middle=values.len()/2;Ok(if values.len()%2==0 {(values[middle-1]+values[middle])/2.0} else {values[middle]})
}
fn launch_counters(root:&Path,config:&Config,variants:&[Variant],log:&str)->Result<()> {
    let mut pairs=log.lines().filter(|line|line.starts_with("PROMPT KERNEL COUNTS ") || line.starts_with("DECODE KERNEL COUNTS "));
    let value=|line:&str|->Result<u64> {let found=line.split_whitespace().filter_map(|word|word.strip_prefix("tp_exchange_fused=")).collect::<Vec<_>>();
        require(found.len()==1,"fused launch counter missing or duplicated")?;found[0].parse().map_err(|_|"invalid fused launch count".into())};
    let mut report=String::from("setting\tjob\trun\tfused_launches\trecorded_exchange\n");
    for variant in variants {for (job,repeats) in std::iter::once(("warmup",1)).chain(config.jobs().iter().map(|job|(job.name,3))) {for repeat in 1..=repeats {
        let prompt=pairs.next().ok_or("prompt launch line missing")?;let decode=pairs.next().ok_or("decode launch line missing")?;
        require(prompt.starts_with("PROMPT ") && decode.starts_with("DECODE "),"launch line order differs")?;
        let delta=value(decode)?.checked_sub(value(prompt)?).ok_or("cumulative launch counter decreased")?;
        let kinds=fields(&root.join(&variant.name).join(job).join(format!("run{repeat}/RECORDED_KINDS")))?;
        let recorded=integer(&kinds,"resident_launch_exchange")?;
        require(delta==recorded && (delta>0)==(variant.cut.id=="exchange"),"fused launch delta differs from active switch or recorded kinds")?;
        report.push_str(&format!("{}\t{job}\t{repeat}\t{delta}\t{recorded}\n",variant.name));
    }}}require(pairs.next().is_none(),"extra launch-count lines")?;save(root.join("EXCHANGE-LAUNCH-COUNTS.tsv"),report)
}
fn defer_counters(root:&Path,config:&Config,variants:&[Variant])->Result<()> {
    let mut report=String::from("setting\tjob\trun\tadmitted\tfold_accepted\tmiddle_accepted\tfold_fallback\tmiddle_fallback\tfold_removed_local_words\tmiddle_removed_local_words\tadmission_seconds\n");let mut valid=true;
    for variant in variants {for (job,repeats) in std::iter::once(("warmup",1)).chain(config.jobs().iter().map(|job|(job.name,3))) {
        let directory=root.join(&variant.name).join(job);let admission=fields(&directory.join("DEFER_GATHER_ADMISSION"))?;let enabled=variant.cut.id=="defer";
        require(admission.len()==4 && admission.get("enabled").map(String::as_str)==Some(if enabled {"true"} else {"false"}) && admission.get("scope").map(String::as_str)==Some("complete_scale_metadata_before_session"),"defer admission schema or switch differs")?;
        let admitted=integer(&admission,"admitted")?;let seconds=positive(&admission,"admission_seconds")?;valid &= admitted==if enabled {42} else {0};
        for repeat in 1..=repeats {let values=fields(&directory.join(format!("run{repeat}/DEFER_GATHER")))?;require(values.len()==8 && values.get("scope").map(String::as_str)==Some("commands_completed_by_final_sync"),"defer work schema differs")?;
            let steps=integer(&values,"decode_steps")?;require(steps==63,"defer decode step count differs")?;
            let names=["fold_accepted","middle_accepted","fold_fallback","middle_fallback","fold_removed_local_words","middle_removed_local_words"];
            let work=names.iter().map(|key|integer(&values,key)).collect::<Result<Vec<_>>>()?;valid &= work.iter().all(|value|value%steps==0);
            valid &= if enabled {work[0]>0 && work[1]>0 && work[4]>0 && work[5]>0} else {work.iter().all(|value|*value==0)};
            report.push_str(&format!("{}\t{job}\t{repeat}\t{admitted}\t{}\t{}\t{}\t{}\t{}\t{}\t{seconds:.9}\n",variant.name,work[0],work[1],work[2],work[3],work[4],work[5]));
        }
    }}save(root.join("DEFER-ADMISSION-COUNTS.tsv"),report)?;
    require(valid,"DEFER-GATHER speed VOID: require admitted=42 and nonzero whole-step fold/middle work on both ranks; head work must be zero")
}
#[derive(Clone)] struct Outcome {rate:Option<f64>,exact:bool,counters:bool,live:bool}
fn screen(run:&Path,variant:&Variant,pid:u64)->Result<()> {
    let values=fields(&run.join("SCREEN"))?;
    require(values.get("name").map(String::as_str)==Some(variant.name.as_str()),"screen label differs")?;
    require(integer(&values,"pid")?==pid,"screen process changed")?;
    require(values.get("resident_weights_ready").map(String::as_str)==Some("true"),"weights not resident")?;
    for key in ["mhc","pdl","queue"] {require(values.get(key).map(String::as_str)==Some(if variant.cuts.split(',').any(|cut|cut==key) {"true"} else {"false"}),format!("wrong {key} switch"))?;}
    require((integer(&values,"pdl_windows")?>0)==variant.cuts.split(',').any(|cut|cut=="pdl"),"PDL did not activate as declared")?;
    require(values.get("conv").map(String::as_str)==Some("true") && integer(&values,"parameter_static_hits")?>0,"head memo path is not active")?;
    require(values.get("exchange_fuse").map(String::as_str)==Some(if variant.cut.id=="exchange" {"true"} else {"false"}),"exchange switch differs")?;
    let identity=text(run.parent().ok_or("run parent missing")?.join("IDENTITY"))?;
    require(identity.lines().any(|line|line==format!("exchange_fuse={}",variant.cut.id=="exchange")),"exchange identity differs")?;
    require(identity.lines().any(|line|line==format!("defer_gather={}",variant.cut.id=="defer")),"defer identity differs")?;
    if variant.cut.id=="pdl-only" {let kinds=fields(&run.join("RECORDED_KINDS"))?;
        for key in ["resident_launch_mhc_rms","resident_launch_mhc_dots","resident_launch_mhc_finish"] {require(integer(&kinds,key)?==0,"PDL-only recorded MHC")?;}}
    marker(&run.join("EXACTNESS"),"PASS")
}
fn gpu(directory:&Path)->Result<()> {
    let before=fields(&directory.join("GPU_BEFORE"))?;let after=fields(&directory.join("GPU_AFTER"))?;
    require(integer(&after,"epoch_seconds")?>=integer(&before,"epoch_seconds")?,"job clock moved backward")?;
    for (name,state) in [("GPU_BEFORE",&before),("GPU_AFTER",&after)] {require(integer(state,"temperature_c")?<=120,"invalid GPU temperature")?;
        let raw=text(directory.join(format!("{name}.raw")))?;let raw=raw.trim().split(',').map(str::trim).collect::<Vec<_>>();
        require(raw.len()==3 && raw[0]==state["temperature_c"] && raw[1]==state["sm_mhz"],"raw GPU telemetry differs")?;
        require(integer(state,"sm_mhz")?>0,"invalid GPU SM clock")?;
        if state.get("memory_mhz").map(String::as_str)==Some("unavailable") {require(["[N/A]","N/A"].contains(&raw[2]),"missing GPU clock lacks an explicit unavailable marker")?;}
        else {require(integer(state,"memory_mhz")?>0 && raw[2]==state["memory_mhz"],"invalid GPU memory clock")?;}
    }Ok(())
}
fn paired(before:f64,current:f64,after:f64)->Result<(f64,f64,f64)> {
    require([before,current,after].iter().all(|value|value.is_finite() && *value>0.0),"invalid paired rates")?;
    let reference=(before+after)/2.0;Ok((reference,100.0*(current/reference-1.0),100.0*(after-before).abs()/reference))
}
pub(super) fn check(config:&Config)->Result<String> {
    let variants=variants();let rules=policy::load(&config.root.join("COUNTER-DECLARATIONS.tsv"))?;
    let mut global=Vec::new();if let Err(error)=reference(config) {global.push(error);}
    let mut pids=BTreeMap::new();let mut differences=String::new();let mut launch_errors=Vec::new();
    for box_name in &config.boxes {let root=config.root.join(box_name);let counted=text(root.join("run.log")).and_then(|log|launch_counters(&root,config,&variants,&log));
        if let Err(error)=counted {launch_errors.push(format!("COUNTERS FAIL {box_name}: {error}\n"));}
        if let Err(error)=defer_counters(&root,config,&variants) {launch_errors.push(format!("COUNTERS FAIL {box_name}: {error}\n"));}}
    for error in &launch_errors {differences.push_str(error);}
    for (rank,box_name) in config.boxes.iter().enumerate() {let root=config.root.join(box_name);let checked:Result<()>=(|| {
        marker(&root.join("BUILD-REVISION"),&config.revision)?;marker(&root.join("run.exit"),"0")?;
        let log=text(root.join("run.log"))?;require(log.lines().filter(|line|line.starts_with("RESIDENT READY ")).count()==1,"model was not loaded exactly once")?;
        require(log.lines().filter(|line|line.starts_with("FLYWHEEL VARIANT ")).count()==variants.len(),"variant inventory differs")?;
        let tokens=log.lines().filter(|line|line.starts_with("RESIDENT TOKEN ")).collect::<Vec<_>>();
        require(tokens.len()==variants.len()*(64+config.jobs().iter().map(|job|job.output*3).sum::<usize>()) && tokens.iter().all(|line|line.contains(&format!("rank={rank} "))),"token log inventory differs")?;
        let pid=text(root.join("engine.pid"))?.trim().parse::<u64>().map_err(|_|"invalid engine PID")?;require(pid>0,"zero engine PID")?;pids.insert(box_name.clone(),pid);Ok(())
    })();if let Err(error)=checked {global.push(format!("{box_name}: {error}"));}}
    let mut outcomes=BTreeMap::new();let mut metrics=String::from("setting\tcut\tpass\tbox\tinput\trun\tdecode_tok_s\tprompt_seconds\tcaptures\n");
    let mut peaks=String::from("setting\tbox\tinput\trun\tpeak_after_prompt\tpeak_after_decode\n");
    let mut live_report=String::from("head\tbox\tinput\trun\tfield\tinitial\tcurrent\tequal\n");
    for variant in &variants {
        let mut warm_errors=global.clone();
        for box_name in &config.boxes {let directory=config.root.join(box_name).join(&variant.name).join("warmup");let run=directory.join("run1");
            let checked:Result<()>=(|| {marker(&directory.join("COMPLETE"),"RESIDENT DECODE SUITE PASS")?;gpu(&directory)?;inventory(&run,&JOBS[0])?;
                screen(&run,variant,*pids.get(box_name).ok_or("missing rank PID")?)?;
                equal(&config.root.join("golden").join(JOBS[0].name).join("run1"),&run,&JOBS[0])?;Ok(())})();
            if let Err(error)=checked {warm_errors.push(format!("{} {box_name} warmup: {error}",variant.name));}
        }
        for job in config.jobs() {let mut exact=warm_errors.is_empty();let mut counters_ok=launch_errors.is_empty();let mut live_ok=true;let mut rates=Vec::new();
            for error in &warm_errors {differences.push_str(&format!("EXACTNESS FAIL {error}\n"));}
            for box_name in &config.boxes {let directory=config.root.join(box_name).join(&variant.name).join(job.name);
                if let Err(error)=marker(&directory.join("COMPLETE"),"RESIDENT DECODE SUITE PASS").and_then(|_|gpu(&directory)) {exact=false;differences.push_str(&format!("EXACTNESS FAIL {} {box_name} {}: {error}\n",variant.name,job.name));}
                for repeat in 1..=3 {let run=directory.join(format!("run{repeat}"));
                    let checked:Result<()>=(|| {inventory(&run,job)?;screen(&run,variant,*pids.get(box_name).ok_or("missing PID")?)?;
                        equal(&config.root.join(&config.boxes[0]).join("head-00").join(job.name).join("run1"),&run,job)?;
                        equal(&config.root.join("golden").join(job.name).join("run1"),&run,job)?;
                        equal(&config.root.join("reference").join(box_name).join(job.name).join(format!("run{repeat}")),&run,job)?;Ok(())})();
                    if let Err(error)=checked {exact=false;differences.push_str(&format!("EXACTNESS FAIL {} {box_name} {} run{repeat}: {error}\n",variant.name,job.name));}
                    match policy::counters(&rules,variant,job,repeat,&run,&config.root.join(box_name).join("head-00").join(job.name).join(format!("run{repeat}"))) {
                        Ok(errors)=>{if !errors.is_empty() {counters_ok=false;}for error in errors {differences.push_str(&format!("COUNTERS FAIL {} {box_name} {} run{repeat}: {error}\n",variant.name,job.name));}},
                        Err(error)=>{counters_ok=false;differences.push_str(&format!("COUNTERS FAIL {} {box_name} {} run{repeat}: {error}\n",variant.name,job.name));},
                    }
                    let measured:Result<()>=(|| {let timing=fields(&run.join("TIMING"))?;let prompt=fields(&run.join("PREFILL_TIMING"))?;let counts=fields(&run.join("COUNTS"))?;
                        let rate=positive(&timing,"decode_tokens_per_second")?;let prompt=positive(&prompt,"prompt_seconds")?;
                        if box_name==&config.boxes[0] {rates.push(rate);}
                        metrics.push_str(&format!("{}\t{}\t{}\t{box_name}\t{}\t{repeat}\t{rate:.9}\t{prompt:.9}\t{}\n",variant.name,variant.cut.id,variant.pass,job.input,integer(&counts,"graph_captures")?));
                        peaks.push_str(&format!("{}\t{box_name}\t{}\t{repeat}\t{}\t{}\n",variant.name,job.input,integer(&counts,"peak_bytes_after_prompt")?,integer(&counts,"peak_bytes_after_decode")?));
                        if variant.pass==0 {let base=fields(&config.root.join(box_name).join("head-00").join(job.name).join(format!("run{repeat}/COUNTS")))?;
                            for key in ["live_bytes_after_prompt","live_bytes_after_decode"] {let first=integer(&base,key)?;let current=integer(&counts,key)?;live_ok &= first==current;
                                live_report.push_str(&format!("{}\t{box_name}\t{}\t{repeat}\t{key}\t{first}\t{current}\t{}\n",variant.name,job.input,first==current));}}
                        Ok(())})();
                    if let Err(error)=measured {exact=false;counters_ok=false;differences.push_str(&format!("METRICS FAIL {} {box_name} {} run{repeat}: {error}\n",variant.name,job.name));}
                }
            }
            outcomes.insert((variant.name.clone(),job.input),Outcome {rate:if rates.len()==3 {Some(median(rates)?)} else {None},exact,counters:counters_ok,live:live_ok});
        }
    }
    let mut floors=BTreeMap::<(usize,usize),f64>::new();let mut pairs=BTreeMap::new();
    for (position,variant) in variants.iter().enumerate().filter(|(_,variant)|variant.pass>0) {for job in config.jobs() {
        let before=&outcomes[&(variants[position-1].name.clone(),job.input)];let current=&outcomes[&(variant.name.clone(),job.input)];let after=&outcomes[&(variants[position+1].name.clone(),job.input)];
        if let (Some(before_rate),Some(rate),Some(after_rate))=(before.rate,current.rate,after.rate) {let (reference,delta,spread)=paired(before_rate,rate,after_rate)?;
            let floor=floors.entry((variant.pass,job.input)).or_default();*floor=floor.max(spread);
            if variant.cut.id=="conv" {*floor=floor.max(delta.abs());}
            pairs.insert((variant.pass,variant.cut.id,job.input),(before_rate,rate,after_rate,reference,delta,spread,before.exact && after.exact && current.exact,before.counters && after.counters && current.counters,before.live && after.live));
        }
    }}
    let mut paired_table=String::from("cut\tpass\tinput\thead_before\tvariant\thead_after\treference_mean\tdelta_pct\tlocal_spread_pct\tpass_floor_pct\texactness\tcounters\thead_live_equal\n");
    let mut table=String::from("cut\texactness\tdeclared_counters\thead_live_equal\tp1_15_pct\tp1_127_pct\tp2_15_pct\tp2_127_pct\tp1_15_floor\tp1_127_floor\tp2_15_floor\tp2_127_floor\tscreen_candidate\n");
    for cut in &CUTS[1..] {let mut exact=true;let mut counters_ok=true;let mut live=true;let mut beats=true;let mut deltas=Vec::new();let mut bands=Vec::new();
        for pass in 1..=2 {for job in config.jobs() {match pairs.get(&(pass,cut.id,job.input)) {
            Some(&(before,rate,after,reference,delta,spread,bytes,counters,heads))=>{let floor=floors[&(pass,job.input)];exact &= bytes;counters_ok &= counters;live &= heads;beats &= delta>floor;
                deltas.push(if cut.id=="defer" && !launch_errors.is_empty() {"VOID".into()} else {format!("{delta:.4}")});bands.push(format!("{floor:.4}"));
                paired_table.push_str(&format!("{}\t{pass}\t{}\t{before:.9}\t{rate:.9}\t{after:.9}\t{reference:.9}\t{delta:.4}\t{spread:.4}\t{floor:.4}\t{}\t{}\t{}\n",cut.id,job.input,verdict(bytes),verdict(counters),verdict(heads)));},
            None=>{exact=false;counters_ok=false;live=false;beats=false;deltas.push("NA".into());bands.push("NA".into());},
        }}}
        table.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\n",cut.id,verdict(exact),verdict(counters_ok),verdict(live),deltas.join("\t"),bands.join("\t"),if cut.id!="conv" && exact && counters_ok && live && beats {"REVIEW-REQUIRED"} else {"NO"}));
    }
    let mut standard=format!("# standard_head={REFERENCE_BASE}; source={}\ninput\tstandard_head_tok_s\toff_state_tok_s\tdelta_pct\n",text(config.root.join("REFERENCE-SOURCE.txt"))?.trim());
    for job in config.jobs() {let root=config.root.join("reference").join(&config.boxes[0]).join(job.name);let rates=(1..=3).map(|repeat|fields(&root.join(format!("run{repeat}/TIMING"))).and_then(|values|positive(&values,"decode_tokens_per_second"))).collect::<Result<Vec<_>>>()?;
        let baseline=median(rates)?;if let Some(rate)=outcomes[&("head-00".to_owned(),job.input)].rate {standard.push_str(&format!("{}\t{baseline:.9}\t{rate:.9}\t{:.4}\n",job.input,100.0*(rate/baseline-1.0)));}}
    table.insert_str(0,&format!("# standard_head={REFERENCE_BASE}; source={}; rate_rank={}; selection=adjacent-in-process-heads-only; conv=head-reset-control\n",text(config.root.join("REFERENCE-SOURCE.txt"))?.trim(),config.boxes[0]));
    save(config.root.join("TABLE.tsv"),&table)?;save(config.root.join("PAIRED.tsv"),paired_table)?;save(config.root.join("METRICS.tsv"),metrics)?;save(config.root.join("PEAKS.tsv"),peaks)?;
    save(config.root.join("HEAD-LIVE.tsv"),live_report)?;save(config.root.join("STANDARD-HEAD.tsv"),standard)?;save(config.root.join("DIFFERENCES.txt"),differences)?;
    save(config.root.join("SCREEN-STATUS"),format!("SCREEN ONLY; merge gate not run; independent review required\nsettings={} cuts={} measured_runs_per_rank={}\n",variants.len(),CUTS.len()-1,variants.len()*config.jobs().len()*3))?;Ok(table)
}
fn verdict(pass:bool)->&'static str {if pass {"PASS"} else {"FAIL"}}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn unavailable_memory_clock_requires_matching_raw_evidence() {
        let root=std::env::temp_dir().join(format!("flywheel4-gpu-{}",std::process::id()));fs::create_dir(&root).unwrap();
        for name in ["GPU_BEFORE","GPU_AFTER"] {
            save(root.join(name),"epoch_seconds=1 temperature_c=43 sm_mhz=208 memory_mhz=unavailable\n").unwrap();
            save(root.join(format!("{name}.raw")),"43, 208, [N/A]\n").unwrap();
        }
        assert!(gpu(&root).is_ok());save(root.join("GPU_AFTER.raw"),"43, 208, broken\n").unwrap();assert!(gpu(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test] fn adjacent_mean_and_spread_do_not_use_the_first_head() {
        assert_eq!(paired(10.0,12.0,14.0).unwrap(),(12.0,0.0,100.0*4.0/12.0));
        assert!(paired(0.0,1.0,1.0).is_err());assert!(paired(1.0,f64::NAN,1.0).is_err());
    }
    #[test] fn the_orders_are_shuffled_complete_and_bracketed() {
        let variants=variants();assert_eq!(variants.len(),9);let mut orders=Vec::new();
        for pass in 1..=2 {let order=variants.iter().filter(|variant|variant.pass==pass).map(|variant|variant.cut.id).collect::<Vec<_>>();assert_eq!(order.iter().collect::<std::collections::BTreeSet<_>>().len(),2);orders.push(order);}
        assert_ne!(orders[0],orders[1]);
        for (position,variant) in variants.iter().enumerate().filter(|(_,variant)|variant.pass>0) {assert_eq!(variants[position-1].pass,0);assert_eq!(variants[position+1].pass,0);assert!(!variant.cuts.contains("queue"));}
        assert_eq!(variants.iter().map(|variant|&variant.name).collect::<std::collections::BTreeSet<_>>().len(),9);
    }
    fn fixture_run(root:&Path,job:&Job,variant:&Variant,pid:u64,rate:f64) {
        fs::create_dir_all(root).unwrap();save(root.join("COMPLETE"),"RESIDENT DECODE RUN PASS\n").unwrap();
        save(root.join("INPUT_IDS"),"0\n".repeat(job.input)).unwrap();save(root.join("OUTPUT_IDS"),"0\n".repeat(job.output)).unwrap();
        save(root.join("PROMPT_LOGITS"),"fixture\n").unwrap();save(root.join("TEXT"),"fixture\n").unwrap();
        for token in 0..job.output {save(root.join(format!("{token}-logits.f32")),0.0f32.to_le_bytes()).unwrap();}
        save(root.join("TIMING"),format!("decode_steps=63 decode_seconds={:.12} decode_tokens_per_second={rate}\n",63.0/rate)).unwrap();
        save(root.join("PREFILL_TIMING"),format!("prompt_tokens={} prompt_seconds=1\n",job.input)).unwrap();
        save(root.join("COUNTS"),"decode_steps=63 graph_replays=1 graph_captures=1 stream_waits=1 host_launches=0 live_bytes_after_prompt=100 live_bytes_after_decode=100 peak_bytes_after_prompt=100 peak_bytes_after_decode=100\n").unwrap();
        save(root.join("PROJECTION_DISPATCH"),"decode_steps=63 project_f32_ordered=1 project_bf16_ordered=1 causal_fast=2\n").unwrap();
        let enabled=|cut|variant.cuts.split(',').any(|value|value==cut);let mhc=usize::from(enabled("mhc"));
        let (put,wide,scatter,exchange)=if variant.cut.id=="exchange" {(3,0,0,7)} else {(4,6,7,0)};
        save(root.join("RECORDED_KINDS"),format!("decode_steps=63 resident_launch_mhc_rms={mhc} resident_launch_mhc_dots={mhc} resident_launch_mhc_finish={mhc} resident_launch_put={put} resident_launch_put_wide={wide} resident_launch_take_scatter={scatter} resident_launch_exchange={exchange} resident_launch_causal=2 resident_launch_causal_fast=2\n")).unwrap();
        let deferred=variant.cut.id=="defer";let count=if deferred {63} else {0};
        save(root.join("DEFER_GATHER"),format!("decode_steps=63 fold_accepted={count} fold_fallback=0 fold_removed_local_words={count} middle_accepted={count} middle_fallback=0 middle_removed_local_words={count} scope=commands_completed_by_final_sync\n")).unwrap();
        save(root.parent().unwrap().join("DEFER_GATHER_ADMISSION"),format!("enabled={deferred} admitted={} admission_seconds=0.01 scope=complete_scale_metadata_before_session\n",if deferred {42} else {0})).unwrap();
        save(root.join("EXCHANGE_STAGE"),format!("decode_steps=63 accepted=1 original_messages=1 local_words={}\n",if deferred {74} else {200})).unwrap();
        save(root.parent().unwrap().join("IDENTITY"),format!("exchange_fuse={}\ndefer_gather={deferred}\n",variant.cut.id=="exchange")).unwrap();
        save(root.join("SCREEN"),format!("name={} pid={pid} resident_weights_ready=true conv=true exchange_fuse={} mhc={} pdl={} queue=false pdl_windows={} parameter_static_hits=1\n",variant.name,variant.cut.id=="exchange",enabled("mhc"),enabled("pdl"),usize::from(enabled("pdl")))).unwrap();save(root.join("EXACTNESS"),"PASS\n").unwrap();
    }
    fn fixture()->Config {
        static NEXT:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let sequence=NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
        let root=std::env::temp_dir().join(format!("flywheel4-full-{}-{nonce}-{sequence}",std::process::id()));fs::create_dir(&root).unwrap();
        let config=Config {root,tag:"test".into(),revision:"a".repeat(40),boxes:["box-a".into(),"box-b".into()],network:"unused".into(),long:false};let variants=variants();save(config.root.join("REFERENCE-SOURCE.txt"),"synthetic-fixture\n").unwrap();
        for job in config.jobs() {fixture_run(&config.root.join("golden").join(job.name).join("run1"),job,&variants[0],1,1.0);}
        for (rank,box_name) in config.boxes.iter().enumerate() {let reference=config.root.join("reference").join(box_name);fs::create_dir_all(&reference).unwrap();save(reference.join("BUILD-REVISION"),REFERENCE_BASE).unwrap();save(reference.join("glm.exit"),"0\n").unwrap();save(reference.join("glm-run.log"),"RESIDENT READY fixture\n").unwrap();
            for job in config.jobs() {let directory=reference.join(job.name);for repeat in 1..=3 {fixture_run(&directory.join(format!("run{repeat}")),job,&variants[0],rank as u64+1,1.0);}save(directory.join("COMPLETE"),"RESIDENT DECODE SUITE PASS\n").unwrap();}
            let root=config.root.join(box_name);fs::create_dir(&root).unwrap();save(root.join("BUILD-REVISION"),&config.revision).unwrap();save(root.join("run.exit"),"0\n").unwrap();save(root.join("engine.pid"),format!("{}\n",rank+1)).unwrap();
            let mut launches=String::new();let mut fused=0;for variant in &variants {for _ in 0..7 {launches.push_str(&format!("PROMPT KERNEL COUNTS tp_exchange_fused={fused}\n"));if variant.cut.id=="exchange" {fused+=7;}launches.push_str(&format!("DECODE KERNEL COUNTS tp_exchange_fused={fused}\n"));}}
            save(root.join("run.log"),format!("RESIDENT READY fixture\n{}{}{launches}","FLYWHEEL VARIANT fixture\n".repeat(variants.len()),format!("RESIDENT TOKEN rank={rank} fixture\n").repeat(variants.len()*448))).unwrap();
            for variant in &variants {let rate=if variant.pass==0 || variant.cut.id=="conv" {1.0} else {1.1};
                for job in std::iter::once(Job {name:"warmup",..JOBS[0]}).chain(config.jobs().iter().copied()) {let directory=root.join(&variant.name).join(job.name);let repeats=if job.name=="warmup" {1} else {3};
                    for repeat in 1..=repeats {fixture_run(&directory.join(format!("run{repeat}")),&job,variant,rank as u64+1,rate);}
                    save(directory.join("COMPLETE"),"RESIDENT DECODE SUITE PASS\n").unwrap();for name in ["GPU_BEFORE","GPU_AFTER"] {save(directory.join(name),"epoch_seconds=1 temperature_c=50 sm_mhz=1000 memory_mhz=1000\n").unwrap();save(directory.join(format!("{name}.raw")),"50, 1000, 1000\n").unwrap();}
                }
            }
        }
        policy::declare(&config).unwrap();config
    }
    #[test] fn complete_fixture_rejects_corruption_missing_warmup_and_undeclared_counters() {
        let config=fixture();let table=check(&config).unwrap();assert_eq!(table.lines().filter(|line|line.ends_with("REVIEW-REQUIRED")).count(),1);
        let root=config.root.join("box-b/p1-conv/glm-tokens64/run2");save(root.join("0-logits.f32"),1.0f32.to_le_bytes()).unwrap();
        let table=check(&config).unwrap();assert!(table.lines().any(|line|line.starts_with("conv\tFAIL\t")));
        save(root.join("0-logits.f32"),0.0f32.to_le_bytes()).unwrap();fs::remove_file(config.root.join("box-a/p2-defer/warmup/run1/EXACTNESS")).unwrap();
        let table=check(&config).unwrap();assert!(table.lines().any(|line|line.starts_with("defer\tFAIL\t")));
        let counts=root.join("COUNTS");save(&counts,format!("{}unannounced=1\n",text(&counts).unwrap())).unwrap();
        let table=check(&config).unwrap();assert!(table.lines().any(|line|line.starts_with("conv\tPASS\tFAIL\t")));
        fs::remove_dir_all(config.root).unwrap();
    }
    #[test] fn state_and_exchange_counts_fail_closed() {
        let config=fixture();let counts=config.root.join("box-a/head-01/glm-tokens64/run2/COUNTS");
        save(&counts,text(&counts).unwrap().replace("live_bytes_after_decode=100","live_bytes_after_decode=101")).unwrap();
        let table=check(&config).unwrap();assert!(!table.lines().any(|line|line.ends_with("REVIEW-REQUIRED")));
        save(&counts,text(&counts).unwrap().replace("live_bytes_after_decode=101","live_bytes_after_decode=100")).unwrap();
        let kinds=config.root.join("box-b/p1-defer/glm-tokens64/run2/RECORDED_KINDS");
        save(&kinds,text(&kinds).unwrap().replace("resident_launch_put=4","resident_launch_put=5")).unwrap();
        let table=check(&config).unwrap();assert!(table.lines().any(|line|line.starts_with("defer\tPASS\tFAIL\t")));
        fs::remove_dir_all(config.root).unwrap();
    }
    #[test] fn incomplete_admission_voids_speed_without_hiding_exactness() {
        let config=fixture();let path=config.root.join("box-b/p2-defer/out127/DEFER_GATHER_ADMISSION");
        save(&path,text(&path).unwrap().replace("admitted=42","admitted=41")).unwrap();
        let table=check(&config).unwrap();let row=table.lines().find(|line|line.starts_with("defer\t")).unwrap();assert!(row.starts_with("defer\tPASS\tFAIL\t"));assert_eq!(row.matches("VOID").count(),4);assert!(row.ends_with("NO"));
        fs::remove_dir_all(config.root).unwrap();
    }
    #[test] fn checked_causal_mix_requires_conservation_and_matching_dispatch() {
        let config=fixture();let run=config.root.join("box-b/p1-defer/glm-tokens64/run2");let kinds=run.join("RECORDED_KINDS");let dispatch=run.join("PROJECTION_DISPATCH");
        save(&kinds,text(&kinds).unwrap().replace("resident_launch_causal=2","resident_launch_causal=3").replace("resident_launch_causal_fast=2","resident_launch_causal_fast=1")).unwrap();
        save(&dispatch,text(&dispatch).unwrap().replace("causal_fast=2","causal_fast=1")).unwrap();
        let table=check(&config).unwrap();assert!(table.lines().any(|line|line.starts_with("defer\tPASS\tPASS\t") && line.ends_with("REVIEW-REQUIRED")));
        save(&dispatch,text(&dispatch).unwrap().replace("causal_fast=1","causal_fast=2")).unwrap();
        let table=check(&config).unwrap();assert!(table.lines().any(|line|line.starts_with("defer\tPASS\tFAIL\t")));
        fs::remove_dir_all(config.root).unwrap();
    }
}
