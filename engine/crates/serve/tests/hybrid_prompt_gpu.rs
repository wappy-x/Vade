#![cfg(target_os = "linux")]
use spark_engine::{greedy_token, hybrid::{Model, Session}};
use spark_serve::{Engine, hybrid::Hybrid};

fn bits(row: &[f32]) -> Vec<u32> { row.iter().map(|v| v.to_bits()).collect() }
fn answer(session: &mut Session<'_, '_, '_>, first: &[f32]) -> Result<(Vec<u32>, Vec<Vec<u32>>), String> {
    let mut ids = vec![greedy_token(first)?];
    let mut rows = vec![bits(first)];
    for _ in 1..8 {
        ids.push(session.step_greedy(*ids.last().unwrap())?);
        rows.push(bits(&session.decode_logits(1)?.unwrap()[0]));
    }
    Ok((ids, rows))
}

#[test]
#[ignore = "skipped: no GPU; requires VADE_TYPED_35B_CHECKPOINT and a qualified GPU"]
fn typed_shared_prompt_sessions_match_solo_bytes() -> Result<(), String> {
    let root = std::env::var_os("VADE_TYPED_35B_CHECKPOINT")
        .ok_or("explicit GPU gate requires VADE_TYPED_35B_CHECKPOINT")?;
    let root = std::path::Path::new(&root);
    let tokenizer = engine_tokenizer::Tokenizer::from_directory(root).map_err(|e| e.to_string())?;
    let prompts: Vec<Vec<u32>> = [("Explain why addition is associative. ",1025), ("Count from one to five. ",513)]
        .into_iter().map(|(text,n)| tokenizer.encode(text).map(|ids| ids.into_iter().cycle().take(n).collect()))
        .collect::<Result<_,_>>().map_err(|e| e.to_string())?;
    spark_gpu_probe::driver::with_backend(0, spark_gpu_probe::layer_ptx()?, 100*1024*1024*1024, |b| {
        let model = Model::load_generic(b, root, 2048, 1)?;
        assert!(model.fixed_rows());
        let mut solo = Vec::new();
        for prompt in &prompts {
            let mut session = model.plain_session()?;
            let logits = session.prefill(prompt)?;
            solo.push(answer(&mut session, &logits)?);
        }
        let engine = Hybrid::new(&model, model.config().eos_token as u32).with_shared_prompt()?;
        engine.reserve_workspace()?;
        let (mut a, mut b) = (engine.session()?, engine.session()?);
        // A1, B1, A2, B2: A grows scores to stride 2048; B needs 1024 but inherits 2048.
        // The >512 lengths deliberately reach the score-buffer path, unlike tiny prompts.
        a.prefill_advance(&prompts[0][..769])?;
        b.prefill_advance(&prompts[1][..257])?;
        let la = a.prefill(&prompts[0][769..])?;
        let lb = b.prefill(&prompts[1][257..])?;
        let mut fork = a.fork()?;
        let mut ids = [vec![greedy_token(&la)?], vec![greedy_token(&lb)?], vec![greedy_token(&la)?]];
        let mut rows = [vec![bits(&la)], vec![bits(&lb)], vec![bits(&la)]];
        for _ in 1..8 {
            for (n, session) in [&mut a, &mut b, &mut fork].into_iter().enumerate() {
                ids[n].push(session.step_greedy(*ids[n].last().unwrap())?);
                rows[n].push(bits(&session.decode_logits(1)?.unwrap()[0]));
            }
        }
        for (n, expected) in [0,1,0].into_iter().enumerate() {
            assert_eq!(ids[n], solo[expected].0, "session {n}: greedy IDs");
            assert_eq!(rows[n], solo[expected].1, "session {n}: full logit bytes");
            assert_eq!(tokenizer.decode(&ids[n], false).map_err(|e| e.to_string())?.as_bytes(),
                tokenizer.decode(&solo[expected].0, false).map_err(|e| e.to_string())?.as_bytes(), "session {n}: answer bytes");
        }
        Ok(())
    })
}
