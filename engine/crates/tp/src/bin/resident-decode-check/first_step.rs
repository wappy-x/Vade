//! One box, small checkpoint, the real served stack (Factory, Split, Device, resident queue): what the first decode
//! steps of a session cost beside a steady step, in host ms, new pool buffers, frees and graph captures.
//!
//! The cells are the served ones in served order, prompt lengths 15, 127, 1040 and 4158 (one small group, one
//! group, four groups and a tail of 16, sixteen groups and a tail of 62), run twice on one backend. The small
//! checkpoint's buffers are megabytes where the model's are gigabytes, so the vector pool's cap and trim rooms are
//! scaled down to this checkpoint (Backend::scale_vector_pool): its spares after a trim are 0.69 of the long cell's
//! decode set, as on the model (926 MB against 1.35 GB at 4,158 tokens). Without the scale nothing here ever meets
//! a bound, and the fixture passes whatever the code does. The decode reserve's cap is twice the decode sets of the
//! four cells as this checkpoint has them: a small checkpoint cannot show that the model's 1 GiB holds the model's
//! sets (its sizes do not keep the model's proportions); the served screen's first-step counts show that.
//!
//! Two arms, each in its own backend. CONTROL: no decode reserve, a new memo slab for each session, no trim at the
//! end of a prompt (Device::set_first_step(false)): it must show first-step allocations in the second round, or
//! this fixture can prove nothing. V3: from the second round on, decode steps 1 to 3 of every cell allocate nothing
//! and free nothing. Logits are equal across rounds and across arms, bit for bit.
use super::*;
use spark_gpu_probe::driver::Backend;
use spark_tp::{bounded_decoder::Device,row_split::{Exchange,Factory,Split}};
use spark_serve::{Engine,Session};
use engine_model::{plan::{description::{TokenMixer,TextDescription},layer::Topology},safetensors::Dtype};
#[allow(dead_code)]
#[path="../../../tests/support/attention_checkpoint.rs"] pub(super) mod attention;

struct OneRank;
impl Exchange for OneRank {
    fn topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn gather(&mut self,local:&[f32],_:usize)->Result<Vec<f32>> {Ok(local.to_vec())}
    fn command(&mut self,_:&mut [u8])->Result<()> {Ok(())}
    fn complete(&mut self)->Result<()> {Ok(())}
}
fn checkpoint(latents:usize,capacity:u64)->attention::Checkpoint {
    attention::Checkpoint::build_with(Dtype::Bf16,true,capacity,false,|d:&mut TextDescription| {
        let mut latent=d.layers[1];if let TokenMixer::LatentAttention(a)=&mut latent.mixer {a.indexer.token_budget=2048;a.indexer.pool=4;}
        d.layers.truncate(1);for _ in 0..latents {d.layers.push(latent);}
    })
}
fn prompt_tokens(length:usize)->Vec<u32> {(0..length).map(|i|1+(i as u32*5)%15).collect()}
struct Snap {pool:spark_gpu_probe::KvArenaStats,copies:spark_gpu_probe::resident_launch::CopyCounts,held:usize,bound:(usize,usize,usize)}
fn snap(b:&Backend<'_>)->Snap {
    let pool=b.vector_pool_stats();let held=pool.allocated_bytes.saturating_sub(pool.direct_freed_bytes).saturating_sub(pool.evicted_bytes);
    Snap {held,pool,copies:b.resident_copy_counts(),bound:b.vector_pool_bound()}
}
fn report(lines:&mut String,case:&str,session:usize,step:i64,ms:f64,before:&Snap,after:&Snap) {
    let line=format!("FIRST_STEP case={case} session={session} step={step} host_ms={ms:.3} new_buffers={} new_bytes={} reused={} direct_freed={} direct_freed_bytes={} evicted={} evicted_bytes={} \
        held_before={} held_after={} spares_before={} cap={} captures_new={} captures_seen={} piece_evictions={}\n",
        after.pool.allocated_count.saturating_sub(before.pool.allocated_count),after.pool.allocated_bytes.saturating_sub(before.pool.allocated_bytes),
        after.pool.reused_count.saturating_sub(before.pool.reused_count),after.pool.direct_freed_count.saturating_sub(before.pool.direct_freed_count),
        after.pool.direct_freed_bytes.saturating_sub(before.pool.direct_freed_bytes),after.pool.evicted_count.saturating_sub(before.pool.evicted_count),
        after.pool.evicted_bytes.saturating_sub(before.pool.evicted_bytes),before.held,after.held,before.bound.1,before.bound.2,
        after.copies.captures_new_key.saturating_sub(before.copies.captures_new_key),after.copies.captures_seen_key.saturating_sub(before.copies.captures_seen_key),
        after.copies.piece_evictions.saturating_sub(before.copies.piece_evictions));
    print!("{line}");lines.push_str(&line);
}
const CELLS:[(&str,usize);4]=[("c15",15),("c127",127),("c1040",1040),("c4158",4158)];
/// Decode steps counted for each session: the three first steps and three steady ones.
const STEPS:i64=6;
type Counts=[[(usize,usize,usize);3];4];
/// New bytes of the first three decode steps of each cell in the first round: its decode set.
type Sets=[usize;4];
/// One arm: two rounds of the four cells. Returns the logits of both rounds, for each cell of the second round the
/// (new buffers, frees at return, evictions) of its first three steps, and each cell's decode set of the first round.
/// `reserve`: the decode reserve's cap for a v3 arm, 0 for the control arm.
fn arm(c:&attention::Checkpoint,name:&str,reserve:usize,shift:u32,lines:&mut String)->Result<(Vec<Vec<Vec<u32>>>,Counts,Sets)> {
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1usize<<34,|b| {
        b.configure_vector_pool(1<<30);b.scale_vector_pool(shift);
        let d=Device::new(b);d.set_first_step(reserve!=0);if reserve!=0 {b.configure_decode_reserve(reserve);}
        let split=Split::new(&d,OneRank)?;let service=Factory::new(&c.admission,split,4200,vec![0])?;service.negotiate_available_bytes(1usize<<40)?;
        let mut rounds:Vec<Vec<Vec<u32>>>=Vec::new();let mut counts:Counts=Default::default();let mut sets:Sets=Default::default();
        for round in 0..2usize {
            let mut all:Vec<Vec<u32>>=Vec::new();
            for (cell,(case,len)) in CELLS.into_iter().enumerate() {
                let case=format!("{name}:{case}");let prompt=prompt_tokens(len);let mut s=service.session()?;
                let before=snap(b);let t=std::time::Instant::now();let logits=s.prefill(&prompt)?;let ms=t.elapsed().as_secs_f64()*1000.0;let after=snap(b);
                report(lines,&case,round,-1,ms,&before,&after);all.push(logits.iter().map(|v|v.to_bits()).collect());
                let mut token=1u32;
                for step in 0..STEPS {
                    let before=snap(b);let t=std::time::Instant::now();let logits=s.step(token)?;let ms=t.elapsed().as_secs_f64()*1000.0;let after=snap(b);
                    report(lines,&case,round,step,ms,&before,&after);all.push(logits.iter().map(|v|v.to_bits()).collect());
                    let new=after.pool.allocated_count.saturating_sub(before.pool.allocated_count);
                    if round==0 && step<3 {sets[cell]+=after.pool.allocated_bytes.saturating_sub(before.pool.allocated_bytes);}
                    if round>0 && step<3 {
                        let freed=after.pool.direct_freed_count.saturating_sub(before.pool.direct_freed_count);
                        counts[cell][step as usize]=(new,freed,after.pool.evicted_count.saturating_sub(before.pool.evicted_count));
                    }
                    token=1+(step as u32*5)%15;
                }
                drop(s);
                let rest=snap(b);let reserve=b.decode_reserve_bound();
                let line=format!("FIRST_STEP case={case} session={round} step=retire held_after={} pool_spares_after_trim={} pool_cap={} \
                    decode_reserve_bytes={} decode_reserve_peak={} decode_reserve_cap={}\n",
                    rest.held,rest.bound.1,rest.bound.2,reserve.0,reserve.1,reserve.2);
                print!("{line}");lines.push_str(&line);
            }
            rounds.push(all);
        }
        service.shutdown()?;Ok((rounds,counts,sets))
    })
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let mut lines=String::new();
    let c=checkpoint(11,4200);
    // The scale: one unscaled control arm gives each cell's decode set in this checkpoint's bytes. The model keeps
    // 926,258,176 B of spares after a trim, 0.69 of its long cell's decode set (1.35 GB).
    let (_,_,sets)=arm(&c,"measure",0,0,&mut String::new())?;
    if sets[3]==0 {return Err("first-step: the long cell allocated nothing in its first steps, so the fixture has no scale".into());}
    let ceiling=2_000_000_000usize-(1<<30);let mut shift=0u32;
    while shift<30 && (ceiling>>(shift+1))*100>=sets[3]*69 {shift+=1;}
    let reserve=2*sets.iter().sum::<usize>();
    let line=format!("FIRST_STEP SCALE decode_set_bytes c15={} c127={} c1040={} c4158={} shift={shift} pool_cap={} \
        spares_after_trim={} decode_reserve_cap={reserve}\n",
        sets[0],sets[1],sets[2],sets[3],2_000_000_000usize>>shift,ceiling>>shift);
    print!("{line}");lines.push_str(&line);
    let (control,control_counts,_)=arm(&c,"control",0,shift,&mut lines)?;
    let (v3,counts,_)=arm(&c,"v3",reserve,shift,&mut lines)?;
    if control[0]!=control[1] || v3[0]!=v3[1] {return Err("first-step: second-round logits differ from the first round".into());}
    if control!=v3 {return Err("first-step: the v3 arm's logits differ from the control arm's".into());}
    for (arm,counts) in [("control",&control_counts),("v3",&counts)] {
        let line=format!("FIRST_STEP SECOND_ROUND arm={arm} steps_1_to_3 (new_buffers, freed_at_return, evicted): {}\n",
            CELLS.iter().zip(counts.iter()).map(|((case,_),steps)|format!("{case} {steps:?}")).collect::<Vec<_>>().join(" "));
        print!("{line}");lines.push_str(&line);
    }
    // The control must show the disease: new buffers in the first steps of the second round, at three cells or more.
    let sick=control_counts.iter().filter(|steps|steps.iter().any(|step|step.0!=0)).count();
    if sick<3 {return Err(format!("first-step: the control arm allocates in the first steps of only {sick} cells; the fixture cannot fail, so it proves nothing"));}
    for ((case,_),steps) in CELLS.iter().zip(counts.iter()) {for (step,count) in steps.iter().enumerate() {
        if *count!=(0,0,0) {return Err(format!("first-step v3 {case} second round step {}: new_buffers={} freed_at_return={} evicted={}, want 0/0/0",step+1,count.0,count.1,count.2));}}}
    let line=format!("FIRST_STEP COMPLETE logits_equal_across_rounds_and_arms=1 control_cells_with_first_step_allocations={sick} \
        v3_second_round_steps_1_to_3_allocate_and_free_nothing=1 cells=15,127,1040,4158 scale_shift={shift} scope=one_box_fixture_not_served_grade\n");
    print!("{line}");lines.push_str(&line);
    write(&out.join("LINES"),lines.as_bytes())?;write(&out.join("COMPLETE"),b"FIRST-STEP PASS\n")?;Ok(())
}
