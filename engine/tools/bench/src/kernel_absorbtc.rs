//! Latent fold timing only. Device correctness is checked by absorb-tc-rows.
//! Base uses the scalar grid; candidate uses the tensor-core grid. No token proof.
#![forbid(unsafe_code)]
use super::{Gpu, Result, ptr, sha256};
use std::time::Instant;
const GUARD: [u8; 16] = [0x5a; 16];
/// The third form of the tile (seat F8TC): bits against the first form, and time.
#[path = "kernel_latent3.rs"] pub(crate) mod latent3;
/// The small form of query and project in the tile (seat CHATS4): bits of the new image against the old, and time.
#[path = "kernel_latentsmall.rs"] pub(crate) mod latentsmall;
/// Mirror of gpu/src/prompt_ops.rs `Shape` and `Arguments` (repr(C)). The op numbers follow the enum order there:
/// AbsorbQuery 21, AbsorbScores 22, AbsorbValues 24, AbsorbProject 25. The kernel traps on any other op.
#[repr(C)] #[derive(Clone, Copy)]
struct Inner { op: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, epsilon: f32, bound: f32 }
#[repr(C)] #[derive(Clone, Copy)]
struct PShape { op: u32, tokens: u32, heads: u32, dim: u32, taps: u32, rows: u32, cols: u32, first: u32, slots: u32, inner: Inner }
#[repr(C)] #[derive(Clone, Copy)]
struct Arguments { input: [u64; 5], lengths: [u64; 5], weight: u64, weights: u64, output: u64, units: u64, words: u64, shape: PShape }
const OPS: [(u32, &str); 4] = [(21, "query"), (22, "scores"), (24, "values"), (25, "project")];
/// The tile of gpu/src/latent_absorb_rows.rs: a block owns 32 A rows x 64 B rows of one outer index, 256 threads.
const TILE_A: usize = 32; const TILE_B: usize = 64; const THREADS: u32 = 256;
struct Rng(u64);
impl Rng {
    fn step(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f32 { ((self.step() >> 40) as f32 / 16777216.0) * 2.0 - 1.0 }
}
/// The activation classes of kernel_plainp.rs: 0 plain, 2 denormals and signed zeros, 4 overflow with NaN and inf,
/// 5 huge/tiny alternation, 6 add-order sensitive (2^24 steps, cancelling pairs). 9: probabilities in [0, 1).
fn pattern(kind: u32, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed | 1);
    (0..n).map(|i| {
        let u = r.unit(); let s = r.step();
        match kind {
            0 => u * 4.0,
            2 => match s % 8 { 0 => 0.0, 1 => -0.0, 2 | 3 => f32::from_bits((s >> 8) as u32 % 8388607 + 1) * if s & 64 == 0 { 1.0 } else { -1.0 }, _ => u * 1e-3 },
            4 => { let v = u.signum() * (1.0 + u.abs()) * 2f32.powi(80); if i % 4099 == 7 { f32::NAN } else if i % 4099 == 11 { f32::INFINITY } else if i % 4099 == 13 { f32::NEG_INFINITY } else { v } }
            5 => if i % 2 == 0 { u * 4096.0 + 8192.0 } else { u * 1e-3 },
            6 => match i % 4 { 0 => 16777216.0, 1 => 1.0 + (s % 3) as f32, 2 => -16777216.0, _ => 1.0 },
            _ => (u + 1.0) * 0.5,
        }
    }).collect()
}
fn le(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
/// Two BF16 patterns a word, as the loader packs the key and value weights.
fn bf16_bits(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect() }
/// 16-byte guards on both sides; the data address stays 16-byte aligned (allocations are 256-byte aligned).
#[derive(Clone, Copy)]
struct Buffer { base: u64, address: u64, bytes: usize }
impl Buffer {
    fn raw(g: &mut Gpu, data: &[u8]) -> Result<Self> {
        let base = g.upload(&[GUARD.as_slice(), data, GUARD.as_slice()].concat())?;
        Ok(Self { base, address: base + 16, bytes: data.len() })
    }
    fn new(g: &mut Gpu, values: &[f32]) -> Result<Self> { Self::raw(g, &le(values)) }
    fn read(&self, g: &Gpu) -> Result<Vec<u8>> { g.read(self.base, self.bytes + 32) }

}
/// One launch shape. `rows` is the cache length after the group, `slots` the selection capacity.
#[derive(Clone, Copy)]
struct Case { tokens: usize, heads: usize, dim: usize, taps: usize, cols: usize, rows: usize, slots: usize }
impl Case {
    fn first(&self) -> usize { self.slots.min(self.rows) }
    fn model(tokens: usize, rows: usize) -> Self { Self { tokens, heads: 32, dim: 256, taps: 256, cols: 512, rows, slots: 2051 } }
    /// (outer count, A rows, B rows) and the output words of one operation, as latent_absorb_rows::tile and lengths.
    fn tile(&self, op: u32) -> (usize, usize, usize, usize) {
        let (m, h, v, k, c) = (self.tokens, self.heads, self.taps, self.cols, self.first());
        match op { 21 => (h, m, k, m * h * k), 25 => (h, m, v, m * h * v), 22 => (m, h, c, m * h * c), _ => (m, h, k, m * h * k) }
    }
    fn blocks(&self, op: u32) -> Result<u32> {
        let (outer, a, b, _) = self.tile(op);
        u32::try_from(outer * a.div_ceil(TILE_A) * b.div_ceil(TILE_B)).map_err(|_| "latent tile grid overflow".to_string())
    }
}
/// Selected rows `[tokens, slots+1]`, the count last. `served`: token t sees context rows-tokens+t+1 and keeps
/// 4 x min(context/4, 512) pool rows plus the tail rows, as the bounded decoder does (pools spread over the context,
/// in selection order, not sorted). Otherwise: counts of every residue, some zero, IDs spread over the cache.
fn selection(c: &Case, served: bool, seed: u64) -> Vec<f32> {
    let (m, r, n, cap) = (c.tokens, c.rows, c.slots, c.first()); let mut out = vec![0.0f32; m * (n + 1)];
    for t in 0..m {
        let row = &mut out[t * (n + 1)..(t + 1) * (n + 1)]; let mut count = 0;
        if served {
            let context = r - m + t + 1; let pools = context / 4; let kept = pools.min(512);
            let mut stride = (pools * 5 / 8) | 1; while gcd(stride, pools.max(1)) != 1 { stride += 2; }
            for j in 0..kept { let pool = if pools <= 512 { j } else { (j * stride + t * 7) % pools }; for e in 0..4 { if count < cap { row[count] = (pool * 4 + e) as f32; count += 1; } } }
            for e in 0..context % 4 { if count < cap { row[count] = (pools * 4 + e) as f32; count += 1; } }
        } else {
            let want = if t % 11 == 5 { 0 } else { 1 + (t * 37 + seed as usize * 13) % cap };
            let mut stride = (r * 3 / 7) | 1; while gcd(stride, r) != 1 { stride += 2; }
            for j in 0..want.min(r) { row[count] = ((j * stride + t * 29 + seed as usize) % r) as f32; count += 1; }
        }
        row[n] = count as f32;
    }
    out
}
fn gcd(a: usize, b: usize) -> usize { if b == 0 { a } else { gcd(b, a % b) } }
/// The operands of the four operations of one shape. Each operation has its own synthetic operands.
struct Fixture { query: Buffer, absorbed: Buffer, probability: Buffer, mixed: Buffer, latent: Buffer, selected: Buffer, weight: Buffer }
fn fixture(g: &mut Gpu, c: &Case, kind: u32, served: bool, seed: u64) -> Result<Fixture> {
    let (m, h, d, v, k, r) = (c.tokens, c.heads, c.dim, c.taps, c.cols, c.rows);
    Ok(Fixture { query: Buffer::new(g, &pattern(kind, m * h * d, seed))?, absorbed: Buffer::new(g, &pattern(kind, m * h * k, seed + 1))?,
        probability: Buffer::new(g, &pattern(if kind == 0 { 9 } else { kind }, m * h * c.first(), seed + 2))?, mixed: Buffer::new(g, &pattern(kind, m * h * k, seed + 3))?,
        latent: Buffer::new(g, &pattern(kind, r * k, seed + 4))?, selected: Buffer::new(g, &selection(c, served, seed))?,
        weight: Buffer::raw(g, &bf16_bits(&pattern(kind, h * (d + v) * k, seed + 5)))? })
}
fn arguments(c: &Case, op: u32, fx: &Fixture, output: u64) -> Arguments {
    let inner = Inner { op: 0, heads: 0, dim: 0, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 };
    let (m, h, d, v, k, r, n) = (c.tokens, c.heads, c.dim, c.taps, c.cols, c.rows, c.slots);
    let (selected, weights) = ((m * (n + 1)) as u64, (h * (d + v) * k / 2) as u64);
    let (input, lengths, weight) = match op {
        21 => ([fx.query.address, 4, 4, 4, 4], [(m * h * d) as u64, 0, 0, 0, 0], fx.weight.address),
        22 => ([fx.absorbed.address, fx.latent.address, fx.selected.address, 4, 4], [(m * h * k) as u64, (r * k) as u64, selected, 0, 0], 4),
        24 => ([fx.probability.address, fx.latent.address, fx.selected.address, 4, 4], [(m * h * c.first()) as u64, (r * k) as u64, selected, 0, 0], 4),
        _ => ([fx.mixed.address, fx.selected.address, 4, 4, 4], [(m * h * k) as u64, selected, 0, 0, 0], fx.weight.address),
    };
    Arguments { input, lengths, weight, weights: if weight == 4 { 0 } else { weights }, output, units: c.tile(op).3 as u64, words: 1,
        shape: PShape { op, tokens: m as u32, heads: h as u32, dim: d as u32, taps: v as u32, rows: r as u32, cols: k as u32, first: c.first() as u32, slots: n as u32, inner } }
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a,b|a.total_cmp(b));v[v.len()/2] }
pub(crate) fn run(base_path:&str,candidate_path:&str)->Result<()> {
    let ia=std::fs::read(base_path).map_err(|e|e.to_string())?;
    let ib=std::fs::read(candidate_path).map_err(|e|e.to_string())?;
    println!("ABSORB BENCH base={} candidate={} timing=launch_and_sync",sha256::digest(&ia),sha256::digest(&ib));
    let mut g=Gpu::new()?;let ma=g.module(&ia)?;let mb=g.module(&ib)?;
    let fs=[g.function(ma,"latent_absorption_tile")?,g.function(mb,"latent_absorption_tile")?];
    for tokens in [1usize,2,4,8,256,1024] {
        let c=Case::model(tokens,8192);let fx=fixture(&mut g,&c,0,true,31)?;let mut totals=[0.0f64;4];
        for (op,name) in OPS {
            let out=Buffer::new(&mut g,&vec![0.0;c.tile(op).3])?;
            let mut a=arguments(&c,op,&fx,out.address);let (outer,n,m,_)=c.tile(op);
            let grids=[c.blocks(op)?,(outer*m.div_ceil(128)*n.div_ceil(32)) as u32];
            let mut times:[Vec<f64>;4]=std::array::from_fn(|_|Vec::new());
            for arm in [0,1] {g.launch(fs[arm],[grids[arm],1,1],THREADS,0,&mut [ptr(&mut a)])?;g.sync()?;}
            for _ in 0..7 {for (cell,arm) in [0usize,1,1,0].into_iter().enumerate() {
                g.sync()?;let start=Instant::now();g.launch(fs[arm],[grids[arm],1,1],THREADS,0,&mut [ptr(&mut a)])?;g.sync()?;
                times[cell].push(start.elapsed().as_secs_f64()*1e3);
            }}
            let ms=times.map(median);for cell in 0..4 {totals[cell]+=ms[cell];}
            let bytes=out.read(&g)?;
            if bytes[..16]!=GUARD || bytes[bytes.len()-16..]!=GUARD {return Err("output guard changed".into());}
            println!("ABSORB BENCH op={name} tokens={tokens} base1_ms={:.4} candidate1_ms={:.4} candidate2_ms={:.4} base2_ms={:.4}",ms[0],ms[1],ms[2],ms[3]);
        }
        println!("ABSORB BENCH total tokens={tokens} base1_ms={:.4} candidate1_ms={:.4} candidate2_ms={:.4} base2_ms={:.4}",totals[0],totals[1],totals[2],totals[3]);
        g.clear()?;
    }
    step_times(&mut g,ma,mb,1)?;
    step_times(&mut g,ma,mb,13)?;
    Ok(())
}
#[repr(C)]
struct StepArguments {input:[u64;4],weight:u64,output:u64,shape:Inner}
/// One decode row. `copies` 1: every launch reads the same operands (they stay in the cache: the hot floor of a
/// launch). `copies` above 1: the launches rotate through that many separate copies of the weights and of the cache
/// rows, so a launch reads operands that left the cache, as a served step does (11 layers and 4 GB of other weights lie
/// between two launches of one layer).
fn step_times(g:&mut Gpu,ma:super::H,mb:super::H,copies:usize)->Result<()> {
    let label=if copies==1 {"STEP"} else {"COLD STEP"};
    for live in [2051usize,256,16] {
        let c=Case::model(1,live);let mut fxs=Vec::new();let mut counts=Vec::new();
        for copy in 0..copies {fxs.push(fixture(g,&c,0,true,61+copy as u64*7)?);counts.push(Buffer::new(g,&[live as f32])?);}
        let mut totals=[0.0;4];
        for (i,(op,name)) in OPS.into_iter().enumerate() {
            let words=c.tile(op).3;let out=Buffer::new(g,&vec![0.0;words])?;
            let entry=match op {22=>"latent_absorption_scores",24=>"latent_absorption_values",_=>"latent_absorption"};
            let fs=[g.function(ma,entry)?,g.function(mb,entry)?];
            let mut args=Vec::new();
            for copy in 0..copies {let pa=arguments(&c,op,&fxs[copy],out.address);
                args.push(StepArguments {input:[pa.input[0],fxs[copy].latent.address,counts[copy].address,4],weight:pa.weight,output:out.address,
                    shape:Inner {op:41+i as u32,heads:32,dim:256,taps:256,rows:live as u32,cols:512,epsilon:0.0,bound:0.0}});}
            let old=match op {22=>(8*live.next_multiple_of(256)).div_ceil(4),24=>8*16,_=>words.div_ceil(4)};
            let (outer,n,m,_)=c.tile(op);let candidate=outer*(if op==22 {m.next_multiple_of(256)} else {m}).div_ceil(if op==22 || op==24 {32} else {128})*n.div_ceil(32);
            let grids=[old as u32,candidate as u32];let threads=[if op==24 {1024} else {128},256];
            let mut times:[Vec<f64>;4]=std::array::from_fn(|_|Vec::new());let mut turn=0usize;
            for arm in [0,1] {g.launch(fs[arm],[grids[arm],1,1],threads[arm],0,&mut [ptr(&mut args[0])])?;g.sync()?;}
            for _ in 0..7 {for (cell,arm) in [0usize,1,1,0].into_iter().enumerate() {
                turn+=1;let a=&mut args[turn%copies];
                g.sync()?;let start=Instant::now();g.launch(fs[arm],[grids[arm],1,1],threads[arm],0,&mut [ptr(a)])?;g.sync()?;
                times[cell].push(start.elapsed().as_secs_f64()*1e3);
            }}
            let ms=times.map(median);for cell in 0..4 {totals[cell]+=ms[cell];}
            println!("ABSORB BENCH {label} op={name} selected={live} base1_ms={:.4} candidate1_ms={:.4} candidate2_ms={:.4} base2_ms={:.4}",ms[0],ms[1],ms[2],ms[3]);
        }
        println!("ABSORB BENCH {label} total selected={live} base1_ms={:.4} candidate1_ms={:.4} candidate2_ms={:.4} base2_ms={:.4}",totals[0],totals[1],totals[2],totals[3]);g.clear()?;
    }
    Ok(())
}

pub(in crate::cuda) fn diag_replay(g:&mut Gpu,module:super::H)->Result<()> {
 for rows in [1024usize,4096,8192] {let c=Case::model(1024,rows);let fx=fixture(g,&c,0,true,43)?;
  let selection=selection(&c,true,43);let selected=selection.chunks(c.slots+1).map(|r|r[c.slots]as usize).sum::<usize>();
  for(op,name)in[(22u32,"latent_scores3"),(24,"latent_values3")]{let words=c.tile(op).3;let out=Buffer::new(g,&vec![f32::NAN;words])?;let a=arguments(&c,op,&fx,out.address);let f=g.function(module,name)?;
   let launch=|g:&Gpu|{let mut a=a;g.launch(f,[if op==22{c.tokens as u32}else{4*c.tokens as u32},1,1],256,0,&mut [ptr(&mut a)])};
   super::diag_limits::replay(g,name,&format!("M=1024 heads=32 latent=512 cache_rows={rows} capacity={} selected_total={selected} source=REGENERATED_SERVED_SCHEDULE",c.first()),out.address,words,None,&launch)?;
  }g.clear()?;
 }Ok(())
}
