//! Standalone byte gate. It uses synthetic tensors and never opens weights.
#[cfg(not(target_os = "linux"))]
fn main() { eprintln!("swiglu-bytes requires a CUDA Linux host"); std::process::exit(1); }
#[cfg(target_os = "linux")]
fn main() {
    if let Err(e) = device::run() { eprintln!("FAIL swiglu-bytes: {e}"); std::process::exit(1); }
}

#[cfg(target_os = "linux")]
mod device {
    use engine_model::{plan::ConfigPlan, swiglu::SwiGlu};
    use spark_gpu_probe::driver::{Backend, with_backend};
    use std::{ffi::{CStr, CString}, fs};
    const WIDTHS: [usize; 8] = [1, 7, 31, 256, 257, 640, 2048, 8192];
    const ROWS: [usize; 3] = [1, 3, 17];
    fn inputs(len: usize) -> (Vec<f32>, Vec<f32>) {
        let edge = [-64.0, -20.0, -10.0001, -10.0, -9.9999, -1.0, -0.0,
            0.0, f32::from_bits(1), -f32::from_bits(1), 1.0, 9.9999, 10.0, 10.0001, 20.0, 64.0];
        let mut s = 0x12345678u32;
        let mut next = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s as i32 as f32) / 33554432.0 };
        let gate = (0..len).map(|i| if i < edge.len() { edge[i] } else { next() }).collect();
        let up = (0..len).map(|i| if i < edge.len() { edge[edge.len()-1-i] } else { next() }).collect();
        (gate, up)
    }
    fn launch(b: &Backend<'_>, gate: &[f32], up: &[f32], bf16: bool, limit: Option<f32>, chunk: usize)
        -> Result<Vec<u32>, String> {
        let mut buffers = b.vector_arena(&[gate.len(), up.len()])?;
        let mut u = buffers.pop().ok_or("missing up buffer")?;
        let mut g = buffers.pop().ok_or("missing gate buffer")?;
        b.write_vector(&mut g, gate)?; b.write_vector(&mut u, up)?;
        for first in (0..gate.len()).step_by(chunk) {
            let end = (first + chunk).min(gate.len());
            b.swiglu_config_device(&mut g.slice(first..end)?, &u.slice(first..end)?, bf16, limit)?;
        }
        Ok(b.read_vector_values(&g)?.into_iter().map(f32::to_bits).collect())
    }
    fn compare(got: &[u32], expected: &[u32], cell: &str) -> Result<(), String> {
        if got.len() != expected.len() { return Err(format!("{cell}: output length differs")); }
        if let Some(i) = got.iter().zip(expected).position(|(g,e)| g != e) {
            return Err(format!("{cell}: element={i} got={:08x} expected={:08x}", got[i], expected[i]));
        }
        Ok(())
    }
    pub fn run() -> Result<(), String> {
        let args = std::env::args_os().skip(1).collect::<Vec<_>>();
        let [baseline] = args.as_slice() else { return Err("usage: swiglu-bytes <parent-layer.ptx>".into()); };
        let bytes = fs::read(baseline).map_err(|e| e.to_string())?;
        let base = CString::new(bytes).map_err(|e| e.to_string())?;
        let ConfigPlan::Deferred(p) = ConfigPlan::from_config(include_bytes!("../../../model/tests/fixtures/glm/config.json"))
            .map_err(|e| e.to_string())? else { return Err("missing deferred operation fixture".into()); };
        let cap = p.swiglu().map_err(|e| e.to_string())?;
        println!("SWIGLU_BYTES revision={} parent=9c7a224505c48fce3fb573547c037f32001ed0a6 config_bound={:?}",
            include_str!("../../../../REVISION").trim(), cap.limit());
        let old = collect_uncapped(&base)?;
        let mut cells = 0;
        with_backend(0, spark_gpu_probe::layer_ptx()?, 64*1024*1024, |b| {
            let mut g = b.vector(1)?; let u = b.vector(1)?;
            for limit in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX, f32::from_bits(1)] {
                if b.swiglu_config_device(&mut g,&u,true,Some(limit)).is_ok() {
                    return Err(format!("invalid bound accepted: {limit}"));
                }
            }
            let mut at = 0;
            for width in WIDTHS { for rows in ROWS { for bf16 in [false, true] {
                let (gate, up) = inputs(width*rows);
                for limit in [None, cap.limit(), Some(3.5), Some(1.00390625)] {
                    let cell = format!("width={width} rows={rows} bf16={bf16} limit={limit:?}");
                    let expected = if limit.is_none() { old[at].clone() } else {
                        let c = SwiGlu::new(limit.map(f64::from)).map_err(|e| e.to_string())?;
                        gate.iter().zip(&up).map(|(&g,&u)| c.reference(g,u,bf16).to_bits()).collect()
                    };
                    for repeat in 0..3 {
                        compare(&launch(b,&gate,&up,bf16,limit,gate.len())?,&expected,&format!("{cell} repeat={repeat}"))?;
                    }
                    for chunk in [width, width+1] {
                        compare(&launch(b,&gate,&up,bf16,limit,chunk)?,&expected,&format!("{cell} chunk={chunk}"))?;
                    }
                    cells += 1;
                    println!("PASS {cell} elements={} bytes={} repeats=3 chunks={width},{}",
                        expected.len(),expected.len()*4,width+1);
                }
                at += 1;
            } } }
            Ok(())
        })?;
        println!("PASS swiglu-bytes cells={cells} capped_vs_host uncapped_vs_parent repeat chunk");
        Ok(())
    }
    fn collect_uncapped(ptx: &CStr) -> Result<Vec<Vec<u32>>, String> {
        with_backend(0,ptx,64*1024*1024,|b| {
            let mut out = Vec::new();
            for width in WIDTHS { for rows in ROWS { for bf16 in [false,true] {
                let (g,u) = inputs(width*rows);
                out.push(launch(b,&g,&u,bf16,None,g.len())?);
            } } }
            Ok(out)
        })
    }
}
