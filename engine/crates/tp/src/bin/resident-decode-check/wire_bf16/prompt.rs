//! One-device replay of the native fold and the mixed router stage. Peer rank
//! packets are made on the same device. Timing measures one rank's work.
use super::*;
use spark_gpu_probe::driver::DeviceVector;
use spark_tp::resident_weights::wire_packet::{self, Format, Layout};

struct Arm {
    formats: Vec<Format>,
    layout: Layout,
    wire_words: usize,
}

impl Arm {
    fn new(ranks: usize, tokens: usize, parts: &[(usize, Format)], compressed: bool) -> Result<Self> {
        let formats = parts.iter().map(|&(_, format)| if compressed { format } else { Format::Fp32 }).collect::<Vec<_>>();
        let local = parts.iter().zip(&formats).map(|(&(full, _), &format)| ((full / ranks) * tokens, format)).collect::<Vec<_>>();
        let layout = Layout::new(ranks, &local)?;
        Ok(Self { formats, wire_words: layout.words / ranks, layout })
    }

    fn write_rank<'d>(&self, b: &Backend<'d>, wire: &mut DeviceVector<'d>, rank: usize, parts: &[DeviceVector<'d>]) -> Result<()> {
        let mut at = rank * self.wire_words;
        for (part, &format) in parts.iter().zip(&self.formats) {
            at = wire_packet::write_part(b, part, wire, at, format)?;
        }
        Ok(())
    }

    fn receive<'d>(&self, b: &Backend<'d>, wire: &DeviceVector<'d>, ranks: usize, tokens: usize, widths: &[(usize, Format)]) -> Result<Vec<DeviceVector<'d>>> {
        let transpose = self.formats.len() > 1;
        let grouped = if transpose {
            let mut grouped = b.vector(self.layout.words)?;
            b.resident_scatter_copy(wire, &mut grouped, &self.layout.copies)?;
            Some(grouped.into_resident_shared()?)
        } else { None };
        let grouped = grouped.as_ref().unwrap_or(wire);
        self.layout.sections.iter().zip(widths).zip(&self.formats).map(|((section, &(full, _)), &format)| {
            let source = grouped.resident_range(section.clone())?;
            let local = full / ranks;
            let shape = match format {
                Format::Bf16 => wire_packet::unpack_shape(tokens, local, ranks, transpose, transpose)?,
                Format::Fp32 if transpose => Shape { rows: full as u32, cols: tokens as u32, first: 1, ..Shape::new(Op::Transpose, 1).unwrap() },
                Format::Fp32 => Shape::rank_rows(tokens, local, ranks, false).ok_or("fold layout")?,
            };
            b.prompt_operation(shape, [Some(&source), None, None, None, None], None)
        }).collect()
    }
}

fn operands(rank: usize, part: usize, words: usize, format: Format) -> Vec<f32> {
    (0..words).map(|word| {
        let high = ((word * 289 + rank * 173 + part * 443) % 0x7f80) as u32;
        let raw = match word {
            0 => 0,
            1 => 0x8000_0000,
            _ => (high << 16) | if format == Format::Fp32 { (word as u32 & 0xffff) | 1 } else { 0 },
        };
        f32::from_bits(raw)
    }).collect()
}

fn case(b: &Backend<'_>, family: &str, widths: &[(usize, Format)], ranks: usize, tokens: usize) -> Result<usize> {
    let sources = (0..ranks).map(|rank| widths.iter().enumerate().map(|(part, &(full, format))| {
        b.upload_vector(&operands(rank, part, tokens * full / ranks, format))
    }).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
    let arms = [Arm::new(ranks, tokens, widths, false)?, Arm::new(ranks, tokens, widths, true)?];
    let queue = b.resident_queue()?;
    b.enable_resident_launches()?;
    let mut wires = [b.mapped_vector(arms[0].layout.words)?, b.mapped_vector(arms[1].layout.words)?];
    for (arm, wire) in arms.iter().zip(&mut wires) {
        for (rank, parts) in sources.iter().enumerate() { arm.write_rank(b, wire, rank, parts)?; }
    }
    let [old, new] = [arms[0].receive(b, &wires[0], ranks, tokens, widths)?, arms[1].receive(b, &wires[1], ranks, tokens, widths)?];
    let mut compared = 0;
    for (part, (a, c)) in old.iter().zip(&new).enumerate() {
        let reference = b.read_vector_values(a)?;
        let got = b.read_vector_values(c)?;
        if reference.len()!=got.len() || reference.iter().map(|v| v.to_bits()).ne(got.iter().map(|v| v.to_bits())) {
            return Err(format!("prompt wire differs family={family} ranks={ranks} tokens={tokens} part={part}"));
        }
        if !got.iter().any(|word| word.to_bits()==0) || !got.iter().any(|word| word.to_bits()==0x8000_0000) {
            return Err("prompt wire signed-zero controls are absent".into());
        }
        compared += got.len();
    }
    // Use the wrong source coordinate for the first rank. The device buffer
    // must still be valid, so the control detects a change in the output words.
    arms[1].write_rank(b, &mut wires[1], 0, &sources[1])?;
    let wrong = arms[1].receive(b, &wires[1], ranks, tokens, widths)?;
    let good = b.read_vector_values(&new[widths.len()-1])?;
    let bad = b.read_vector_values(&wrong[widths.len()-1])?;
    if good.iter().map(|v| v.to_bits()).eq(bad.iter().map(|v| v.to_bits())) {
        return Err("prompt wire coordinate control gave the reference words".into());
    }
    let mut samples = [Vec::new(), Vec::new()];
    for _ in 0..9 {
        for index in [0, 1, 1, 0] {
            let arm = &arms[index];
            let wire = &mut wires[index];
            b.synchronize()?;
            let start = Instant::now();
            arm.write_rank(b, wire, 0, &sources[0])?;
            let output = arm.receive(b, wire, ranks, tokens, widths)?;
            b.synchronize()?;
            samples[index].push(start.elapsed().as_secs_f64()*1e6);
            drop(output);
        }
    }
    drop(queue);
    for values in &mut samples { values.sort_by(f64::total_cmp); }
    println!("PROMPT WIRE CASE family={family} ranks={ranks} tokens={tokens} words={compared} wrong_coordinate=refused old_us={:.3} pack_unpack_us={:.3} old_bytes={} wire_bytes={}",
        samples[0][samples[0].len()/2], samples[1][samples[1].len()/2], arms[0].wire_words*4, arms[1].wire_words*4);
    Ok(compared)
}

pub fn run(out: &Path) -> Result<()> {
    std::fs::create_dir(out).map_err(|e| e.to_string())?;
    let fold = [(4096, Format::Bf16)];
    let mixed = [(288, Format::Fp32), (2048, Format::Bf16), (2048, Format::Bf16)];
    let (mut cases, mut words) = (0, 0);
    spark_gpu_probe::driver::with_backend(0, spark_gpu_probe::layer_ptx()?, 2<<30, |b| {
        for ranks in [2, 4] {
            for tokens in [17, 256, 512, 768, 1024] {
                for (family, parts) in [("fold", &fold[..]), ("mixed", &mixed[..])] {
                    words += case(b, family, parts, ranks, tokens)?;
                    cases += 1;
                }
            }
        }
        Ok(())
    })?;
    for kind in [0, 1] {
        let child = std::process::Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
            .args(["wire-bf16-reject", &kind.to_string()]).output().map_err(|e| e.to_string())?;
        let proof = String::from_utf8(child.stdout).map_err(|e| e.to_string())?;
        if !child.status.success() || !proof.contains(&format!("WIRE REJECT PASS kind={kind} launched=true refused=true")) {
            return Err(format!("prompt wire invalid-word control did not stop the device at position {kind}"));
        }
        print!("{proof}");
    }
    let line = format!("PROMPT WIRE PASS cases={cases} words={words} differing_words=0 coordinate_controls={cases} invalid_word_controls=2\n");
    super::super::write(&out.join("COMPLETE"), line.as_bytes())?;
    print!("{line}");
    Ok(())
}
