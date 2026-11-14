//! CPU reference proof for the chunk-64 GDN scan:
//! absolute chunk starts, state snapshots only at 64-aligned boundaries, the open
//! chunk held as a tail, one canonical row function. A bulk chunk path written in
//! the GPU kernel's loop order must give the same bits as the row path, for every
//! way the token stream is cut into calls (prompt pieces, decode, MTP verify).
//! Canonical FP32 arithmetic (form DS, identical to flash_next_gdn_chunk.rs):
//!   gamma_i = gamma_{i-1} + g_i (gamma_{-1} = +0) ; e_i = exp(gamma_i)
//!   kk_ij = sum_d k_i k_j ; qk_ij = sum_d q_i k_j (d ascending, fma from +0)
//!   A_ij = (kk_ij * exp(gamma_i - gamma_j)) * beta_i (j < i)
//!   r_i = fma(-e_i, k_i.S_c, v_i) * beta_i ; v'_i = r_i, then
//!   v'_i = fma(-A_ij, v'_j, v'_i) for j = 0 .. i-1 ascending
//!   o_i = fma(e_i, q_i.S_c, sum_{j<=i} (qk_ij * exp(gamma_i - gamma_j)) v'_j)
//!   at row 63: S' = fma(S_c, e_63, sum_j (k_j * exp(gamma_63 - gamma_j)) v'_j)
const D: usize = 128;
const C: usize = 64;
fn fma(a: f32, b: f32, c: f32) -> f32 { a.mul_add(b, c) }
fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).fold(0.0f32, |s, (x, y)| fma(*x, *y, s)) }

#[derive(Clone)]
struct Row { q: Vec<f32>, k: Vec<f32>, v: Vec<f32>, beta: f32, g: f32 }
/// Recurrent state of one value head: S at the last 64-aligned boundary plus the
/// open chunk's rows (k, gamma, v'). Snapshots copy both.
#[derive(Clone, PartialEq)]
struct Head { s: Vec<f32>, k: Vec<Vec<f32>>, gamma: Vec<f32>, w: Vec<Vec<f32>> }
#[allow(clippy::assign_op_pattern, clippy::needless_range_loop)] // index loops and explicit `+` keep the arithmetic order.
impl Head {
    fn new(s: Vec<f32>) -> Self { Head { s, k: vec![], gamma: vec![], w: vec![] } }
    fn column(&self, x: &[f32], v: usize) -> f32 {
        let mut acc = 0.0f32;
        for d in 0..D { acc = fma(x[d], self.s[d * D + v], acc); }
        acc
    }
    /// The canonical row function: decode, MTP verify and mid-chunk prompt entry.
    fn row(&mut self, x: &Row) -> Vec<f32> {
        let i = self.k.len();
        let gamma = (if i == 0 { 0.0 } else { self.gamma[i - 1] }) + x.g;
        let e = gamma.exp();
        let a: Vec<f32> = (0..i).map(|j| (dot(&x.k, &self.k[j]) * (gamma - self.gamma[j]).exp()) * x.beta).collect();
        let mut w: Vec<f32> = (0..D).map(|v| fma(-e, self.column(&x.k, v), x.v[v]) * x.beta).collect();
        for j in 0..i {
            for v in 0..D { w[v] = fma(-a[j], self.w[j][v], w[v]); }
        }
        let p: Vec<f32> = (0..=i).map(|j| {
            let kj = if j == i { &x.k } else { &self.k[j] };
            dot(&x.q, kj) * (gamma - if j == i { gamma } else { self.gamma[j] }).exp()
        }).collect();
        let mut o = vec![0.0f32; D];
        for v in 0..D {
            let mut acc = 0.0f32;
            for j in 0..i { acc = fma(p[j], self.w[j][v], acc); }
            acc = fma(p[i], w[v], acc);
            o[v] = fma(e, self.column(&x.q, v), acc);
        }
        self.k.push(x.k.clone()); self.gamma.push(gamma); self.w.push(w);
        if self.k.len() == C { self.close(); }
        o
    }
    /// Row 63 accepted: fold the chunk into S (the only state write) and clear the tail.
    fn close(&mut self) {
        let last = self.gamma[C - 1];
        let decay = last.exp();
        let f: Vec<f32> = (0..C).map(|j| (last - self.gamma[j]).exp()).collect();
        let mut s = self.s.clone();
        for d in 0..D {
            for v in 0..D {
                let mut u = 0.0f32;
                for j in 0..C { u = fma(self.k[j][d] * f[j], self.w[j][v], u); }
                s[d * D + v] = fma(self.s[d * D + v], decay, u);
            }
        }
        *self = Head::new(s);
    }
    /// Bulk path for a whole aligned chunk in the GPU kernel's loop order
    /// (4x4 register tiles for kk/qk, right-looking substitution, P.v' bounded at j <= i).
    fn bulk(&mut self, rows: &[Row]) -> Vec<Vec<f32>> {
        assert!(self.k.is_empty() && rows.len() == C);
        let mut gamma = vec![0.0f32; C];
        let mut acc = 0.0f32;
        for i in 0..C { acc = acc + rows[i].g; gamma[i] = acc; }
        let mut kk = vec![0.0f32; C * C];
        let mut qk = vec![0.0f32; C * C];
        for ib in (0..C).step_by(4) {
            for jb in (0..=ib).step_by(4) {
                let mut tk = [[0.0f32; 4]; 4];
                let mut tq = [[0.0f32; 4]; 4];
                for d in 0..D {
                    for x in 0..4 {
                        for y in 0..4 {
                            tk[x][y] = fma(rows[ib + x].k[d], rows[jb + y].k[d], tk[x][y]);
                            tq[x][y] = fma(rows[ib + x].q[d], rows[jb + y].k[d], tq[x][y]);
                        }
                    }
                }
                for x in 0..4 { for y in 0..4 { kk[(ib + x) * C + jb + y] = tk[x][y]; qk[(ib + x) * C + jb + y] = tq[x][y]; } }
            }
        }
        let mut a = vec![0.0f32; C * C];
        let mut p = vec![0.0f32; C * C];
        for i in 0..C {
            for j in 0..C {
                if j < i { a[i * C + j] = (kk[i * C + j] * (gamma[i] - gamma[j]).exp()) * rows[i].beta; }
                if j <= i { p[i * C + j] = qk[i * C + j] * (gamma[i] - gamma[j]).exp(); }
            }
        }
        let e: Vec<f32> = gamma.iter().map(|g| g.exp()).collect();
        let mut w: Vec<Vec<f32>> = (0..C).map(|j| (0..D).map(|v| {
            let mut ks = 0.0f32;
            for d in 0..D { ks = fma(rows[j].k[d], self.s[d * D + v], ks); }
            fma(-e[j], ks, rows[j].v[v]) * rows[j].beta
        }).collect()).collect();
        // Right-looking substitution: once v'_j is final, every later row takes its term.
        // Negative control: GDN_PROOF_NEGATIVE=1 applies the terms in descending j and must fail.
        let negative = std::env::var("GDN_PROOF_NEGATIVE").is_ok_and(|v| v == "1");
        if negative {
            for m in 1..C { for j in (0..m).rev() { for v in 0..D { w[m][v] = fma(-a[m * C + j], w[j][v], w[m][v]); } } }
        } else {
            for j in 0..C {
                for m in j + 1..C { for v in 0..D { w[m][v] = fma(-a[m * C + j], w[j][v], w[m][v]); } }
            }
        }
        let mut out = vec![vec![0.0f32; D]; C];
        for i in 0..C {
            for v in 0..D {
                let mut acc = 0.0f32;
                for j in 0..=i { acc = fma(p[i * C + j], w[j][v], acc); }
                let mut qs = 0.0f32;
                for d in 0..D { qs = fma(rows[i].q[d], self.s[d * D + v], qs); }
                out[i][v] = fma(e[i], qs, acc);
            }
        }
        let last = gamma[C - 1];
        let f: Vec<f32> = (0..C).map(|j| (last - gamma[j]).exp()).collect();
        let mut s = self.s.clone();
        for d in 0..D {
            for v in 0..D {
                let mut u = 0.0f32;
                for j in 0..C { u = fma(rows[j].k[d] * f[j], w[j][v], u); }
                s[d * D + v] = fma(self.s[d * D + v], e[C - 1], u);
            }
        }
        *self = Head::new(s);
        out
    }
    /// One call over rows: finish an open chunk with the row function, full
    /// aligned chunks in bulk, a trailing partial chunk into the tail (no fold).
    fn call(&mut self, rows: &[Row]) -> Vec<Vec<f32>> {
        let mut out = vec![];
        let mut at = 0;
        while at < rows.len() && !self.k.is_empty() { out.push(self.row(&rows[at])); at += 1; }
        while rows.len() - at >= C { out.extend(self.bulk(&rows[at..at + C])); at += C; }
        while at < rows.len() { out.push(self.row(&rows[at])); at += 1; }
        out
    }
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}
fn stream(n: usize, seed: u64) -> Vec<Row> {
    let mut r = Rng(seed);
    (0..n).map(|_| {
        let mut q: Vec<f32> = (0..D).map(|_| r.unit() * 2. - 1.).collect();
        let mut k: Vec<f32> = (0..D).map(|_| r.unit() * 2. - 1.).collect();
        let (nq, nk) = (q.iter().map(|x| x * x).sum::<f32>().sqrt(), k.iter().map(|x| x * x).sum::<f32>().sqrt());
        q.iter_mut().for_each(|x| *x = *x / nq / (D as f32).sqrt());
        k.iter_mut().for_each(|x| *x /= nk);
        let v = (0..D).map(|_| r.unit() * 2. - 1.).collect();
        let beta = 1. / (1. + (-(r.unit() * 6. - 3.)).exp());
        let g = -((r.unit() * (0.001f32.ln().abs())) - 0.001f32.ln().abs()).exp();
        Row { q, k, v, beta, g }
    }).collect()
}
fn bits(o: &[Vec<f32>]) -> Vec<u32> { o.iter().flatten().map(|x| x.to_bits()).collect() }

/// Runs every split; returns (cases, failures). Prints one line per case.
pub fn run() -> (usize, usize) {
    let total = 3 * C + 7; // 199 tokens: three boundaries plus a partial chunk
    let rows = stream(total, 0x51ed_270b_c0ff_ee11);
    let mut init = Rng(7);
    let s0: Vec<f32> = (0..D * D).map(|_| init.unit() * 0.2 - 0.1).collect();
    // Reference: the row function alone (pure decode from token 0).
    let mut reference = Head::new(s0.clone());
    let want: Vec<Vec<f32>> = rows.iter().map(|x| reference.row(x)).collect();
    let want_bits = bits(&want);
    let mut cases = 0;
    let mut failures = 0;
    let mut check = |name: String, out: Vec<Vec<f32>>, head: &Head| {
        cases += 1;
        let same = bits(&out) == want_bits && *head == reference;
        if !same { failures += 1; }
        println!("gdn-cpu-proof {name}: {}", if same { "bytes equal (outputs + boundary state + tail)" } else { "MISMATCH" });
    };
    // One prompt over everything (bulk for the three full chunks, tail for 7 rows).
    let mut h = Head::new(s0.clone());
    let out = h.call(&rows);
    check("prompt 0..199".into(), out, &h);
    // Prompt split at 63 / 64 / 65 (and 127/128/129), then the rest as one prompt.
    for cut in [1, 63, 64, 65, 127, 128, 129, 191, 192, 193] {
        let mut h = Head::new(s0.clone());
        let mut out = h.call(&rows[..cut]);
        out.extend(h.call(&rows[cut..]));
        check(format!("prompt 0..{cut} + prompt {cut}..199"), out, &h);
    }
    // Prompt then plain decode (one token per call).
    for cut in [63, 64, 65] {
        let mut h = Head::new(s0.clone());
        let mut out = h.call(&rows[..cut]);
        for x in &rows[cut..] { out.extend(h.call(std::slice::from_ref(x))); }
        check(format!("prompt 0..{cut} + decode"), out, &h);
    }
    // Uneven pieces crossing boundaries (chunked prefill with arbitrary sizes).
    {
        let mut h = Head::new(s0.clone());
        let mut out = vec![];
        let mut at = 0;
        for piece in [5, 70, 1, 60, 30, 33] { let end = (at + piece).min(total); out.extend(h.call(&rows[at..end])); at = end; }
        out.extend(h.call(&rows[at..]));
        check("pieces 5,70,1,60,30,33,rest".into(), out, &h);
    }
    // MTP: verify spans of 3 rows on a snapshot; accept 0..=3, roll back the rest.
    for accept in 0..=3usize {
        let mut h = Head::new(s0.clone());
        let mut out = h.call(&rows[..62]);
        let mut at = 62;
        while at < total {
            let span = 3.min(total - at);
            let snapshot = h.clone();
            let verified = h.call(&rows[at..at + span]);
            let keep = accept.max(1).min(span); // the bonus token always advances one row
            h = snapshot;
            let kept = h.call(&rows[at..at + keep]);
            assert_eq!(bits(&kept), bits(&verified[..keep]), "verify rows differ from accepted rows");
            out.extend(kept);
            at += keep;
        }
        check(format!("prompt 0..62 + MTP verify 3, accept {accept}"), out, &h);
    }
    println!("gdn-cpu-proof: {cases} cases, {failures} mismatches");
    (cases, failures)
}
