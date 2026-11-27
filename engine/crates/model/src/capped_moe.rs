//! Host reference of a routed expert block with soft-capped activation, for
//! one token: the router pick, one expert, and the fold of the picked experts.
//!
//! Source (read for the math, no code copied): the public `transformers`
//! release v5.24.0, `models/inkling2_loop/modeling_inkling2_loop.py`.
//! Router: lines 1152-1172. Activation: lines 760-771. One expert and the
//! sum: lines 787-811. Row order of the gate-and-up matrix:
//! `conversion_mapping.py` lines 2658-2667 and `core_model_loading.py`
//! lines 232-251.
//!
//! Number contract of each step (FP32 = 32-bit float):
//! - Router: FP32 for the logits, the scores, the pick and the weights.
//! - `W13 * z` and `W2 * act`: BF16 inputs and BF16 weights. Each output is
//!   one rising FP32 chain from 0.0 over the columns, each product rounded
//!   before its add, then rounded to BF16.
//! - Activation: FP32, then rounded to BF16.
//! - Fold: each term `weight * output` in FP32, added in FP32, and the sum
//!   rounded to BF16 one time.
//!
//! Orders that the source leaves open are fixed here and hold in every path:
//! - The pick is in falling `score + bias`. On equal values the lower expert
//!   number comes first. The weight sum and the fold add in that order.
//! - Sums inside a matrix product are the rising chain above.
//!
//! So the bits of this reference are stable. Last-bit parity with the source
//! is not claimed. The weights of an expert come from
//! [`crate::expert_widen::widen`] only. That function is the seam for a
//! later form that keeps the stored precision.
use crate::{Result, invalid, expert_widen::{self, StoredMatrix}, flash::nvfp4::bf16_to_f32,
    norm_cap::{SoftCap, check_cap, round_bf16, sigmoid}, shapes::MAX_EXPERTS};

/// Added to the sum of the picked scores before the divide. Source line 1171.
pub const ROUTE_EPSILON: f32 = 1e-20;

/// Row order of a gate-and-up matrix with `2 * width` rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateUpRows {
    /// Row `2i` is gate row `i`. Row `2i + 1` is up row `i`.
    Interleaved,
    /// Rows `0..width` are the gate rows. Rows `width..2 * width` are the up rows.
    Halves,
}
/// The order in the stored matrix. The ignored test `capped_moe_trace` fails with the other order.
pub const STORED_GATE_UP_ROWS: GateUpRows = GateUpRows::Interleaved;
impl GateUpRows {
    pub fn gate(self, i: usize, _width: usize) -> usize { match self { Self::Interleaved => 2 * i, Self::Halves => i } }
    pub fn up(self, i: usize, width: usize) -> usize { match self { Self::Interleaved => 2 * i + 1, Self::Halves => width + i } }
}

/// Router sizes and scale. The values come from the config, not from a name.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Routing { experts: usize, selected: usize, scale: f32 }
/// The picked experts in fold order, each with its weight.
#[derive(Debug, Clone, PartialEq)]
pub struct Route { pub ids: Vec<usize>, pub weights: Vec<f32> }

impl Routing {
    pub fn new(experts: usize, selected: usize, scale: f64) -> Result<Self> {
        if experts == 0 || experts as u64 > MAX_EXPERTS || selected == 0 || selected > experts {
            return Err(invalid("router needs 1 or more picked experts and not more than the expert count"));
        }
        Ok(Self { experts, selected, scale: check_cap(scale, "route scale")? })
    }
    pub fn experts(&self) -> usize { self.experts }
    pub fn selected(&self) -> usize { self.selected }
    pub fn scale(&self) -> f32 { self.scale }

    /// `logit[e] = weight[e] . z` in FP32. `weight` is `[experts, hidden]`, row-major.
    pub fn logits(&self, weight: &[f32], z: &[f32]) -> Result<Vec<f32>> {
        if z.is_empty() || self.experts.checked_mul(z.len()) != Some(weight.len()) { return Err(invalid("router weight must be experts by hidden")); }
        let logits = rows(weight, z);
        if logits.iter().any(|v| !v.is_finite()) { return Err(invalid("router logit is not finite")); }
        Ok(logits)
    }

    /// Pick and weigh the experts of one token.
    /// 1. `score = sigmoid(logit)`.
    /// 2. Pick by falling `score + bias`, the lower expert number first on
    ///    equal values. The bias is for the pick only.
    /// 3. `weight = score / (sum of the picked scores + 1e-20) * scale`. The
    ///    sum adds the picked scores from 0.0 in pick order.
    pub fn route(&self, logits: &[f32], bias: &[f32]) -> Result<Route> {
        if logits.len() != self.experts || bias.len() != self.experts { return Err(invalid("router needs one logit and one bias for each expert")); }
        let scores: Vec<f32> = logits.iter().map(|&v| sigmoid(v)).collect();
        let mut choice: Vec<f32> = scores.iter().zip(bias).map(|(&s, &b)| s + b).collect();
        if logits.iter().chain(bias).chain(&choice).any(|v| !v.is_finite()) { return Err(invalid("router logit, bias or corrected score is not finite")); }
        let (mut ids, mut weights, mut sum) = (Vec::with_capacity(self.selected), Vec::with_capacity(self.selected), 0.0f32);
        for _ in 0..self.selected {
            // Rising expert numbers and a strict comparison keep the lower number on a tie.
            let (mut best, mut id) = (f32::NEG_INFINITY, 0usize);
            for (expert, &value) in choice.iter().enumerate() { if value > best { best = value; id = expert; } }
            choice[id] = f32::NEG_INFINITY;
            sum += scores[id];
            ids.push(id); weights.push(scores[id]);
        }
        let denominator = sum + ROUTE_EPSILON;
        for weight in &mut weights { *weight = *weight / denominator * self.scale; }
        if weights.iter().any(|v| !v.is_finite()) { return Err(invalid("router weight is not finite")); }
        Ok(Route { ids, weights })
    }
}

/// `cap(gate) * sigmoid(gate) * cap(up)`, each cap a [`SoftCap`]. The sigmoid reads the raw gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CappedActivation { gate: SoftCap, up: SoftCap }
impl CappedActivation {
    pub fn new(gate_cap: f64, up_cap: f64) -> Result<Self> {
        Ok(Self { gate: SoftCap::new(gate_cap, "gate soft cap")?, up: SoftCap::new(up_cap, "up soft cap")? })
    }
    /// FP32 in, FP32 out, multiplied from left to right. The caller rounds the result.
    pub fn apply(self, gate: f32, up: f32) -> f32 { (self.gate.apply(gate) * sigmoid(gate)) * self.up.apply(up) }
}

/// One expert: a gate-and-up matrix `[2 * width, hidden]` in stored row order
/// and a down matrix `[hidden, width]`. All weights are BF16 values.
#[derive(Debug, Clone, PartialEq)]
pub struct Expert { hidden: usize, width: usize, gate_up: Vec<f32>, down: Vec<f32> }
impl Expert {
    /// Make the expert numbers from the stored bytes. This is the seam.
    pub fn from_stored(hidden: usize, width: usize, gate_up: &StoredMatrix<'_>, down: &StoredMatrix<'_>) -> Result<Self> {
        if width.checked_mul(2) != Some(gate_up.rows) || gate_up.columns != hidden || down.rows != hidden || down.columns != width {
            return Err(invalid("stored expert matrices do not match hidden and width"));
        }
        Self::from_bf16(hidden, width, &expert_widen::widen(gate_up)?, &expert_widen::widen(down)?)
    }
    /// Take BF16 bits that are already widened, row-major.
    pub fn from_bf16(hidden: usize, width: usize, gate_up: &[u16], down: &[u16]) -> Result<Self> {
        let cells = hidden.checked_mul(width).filter(|&n| n > 0).ok_or_else(|| invalid("expert shape is empty or too large"))?;
        if cells.checked_mul(2) != Some(gate_up.len()) || down.len() != cells { return Err(invalid("expert weights do not match hidden and width")); }
        let values = |bits: &[u16]| -> Result<Vec<f32>> {
            let values: Vec<f32> = bits.iter().map(|&b| bf16_to_f32(b)).collect();
            if values.iter().any(|v| !v.is_finite()) { return Err(invalid("expert weight is not finite")); }
            Ok(values)
        };
        Ok(Self { hidden, width, gate_up: values(gate_up)?, down: values(down)? })
    }
    pub fn hidden(&self) -> usize { self.hidden }
    pub fn width(&self) -> usize { self.width }

    /// `z` holds `hidden` BF16 values. The result holds `hidden` BF16 values.
    /// 1. `gu = round(W13 * z)`.
    /// 2. `act[i] = round(activation(gate[i], up[i]))` with the rows of
    ///    [`STORED_GATE_UP_ROWS`].
    /// 3. `d = round(W2 * act)`.
    pub fn run(&self, z: &[f32], activation: CappedActivation) -> Result<Vec<f32>> {
        if z.len() != self.hidden || z.iter().any(|&v| !v.is_finite() || round_bf16(v).to_bits() != v.to_bits()) {
            return Err(invalid("expert input must be hidden finite BF16 values"));
        }
        let gate_up = rounded(rows(&self.gate_up, z), "gate-and-up product")?;
        let act: Vec<f32> = (0..self.width).map(|i| activation.apply(
            gate_up[STORED_GATE_UP_ROWS.gate(i, self.width)], gate_up[STORED_GATE_UP_ROWS.up(i, self.width)])).collect();
        let act = rounded(act, "expert activation")?;
        rounded(rows(&self.down, &act), "down product")
    }
}

/// Fold the expert outputs of one token: `round(((0 + w0 * d0) + w1 * d1) + ...)`.
/// The terms are added in the order given, which is the pick order of the
/// router. Each product is rounded to FP32 before its add. One BF16 rounding.
pub fn fold(weights: &[f32], outputs: &[Vec<f32>]) -> Result<Vec<f32>> {
    let width = outputs.first().map_or(0, Vec::len);
    if width == 0 || weights.len() != outputs.len() || outputs.iter().any(|o| o.len() != width)
        || weights.iter().chain(outputs.iter().flatten()).any(|v| !v.is_finite()) {
        return Err(invalid("fold needs one finite weight for each expert output of equal width"));
    }
    let mut sum = vec![0.0f32; width];
    for (&weight, output) in weights.iter().zip(outputs) {
        for (total, &value) in sum.iter_mut().zip(output) { *total += value * weight; }
    }
    rounded(sum, "expert fold")
}

/// The whole expert block for one token. `z` is the BF16 input of the block.
/// `router_weight` is `[experts, hidden]` FP32. `expert` gives the numbers of
/// one expert by its number. Only picked experts are asked for.
pub fn moe_token(routing: &Routing, activation: CappedActivation, router_weight: &[f32], bias: &[f32], z: &[f32],
    mut expert: impl FnMut(usize) -> Result<Expert>) -> Result<(Route, Vec<f32>)> {
    let route = routing.route(&routing.logits(router_weight, z)?, bias)?;
    let mut outputs = Vec::with_capacity(route.ids.len());
    for &id in &route.ids { outputs.push(expert(id)?.run(z, activation)?); }
    let output = fold(&route.weights, &outputs)?;
    Ok((route, output))
}

/// One rising FP32 chain a row, from 0.0. Each product is rounded before its add.
fn rows(matrix: &[f32], x: &[f32]) -> Vec<f32> {
    matrix.chunks_exact(x.len()).map(|row| { let mut sum = 0.0f32; for (&w, &v) in row.iter().zip(x) { sum += w * v; } sum }).collect()
}
fn rounded(mut values: Vec<f32>, what: &str) -> Result<Vec<f32>> {
    for value in &mut values { *value = round_bf16(*value); }
    if values.iter().any(|v| !v.is_finite()) { return Err(invalid(format!("{what} is not finite in BF16"))); }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_widen::Stored;
    fn bits(values: &[f32]) -> Vec<u16> { values.iter().map(|&v| (round_bf16(v).to_bits() >> 16) as u16).collect() }
    fn activation() -> CappedActivation { CappedActivation::new(4.0, 25.0).unwrap() }
    /// The same formula in FP64, for hand values.
    fn act64(gate: f64, up: f64) -> f64 { (4.0 * (gate / 4.0).tanh()) * (1.0 / (1.0 + (-gate).exp())) * (25.0 * (up / 25.0).tanh()) }
    /// Within 1 part in 64: room for the three BF16 roundings of an expert and the one of the fold.
    fn near(a: f32, b: f64) -> bool { ((a as f64) - b).abs() <= b.abs() / 64.0 }

    #[test]
    fn tie_for_place_8_takes_the_lower_expert_number() {
        // 10 experts, pick 8. All logits 0, so all scores are 0.5.
        // Experts 3, 5, 7 and 8 have bias 0 and tie for the last places. The other 6 have bias 1.
        // Places 7 and 8 go to experts 3 and 5. Experts 7 and 8 are not picked.
        let routing = Routing::new(10, 8, 8.0).unwrap();
        let bias = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let route = routing.route(&[0.0; 10], &bias).unwrap();
        assert_eq!(route.ids, [0, 1, 2, 4, 6, 9, 3, 5]);
        // Sum of the 8 scores is 4. Each weight is 0.5 / 4 * 8 = 1. The 8 weights add up to 8.
        assert_eq!(route.weights, [1.0; 8]);
        for _ in 0..3 { assert_eq!(routing.route(&[0.0; 10], &bias).unwrap(), route); }
        // Signed zeros are equal values: the lower number still comes first.
        let zeros = Routing::new(3, 2, 1.0).unwrap().route(&[0.0; 3], &[-0.0, 0.0, -0.0]).unwrap();
        assert_eq!(zeros.ids, [0, 1]);
    }
    #[test]
    fn bias_picks_but_does_not_weigh() {
        // Expert 0: logit 0, score 0.5, bias 0.25. Expert 2: logit 100, score 1.0, bias -0.25.
        // Both have score + bias = 0.75: a tie of two different scores. Expert 0 comes first.
        // Expert 1: score 0.5, bias 0.5: 1.0, the first place. Expert 3: score 1.0, bias -2: out.
        let routing = Routing::new(4, 3, 8.0).unwrap();
        let route = routing.route(&[0.0, 0.0, 100.0, 100.0], &[0.25, 0.5, -0.25, -2.0]).unwrap();
        assert_eq!(route.ids, [1, 0, 2]);
        // Weights use the scores without the bias: [0.5, 0.5, 1.0] / 2.0 * 8.
        assert_eq!(route.weights, [2.0, 2.0, 4.0]);
        // Without the bias the pick would be the two scores of 1.0 first.
        assert_eq!(routing.route(&[0.0, 0.0, 100.0, 100.0], &[0.0; 4]).unwrap().ids, [2, 3, 0]);
    }
    #[test]
    fn weight_sum_follows_the_pick_order_and_keeps_the_epsilon() {
        // One picked expert: weight = score / (score + 1e-20) * scale. For a score near 1e-22 the epsilon decides.
        let tiny = Routing::new(2, 1, 1.0).unwrap().route(&[-50.0, -60.0], &[0.0; 2]).unwrap();
        assert_eq!(tiny.ids, [0]);
        let score = sigmoid(-50.0);
        assert_eq!(tiny.weights, [score / (score + ROUTE_EPSILON)]);
        assert!(tiny.weights[0] > 0.015 && tiny.weights[0] < 0.025);
        // Divide first, then scale: 0.75 / 1.25 * 8 = 4.8 and 0.5 / 1.25 * 8 = 3.2.
        let route = Routing::new(2, 2, 8.0).unwrap().route(&[3.0f32.ln(), 0.0], &[0.0; 2]).unwrap();
        assert!((route.weights[0] - 4.8).abs() < 1e-5 && (route.weights[1] - 3.2).abs() < 1e-5);
        let (a, b) = (sigmoid(3.0f32.ln()), 0.5f32);
        assert_eq!(route.weights, [a / ((0.0 + a) + b + ROUTE_EPSILON) * 8.0, b / ((0.0 + a) + b + ROUTE_EPSILON) * 8.0]);
    }
    #[test]
    fn router_logits_are_rising_fp32_rows() {
        let routing = Routing::new(2, 1, 8.0).unwrap();
        // Row 0: 1*2 + 2*3 + 3*4 = 20. Row 1: (2^24 * 1 + 1 * 1) - 2^24 * 1 = 0 in a rising FP32 chain.
        let weight = [1.0, 2.0, 3.0, 16_777_216.0, 1.0, -16_777_216.0];
        assert_eq!(routing.logits(&weight, &[2.0, 3.0, 4.0]).unwrap(), [20.0, 16_777_216.0 * 2.0 + 3.0 - 16_777_216.0 * 4.0]);
        assert_eq!(routing.logits(&weight, &[1.0, 1.0, 1.0]).unwrap(), [6.0, 0.0]);
        assert!(routing.logits(&weight, &[1.0, 1.0]).is_err());
    }
    #[test]
    fn activation_hand_values() {
        // gate 0 gives 0: tanh(0) = 0. The sigmoid reads the raw gate, not the capped one.
        assert_eq!(activation().apply(0.0, 7.0), 0.0);
        // gate 3, up 7: 4 tanh(0.75) = 2.540596, sigmoid(3) = 0.952574, 25 tanh(0.28) = 6.822627. Product 16.51148.
        assert!((act64(3.0, 7.0) - 16.51148).abs() < 1e-4);
        assert!(((activation().apply(3.0, 7.0) as f64) - act64(3.0, 7.0)).abs() < 1e-4);
        // Far above the caps: 4 * 1 * 25 = 100. Far below on the gate: the sigmoid gives 0.
        assert_eq!(activation().apply(1000.0, 1000.0), 100.0);
        assert_eq!(activation().apply(1000.0, -1000.0), -100.0);
        assert!(activation().apply(-1000.0, 1000.0).abs() < 1e-30);
        // A capped sigmoid would give sigmoid(4) = 0.982 here. The raw gate gives 1.0.
        assert_eq!(activation().apply(100.0, 2500.0), 100.0);
        assert!(CappedActivation::new(0.0, 25.0).is_err() && CappedActivation::new(4.0, f64::NAN).is_err());
    }
    #[test]
    fn gate_and_up_rows_follow_the_named_order() {
        assert_eq!(STORED_GATE_UP_ROWS, GateUpRows::Interleaved);
        assert_eq!((GateUpRows::Interleaved.gate(1, 2), GateUpRows::Interleaved.up(1, 2)), (2, 3));
        assert_eq!((GateUpRows::Halves.gate(1, 2), GateUpRows::Halves.up(1, 2)), (1, 3));
        // hidden 2, width 2, z = [1, 2]. Stored rows give gu = [0, 5, 3, 7].
        // Interleaved: gate = [0, 3], up = [5, 7]. So act = [0, f(3, 7)] = [0, 16.5].
        // The other order would read gate = [0, 5], up = [3, 7] and give f(5, 7) = 23.0.
        let gate_up = bits(&[0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 3.0]);
        let down = bits(&[1.0, 1.0, 0.0, 2.0]);
        let expert = Expert::from_bf16(2, 2, &gate_up, &down).unwrap();
        let d = expert.run(&[1.0, 2.0], activation()).unwrap();
        let act = round_bf16(activation().apply(3.0, 7.0));
        assert_eq!(act, 16.5);
        // Down rows [1, 1] and [0, 2]: d = [0 + act, 2 * act].
        assert_eq!(d, [16.5, 33.0]);
        assert!(near(d[0], act64(3.0, 7.0)) && !near(d[0], act64(5.0, 7.0)));
    }
    #[test]
    fn expert_rounds_to_bf16_at_each_of_its_three_steps() {
        // hidden 1, width 1. gate row [1 + 1/128], up row [1], z = 1 + 1/128.
        // gu[0] = (1 + 1/128)^2 = 1 + 2/128 + 1/16384, which rounds to 1 + 2/128 in BF16.
        let w = 1.0 + 1.0 / 128.0f32;
        let expert = Expert::from_bf16(1, 1, &bits(&[w, 1.0]), &bits(&[w])).unwrap();
        let gate = round_bf16(w * w);
        assert_eq!(gate, 1.0 + 2.0 / 128.0);
        let act = round_bf16(activation().apply(gate, w));
        assert_eq!(expert.run(&[w], activation()).unwrap(), [round_bf16(w * act)]);
        // An input that is not a BF16 value is refused.
        assert!(expert.run(&[1.0 + 1.0 / 1024.0], activation()).is_err());
        assert!(expert.run(&[1.0, 1.0], activation()).is_err());
        assert!(expert.run(&[f32::NAN], activation()).is_err());
    }
    #[test]
    fn fold_adds_in_pick_order_in_fp32_and_rounds_once() {
        // 2^24 + 1 is not an FP32 value. Pick order A, B, C gives (2^24 + 1) - 2^24 = 0.
        // Pick order A, C, B gives (2^24 - 2^24) + 1 = 1. The fold does not sort.
        let (a, b, c) = (vec![16_777_216.0f32], vec![1.0f32], vec![-16_777_216.0f32]);
        assert_eq!(fold(&[1.0; 3], &[a.clone(), b.clone(), c.clone()]).unwrap(), [0.0]);
        assert_eq!(fold(&[1.0; 3], &[a, c, b]).unwrap(), [1.0]);
        // 256 + 1 - 256 = 1 in FP32. A BF16 rounding after each add would give 0.
        assert_eq!(fold(&[1.0; 3], &[vec![256.0], vec![1.0], vec![-256.0]]).unwrap(), [1.0]);
        // The weight stays FP32: 1.5 * 1.003 rounds once at the end.
        let weight = 1.003f32;
        assert_eq!(fold(&[weight], &[vec![1.5]]).unwrap(), [round_bf16(1.5 * weight)]);
        assert_ne!(fold(&[weight], &[vec![1.5]]).unwrap(), [round_bf16(1.5 * round_bf16(weight))]);
        // Two columns fold apart: [1*2 + 3*4, 2*2 + 5*4].
        assert_eq!(fold(&[2.0, 4.0], &[vec![1.0, 2.0], vec![3.0, 5.0]]).unwrap(), [14.0, 24.0]);
        assert!(fold(&[1.0], &[vec![1.0], vec![2.0]]).is_err());
        assert!(fold(&[1.0, 1.0], &[vec![1.0], vec![2.0, 3.0]]).is_err());
        assert!(fold(&[], &[]).is_err());
        assert!(fold(&[f32::NAN], &[vec![1.0]]).is_err());
    }
    #[test]
    fn whole_block_reads_stored_experts_through_the_seam() {
        // hidden 16, width 16, 3 experts, pick 2. Packed 4-bit experts: every code is 2 (+1.0),
        // block scale 1.0. Expert e has amax e + 1 and denominator 16, so each weight is (e + 1) / 16.
        let (hidden, width) = (16usize, 16usize);
        let (codes13, scales13, codes2, scales2) = (vec![0x22u8; 32 * 8], vec![0x38u8; 32], vec![0x22u8; 16 * 8], vec![0x38u8; 16]);
        let load = |e: usize| {
            let amax = (e + 1) as f32;
            Expert::from_stored(hidden, width,
                &StoredMatrix { rows: 32, columns: 16, data: Stored::Packed4 { codes: &codes13, scales: &scales13, amax, denominator: 16.0 } },
                &StoredMatrix { rows: 16, columns: 16, data: Stored::Packed4 { codes: &codes2, scales: &scales2, amax, denominator: 16.0 } })
        };
        // z = 16 values of 0.5: sum 8. Expert e: every gate and up output is 8 * (e + 1) / 16 = (e + 1) / 2.
        let z = [0.5f32; 16];
        let routing = Routing::new(3, 2, 8.0).unwrap();
        // Router rows: expert 0 gets logit 0, expert 1 gets 8 * 0 = 0, expert 2 gets 0. Bias picks 2 then 0.
        let router_weight = [0.0f32; 48];
        let mut asked = Vec::new();
        let (route, output) = moe_token(&routing, activation(), &router_weight, &[0.5, -1.0, 1.0], &z, |e| { asked.push(e); load(e) }).unwrap();
        assert_eq!((route.ids.clone(), asked), (vec![2, 0], vec![2, 0]));
        assert_eq!(route.weights, [4.0, 4.0]);
        // Expert 2: gate = up = 1.5. act = round(f(1.5, 1.5)). d = round(16 * act * 3/16) for each of 16 outputs.
        // Expert 0: gate = up = 0.5. act = round(f(0.5, 0.5)). d = round(16 * act * 1/16).
        let d = |e: usize, g: f32| { let act = round_bf16(activation().apply(g, g)); let w = (e + 1) as f32 / 16.0;
            let mut sum = 0.0f32; for _ in 0..16 { sum += w * act; } round_bf16(sum) };
        let expected = round_bf16((0.0 + d(2, 1.5) * 4.0) + d(0, 0.5) * 4.0);
        assert_eq!(output, [expected; 16]);
        // Hand value: f(1.5, 1.5) = 1.7558, times 3 = 5.267. f(0.5, 0.5) = 0.1548. Sum times 4 = 21.69.
        assert!(near(expected, 4.0 * (3.0 * act64(1.5, 1.5) + act64(0.5, 0.5))));
        assert!((act64(1.5, 1.5) - 1.7558).abs() < 1e-4 && (act64(0.5, 0.5) - 0.1548).abs() < 1e-4);
        // An 8-bit float expert goes through the same seam: hidden 32, width 32, every weight 1/32.
        let (v13, s13, v2, s2) = (vec![0x38u8; 64 * 32], vec![122u8; 64], vec![0x38u8; 32 * 32], vec![122u8; 32]);
        let wide = Expert::from_stored(32, 32,
            &StoredMatrix { rows: 64, columns: 32, data: Stored::Float8 { values: &v13, scales: &s13 } },
            &StoredMatrix { rows: 32, columns: 32, data: Stored::Float8 { values: &v2, scales: &s2 } }).unwrap();
        // z = 32 values of 2: every gate and up output is 2. Every down output is act = round(f(2, 2)).
        assert_eq!(wide.run(&[2.0; 32], activation()).unwrap(), [round_bf16(activation().apply(2.0, 2.0)); 32]);
        // A shape that does not match is refused before any decode.
        assert!(Expert::from_stored(16, 8,
            &StoredMatrix { rows: 32, columns: 16, data: Stored::Packed4 { codes: &codes13, scales: &scales13, amax: 1.0, denominator: 16.0 } },
            &StoredMatrix { rows: 16, columns: 16, data: Stored::Packed4 { codes: &codes2, scales: &scales2, amax: 1.0, denominator: 16.0 } }).is_err());
    }
    #[test]
    fn bad_router_inputs_are_refused() {
        assert!(Routing::new(0, 1, 8.0).is_err() && Routing::new(4, 0, 8.0).is_err() && Routing::new(4, 5, 8.0).is_err());
        assert!(Routing::new(1025, 8, 8.0).is_err() && Routing::new(4, 2, 0.0).is_err() && Routing::new(4, 2, f64::INFINITY).is_err());
        let routing = Routing::new(4, 2, 8.0).unwrap();
        assert!(routing.route(&[0.0; 3], &[0.0; 4]).is_err() && routing.route(&[0.0; 4], &[0.0; 3]).is_err());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(routing.route(&[bad, 0.0, 0.0, 0.0], &[0.0; 4]).is_err());
            assert!(routing.route(&[0.0; 4], &[0.0, bad, 0.0, 0.0]).is_err());
        }
        assert!(routing.route(&[0.0; 4], &[f32::MAX; 4]).is_ok());
    }
}
