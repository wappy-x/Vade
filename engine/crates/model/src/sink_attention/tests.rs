use super::*;
use crate::short_conv::{FixedTaps, TokenTaps};

fn bits(values: &[f32]) -> Vec<u32> { values.iter().map(|v| v.to_bits()).collect() }
fn same_bits(actual: &[f32], expected: &[f32]) { assert_eq!(bits(actual), bits(expected)); }
fn message<T: std::fmt::Debug>(result: Result<T>) -> String { result.unwrap_err().to_string() }

/// Owned weights of one layer. `k_taps` and `v_taps` hold fixed taps, or
/// static taps followed by the tap projection when `token_taps` is true.
#[derive(Clone)]
struct Layer {
    config: Config, token_taps: bool,
    q: Vec<f32>, q_scale: Vec<f32>, k: Vec<f32>, v: Vec<f32>, k_taps: (Vec<f32>, Vec<f32>), v_taps: (Vec<f32>, Vec<f32>),
    rel: Vec<f32>, rel_scale: Vec<f32>, table: Vec<f32>, sink: Vec<f32>, gate: Vec<f32>, o: Vec<f32>,
}
impl Layer {
    fn taps<'a>(&self, data: &'a (Vec<f32>, Vec<f32>)) -> Taps<'a> {
        let d = self.config;
        if self.token_taps { Taps::TokenMade(TokenTaps::new(d.kv_width(), d.num_kv_heads, d.conv_taps, d.hidden_size, &data.0, &data.1).unwrap()) }
        else { Taps::Fixed(FixedTaps::new(d.kv_width(), d.conv_taps, &data.0).unwrap()) }
    }
    fn weights(&self) -> Weights<'_> {
        Weights { q_proj: &self.q, q_scale: &self.q_scale, k_proj: &self.k, v_proj: &self.v,
            k_conv: self.taps(&self.k_taps), v_conv: self.taps(&self.v_taps), rel_proj: &self.rel, rel_scale: &self.rel_scale,
            rel_table: &self.table, sink_proj: &self.sink, gate_proj: &self.gate, o_proj: &self.o }
    }
    fn reference(&self) -> Reference<'_> { Reference::new(self.config, self.weights()).unwrap() }
    fn states(&self) -> (State, VisitHistories) { (State::new(self.config).unwrap(), self.config.histories().unwrap()) }
    /// Run every token from a new state and return the stages of each token.
    fn run(&self, xq: &[Vec<f32>], xk: &[Vec<f32>], xv: &[Vec<f32>]) -> Vec<Parts> {
        let (reference, (mut state, mut conv)) = (self.reference(), self.states());
        (0..xq.len()).map(|t| reference.step_parts(&mut state, &mut conv, &xq[t], &xk[t], &xv[t]).unwrap()).collect()
    }
}

// The hand layer: hidden 12, 2 query heads, 1 key and value head, head size 2,
// relative width 1. The projections pick input elements: query input
// [0..4] is the query, [4..6] the relative state, [6..8] the sink and [8..12]
// the gate. Key and value inputs [0..2] are the key and the value. The 3 fixed
// taps are zero, so a convolution gives its input times 1 / sqrt(4) = 0.5.
const HAND: usize = 12;
fn pick(rows: usize, cols: usize, ones: &[(usize, usize)]) -> Vec<f32> {
    let mut m = vec![0.0; rows * cols];
    for &(r, c) in ones { m[r * cols + c] = 1.0; }
    m
}
fn hand(reach: Reach, table: &[f32], qk_clip: f32, q_scale: [f32; 4]) -> Layer {
    let config = Config { hidden_size: HAND, num_heads: 2, num_kv_heads: 1, head_dim: 2, rel_dim: 1, conv_taps: 3,
        qk_clip, rel_clip: 8.0, rms_norm_eps: 1e-6, reach };
    let diagonal: Vec<(usize, usize)> = (0..4).map(|i| (i, i)).collect();
    Layer { config, token_taps: false, q: pick(4, HAND, &diagonal), q_scale: q_scale.to_vec(),
        k: pick(2, HAND, &[(0, 0), (1, 1)]), v: pick(2, HAND, &[(0, 0), (1, 1)]),
        k_taps: (vec![0.0; 6], Vec::new()), v_taps: (vec![0.0; 6], Vec::new()),
        rel: pick(2, HAND, &[(0, 4), (1, 5)]), rel_scale: vec![1.0; 2], table: table.to_vec(),
        sink: pick(2, HAND, &[(0, 6), (1, 7)]), gate: pick(4, HAND, &[(0, 8), (1, 9), (2, 10), (3, 11)]),
        o: pick(HAND, 4, &diagonal) }
}
fn window3(table: &[f32]) -> Layer { hand(Reach::Window { window: 3 }, table, 32.0, [1.0; 4]) }
fn xq(q: [f32; 4], rel: [f32; 2], sink: [f32; 2], gate: [f32; 4]) -> Vec<f32> { [&q[..], &rel[..], &sink[..], &gate[..]].concat() }
fn xkv(a: f32, b: f32) -> Vec<f32> { let mut x = vec![0.0; HAND]; x[0] = a; x[1] = b; x }
const OPEN: [f32; 4] = [40.0; 4]; // sigmoid(40) is exactly 1 in f32.
// Raw values (4, 0), (2, 2), (2, -2), (-2, 2), (0, 4) give the finished values
// (1.4140625, 0), (1, 1), (1, -1), (-1, 1), (0, 1.4140625): times 0.5, then the plain norm of 2 numbers.
const RAW_VALUES: [(f32, f32); 5] = [(4.0, 0.0), (2.0, 2.0), (2.0, -2.0), (-2.0, 2.0), (0.0, 4.0)];
fn raw_values(count: usize) -> Vec<Vec<f32>> { RAW_VALUES[..count].iter().map(|&(a, b)| xkv(a, b)).collect() }

#[test]
fn a_sequence_start_has_hand_values() {
    let layer = window3(&[0.0; 3]);
    // Head 0: q (1, 0) with key (1, 2): 1 * f32(1 / sqrt(2)) = 0.70710677, BF16 0.70703125.
    // The sink has the same score, so the weight is 0.5. Head 1: q (0, 2): 4 * 0.70710677 =
    // 2.828427, BF16 2.828125. Its sink is far below, so the weight is 1. Gate logits 0 give 0.5.
    let parts = layer.run(&[xq([1.0, 0.0, 0.0, 2.0], [0.0; 2], [0.70703125, -100.0], [0.0; 4])], &[xkv(2.0, 4.0)], &[xkv(2.0, 2.0)]);
    let p = &parts[0];
    same_bits(&p.query, &[1.0, 0.0, 0.0, 2.0]);
    same_bits(&p.key, &[1.0, 2.0]);
    same_bits(&p.value, &[1.0, 1.0]);
    same_bits(&p.scores, &[0.70703125, 2.828125]);
    same_bits(&p.weights, &[0.5, 1.0]);
    same_bits(&p.mixed, &[0.5, 0.5, 1.0, 1.0]);
    same_bits(&p.gated, &[0.25, 0.25, 0.5, 0.5]);
    same_bits(&p.output, &[0.25, 0.25, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
#[allow(clippy::excessive_precision)] // The literals are exact BF16 values, written with every decimal digit.
fn the_window_edge_drops_the_oldest_token() {
    let layer = window3(&[0.0; 3]);
    let (reference, (mut state, mut conv)) = (layer.reference(), layer.states());
    // All scores and the sink are 0. Each visible key has the weight 1 / (keys + 1).
    // Token 3 sees tokens 1, 2, 3 and not token 0: 0.25 * ((1, 1) + (1, -1) + (-1, 1)).
    // With token 0 still visible the weights would be 0.2 and the result would differ.
    let expected: [[f32; 2]; 5] = [[0.70703125, 0.0], [0.8046875, 0.333984375], [0.8515625, 0.0], [0.25, 0.25], [0.0, 0.353515625]];
    let weights: [&[f32]; 5] = [&[0.5], &[0.333984375; 2], &[0.25; 3], &[0.25; 3], &[0.25; 3]];
    let query = xq([0.0; 4], [0.0; 2], [0.0; 2], OPEN);
    for (t, value) in raw_values(5).iter().enumerate() {
        let p = reference.step_parts(&mut state, &mut conv, &query, &xkv(0.0, 0.0), value).unwrap();
        same_bits(&p.weights, &[weights[t], weights[t]].concat());
        same_bits(&p.output[..4], &[expected[t], expected[t]].concat());
        // The state keeps only the tokens that the next token can see.
        assert_eq!((state.position(), state.len()), (t + 1, (t + 1).min(2)));
    }
    same_bits(state.values(), &[-1.0, 1.0, 0.0, 1.4140625]);
}

#[test]
fn a_distance_outside_the_table_has_no_term() {
    // Full reach with a table of 2 columns, both -8. Head 0 has the relative state 8, so the
    // keys at distance 0 and 1 get -64. A key at distance 2 or more gets no term and wins.
    // Head 1 has the relative state 0. Positions are far under the floor, so the scale is 1.
    let layer = hand(Reach::Full { rel_extent: 2, log_alpha: 0.1, log_floor: 8192.0 }, &[-8.0, -8.0], 32.0, [1.0; 4]);
    let query = xq([0.0; 4], [100.0, 0.0], [-100.0; 2], OPEN);
    let parts = layer.run(&vec![query; 4], &vec![xkv(0.0, 0.0); 4], &raw_values(4));
    same_bits(&parts[3].relative, &[8.0, 0.0]);
    same_bits(&parts[2].scores, &[0.0, -64.0, -64.0, 0.0, 0.0, 0.0]);
    same_bits(&parts[3].scores, &[0.0, 0.0, -64.0, -64.0, 0.0, 0.0, 0.0, 0.0]);
    same_bits(&parts[3].weights[..2], &[0.5, 0.5]);
    assert!(parts[3].weights[2] < 1e-27 && parts[3].weights[3] < 1e-27);
    // Head 0: token 0 alone, both tokens, token 0 alone, tokens 0 and 1. Head 1: the mean of all.
    same_bits(&parts[0].mixed, &[1.4140625, 0.0, 1.4140625, 0.0]);
    same_bits(&parts[1].mixed, &[1.203125, 0.5, 1.203125, 0.5]);
    same_bits(&parts[2].mixed, &[1.4140625, 0.0, 1.140625, 0.0]);
    same_bits(&parts[3].mixed, &[1.203125, 0.5, 0.6015625, 0.25]);
    // Full reach keeps every token.
    let (reference, (mut state, mut conv)) = (layer.reference(), layer.states());
    reference.scan(&mut state, &mut conv, &[0.0; 4 * HAND], &[0.0; 4 * HAND], &raw_values(4).concat()).unwrap();
    assert_eq!((state.position(), state.len()), (4, 4));
}

#[test]
fn the_relative_term_reads_the_column_of_the_distance() {
    // Table [1, 2, 4] and relative states 1 and -0.5. With q = 0 a score is the term alone.
    // The oldest visible key comes first. Token 3 sees the distances 2, 1, 0 again.
    let layer = window3(&[1.0, 2.0, 4.0]);
    let query = xq([0.0; 4], [1.0, -0.5], [-100.0; 2], [0.0; 4]);
    let parts = layer.run(&vec![query; 4], &vec![xkv(0.0, 0.0); 4], &raw_values(4));
    same_bits(&parts[0].scores, &[1.0, -0.5]);
    same_bits(&parts[1].scores, &[2.0, 1.0, -1.0, -0.5]);
    same_bits(&parts[2].scores, &[4.0, 2.0, 1.0, -2.0, -1.0, -0.5]);
    same_bits(&parts[3].scores, &[4.0, 2.0, 1.0, -2.0, -1.0, -0.5]);
}

#[test]
fn the_position_scale_follows_the_token_position() {
    let reach = Reach::Full { rel_extent: 2, log_alpha: 0.5, log_floor: 2.0 };
    // 1 + 0.5 * ln(max((position + 1) / 2, 1)): 1, 1, 1.2027326, 1.3465736, and 1.6931472 at position 7.
    assert_eq!([0, 1, 2, 3, 7].map(|p| reach.position_scale(p).unwrap().to_bits()), [0x3f80_0000, 0x3f80_0000, 0x3f99_f324, 0x3fac_5c86, 0x3fd8_b90c]);
    assert_eq!(Reach::Window { window: 3 }.position_scale(9), None);
    // Alpha 0.1 and floor 8192: exactly 1 up to position 8191, then a slow rise.
    let wide = Reach::Full { rel_extent: 2, log_alpha: 0.1, log_floor: 8192.0 };
    assert_eq!([8191, 8192, 1_048_575].map(|p| wide.position_scale(p).unwrap().to_bits()), [0x3f80_0000, 0x3f80_0066, 0x3fbe_1b22]);
    // A floor that is not a power of two shows the form of the division: a product with f32(1 / 3).
    // Position 13: 14 * 0.33333334, and the scale is 2.540445. A division by 3 gives one f32 step less.
    let third = Reach::Full { rel_extent: 2, log_alpha: 1.0, log_floor: 3.0 };
    assert_eq!([2, 3, 13].map(|p| third.position_scale(p).unwrap().to_bits()), [0x3f80_0000, 0x3fa4_d2c4, 0x4022_96a7]);

    // Query (1, 0) and (3, 4), relative states 2 and 100 (clipped to 8). Each is rounded to BF16
    // after the clip, multiplied by the scale in f32 and rounded again.
    let layer = hand(reach, &[0.0, 0.0], 32.0, [1.0; 4]);
    let query = xq([1.0, 0.0, 3.0, 4.0], [2.0, 100.0], [-100.0; 2], [0.0; 4]);
    let parts = layer.run(&vec![query; 4], &vec![xkv(0.0, 0.0); 4], &vec![xkv(2.0, 2.0); 4]);
    same_bits(&parts[1].query, &[1.0, 0.0, 3.0, 4.0]);
    same_bits(&parts[1].relative, &[2.0, 8.0]);
    same_bits(&parts[2].query, &[1.203125, 0.0, 3.609375, 4.8125]);
    same_bits(&parts[2].relative, &[2.40625, 9.625]);
    same_bits(&parts[3].query, &[1.34375, 0.0, 4.03125, 5.375]);
    same_bits(&parts[3].relative, &[2.6875, 10.75]);
}

#[test]
fn the_learned_scale_comes_before_the_clip() {
    // Threshold 2.5. Head 0: (3, 4) has the length 5 and becomes (1.5, 2). Head 1: the scale
    // (2, 0.5) makes (6, 2) with the length 6.3245554: factor 0.39528465, BF16 (2.375, 0.7890625).
    // The key (3, 4) has no learned scale and becomes (1.5, 2).
    let layer = hand(Reach::Window { window: 3 }, &[0.0; 3], 2.5, [1.0, 1.0, 2.0, 0.5]);
    let parts = layer.run(&[xq([3.0, 4.0, 3.0, 4.0], [0.0; 2], [0.0; 2], [0.0; 4])], &[xkv(6.0, 8.0)], &[xkv(2.0, 2.0)]);
    same_bits(&parts[0].query, &[1.5, 2.0, 2.375, 0.7890625]);
    same_bits(&parts[0].key, &[1.5, 2.0]);
    // A vector under the threshold does not change.
    same_bits(&length_clip(&[0.25, -0.5], 2.5).unwrap(), &[0.25, -0.5]);
    assert_eq!(CLIP_EPSILON, 1e-6);
}

#[test]
fn a_number_over_a_tensor_value_is_two_rounded_steps() {
    // T = 22.627417. T * f32(1 / 5) = 4.5254836 and T * f32(1 / 45) = 0.5028315: one f32 step above
    // the one division T / 5 and T / 45. With 30 the two forms give the same value.
    let t = f32::from_bits(0x41b5_04f3);
    assert_eq!([5.0f32, 45.0, 30.0].map(|v| number_over(t, v).to_bits()), [0x4090_d0c3, 0x3f00_b991, 0x3f41_1659]);
    assert_eq!([(t / 5.0).to_bits(), (t / 45.0).to_bits()], [0x4090_d0c2, 0x3f00_b990]);
    // The f32 results of a clip, before the BF16 round. (27, 36) has the length 45. One division
    // would give 0x4159_3923 and 0x4190_d0c2. (3, 4) with the threshold 2.5 has the length 5.000001.
    let bits = |x: &[f32], threshold: f32| length_clip(x, threshold).unwrap().iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
    assert_eq!(bits(&[27.0, 36.0], t), [0x4159_3925, 0x4190_d0c3]);
    assert_eq!(bits(&[3.0, 4.0], 2.5), [0x3fbf_fffd, 0x3fff_fffc]);
}

#[test]
fn a_zero_gate_halves_and_a_closed_gate_removes() {
    // Gate logits 0, -100, 40, 0 on the head outputs (1, 1, 1, 1): sigmoid gives 0.5, about 1.6e-38, 1, 0.5.
    let layer = window3(&[0.0; 3]);
    let parts = layer.run(&[xq([0.0; 4], [0.0; 2], [-100.0; 2], [0.0, -100.0, 40.0, 0.0])], &[xkv(0.0, 0.0)], &[xkv(2.0, 2.0)]);
    same_bits(&parts[0].mixed, &[1.0; 4]);
    assert_eq!(bits(&parts[0].gated), [0x3f00_0000, 0x00b3_0000, 0x3f80_0000, 0x3f00_0000]);
    same_bits(&parts[0].output[..4], &parts[0].gated);
    assert!(parts[0].gated[1] > 0.0 && parts[0].gated[1] < 1e-37);
    // A gate projection of zeros is a zero gate for every element: the output is half of the head outputs.
    let mut zero = layer.clone();
    zero.gate = vec![0.0; 4 * HAND];
    let parts = zero.run(&[xq([0.0; 4], [0.0; 2], [-100.0; 2], OPEN)], &[xkv(0.0, 0.0)], &[xkv(2.0, 2.0)]);
    same_bits(&parts[0].output[..4], &[0.5; 4]);
}

#[test]
fn a_score_is_rounded_three_times() {
    // q (1, 1), key (1.3515625, 1), relative term 1.0078125 (state 1 times the table value).
    // Dot 2.3515625, BF16 2.34375. Times 0.70710677 = 1.6572815, BF16 1.65625.
    // Plus 1.0078125 = 2.6640625, BF16 2.65625. With one rounding fewer at any place
    // the result is 2.671875.
    let layer = window3(&[1.0078125, 0.0, 0.0]);
    let parts = layer.run(&[xq([1.0, 1.0, 0.0, 0.0], [1.0, 0.0], [0.0; 2], [0.0; 4])], &[xkv(2.703125, 2.0)], &[xkv(2.0, 2.0)]);
    same_bits(&parts[0].key, &[1.3515625, 1.0]);
    same_bits(&parts[0].scores, &[2.65625, 0.0]);
    assert_eq!(score_scale(2).to_bits(), 0x3f35_04f3);
    assert_eq!(score_scale(4), 0.5);
}

#[test]
#[allow(clippy::excessive_precision)] // The literals are exact BF16 values, written with every decimal digit.
fn weights_are_rounded_before_they_meet_the_values() {
    // 4 keys and the sink have equal scores: each weight is 0.2 in f32 and 0.2001953125 in BF16.
    // The finished values are (1.1796875, 0.78515625), (1.390625, -0.27734375), (0.5234375, 1.3125)
    // and (-1, 1). BF16 weights give (0.419921875, 0.56640625). f32 weights would give
    // (0.41796875, 0.5625).
    let layer = hand(Reach::Window { window: 4 }, &[0.0; 4], 32.0, [1.0; 4]);
    let values: Vec<Vec<f32>> = [(6.0, 4.0), (5.0, -1.0), (2.0, 5.0), (-1.0, 1.0)].iter().map(|&(a, b)| xkv(a, b)).collect();
    let parts = layer.run(&vec![xq([0.0; 4], [0.0; 2], [0.0; 2], OPEN); 4], &vec![xkv(0.0, 0.0); 4], &values);
    same_bits(&parts[2].value, &[0.5234375, 1.3125]);
    same_bits(&parts[3].weights, &[0.2001953125; 8]);
    same_bits(&parts[3].mixed, &[0.419921875, 0.56640625, 0.419921875, 0.56640625]);
}

#[test]
fn query_heads_share_key_and_value_heads() {
    assert_eq!([0, 1, 2, 3, 4, 5, 23].map(|h| shared_head(h, 24, 8)), [0, 0, 0, 1, 1, 1, 7]);
    // 4 query heads on 2 key and value heads, head size 1, hidden 8. Query heads 0 and 1 read
    // head 0. Query heads 2 and 3 read head 1. Keys are 3 and 5 (6 and 10 times 0.5). A value
    // of 1 number has the plain norm 1 with its sign. The scale for head size 1 is 1.
    let config = Config { hidden_size: 8, num_heads: 4, num_kv_heads: 2, head_dim: 1, rel_dim: 1, conv_taps: 3,
        qk_clip: 32.0, rel_clip: 8.0, rms_norm_eps: 1e-6, reach: Reach::Window { window: 2 } };
    let diagonal: Vec<(usize, usize)> = (0..4).map(|i| (i, i)).collect();
    let layer = Layer { config, token_taps: false, q: pick(4, 8, &diagonal), q_scale: vec![1.0; 4],
        k: pick(2, 8, &[(0, 0), (1, 1)]), v: pick(2, 8, &[(0, 2), (1, 3)]), k_taps: (vec![0.0; 6], Vec::new()), v_taps: (vec![0.0; 6], Vec::new()),
        rel: vec![0.0; 4 * 8], rel_scale: vec![1.0; 4], table: vec![0.0; 2], sink: vec![0.0; 4 * 8], gate: vec![0.0; 4 * 8], o: pick(8, 4, &diagonal) };
    let parts = layer.run(&[vec![1.0, 2.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]], &[vec![6.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        &[vec![0.0, 0.0, 4.0, -4.0, 0.0, 0.0, 0.0, 0.0]]);
    same_bits(&parts[0].key, &[3.0, 5.0]);
    same_bits(&parts[0].value, &[1.0, -1.0]);
    same_bits(&parts[0].scores, &[3.0, 6.0, 5.0, 10.0]);
    assert!(parts[0].mixed[0] > 0.0 && parts[0].mixed[1] > 0.0 && parts[0].mixed[2] < 0.0 && parts[0].mixed[3] < 0.0);
}

#[test]
fn the_fixed_sums_have_their_named_orders() {
    let place = |count: usize, values: &[(usize, f32)]| { let mut x = vec![0.0f32; count]; for &(at, v) in values { x[at] = v; } x };
    // 4096 * 4096 = 2^24, and 2^24 + 1 is not an f32: a 1 that meets 2^24 alone is lost.
    // The 32-lane tree: lanes 0 and 2 join before lane 1, so 1 + 1 = 2 survives. A chain gives 2^24.
    assert_eq!(square_tree(&[1.0, 4096.0, 1.0]), 16_777_218.0);
    // Index 32 goes to lane 0 again: lane 0 holds 1 + 1 before the join.
    assert_eq!(square_tree(&place(40, &[(0, 1.0), (1, 4096.0), (32, 1.0)])), 16_777_218.0);
    // Lanes 1 and 17 join at the distance 16, before they meet lane 0.
    assert_eq!(square_tree(&place(40, &[(1, 1.0), (17, 1.0), (0, 4096.0)])), 16_777_218.0);
    assert_eq!(square_tree(&place(40, &[(0, 4096.0), (16, 1.0), (1, 1.0)])), 16_777_216.0);
    // The norm tree sums each chunk of 128 first: the two values of the second chunk join to 2.
    let two_chunks = place(130, &[(0, 4096.0), (128, 1.0), (129, 1.0)]);
    assert_eq!((norm_squares(&two_chunks), square_tree(&two_chunks)), (16_777_218.0, 16_777_216.0));
    // One chunk: the norm tree is the 32-lane tree.
    let one_chunk: Vec<f32> = (0..128).map(|i| 1.0 + i as f32 / 64.0).collect();
    assert_eq!(norm_squares(&one_chunk).to_bits(), square_tree(&one_chunk).to_bits());

    // The distance order: the same lanes and join inside a block of 1,024 distances.
    let by = |terms: Vec<f32>| distance_sum(terms.len(), |distance| terms[distance]);
    assert_eq!(by(vec![1.0, 16_777_216.0, 1.0]), 16_777_218.0);
    assert_eq!(by(place(40, &[(1, 1.0), (17, 1.0), (0, 16_777_216.0)])), 16_777_218.0);
    // A new block starts at the distance 1,024: its two terms join to 2 before they meet the first block.
    assert_eq!(by(place(1026, &[(0, 16_777_216.0), (1024, 1.0), (1025, 1.0)])), 16_777_218.0);
    // The block sums are a chain: 1, then 2^24, then 1. Both ones are lost.
    assert_eq!(by(place(2050, &[(0, 1.0), (1024, 16_777_216.0), (2048, 1.0)])), 16_777_216.0);
    assert_eq!((SUM_LANES, NORM_CHUNK, DISTANCE_BLOCK), (32, 128, 1024));
    assert_eq!(distance_sum(1, |_| 0.375), 0.375);
}

// Generated layers. The same formulas are in the separate calculator that made the literal bits below.
fn generated(count: usize, salt: usize) -> Vec<f32> { (0..count).map(|i| (((i * 17 + salt * 13) % 29) as i32 - 14) as f32 / 32.0).collect() }
fn generated_layer(config: Config, token_taps: bool) -> Layer {
    let (d, h) = (config, config.hidden_size);
    let (q, kv, extent) = (d.num_heads * d.head_dim, d.kv_width(), d.reach.rel_extent());
    let taps = |salt| if token_taps { (generated(d.conv_taps * kv, salt), generated(d.num_kv_heads * d.conv_taps * h, salt + 1)) }
        else { (generated(kv * d.conv_taps, salt), Vec::new()) };
    let plus_one = |values: Vec<f32>| values.into_iter().map(|v| v + 1.0).collect();
    Layer { config, token_taps, q: generated(q * h, 1), q_scale: plus_one(generated(q, 2)), k: generated(kv * h, 3), v: generated(kv * h, 4),
        k_taps: taps(11), v_taps: taps(13), rel: generated(d.num_heads * d.rel_dim * h, 5), rel_scale: plus_one(generated(d.num_heads * d.rel_dim, 6)),
        table: generated(d.rel_dim * extent, 7), sink: generated(d.num_heads * h, 8), gate: generated(q * h, 9), o: generated(h * q, 10) }
}
fn generated_inputs(tokens: usize, hidden: usize, salt: usize) -> Vec<Vec<f32>> {
    generated(tokens * hidden, salt).chunks_exact(hidden).map(|row| row.iter().map(|v| 4.0 * v).collect()).collect()
}
fn generated_config(reach: Reach) -> Config {
    Config { hidden_size: 7, num_heads: 4, num_kv_heads: 2, head_dim: 4, rel_dim: 3, conv_taps: 4, qk_clip: 1.5, rel_clip: 0.75, rms_norm_eps: 1e-6, reach }
}
fn fold(parts: &[Parts]) -> u64 {
    let mut fold = 0u64;
    for p in parts {
        for stage in [&p.query, &p.key, &p.value, &p.relative, &p.sink, &p.scores, &p.weights, &p.mixed, &p.gated, &p.output] {
            for v in stage { fold = fold.wrapping_mul(31).wrapping_add(u64::from(v.to_bits())); }
        }
    }
    fold
}

#[test]
fn generated_layers_match_the_separate_calculator() {
    // The literal bits come from an exact-fraction calculator in another language. It was
    // written from the operation's description, with each f32 step rounded one time.
    let inputs = |tokens, salt| generated_inputs(tokens, 7, salt);
    let window = generated_layer(generated_config(Reach::Window { window: 3 }), false).run(&inputs(7, 21), &inputs(7, 22), &inputs(7, 23));
    assert_eq!(fold(&window), 0x3967_54a6_5c70_0000);
    assert_eq!(bits(&window[6].output), [0xbe3a_0000, 0x3bf7_0000, 0xbcfe_0000, 0xbe3f_0000, 0x3d91_0000, 0x3f40_0000, 0x3dce_0000]);
    assert_eq!(bits(&window[6].weights[..3]), [0x3e70_0000, 0x3e95_0000, 0x3ea7_0000]);
    // The full layer runs 40 tokens: more than 32 keys, so a lane of the distance order holds more than one term.
    let full = generated_layer(generated_config(Reach::Full { rel_extent: 3, log_alpha: 0.25, log_floor: 2.0 }), true).run(&inputs(40, 21), &inputs(40, 22), &inputs(40, 23));
    assert_eq!(fold(&full), 0x15b0_cb15_162b_0000);
    assert_eq!(bits(&full[39].output), [0x3dc8_0000, 0xbdf6_0000, 0xbf12_0000, 0xbf08_0000, 0x3e97_0000, 0xbce0_0000, 0x3e89_0000]);
    assert_eq!(bits(&full[39].weights[33..40]), [0x3d3f_0000, 0x3d3e_0000, 0x3d2d_0000, 0x3c73_0000, 0x3cc9_0000, 0x3b1b_0000, 0x3c6d_0000]);
    same_bits(&full[39].query[12..], &[-1.125, -0.78125, -0.9921875, -1.5078125]);
}

// A whole-sequence oracle. It does not call the reference or its helpers. It shares only the
// BF16 round and the exponential. It computes every key and value first, then each query.
// Its fixed sums are written again here: lanes as an array and a join by halves.
fn rb(x: f32) -> f32 { round_bf16(x) }
fn product(matrix: &[f32], x: &[f32]) -> Vec<f32> {
    matrix.chunks_exact(x.len()).map(|row| rb(row.iter().zip(x).fold(0.0f32, |sum, (a, b)| sum + a * b))).collect()
}
/// Terms in index order: lane `index mod 32`, then halves of 16, 8, 4, 2 and 1 lanes.
fn lanes_sum(terms: impl Iterator<Item = f32>) -> f32 {
    let mut lanes = [0.0f32; 32];
    for (index, term) in terms.enumerate() { lanes[index % 32] += term; }
    let mut width = 16;
    while width >= 1 { for lane in 0..width { lanes[lane] += lanes[lane + width]; } width /= 2; }
    lanes[0]
}
fn by_distance(terms: &[f32]) -> f32 { terms.chunks(1024).fold(0.0f32, |total, block| total + lanes_sum(block.iter().copied())) }
fn length(x: &[f32]) -> f32 { lanes_sum(x.iter().map(|v| v * v)).sqrt() }
fn oracle_conv(layer: &Layer, data: &(Vec<f32>, Vec<f32>), raw: &[Vec<f32>], source: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let (d, channels) = (layer.config, layer.config.kv_width());
    let taps = d.conv_taps;
    (0..raw.len()).map(|t| (0..channels).map(|c| {
        let made = if layer.token_taps { product(&data.1, &source[t]) } else { Vec::new() };
        // From the oldest token of the window to the current one.
        let mut sum = 0.0f32;
        for back in (0..taps).rev() {
            let past = if t >= back { raw[t - back][c] } else { 0.0 };
            let tap = if layer.token_taps { data.0[back * channels + c] + made[(c / d.head_dim) * taps + back] } else { data.0[c * taps + (taps - 1 - back)] };
            sum += tap * past;
        }
        let terms = if layer.token_taps { 2 * taps + 1 } else { taps + 1 };
        rb((sum + raw[t][c]) * (1.0 / (terms as f64).sqrt()) as f32)
    }).collect()).collect()
}
fn oracle(layer: &Layer, xq: &[Vec<f32>], xk: &[Vec<f32>], xv: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let d = layer.config;
    let (hd, rd, group) = (d.head_dim, d.rel_dim, d.num_heads / d.num_kv_heads);
    let raw_keys: Vec<Vec<f32>> = xk.iter().map(|x| product(&layer.k, x)).collect();
    let raw_values: Vec<Vec<f32>> = xv.iter().map(|x| product(&layer.v, x)).collect();
    let keys: Vec<Vec<f32>> = oracle_conv(layer, &layer.k_taps, &raw_keys, xk).iter().map(|row| row.chunks_exact(hd).flat_map(|head| {
        let factor = (d.qk_clip * (1.0 / (length(head) + 1e-6))).min(1.0);
        head.iter().map(|&v| rb(v * factor)).collect::<Vec<f32>>()
    }).collect()).collect();
    let values: Vec<Vec<f32>> = oracle_conv(layer, &layer.v_taps, &raw_values, xv).iter().map(|row| row.chunks_exact(hd).flat_map(|head| {
        let squares = lanes_sum(head.chunks(128).map(|chunk| lanes_sum(chunk.iter().map(|v| v * v))));
        let inverse = 1.0 / (squares / hd as f32 + d.rms_norm_eps).sqrt();
        head.iter().map(|&v| rb(v * inverse)).collect::<Vec<f32>>()
    }).collect()).collect();
    let (extent, first_of, tau_of): (usize, Box<dyn Fn(usize) -> usize>, Box<dyn Fn(usize) -> f32>) = match d.reach {
        Reach::Window { window } => (window, Box::new(move |p| (p + 1).saturating_sub(window)), Box::new(|_| 1.0)),
        Reach::Full { rel_extent, log_alpha, log_floor } => (rel_extent, Box::new(|_| 0),
            Box::new(move |p| 1.0 + log_alpha * ln(((p + 1) as f32 * (1.0 / log_floor) as f32).max(1.0)))),
    };
    let states = |raw: Vec<f32>, scales: &[f32], width: usize, threshold: f32, tau: f32| -> Vec<f32> {
        raw.chunks_exact(width).zip(scales.chunks_exact(width)).flat_map(|(head, scale)| {
            let scaled: Vec<f32> = head.iter().zip(scale).map(|(v, s)| v * s).collect();
            let factor = (threshold * (1.0 / (length(&scaled) + 1e-6))).min(1.0);
            scaled.iter().map(|&v| rb(rb(v * factor) * tau)).collect::<Vec<f32>>()
        }).collect()
    };
    (0..xq.len()).map(|p| {
        let x = &xq[p];
        let query = states(product(&layer.q, x), &layer.q_scale, hd, d.qk_clip, tau_of(p));
        let relative = states(product(&layer.rel, x), &layer.rel_scale, rd, d.rel_clip, tau_of(p));
        let (sink, gate) = (product(&layer.sink, x), product(&layer.gate, x));
        // Keys by distance: the token itself first.
        let visible: Vec<usize> = (first_of(p)..=p).rev().collect();
        let mut gated = Vec::new();
        for head in 0..d.num_heads {
            let (q, r, shared) = (&query[head * hd..(head + 1) * hd], &relative[head * rd..(head + 1) * rd], head / group);
            let scores: Vec<f32> = visible.iter().map(|&j| {
                let key = &keys[j][shared * hd..(shared + 1) * hd];
                let dot = rb(q.iter().zip(key).fold(0.0f32, |sum, (a, b)| sum + a * b));
                let term = if p - j < extent { rb((0..rd).fold(0.0f32, |sum, i| sum + r[i] * layer.table[i * extent + (p - j)])) } else { 0.0 };
                rb(rb(dot * (1.0 / (hd as f64).sqrt()) as f32) + term)
            }).collect();
            let max = scores.iter().fold(sink[head], |m, &s| m.max(s));
            let exps: Vec<f32> = scores.iter().map(|&s| exp(s - max)).collect();
            let total = by_distance(&exps) + exp(sink[head] - max);
            for i in 0..hd {
                let terms: Vec<f32> = visible.iter().zip(&exps).map(|(&j, &e)| rb(e / total) * values[j][shared * hd + i]).collect();
                gated.push(rb(rb(by_distance(&terms)) * (1.0 / (1.0 + exp(-gate[head * hd + i])))));
            }
        }
        product(&layer.o, &gated)
    }).collect()
}

#[test]
fn steps_match_a_whole_sequence_oracle() {
    let small = |reach, num_heads, num_kv_heads, conv_taps| Config { hidden_size: 6, num_heads, num_kv_heads, head_dim: 3, rel_dim: 2,
        conv_taps, qk_clip: 1.25, rel_clip: 0.5, rms_norm_eps: 1e-6, reach };
    let full = Reach::Full { rel_extent: 3, log_alpha: 0.25, log_floor: 2.0 };
    let cases = [
        (generated_config(Reach::Window { window: 3 }), false), (generated_config(full), true),
        // Each kind of reach also runs with the other kind of convolution.
        (generated_config(Reach::Window { window: 5 }), true), (generated_config(Reach::Full { rel_extent: 7, log_alpha: 0.1, log_floor: 4.0 }), false),
        // One query head for each key and value head, one tap, and a window of one token.
        (small(Reach::Window { window: 1 }, 2, 2, 1), false), (small(full, 3, 1, 2), true), (small(Reach::Window { window: 2 }, 6, 2, 3), true),
    ];
    for (config, token_taps) in cases {
        let layer = generated_layer(config, token_taps);
        let h = config.hidden_size;
        let (xq, xk, xv) = (generated_inputs(11, h, 31), generated_inputs(11, h, 32), generated_inputs(11, h, 33));
        let expected = oracle(&layer, &xq, &xk, &xv);
        let stepped: Vec<Vec<f32>> = layer.run(&xq, &xk, &xv).into_iter().map(|p| p.output).collect();
        assert_eq!(stepped.iter().map(|v| bits(v)).collect::<Vec<_>>(), expected.iter().map(|v| bits(v)).collect::<Vec<_>>(), "{config:?}");
        assert!(stepped.iter().flatten().any(|&v| v != 0.0));

        // One scan, and a scan in chunks of 4, 0, 1 and 6 tokens, give the same bits and the same states.
        let reference = layer.reference();
        let (mut state, mut conv) = layer.states();
        let whole = reference.scan(&mut state, &mut conv, &xq.concat(), &xk.concat(), &xv.concat()).unwrap();
        same_bits(&whole, &expected.concat());
        let (mut chunk_state, mut chunk_conv) = layer.states();
        let (mut chunked, mut at) = (Vec::new(), 0);
        for count in [4, 0, 1, 6] {
            let range = at..at + count;
            chunked.extend(reference.scan(&mut chunk_state, &mut chunk_conv, &xq[range.clone()].concat(), &xk[range.clone()].concat(), &xv[range].concat()).unwrap());
            at += count;
        }
        same_bits(&chunked, &whole);
        assert_eq!((&chunk_state, &chunk_conv), (&state, &conv));
        assert_eq!(state.position(), 11);
    }
}

#[test]
fn config_reads_each_layer_kind_from_json() {
    let text = br#"{"model_type": "any", "text_config": {"hidden_size": 12, "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 4,
        "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2, "swa_head_dim": 8, "sliding_window_size": 16, "rel_extent": 32, "d_rel": 3,
        "sconv_kernel_size": 4, "qk_l2_clip_threshold": 2.5, "rel_l2_clip_threshold": 8.0, "rms_norm_eps": 1e-06,
        "log_scaling_alpha": 0.5, "log_scaling_n_floor": 64, "q_bias": false, "o_bias": false, "use_value_norm_no_gamma": true}}"#;
    let window = Config::from_json(text, true).unwrap();
    assert_eq!(window, Config { hidden_size: 12, num_heads: 4, num_kv_heads: 2, head_dim: 8, rel_dim: 3, conv_taps: 4, qk_clip: 2.5,
        rel_clip: 8.0, rms_norm_eps: 1e-6, reach: Reach::Window { window: 16 } });
    assert_eq!((window.kv_width(), window.reach.rel_extent(), window.histories().unwrap().elements()), (16, 16, (16 + 16 + 12 + 12) * 3));
    let full = Config::from_json(text, false).unwrap();
    assert_eq!((full.num_heads, full.num_kv_heads, full.head_dim), (6, 2, 4));
    assert_eq!(full.reach, Reach::Full { rel_extent: 32, log_alpha: 0.5, log_floor: 64.0 });

    let text = std::str::from_utf8(text).unwrap();
    let changed = |from: &str, to: &str, window: bool| { assert!(text.contains(from)); message(Config::from_json(text.replace(from, to).as_bytes(), window)) };
    assert!(changed("\"q_bias\": false", "\"q_bias\": true", true).contains("projection bias"));
    assert!(changed("\"o_bias\": false", "\"o_bias\": true", false).contains("projection bias"));
    assert!(changed("\"use_value_norm_no_gamma\": true", "\"use_value_norm_no_gamma\": false", true).contains("weighted value norm"));
    assert!(changed("\"swa_num_attention_heads\": 4", "\"swa_num_attention_heads\": 3", true).contains("multiple of key and value heads"));
    assert!(changed("\"sliding_window_size\": 16", "\"sliding_window_size\": 0", true).contains("positive"));
    assert!(changed("\"qk_l2_clip_threshold\": 2.5", "\"qk_l2_clip_threshold\": 0.0", true).contains("clip threshold"));
    assert!(changed("\"log_scaling_n_floor\": 64", "\"log_scaling_n_floor\": 0", false).contains("position scale"));
    // A field of the other layer kind is not read. A missing field has no default.
    assert!(Config::from_json(text.replace("\"log_scaling_n_floor\": 64", "\"log_scaling_n_floor\": 0").as_bytes(), true).is_ok());
    assert!(!changed("\"d_rel\": 3,", "", true).is_empty());
    assert!(!changed("\"sliding_window_size\": 16,", "", true).is_empty());
    assert!(Config::from_json(text.replace("\"sliding_window_size\": 16,", "").as_bytes(), false).is_ok());
}

#[test]
fn bad_shapes_and_values_are_refused_and_change_nothing() {
    let layer = generated_layer(generated_config(Reach::Window { window: 3 }), false);
    let reference = layer.reference();
    let (xq, xk, xv) = (generated_inputs(3, 7, 21), generated_inputs(3, 7, 22), generated_inputs(3, 7, 23));
    let (mut state, mut conv) = layer.states();
    reference.step(&mut state, &mut conv, &xq[0], &xk[0], &xv[0]).unwrap();
    let before = (state.clone(), conv.clone());

    assert!(message(reference.step(&mut state, &mut conv, &xq[1][..6], &xk[1], &xv[1])).contains("input width"));
    let mut inexact = xk[1].clone();
    inexact[2] = 1.0 + f32::EPSILON;
    assert!(message(reference.step(&mut state, &mut conv, &xq[1], &inexact, &xv[1])).contains("exact BF16"));
    let mut infinite = xv[1].clone();
    infinite[0] = f32::INFINITY;
    assert!(message(reference.step(&mut state, &mut conv, &xq[1], &xk[1], &infinite)).contains("exact BF16"));
    // A value projection that overflows stops the step after the key is ready.
    let mut wide = layer.clone();
    wide.v[..7].fill(1.0);
    let largest = vec![f32::from_bits(0x7f7f_0000); 7];
    assert!(message(wide.reference().step(&mut state, &mut conv, &xq[1], &xk[1], &largest)).contains("not finite"));
    // A chunk with a bad last token changes no state.
    let chunk = |x: &[Vec<f32>]| x[1..].concat();
    let mut bad = chunk(&xv);
    bad[13] = f32::NAN;
    assert!(reference.scan(&mut state, &mut conv, &chunk(&xq), &chunk(&xk), &bad).is_err());
    assert!(message(reference.scan(&mut state, &mut conv, &chunk(&xq), &chunk(&xk)[..7], &chunk(&xv))).contains("chunk width"));
    // A state or histories of another layer shape are refused.
    let other = generated_config(Reach::Window { window: 4 });
    assert!(message(reference.step(&mut State::new(other).unwrap(), &mut conv, &xq[1], &xk[1], &xv[1])).contains("state config"));
    assert!(message(reference.step(&mut state, &mut VisitHistories::new(8, 7, 3).unwrap(), &xq[1], &xk[1], &xv[1])).contains("history shape"));
    assert_eq!((&state, &conv), (&before.0, &before.1));
    // The state still works: the next two tokens match a run from the start.
    let tail = reference.scan(&mut state, &mut conv, &chunk(&xq), &chunk(&xk), &chunk(&xv)).unwrap();
    same_bits(&tail, &layer.run(&xq, &xk, &xv)[1..].iter().flat_map(|p| p.output.clone()).collect::<Vec<f32>>());

    // Weight checks at construction.
    let mut short = layer.clone();
    short.table.pop();
    assert!(message(Reference::new(short.config, short.weights()).map(|_| ())).contains("weight shape"));
    let mut inexact = layer.clone();
    inexact.sink[0] = 0.1;
    assert!(message(Reference::new(inexact.config, inexact.weights()).map(|_| ())).contains("exact BF16"));
    let mut fewer_taps = layer.config;
    fewer_taps.conv_taps = 3;
    assert!(message(Reference::new(fewer_taps, layer.weights()).map(|_| ())).contains("convolution shape"));
    let mut heads = layer.config;
    heads.num_kv_heads = 3;
    assert!(message(State::new(heads)).contains("multiple of key and value heads"));
    heads.num_kv_heads = 0;
    assert!(message(heads.validate()).contains("positive"));
}
