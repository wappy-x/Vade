use super::*;

fn bits(values: &[f32]) -> Vec<u32> { values.iter().map(|v| v.to_bits()).collect() }
fn same_bits(actual: &[f32], expected: &[f32]) { assert_eq!(bits(actual), bits(expected)); }
fn message<T: std::fmt::Debug>(result: Result<T>) -> String { result.unwrap_err().to_string() }

// Two channels, four taps. Channel 0 has taps [1, 2, 4, 8] and inputs 1, 16, 256, 4096, -8.
const FIXED: [f32; 8] = [1.0, 2.0, 4.0, 8.0, 0.5, -1.0, 0.25, 3.0];
const FIXED_INPUT: [[f32; 2]; 5] = [[1.0, 2.0], [16.0, -1.0], [256.0, 0.5], [4096.0, 3.0], [-8.0, 1.5]];

#[test]
fn tap_layout_is_named_once() {
    // A wrong layout is a change of one constant or of `projected_tap`.
    assert_eq!(FIXED_TAP_ORDER, TapOrder::OldestFirst);
    assert_eq!(TOKEN_TAP_ORDER, TapOrder::NewestFirst);
    assert_eq!([0, 1, 2, 3].map(|tap| FIXED_TAP_ORDER.lag(tap, 4)), [3, 2, 1, 0]);
    assert_eq!([0, 1, 2, 3].map(|tap| TOKEN_TAP_ORDER.lag(tap, 4)), [0, 1, 2, 3]);
    assert_eq!([0, 1, 2, 3].map(|lag| (FIXED_TAP_ORDER.tap(lag, 4), TOKEN_TAP_ORDER.tap(lag, 4))), [(3, 0), (2, 1), (1, 2), (0, 3)]);
    assert_eq!([projected_tap(0, 0, 4), projected_tap(0, 3, 4), projected_tap(1, 0, 4), projected_tap(7, 2, 4)], [0, 3, 4, 30]);
    assert_eq!(Slot::ALL.map(Slot::index), [0, 1, 2, 3]);
}

#[test]
fn a_convolution_multiplies_one_time_by_the_f32_reciprocal() {
    // The reciprocal is made in f64 and rounded to f32 one time: 1 / sqrt(5) = 0.4472136, 1 / 3 = 0.33333334.
    assert_eq!((reciprocal(2.0), reciprocal(3.0).to_bits(), reciprocal(30.0).to_bits()), (0.5, 0x3eaa_aaab, 0x3d08_8889));
    assert_eq!((factor(4), factor(5).to_bits(), factor(7).to_bits(), factor(9).to_bits()), (0.5, 0x3ee4_f92e, 0x3ec1_848f, 0x3eaa_aaab));
    // One tap 0.22721708 on the input 1: the sum with the input is 1.2272171 in f32. Times
    // 0.4472136 it rounds to 0.55078125. A division by f32(sqrt(5)) would round to 0.546875.
    let taps = [0.0, 0.0, 0.0, f32::from_bits(0x3e68_ab98)];
    let conv = FixedTaps::new(1, 4, &taps).unwrap();
    assert_eq!(conv.apply(&History::new(1, 4).unwrap(), &[1.0]).unwrap()[0].to_bits(), 0x3f0d_0000);
}

#[test]
fn the_tap_chain_runs_from_the_oldest_token() {
    // Tap products 1, 2^25, -2^25 and 0 from the oldest token to the current one. From the oldest
    // token the 1 is lost in 2^25, and the chain ends at 0: (0 + 3) / 2.236068 = 1.3416407, BF16
    // 1.34375. A chain from the current token would keep the 1 and give 1.7890625.
    let big = 33_554_432.0;
    let fixed = FixedTaps::new(1, 4, &[1.0, 1.0, -1.0, 0.0]).unwrap();
    let mut history = History::new(1, 4).unwrap();
    for u in [1.0, big, big] { fixed.step(&mut history, &[u]).unwrap(); }
    same_bits(&fixed.step(&mut history, &[3.0]).unwrap(), &[1.34375]);
    // The same products with token-made taps, which are stored current token first:
    // (0 + 3) / 3 = 1. The other chain would give 1.3359375.
    let made = TokenTaps::new(1, 1, 4, 1, &[0.0, -1.0, 1.0, 1.0], &[0.0; 4]).unwrap();
    let mut history = History::new(1, 4).unwrap();
    for u in [1.0, big, big] { made.step(&mut history, &[u], &[0.0]).unwrap(); }
    same_bits(&made.step(&mut history, &[3.0], &[0.0]).unwrap(), &[1.0]);
}

#[test]
#[allow(clippy::excessive_precision)] // The literals are exact BF16 values, written with every decimal digit.
fn fixed_taps_weight_the_oldest_token_first() {
    let conv = FixedTaps::new(2, 4, &FIXED).unwrap();
    let mut history = History::new(2, 4).unwrap();
    // Token 0, channel 0: the three tokens before the text are zero.
    // (8 * 1 + 1) * f32(1 / sqrt(5)) = 4.0249224, and the nearest BF16 value is 4.03125.
    // Token 3, channel 0: 1 * 1 + 2 * 16 + 4 * 256 + 8 * 4096 + 4096 = 37921.
    // 37921 / 2.236068 = 16958.79, and the nearest BF16 value is 16896.
    // The opposite tap order gives 3920 for token 3.
    let expected: [[f32; 2]; 5] = [[4.03125, 3.578125], [66.0, -1.5625], [1056.0, -0.11181640625], [16896.0, 6.3125], [7520.0, 2.578125]];
    for (input, want) in FIXED_INPUT.iter().zip(expected) { same_bits(&conv.step(&mut history, input).unwrap(), &want); }
    // The history keeps the last three inputs, oldest first.
    same_bits(history.inputs(), &[256.0, 0.5, 4096.0, 3.0, -8.0, 1.5]);
}

#[test]
fn three_taps_have_an_exact_scale() {
    // (taps + 1) = 4, so the divisor is 2. Token 3: (2 * 5 - 1 * -2 + 0.5 * 7 + 7) / 2 = 11.25.
    let conv = FixedTaps::new(1, 3, &[2.0, -1.0, 0.5]).unwrap();
    let mut history = History::new(1, 3).unwrap();
    let outputs: Vec<f32> = [3.0, 5.0, -2.0, 7.0].iter().map(|&u| conv.step(&mut history, &[u]).unwrap()[0]).collect();
    same_bits(&outputs, &[2.25, 2.25, -1.0, 11.25]);
}

#[test]
fn the_result_is_rounded_to_bf16_one_time() {
    // (0.5 * 1.0078125 + 1.0078125) / 2.236068 = 0.67606115 in f32. BF16 keeps 0.67578125.
    let conv = FixedTaps::new(1, 4, &[0.0, 0.0, 0.0, 0.5]).unwrap();
    let output = conv.apply(&History::new(1, 4).unwrap(), &[1.0078125]).unwrap();
    assert_eq!(output[0].to_bits(), 0x3f2d_0000);
    assert_ne!(output[0].to_bits(), 0x3f2d_1258);
}

// Four channels in two heads, four taps, taps made from a source of width three.
const STATIC: [f32; 16] = [0.5, -0.25, 1.0, 2.0, 1.0, 0.5, -1.0, 0.25, -0.5, 2.0, 0.125, 1.0, 0.25, 1.0, 0.5, -2.0];
const PROJ: [f32; 24] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.5, 0.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0];
const TOKEN_INPUT: [[f32; 4]; 5] = [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 8.0], [4.0, 0.25, -2.0, 1.0], [-1.0, 16.0, 0.5, 2.0], [2.0, 1.0, 1.0, -4.0]];
const SOURCE: [[f32; 3]; 5] = [[1.0, 0.0, 0.0], [0.5, 0.25, -1.0], [0.0, 2.0, 1.0], [-1.0, 0.5, 0.25], [0.125, -0.5, 2.0]];

#[test]
#[allow(clippy::excessive_precision)] // The literals are exact BF16 values, written with every decimal digit.
fn token_taps_weight_the_current_token_first() {
    let conv = TokenTaps::new(4, 2, 4, 3, &STATIC, &PROJ).unwrap();
    // Token 1 projects [head][tap] = [0.5, 0.25, -1, 0.75] and [0.125, 1, 1, -0.75].
    same_bits(&conv.projected(&SOURCE[1]).unwrap(), &[0.5, 0.25, -1.0, 0.75, 0.125, 1.0, 1.0, -0.75]);
    // Token 0, channel 0: taps are [0.5 + 1, 1 + 0, -0.5 + 0, 0.25 + 1]. Only tap 0 meets a token.
    // (1.5 * 1 + 1) / 3 = 0.8333333, and the nearest BF16 value is 0.83203125.
    let expected: [[f32; 4]; 5] = [[0.83203125, 1.1640625, 2.0, 4.0], [0.75, 0.08349609375, 1.4140625, 10.0],
        [2.671875, 1.2265625, -3.203125, 0.66796875], [1.7109375, -1.6640625, 1.2109375, -2.171875],
        [2.890625, 0.416015625, 1.1640625, -5.75]];
    let mut history = History::new(4, 4).unwrap();
    for ((input, source), want) in TOKEN_INPUT.iter().zip(&SOURCE).zip(expected) {
        same_bits(&conv.step(&mut history, input, source).unwrap(), &want);
    }
}

#[test]
fn a_zero_projection_leaves_the_static_taps() {
    let zero = [0.0; 24];
    let conv = TokenTaps::new(4, 2, 4, 3, &STATIC, &zero).unwrap();
    let mut history = History::new(4, 4).unwrap();
    let mut last = Vec::new();
    for (input, source) in TOKEN_INPUT.iter().zip(&SOURCE).take(4) { last = conv.step(&mut history, input, source).unwrap(); }
    same_bits(&last, &[0.83203125, 4.03125, 1.5859375, 2.078125]);
}

#[test]
fn both_kinds_share_one_entry() {
    let fixed = Taps::Fixed(FixedTaps::new(2, 4, &FIXED).unwrap());
    let made = Taps::TokenMade(TokenTaps::new(4, 2, 4, 3, &STATIC, &PROJ).unwrap());
    assert_eq!((fixed.channels(), fixed.taps(), made.channels(), made.taps()), (2, 4, 4, 4));
    // Fixed taps do not read the source vector.
    same_bits(&fixed.apply(&History::new(2, 4).unwrap(), &FIXED_INPUT[0], &[]).unwrap(), &[4.03125, 3.578125]);
    same_bits(&made.apply(&History::new(4, 4).unwrap(), &TOKEN_INPUT[0], &SOURCE[0]).unwrap(), &[0.83203125, 1.1640625, 2.0, 4.0]);
}

#[test]
fn a_visit_has_four_separate_histories() {
    let conv = FixedTaps::new(2, 4, &FIXED).unwrap();
    let mut visit = VisitHistories::new(2, 3, 4).unwrap();
    // (2 + 2 + 3 + 3) channels times (4 - 1) kept inputs.
    assert_eq!(visit.elements(), 30);
    assert_eq!(Slot::ALL.map(|slot| visit.get(slot).channels()), [2, 2, 3, 3]);
    conv.step(visit.get_mut(Slot::Key), &FIXED_INPUT[0]).unwrap();
    same_bits(visit.get(Slot::Key).inputs(), &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
    assert!(visit.get(Slot::Value).inputs().iter().all(|&v| v == 0.0));
    assert!(visit.get(Slot::AfterAttention).inputs().iter().chain(visit.get(Slot::AfterExperts).inputs()).all(|&v| v == 0.0));
    // A second visit of the same layer starts from zeros and gives the first output again.
    let mut other = VisitHistories::new(2, 3, 4).unwrap();
    same_bits(&conv.step(other.get_mut(Slot::Key), &FIXED_INPUT[0]).unwrap(), &[4.03125, 3.578125]);
    same_bits(&conv.step(visit.get_mut(Slot::Key), &FIXED_INPUT[1]).unwrap(), &[66.0, -1.5625]);
}

#[test]
fn one_tap_keeps_no_history() {
    // (3 * 2 + 2) * f32(1 / sqrt(2)) = 5.656854, and the nearest BF16 value is 5.65625.
    let conv = FixedTaps::new(1, 1, &[3.0]).unwrap();
    let mut history = History::new(1, 1).unwrap();
    same_bits(&conv.step(&mut history, &[2.0]).unwrap(), &[5.65625]);
    assert!(history.inputs().is_empty());
}

#[test]
fn bad_shapes_and_values_are_refused_and_change_nothing() {
    assert!(message(FixedTaps::new(2, 4, &FIXED[..7])).contains("fixed tap shape"));
    assert!(message(FixedTaps::new(0, 4, &[])).contains("positive"));
    assert!(message(FixedTaps::new(1, 1, &[f32::NAN])).contains("finite"));
    assert!(message(TokenTaps::new(4, 3, 4, 3, &STATIC, &PROJ)).contains("multiple of heads"));
    assert!(message(TokenTaps::new(4, 2, 4, 3, &STATIC[..15], &PROJ)).contains("static tap shape"));
    assert!(message(TokenTaps::new(4, 2, 4, 2, &STATIC, &PROJ)).contains("tap projection shape"));
    let mut inexact = PROJ;
    inexact[0] = 1.0 + f32::EPSILON;
    assert!(message(TokenTaps::new(4, 2, 4, 3, &STATIC, &inexact)).contains("exact BF16"));
    assert!(message(History::new(usize::MAX, 4)).contains("overflow"));

    let conv = FixedTaps::new(2, 4, &FIXED).unwrap();
    let mut history = History::new(2, 4).unwrap();
    conv.step(&mut history, &FIXED_INPUT[0]).unwrap();
    let before = history.clone();
    assert!(message(conv.step(&mut history, &[1.0])).contains("input width"));
    assert!(message(conv.step(&mut history, &[1.0, f32::INFINITY])).contains("finite"));
    assert!(message(conv.step(&mut history, &[f32::MAX, 0.0])).contains("not finite"));
    assert!(message(conv.step(&mut History::new(2, 3).unwrap(), &FIXED_INPUT[0])).contains("history shape"));
    assert_eq!(history, before);
    let made = TokenTaps::new(4, 2, 4, 3, &STATIC, &PROJ).unwrap();
    let mut history = History::new(4, 4).unwrap();
    assert!(message(made.step(&mut history, &TOKEN_INPUT[0], &[1.0, 0.0])).contains("source width"));
    assert!(message(made.step(&mut history, &TOKEN_INPUT[0], &[1.0 + f32::EPSILON, 0.0, 0.0])).contains("exact BF16"));
    assert_eq!(history, History::new(4, 4).unwrap());
}
