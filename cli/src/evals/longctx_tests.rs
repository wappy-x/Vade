//! Long-context bucket, summation, retrieval and exact-integer comparison tests.
use super::*;
#[test]
fn bucket_boundaries_and_summation() {
    for (i, (lo, hi)) in BUCKETS.iter().copied().enumerate() {
        assert_eq!(bucket(lo), Ok(i));
        assert_eq!(bucket(hi - 1), Ok(i));
    }
    assert!(bucket(131072).is_err());
    assert_eq!(fsum([1e100, 1.0, 1e-100]).unwrap(), 1e100);
    assert_eq!(
        fsum([1.0, 2f64.powi(-53), 2f64.powi(-54)]).unwrap(),
        f64::from_bits(1.0f64.to_bits() + 1)
    );
    assert_eq!(splitlines("a\r\nb\u{85}c\n"), vec!["a", "b", "c"]);
    assert_eq!(
        Logit::Int(9_007_199_254_740_993)
            .difference(&Logit::Int(9_007_199_254_740_992))
            .unwrap(),
        1.0
    );
    assert_eq!(
        Logit::Int(9_007_199_254_740_993).compare(&Logit::Float(9_007_199_254_740_992.0)),
        std::cmp::Ordering::Greater
    );
    assert_eq!(
        Logit::Float(-0.0).compare(&Logit::Float(0.0)),
        std::cmp::Ordering::Equal
    );
    let big = "18446744073709551616";
    let next = "18446744073709551617";
    assert_eq!(integer_difference(next, big), "1");
    let a = Logit::Big(big.into());
    let b = Logit::Big(next.into());
    assert_eq!(b.difference(&a).unwrap(), 1.0);
    assert!(
        b.compare(&Logit::Float(18_446_744_073_709_551_616.0))
            .is_gt()
    );
    assert_eq!(TokenId::Big(big.into()).to_string(), big);
    assert!(TokenId::Big(big.into()) > TokenId::Small(i64::MAX));
}
