//! Verification metadata transitions, independent of GPU ownership.

/// Rejected rows cannot be committed, even if the forward advanced position
/// before the row verdict arrived. The entry rollback is then a no-op.
pub fn reject<T>(verification: &mut Option<T>, position: &mut usize, valid: &mut bool, entry: usize) {
    *verification = None;
    *position = entry;
    *valid = false;
}

/// Return the accepted row count; zero means no retained verification.
pub fn committed_rows(verification: Option<(usize, usize)>, current: usize, valid: bool,
    position: usize) -> Result<usize, String> {
    let Some((entry, rows)) = verification else {
        return if position == current { Ok(0) }
            else { Err("Flash rollback needs retained verification prefixes".into()) };
    };
    if !valid || position <= entry || position - entry > rows {
        return Err("invalid Flash verified prefix".into());
    }
    Ok(position - entry)
}

#[cfg(test)]
#[test]
fn rejected_rows_cannot_commit() {
    for advanced in [0, 1, 4] {
        let (mut v, mut position, mut valid) = (Some((10, 4)), 10 + advanced, advanced != 0);
        reject(&mut v, &mut position, &mut valid, 10);
        assert_eq!((v, position, valid), (None, 10, false));
        assert_eq!(committed_rows(v, position, valid, 10).unwrap(), 0);
        assert!(committed_rows(v, position, valid, 11).is_err());
    }
}
