//! Reject private literals before any export writes.

pub fn scan(name: &str, data: &[u8], denylist: &[String]) -> Result<(), String> {
    // Reference logit rows: fixtures/<model>/<case>-<position>.f32, one f32 per vocabulary entry.
    let fixture = name.starts_with("fixtures/") && name.ends_with(".f32");
    // Numeric fixtures waive UTF-8 validation, never literal scanning. Random bytes spell every
    // short word eventually, so in numeric rows a literal only counts inside a run of readable text.
    let hit = |bytes: &[u8]| denylist.iter().any(|entry| literal(bytes, entry.as_bytes()));
    if name.ends_with(".png") {
        if hit(name.as_bytes()) || !png_pixels_only(data) {
            return Err(format!("PNG must hold pixels only, no text or metadata chunks: {name}"));
        }
        return Ok(());
    }
    if if fixture { printable_runs(data, 12).any(hit) } else { hit(data) } {
        return Err(format!("private identifier detected: {name}"));
    }
    if fixture {
        if data.len() % 4 != 0
            || data.len() < 32_000 * 4
            || data.chunks_exact(4).any(|chunk| {
                let value = f32::from_le_bytes(chunk.try_into().unwrap());
                !value.is_finite() || value.abs() > 1e6
            })
        {
            return Err(format!("invalid reference logit row: {name}"));
        }
        return Ok(()); // numeric rows carry no text; the literal patterns below do not apply
    } else {
        std::str::from_utf8(data).map_err(|_| format!("invalid UTF-8 content: {name}"))?;
        if data.contains(&0) {
            return Err(format!("binary content rejected: {name}"));
        }
    }
    let patterns: &[(&str, fn(&[u8]) -> bool)] = &[
        ("personal absolute path", home_path),
        ("IPv4 address", ipv4),
        ("private key", private_key),
        ("MAC address", mac),
        ("access token", access_token),
    ];
    for (label, found) in patterns {
        if found(data) || found(name.as_bytes()) {
            return Err(format!("{label} detected: {name}"));
        }
    }
    if credential_assignment(data, !fixture) || credential_assignment(name.as_bytes(), true) {
        return Err(format!("credential assignment detected: {name}"));
    }
    // Most literals are ASCII. Fold Unicode once when needed, retaining the
    // original ASCII word-boundary rule for private identifiers.
    let unicode = denylist.iter().any(|entry| !entry.is_ascii());
    let folded = if !fixture && (unicode || !data.is_ascii()) {
        Some(fold(std::str::from_utf8(data).unwrap()))
    } else if fixture && unicode {
        Some(fold(
            &data.iter().map(|&b| char::from(b)).collect::<String>(),
        ))
    } else {
        None
    };
    let folded_name = fold(name);
    for entry in denylist {
        if literal(folded_name.as_bytes(), fold(entry).as_bytes())
            || folded
                .as_ref()
                .is_some_and(|text| literal(text.as_bytes(), fold(entry).as_bytes()))
        {
            return Err(format!("private identifier detected: {name}"));
        }
    }
    Ok(())
}

fn fold(text: &str) -> String {
    text.chars()
        .map(|c| {
            let lower = c.to_lowercase().next().unwrap();
            let mut upper = lower.to_uppercase();
            let first = upper.next().unwrap();
            if upper.next().is_none() {
                first.to_lowercase().next().unwrap()
            } else {
                lower
            }
        })
        .collect()
}

fn word(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn left_boundary(data: &[u8], start: usize) -> bool {
    start == 0 || !word(data[start - 1])
}

fn right_boundary(data: &[u8], end: usize) -> bool {
    end == data.len() || !word(data[end])
}

fn home_path(data: &[u8]) -> bool {
    (0..data.len()).any(|start| {
        [b"/Users/".as_slice(), b"/home/", b"/Volumes/"]
            .iter()
            .any(|prefix| {
                let tail = &data[start..];
                if !tail.starts_with(prefix) {
                    return false;
                }
                let user = &tail[prefix.len()..];
                let length = user
                    .iter()
                    .take_while(|&&b| word(b) || b == b'.' || b == b'-')
                    .count();
                length != 0 && user.get(length) == Some(&b'/')
            })
    })
}

fn ipv4(data: &[u8]) -> bool {
    for start in 0..data.len() {
        if !data[start].is_ascii_digit() || !left_boundary(data, start) {
            continue;
        }
        let mut end = start;
        let mut values = [0u16; 4];
        let mut widths = [0; 4];
        for part in 0..4 {
            while widths[part] < 3 && data.get(end).is_some_and(u8::is_ascii_digit) {
                values[part] = values[part] * 10 + (data[end] - b'0') as u16;
                widths[part] += 1;
                end += 1;
            }
            if widths[part] == 0 || (part != 3 && data.get(end) != Some(&b'.')) {
                break;
            }
            if part != 3 {
                end += 1;
            }
        }
        if widths.iter().all(|&width| width != 0) && right_boundary(data, end) {
            // Only the canonical localhost and wildcard spellings, and the RFC 5737
            // documentation ranges (never routable), are generic.
            let localhost = values == [127, 0, 0, 1] && widths == [3, 1, 1, 1];
            let wildcard = values == [0; 4] && widths == [1; 4];
            let documentation = matches!(values[..3], [192, 0, 2] | [198, 51, 100] | [203, 0, 113]);
            if !localhost && !wildcard && !documentation {
                return true;
            }
        }
    }
    false
}

fn private_key(data: &[u8]) -> bool {
    let begin = b"-----BEGIN ";
    data.windows(begin.len()).enumerate().any(|(i, part)| {
        part == begin
            && [b"".as_slice(), b"RSA ", b"EC ", b"OPENSSH "]
                .iter()
                .any(|kind| {
                    let tail = &data[i + begin.len()..];
                    tail.starts_with(kind) && tail[kind.len()..].starts_with(b"PRIVATE KEY-----")
                })
    })
}

fn mac(data: &[u8]) -> bool {
    data.windows(17).enumerate().any(|(start, candidate)| {
        left_boundary(data, start)
            && right_boundary(data, start + candidate.len())
            && candidate.iter().enumerate().all(|(i, byte)| {
                if i % 3 == 2 {
                    *byte == b':'
                } else {
                    byte.is_ascii_hexdigit()
                }
            })
    })
}

fn token_tail(tail: &[u8], prefix: &[u8], extra: &[u8]) -> bool {
    if !tail.starts_with(prefix) {
        return false;
    }
    let suffix = &tail[prefix.len()..];
    for (i, &byte) in suffix.iter().enumerate() {
        if !byte.is_ascii_alphanumeric() && !extra.contains(&byte) {
            break;
        }
        if i >= 19 && word(byte) && right_boundary(suffix, i + 1) {
            return true;
        }
    }
    false
}

fn access_token(data: &[u8]) -> bool {
    for start in 0..data.len() {
        if !left_boundary(data, start) {
            continue;
        }
        let tail = &data[start..];
        if (tail.starts_with(b"gh")
            && tail.get(2).is_some_and(|b| b"pousr".contains(b))
            && tail.get(3) == Some(&b'_')
            && token_tail(tail, &tail[..4], b""))
            || token_tail(tail, b"github_pat_", b"_")
            || token_tail(tail, b"hf_", b"")
            || [b"sk-".as_slice(), b"sk-proj-", b"sk-svcacct-"]
                .iter()
                .any(|prefix| token_tail(tail, prefix, b"_-"))
            || (tail.starts_with(b"AKIA")
                && tail.len() >= 20
                && tail[4..20]
                    .iter()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
                && right_boundary(tail, 20))
        {
            return true;
        }
    }
    false
}

fn credential_assignment(data: &[u8], utf8: bool) -> bool {
    let text = utf8.then(|| std::str::from_utf8(data).unwrap());
    let next = |offset: usize| {
        if let Some(text) = text {
            text[offset..].chars().next().map(|c| (c, c.len_utf8()))
        } else {
            data.get(offset).map(|&b| (char::from(b), 1))
        }
    };
    let space = |c: char| c.is_whitespace() || matches!(c, '\u{001c}'..='\u{001f}');
    (0..data.len())
        .filter(|&start| start == 0 || data[start - 1] == b'\n')
        .any(|mut offset| {
            let whitespace = |offset: &mut usize| {
                while let Some((c, width)) = next(*offset) {
                    if !space(c) {
                        break;
                    }
                    *offset += width;
                }
            };
            whitespace(&mut offset);
            if matches!(data.get(offset), Some(b'\'' | b'"')) {
                offset += 1;
            }
            let start = offset;
            while data.get(offset).is_some_and(|&b| word(b)) {
                offset += 1;
            }
            let key = &data[start..offset];
            let recognized = [
                b"password".as_slice(),
                b"passwd",
                b"api_key",
                b"access_token",
                b"secret",
                b"secret_access_key",
            ]
            .iter()
            .any(|suffix| {
                key.eq_ignore_ascii_case(suffix)
                    || (key.len() > suffix.len() + 1
                        && key[0].is_ascii_alphabetic()
                        && key[key.len() - suffix.len() - 1] == b'_'
                        && key[key.len() - suffix.len()..].eq_ignore_ascii_case(suffix))
            });
            if !recognized {
                return false;
            }
            if matches!(data.get(offset), Some(b'\'' | b'"')) {
                offset += 1;
            }
            whitespace(&mut offset);
            if !matches!(data.get(offset), Some(b':' | b'=')) {
                return false;
            }
            offset += 1;
            whitespace(&mut offset);
            if matches!(data.get(offset), Some(b'\'' | b'"')) {
                offset += 1;
            }
            for _ in 0..8 {
                let Some((c, width)) = next(offset) else {
                    return false;
                };
                if space(c) || matches!(c, '\'' | '"') {
                    return false;
                }
                offset += width;
            }
            true
        })
}

fn literal(data: &[u8], entry: &[u8]) -> bool {
    if entry.len() > data.len() {
        return false;
    }
    (0..=data.len() - entry.len()).any(|start| {
        left_boundary(data, start)
            && right_boundary(data, start + entry.len())
            && data[start..start + entry.len()].eq_ignore_ascii_case(entry)
    })
}

/// Maximal runs of printable ASCII of at least `min` bytes: where real text would hide in binary data.
fn printable_runs(data: &[u8], min: usize) -> impl Iterator<Item = &[u8]> {
    data.split(|b| !(0x20..0x7f).contains(b)).filter(move |run| run.len() >= min)
}

/// Compressed pixels spell random words, so images are not scanned for literals; instead every
/// chunk that could carry text (tEXt, zTXt, iTXt, eXIf, ...) is rejected.
fn png_pixels_only(data: &[u8]) -> bool {
    if !data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return false;
    }
    let mut at = 8;
    while at + 12 <= data.len() {
        let len = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        let next = at + 12 + len;
        let kind = &data[at + 4..at + 8];
        if next > data.len()
            || !matches!(kind, b"IHDR" | b"PLTE" | b"IDAT" | b"IEND" | b"gAMA" | b"cHRM" | b"sRGB" | b"pHYs")
        {
            return false;
        }
        if kind == b"IEND" {
            return next == data.len();
        }
        at = next;
    }
    false
}
