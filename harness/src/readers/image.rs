//! PNG, JPEG, GIF and WebP images: one line with the format, the size in
//! pixels and the path, so the agent knows what the file is and where it
//! is. Vade runs no text recognition (OCR), so no text comes from the
//! pixels. Only the headers are read: the PNG `IHDR` chunk, the JPEG frame
//! header (the segments before it are walked by their lengths), the GIF
//! screen descriptor and frame blocks, and the WebP `VP8`, `VP8L` or `VP8X`
//! chunk. A header that is cut or broken is a damaged file.
use super::{Caps, Doc, Format, ReadError, Reader, scanned};

pub(super) struct ImageReader;

impl Reader for ImageReader {
    fn sniff(&self, bytes: &[u8], _: &str) -> Option<Format> {
        if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            Some(Format::Image("PNG"))
        } else if bytes.starts_with(b"\xff\xd8\xff") {
            Some(Format::Image("JPEG"))
        } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
            Some(Format::Image("GIF"))
        } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
            Some(Format::Image("WebP"))
        } else {
            None
        }
    }

    fn to_markdown(&self, bytes: &[u8], format: Format, _: &Caps) -> Result<Doc, ReadError> {
        let bad = |why: &str| ReadError::Malformed(format, why.to_owned());
        let info = match format {
            Format::Image("PNG") => png(bytes),
            Format::Image("JPEG") => jpeg(bytes),
            Format::Image("GIF") => gif(bytes),
            Format::Image("WebP") => webp(bytes),
            _ => Err("the format is not an image"),
        }
        .map_err(bad)?;
        if info.width == 0 || info.height == 0 {
            return Err(bad("the image has no pixels"));
        }
        let mut doc = Doc::new(String::new(), 1);
        doc.markdown = format!("{}: {} x {} pixels{}.", format.label(), info.width, info.height, info.detail);
        doc.warnings.push("image pixels (no text recognition)".to_owned());
        Ok(doc)
    }
}

/// The body for the image file `name`: its line from [`ImageReader`], the
/// path, and why no text comes from it.
pub(super) fn with_path(name: &str, doc: &mut Doc) {
    doc.markdown = format!("{} The file is {name}. Vade does not run text recognition (OCR), so no text was read from the pixels.\n", doc.markdown);
}

struct Info {
    width: u64,
    height: u64,
    detail: String,
}

fn be16(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from(u16::from_be_bytes(b.get(at..at + 2)?.try_into().ok()?)))
}

fn le16(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?)))
}

fn le24(b: &[u8], at: usize) -> Option<u64> {
    let s = b.get(at..at + 3)?;
    Some(u64::from(s[0]) | u64::from(s[1]) << 8 | u64::from(s[2]) << 16)
}

fn png(b: &[u8]) -> Result<Info, &'static str> {
    const CUT: &str = "the PNG header is cut short";
    if b.get(12..16) != Some(b"IHDR") {
        return Err(if b.len() < 16 { CUT } else { "the first PNG chunk is not IHDR" });
    }
    let word = |at: usize| b.get(at..at + 4).map(|s| u64::from(u32::from_be_bytes(s.try_into().unwrap())));
    // The size comes only from inside the header chunk, which is 13 bytes
    // and its check value.
    if word(8) != Some(13) {
        return Err("the PNG header chunk does not have 13 bytes");
    }
    if b.len() < 33 {
        return Err(CUT);
    }
    let (width, height) = (word(16).ok_or(CUT)?, word(20).ok_or(CUT)?);
    let (depth, color) = (*b.get(24).ok_or(CUT)?, *b.get(25).ok_or(CUT)?);
    let kind = match color {
        0 => "grayscale",
        2 => "RGB",
        3 => "indexed color",
        4 => "grayscale with alpha",
        6 => "RGBA",
        _ => return Err("the PNG color type is not valid"),
    };
    scanned(26);
    Ok(Info { width, height, detail: format!(", {depth}-bit {kind}") })
}

fn jpeg(b: &[u8]) -> Result<Info, &'static str> {
    const CUT: &str = "the JPEG data ends before the frame header";
    let mut at = 2;
    loop {
        // Fill bytes (0xff) may come before a marker.
        while b.get(at) == Some(&0xff) && b.get(at + 1) == Some(&0xff) {
            at += 1;
        }
        let (&ff, &marker) = (b.get(at).ok_or(CUT)?, b.get(at + 1).ok_or(CUT)?);
        if ff != 0xff {
            return Err("a JPEG segment does not start with a marker");
        }
        if matches!(marker, 0xd8 | 0x01 | 0xd0..=0xd7) {
            at += 2;
            continue;
        }
        if matches!(marker, 0xd9 | 0xda) {
            return Err("the JPEG data has no frame header");
        }
        let len = be16(b, at + 2).ok_or(CUT)? as usize;
        if len < 2 {
            return Err("a JPEG segment length is not valid");
        }
        if matches!(marker, 0xc0..=0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf) {
            // The size comes only from inside the frame header: 8 bytes and
            // 3 for each component, all in the file.
            let parts = *b.get(at + 9).ok_or(CUT)?;
            if len < 8 + 3 * usize::from(parts) {
                return Err("the JPEG frame header is shorter than its fields");
            }
            if b.len() < at + 2 + len {
                return Err(CUT);
            }
            let (height, width) = (be16(b, at + 5).ok_or(CUT)?, be16(b, at + 7).ok_or(CUT)?);
            scanned(at + 10);
            let kind = match parts {
                1 => "grayscale",
                3 => "color",
                4 => "CMYK",
                _ => "color",
            };
            let mode = if matches!(marker, 0xc2 | 0xc6 | 0xca | 0xce) { ", progressive" } else { "" };
            return Ok(Info { width, height, detail: format!(", {kind}{mode}") });
        }
        at += 2 + len;
    }
}

fn gif(b: &[u8]) -> Result<Info, &'static str> {
    const CUT: &str = "the GIF data is cut short";
    let (width, height) = (le16(b, 6).ok_or(CUT)?, le16(b, 8).ok_or(CUT)?);
    let flags = *b.get(10).ok_or(CUT)?;
    let mut at = 13 + if flags & 0x80 != 0 { 3 << ((flags & 7) + 1) } else { 0 };
    // Count the frames: walk the blocks by their sizes to the trailer.
    let mut frames = 0u64;
    let skip_blocks = |mut at: usize| -> Option<usize> {
        loop {
            let n = *b.get(at)? as usize;
            at += 1 + n;
            if n == 0 {
                return Some(at);
            }
        }
    };
    loop {
        match b.get(at) {
            Some(0x2c) => {
                let local = *b.get(at + 9).ok_or(CUT)?;
                at += 10 + if local & 0x80 != 0 { 3 << ((local & 7) + 1) } else { 0 };
                at = skip_blocks(at + 1).ok_or(CUT)?;
                frames += 1;
            }
            Some(0x21) => at = skip_blocks(at + 2).ok_or(CUT)?,
            // A file cut after its first frame still shows its size.
            Some(0x3b) | None if frames > 0 => break,
            None => return Err(CUT),
            Some(_) => return Err("a GIF block has a type that is not valid"),
        }
    }
    scanned(at.min(b.len()));
    let detail = if frames > 1 { format!(", animated, {frames} frames") } else { String::new() };
    Ok(Info { width, height, detail })
}

fn webp(b: &[u8]) -> Result<Info, &'static str> {
    const CUT: &str = "the WebP header is cut short";
    scanned(b.len().min(30));
    let word = |at: usize| b.get(at..at + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()) as usize);
    let (riff, len) = (word(4).ok_or(CUT)?, word(16).ok_or(CUT)?);
    // The size comes only from inside the first chunk: it has the bytes that
    // its kind needs, and it ends inside the RIFF data.
    let need = match b.get(12..16).ok_or(CUT)? {
        b"VP8 " | b"VP8X" => 10,
        b"VP8L" => 5,
        _ => 0,
    };
    if len < need {
        return Err("the first WebP chunk is shorter than its header");
    }
    if 20 + len > riff.saturating_add(8) {
        return Err("the first WebP chunk ends after the RIFF data");
    }
    if b.len() < 20 + need {
        return Err(CUT);
    }
    match b.get(12..16).ok_or(CUT)? {
        b"VP8 " => {
            if b.get(23..26).ok_or(CUT)? != b"\x9d\x01\x2a" {
                return Err("the WebP VP8 frame has no start code");
            }
            Ok(Info { width: le16(b, 26).ok_or(CUT)? & 0x3fff, height: le16(b, 28).ok_or(CUT)? & 0x3fff, detail: ", lossy".into() })
        }
        b"VP8L" => {
            if b.get(20) != Some(&0x2f) {
                return Err(if b.len() <= 20 { CUT } else { "the WebP VP8L data has no signature" });
            }
            let bits = u64::from(u32::from_le_bytes(b.get(21..25).ok_or(CUT)?.try_into().unwrap()));
            Ok(Info { width: (bits & 0x3fff) + 1, height: (bits >> 14 & 0x3fff) + 1, detail: ", lossless".into() })
        }
        b"VP8X" => {
            let flags = *b.get(20).ok_or(CUT)?;
            let detail = if flags & 0x02 != 0 { ", animated" } else { "" };
            Ok(Info { width: le24(b, 24).ok_or(CUT)? + 1, height: le24(b, 27).ok_or(CUT)? + 1, detail: detail.into() })
        }
        _ => Err("the first WebP chunk is not VP8, VP8L or VP8X"),
    }
}
