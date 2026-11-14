//! Fixed local files and their saved Markdown. Broken inputs use byte cuts,
//! wrong names, byte changes and a fixed random seed. No file leaves the host.
use super::*;
use super::office_tests::{Entry, deflate_bomb, deflated, stored, zip_file, zip_with};
use super::zip::{SharedCap, Zip, shared_left};

struct Case {
    name: &'static str,
    bytes: &'static [u8],
    markdown: &'static str,
}

macro_rules! cases {
    ($($name:literal),+ $(,)?) => {
        &[$(Case {
            name: $name,
            bytes: include_bytes!(concat!("../../tests/reader-corpus/", $name)),
            markdown: include_str!(concat!("../../tests/reader-corpus/golden/", $name, ".md.golden")),
        }),+]
    };
}

const CORPUS: &[Case] = cases![
    "note.txt", "utf16-le.text", "utf16-be.text", "latin1.text", "note.md", "note.html", "note.csv", "note.tsv",
    "note.json", "note.yaml", "note.toml", "note.xml", "note.rs", "note.rtf", "note.pdf", "note.docx", "note.xlsx",
    "note.pptx", "note.epub", "note.odt", "note.ods", "note.odp", "note.png", "note.jpg", "note.gif", "note.webp",
    "note.zip", "note.tar", "note.gz", "note.tgz",
];

fn small() -> Caps {
    Caps { max_chars: 256, max_pages: 1, max_rows: 2, ..Caps::default() }
}

/// A panic fails the test and names the input. A refusal is a valid result.
/// Test both the default caps and small caps, so a small cap cannot hide a fault.
fn probe(bytes: &[u8], name: &str, caps: &Caps) {
    for caps in [Caps::default(), *caps] {
        let result = std::panic::catch_unwind(|| read(bytes, name, &caps))
            .unwrap_or_else(|_| panic!("reader panic: {name}, {} bytes, {caps:?}", bytes.len()));
        if let Ok((_, doc)) = result {
            assert!(doc.chars <= caps.max_chars, "{name}: {} chars", doc.chars);
            assert_eq!(doc.chars, doc.markdown.chars().count(), "{name}");
        }
    }
}

#[test]
fn corpus_matches_saved_markdown() {
    let mut formats = std::collections::HashSet::new();
    let mut covered = vec![false; READERS.len()];
    for case in CORPUS {
        let (format, doc) = read(case.bytes, case.name, &Caps::default()).unwrap();
        assert_eq!(render(case.name, format, &doc), case.markdown, "{}", case.name);
        formats.insert(format);
        let selected = sniff(case.bytes, case.name).unwrap().1.unwrap();
        for (r, seen) in READERS.iter().zip(&mut covered) {
            *seen |= std::ptr::eq(*r, selected);
        }
    }
    assert!(covered.iter().all(|seen| *seen), "each registered reader needs a golden");
    println!("corpus: {} files, {} formats, {} readers", CORPUS.len(), formats.len(), covered.len());
}

#[test]
fn corpus_every_byte_cut_never_panics() {
    let caps = small();
    let mut count = 0;
    for case in CORPUS {
        for cut in 0..case.bytes.len() {
            let result = std::panic::catch_unwind(|| probe(&case.bytes[..cut], case.name, &caps));
            assert!(result.is_ok(), "{} cut at {cut}", case.name);
            count += 1;
        }
    }
    println!("corpus: {count} byte cuts, no panics");
}

#[test]
fn corpus_byte_changes_never_panic() {
    let caps = small();
    let mut count = 0;
    for case in CORPUS {
        let mut bytes = case.bytes.to_vec();
        for at in 0..bytes.len() {
            bytes[at] ^= 0xff;
            let result = std::panic::catch_unwind(|| probe(&bytes, case.name, &caps));
            assert!(result.is_ok(), "{} changed byte {at}", case.name);
            bytes[at] ^= 0xff;
            count += 1;
        }
    }
    println!("corpus: {count} byte changes, no panics");
}

#[test]
fn corpus_wrong_files_never_panic() {
    let caps = small();
    let mut count = 0;
    let wrong: &[&[u8]] = &[b"", b"wrong file", b"\0\xff\0\x80", b"PK\x03\x04", b"%PDF-", b"{\\rtf", b"\xff\xfe\0"];
    for case in CORPUS {
        let (format, reader) = sniff(case.bytes, case.name).unwrap();
        let reader = reader.unwrap();
        // Call the selected reader too. Magic bytes can select another reader
        // through `read`, even when the name says this reader's file type.
        for bytes in wrong.iter().copied().chain(CORPUS.iter().map(|c| c.bytes)) {
            probe(bytes, case.name, &caps);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| reader.to_markdown(bytes, format, &caps)));
            assert!(result.is_ok(), "{} direct read, {} bytes", case.name, bytes.len());
            count += 3;
        }
    }
    println!("corpus: {count} wrong-file reads, no panics");
}

#[test]
fn corpus_fixed_random_bytes_never_panic() {
    let caps = small();
    let mut seed = 0x6e65_7665_7270_616eu64;
    let mut count = 0;
    for len in 0..256 {
        let bytes: Vec<u8> = (0..len).map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u8
        }).collect();
        for case in CORPUS {
            probe(&bytes, case.name, &caps);
            count += 1;
        }
    }
    println!("corpus: {count} fixed random inputs, no panics");
}

#[test]
fn corpus_package_parts_never_panic_with_valid_zip_checksums() {
    let caps = small();
    let mut count = 0;
    for case in CORPUS.iter().filter(|c| is_zip(c.bytes) && !c.name.ends_with(".zip")) {
        let format = sniff(case.bytes, case.name).unwrap().0;
        let mut zip = Zip::open(case.bytes, format).unwrap();
        let names: Vec<String> = zip.list().map(|e| e.name.to_owned()).collect();
        let original: Vec<Vec<u8>> = names.iter().map(|name| zip.part(name).unwrap().unwrap().data).collect();
        let mut entries: Vec<Entry> = names.iter().zip(&original).map(|(name, data)| Entry { name, data: data.clone(), deflate: false, raw: None }).collect();
        for part in 0..entries.len() {
            if names[part] == "mimetype" {
                continue;
            }
            // Rebuild the container after every cut. A bad CRC must not hide
            // a fault in the XML scanner or in a document reader.
            for cut in 0..original[part].len() {
                entries[part].data.truncate(cut);
                let bytes = zip_file(&entries);
                let result = std::panic::catch_unwind(|| probe(&bytes, case.name, &caps));
                assert!(result.is_ok(), "{}: {} cut at {cut}", case.name, names[part]);
                entries[part].data.clone_from(&original[part]);
                count += 1;
            }
            for at in 0..original[part].len() {
                entries[part].data[at] = b'<';
                let bytes = zip_file(&entries);
                let result = std::panic::catch_unwind(|| probe(&bytes, case.name, &caps));
                assert!(result.is_ok(), "{}: {} changed byte {at}", case.name, names[part]);
                entries[part].data[at] = original[part][at];
                count += 1;
            }
        }
    }
    println!("corpus: {count} changed package parts with valid checksums, no panics");
}

#[test]
fn corpus_refuses_input_past_the_byte_cap_before_parsing() {
    for case in CORPUS {
        for cap in [0, case.bytes.len() - 1] {
            SCANNED.with(|n| n.set(0));
            let caps = Caps { max_bytes: cap, ..Caps::default() };
            assert_eq!(read(case.bytes, case.name, &caps).unwrap_err(), ReadError::TooLarge { bytes: case.bytes.len() as u64, cap });
            assert_eq!(SCANNED.with(|n| n.get()), 0, "{} parsed oversized input", case.name);
        }
        let caps = Caps { max_bytes: case.bytes.len(), ..Caps::default() };
        assert_eq!(read_markdown(case.bytes, case.name, &caps).unwrap(), case.markdown, "{} exact byte cap", case.name);
    }
}

#[test]
fn corpus_keeps_output_within_the_char_and_page_caps() {
    for case in CORPUS {
        for max_chars in [0, 1, 31, 128] {
            let caps = Caps { max_chars, ..small() };
            probe(case.bytes, case.name, &caps);
        }
        let (_, doc) = read(case.bytes, case.name, &small()).unwrap();
        let markers = doc.markdown.lines().filter(|l| ["<!-- page ", "<!-- slide ", "<!-- chapter ", "<!-- sheet ", "<!-- file "].iter().any(|m| l.starts_with(m))).count();
        assert!(markers <= 1, "{}: {markers} markers at page cap 1", case.name);
    }
}

#[test]
fn zip_parts_stop_at_the_decode_caps_even_with_wrong_sizes() {
    for deflate in [false, true] {
        let entry = if deflate { deflated("part.xml", &"a".repeat(8192)) } else { stored("part.xml", &"a".repeat(8192)) };
        let file = zip_file(&[entry]);
        for limit in [0, 1, 31, 4096] {
            let shared = SharedCap::new(limit);
            let mut zip = Zip::open_with(&file, Format::Docx, limit, limit).unwrap();
            let part = zip.part("part.xml").unwrap().unwrap();
            assert!(part.cut && part.data.len() <= limit);
            assert_eq!(shared_left(), limit - part.data.len());
            // A second use of the part has no room left.
            assert!(zip.part("part.xml").unwrap().unwrap().data.is_empty());
            drop(shared);
        }
    }
    let file = zip_with(&[Entry { name: "part.xml", data: vec![b'a'; 32], deflate: true, raw: Some(deflate_bomb(64)) }], |_, record| {
        record[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
    });
    let mut zip = Zip::open_with(&file, Format::Docx, 97, 97).unwrap();
    let part = zip.part("part.xml").unwrap().unwrap();
    assert!(part.cut && part.data.len() <= 97);
    assert!(zip.part("part.xml").unwrap().unwrap().data.is_empty());
}

#[test]
fn corpus_extreme_header_fields_never_panic() {
    let caps = small();
    let mut count = 0;
    for case in CORPUS {
        let mut bytes = case.bytes.to_vec();
        // Headers and trailers hold counts, lengths and offsets. Set each
        // field-sized window to zero or to all ones, also across byte cuts.
        for at in 0..bytes.len().saturating_sub(7) {
            if at >= 64 && at + 64 < bytes.len() {
                continue;
            }
            for fill in [0, 0xff] {
                bytes[at..at + 8].fill(fill);
                let result = std::panic::catch_unwind(|| probe(&bytes, case.name, &caps));
                assert!(result.is_ok(), "{} header at {at}, fill {fill}", case.name);
                bytes[at..at + 8].copy_from_slice(&case.bytes[at..at + 8]);
                count += 1;
            }
        }
    }
    println!("corpus: {count} extreme header fields, no panics");
}
