use super::Limits;
use super::budget::{Budget, Exhausted, Kind};
use super::content::{Fonts, PageText};
use super::document::Document;
use super::inflate::{inflate, zlib_stored};
use std::sync::atomic::{AtomicBool, Ordering};

pub(super) static NEVER: AtomicBool = AtomicBool::new(false);

/// A budget with the default limits and a cancel flag that is never set.
pub(super) fn budget() -> Budget<'static> {
    Budget::new(&Limits::DEFAULT, &NEVER)
}

pub(super) fn text(data: &[u8]) -> Result<String, String> {
    super::text(data, &AtomicBool::new(false))
}

/// A stream object body with a correct /Length.
pub(super) fn stream(dict: &str, data: &[u8]) -> Vec<u8> {
    let mut out = format!("<< /Length {} {dict} >>\nstream\n", data.len()).into_bytes();
    out.extend_from_slice(data);
    out.extend_from_slice(b"\nendstream");
    out
}

/// A classic PDF: object N is `objects[N - 1]`, object 1 is the catalog.
pub(super) fn pdf(objects: &[Vec<u8>], trailer_extra: &str) -> Vec<u8> {
    pdf_from(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec(), objects, trailer_extra)
}

pub(super) fn pdf_from(mut out: Vec<u8>, objects: &[Vec<u8>], trailer_extra: &str) -> Vec<u8> {
    let mut offsets = Vec::new();
    for (i, body) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes());
    for o in offsets {
        out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R {trailer_extra} >>\nstartxref\n{xref}\n%%EOF\n", objects.len() + 1).as_bytes());
    out
}

pub(super) fn b(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

/// Catalog, page tree, one page per content stream, and font F1 (object 3).
pub(super) fn simple(font: &str, pages: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let kids: Vec<String> = (0..pages.len()).map(|i| format!("{} 0 R", 4 + 2 * i)).collect();
    let mut objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b(&format!("<< /Type /Pages /Kids [{}] /Count {} /Resources << /Font << /F1 3 0 R >> >> >>", kids.join(" "), pages.len())),
        b(font),
    ];
    for (i, content) in pages.iter().enumerate() {
        objects.push(b(&format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {} 0 R >>", 5 + 2 * i)));
        objects.push(content.clone());
    }
    objects
}

pub(super) const HELVETICA: &str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";

#[test]
fn plain_two_pages_with_line_moves() {
    let p1 = stream("", b"BT /F1 12 Tf 72 720 Td (Hello, World!) Tj 0 -14 Td (Second line) Tj ET");
    let p2 = stream("", b"BT /F1 12 Tf 14 TL 72 720 Td (Top) Tj T* (Next) Tj (more) ' 0 0 Td ET\nBT 1 0 0 1 72 600 Tm (Low) Tj ET");
    let out = text(&pdf(&simple(HELVETICA, &[p1, p2]), "")).unwrap();
    assert_eq!(
        out,
        "PDF text layer extracted by Vade: 2 pages. Layout is approximate; text follows content-stream order.\n\
         page 1 of 2\nHello, World!\nSecond line\n\npage 2 of 2\nTop\nNext\nmore\nLow\n"
    );
}

#[test]
fn word_gaps_from_widths_and_kerning() {
    let font = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /FirstChar 32 /LastChar 122 /Widths 6 0 R >>";
    // Every glyph 500 units wide: at 10 pt each advances 5 units.
    let widths = b(&format!("[{}]", vec!["500"; 91].join(" ")));
    let content = stream("", b"BT /F1 10 Tf 0 0 Td [(Hel) -20 (lo) -400 (World)] TJ (A) Tj 7 0 Td (B) Tj 0.5 0 Td (C) Tj ET");
    let mut objects = simple(font, &[content]);
    objects.push(widths);
    let out = text(&pdf(&objects, "")).unwrap();
    // "-20" moves 0.2 (no space) and "-400" moves 4 (a space). Td moves from the
    // line start: "B" starts behind the pen (a space); "C" at 7.5 overlaps "B" (none).
    assert!(out.contains("page 1 of 1\nHello WorldA BC\n"), "{out}");
}

#[test]
fn flate_content_with_winansi() {
    let content = zlib_stored(b"BT /F1 12 Tf 72 720 Td (caf\xe9 \x93quoted\x94 \x80 5) Tj ET");
    let font = "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman /Encoding /WinAnsiEncoding >>";
    let out = text(&pdf(&simple(font, &[stream("/Filter /FlateDecode", &content)]), "")).unwrap();
    assert!(out.contains("page 1 of 1\ncaf\u{e9} \u{201c}quoted\u{201d} \u{20ac} 5\n"), "{out}");
}

#[test]
fn filter_chain_ascii85_over_flate() {
    let raw = zlib_stored(b"BT /F1 12 Tf (chained) Tj ET");
    let mut a85 = String::from("<~");
    for chunk in raw.chunks(4) {
        let mut g = [0u8; 4];
        g[..chunk.len()].copy_from_slice(chunk);
        let mut v = u32::from_be_bytes(g);
        let mut d = [0u8; 5];
        for x in d.iter_mut().rev() {
            *x = (v % 85) as u8 + b'!';
            v /= 85;
        }
        a85.push_str(std::str::from_utf8(&d[..chunk.len() + 1]).unwrap());
    }
    a85.push_str("~>");
    let s = stream("/Filter [/ASCII85Decode /FlateDecode]", a85.as_bytes());
    assert!(text(&pdf(&simple(HELVETICA, &[s]), "")).unwrap().contains("\nchained\n"));
    let hex: String = b"BT /F1 12 Tf (hex) Tj ET".iter().map(|b| format!("{b:02x}")).collect();
    let s = stream("/Filter /AHx", format!("{hex}>").as_bytes());
    assert!(text(&pdf(&simple(HELVETICA, &[s]), "")).unwrap().contains("\nhex\n"));
    let s = stream("/Filter /RunLengthDecode", b"\x0dBT /F1 12 Tf (\xfex\x0b) Tj ET word\x80");
    assert!(text(&pdf(&simple(HELVETICA, &[s]), "")).unwrap().contains("\nxxx\n"));
}

/// Objects 1..=4 live in object stream 5; an xref stream (object 6) with PNG
/// Up prediction indexes them.
#[test]
fn xref_stream_and_object_stream() {
    let inner = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".into(),
        "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 7 0 R >>".into(),
        HELVETICA.into(),
    ];
    let mut header = String::new();
    let mut body = String::new();
    for (i, o) in inner.iter().enumerate() {
        header.push_str(&format!("{} {} ", i + 1, body.len()));
        body.push_str(o);
        body.push(' ');
    }
    let objstm = format!("{header}\n{body}");
    let mut out = b"%PDF-1.5\n".to_vec();
    let at5 = out.len();
    out.extend_from_slice(b"5 0 obj\n");
    out.extend_from_slice(&stream(&format!("/Type /ObjStm /N 4 /First {} /Filter /FlateDecode", header.len() + 1), &zlib_stored(objstm.as_bytes())));
    out.extend_from_slice(b"\nendobj\n");
    let at7 = out.len();
    out.extend_from_slice(b"7 0 obj\n");
    out.extend_from_slice(&stream("", b"BT /F1 9 Tf (from an object stream) Tj ET"));
    out.extend_from_slice(b"\nendobj\n");
    let at6 = out.len();
    // Rows: type (1 byte), field 2 (2 bytes), field 3 (1 byte); objects 0..=7.
    let rows: Vec<[u8; 4]> = vec![
        [0, 0, 0, 255],
        [2, 0, 5, 0],
        [2, 0, 5, 1],
        [2, 0, 5, 2],
        [2, 0, 5, 3],
        [1, (at5 >> 8) as u8, at5 as u8, 0],
        [1, (at6 >> 8) as u8, at6 as u8, 0],
        [1, (at7 >> 8) as u8, at7 as u8, 0],
    ];
    let mut predicted = Vec::new();
    let mut prev = [0u8; 4];
    for r in &rows {
        predicted.push(2);
        predicted.extend(r.iter().zip(prev).map(|(a, p)| a.wrapping_sub(p)));
        prev = *r;
    }
    out.extend_from_slice(b"6 0 obj\n");
    let dict = "/Type /XRef /Size 8 /W [1 2 1] /Root 1 0 R /Filter /FlateDecode /DecodeParms << /Predictor 12 /Columns 4 >>";
    out.extend_from_slice(&stream(dict, &zlib_stored(&predicted)));
    out.extend_from_slice(format!("\nendobj\nstartxref\n{at6}\n%%EOF\n").as_bytes());
    let got = text(&out).unwrap();
    assert!(got.contains("page 1 of 1\nfrom an object stream\n"), "{got}");
}

#[test]
fn identity_h_with_tounicode() {
    let cmap = b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n\
        1 begincodespacerange <0000> <FFFF> endcodespacerange\n\
        2 beginbfchar <0003> <0020> <0010> <D835DC00> endbfchar\n\
        2 beginbfrange <0020> <0022> <0041> <0030> <0031> [<00660069> <00E9>] endbfrange\n\
        endcmap CMapName currentdict /CMap defineresource pop end end";
    let objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b("<< /Type /Pages /Kids [4 0 R] /Count 1 >>"),
        b("<< /Type /Font /Subtype /Type0 /BaseFont /ABCDEF+Noto /Encoding /Identity-H /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>"),
        b("<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>"),
        stream("", b"BT /F1 11 Tf 50 700 Td <0020002100220003003000310010> Tj <0099> Tj ET"),
        b("<< /Type /Font /Subtype /CIDFontType2 /BaseFont /ABCDEF+Noto /DW 1000 /W [32 [600 600 600] 48 49 500] >>"),
        stream("", cmap),
    ];
    let out = text(&pdf(&objects, "")).unwrap();
    assert!(out.contains("page 1 of 1\nABC fi\u{e9}\u{1d400}\n"), "{out}");
    assert!(out.contains("[1 characters were left out: their fonts have no Unicode mapping]"), "{out}");
}

#[test]
fn differences_and_glyph_names() {
    let font = "<< /Type /Font /Subtype /Type1 /BaseFont /Custom /Encoding << /BaseEncoding /WinAnsiEncoding /Differences [65 /uni00E9 /f_i 1 /quoteright /alpha /Omega.sc /g123] >> >>";
    let content = stream("", b"BT /F1 12 Tf (ABC\x01\x02\x03\x04Z) Tj ET");
    let out = text(&pdf(&simple(font, &[content]), "")).unwrap();
    assert!(out.contains("page 1 of 1\n\u{e9}fiC\u{2019}\u{3b1}\u{3a9}Z\n"), "{out}");
    assert!(out.contains("[1 characters were left out"), "{out}");
    assert_eq!(super::font::glyph_text("u1F600").as_deref(), Some("\u{1f600}"));
    assert_eq!(super::font::glyph_text("uniD83DDE00").as_deref(), Some("\u{1f600}"));
}

#[test]
fn macroman_and_standard_encodings() {
    let mac = "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman /Encoding /MacRomanEncoding >>";
    let out = text(&pdf(&simple(mac, &[stream("", b"BT /F1 12 Tf (\x80\x8e\xd2\xd3\xa5\xff) Tj ET")]), "")).unwrap();
    assert!(out.contains("\n\u{c4}\u{e9}\u{201c}\u{201d}\u{2022}\u{2c7}\n"), "{out}");
    let out = text(&pdf(&simple(HELVETICA, &[stream("", b"BT /F1 12 Tf (it`s don't \xae\xd0) Tj ET")]), "")).unwrap();
    assert!(out.contains("\nit\u{2018}s don\u{2019}t fi\u{2014}\n"), "{out}");
}

#[test]
fn form_xobject_and_inline_image() {
    let objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        b("<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Fm1 5 0 R /Im1 6 0 R >> /Font << /F1 7 0 R >> >> /Contents 4 0 R >>"),
        stream("", b"BT /F1 10 Tf 0 700 Td (Body) Tj ET\nq /Fm1 Do Q /Im1 Do q 1 0 0 1 0 -50 cm /Fm1 Do Q\nBI /W 4 /H 1 /BPC 8 /CS /G ID \x00(Tj)\xffEI x EI\nBT /F1 10 Tf 0 600 Td (After) Tj ET"),
        stream("/Type /XObject /Subtype /Form /BBox [0 0 600 100] /Matrix [1 0 0 1 0 -400] /Resources << /XObject << /Self 5 0 R >> /Font << /F1 7 0 R >> >>", b"BT /F1 10 Tf 0 750 Td (Footer) Tj ET /Self Do"),
        stream("/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 /ColorSpace /DeviceGray /Filter /DCTDecode", b"\xff\xd8"),
        b(HELVETICA),
    ];
    let out = text(&pdf(&objects, "")).unwrap();
    assert!(out.contains("page 1 of 1\nBody\nFooter\nFooter\nAfter\n"), "{out}");
}

#[test]
fn encrypted_is_refused() {
    let mut objects = simple(HELVETICA, &[stream("", b"BT /F1 12 Tf (secret) Tj ET")]);
    objects.push(b("<< /Filter /Standard /V 2 /R 3 /O <00> /U <00> /P -4 >>"));
    let err = text(&pdf(&objects, "/Encrypt 6 0 R /ID [<01> <01>]")).unwrap_err();
    assert_eq!(err, "PDF is encrypted; Vade does not decrypt PDF files");
}

#[test]
fn image_only_reports_no_text_layer() {
    let objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b("<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>"),
        b("<< /Type /Page /Parent 2 0 R /Contents 4 0 R /Resources << /XObject << /Im 6 0 R >> >> >>"),
        stream("", b"q 612 0 0 792 0 0 cm /Im Do Q"),
        b("<< /Type /Page /Parent 2 0 R /Contents 4 0 R /Resources << /XObject << /Im 6 0 R >> >> >>"),
        stream("/Type /XObject /Subtype /Image /Filter /JBIG2Decode", b"\x97JB2"),
    ];
    assert_eq!(
        text(&pdf(&objects, "")).unwrap(),
        "no text layer, 2 pages: this PDF has no extractable text (scanned images or text drawn as shapes); Vade does not run OCR.\n"
    );
}

#[test]
fn damaged_xref_is_rebuilt_by_scan() {
    let good = pdf(&simple(HELVETICA, &[stream("", b"BT /F1 12 Tf (survives) Tj ET")]), "");
    // Shift every byte after the header so each offset in the table is wrong.
    let mut shifted = b"%PDF-1.4\n% junk junk junk\n".to_vec();
    shifted.extend_from_slice(&good[9..]);
    assert!(text(&shifted).unwrap().contains("page 1 of 1\nsurvives\n"));
    // No xref and no trailer at all: the catalog is found by its /Type.
    let cut = good.windows(4).position(|w| w == b"xref").unwrap();
    assert!(text(&good[..cut]).unwrap().contains("\nsurvives\n"));
}

#[test]
fn incremental_update_wins() {
    let mut file = pdf(&simple(HELVETICA, &[stream("", b"BT /F1 12 Tf (old) Tj ET")]), "");
    let prev = file.windows(9).rposition(|w| w == b"startxref").unwrap();
    let prev: usize = std::str::from_utf8(&file[prev + 10..]).unwrap().lines().next().unwrap().parse().unwrap();
    let at = file.len();
    file.extend_from_slice(b"5 0 obj\n");
    file.extend_from_slice(&stream("", b"BT /F1 12 Tf (new) Tj ET"));
    file.extend_from_slice(b"\nendobj\n");
    let xref = file.len();
    file.extend_from_slice(format!("xref\n5 1\n{at:010} 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R /Prev {prev} >>\nstartxref\n{xref}\n%%EOF\n").as_bytes());
    let out = text(&file).unwrap();
    assert!(out.contains("\nnew\n") && !out.contains("old"), "{out}");
}

#[test]
fn not_a_pdf_and_no_pages() {
    assert!(!super::is_pdf(b"hello"));
    assert!(super::is_pdf(b"\xff\xfe junk %PDF-1.7") && super::is_pdf(b"%PDF-1.7"));
    assert!(!super::is_pdf(b"let header = b\"%PDF-1.4\";"));
    assert!(text(b"plain text").is_err());
    let objects = vec![b("<< /Type /Catalog /Pages 2 0 R >>"), b("<< /Type /Pages /Kids [] /Count 0 >>")];
    assert_eq!(text(&pdf(&objects, "")).unwrap_err(), "PDF has no pages");
}

/// Flipped and truncated bytes must never panic.
#[test]
fn corrupt_input_never_panics() {
    let mut objects = simple(HELVETICA, &[stream("/Filter /FlateDecode", &zlib_stored(b"BT /F1 12 Tf (x) Tj [(a) 5 (b)] TJ ET"))]);
    objects.push(b("<< /Type /ObjStm /N 99 /First 1 >>"));
    let good = pdf(&objects, "");
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for round in 0..3000 {
        let mut bad = good.clone();
        for _ in 0..1 + round % 6 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let i = (seed % bad.len() as u64) as usize;
            bad[i] = (seed >> 32) as u8;
        }
        if round % 5 == 0 {
            bad.truncate((seed >> 40) as usize % bad.len());
        }
        let _ = text(&bad);
    }
}

pub(super) fn gzip_body(hex: &str) -> Vec<u8> {
    let bytes: Vec<u8> = (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect();
    bytes[10..bytes.len() - 8].to_vec()
}

/// Vectors from `gzip -9n` (header and trailer removed, so raw deflate).
#[test]
fn inflate_fixed_and_dynamic_vectors() {
    let mut out = Vec::new();
    inflate(&gzip_body("1f8b0800000000000203f348cdc9c9d751087071530400104554970b000000"), 1 << 20, &mut out, &budget()).unwrap();
    assert_eq!(out, b"Hello, PDF!");
    let dynamic = "1f8b08000000000002039dd53b4e43311484e19e559c121af079f8d5460a554a6f20527c13102248dcfd0baadc115598c2e5b8f8f4cbde0d7979555193b1483529adc838c9e3e1ed738aca7591f532e534978fe33a659ddfab7c1dcff3f949c6bbecc7c3eeeffef7dcf646ec73dbf64eec23b67d107b4fdb3e137b05bf42ec13f8d5ffef7303bf46ec2bf875625fc04f13714160804481d941508904b302a1120de604864a44181d11890aa3222291611444243a8c404422c47044244a0c0344234a8c04884694e81ddf42a244af806844895e00d188123d232251a23b221225ba212251a22744244ab48e884489d600d18912ad00a213255a0644274a34c76f9928d10c109d28d114118912b5232251a23644244ad4828844899a119128510310e38e127f007056a0471f0a0000";
    let expected: String = (1..=40).map(|i| format!("BT /F1 12 Tf 72 {} Td (Line {i} of the deflate test page.) Tj ET\n", 700 - i * 14)).collect();
    out.clear();
    inflate(&gzip_body(dynamic), 1 << 20, &mut out, &budget()).unwrap();
    assert_eq!(out.len(), 2591);
    assert_eq!(out, expected.as_bytes());
    // The output cap stops a stream early and keeps what was decoded.
    out.clear();
    assert!(inflate(&gzip_body(dynamic), 100, &mut out, &budget()).is_err());
    assert_eq!(out, &expected.as_bytes()[..100]);
}

#[test]
fn inflate_stored_and_hand_built_blocks() {
    let data: Vec<u8> = (0..70_000u32).map(|i| (i * 7 % 251) as u8).collect();
    let mut out = Vec::new();
    inflate(&zlib_stored(&data), 1 << 20, &mut out, &budget()).unwrap();
    assert_eq!(out, data);
    out.clear();
    inflate(&zlib_stored(b""), 16, &mut out, &budget()).unwrap();
    assert!(out.is_empty());
    // Fixed Huffman by hand: literal 'a' (0x61 -> code 0x91, 8 bits), then
    // length 3 distance 1 (code 257 -> 0000001, distance code 00000), end 0000000.
    // Bits LSB first: 1 (final) 10 (fixed) | 10001001 reversed... built below.
    let mut bits: Vec<u8> = vec![1, 1, 0];
    let push = |code: u32, len: u32, bits: &mut Vec<u8>| (0..len).rev().for_each(|k| bits.push((code >> k & 1) as u8));
    push(0x30 + 0x61, 8, &mut bits);
    push(1, 7, &mut bits);
    push(0, 5, &mut bits);
    push(0, 7, &mut bits);
    let pack = |bits: &[u8]| -> Vec<u8> { bits.chunks(8).map(|c| c.iter().enumerate().fold(0, |a, (i, &b)| a | b << i)).collect() };
    out.clear();
    inflate(&pack(&bits), 16, &mut out, &budget()).unwrap();
    assert_eq!(out, b"aaaa");
    // A copy with no output before it reaches before the start.
    let mut far = vec![1, 1, 0];
    push(1, 7, &mut far);
    push(0, 5, &mut far);
    let far = pack(&far);
    // Errors: reserved block type, distance before the start, truncation.
    for bad in [&[0x07u8][..], &far[..], &gzip_body("1f8b0800000000000203f348cdc9c9d751087071530400104554970b000000")[..6]] {
        out.clear();
        assert!(inflate(bad, 16, &mut out, &budget()).is_err(), "{bad:02x?}");
    }
}

#[test]
fn encoding_tables_have_full_size() {
    assert_eq!(super::font::MAC_80.chars().count(), 128);
}

/// A one-font PDF with one page per string, for tests outside this module.
pub(crate) fn sample(pages: &[&str]) -> Vec<u8> {
    let streams: Vec<Vec<u8>> = pages.iter().map(|t| stream("", format!("BT /F1 12 Tf 72 720 Td ({t}) Tj ET").as_bytes())).collect();
    pdf(&simple(HELVETICA, &streams), "")
}

/// The same file with no binary comment line: every byte is ASCII.
pub(crate) fn sample_ascii(pages: &[&str]) -> Vec<u8> {
    let streams: Vec<Vec<u8>> = pages.iter().map(|t| stream("", format!("BT /F1 12 Tf 72 720 Td ({t}) Tj ET").as_bytes())).collect();
    pdf_from(b"%PDF-1.4\n".to_vec(), &simple(HELVETICA, &streams), "")
}

#[test]
fn page_and_text_caps_are_reported() {
    let pages = vec![stream("", b"BT /F1 12 Tf (p) Tj ET"); super::MAX_PAGES + 1];
    let out = text(&pdf(&simple(HELVETICA, &pages), "")).unwrap();
    assert!(out.ends_with(&format!("page 2000 of 2001\np\n\n[stopped after page 2000 of 2001: page limit {}]\n", super::MAX_PAGES)), "{}", &out[out.len() - 200..]);
    // Two half-limit pages plus their header lines pass the limit inside page 2.
    let big = format!("BT /F1 12 Tf ({}) Tj ET", "w".repeat(super::MAX_TEXT_BYTES / 2));
    let out = text(&pdf(&simple(HELVETICA, &vec![stream("", big.as_bytes()); 3]), "")).unwrap();
    assert!(out.ends_with("w\n\n[text stopped inside page 2 of 3: extracted text limit 4 MiB]\n"), "{}", &out[out.len() - 200..]);
    assert!(out.len() < super::MAX_TEXT_BYTES + 400);
}

/// Many bfrange entries, written out of order: lookup must still find each.
#[test]
fn tounicode_with_many_ranges() {
    let mut ranges: Vec<String> = (0..3000u32).map(|i| format!("<{:04X}> <{:04X}> <{:04X}>", i * 2, i * 2 + 1, 0x4e00 + i * 2)).collect();
    ranges.reverse();
    let cmap = format!("begincmap 1 begincodespacerange <0000> <FFFF> endcodespacerange 3000 beginbfrange {} endbfrange endcmap", ranges.join("\n"));
    let objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b("<< /Type /Pages /Kids [4 0 R] /Count 1 >>"),
        b("<< /Type /Font /Subtype /Type0 /BaseFont /CJK /Encoding /Identity-H /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>"),
        b("<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>"),
        stream("", b"BT /F1 11 Tf <00000001176E176F> Tj ET"),
        b("<< /Type /Font /Subtype /CIDFontType0 /BaseFont /CJK >>"),
        stream("", cmap.as_bytes()),
    ];
    let out = text(&pdf(&objects, "")).unwrap();
    assert!(out.contains("page 1 of 1\n\u{4e00}\u{4e01}\u{656e}\u{656f}\n"), "{out}");
}

pub(super) fn read_with(data: &[u8], limits: Limits) -> String {
    super::read(data, &limits, &AtomicBool::new(false)).unwrap()
}

/// Forms 5..=12 each call the next `fan` times; form 12 shows `leaf`. The
/// chain is acyclic and 8 deep: 64 calls each give 64^7 leaf calls.
pub(super) fn form_chain(fan: usize, leaf: &str) -> Vec<u8> {
    form_chain_pages(fan, leaf, 1)
}

/// `form_chain` with `pages` pages that share one content stream.
pub(super) fn form_chain_pages(fan: usize, leaf: &str, pages: usize) -> Vec<u8> {
    let page = "<< /Type /Page /Parent 2 0 R /Resources << /XObject << /N 5 0 R >> /Font << /F1 13 0 R >> >> /Contents 4 0 R >>";
    let kids: Vec<String> = std::iter::once(3).chain(14..13 + pages).map(|n| format!("{n} 0 R")).collect();
    let mut objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b(&format!("<< /Type /Pages /Kids [{}] /Count {pages} >>", kids.join(" "))),
        b(page),
        stream("", b"BT /F1 10 Tf 0 700 Td (Body) Tj ET /N Do BT /F1 10 Tf 0 600 Td (tail) Tj ET"),
    ];
    for k in 0..8 {
        let (resources, body) = if k == 7 { ("/Font << /F1 13 0 R >>".to_string(), leaf.to_string()) } else { (format!("/XObject << /N {} 0 R >>", 6 + k), "/N Do ".repeat(fan)) };
        objects.push(stream(&format!("/Type /XObject /Subtype /Form /BBox [0 0 1 1] /Resources << {resources} >>"), body.as_bytes()));
    }
    objects.push(b(HELVETICA));
    objects.extend((1..pages).map(|_| b(page)));
    pdf(&objects, "")
}

/// Astra BLOCK 1: depth and cycle guards alone let an acyclic chain run
/// 64^7 form calls. The per-page budget stops it, and the page text after
/// the chain still comes out.
#[test]
fn form_calls_have_a_page_budget() {
    let started = std::time::Instant::now();
    let out = text(&form_chain(64, "")).unwrap();
    assert!(out.contains("page 1 of 1\nBody\ntail\n") && out.contains("[work limit reached on page 1: some content there was not read]"), "{out}");
    assert!(started.elapsed() < std::time::Duration::from_secs(20), "{:?}", started.elapsed());
    // 10 calls: forms 5..=12 (one leaf), then two more leaves.
    let out = read_with(&form_chain(64, "BT /F1 1 Tf (x) Tj ET"), Limits { form_calls: 10, ..Limits::DEFAULT });
    assert!(out.contains("\nBody\nxxx\ntail\n"), "{out}");
    // Form content bytes: form 5 is 64 * 6 = 384 bytes, so a 200-byte budget
    // cuts it and skips every call inside it.
    let out = read_with(&form_chain(64, "BT /F1 1 Tf (x) Tj ET"), Limits { form_bytes: 200, ..Limits::DEFAULT });
    assert!(out.contains("\nBody\ntail\n") && out.contains("[work limit reached on page 1"), "{out}");
    // The whole read has a budget too: 3 pages share the chain and 10 calls.
    let out = read_with(&form_chain_pages(64, "BT /F1 1 Tf (x) Tj ET", 3), Limits { document_form_calls: 10, ..Limits::DEFAULT });
    assert!(out.contains("page 1 of 3\nBody\nxxx\ntail\n\npage 2 of 3\nBody\ntail\n\npage 3 of 3\nBody\ntail\n"), "{out}");
    assert!(out.contains("[work limit reached on page 1, 2, 3: some content there was not read]"), "{out}");
}

/// The cancel flag stops a read: between pages, and inside one page's
/// operator loop.
#[test]
fn cancel_stops_a_read() {
    let file = form_chain(64, "");
    let set = AtomicBool::new(true);
    assert_eq!(super::text(&file, &set).unwrap_err(), "PDF read cancelled");
    let flag = AtomicBool::new(false);
    let budget = Budget::new(&Limits::DEFAULT, &flag);
    let doc = Document::open(&file, &budget).unwrap();
    let pages = doc.pages(1).list;
    flag.store(true, Ordering::Relaxed);
    let mut fonts = Fonts::default();
    let mut reader = PageText::new(&doc, &mut fonts, &Limits::DEFAULT, 1000);
    reader.run(b"BT /F1 10 Tf (Body) Tj ET", &pages[0].resources);
    assert_eq!((reader.out.as_str(), budget.stopped()), ("", Some(Exhausted::Cancelled)));
}

/// Astra BLOCK 2: an unfiltered stream had no charge, so one /Contents array
/// that repeats it could join any amount of content. Each use is charged now.
#[test]
fn repeated_streams_are_charged_each_use() {
    let refs = vec!["5 0 R"; 200].join(" ");
    let objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b("<< /Type /Pages /Kids [4 0 R] /Count 1 /Resources << /Font << /F1 3 0 R >> >> >>"),
        b(HELVETICA),
        b(&format!("<< /Type /Page /Parent 2 0 R /Contents [{refs} 6 0 R] >>")),
        stream("", "0 0 m 1 1 l S\n".repeat(70).as_bytes()),
        stream("", b"BT /F1 12 Tf (tail) Tj ET"),
    ];
    let file = pdf(&objects, "");
    assert!(text(&file).unwrap().contains("\ntail\n"));
    // The whole-read budget stops the read; a page limit skips the rest of the page.
    let stop = "[work limit reached on page 1 of 1 (bytes decoded limit): the read stopped there]";
    let skip = "[work limit reached on page 1: some content there was not read]";
    for (limits, note) in [(Limits { stream_bytes: 50_000, ..Limits::DEFAULT }, stop), (Limits { page_bytes: 50_000, ..Limits::DEFAULT }, skip), (Limits { page_streams: 200, ..Limits::DEFAULT }, skip)] {
        let out = read_with(&file, limits);
        assert!(!out.contains("tail") && out.contains(note), "{out}");
    }
}

/// Pages that share one /Contents array of references to a non-stream
/// object charge no stream bytes, so the entries read per page are bounded.
/// Before, 2,000 pages that share 4,000,000 entries ran for over 60 s.
#[test]
fn contents_entries_per_page_are_bounded() {
    let mut objects = simple(HELVETICA, &[stream("", b"BT /F1 12 Tf (first) Tj ET"), b("null")]);
    objects[5] = b(&format!("<< /Type /Page /Parent 2 0 R /Contents [5 0 R {} 5 0 R] >>", "7 0 R ".repeat(5_000)));
    objects.push(b("null"));
    let out = text(&pdf(&objects, "")).unwrap();
    assert!(out.contains("page 1 of 2\nfirst\n\npage 2 of 2\nfirst\n\n[work limit reached on page 2:"), "{out}");
}

/// Pages nodes that share one /Kids array: the walk stays within its node
/// limit. Before, 400,000 such nodes (a 36 MB file) used 1.7 GB.
#[test]
fn shared_kids_arrays_stay_bounded() {
    let mut objects = vec![b("<< /Type /Catalog /Pages 4 0 R >>"), b(""), b(""), b("<< /Type /Pages /Kids 2 0 R /Count 1 >>")];
    objects[1] = b(&format!("[3 0 R {}]", (4..504).map(|n| format!("{n} 0 R")).collect::<Vec<_>>().join(" ")));
    objects[2] = b("<< /Type /Page /Contents 504 0 R /Resources << /Font << /F1 505 0 R >> >> >>");
    objects.extend((5..504).map(|_| b("<< /Type /Pages /Kids 2 0 R /Count 1 >>")));
    objects.push(stream("", b"BT /F1 12 Tf (leaf) Tj ET"));
    objects.push(b(HELVETICA));
    let out = text(&pdf(&objects, "")).unwrap();
    assert!(out.contains("1 pages") && out.contains("page 1 of 1\nleaf\n"), "{out}");
}

/// Each filter stops at the remaining budget; no append crosses it.
#[test]
fn filters_stop_at_the_budget() {
    let lzw: Vec<u8> = {
        // 20 literal codes 'x' (9 bits each, most significant bit first), then end of data.
        let codes: Vec<u32> = std::iter::repeat_n(u32::from(b'x'), 20).chain([257]).collect();
        let bits: Vec<u8> = codes.iter().flat_map(|&c| (0..9).rev().map(move |k| (c >> k & 1) as u8)).collect();
        bits.chunks(8).map(|c| c.iter().enumerate().fold(0, |a, (i, &b)| a | b << (7 - i))).collect()
    };
    let streams = [
        stream("", &[b'u'; 100]),
        stream("/Filter /FlateDecode", &zlib_stored(&[b'f'; 150])),
        stream("/Filter /LZWDecode", &lzw),
        stream("/Filter /RunLengthDecode", b"\x81r\x80"),
        stream("/Filter /ASCII85Decode", b"zzzzzzzz~>"),
        stream("/Filter /ASCIIHexDecode", "61".repeat(40).as_bytes()),
        stream("/Filter /FlateDecode /DecodeParms << /Predictor 12 /Colors 32 /BitsPerComponent 16 /Columns 1048576 >>", &zlib_stored(b"\x02")),
    ];
    // Objects 3..=9 after the catalog and an empty page tree.
    let mut objects = vec![b("<< /Type /Catalog /Pages 2 0 R >>"), b("<< /Type /Pages /Kids [] /Count 0 >>")];
    objects.extend(streams);
    let file = pdf(&objects, "");
    let limited = |bytes| Budget::new(&Limits { stream_bytes: bytes, ..Limits::DEFAULT }, &NEVER);
    let size = |n: u32, doc: &Document, cap: usize| doc.stream_data_within(&doc.get(n, 0).unwrap(), cap).map(|d| d.len());
    // Exactly the budget passes; the next use stops the budget.
    let b = limited(100);
    let doc = Document::open(&file, &b).unwrap();
    assert_eq!(size(3, &doc, usize::MAX), Some(100));
    assert!(!doc.take_cut() && b.stopped().is_none());
    assert_eq!((size(3, &doc, usize::MAX), b.stopped()), (None, Some(Exhausted::Limit(Kind::Decoded))));
    let full = budget();
    let doc = Document::open(&file, &full).unwrap();
    let all: Vec<Option<usize>> = (4..=8).map(|n| size(n, &doc, usize::MAX)).collect();
    assert_eq!(all, [Some(150), Some(20), Some(128), Some(32), Some(40)]);
    // A caller cap cuts each filter at the cap and marks the cut.
    for n in 4..=8 {
        assert_eq!(size(n, &doc, 10), Some(10), "object {n}");
        assert!(doc.take_cut());
    }
    // Each filter charges its output first: a 10-byte budget stops each one.
    for n in 4..=8 {
        let b = limited(10);
        let doc = Document::open(&file, &b).unwrap();
        assert_eq!((size(n, &doc, usize::MAX), b.stopped()), (None, Some(Exhausted::Limit(Kind::Decoded))), "object {n}");
    }
    // The predictor does not pad a short row: 1 input byte, 0 data bytes.
    assert_eq!(size(9, &doc, usize::MAX), Some(0));
}

/// Astra BLOCK 3: one Tj decoded the whole string into a temporary before
/// any limit check. Glyphs now go straight to the page text and stop there.
#[test]
fn text_operators_stop_at_the_text_limit() {
    let cmap = format!("begincmap 1 begincodespacerange <00> <FF> endcodespacerange 1 beginbfchar <01> <{}> endbfchar endcmap", "0041".repeat(256));
    let codes = "01".repeat(20_000);
    let objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b("<< /Type /Pages /Kids [4 0 R] /Count 1 >>"),
        b("<< /Type /Font /Subtype /Type1 /BaseFont /X /ToUnicode 6 0 R >>"),
        b("<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>"),
        stream("", format!("BT /F1 12 Tf <{codes}> Tj ET").as_bytes()),
        stream("", cmap.as_bytes()),
    ];
    let file = pdf(&objects, "");
    let b = budget();
    let doc = Document::open(&file, &b).unwrap();
    let pages = doc.pages(1).list;
    for content in [format!("BT /F1 12 Tf <{codes}> Tj ET"), format!("BT /F1 12 Tf [{}] TJ ET", "<01> -5 ".repeat(20_000))] {
        let mut fonts = Fonts::default();
        let mut reader = PageText::new(&doc, &mut fonts, &Limits::DEFAULT, 1000);
        reader.run(content.as_bytes(), &pages[0].resources);
        assert!(reader.full && reader.out.len() <= 1000 && reader.out.len() > 500, "{}", reader.out.len());
    }
    let out = read_with(&file, Limits { text_bytes: 100, ..Limits::DEFAULT });
    assert!(out.ends_with("\n[text stopped inside page 1 of 1: extracted text limit 0 MiB]\n"), "{out}");
}

/// Revision 1 of `simple` plus a hybrid update: the classic table marks
/// object 4 (the page) free, and its /XRefStm puts the new page 4 in object
/// stream 6; the new page shows NEW through content 7.
pub(super) fn hybrid_update() -> Vec<u8> {
    let mut file = pdf(&simple(HELVETICA, &[stream("", b"BT /F1 12 Tf (OLD) Tj ET")]), "");
    let prev = last_startxref(&file);
    let page = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 7 0 R >>";
    let at6 = file.len();
    file.extend_from_slice(b"6 0 obj\n");
    file.extend_from_slice(&stream("/Type /ObjStm /N 1 /First 4", format!("4 0 {page}").as_bytes()));
    file.extend_from_slice(b"\nendobj\n");
    let at7 = file.len();
    file.extend_from_slice(b"7 0 obj\n");
    file.extend_from_slice(&stream("", b"BT /F1 12 Tf (NEW) Tj ET"));
    file.extend_from_slice(b"\nendobj\n");
    // Rows for objects 4, 7 and 8: type (1 byte), field 2 (3 bytes), field 3 (1 byte).
    let at8 = file.len();
    let mut rows = vec![2, 0, 0, 6, 0];
    for at in [at7, at8] {
        rows.push(1);
        rows.extend_from_slice(&(at as u32).to_be_bytes()[1..]);
        rows.push(0);
    }
    file.extend_from_slice(b"8 0 obj\n");
    file.extend_from_slice(&stream("/Type /XRef /Size 9 /W [1 3 1] /Index [4 1 7 2]", &rows));
    file.extend_from_slice(b"\nendobj\n");
    let xref = file.len();
    file.extend_from_slice(format!("xref\n4 1\n0000000000 00001 f \n6 1\n{at6:010} 00000 n \ntrailer\n<< /Size 9 /Root 1 0 R /Prev {prev} /XRefStm {at8} >>\nstartxref\n{xref}\n%%EOF\n").as_bytes());
    file
}

pub(super) fn last_startxref(file: &[u8]) -> usize {
    let at = file.windows(9).rposition(|w| w == b"startxref").unwrap();
    std::str::from_utf8(&file[at + 10..]).unwrap().lines().next().unwrap().parse().unwrap()
}

/// Astra BLOCK 4: the /XRefStm entry replaces the same revision's free
/// entry; before, the free entry won and the recovery scan found OLD.
#[test]
fn hybrid_xrefstm_replaces_the_free_entry() {
    let out = text(&hybrid_update()).unwrap();
    assert!(out.contains("page 1 of 1\nNEW\n") && !out.contains("OLD"), "{out}");
}

/// A current free entry is a deleted object: the recovery scan must not
/// bring back its old content. An object stream later in the file also wins
/// over an older plain copy when the scan rebuilds a damaged table.
#[test]
fn current_free_entries_stay_deleted() {
    let mut file = pdf(&simple(HELVETICA, &[stream("", b"BT /F1 12 Tf (OLD) Tj ET")]), "");
    let prev = last_startxref(&file);
    let xref = file.len();
    file.extend_from_slice(format!("xref\n5 1\n0000000000 00001 f \ntrailer\n<< /Size 6 /Root 1 0 R /Prev {prev} >>\nstartxref\n{xref}\n%%EOF\n").as_bytes());
    let out = text(&file).unwrap();
    assert!(out.starts_with("no text layer, 1 pages:") && !out.contains("OLD"), "{out}");
    // No startxref: the reader scans the file for objects.
    let damaged: Vec<u8> = String::from_utf8_lossy(&hybrid_update()).replace("startxref", "startxreF").into_bytes();
    let out = text(&damaged).unwrap();
    assert!(out.contains("page 1 of 1\nNEW\n") && !out.contains("OLD"), "{out}");
}

/// Astra SHOULD-FIX 5: "/GS1 gs" sets the font through an ExtGState.
#[test]
fn extgstate_sets_the_font() {
    let objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        b("<< /Type /Page /Parent 2 0 R /Resources << /ExtGState << /GS1 << /Type /ExtGState /Font [5 0 R 12] >> /GS2 6 0 R >> >> /Contents 4 0 R >>"),
        stream("", b"BT /GS1 gs (Hello) Tj 0 -20 Td /GS2 gs (World) Tj ET"),
        b(HELVETICA),
        b("<< /Type /ExtGState /Font [5 0 R 10] >>"),
    ];
    let out = text(&pdf(&objects, "")).unwrap();
    assert!(out.contains("page 1 of 1\nHello\nWorld\n"), "{out}");
}

/// Astra SHOULD-FIX 6: blank pages up to the page cap must not hide a later
/// text page; the verdict covers only the pages read and keeps the note.
#[test]
fn page_cap_note_on_a_blank_read() {
    let mut pages = vec![stream("", b"0 0 m 1 1 l S"); super::MAX_PAGES];
    pages.push(stream("", b"BT /F1 12 Tf (late) Tj ET"));
    assert_eq!(
        text(&pdf(&simple(HELVETICA, &pages), "")).unwrap(),
        "no text layer in the first 2000 of 2001 pages: these pages have no extractable text (scanned images or text drawn as shapes); Vade does not run OCR.\n\n[stopped after page 2000 of 2001: page limit 2000]\n"
    );
}
