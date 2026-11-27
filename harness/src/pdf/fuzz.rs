//! Hostile-input tests. Each probe from the two reviews of this reader
//! (db7e382 and 858f7b4) is a fixture generated here. Each parser and
//! decoder entry has a test that runs its budget out. The total-work test
//! runs every fixture, and mutations of the small ones, under small limits:
//! once the budget stops, at most `LATE` more charges are made, so no loop
//! runs on after the stop.
use super::budget::{Budget, Exhausted, Kind};
use super::content::{Fonts, PageText};
use super::document::Document;
use super::filters::{ascii85, ascii_hex, lzw, predict, run_length};
use super::font::{CMap, Font};
use super::inflate::{self, inflate, zlib_stored};
use super::syntax::{CONTENT_ITEMS, Lexer};
use super::tests::{HELVETICA, NEVER, b, budget, form_chain, hybrid_update, pdf, read_with, simple, stream, text};
use super::{Limits, MAX_PAGES};
use std::sync::atomic::AtomicBool;

/// Charges refused after a stop, at most, in one read.
const LATE: usize = 64;

fn limited(kind: Kind, n: usize) -> Limits {
    let mut l = Limits::DEFAULT;
    match kind {
        Kind::Read => l.read_bytes = n,
        Kind::Decoded => l.stream_bytes = n,
        Kind::Objects => l.objects = n,
        Kind::Retained => l.retained_bytes = n,
        Kind::Tables => l.tables = n,
        Kind::FontEntries => l.font_entries = n,
        Kind::Steps => l.steps = n,
    }
    l
}

/// A read of `file` under `limits`, with the budget it used.
fn run(file: &[u8], limits: &Limits) -> (Result<String, String>, usize, Option<Exhausted>) {
    let budget = Budget::new(limits, &NEVER);
    let out = super::read_with(file, limits, &budget);
    (out, budget.late(), budget.stopped())
}

/// Catalog 1, pages 2, page 3 (resources `res`, content 4), then `rest` as
/// objects 5 and up.
fn one_page(res: &str, content: &[u8], rest: Vec<Vec<u8>>) -> Vec<u8> {
    let mut objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        b(&format!("<< /Type /Page /Parent 2 0 R /Resources << {res} >> /Contents 4 0 R >>")),
        stream("", content),
    ];
    objects.extend(rest);
    pdf(&objects, "")
}

/// A Type0 Identity-H font (object 5) with descendant 6 and ToUnicode 7;
/// `extra` objects follow as 8 and up.
fn cid_file(descendant: &str, to_unicode: Vec<u8>, content: &[u8], extra: Vec<Vec<u8>>) -> Vec<u8> {
    let mut rest = vec![
        b("<< /Type /Font /Subtype /Type0 /BaseFont /Test /Encoding /Identity-H /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>"),
        b(descendant),
        to_unicode,
    ];
    rest.extend(extra);
    one_page("/Font << /F1 5 0 R >>", content, rest)
}

const AB_CMAP: &[u8] = b"begincmap 1 begincodespacerange <0000> <FFFF> endcodespacerange 2 beginbfchar <0001> <0041> <0002> <0042> endbfchar endcmap";

// ---- Review db7e382 (r1) ----

/// r1 BLOCK 2: one /Contents array that repeats an unfiltered stream.
fn repeated_stream(copies: usize, size: usize) -> Vec<u8> {
    let refs = vec!["5 0 R"; copies].join(" ");
    let objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b("<< /Type /Pages /Kids [4 0 R] /Count 1 /Resources << /Font << /F1 3 0 R >> >> >>"),
        b(HELVETICA),
        b(&format!("<< /Type /Page /Parent 2 0 R /Contents [{refs} 6 0 R] >>")),
        stream("", &vec![b' '; size]),
        stream("", b"BT /F1 12 Tf (tail) Tj ET"),
    ];
    pdf(&objects, "")
}

/// r1 BLOCK 3: one Tj whose codes each map to 256 characters.
fn expanding_tj(codes: usize) -> Vec<u8> {
    let cmap = format!("begincmap 1 begincodespacerange <00> <FF> endcodespacerange 1 beginbfchar <01> <{}> endbfchar endcmap", "0041".repeat(256));
    one_page(
        "/Font << /F1 5 0 R >>",
        format!("BT /F1 12 Tf <{}> Tj ET", "01".repeat(codes)).as_bytes(),
        vec![b("<< /Type /Font /Subtype /Type1 /BaseFont /X /ToUnicode 6 0 R >>"), stream("", cmap.as_bytes())],
    )
}

#[test]
fn r1_block2_repeated_stream_stops_at_the_decode_budget() {
    let file = repeated_stream(600, 4096);
    assert!(text(&file).unwrap().contains("\ntail\n"));
    let (out, late, stop) = run(&file, &limited(Kind::Decoded, 100_000));
    let out = out.unwrap();
    assert!(!out.contains("tail") && out.contains("[work limit reached on page 1 of 1 (bytes decoded limit): the read stopped there]"), "{out}");
    assert!(late <= LATE && stop == Some(Exhausted::Limit(Kind::Decoded)), "{late} {stop:?}");
}

#[test]
fn r1_block3_expanding_tj_stops_at_the_text_limit() {
    let out = read_with(&expanding_tj(20_000), Limits { text_bytes: 1000, ..Limits::DEFAULT });
    assert!(out.len() < 1300 && out.ends_with("[text stopped inside page 1 of 1: extracted text limit 1000 bytes]\n"), "{out}");
}

// ---- Review 858f7b4 (r2) ----

/// r2 BLOCK 1: a /W array of `repeats` copies of the range `0 65535 500`.
/// Before, each copy inserted 65,536 widths: 2,000 copies made 131,072,000
/// inserts and took 1,963 ms after a cancel at 10 ms.
fn overlapping_widths(repeats: usize) -> Vec<u8> {
    let w = "0 65535 500 ".repeat(repeats);
    cid_file(&format!("<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Test /W [{w}] >>"), stream("", AB_CMAP), b"BT /F1 12 Tf <0001> Tj ET", vec![])
}

/// r2 BLOCK 1: a /W array whose entries all point at one shared width array.
fn shared_widths(repeats: usize, len: usize) -> Vec<u8> {
    let w = "1 8 0 R ".repeat(repeats);
    let widths = b(&format!("[{}]", "500 ".repeat(len)));
    cid_file(&format!("<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Test /W [{w}] >>"), stream("", AB_CMAP), b"BT /F1 12 Tf <0001> Tj ET", vec![widths])
}

#[test]
fn r2_block1_cid_width_ranges_are_one_entry_each() {
    let file = overlapping_widths(2000);
    let b = budget();
    let out = super::read_with(&file, &Limits::DEFAULT, &b).unwrap();
    assert!(out.contains("page 1 of 1\nA\n"), "{out}");
    // One entry per range and one per array member: not 65,536 per range.
    assert!(b.used(Kind::FontEntries) < 2000 * 4 + 1000, "{}", b.used(Kind::FontEntries));
    // A shared array is charged on every use, so the font-entry limit stops it.
    let out = run(&shared_widths(2000, 1000), &limited(Kind::FontEntries, 100_000));
    assert!(out.0.as_ref().unwrap().contains("(font entries limit): the read stopped there]") && out.1 <= LATE, "{out:?}");
    // Indirect members of a width list are resolved, not read as 0.
    let file = cid_file("<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Test /W [1 [8 0 R 8 0 R]] >>", stream("", AB_CMAP), b"BT /F1 12 Tf <0001> Tj 1 0 0 1 6 0 Tm <0002> Tj ET", vec![super::tests::b("500")]);
    assert!(text(&file).unwrap().contains("page 1 of 1\nAB\n"));
}

/// `n` empty fixed-Huffman blocks in a zlib wrapper: 10 bits each, no output.
fn empty_blocks(n: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(n * 10);
    for i in 0..n {
        bits.push(u8::from(i + 1 == n));
        bits.extend_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0, 0]);
    }
    let mut out = vec![0x78, 0x01];
    out.extend(bits.chunks(8).map(|c| c.iter().enumerate().fold(0u8, |a, (i, &bit)| a | bit << i)));
    out.extend_from_slice(&1u32.to_be_bytes());
    out
}

/// r2 BLOCK 2: a page that repeats a stream of 50,000 empty blocks 20
/// times. Before, it decoded every block after a cancel at 5 ms (1.29 s).
fn empty_flate_page() -> Vec<u8> {
    let objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b("<< /Type /Pages /Kids [4 0 R] /Count 1 /Resources << /Font << /F1 3 0 R >> >> >>"),
        b(HELVETICA),
        b(&format!("<< /Type /Page /Parent 2 0 R /Contents [{} 6 0 R] >>", vec!["5 0 R"; 20].join(" "))),
        stream("/Filter /FlateDecode", &empty_blocks(50_000)),
        stream("", b"BT /F1 12 Tf (tail) Tj ET"),
    ];
    pdf(&objects, "")
}

#[test]
fn r2_block2_empty_flate_blocks_are_charged() {
    let mut out = Vec::new();
    let b = budget();
    assert_eq!(inflate(&empty_blocks(50_000), 1 << 20, &mut out, &b), Ok(62_506));
    assert!(out.is_empty() && b.used(Kind::Steps) == 50_000, "{}", b.used(Kind::Steps));
    let file = empty_flate_page();
    assert!(text(&file).unwrap().contains("\ntail\n"));
    let (out, late, stop) = run(&file, &limited(Kind::Steps, 200_000));
    let out = out.unwrap();
    assert!(!out.contains("tail") && out.contains("(work steps limit): the read stopped there]"), "{out}");
    assert!(late <= LATE && stop == Some(Exhausted::Limit(Kind::Steps)));
}

/// r2 BLOCK 3: 32 page-tree kids that are streams with no /Length. Each
/// body runs to the one "endstream" at the end, so the bodies overlap and
/// hold the later object headers. Before, each body was copied and cached.
fn overlapping_stream_pages() -> Vec<u8> {
    let kids: Vec<String> = (3..35).map(|n| format!("{n} 0 R")).collect();
    let mut objects = vec![b("<< /Type /Catalog /Pages 2 0 R >>"), b(&format!("<< /Type /Pages /Kids [{}] /Count 32 >>", kids.join(" ")))];
    for k in 0..32 {
        let mut body = b("<< /Type /Page /Contents 35 0 R >>\nstream\n");
        body.extend(std::iter::repeat_n(b'x', 64 << 10));
        if k == 31 {
            body.extend_from_slice(b"\nendstream");
        }
        objects.push(body);
    }
    objects.push(stream("", b"BT (x) Tj ET"));
    pdf(&objects, "")
}

#[test]
fn r2_block3_stream_bodies_are_not_copied_or_cached() {
    let b = budget();
    assert_eq!(super::read_with(&overlapping_stream_pages(), &Limits::DEFAULT, &b).unwrap_err(), "PDF has no pages");
    // Only the parsed dictionaries are cached: no stream body bytes.
    assert!(b.used(Kind::Retained) < 64 << 10, "{}", b.used(Kind::Retained));
}

/// r2 BLOCK 4: one content-stream operand of 500,000 integers.
#[test]
fn r2_block4_container_operands_have_an_item_limit() {
    let content = format!("BT /F1 12 Tf (before) Tj ET [{}] pop BT /F1 12 Tf (after) Tj ET", "0 ".repeat(500_000));
    let b = budget();
    let out = super::read_with(&one_page("/Font << /F1 5 0 R >>", content.as_bytes(), vec![super::tests::b(HELVETICA)]), &Limits::DEFAULT, &b).unwrap();
    assert!(out.contains("page 1 of 1\nbefore\n") && out.contains("[work limit reached on page 1: some content there was not read]"), "{out}");
    assert!(b.used(Kind::Objects) < CONTENT_ITEMS + 1000, "{}", b.used(Kind::Objects));
}

/// r2 BLOCK 5: the hybrid update's classic table has an in-use entry for
/// the old page; its /XRefStm puts the new page in an object stream.
#[test]
fn r2_block5_xrefstm_wins_over_an_in_use_entry() {
    let file = hybrid_update();
    let old = file.windows(8).position(|w| w == b"\n4 0 obj").unwrap() + 1;
    let free = b"xref\n4 1\n0000000000 00001 f \n";
    let at = file.windows(free.len()).rposition(|w| w == free).unwrap();
    let mut in_use = file.clone();
    in_use[at..at + free.len()].copy_from_slice(format!("xref\n4 1\n{old:010} 00000 n \n").as_bytes());
    for f in [file, in_use] {
        let out = text(&f).unwrap();
        assert!(out.contains("page 1 of 1\nNEW\n") && !out.contains("OLD"), "{out}");
    }
}

/// r2 SHOULD-FIX 6: the generic adapter, which the native child adapter
/// uses, passes the call's cancel flag to the read callback.
#[test]
fn r2_should_fix6_workspace_tools_pass_the_cancel_flag() {
    use crate::agent::{ToolExecutor, ToolInvocation, WorkspaceTools};
    let flag = AtomicBool::new(false);
    let mut same = false;
    let mut tools = WorkspaceTools::new(
        |_: &str, _: usize, _: usize, cancel: &AtomicBool| {
            same = std::ptr::eq(cancel, &flag);
            Ok(String::new())
        },
        |_: &str| Ok(Vec::new()),
    );
    let call = crate::agent_wire::ToolCall { id: "pdf".into(), name: "workspace_read".into(), arguments: r#"{"path":"a.pdf"}"#.into() };
    let args = crate::json::parse(&call.arguments).unwrap();
    tools.execute(ToolInvocation { call: &call, arguments: &args, max_output_bytes: 4096 }, &flag).unwrap();
    drop(tools);
    assert!(same);
}

/// Inline image data that holds " EI " and then operators, before the real text.
fn inline_image(dict: &str, data: &[u8]) -> Vec<u8> {
    let mut content = format!("BI {dict} ID ").into_bytes();
    content.extend_from_slice(data);
    content.extend_from_slice(b"\nEI\nBT /F1 10 Tf (REAL) Tj ET");
    one_page("/Font << /F1 5 0 R >>", &content, vec![b(HELVETICA)])
}

/// r2 SHOULD-FIX 7: image bytes that look like " EI " plus operators are
/// skipped by the image's own size or encoding. Before: "FAKE REAL".
#[test]
fn r2_should_fix7_inline_image_ends_at_its_data_size() {
    let pixels = b" EI BT /F1 10 Tf (FAKE) Tj ET ";
    let hex: String = pixels.iter().map(|b| format!("{b:02x}")).collect();
    let cases = [
        inline_image(&format!("/W {} /H 1 /BPC 8 /CS /G", pixels.len()), pixels),
        inline_image(&format!("/W {} /H 1 /BPC 8 /CS /RGB", pixels.len() / 3), pixels),
        inline_image(&format!("/W {} /H 1 /BPC 8 /CS /G /F /Fl", pixels.len()), &zlib_stored(pixels)),
        inline_image(&format!("/W {} /H 1 /BPC 8 /CS /G /F /AHx", pixels.len()), format!("{hex}>").as_bytes()),
    ];
    for file in cases {
        let out = text(&file).unwrap();
        assert!(out.contains("page 1 of 1\nREAL\n") && !out.contains("FAKE"), "{out}");
    }
    // An encoding whose end is unknown: the rest of the stream is skipped and
    // the page is marked, never read as operators.
    let out = text(&inline_image("/W 30 /H 1 /BPC 1 /IM true /F /CCF", pixels)).unwrap();
    assert!(!out.contains("FAKE") && out.contains("[work limit reached on page 1:"), "{out}");
}

/// r2 SHOULD-FIX 8: a branch 70 levels deep before a shallow text page.
#[test]
fn r2_should_fix8_deep_branch_keeps_its_siblings() {
    let mut objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b("<< /Type /Pages /Kids [6 0 R 3 0 R] /Count 1 >>"),
        b("<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"),
        stream("", b"BT /F1 12 Tf (shallow) Tj ET"),
        b(HELVETICA),
    ];
    objects.extend((6..76).map(|n| b(&if n < 75 { format!("<< /Type /Pages /Kids [{} 0 R] >>", n + 1) } else { "<< /Type /Pages /Kids [] >>".to_string() })));
    let out = text(&pdf(&objects, "")).unwrap();
    assert!(out.contains("page 1 of 1\nshallow\n") && out.contains("[page tree: a limit skipped some branches"), "{out}");
}

/// Forms 5.. in a chain `depth` long; the last one shows "visible".
fn form_depth(depth: usize) -> Vec<u8> {
    let rest = (0..depth).map(|k| {
        let (res, body) = if k + 1 == depth { ("/Font << /F1 4 0 R >>".to_string(), "BT /F1 10 Tf (visible) Tj ET".to_string()) } else { (format!("/XObject << /N {} 0 R >>", 6 + k), "/N Do".to_string()) };
        stream(&format!("/Type /XObject /Subtype /Form /BBox [0 0 1 1] /Resources << {res} >>"), body.as_bytes())
    });
    // Forms are objects 5 and up; the page content comes after them.
    let content = 5 + depth;
    let mut objects = vec![
        b("<< /Type /Catalog /Pages 2 0 R >>"),
        b("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        b(&format!("<< /Type /Page /Parent 2 0 R /Resources << /XObject << /N 5 0 R >> >> /Contents {content} 0 R >>")),
        b(HELVETICA),
    ];
    objects.extend(rest);
    objects.push(stream("", b"/N Do"));
    pdf(&objects, "")
}

/// r2 SHOULD-FIX 9: text behind 9 nested forms is not called absent.
#[test]
fn r2_should_fix9_form_depth_marks_the_page() {
    assert!(text(&form_depth(8)).unwrap().contains("page 1 of 1\nvisible\n"));
    let out = text(&form_depth(9)).unwrap();
    assert!(out.starts_with("no text found in the content read, 1 pages: a work limit skipped some content") && out.contains("[work limit reached on page 1:"), "{out}");
}

/// r2 SHOULD-FIX 10: a ToUnicode CMap inherits its /UseCMap base.
#[test]
fn r2_should_fix10_tounicode_usecmap_is_applied() {
    let base = b"begincmap 1 begincodespacerange <0000> <FFFF> endcodespacerange 1 beginbfchar <0001> <0041> endbfchar endcmap";
    let derived = b"begincmap 1 begincodespacerange <0000> <FFFF> endcodespacerange 1 beginbfchar <0002> <0042> endbfchar endcmap";
    let descendant = "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Test >>";
    let file = cid_file(descendant, stream("/UseCMap 8 0 R", derived), b"BT /F1 12 Tf <00010002> Tj ET", vec![stream("", base)]);
    let out = text(&file).unwrap();
    assert!(out.contains("page 1 of 1\nAB\n") && !out.contains("left out"), "{out}");
    // A /UseCMap cycle stops at the base limit.
    let file = cid_file(descendant, stream("/UseCMap 7 0 R", derived), b"BT /F1 12 Tf <0002> Tj ET", vec![]);
    assert!(text(&file).unwrap().contains("page 1 of 1\nB\n"));
}

/// r2 SHOULD-FIX 11: "/Contents 5 0 R" after object 5 was reused as
/// generation 1 is a stale reference and reads as null.
#[test]
fn r2_should_fix11_stale_generation_reads_as_null() {
    let mut file = pdf(&simple(HELVETICA, &[stream("", b"BT /F1 12 Tf (OLD) Tj ET")]), "");
    let prev = super::tests::last_startxref(&file);
    let at = file.len();
    file.extend_from_slice(b"5 1 obj\n");
    file.extend_from_slice(&stream("", b"BT /F1 12 Tf (NEW_GENERATION) Tj ET"));
    file.extend_from_slice(b"\nendobj\n");
    let xref = file.len();
    file.extend_from_slice(format!("xref\n5 1\n{at:010} 00001 n \ntrailer\n<< /Size 6 /Root 1 0 R /Prev {prev} >>\nstartxref\n{xref}\n%%EOF\n").as_bytes());
    let out = text(&file).unwrap();
    assert!(out.starts_with("no text layer, 1 pages:") && !out.contains("NEW") && !out.contains("OLD"), "{out}");
}

/// r2 font probe: 1,000 code-space ranges and 100,000 glyphs. Before, each
/// glyph checked every range. The merged code space is one binary search.
#[test]
fn r2_many_code_space_ranges_stay_cheap() {
    let cmap = format!("begincmap {} begincodespacerange {} <0001> <0001> endcodespacerange 1 beginbfchar <0001> <0041> endbfchar endcmap", 1001, "<0000> <0000> ".repeat(1000));
    let content = format!("BT /F1 12 Tf <{}> Tj ET", "0001".repeat(100_000));
    let b = budget();
    let out = super::read_with(&cid_file("<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Test >>", stream("", cmap.as_bytes()), content.as_bytes(), vec![]), &Limits::DEFAULT, &b).unwrap();
    assert!(out.contains(&format!("\n{}\n", "A".repeat(100_000))));
    assert!(b.used(Kind::Steps) < 200_000, "{}", b.used(Kind::Steps));
}

// ---- One budget-exhaustion test per parser and decoder entry ----

#[test]
fn lexer_charges_bytes_objects_and_depth() {
    let data = b"[1 2 3 [4 5] << /A (string) /B <414243> >> name]";
    let full = budget();
    assert!(Lexer::new(data, 0, &full).object(true).is_some());
    for (kind, n) in [(Kind::Read, 10), (Kind::Objects, 3)] {
        let b = Budget::new(&limited(kind, n), &NEVER);
        assert!(Lexer::new(data, 0, &b).object(true).is_none());
        assert_eq!(b.stopped(), Some(Exhausted::Limit(kind)));
        assert!(b.late() <= 4, "{kind:?} {}", b.late());
    }
    // Depth past the limit skips that object alone; the budget goes on.
    let deep = format!("{}1{}", "[".repeat(100), "]".repeat(100));
    let b = budget();
    assert!(Lexer::new(deep.as_bytes(), 0, &b).object(true).is_none());
    assert_eq!((b.stopped(), Lexer::new(data, 0, &b).object(true).is_some()), (None, true));
    // An operand past the item limit is dropped and marked.
    let wide = format!("[{}]", "0 ".repeat(CONTENT_ITEMS + 1));
    let mut lex = Lexer::with_items(wide.as_bytes(), 0, &b, CONTENT_ITEMS);
    assert!(lex.object(false).is_none() && lex.over);
}

#[test]
fn inflate_charges_blocks_tables_and_output() {
    let dynamic = super::tests::gzip_body("1f8b08000000000002039dd53b4e43311484e19e559c121af079f8d5460a554a6f20527c13102248dcfd0baadc115598c2e5b8f8f4cbde0d7979555193b1483529adc838c9e3e1ed738aca7591f532e534978fe33a659ddfab7c1dcff3f949c6bbecc7c3eeeffef7dcf646ec73dbf64eec23b67d107b4fdb3e137b05bf42ec13f8d5ffef7303bf46ec2bf875625fc04f13714160804481d941508904b302a1120de604864a44181d11890aa3222291611444243a8c404422c47044244a0c0344234a8c04884694e81ddf42a244af806844895e00d188123d232251a23b221225ba212251a22744244ab48e884489d600d18912ad00a213255a0644274a34c76f9928d10c109d28d114118912b5232251a23644244ad4828844899a119128510310e38e127f007056a0471f0a0000");
    for (input, kind, n) in [(empty_blocks(1000), Kind::Steps, 100), (dynamic.clone(), Kind::Tables, 2), (dynamic, Kind::Decoded, 100), (zlib_stored(&[7; 500]), Kind::Decoded, 100)] {
        let b = Budget::new(&limited(kind, n), &NEVER);
        let mut out = Vec::new();
        assert_eq!(inflate(&input, 1 << 20, &mut out, &b), Err(inflate::Error::Budget(Exhausted::Limit(kind))));
        assert!(out.len() <= 100 && b.late() == 0, "{kind:?}");
    }
}

#[test]
fn filters_charge_before_each_byte() {
    let b = Budget::new(&limited(Kind::Decoded, 10), &NEVER);
    assert!(ascii_hex(&[b'6'; 100], usize::MAX, &b).is_err());
    let b = Budget::new(&limited(Kind::Decoded, 10), &NEVER);
    assert!(ascii85(b"zzzzzzzz~>", usize::MAX, &b).is_err());
    let b = Budget::new(&limited(Kind::Decoded, 10), &NEVER);
    assert!(run_length(b"\x81r\x81r\x80", usize::MAX, &b).is_err());
    // 20 literal 9-bit codes, most significant bit first: 23 bytes.
    let codes: Vec<u8> = std::iter::repeat_n(u32::from(b'x'), 20).flat_map(|c| (0..9).rev().map(move |k| (c >> k & 1) as u8)).collect::<Vec<u8>>().chunks(8).map(|c| c.iter().enumerate().fold(0, |a, (i, &bit)| a | bit << (7 - i))).collect();
    let b = Budget::new(&limited(Kind::Read, 10), &NEVER);
    assert!(lzw(&codes, true, usize::MAX, &b).is_err());
    let b = Budget::new(&limited(Kind::Decoded, 10), &NEVER);
    let parms = super::syntax::Dict::from([(b"Predictor".to_vec(), super::syntax::Object::Int(12))]);
    assert!(predict(vec![2; 100], Some(&parms), &b).is_err());
    assert_eq!(b.stopped(), Some(Exhausted::Limit(Kind::Decoded)));
}

#[test]
fn document_open_and_lookups_charge_the_budget() {
    let file = pdf(&simple(HELVETICA, &[stream("", b"BT /F1 12 Tf (x) Tj ET")]), "");
    for kind in [Kind::Read, Kind::Tables, Kind::Retained, Kind::Steps, Kind::Objects] {
        let n = if kind == Kind::Tables { 0 } else { 3 };
        let b = Budget::new(&limited(kind, n), &NEVER);
        assert!(Document::open(&file, &b).is_err(), "{kind:?}");
        assert_eq!(b.stopped(), Some(Exhausted::Limit(kind)));
        assert!(b.late() <= LATE, "{kind:?} {}", b.late());
        let (out, late, _) = run(&file, &limited(kind, n));
        assert!(out.unwrap_err().starts_with("PDF work limit reached while reading the file structure:") && late <= LATE);
    }
    // The page walk stops at its step budget and says the tree is incomplete.
    let pages = vec![stream("", b"BT /F1 12 Tf (p) Tj ET"); 50];
    let file = pdf(&simple(HELVETICA, &pages), "");
    let opened = budget();
    Document::open(&file, &opened).unwrap();
    let b = Budget::new(&limited(Kind::Steps, opened.used(Kind::Steps) + 20), &NEVER);
    let doc = Document::open(&file, &b).unwrap();
    let tree = doc.pages(MAX_PAGES);
    assert!(tree.incomplete && tree.total < 50 && b.stopped() == Some(Exhausted::Limit(Kind::Steps)), "{}", tree.total);
}

#[test]
fn cmap_and_font_loads_charge_font_entries() {
    let b = Budget::new(&limited(Kind::FontEntries, 2), &NEVER);
    assert!(CMap::parse(AB_CMAP, &b).is_none());
    assert_eq!(b.stopped(), Some(Exhausted::Limit(Kind::FontEntries)));
    let file = overlapping_widths(50);
    let b = Budget::new(&limited(Kind::FontEntries, 20), &NEVER);
    let doc = Document::open(&file, &b).unwrap();
    let font = doc.get(5, 0).unwrap();
    assert!(Font::load(&doc, font.as_dict().unwrap()).is_none());
    assert_eq!(b.stopped(), Some(Exhausted::Limit(Kind::FontEntries)));
}

#[test]
fn page_text_charges_operators_and_glyphs() {
    let file = pdf(&simple(HELVETICA, &[stream("", b"BT /F1 12 Tf (x) Tj ET")]), "");
    let b = Budget::new(&limited(Kind::Steps, 5_000), &NEVER);
    let doc = Document::open(&file, &b).unwrap();
    let pages = doc.pages(1).list;
    let mut fonts = Fonts::default();
    let mut reader = PageText::new(&doc, &mut fonts, &Limits::DEFAULT, usize::MAX);
    reader.run("BT /F1 12 Tf (abc) Tj ET ".repeat(10_000).as_bytes(), &pages[0].resources);
    assert!(reader.out.len() < 20_000 && b.stopped() == Some(Exhausted::Limit(Kind::Steps)) && b.late() <= 4, "{} {}", reader.out.len(), b.late());
}

/// Every whole-read limit, reached on a file that uses every parser: the
/// read stops, names the limit, and makes few charges after the stop.
#[test]
fn each_whole_read_limit_is_named() {
    let file = form_chain(4, "BT /F1 1 Tf (leaf) Tj ET");
    let full = budget();
    super::read_with(&file, &Limits::DEFAULT, &full).unwrap();
    for kind in [Kind::Read, Kind::Decoded, Kind::Objects, Kind::Retained, Kind::Tables, Kind::FontEntries, Kind::Steps] {
        // Half of what the whole read needs: the stop falls inside it.
        let (out, late, stop) = run(&file, &limited(kind, full.used(kind) / 2));
        let what = Exhausted::Limit(kind).what();
        let out = out.unwrap_or_else(|e| e);
        assert!(out.contains(&format!("{what} limit")) && stop == Some(Exhausted::Limit(kind)) && late <= LATE, "{kind:?} {late} {out}");
    }
}

// ---- Total work ----

/// Every fixture above, at sizes that keep a debug test fast.
fn corpus() -> Vec<Vec<u8>> {
    vec![
        form_chain(64, ""),
        repeated_stream(64, 1024),
        expanding_tj(2_000),
        hybrid_update(),
        overlapping_widths(200),
        shared_widths(200, 100),
        empty_flate_page(),
        overlapping_stream_pages(),
        one_page("/Font << /F1 5 0 R >>", format!("[{}] pop (x) Tj", "0 ".repeat(100_000)).as_bytes(), vec![b(HELVETICA)]),
        inline_image("/W 30 /H 1 /BPC 8 /CS /G", b" EI BT /F1 10 Tf (FAKE) Tj ET "),
        form_depth(9),
        cid_file("<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Test >>", stream("/UseCMap 7 0 R", AB_CMAP), b"BT /F1 12 Tf <0001> Tj ET", vec![]),
    ]
}

/// The total-work bound: under any limits, a read of any input charges at
/// most the limits, and once the budget stops it refuses at most `LATE`
/// more charges before it returns. With the rule that every loop iteration
/// and allocation charges first, this bounds the work of every read. The
/// same input and limits also give the same output.
#[test]
fn total_work_is_bounded_for_every_fixture_and_mutation() {
    let small = Limits { read_bytes: 1 << 20, stream_bytes: 1 << 20, objects: 1 << 15, retained_bytes: 1 << 20, tables: 512, font_entries: 1 << 14, steps: 1 << 16, ..Limits::DEFAULT };
    let kinds = [Kind::Read, Kind::Decoded, Kind::Objects, Kind::Retained, Kind::Tables, Kind::FontEntries, Kind::Steps];
    let limit = |l: &Limits, k: Kind| [l.read_bytes, l.stream_bytes, l.objects, l.retained_bytes, l.tables, l.font_entries, l.steps][k as usize];
    let check = |file: &[u8], limits: &Limits| {
        let b = Budget::new(limits, &NEVER);
        let out = super::read_with(file, limits, &b);
        assert!(b.late() <= LATE, "late {}", b.late());
        for k in kinds {
            assert!(b.used(k) <= limit(limits, k), "{k:?}");
        }
        out
    };
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for file in corpus() {
        for limits in [Limits::DEFAULT, small] {
            assert_eq!(check(&file, &limits), check(&file, &limits));
        }
        if file.len() > 64 << 10 {
            continue;
        }
        for round in 0..200 {
            let mut bad = file.clone();
            for _ in 0..1 + round % 8 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let i = (seed % bad.len() as u64) as usize;
                bad[i] = (seed >> 32) as u8;
            }
            let _ = check(&bad, &small);
        }
    }
}
