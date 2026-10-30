//! Tests for the readers audit of 2026-10-30: failure modes, archives,
//! repeat runs and ODT numbered lists. The fixtures come from
//! [`office_tests`](super::office_tests).
use super::office_tests::{Entry, RELS, SALES, ap, deflate_bomb, deflate_fixed, deflate_run, deflated, docx, docx_with, epub, odf, pptx, stored, xlsx, zip_file, zip_with};
use super::zip::crc32;
use super::*;

/// The failure modes of the readers audit of 2026-10-30, each through
/// [`read`]: an encrypted or scanned PDF gives one line; a huge sheet gives
/// its first rows and a warning; malformed HTML gives its text; bytes that
/// are not UTF-8 decode to the end of the file. The compression bomb is in
/// `archives_list_members_and_stop_bombs`.
#[test]
fn audit_failure_modes_give_one_line_or_a_warning() {
    // Encrypted PDF: the trailer names an /Encrypt dictionary.
    let plain = crate::pdf::tests::sample(&["secret"]);
    let at = plain.windows(7).rposition(|w| w == b"trailer").unwrap();
    let mut locked = plain[..at + 7].to_vec();
    locked.extend_from_slice(b"\n<< /Encrypt << /Filter /Standard /V 2 /R 3 >> /ID [<01> <01>]");
    let rest = &plain[at + 7..];
    let open = rest.windows(2).position(|w| w == b"<<").unwrap();
    locked.extend_from_slice(&rest[open + 2..]);
    assert_eq!(read(&locked, "a.pdf", &Caps::default()).unwrap_err().to_string(), "this PDF file is encrypted; Vade cannot read encrypted files");
    // Scanned PDF: pages with no text layer.
    let scanned = crate::pdf::tests::sample(&["", ""]);
    let err = read(&scanned, "scan.pdf", &Caps::default()).unwrap_err();
    assert_eq!(err.to_string(), "this PDF file has no text layer (2 pages; scanned images?); Vade does not read images yet");
    // A huge sheet stops at the row cap at the default caps.
    let rows: String = (1..=10_050).map(|r| format!("<row r=\"{r}\"><c r=\"A{r}\"><v>{r}</v></c></row>")).collect();
    let (_, doc) = read(&xlsx(&rows, ""), "big.xlsx", &Caps::default()).unwrap();
    // Row 1 is the header; 10,000 rows follow it.
    let last = doc.markdown.lines().filter(|l| l.starts_with("| ")).last().unwrap().to_owned();
    assert_eq!(last, "| 10001 |");
    assert_eq!(doc.warnings, ["rows after 10000 of 10049 in sheet Sales & costs (row cap)"]);
    // Malformed HTML: nothing closes, tags are cut, and quotes never end.
    for html in ["<div><table><tr><td><ul><li><a href=\"x\"><b>text", "<p>a<b>b<i>c</p>d</b>e</i><li>f<td>g<", "<p title=\"never>text", "</p></li></ul></table>text<"] {
        let (_, doc) = read(html.as_bytes(), "a.html", &Caps::default()).unwrap();
        assert!(doc.markdown.chars().count() < 80 && doc.warnings.is_empty(), "{html}: {doc:?}");
    }
    assert_eq!(read(b"<p>a<b>b<i>c</p>d", "a.html", &Caps::default()).unwrap().1.markdown, "a**b*c\n\nd\n");
    // Bytes that are not UTF-8 decode as Windows-1252 to the end of the file:
    // an undefined byte after the first 64 KiB stays a C1 control, and a NUL
    // byte there makes the file binary.
    let mut late = vec![b'a'; 70_000];
    late.extend_from_slice(b" \x81\x93q\x94");
    let (_, doc) = read(&late, "a.txt", &Caps::default()).unwrap();
    assert!(doc.markdown.ends_with(" \u{81}\u{201c}q\u{201d}"), "{:?}", &doc.markdown[doc.markdown.len() - 8..]);
    assert_eq!(doc.warnings, ["decoded as Windows-1252 (Latin-1); the file is not UTF-8"]);
    late.push(0);
    assert_eq!(read(&late, "a.txt", &Caps::default()).unwrap_err(), ReadError::Binary);
}

/// One tar entry: a ustar header for `name` with the type flag `kind`, then
/// `data` padded to 512 bytes.
fn tar_entry(name: &str, kind: u8, data: &[u8]) -> Vec<u8> {
    let mut h = vec![0u8; 512];
    h[..name.len()].copy_from_slice(name.as_bytes());
    h[100..108].copy_from_slice(b"0000644\0");
    h[124..136].copy_from_slice(format!("{:011o}\0", data.len()).as_bytes());
    h[136..148].copy_from_slice(b"00000000000\0");
    h[156] = kind;
    h[257..263].copy_from_slice(b"ustar\0");
    h[263..265].copy_from_slice(b"00");
    h[148..156].copy_from_slice(b"        ");
    let sum: u32 = h.iter().map(|&b| u32::from(b)).sum();
    h[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
    h.extend_from_slice(data);
    h.resize(h.len().div_ceil(512) * 512, 0);
    h
}

/// Gzip data for `data` with the file name `name` in the header.
fn gzip(name: &str, data: &[u8]) -> Vec<u8> {
    let mut out = b"\x1f\x8b\x08\x08\0\0\0\0\0\x03".to_vec();
    out.extend_from_slice(name.as_bytes());
    out.push(0);
    out.extend_from_slice(&deflate_fixed(data));
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// The ZIP fixture of the archive tests: a folder, two files, a symbolic
/// link, an encrypted entry and a ZIP inside that holds one more ZIP.
fn bundle() -> Vec<u8> {
    let deep = zip_file(&[stored("c.txt", "deep text")]);
    let inner = zip_file(&[stored("b.md", "# Inner\n"), Entry { name: "deep.zip", data: deep, deflate: false, raw: None }]);
    zip_with(
        &[
            stored("docs/", ""),
            stored("docs/a.txt", "alpha"),
            deflated("docs/t.csv", "x,y\n1,2\n"),
            stored("link", "docs/a.txt"),
            stored("secret.txt", "hidden"),
            Entry { name: "inner.zip", data: inner, deflate: false, raw: None },
        ],
        |name, record| match name {
            // Made on Unix (3), mode 0o120777: a symbolic link.
            "link" => {
                record[5] = 3;
                record[38..42].copy_from_slice(&(0o120777u32 << 16).to_le_bytes());
            }
            "secret.txt" => record[8] |= 1,
            _ => {}
        },
    )
}

/// The tar fixture: a folder, a file, a symbolic link and a GNU long name.
fn tarball() -> Vec<u8> {
    let mut tar = tar_entry("notes/", b'5', b"");
    tar.extend(tar_entry("notes/a.txt", b'0', b"tar text\n"));
    tar.extend(tar_entry("notes/link", b'2', b""));
    tar.extend(tar_entry("././@LongLink", b'L', b"notes/a/very/long/path/name.json\0"));
    tar.extend(tar_entry("notes/short", b'0', b"{\"a\":1}"));
    tar.extend(vec![0; 1024]);
    tar
}

/// Readers audit item 4: ZIP, tar and gzip archives list their entries with
/// sizes, then read each file with the reader for its format after a file
/// marker. Links and encrypted entries are listed, not read; an archive
/// three levels deep is not opened; the members share one bytes cap, so a
/// compression bomb stops at it.
#[test]
fn archives_list_members_and_stop_bombs() {
    let out = read_markdown(&bundle(), "bundle.zip", &Caps::default()).unwrap();
    let want = "<!-- vade read: bundle.zip | ZIP archive | pages 3 | chars 817 | skipped: inner.zip/deep.zip (archive past 2 levels; depth cap) -->
6 entries, 352 bytes when unpacked.

| # | Path | Size (bytes) | Type |
| --- | --- | --- | --- |
| 1 | docs/ | 0 | folder |
| 2 | docs/a.txt | 5 | file |
| 3 | docs/t.csv | 8 | file |
| 4 | link | 10 | symbolic link (not followed) |
| 5 | secret.txt | 6 | encrypted (not read) |
| 6 | inner.zip | 323 | file |

<!-- file 1 of 3 -->
## File docs/a.txt

alpha

<!-- file 2 of 3 -->
## File docs/t.csv

| x | y |
| --- | --- |
| 1 | 2 |

<!-- file 3 of 3 -->
## File inner.zip

2 entries, 125 bytes when unpacked.

| # | Path | Size (bytes) | Type |
| --- | --- | --- | --- |
| 1 | inner.zip/b.md | 8 | file |
| 2 | inner.zip/deep.zip | 117 | file |

<!-- file 1 of 2 -->
## File inner.zip/b.md

# Inner

<!-- file 2 of 2 -->
## File inner.zip/deep.zip

(not opened: an archive inside an archive is read 2 levels deep)

<!-- vade read: skipped: inner.zip/deep.zip (archive past 2 levels; depth cap) -->
";
    assert_eq!(out, want);
    let tar = tarball();
    let listing = "4 entries, 16 bytes when unpacked.

| # | Path | Size (bytes) | Type |
| --- | --- | --- | --- |
| 1 | notes/ | 0 | folder |
| 2 | notes/a.txt | 9 | file |
| 3 | notes/link | 0 | symbolic link (not followed) |
| 4 | notes/a/very/long/path/name.json | 7 | file |

<!-- file 1 of 2 -->
## File notes/a.txt

tar text

<!-- file 2 of 2 -->
## File notes/a/very/long/path/name.json

```json
{
  \"a\": 1
}
```
";
    for (bytes, name, format) in [(tar.clone(), "t.tar", "tar archive"), (gzip("t.tar", &tar), "t.tgz", "gzip archive"), (gzip("t.tar", &tar), "", "gzip archive")] {
        let out = read_markdown(&bytes, name, &Caps::default()).unwrap();
        assert_eq!(out, format!("<!-- vade read: {name} | {format} | pages 2 | chars 403 | skipped: none -->\n{listing}"), "{name}");
    }
    // A ustar file under any name is a tar archive; an old one needs the name.
    assert_eq!(sniff(&tar, "notes.txt").map(|(f, _)| f), Some(Format::Archive("tar archive")));
    let one = read(&gzip("notes.txt", b"one file\n"), "notes.txt.gz", &Caps::default()).unwrap().1;
    assert_eq!(one.markdown, "1 entry, 9 bytes when unpacked.\n\n| # | Path | Size (bytes) | Type |\n| --- | --- | --- | --- |\n| 1 | notes.txt | 9 | file |\n\n<!-- file 1 of 1 -->\n## File notes.txt\n\none file\n");
    // Bombs: a member larger than the bytes left is not decoded, one that
    // holds more than its declared size is damaged, and gzip data stops at
    // the cap. The other members still read.
    let small = Caps { max_bytes: 100_000, ..Caps::default() };
    let n = 4_000;
    let big = Entry { name: "big.txt", data: vec![b'a'; 1 + 258 * n], deflate: true, raw: Some(deflate_bomb(n)) };
    let liar = Entry { name: "liar.txt", data: vec![b'a'; 1000], deflate: true, raw: Some(deflate_bomb(n)) };
    let bombs = zip_file(&[big, liar, stored("ok.txt", "fine")]);
    assert!(bombs.len() < 20_000, "{}", bombs.len());
    let (_, doc) = read(&bombs, "bomb.zip", &small).unwrap();
    assert!(doc.markdown.contains("## File big.txt\n\n(not read: the member has 1032001 bytes, more than the 100000 bytes left of the archive bytes cap)\n"), "{}", doc.markdown);
    assert!(doc.markdown.contains("## File liar.txt\n\n(not read: damaged ZIP archive file: the entry liar.txt holds more data than its size; no text was read)\n"));
    assert!(doc.markdown.ends_with("## File ok.txt\n\nfine\n"));
    assert_eq!(doc.warnings, ["big.txt (1032001 bytes; archive bytes cap)"]);
    let mut gzbomb = b"\x1f\x8b\x08\x08\0\0\0\0\0\x03a.txt\0".to_vec();
    gzbomb.extend(deflate_bomb(n));
    gzbomb.extend([0; 8]);
    let (_, doc) = read(&gzbomb, "a.txt.gz", &small).unwrap();
    assert!(doc.markdown.starts_with("1 entry, 100000 bytes when unpacked.\n"), "{}", &doc.markdown[..80]);
    assert_eq!(doc.warnings, ["a.txt after 100000 bytes (archive bytes cap)"]);
    // Damaged archives and archives Vade has no decoder for give one line.
    let mut bad = tar.clone();
    bad[148] ^= 1;
    assert_eq!(read(&bad, "t.tar", &Caps::default()).unwrap_err().to_string(), "damaged tar archive file: the header at byte 0 fails its checksum; no text was read");
    for (magic, line) in [(&b"\xfd7zXZ\0rest"[..], "xz archive"), (b"\x28\xb5\x2f\xfdrest", "zstd archive"), (b"BZh91AY&SYrest", "bzip2 archive"), (b"7z\xbc\xaf\x27\x1crest", "7z archive")] {
        assert_eq!(read(magic, "a.bin", &Caps::default()).unwrap_err().to_string(), format!("Vade does not read {line} files; no text was read"));
    }
    // A small char cap stops the listing and the members with the warning.
    let (_, doc) = read(&bundle(), "bundle.zip", &Caps { max_chars: 100, ..Caps::default() }).unwrap();
    assert_eq!(doc.chars, 100);
    assert!(doc.warnings.contains(&"text after 100 chars (char cap)".to_owned()), "{:?}", doc.warnings);
    for cut in (0..tar.len()).step_by(97) {
        let _ = read(&tar[..cut], "t.tar", &Caps::default());
        let _ = read(&gzip("t.tar", &tar)[..cut.min(200)], "t.tgz", &Caps::default());
    }
}

/// Readers audit item 3: each reader gives the same bytes on every run. The
/// readers keep no clock or environment value, and the maps they hash are
/// for lookups only; each run here (some on a new thread) has new hash keys.
#[test]
fn every_reader_gives_the_same_bytes_every_run() {
    let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
    png.extend_from_slice(&[0, 0, 2, 128, 0, 0, 1, 224, 8, 6, 0, 0, 0, 1, 2, 3, 4]);
    let docx_body = "<w:p><w:pPr><w:pStyle w:val=\"MyHead\"/></w:pPr><w:r><w:t>Head</w:t></w:r></w:p>\
        <w:p><w:pPr><w:pStyle w:val=\"ListBullet\"/></w:pPr><w:r><w:t>item</w:t></w:r></w:p>\
        <w:p><w:pPr><w:numPr><w:ilvl w:val=\"1\"/><w:numId w:val=\"2\"/></w:numPr></w:pPr><w:r><w:t>sub</w:t></w:r></w:p>";
    let odt = "<text:h text:outline-level=\"1\">T</text:h><text:list><text:list-item><text:p>a</text:p></text:list-item></text:list>";
    let ods = "<table:table table:name=\"S\"><table:table-row><table:table-cell><text:p>1</text:p></table:table-cell></table:table-row></table:table>";
    let files: Vec<(Vec<u8>, &str)> = vec![
        (b"<h1>T</h1><ul><li><p>a</p></li></ul><table><tr><td colspan=2>x</td></tr></table>".to_vec(), "a.html"),
        (b"a,b\n1,2\n".to_vec(), "a.csv"),
        (br"{\rtf1\ansi caf\'e9}".to_vec(), "a.rtf"),
        (b"{\"b\":1,\"a\":[2,3]}".to_vec(), "a.json"),
        (b"caf\xe9\n".to_vec(), "a.txt"),
        (b"fn main() {}\n".to_vec(), "a.rs"),
        (docx(docx_body), "a.docx"),
        (xlsx(SALES, ""), "a.xlsx"),
        (pptx(&ap(0, "Hello")), "a.pptx"),
        (epub(None), "a.epub"),
        (odf("text", odt), "a.odt"),
        (odf("spreadsheet", ods), "a.ods"),
        (crate::pdf::tests::sample(&["one", "two"]), "a.pdf"),
        (png, "a.png"),
        (bundle(), "a.zip"),
        (gzip("t.tar", &tarball()), "a.tgz"),
    ];
    for (bytes, name) in &files {
        let first = read_markdown(bytes, name, &Caps::default()).unwrap();
        for run in 0..6 {
            let again = if run % 2 == 0 {
                read_markdown(bytes, name, &Caps::default()).unwrap()
            } else {
                let (bytes, name) = (bytes.clone(), name.to_string());
                std::thread::spawn(move || read_markdown(&bytes, &name, &Caps::default()).unwrap()).join().unwrap()
            };
            assert_eq!(again, first, "{name} run {run}");
        }
    }
}

/// Readers audit item 5 (ODF): ODT numbered lists show their numbers, from
/// the list styles in `content.xml` and `styles.xml`. A nested list with no
/// style takes its parent's style at its own level; `start-value` and
/// `continue-numbering` set the numbers; a bullet level stays a bullet.
#[test]
fn odt_numbered_lists_follow_their_list_styles() {
    let item = |text: &str| format!("<text:list-item><text:p>{text}</text:p></text:list-item>");
    let styles = "<office:document-styles xmlns:office=\"o\" xmlns:style=\"s\" xmlns:text=\"t\"><office:styles>\
        <text:list-style style:name=\"Numbering_20_123\"><text:list-level-style-number text:level=\"1\" style:num-format=\"1\" text:start-value=\"3\"/></text:list-style>\
        </office:styles></office:document-styles>";
    let content = format!(
        "<office:document-content xmlns:office=\"o\" xmlns:style=\"s\" xmlns:text=\"t\"><office:automatic-styles>\
        <text:list-style style:name=\"L1\"><text:list-level-style-number text:level=\"1\" style:num-format=\"1\"/><text:list-level-style-bullet text:level=\"2\" text:bullet-char=\"*\"/>\
        <text:list-level-style-number text:level=\"3\" style:num-format=\"a\"/></text:list-style>\
        <text:list-style style:name=\"L2\"><text:list-level-style-number text:level=\"1\" style:num-format=\"\"/></text:list-style>\
        </office:automatic-styles><office:body><office:text>\
        <text:list text:style-name=\"L1\">{}<text:list-item><text:p>two</text:p><text:list>{}<text:list-item><text:p>b</text:p><text:list>{}{}</text:list></text:list-item></text:list></text:list-item></text:list>\
        <text:p>between</text:p>\
        <text:list text:style-name=\"L1\" text:continue-numbering=\"true\">{}</text:list>\
        <text:list text:style-name=\"L1\"><text:list-item text:start-value=\"7\"><text:p>seven</text:p></text:list-item>{}</text:list>\
        <text:list text:style-name=\"Numbering_20_123\">{}</text:list>\
        <text:list text:style-name=\"L2\">{}</text:list><text:list text:style-name=\"Missing\">{}</text:list>\
        </office:text></office:body></office:document-content>",
        item("one"), item("a"), item("x"), item("y"), item("three"), item("eight"), item("named"), item("plain"), item("unknown")
    );
    let file = zip_file(&[
        stored("mimetype", "application/vnd.oasis.opendocument.text"),
        deflated("styles.xml", styles),
        deflated("content.xml", &content),
    ]);
    let (_, doc) = read(&file, "a.odt", &Caps::default()).unwrap();
    assert_eq!(
        doc.markdown,
        "<!-- page 1 of 1 -->\n## Page 1\n\n1. one\n2. two\n    - a\n    - b\n        1. x\n        2. y\n\nbetween\n\n3. three\n\n7. seven\n8. eight\n\n3. named\n\n- plain\n\n- unknown\n"
    );
}

/// Astra review of 4a72078, finding 2: each list finds its style in a
/// binary search, so 2,000 list styles and 20,000 lists read in well under
/// one second with at most 13 name compares for each list (a scan of all
/// styles for each list does 2,000 and fails the step bound).
#[test]
fn many_odt_list_styles_read_fast() {
    let (n, lists) = (2_000, 20_000);
    let pad = "x".repeat(40);
    let styles: String = (0..n)
        .map(|i| format!("<text:list-style style:name=\"{pad}{i}\"><text:list-level-style-number text:level=\"1\" style:num-format=\"1\" text:start-value=\"5\"/></text:list-style>"))
        .collect();
    let empty: String = (0..lists).map(|i| format!("<text:list text:style-name=\"{pad}{i}-\"/>")).collect();
    let content = format!(
        "<office:document-content xmlns:office=\"o\" xmlns:style=\"s\" xmlns:text=\"t\"><office:automatic-styles>{styles}</office:automatic-styles>\
        <office:body><office:text>{empty}<text:list text:style-name=\"{pad}{}\"><text:list-item><text:p>last</text:p></text:list-item></text:list></office:text></office:body></office:document-content>",
        n - 1
    );
    let file = zip_file(&[stored("mimetype", "application/vnd.oasis.opendocument.text"), deflated("content.xml", &content)]);
    let start = std::time::Instant::now();
    SCANNED.with(|s| s.set(0));
    let (_, doc) = read(&file, "a.odt", &Caps::default()).unwrap();
    let steps = SCANNED.with(|s| s.get());
    assert!(start.elapsed() < std::time::Duration::from_secs(1), "{:?}", start.elapsed());
    // The unzip and XML scans count each byte a few times.
    assert!(steps <= 4 * content.len() + 13 * (lists + 1), "{steps} for {}", content.len());
    assert_eq!(doc.markdown, "<!-- page 1 of 1 -->\n## Page 1\n\n5. last\n");
}

/// Astra review of 4a72078, finding 3: a damaged styles.xml does not refuse
/// the file; its lists read as bullets.
#[test]
fn damaged_odt_styles_give_bullets() {
    let content = "<office:document-content xmlns:office=\"o\" xmlns:text=\"t\"><office:body><office:text>\
        <text:list text:style-name=\"Numbering_20_123\"><text:list-item><text:p>kept</text:p></text:list-item></text:list>\
        </office:text></office:body></office:document-content>";
    let file = zip_file(&[
        stored("mimetype", "application/vnd.oasis.opendocument.text"),
        stored("styles.xml", "<a><b></a>"),
        deflated("content.xml", content),
    ]);
    let (_, doc) = read(&file, "a.odt", &Caps::default()).unwrap();
    assert_eq!(doc.markdown, "<!-- page 1 of 1 -->\n## Page 1\n\n- kept\n");
    assert_eq!(doc.warnings, ["styles.xml (damaged; lists read as bullets)"]);
}

/// Astra review of 4a72078, finding 7: a shared string that the room cuts
/// short is left out and not counted, so its cell is empty, not short.
#[test]
fn cut_shared_string_is_left_out() {
    let file = xlsx("<row><c r=\"A1\" t=\"s\"><v>0</v></c><c r=\"B1\" t=\"s\"><v>1</v></c></row>", "");
    let mut zip = super::zip::Zip::open(&file, Format::Xlsx).unwrap();
    let mut entries = Vec::new();
    for name in ["[Content_Types].xml", "_rels/.rels", "xl/workbook.xml", "xl/_rels/workbook.xml.rels", "xl/styles.xml", "xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"] {
        entries.push((name, String::from_utf8(zip.part(name).unwrap().unwrap().data).unwrap()));
    }
    entries.push(("xl/sharedStrings.xml", format!("<sst><si><t>short</t></si><si><t>{}</t></si></sst>", "y".repeat(100_000))));
    let file = zip_file(&entries.iter().map(|(n, d)| deflated(n, d)).collect::<Vec<_>>());
    let (_, doc) = read(&file, "a.xlsx", &Caps { max_chars: 1000, ..Caps::default() }).unwrap();
    assert!(doc.markdown.starts_with("<!-- sheet 1 of 2 -->\n## Sheet Sales & costs\n\n| short |  |\n"), "{}", &doc.markdown[..200.min(doc.markdown.len())]);
    assert_eq!(doc.warnings, ["shared strings after 1 (memory cap; later ones show as empty cells)"]);
}

/// Astra review of 4a72078, finding 1: text of a nested archive counts once
/// against the char cap, so a file reads the same in a ZIP and in a ZIP in a
/// ZIP.
#[test]
fn nested_archive_text_counts_once() {
    let words: String = (0..300).map(|i| format!("w{i:03} ")).collect();
    let flat = zip_file(&[stored("a.txt", &words)]);
    let nested = zip_file(&[Entry { name: "in.zip", data: flat.clone(), deflate: false, raw: None }]);
    let caps = Caps { max_chars: 2500, ..Caps::default() };
    for (file, name) in [(flat, "flat.zip"), (nested, "nested.zip")] {
        let (_, doc) = read(&file, name, &caps).unwrap();
        assert!(doc.markdown.contains("w000 ") && doc.markdown.contains("w299"), "{name}: {}", doc.markdown);
        assert_eq!(doc.warnings, Vec::<String>::new(), "{name}");
        assert!(doc.chars < 2500, "{name}: {}", doc.chars);
    }
}

/// Astra review of 4a72078, finding 5: the parts that office members decode
/// draw from the archive bytes cap, so DOCX members that each decode about
/// 103 KB stop at a 150,000-byte cap in all.
#[test]
fn office_members_draw_from_the_archive_bytes_cap() {
    let (head, tail, n) = (&b"<w:document xmlns:w=\"w\"><w:body><w:p><w:r><w:t>"[..], &b"</w:t></w:r></w:p></w:body></w:document>"[..], 400);
    let data = [head, &vec![b'a'; 1 + 258 * n], tail].concat();
    let member = zip_file(&[
        stored("[Content_Types].xml", "<Types/>"),
        stored("_rels/.rels", RELS),
        Entry { name: "word/document.xml", data, deflate: true, raw: Some(deflate_run(head, n, tail)) },
    ]);
    assert!(member.len() < 3_000, "{}", member.len());
    let file = zip_file(&[
        Entry { name: "a.docx", data: member.clone(), deflate: false, raw: None },
        Entry { name: "b.docx", data: member.clone(), deflate: false, raw: None },
        Entry { name: "c.docx", data: member, deflate: false, raw: None },
    ]);
    let caps = Caps { max_bytes: 150_000, ..Caps::default() };
    let (_, doc) = read(&file, "bombs.zip", &caps).unwrap();
    let decoded = doc.markdown.bytes().filter(|&b| b == b'a').count();
    assert!(decoded > 100_000 && decoded < 150_000, "{decoded}");
    assert!(doc.warnings.iter().any(|w| w.starts_with("b.docx: word/document.xml after ") && w.ends_with(" bytes (unzip cap)")), "{:?}", doc.warnings);
}

/// Astra review of d06dbdc, finding 1: false end records that name distinct
/// starts inside the real directory join its walk, so they do not use up the
/// walk allowance of the real record (4 x 65,535 records; these eight would
/// walk about 360,000).
#[test]
fn zip_decoys_with_distinct_starts_join_one_walk() {
    let entries: Vec<Entry> = (0..45_000).map(|i| Entry { name: Box::leak(format!("e{i}").into_boxed_str()), data: Vec::new(), deflate: false, raw: None }).collect();
    let mut file = zip_file(&entries);
    let n = file.len();
    let start = u32::from_le_bytes(file[n - 6..n - 2].try_into().unwrap()) as usize;
    let fakes = 8;
    let (mut comment, mut at) = (Vec::new(), start);
    for k in 0..fakes {
        // False record k names record k of the real directory as its start.
        comment.extend_from_slice(b"PK\x05\x06\0\0\0\0\x10\x27\x10\x27");
        comment.extend_from_slice(&((n + 22 * k - at) as u32).to_le_bytes());
        comment.extend_from_slice(&(at as u32).to_le_bytes());
        comment.extend_from_slice(&((22 * (fakes - k - 1)) as u16).to_le_bytes());
        at += 46 + format!("e{k}").len();
    }
    file[n - 2..].copy_from_slice(&(comment.len() as u16).to_le_bytes());
    file.extend_from_slice(&comment);
    let mut zip = super::zip::Zip::open(&file, Format::Docx).unwrap();
    assert_eq!(zip.part("e44999").unwrap().unwrap().data, b"");
}

/// Astra review of d06dbdc, finding 2: an internal relationship target longer
/// than a ZIP entry name, for a part that the reader can do without, is left
/// out at the attribute cap and the text stays. The main document part still
/// needs a path that names a part (`docx_metadata_at_the_work_cap_gives_the_cap_warning`).
#[test]
fn long_relationship_target_is_left_out() {
    let rels = format!("<Relationships><Relationship Id=\"rId9\" Type=\"t/image\" Target=\"{}media/image1.png\"/></Relationships>", "x/../".repeat(14_000));
    let file = docx_with("<w:p><w:r><w:t>kept</w:t></w:r></w:p>", &[("word/_rels/document.xml.rels", &rels)]);
    let (_, doc) = read(&file, "a.docx", &Caps::default()).unwrap();
    assert_eq!(doc.markdown, "<!-- page 1 of 1 -->\n## Page 1\n\nkept\n");
    assert_eq!(doc.warnings, ["attribute values over 4096 bytes left out (attribute cap)"]);
}
