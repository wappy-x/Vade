use super::*;

fn md(bytes: &[u8], name: &str) -> String {
    read(bytes, name, &Caps::default()).unwrap().1.markdown
}

fn format(bytes: &[u8], name: &str) -> Option<Format> {
    sniff(bytes, name).map(|(f, _)| f)
}

#[test]
fn sniff_uses_magic_bytes_before_the_extension() {
    assert_eq!(format(b"%PDF-1.7\n", "notes.txt"), Some(Format::Pdf));
    assert_eq!(format(b"{\\rtf1 hi}", "notes.txt"), Some(Format::Rtf));
    assert_eq!(format(b"<!DOCTYPE html><p>x", "page.txt"), Some(Format::Html));
    assert_eq!(format(b"\x89PNG\r\n\x1a\nrest", "a.md"), Some(Format::Image("PNG")));
    assert_eq!(format(b"PK\x03\x04rest", "report.docx"), Some(Format::Docx));
    assert_eq!(format(b"PK\x03\x04rest", "bundle.zip"), Some(Format::Other("ZIP archive")));
    assert_eq!(format(b"bplist00\xd1\x01\x02", "Info.plist"), Some(Format::Other("binary property list")));
    assert_eq!(format(b"fn main() {}\n", "src/main.rs"), Some(Format::Source("rust")));
    assert_eq!(format(b"all:\n\ttrue\n", "Makefile"), Some(Format::Source("make")));
    assert_eq!(format(b"a,b\n1,2\n", "t.CSV"), Some(Format::Csv));
    assert_eq!(format(b"just words\n", "NOTES"), Some(Format::Text));
    assert_eq!(format(b"just words\n", ""), Some(Format::Text));
    assert_eq!(format(b"\0\x01\x02\x03\0\0\x10binary", "blob.bin"), None);
    assert_eq!(extension("dir.v2/README"), "");
    assert_eq!(extension(".bashrc"), "");
    assert_eq!(extension("a/b.Tar.GZ"), "gz");
}

#[test]
fn unsupported_encrypted_and_binary_files_give_one_line() {
    let caps = Caps::default();
    for (bytes, name, line) in [
        (&b"7z\xbc\xaf\x27\x1c"[..], "a.7z", "Vade does not read 7z archive files; no text was read"),
        (b"II*\0", "a.tif", "Vade does not read TIFF image files; no text was read"),
        (b"\0\x01\x02\x03\0\0\x10", "a.bin", "binary file with no known format; no text was read"),
    ] {
        let e = read(bytes, name, &caps).unwrap_err();
        assert_eq!(e.to_string(), line);
        assert!(!line.contains('\n'));
    }
    assert_eq!(ReadError::Encrypted(Format::Pdf).to_string(), "this PDF file is encrypted; Vade cannot read encrypted files");
    assert_eq!(
        ReadError::NoText(Format::Pdf, 3).to_string(),
        "this PDF file has no text layer (3 pages; scanned images?); Vade does not read images yet"
    );
    let small = Caps { max_bytes: 4, ..caps };
    assert_eq!(read(b"hello", "a.txt", &small).unwrap_err(), ReadError::TooLarge { bytes: 5, cap: 4 });
}

#[test]
fn header_line_and_char_cap_are_stated() {
    let out = read_markdown(b"hello\n", "dir/a.txt", &Caps::default()).unwrap();
    assert_eq!(out, "<!-- vade read: a.txt | text | pages 1 | chars 6 | skipped: none -->\nhello\n");
    let caps = Caps { max_chars: 4, ..Caps::default() };
    let (f, doc) = read("h\u{e9}llo world".as_bytes(), "a.txt", &caps).unwrap();
    assert_eq!((doc.markdown.as_str(), doc.chars), ("h\u{e9}ll", 4));
    let out = render("a.txt", f, &doc);
    assert!(out.starts_with("<!-- vade read: a.txt | text | pages 1 | chars 4 | skipped: text after 4 chars (char cap) -->\n"));
    assert!(out.ends_with("\n<!-- vade read: skipped: text after 4 chars (char cap) -->\n"));
    // A name or warning cannot close the header comment early.
    assert!(render("a-->b.txt", Format::Text, &Doc::new("x".into(), 1)).starts_with("<!-- vade read: a- ->b.txt |"));
}

#[test]
fn markers_and_helpers_follow_the_conventions() {
    assert_eq!(page_marker(2, 9), "\n<!-- page 2 of 9 -->\n## Page 2\n\n");
    assert_eq!(slide_marker(1, 3), "\n<!-- slide 1 of 3 -->\n## Slide 1\n\n");
    assert_eq!(sheet_marker(1, 2, "Q1\nsales"), "\n<!-- sheet 1 of 2 -->\n## Sheet Q1 sales\n\n");
    assert_eq!(page_cap_warning(2000, 5000, "pages"), "pages after 2000 of 5000 (pages cap)");
    assert_eq!(fence("rust", "fn a() {}"), "```rust\nfn a() {}\n```\n");
    assert_eq!(fence("md", "a ```` b\n"), "`````md\na ```` b\n`````\n");
    assert_eq!(cell(" a|b\nc "), "a\\|b<br>c");
    let rows = vec![vec!["h1".into(), "h2".into()], vec!["x".into()]];
    assert_eq!(table(&rows), "| h1 | h2 |\n| --- | --- |\n| x |  |\n");
    assert_eq!(table(&[]), "");
}

#[test]
fn text_encodings_are_detected() {
    assert_eq!(md(b"plain \xc3\xa9\n", "a.txt"), "plain \u{e9}\n");
    assert_eq!(md(b"\xef\xbb\xbfbom\n", "a.txt"), "bom\n");
    let le: Vec<u8> = b"\xff\xfe".iter().copied().chain("hi \u{e9}\n".encode_utf16().flat_map(u16::to_le_bytes)).collect();
    let be: Vec<u8> = b"\xfe\xff".iter().copied().chain("hi \u{e9}\n".encode_utf16().flat_map(u16::to_be_bytes)).collect();
    let bare: Vec<u8> = "no mark here\n".encode_utf16().flat_map(u16::to_le_bytes).collect();
    for (bytes, note) in [(le, "decoded from UTF-16 LE"), (be, "decoded from UTF-16 BE")] {
        let (_, doc) = read(&bytes, "a.txt", &Caps::default()).unwrap();
        assert_eq!((doc.markdown.as_str(), doc.warnings), ("hi \u{e9}\n", vec![note.to_owned()]));
    }
    let (f, doc) = read(&bare, "a.log", &Caps::default()).unwrap();
    assert_eq!((f, doc.markdown.as_str()), (Format::Text, "no mark here\n"));
    assert_eq!(doc.warnings, ["decoded from UTF-16 LE (no byte order mark)"]);
    // A sample cut inside a surrogate pair still decodes.
    let cut: Vec<u8> = b"\xff\xfe".iter().copied().chain("a\u{1f600}".encode_utf16().flat_map(u16::to_le_bytes)).take(6).collect();
    assert_eq!(md(&cut, "a.txt"), "a\u{fffd}");
    let (_, doc) = read(b"caf\xe9 \x93q\x94 \x80", "a.txt", &Caps::default()).unwrap();
    assert_eq!(doc.markdown, "caf\u{e9} \u{201c}q\u{201d} \u{20ac}");
    assert_eq!(doc.warnings, ["decoded as Windows-1252 (Latin-1); the file is not UTF-8"]);
}

#[test]
fn markdown_passes_through_and_data_is_fenced() {
    assert_eq!(md(b"# T\n\n- a\n", "README.md"), "# T\n\n- a\n");
    assert_eq!(md(b"{\"a\":[1,2.50,{}],\"b\":\"x,y:{\\\"\"}", "d.json"), "```json\n{\n  \"a\": [\n    1,\n    2.50,\n    {}\n  ],\n  \"b\": \"x,y:{\\\"\"\n}\n```\n");
    // Already multi-line or invalid JSON is kept as written.
    assert_eq!(md(b"{\n    \"a\": 1\n,\n\"b\": 2}", "d.json"), "```json\n{\n    \"a\": 1\n,\n\"b\": 2}\n```\n");
    assert_eq!(md(b"{\"a\":", "d.json"), "```json\n{\"a\":\n```\n");
    assert_eq!(md(b"a: 1\n", "c.yml"), "```yaml\na: 1\n```\n");
    assert_eq!(md(b"[x]\ny = 1\n", "c.toml"), "```toml\n[x]\ny = 1\n```\n");
    assert_eq!(md(b"<?xml version=\"1.0\"?><a/>", "c.xml"), "```xml\n<?xml version=\"1.0\"?><a/>\n```\n");
    assert_eq!(md(b"print(1)\n", "s.py"), "```python\nprint(1)\n```\n");
    assert_eq!(md(b"x = \"```\"\n", "s.rb"), "````ruby\nx = \"```\"\n````\n");
}

#[test]
fn html_converts_to_markdown() {
    let html = br#"<!DOCTYPE html><html><head><title>T</title><style>p{color:red}</style></head>
<body><script>alert("x<p>")</script>
<h1>Title &amp; more</h1><p>One <b>bold</b> and <em>soft</em> word,
   <a href="https://example.org/a b">a link</a> and <code>x()</code>.</p>
<p>Second&nbsp;para&#8212;dash &#x41; &lt;tag&gt; &bogus; &copy;
<ul><li>apple<li>pear<ol start="3"><li>three</li><li>four</li></ol></li></ul>
<pre class="language-rust"><code>fn main() {
    println!("&lt;hi&gt;");
}</code></pre>
<blockquote><p>quoted</p><p>twice</p></blockquote>
<table><tr><th>Name</th><th>Note</th></tr><tr><td>a|b</td><td>x<br>y</td></tr><tr><td>c</td></table>
<img src="i.png" alt="chart"><hr><p>end</body></html>"#;
    let (f, doc) = read(html, "page.html", &Caps::default()).unwrap();
    assert_eq!(f, Format::Html);
    let expected = "# Title & more\n\n\
One **bold** and *soft* word, [a link](https://example.org/a%20b) and `x()`.\n\n\
Second para\u{2014}dash A <tag> &bogus; \u{a9}\n\n\
- apple\n- pear\n  3. three\n  4. four\n\n\
```rust\nfn main() {\n    println!(\"<hi>\");\n}\n```\n\n\
> quoted\n>\n> twice\n\n\
| Name | Note |\n| --- | --- |\n| a\\|b | x<br>y |\n| c |  |\n\n\
![chart](i.png)\n\n---\n\nend\n";
    assert_eq!(doc.markdown, expected);
    assert_eq!(doc.warnings, ["1 script, style or embedded blocks"]);
    assert_eq!(html_to_markdown("<h3>A</h3>text<br>next"), "### A\n\ntext\nnext\n");
    // A head with no close tag does not swallow the body.
    assert_eq!(html_to_markdown("<html><head><title>t<body><p>body"), "t\n\nbody\n");
}

#[test]
fn csv_and_tsv_become_tables() {
    let csv = b"name,note,n\r\n\"Smith, J\",\"said \"\"hi\"\"\nthen left\",1\r\npipe,a|b,2\r\n";
    assert_eq!(
        md(csv, "people.csv"),
        "| name | note | n |\n| --- | --- | --- |\n| Smith, J | said \"hi\"<br>then left | 1 |\n| pipe | a\\|b | 2 |\n"
    );
    assert_eq!(md(b"a;b;c\n1;2,5;3\n", "eu.csv"), "| a | b | c |\n| --- | --- | --- |\n| 1 | 2,5 | 3 |\n");
    assert_eq!(md(b"a\tb\n1\t2\n", "t.tsv"), "| a | b |\n| --- | --- |\n| 1 | 2 |\n");
    let caps = Caps { max_rows: 2, ..Caps::default() };
    let (_, doc) = read(b"h\n1\n2\n3\n4\n", "c.csv", &caps).unwrap();
    assert_eq!(doc.markdown, "| h |\n| --- |\n| 1 |\n| 2 |\n");
    assert_eq!(doc.warnings, ["rows after 2 of 4 (row cap)"]);
    assert_eq!(md(b"", "empty.csv"), "");
}

#[test]
fn rtf_becomes_plain_text() {
    let rtf = br"{\rtf1\ansi\ansicpg1252\deff0{\fonttbl{\f0\fswiss Helvetica;}}{\colortbl;\red0\green0\blue0;}
{\*\generator Writer 1.0;}{\info{\author A}}\f0\fs24 Hello \b bold\b0  caf\'e9\par
Line\line two \{braces\} \\ and \u8212\'97 dash\uc0\u8364  euro\par
{\pict\pngblip 89504e47}{\field{\*\fldinst HYPERLINK x}{\fldrslt link}}\tab end}";
    let (f, doc) = read(rtf, "a.rtf", &Caps::default()).unwrap();
    assert_eq!(f, Format::Rtf);
    assert_eq!(doc.markdown, "Hello bold caf\u{e9}\nLine\ntwo {braces} \\ and \u{2014} dash\u{20ac} euro\nlink\tend\n");
    assert_eq!(doc.warnings, ["1 pictures or objects"]);
    let emoji = read(br"{\rtf1 a\u-10179?\u-8704?b}", "e.rtf", &Caps::default()).unwrap().1;
    assert_eq!(emoji.markdown, "a\u{1f600}b\n");
    assert!(matches!(read(b"not rtf", "a.rtf", &Caps::default()), Err(ReadError::Malformed(Format::Rtf, _))));
}

#[test]
fn readers_do_not_panic_on_cut_or_flipped_input() {
    let samples: [(&[u8], &str); 4] = [
        (b"<table><tr><td><a href='x'>l<pre>p</table></td><li>&#xFFFFFFFF;&#0;<", "a.html"),
        (b"\"a,\"\"b\n\"x", "a.csv"),
        (br"{\rtf1 {\*\x {\u-3\'zz}}\u99999999999 \'", "a.rtf"),
        (b"{\"a\":[\"\\\"\",{}]}", "a.json"),
    ];
    for (bytes, name) in samples {
        for cut in 0..=bytes.len() {
            let _ = read(&bytes[..cut], name, &Caps::default());
        }
        for i in 0..bytes.len() {
            let mut flipped = bytes.to_vec();
            flipped[i] ^= 0xa5;
            let _ = read(&flipped, name, &Caps::default());
        }
    }
}

/// Astra review of 075165a, BLOCK 1: the caps apply while a reader converts,
/// so a small file cannot build a large result first.
#[test]
fn caps_bound_the_work_during_conversion() {
    let tiny = Caps { max_chars: 1, ..Caps::default() };
    // 6,000 bytes: one 2,000-cell header row, then 2,000 one-cell rows.
    let csv = format!("{}\n{}", ",".repeat(1999), "x\n".repeat(2000));
    let doc = csv::CsvReader.to_markdown(csv.as_bytes(), Format::Csv, &tiny).unwrap();
    assert!(doc.markdown.chars().count() <= 1, "{} bytes", doc.markdown.len());
    assert_eq!(doc.warnings, ["text after 1 chars (char cap)"]);
    let full = csv::CsvReader.to_markdown(csv.as_bytes(), Format::Csv, &Caps::default()).unwrap();
    assert!(full.markdown.chars().count() <= MAX_CHARS, "{} chars", full.markdown.chars().count());
    // 19,000 bytes of nested lists indented every item by its depth.
    let html = format!("{}{}", "<ul><li>x".repeat(1000), "</li></ul>".repeat(1000));
    let doc = html::HtmlReader.to_markdown(html.as_bytes(), Format::Html, &tiny).unwrap();
    assert!(doc.markdown.len() <= 64 * 1024, "{} bytes", doc.markdown.len());
    let doc = html::HtmlReader.to_markdown(html.as_bytes(), Format::Html, &Caps::default()).unwrap();
    assert!(doc.markdown.len() <= 200 * 1000, "{} bytes", doc.markdown.len());
    // Nested quotes repeat their prefix on every level.
    let quotes = format!("{}x{}", "<blockquote>".repeat(5000), "</blockquote>".repeat(5000));
    let doc = html::HtmlReader.to_markdown(quotes.as_bytes(), Format::Html, &Caps::default()).unwrap();
    assert!(doc.markdown.len() <= 64 * 1024, "{} bytes", doc.markdown.len());
    // A wide HTML table pads each short row to the widest row.
    let table = format!("<table><tr>{}</tr>{}</table>", "<td>h".repeat(3000), "<tr><td>x".repeat(3000));
    let doc = html::HtmlReader.to_markdown(table.as_bytes(), Format::Html, &tiny).unwrap();
    assert!(doc.markdown.len() <= 64 * 1024, "{} bytes", doc.markdown.len());
    // Text, source and RTF stop at the char cap too.
    let text = "word ".repeat(1000);
    for (reader, format, bytes) in [
        (&text::TextReader as &dyn Reader, Format::Text, text.clone().into_bytes()),
        (&text::TextReader, Format::Source("rust"), text.clone().into_bytes()),
        (&rtf::RtfReader, Format::Rtf, format!("{{\\rtf1 {text}}}").into_bytes()),
    ] {
        let doc = reader.to_markdown(&bytes, format, &Caps { max_chars: 10, ..Caps::default() }).unwrap();
        assert!(doc.markdown.chars().count() <= 10 + 16, "{format}: {:?}", doc.markdown);
        assert_eq!(doc.warnings, ["text after 10 chars (char cap)"], "{format}");
    }
    // An XHTML part (EPUB) converts within its reader's budget.
    let mut budget = Budget::new(&Caps { max_chars: 3, ..Caps::default() });
    assert_eq!(html_to_markdown_in("<p>hello</p><p>more</p>", &mut budget), "hel\n");
    assert_eq!(budget.warning(), ["text after 3 chars (char cap)"]);
    // The registry does not repeat the reader's char cap warning.
    let (_, doc) = read(csv.as_bytes(), "w.csv", &tiny).unwrap();
    assert_eq!((doc.chars, doc.warnings), (1, vec!["text after 1 chars (char cap)".to_owned()]));
}

/// Astra BLOCK 1: an unclosed head tag is searched for once, not once per
/// tag. Astra SHOULD-FIX 5 of 7d418cd: the test counts the bytes that close
/// tag searches look at, not the time, so a slow host cannot fail it.
#[test]
fn repeated_unclosed_head_tags_stay_linear() {
    let searched = |html: &str| {
        SCANNED.with(|n| n.set(0));
        let markdown = html_to_markdown(html);
        (markdown, SCANNED.with(|n| n.get()))
    };
    let html = format!("{}<p>end", "<head>".repeat(40_000));
    let (markdown, bytes) = searched(&html);
    assert_eq!(markdown, "end\n");
    assert!(bytes <= html.len(), "{bytes} bytes searched in {}", html.len());
    // Nested drop elements count their depth in the same single pass.
    let html = format!("<p>a</p>{}{}<p>b", "<template><svg>".repeat(20_000), "</svg></template>".repeat(20_000));
    let (markdown, bytes) = searched(&html);
    assert_eq!(markdown, "a\n\nb\n");
    assert!(bytes <= html.len(), "{bytes} bytes searched in {}", html.len());
}

/// Astra BLOCK 3: a list number at the integer limit neither panics nor
/// prints a number that Markdown does not accept (more than nine digits).
#[test]
fn ordered_list_numbers_saturate() {
    assert_eq!(html_to_markdown("<ol start=\"18446744073709551615\"><li>x</li><li>y</li></ol>"), "999999999. x\n999999999. y\n");
    assert_eq!(html_to_markdown("<ol start=\"999999999999999999999999\"><li>x</li></ol>"), "999999999. x\n");
    assert_eq!(html_to_markdown("<ol start=\"999999998\"><li>x<li>y<li>z</ol>"), "999999998. x\n999999999. y\n999999999. z\n");
    assert_eq!(html_to_markdown("<ol start=\"-4\"><li>x</li></ol>"), "0. x\n");
    assert_eq!(html_to_markdown("<ol start=\"x\"><li>x</li></ol>"), "1. x\n");
}

/// Astra SHOULD-FIX 6: spaces around `=` and XHTML empty elements.
#[test]
fn html_attribute_spacing_and_empty_elements() {
    assert_eq!(html_to_markdown("<a href = \"https://example.org\">link</a>"), "[link](https://example.org)\n");
    assert_eq!(html_to_markdown("<a\nhref\t=\n'https://example.org/b' title = x>b</a>"), "[b](https://example.org/b)\n");
    assert_eq!(html_to_markdown("<img alt = \"diagram\">"), "[image: diagram]\n");
    assert_eq!(html_to_markdown("<img src = i.png alt= chart />"), "![chart](i.png)\n");
    assert_eq!(html_to_markdown("<blockquote/>outside"), "outside\n");
    assert_eq!(html_to_markdown("<pre/>outside"), "outside\n");
    assert_eq!(html_to_markdown("<a href=\"x\"/>outside"), "outside\n");
    assert_eq!(html_to_markdown("<ol/><li>x</li>"), "- x\n");
    assert_eq!(html_to_markdown("<p>a<b/>b<h2/>c<br/>d</p>"), "ab\n\nc\nd\n");
    assert_eq!(html_to_markdown("<table/>after"), "after\n");
}

/// Astra SHOULD-FIX 7: `\binN` payloads and control-word Unicode fallbacks.
#[test]
fn rtf_binary_payloads_and_control_word_fallbacks() {
    let rtf = |s: &[u8]| read(s, "a.rtf", &Caps::default()).unwrap().1;
    let doc = rtf(br"{\rtf1 A{\pict\bin1 {}B}");
    assert_eq!((doc.markdown.as_str(), doc.warnings), ("AB\n", vec!["1 pictures or objects".to_owned()]));
    assert_eq!(rtf(b"{\\rtf1 A{\\*\\blob\\bin3 }{\\}C}").markdown, "AC\n");
    assert_eq!(rtf(br"{\rtf1 A\u8212\emdash B}").markdown, "A\u{2014}B\n");
    assert_eq!(rtf(br"{\rtf1\uc2 A\u8212\'97\'97B\u8212\endash\'97C}").markdown, "A\u{2014}B\u{2014}C\n");
    // A huge or negative byte count stops at the end of the file.
    assert_eq!(rtf(br"{\rtf1 A\bin99999999999 xyz}").markdown, "A\n");
    assert_eq!(rtf(br"{\rtf1 A\bin-5 B}").markdown, "AB\n");
}

/// Astra SHOULD-FIX 8: the declared ANSI code page decodes `\'hh` bytes.
#[test]
fn rtf_code_pages_decode_or_are_named() {
    let rtf = |s: &[u8]| read(s, "a.rtf", &Caps::default()).unwrap().1;
    let doc = rtf(br"{\rtf1\ansi\ansicpg1251 \'cf\'f0\'e8\'e2\'e5\'f2}");
    assert_eq!((doc.markdown.as_str(), doc.warnings.len()), ("\u{41f}\u{440}\u{438}\u{432}\u{435}\u{442}\n", 0));
    assert_eq!(rtf(br"{\rtf1\ansi\ansicpg1250 \'8a\'e8}").markdown, "\u{160}\u{10d}\n");
    assert_eq!(rtf(br"{\rtf1\mac \'8e\'a5}").markdown, "\u{e9}\u{2022}\n");
    assert_eq!(rtf(br"{\rtf1\ansi\ansicpg1252 caf\'e9}").markdown, "caf\u{e9}\n");
    // An unsupported code page never prints guessed letters.
    let doc = rtf(br"{\rtf1\ansi\ansicpg932 A\'82\'a0B \u12354?}");
    assert_eq!(doc.markdown, "A\u{fffd}\u{fffd}B \u{3042}\n");
    assert_eq!(doc.warnings, ["2 bytes in code page 932 (not supported; shown as U+FFFD)"]);
    // A font charset wins over the document code page. macOS writes
    // \fcharset0 (Windows-1252) fonts under \mac, \ansicpg932 or \ansicpg65001.
    assert_eq!(rtf(br"{\rtf1\mac\ansicpg10000{\fonttbl\f0\fnil\fcharset0 Lucida;}\f0 prena\'9aate}").markdown, "prena\u{161}ate\n");
    let fonts = br"{\rtf1\ansi\ansicpg932{\fonttbl\f0\fnil\fcharset0 L;\f1\fswiss\fcharset204 C;}\f0 \'93q\'94 {\f1 \'cf}\'e9}";
    let doc = rtf(fonts);
    assert_eq!((doc.markdown.as_str(), doc.warnings.len()), ("\u{201c}q\u{201d} \u{41f}\u{e9}\n", 0));
    assert_eq!(rtf(br"{\rtf1\ansi\deff0{\fonttbl{\f0\fcharset204 Arial;}}\'cf}").markdown, "\u{41f}\n");
    assert_eq!(rtf(br"{\rtf1\ansi\ansicpg1251{\fonttbl{\f0\fcharset1 A;}}\f0\'cf}").markdown, "\u{41f}\n");
    let doc = rtf(br"{\rtf1\ansi{\fonttbl{\f0\fcharset128 MS;}}\f0 \'82\'a0 ok}");
    assert_eq!((doc.markdown.as_str(), doc.warnings), ("\u{fffd}\u{fffd} ok\n", vec!["2 bytes in code page 932 (not supported; shown as U+FFFD)".to_owned()]));
}

/// Astra SHOULD-FIX 9: an unfinished quoted field is an error, not a merged record.
#[test]
fn csv_unterminated_quote_is_an_error() {
    let e = read(b"name,note\nAlice,\"first\nBob,second\n", "a.csv", &Caps::default()).unwrap_err();
    assert_eq!(e, ReadError::Malformed(Format::Csv, "the quoted field in record 2 has no closing quote".into()));
    assert_eq!(e.to_string(), "damaged CSV file: the quoted field in record 2 has no closing quote; no text was read");
    assert!(read(b"a\tb\n\"x\ty\n", "a.tsv", &Caps::default()).is_err());
    assert_eq!(md(b"a,b\n\"x\"\"\",y\n", "ok.csv"), "| a | b |\n| --- | --- |\n| x\" | y |\n");
}

/// Astra SHOULD-FIX 10: every hyphen pair is broken, overlapping ones too.
#[test]
fn header_escapes_every_hyphen_run() {
    let out = render("a--->b.txt", Format::Text, &Doc::new("x".into(), 1));
    assert!(out.starts_with("<!-- vade read: a- - ->b.txt | text |"), "{out}");
    let mut doc = Doc::new("x".into(), 1);
    doc.warnings.push("w---x--!>".into());
    let out = render("----", Format::Text, &doc);
    let header = out.lines().next().unwrap();
    assert_eq!(header, "<!-- vade read: - - - - | text | pages 1 | chars 0 | skipped: w- - -x- -!> -->");
    for line in out.lines().filter(|l| l.starts_with("<!--")) {
        assert!(!line[4..line.len() - 3].contains("--"), "{line}");
    }
}

/// Astra SHOULD-FIX 12: ZIP magic wins over a text extension.
#[test]
fn zip_magic_wins_over_text_extensions() {
    assert_eq!(format(b"PK\x03\x04rest", "notes.txt"), Some(Format::Other("ZIP archive")));
    assert_eq!(format(b"PK\x05\x06rest", "data.csv"), Some(Format::Other("ZIP archive")));
    assert_eq!(format(b"PK\x03\x04rest", "book.EPUB"), Some(Format::Epub));
    assert_eq!(format(b"PK\x03\x04rest", "sheet.xlsx"), Some(Format::Xlsx));
    assert_eq!(read(b"PK\x03\x04rest", "notes.txt", &Caps::default()).unwrap_err().to_string(), "Vade does not read ZIP archive files; no text was read");
    assert_eq!(format(b"PK is a plain word\n", "notes.txt"), Some(Format::Text));
}

/// Argon review of 7c9881a, item 1: a cap that stops the scan inside an open
/// block keeps that block's text, and the first stop names the cap.
#[test]
fn a_cap_inside_an_open_block_keeps_its_text() {
    for (html, kept) in [
        ("<pre>hello world</pre>", "```\nh"),
        ("<table><tr><td>hello</td><td>world</td></tr></table>", "| hel"),
        ("<blockquote>hello world</blockquote>", "> hel"),
        ("<p><a href=\"x\">hello world</a></p>", "[hell"),
    ] {
        let (_, doc) = read(html.as_bytes(), "a.html", &Caps { max_chars: 5, ..Caps::default() }).unwrap();
        assert_eq!((doc.markdown.as_str(), doc.chars), (kept, 5), "{html}");
        assert_eq!(doc.warnings, ["text after 5 chars (char cap)"], "{html}");
    }
    let (_, doc) = read(b"<pre>hello world</pre><p>more text</p>", "a.html", &Caps { max_chars: 14, ..Caps::default() }).unwrap();
    assert_eq!(doc.markdown, "```\nhello worl");
    let mut budget = Budget::new(&Caps { max_chars: 3, ..Caps::default() });
    assert_eq!(budget.text("abcdef"), "abc");
    assert_eq!((budget.text("x"), budget.work("x")), ("", "x"));
    assert!(budget.work(&"y".repeat(MIN_WORK_BYTES)).len() < MIN_WORK_BYTES);
    assert_eq!(budget.warning(), ["text after 3 chars (char cap)"]);
    // A shared budget (EPUB parts) counts the next part's text again.
    let mut budget = Budget::new(&Caps { max_chars: 8, ..Caps::default() });
    assert_eq!(html_to_markdown_in("<blockquote>abc", &mut budget), "> abc\n");
    assert_eq!(html_to_markdown_in("<p>defghijk</p>", &mut budget), "defgh\n");
    assert!(budget.full());
}

/// Argon items 2, 3 and 9: `</header>` does not close `<head>`; entities with
/// no `;` follow the HTML standard; close tags past the nesting cap do not
/// close an outer block.
#[test]
fn html_close_tags_entities_and_deep_close_tags() {
    let doc = md(b"<html><head><title>T</title><body><header><h1>Heading</h1></header><p>Body</p></body></html>", "a.html");
    // The head has no close tag, so it is ignored, not dropped: its title stays.
    assert_eq!(doc, "T\n\n# Heading\n\nBody\n");
    assert_eq!(md(b"<script>a</scripts>b</script><p>c</p>", "a.html"), "c\n");
    let doc = md(b"<a href=\"https://example.org/p?a=1&reg=2&le=3&amp;x=4&copy;\">link</a>", "a.html");
    assert_eq!(doc, "[link](https://example.org/p?a=1&reg=2&le=3&x=4\u{a9})\n");
    // In text a legacy name decodes with no `;`, as in a browser; others do not.
    assert_eq!(md(b"<p>&le=3 &le; &copy 2026 &not_y &amp</p>", "a.html"), "&le=3 \u{2264} \u{a9} 2026 \u{ac}_y &\n");
    let deep = format!("{}deep{}after", "<blockquote>".repeat(40), "</blockquote>".repeat(8));
    let doc = md(deep.as_bytes(), "a.html");
    let depth = |word: &str| doc.lines().find(|l| l.contains(word)).map(|l| l.matches('>').count());
    assert_eq!((depth("deep"), depth("after")), (Some(html::MAX_NESTING), Some(html::MAX_NESTING)), "{doc}");
}

/// Argon item 5: bad bytes after a UTF-8 byte order mark get a note, and a NUL
/// byte after the first 64 KiB still makes the file binary.
#[test]
fn bom_with_bad_bytes_and_late_nul_bytes() {
    let (_, doc) = read(b"\xef\xbb\xbfcaf\xe9\n", "a.txt", &Caps::default()).unwrap();
    assert_eq!(doc.markdown, "caf\u{fffd}\n");
    assert_eq!(doc.warnings, ["the file has a UTF-8 byte order mark, but some bytes are not UTF-8 (shown as U+FFFD)"]);
    assert_eq!(md("\u{feff}caf\u{e9}\n".as_bytes(), "a.txt"), "caf\u{e9}\n");
    let mut late = vec![b'a'; 70_000];
    late.extend_from_slice(b"\0\0\0\0");
    assert_eq!(read(&late, "a.txt", &Caps::default()).unwrap_err(), ReadError::Binary);
    assert_eq!(read(&late, "", &Caps::default()).unwrap_err(), ReadError::Binary);
}

/// Argon review of dc4936a, findings 1 and 2: elements past the nesting cap
/// end with the frame that holds them, and a high surrogate with no low half
/// right after it is U+FFFD.
#[test]
fn deep_elements_end_with_their_frame_and_lone_surrogates_show() {
    let html = format!("<table><tr><td>{}</td></tr></table><blockquote><p>a</p></blockquote><p>b</p>", "<blockquote>".repeat(31));
    assert!(md(html.as_bytes(), "a.html").ends_with("> a\n\nb\n"), "{}", md(html.as_bytes(), "a.html"));
    let deep = "<blockquote>".repeat(40);
    let links = format!("{deep}<a href=\"x\">one</a><a href=\"y\">two<a href=\"z\">three</a></a>{}after", "</blockquote>".repeat(40));
    assert!(md(links.as_bytes(), "a.html").ends_with("onetwothree\n\nafter\n"), "{}", md(links.as_bytes(), "a.html"));
    for (rtf, text) in [
        (r"{\rtf1 A\u-10179?B}", "A\u{fffd}B\n"),
        (r"{\rtf1 A\u-10179?B\u-8704?C}", "A\u{fffd}B\u{fffd}C\n"),
        (r"{\rtf1 A\u-10179?\u-8704?C}", "A\u{1f600}C\n"),
        (r"{\rtf1 A\u-10179?\u-10179?\u-8704?C}", "A\u{fffd}\u{1f600}C\n"),
        (r"{\rtf1 A\u-10179?}", "A\u{fffd}\n"),
    ] {
        assert_eq!(md(rtf.as_bytes(), "a.rtf"), text, "{rtf}");
    }
}

/// Astra review of 7d418cd, BLOCK 1: parser storage (open RTF groups and
/// fonts, HTML rows, cells and lists, tag attributes, CSV fields) counts
/// against the work cap before it is allocated. Astra's inputs, at a char cap
/// of 1, give the same text as before; where storage fills the work cap, the
/// memory cap warning says so. `cli/tests/read_memory.rs` measures the
/// process memory of the same inputs.
#[test]
fn parser_storage_counts_against_the_work_cap() {
    let one = Caps { max_chars: 1, ..Caps::default() };
    let memory = "text after 65536 bytes of conversion work (memory cap)".to_owned();
    let chars = "text after 1 chars (char cap)".to_owned();
    let rtf = format!("{{\\rtf1 {}{}}}", "{".repeat(1_000_000), "}".repeat(1_000_000));
    let (_, doc) = read(rtf.as_bytes(), "a.rtf", &one).unwrap();
    assert_eq!((doc.markdown.as_str(), doc.warnings), ("", vec![memory.clone()]));
    let rows = format!("<table>{}</table>", "<tr></tr>".repeat(125_000));
    let (_, doc) = read(rows.as_bytes(), "a.html", &one).unwrap();
    assert_eq!((doc.markdown.as_str(), doc.warnings), ("", vec![memory.clone()]));
    let fonts: String = (0..10_000).map(|n| format!("{{\\f{n}\\fcharset0 F;}}")).collect();
    let rtf = format!("{{\\rtf1{{\\fonttbl{fonts}}}x}}");
    let (_, doc) = read(rtf.as_bytes(), "a.rtf", &one).unwrap();
    assert_eq!(doc.warnings, [memory]);
    // Lists past the nesting cap, attributes this reader does not use and
    // CSV fields keep no storage: the result is the text alone.
    let lists = format!("{}x{}", "<ul>".repeat(250_000), "</ul>".repeat(250_000));
    assert_eq!(md(lists.as_bytes(), "a.html"), "x\n");
    for (bytes, name) in [(format!("<p {}>x</p>", "a ".repeat(500_000)).into_bytes(), "a.html"), (vec![b','; 1_000_000], "a.csv")] {
        let (_, doc) = read(&bytes, name, &one).unwrap();
        assert_eq!(doc.warnings, [chars.clone()], "{name}");
    }
    // A cell, an attribute value and a text run are cut to the room left.
    let caps = Caps { max_chars: 4096, ..Caps::default() };
    let csv = format!("a,b\n\"{}\",y\n", "x\"\"".repeat(400_000));
    let (_, doc) = read(csv.as_bytes(), "a.csv", &caps).unwrap();
    assert!(doc.markdown.starts_with("| a | b |\n| --- | --- |\n| x\"x\"x\"x\""), "{}", &doc.markdown[..80]);
    // A link target longer than the room stops the reader: a cut target is
    // not the link.
    let html = format!("<p>a</p><a href=\"{}\">x</a>", "h".repeat(2_000_000));
    let (_, doc) = read(html.as_bytes(), "a.html", &caps).unwrap();
    assert_eq!((doc.markdown.as_str(), doc.warnings), ("a\n", vec!["text after 65536 bytes of conversion work (memory cap)".to_owned()]));
    let html = format!("<a href=\"{}\">x</a><p>{}</p>", "h".repeat(10_000), "y &amp; ".repeat(300_000));
    let (_, doc) = read(html.as_bytes(), "a.html", &caps).unwrap();
    assert!(doc.markdown.starts_with("[x](hhh"), "{:?}", doc.markdown.get(..200));
    assert_eq!(doc.chars, 4096);
}

/// Astra review of 7d418cd, SHOULD-FIX 2, 3, 6 and 7: `\plain` goes back to
/// the default font; `\binN` with its data is one fallback character; a cut
/// sequence at the end of a whole file is U+FFFD; nested templates stay hidden.
#[test]
fn font_reset_binary_fallback_cut_end_and_nested_templates() {
    let plain = r"{\rtf1\ansi\deff0{\fonttbl{\f0\fcharset0 Latin;}{\f1\fcharset204 Cyrillic;}}\f1 \'cf\plain \'e9}";
    assert_eq!(md(plain.as_bytes(), "a.rtf"), "\u{41f}\u{e9}\n");
    assert_eq!(md(br"{\rtf1\uc1 A\u8212\bin1 xB}", "a.rtf"), "A\u{2014}B\n");
    assert_eq!(md(br"{\rtf1\uc2 A\u8212\bin1 x\'41B}", "a.rtf"), "A\u{2014}B\n");
    let (_, doc) = read(b"\xef\xbb\xbfhello\xe2\x82", "a.txt", &Caps::default()).unwrap();
    assert_eq!(doc.markdown, "hello\u{fffd}");
    assert_eq!(doc.warnings, ["the file has a UTF-8 byte order mark, but some bytes are not UTF-8 (shown as U+FFFD)"]);
    let (_, doc) = read(b"\xff\xfeh\x00i", "a.txt", &Caps::default()).unwrap();
    assert_eq!((doc.markdown.as_str(), doc.warnings), ("h\u{fffd}", vec!["decoded from UTF-16 LE".to_owned()]));
    let (_, doc) = read(b"<template><template>hidden1</template>hidden2</template><p>visible</p>", "a.html", &Caps::default()).unwrap();
    assert_eq!((doc.markdown.as_str(), doc.warnings), ("visible\n", vec!["1 script, style or embedded blocks".to_owned()]));
    for (html, text) in [
        ("<svg><svg/>hidden</svg><p>visible</p>", "visible\n"),
        ("<SVG><svg >a</svg >b</SVG><p>visible</p>", "visible\n"),
        ("<template><templates>a</template>visible", "visible\n"),
        // Script and style stay raw text: the first close tag ends them.
        ("<script><script>a</script>visible</script>", "visible\n"),
    ] {
        assert_eq!(md(html.as_bytes(), "a.html"), text, "{html}");
    }
}

/// Astra review of 8f038e3, findings 1 to 4: JSON storage and output stay in
/// the budget; skipped RTF data and HTML white space are scanned once; a cell
/// or an attribute keeps its text after leading spaces and entities, or the
/// reader stops with a warning; comments, quoted values and script text in
/// a template hold no tags.
#[test]
fn json_budget_single_scans_kept_text_and_template_tokens() {
    let one = Caps { max_chars: 1, ..Caps::default() };
    let memory = "text after 65536 bytes of conversion work (memory cap)".to_owned();
    let json = format!("[{}0]", "0,".repeat(90_000));
    let (_, doc) = read(json.as_bytes(), "a.json", &one).unwrap();
    assert_eq!((doc.markdown.as_str(), doc.warnings), ("`", vec!["text after 1 chars (char cap)".to_owned()]));
    let small = Caps { max_chars: 100, ..Caps::default() };
    let (_, doc) = read(json.as_bytes(), "a.json", &small).unwrap();
    assert!(doc.markdown.starts_with("```json\n[\n  0,\n  0,\n"), "{}", doc.markdown);
    for bad in ["[1,]", "{\"a\" 1}", "[01]", "\"\\x\"", "[1}", "tru", "{\"a\":1,}", "1 2"] {
        assert_eq!(md(bad.as_bytes(), "a.json"), format!("```json\n{bad}\n```\n"), "{bad}");
    }
    assert_eq!(md(format!("{}{}", "[".repeat(65), "]".repeat(65)).as_bytes(), "a.json").lines().count(), 3);
    // Each scan looks at a source byte once, near-full storage or not.
    let scanned = |bytes: &[u8], name: &str| {
        SCANNED.with(|n| n.set(0));
        let (_, doc) = read(bytes, name, &one).unwrap();
        (doc, SCANNED.with(|n| n.get()))
    };
    let rtf = format!("{{\\rtf1 {}{{\\pict {}}}{}}}", "{".repeat(2044), "a".repeat(1 << 20), "}".repeat(2044));
    let (doc, bytes) = scanned(rtf.as_bytes(), "a.rtf");
    assert_eq!(doc.warnings, ["1 pictures or objects"]);
    assert!(bytes <= rtf.len(), "{bytes} bytes scanned in {}", rtf.len());
    let html = format!("<table>{}{}<tr><td>X</td></tr></table>", "<tr></tr>".repeat(2729), " ".repeat(1 << 20));
    let (doc, bytes) = scanned(html.as_bytes(), "a.html");
    assert_eq!(doc.warnings, [memory]);
    assert!(bytes <= html.len(), "{bytes} bytes scanned in {}", html.len());
    // Leading spaces and references do not use up the room.
    let caps = Caps { max_chars: 4096, ..Caps::default() };
    let csv = format!("a,b\n\"{}TAIL\",z\n", " ".repeat(70_000));
    let (_, doc) = read(csv.as_bytes(), "a.csv", &caps).unwrap();
    assert_eq!((doc.markdown.as_str(), doc.warnings), ("| a | b |\n| --- | --- |\n| TAIL | z |\n", vec![]));
    let alt = format!("<img alt=\"{}TAIL\">", "&#0000000000000000000000000000065;".repeat(600));
    let (_, doc) = read(alt.as_bytes(), "a.html", &caps).unwrap();
    assert_eq!((doc.markdown, doc.warnings), (format!("[image: {}TAIL]\n", "A".repeat(600)), vec![]));
    // Text that does not fit stops the reader with the warning.
    let csv = format!("a,b\n\"x{}TAIL\",z\n", " ".repeat(70_000));
    let (_, doc) = read(csv.as_bytes(), "a.csv", &caps).unwrap();
    assert_eq!((doc.markdown.as_str(), doc.warnings), ("| a | b |\n| --- | --- |\n| x |", vec!["text after 65536 bytes of conversion work (memory cap)".to_owned()]));
    for html in [
        "<template><!-- </template> -->hidden</template><p>visible</p>",
        "<template><div title='</template>'>hidden</div></template><p>visible</p>",
        "<template><script>let x = '</template>';</script>hidden</template><p>visible</p>",
        "<template><!-- <template> --></template><p>visible</p>",
        "<template><template title='>'/>hidden</template><p>visible</p>",
        "<template><style>a::after { content: '<template>' }</style></template><p>visible</p>",
    ] {
        assert_eq!(md(html.as_bytes(), "a.html"), "visible\n", "{html}");
    }
}

/// Astra review of 3552bde, findings 2 and 3: textarea, title and CDATA text
/// and quoted values in close tags hold no tags for the drop scanner, and a
/// prefixed element ends at its prefixed close tag.
#[test]
fn drop_scanner_text_states_and_prefixed_names() {
    for html in [
        "<template><textarea></template>hidden</textarea></template><p>visible</p>",
        "<template><textarea><template></textarea></template><p>visible</p>",
        "<template><title></template></title>hidden</template><p>visible</p>",
        "<svg><![CDATA[</svg>]]><text>hidden</text></svg><p>visible</p>",
        "<svg><![CDATA[<svg>]]></svg><p>visible</p>",
        "<template><div></div title='</template>'>hidden</template><p>visible</p>",
        "<svg:svg xmlns:svg='http://www.w3.org/2000/svg'><svg:text>hidden</svg:text></svg:svg><p>visible</p>",
        "<svg:svg><svg:svg>a</svg:svg>hidden</svg:svg><p>visible</p>",
        "<m:math><m:mi>x</m:mi></m:math><p>visible</p>",
        "<x:script>if (a < b) {}</x:script><p>visible</p>",
    ] {
        let (_, doc) = read(html.as_bytes(), "a.html", &Caps::default()).unwrap();
        assert_eq!((doc.markdown.as_str(), doc.warnings), ("visible\n", vec!["1 script, style or embedded blocks".to_owned()]), "{html}");
    }
}
