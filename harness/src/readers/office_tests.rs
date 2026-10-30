//! Tests for the ZIP container, the XML scanner and the ZIP-based readers.
//! Fixtures are built here: [`zip_file`] writes stored or deflated entries,
//! and [`deflate_fixed`] writes fixed-Huffman deflate data.
use super::xml::{Event, Scanner, unescape, unescape_prefix};

const ATTR_CAP: &str = "attribute values over 4096 bytes left out (attribute cap)";

/// The decoded start of the `descr` attribute of the first `name` element in
/// `xml` (at most 804 bytes), and its alt text.
fn events_tag(xml: &str, name: &str) -> (Option<(String, bool)>, String) {
    let mut scan = Scanner::new(xml, false);
    while let Some(event) = scan.next().unwrap() {
        if let Event::Open(tag) = event
            && tag.local() == name
        {
            let start = tag.attr_prefix("descr", 804).unwrap().map(|(v, whole)| (v.into_owned(), whole));
            return (start, tag.alt_text(&["descr"]).unwrap());
        }
    }
    (None, String::new())
}
use super::zip::{Zip, crc32, resolve};
use super::*;

/// One fixture entry: the name, the data, and the compression method.
pub(super) struct Entry<'a> {
    pub(super) name: &'a str,
    pub(super) data: Vec<u8>,
    pub(super) deflate: bool,
    /// Deflate data to store in place of the encoded `data`.
    pub(super) raw: Option<Vec<u8>>,
}

pub(super) fn stored<'a>(name: &'a str, data: &str) -> Entry<'a> {
    Entry { name, data: data.as_bytes().to_vec(), deflate: false, raw: None }
}

pub(super) fn deflated<'a>(name: &'a str, data: &str) -> Entry<'a> {
    Entry { name, data: data.as_bytes().to_vec(), deflate: true, raw: None }
}

/// A ZIP archive of `entries`. `edit` may change each central directory
/// record (flags at 8, method at 10, CRC at 16, sizes at 20 and 24).
pub(super) fn zip_with(entries: &[Entry], edit: impl Fn(&str, &mut [u8])) -> Vec<u8> {
    let mut out = Vec::new();
    let mut dir = Vec::new();
    for e in entries {
        let packed = match (&e.raw, e.deflate) {
            (Some(raw), _) => raw.clone(),
            (None, true) => deflate_fixed(&e.data),
            (None, false) => e.data.clone(),
        };
        let offset = out.len() as u32;
        let mut fields = Vec::new();
        fields.extend_from_slice(&[20, 0, 0, 0]);
        fields.extend_from_slice(&(if e.deflate { 8u16 } else { 0 }).to_le_bytes());
        fields.extend_from_slice(&[0, 0, 0, 0]);
        fields.extend_from_slice(&crc32(&e.data).to_le_bytes());
        fields.extend_from_slice(&(packed.len() as u32).to_le_bytes());
        fields.extend_from_slice(&(e.data.len() as u32).to_le_bytes());
        fields.extend_from_slice(&(e.name.len() as u16).to_le_bytes());
        fields.extend_from_slice(&[0, 0]);
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&fields);
        out.extend_from_slice(e.name.as_bytes());
        out.extend_from_slice(&packed);
        let mut record = b"PK\x01\x02\x14\x00".to_vec();
        record.extend_from_slice(&fields);
        record.extend_from_slice(&[0; 10]);
        record.extend_from_slice(&offset.to_le_bytes());
        edit(e.name, &mut record);
        dir.extend_from_slice(&record);
        dir.extend_from_slice(e.name.as_bytes());
    }
    let start = out.len() as u32;
    out.extend_from_slice(&dir);
    out.extend_from_slice(b"PK\x05\x06\0\0\0\0");
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(dir.len() as u32).to_le_bytes());
    out.extend_from_slice(&start.to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    out
}

pub(super) fn zip_file(entries: &[Entry]) -> Vec<u8> {
    zip_with(entries, |_, _| {})
}

/// Least-significant-bit-first output for deflate data.
struct Bits {
    out: Vec<u8>,
    bit: u32,
}

impl Bits {
    fn put(&mut self, value: u32, n: u32) {
        for i in 0..n {
            if self.bit == 0 {
                self.out.push(0);
            }
            *self.out.last_mut().unwrap() |= (((value >> i) & 1) as u8) << self.bit;
            self.bit = (self.bit + 1) % 8;
        }
    }

    /// A Huffman code, which deflate stores from its first bit.
    fn code(&mut self, code: u32, n: u32) {
        self.put(code.reverse_bits() >> (32 - n), n);
    }

    fn literal(&mut self, symbol: u32) {
        match symbol {
            0..=143 => self.code(0x30 + symbol, 8),
            144..=255 => self.code(0x190 + symbol - 144, 9),
            256..=279 => self.code(symbol - 256, 7),
            _ => self.code(0xc0 + symbol - 280, 8),
        }
    }
}

/// One final fixed-Huffman block of literals.
pub(super) fn deflate_fixed(data: &[u8]) -> Vec<u8> {
    let mut bits = Bits { out: Vec::new(), bit: 0 };
    bits.put(1, 1);
    bits.put(1, 2);
    for &b in data {
        bits.literal(u32::from(b));
    }
    bits.literal(256);
    bits.out
}

/// Fixed-Huffman data for `b'a'` then `n` copies of 258 bytes at distance 1:
/// a few bytes per 258 bytes of output.
pub(super) fn deflate_bomb(n: usize) -> Vec<u8> {
    let mut bits = Bits { out: Vec::new(), bit: 0 };
    bits.put(1, 1);
    bits.put(1, 2);
    bits.literal(u32::from(b'a'));
    for _ in 0..n {
        bits.literal(285);
        bits.code(0, 5);
    }
    bits.literal(256);
    bits.out
}

/// The events of `xml` as short strings: `<name`, `/name`, `text`.
fn events(xml: &str) -> Result<Vec<String>, String> {
    let mut scan = Scanner::new(xml, false);
    let mut out = Vec::new();
    while let Some(event) = scan.next()? {
        out.push(match event {
            Event::Open(tag) => format!("<{}", tag.name),
            Event::Close(name) => format!("/{name}"),
            Event::Text(t) => unescape(t)?.into_owned(),
            Event::Cdata(t) => format!("cdata:{t}"),
        });
    }
    scan.finish(&mut Budget::new(&Caps::default()))?;
    Ok(out)
}

#[test]
fn xml_scanner_reads_tags_text_and_references() {
    let xml = "<?xml version=\"1.0\"?><!DOCTYPE a [<!ENTITY big \"xxxx\">]><!-- c --><a x='1>2'><b/>t&amp;&#x41;&#66;<![CDATA[<raw>]]></a>";
    assert_eq!(events(xml).unwrap(), ["<a", "<b", "/b", "t&AB", "cdata:<raw>", "/a"]);
    let mut scan = Scanner::new("<w:p w:val=\"a&lt;b\" val='plain' r:id=\"rId1\" id=\"9\"/>", false);
    let Some(Event::Open(tag)) = scan.next().unwrap() else { panic!() };
    assert_eq!((tag.local(), tag.attr("val").unwrap().as_deref()), ("p", Some("plain")));
    assert_eq!(tag.attr("w:val").unwrap().as_deref(), Some("a<b"));
    assert_eq!((tag.attr("id").unwrap().as_deref(), tag.attr("r:id").unwrap().as_deref()), (Some("9"), Some("rId1")));
    assert_eq!(tag.attr("none").unwrap(), None);
    // A declared entity is never expanded, and broken markup fails.
    for (xml, why) in [
        ("<a>&big;</a>", "the entity &big; is not defined"),
        ("<a></b>", "the close tag </b> does not match <a> at byte 3"),
        ("<a>", "ends inside <a>"),
        ("<a x=1/>", "an attribute value has no quotes at byte 0"),
        ("<a><!-- open", "ends inside markup at byte 3"),
        ("</a>", "the close tag </a> has no open tag at byte 0"),
        ("<a>&#0;</a>", "the entity &#0; is not defined"),
    ] {
        assert_eq!(events(xml).unwrap_err(), why, "{xml}");
    }
    let deep = format!("{}{}", "<a>".repeat(xml::MAX_DEPTH + 1), "</a>".repeat(xml::MAX_DEPTH + 1));
    assert!(events(&deep).unwrap_err().contains("deeper than 512"));
    // A part cut at a cap may end anywhere.
    let mut scan = Scanner::new("<a><b>text</b><c x=\"", true);
    while scan.next().unwrap().is_some() {}
    assert!(scan.finish(&mut Budget::new(&Caps::default())).is_ok());
}

#[test]
fn zip_reads_stored_and_deflated_entries_and_checks_them() {
    let file = zip_file(&[stored("a.txt", "plain"), deflated("Dir/B.xml", "<b>deflated text</b>")]);
    let mut zip = Zip::open(&file, Format::Docx).unwrap();
    assert_eq!(zip.part("a.txt").unwrap().unwrap().data, b"plain");
    // Part names do not depend on case.
    let part = zip.part("dir/b.xml").unwrap().unwrap();
    assert_eq!((part.data.as_slice(), part.cut), (&b"<b>deflated text</b>"[..], false));
    assert!(zip.part("missing").unwrap().is_none());
    let damaged = |edit: &dyn Fn(&str, &mut [u8]), name: &str| {
        let file = zip_with(&[deflated("x.xml", "<x/>")], edit);
        Zip::open(&file, Format::Docx).and_then(|mut z| z.part(name).map(|_| ())).unwrap_err().to_string()
    };
    assert_eq!(damaged(&|_, r| r[16] ^= 1, "x.xml"), "damaged DOCX file: the entry x.xml fails its CRC-32 check; no text was read");
    assert_eq!(damaged(&|_, r| r[8] |= 1, "x.xml"), "this DOCX file is encrypted; Vade cannot read encrypted files");
    assert_eq!(damaged(&|_, r| r[10] = 12, "x.xml"), "damaged DOCX file: the entry x.xml uses compression method 12, which Vade does not read; no text was read");
    assert_eq!(damaged(&|_, r| r[24] = 2, "x.xml"), "damaged DOCX file: the entry x.xml holds more data than its size; no text was read");
    assert_eq!(damaged(&|_, r| r[24] = 9, "x.xml"), "damaged DOCX file: the entry x.xml has the wrong size; no text was read");
    assert_eq!(damaged(&|_, r| r[20] = 200, "x.xml"), "damaged DOCX file: an entry is larger than the file; no text was read");
    let twice = zip_file(&[stored("a", "1"), stored("A", "2")]);
    assert!(Zip::open(&twice, Format::Docx).err().unwrap().to_string().contains("two entries are named A"));
    // Every cut of a file fails cleanly or reads; none panics.
    for n in 0..file.len() {
        if let Ok(mut zip) = Zip::open(&file[..n], Format::Docx) {
            let _ = zip.part("dir/b.xml");
        }
    }
    assert_eq!(resolve("word", "media/a%20b.png"), "word/media/a b.png");
    assert_eq!(resolve("ppt/slides", "../media/x.png#frag"), "ppt/media/x.png");
    assert_eq!(resolve("word", "/customXml/item.xml"), "customXml/item.xml");
}

#[test]
fn zip_bombs_stop_at_the_part_cap() {
    // 4,000 copies of 258 bytes: about 1 MB from about 3 KB.
    let data = vec![b'a'; 1 + 258 * 4_000];
    let file = zip_file(&[Entry { name: "b.xml", data: data.clone(), deflate: true, raw: Some(deflate_bomb(4_000)) }]);
    assert!(file.len() < 8_000, "{}", file.len());
    let mut zip = Zip::open(&file, Format::Docx).unwrap();
    assert_eq!(zip.part("b.xml").unwrap().unwrap().data, data);
    let mut zip = Zip::open_with(&file, Format::Docx, 1000, 1 << 20).unwrap();
    let part = zip.part("b.xml").unwrap().unwrap();
    assert_eq!((part.data.len(), part.cut), (1000, true));
    assert_eq!(zip.warnings, ["b.xml after 1000 bytes (unzip cap)"]);
    // The read's total cap stops the decoder too.
    let mut zip = Zip::open_with(&file, Format::Docx, usize::MAX, 5000).unwrap();
    let part = zip.part("b.xml").unwrap().unwrap();
    assert!(part.cut && part.data.len() <= 5000, "{}", part.data.len());
}

pub(super) const RELS: &str = r#"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>"#;

const DOC_RELS: &str = r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.org/a b" TargetMode="External"/>
<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
</Relationships>"#;

const STYLES: &str = r#"<w:styles xmlns:w="w">
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:pPr><w:numPr><w:numId w:val="2"/></w:numPr></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="MyHead"><w:name w:val="Custom"/><w:basedOn w:val="Heading2"/></w:style>
<w:style w:type="paragraph" w:styleId="ListBullet"><w:name w:val="List Bullet"/><w:pPr><w:numPr><w:numId w:val="1"/></w:numPr></w:pPr></w:style>
</w:styles>"#;

const NUMBERING: &str = r#"<w:numbering xmlns:w="w">
<w:abstractNum w:abstractNumId="10"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/></w:lvl><w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/></w:lvl></w:abstractNum>
<w:abstractNum w:abstractNumId="20"><w:lvl w:ilvl="0"><w:start w:val="3"/><w:numFmt w:val="decimal"/></w:lvl><w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="lowerLetter"/></w:lvl></w:abstractNum>
<w:num w:numId="1"><w:abstractNumId w:val="10"/></w:num>
<w:num w:numId="2"><w:abstractNumId w:val="20"/></w:num>
</w:numbering>"#;

/// A DOCX file whose body is `body`, with the styles, lists and links above.
pub(super) fn docx(body: &str) -> Vec<u8> {
    docx_with(body, &[])
}

/// [`docx`] with some parts replaced: (part name, data).
fn docx_with(body: &str, parts: &[(&str, &str)]) -> Vec<u8> {
    let document = format!(r#"<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="w" xmlns:r="r"><w:body>{body}<w:sectPr/></w:body></w:document>"#);
    let defaults = [
        ("[Content_Types].xml", "<Types/>"),
        ("_rels/.rels", RELS),
        ("word/document.xml", document.as_str()),
        ("word/_rels/document.xml.rels", DOC_RELS),
        ("word/styles.xml", STYLES),
        ("word/numbering.xml", NUMBERING),
    ];
    let entries: Vec<Entry> = defaults.iter().map(|&(name, data)| deflated(name, parts.iter().find(|p| p.0 == name).map_or(data, |p| p.1))).collect();
    zip_file(&entries)
}

fn p(text: &str) -> String {
    format!("<w:p><w:r><w:t>{text}</w:t></w:r></w:p>")
}

fn styled(style: &str, text: &str) -> String {
    format!("<w:p><w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>")
}

fn item(num: u32, level: u32, text: &str) -> String {
    format!("<w:p><w:pPr><w:numPr><w:ilvl w:val=\"{level}\"/><w:numId w:val=\"{num}\"/></w:numPr></w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>")
}

fn docx_md(body: &str) -> String {
    read(&docx(body), "a.docx", &Caps::default()).unwrap().1.markdown
}

#[test]
fn docx_reads_headings_lists_links_tables_and_images() {
    let body = [
        styled("Title", "Report"),
        styled("MyHead", "Part &amp; one"),
        "<w:p><w:pPr><w:outlineLvl w:val=\"2\"/></w:pPr><w:r><w:t xml:space=\"preserve\">Outline </w:t></w:r></w:p>".into(),
        "<w:p><w:r><w:t>See </w:t></w:r><w:hyperlink r:id=\"rId3\"><w:r><w:t>the site</w:t></w:r></w:hyperlink><w:r><w:t>.</w:t><w:br/><w:t>Next</w:t><w:tab/><w:t>line</w:t></w:r></w:p>".into(),
        styled("ListBullet", "bullet from style"),
        item(1, 1, "inner bullet"),
        item(2, 0, "three"),
        item(2, 1, "sub a"),
        item(2, 0, "four"),
        p(""),
        "<w:tbl><w:tblPr><w:tblW w:w=\"0\"/></w:tblPr><w:tr><w:tc><w:p><w:r><w:t>Name</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr>\
         <w:tr><w:tc><w:tcPr><w:gridSpan w:val=\"2\"/></w:tcPr><w:p><w:r><w:t>wide|cell</w:t></w:r></w:p><w:p><w:r><w:t>two</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
            .into(),
        "<w:p><w:r><w:drawing><wp:inline><wp:docPr id=\"1\" name=\"Picture 1\" descr=\"A chart\"/><a:graphic><pic:pic><pic:blipFill><a:blip r:embed=\"rId4\"/></pic:blipFill></pic:pic></a:graphic></wp:inline></w:drawing></w:r></w:p>".into(),
    ]
    .concat();
    let (format, doc) = read(&docx(&body), "dir/a.docx", &Caps::default()).unwrap();
    assert_eq!(format, Format::Docx);
    assert_eq!(
        doc.markdown,
        "<!-- page 1 of 1 -->\n## Page 1\n\n# Report\n\n## Part & one\n\n### Outline\n\nSee [the site](https://example.org/a%20b).\nNext\tline\n\n\
         - bullet from style\n    - inner bullet\n3. three\n    1. sub a\n4. four\n\n\
         | Name | Value |\n| --- | --- |\n| wide\\|cell<br>two |  |\n\n[image: A chart (word/media/image1.png)]\n"
    );
    assert_eq!((doc.pages, doc.warnings), (1, vec!["1 images (listed, not read)".to_owned()]));
}

#[test]
fn docx_skips_deleted_text_field_codes_and_fallback_copies() {
    let body = "<w:p><w:r><w:t>kept </w:t></w:r><w:del><w:r><w:delText>gone</w:delText></w:r></w:del>\
        <w:r><w:fldChar w:fldCharType=\"begin\"/></w:r><w:r><w:instrText>PAGE</w:instrText></w:r><w:r><w:fldChar w:fldCharType=\"separate\"/></w:r><w:r><w:t>7</w:t></w:r>\
        <w:r><mc:AlternateContent><mc:Choice><w:drawing><wps:txbx><w:txbxContent><w:p><w:r><w:t>box</w:t></w:r></w:p></w:txbxContent></wps:txbx></w:drawing></mc:Choice>\
        <mc:Fallback><w:pict><w:p><w:r><w:t>box copy</w:t></w:r></w:p></w:pict></mc:Fallback></mc:AlternateContent></w:r></w:p>";
    assert_eq!(docx_md(body), "<!-- page 1 of 1 -->\n## Page 1\n\nkept 7 box\n");
}

#[test]
fn docx_page_markers_follow_breaks() {
    let page = |n: usize, of: usize| page_marker(n, of);
    // Hard breaks: a page break run, pageBreakBefore and a section break.
    let body = [
        p("one"),
        "<w:p><w:r><w:t>two a</w:t><w:br w:type=\"page\"/><w:t>two b</w:t></w:r></w:p>".into(),
        "<w:p><w:pPr><w:pageBreakBefore/></w:pPr><w:r><w:t>three</w:t></w:r></w:p>".into(),
        "<w:p><w:pPr><w:sectPr><w:type w:val=\"continuous\"/></w:sectPr></w:pPr><w:r><w:t>same page</w:t></w:r></w:p>".into(),
        "<w:p><w:pPr><w:sectPr/></w:pPr><w:r><w:t>section end</w:t></w:r></w:p>".into(),
        p("four"),
    ]
    .concat();
    let md = docx_md(&body);
    let want = format!(
        "{}one\n\ntwo a\n{}two b\n{}three\n\nsame page\n\nsection end\n{}four\n",
        page(1, 4).trim_start(),
        page(2, 4),
        page(3, 4),
        page(4, 4)
    );
    assert_eq!(md, want);
    // Word's rendered marks give the pages; hard breaks then add none.
    let body = [
        p("one"),
        "<w:p><w:r><w:br w:type=\"page\"/></w:r><w:r><w:lastRenderedPageBreak/><w:t>two</w:t></w:r></w:p>".into(),
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>a</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:lastRenderedPageBreak/><w:t>b</w:t></w:r></w:p></w:tc></w:tr></w:tbl>".into(),
        p("after"),
    ]
    .concat();
    let want = format!("{}one\n{}two\n\n| a | b |\n| --- | --- |\n{}after\n", page(1, 3).trim_start(), page(2, 3), page(3, 3));
    assert_eq!(docx_md(&body), want);
}

#[test]
fn docx_caps_pages_chars_and_work() {
    let body: String = (1..=5).map(|i| format!("<w:p><w:r><w:t>page {i}</w:t><w:br w:type=\"page\"/></w:r></w:p>")).collect();
    let caps = Caps { max_pages: 2, ..Caps::default() };
    let (_, doc) = read(&docx(&body), "a.docx", &caps).unwrap();
    assert_eq!(doc.markdown, format!("{}page 1\n{}page 2\n", page_marker(1, 6).trim_start(), page_marker(2, 6)));
    assert_eq!((doc.pages, doc.warnings), (6, vec!["pages after 2 of 6 (pages cap)".to_owned()]));
    // A long document stops at the char cap, and its markers still show.
    let body: String = (0..2000).map(|i| p(&format!("paragraph {i} {}", "x".repeat(50)))).collect();
    let caps = Caps { max_chars: 500, ..Caps::default() };
    let (_, doc) = read(&docx(&body), "a.docx", &caps).unwrap();
    assert!(doc.markdown.starts_with("<!-- page 1 of 1 -->\n## Page 1\n\nparagraph 0 x"));
    assert!(doc.chars == 500 && doc.markdown.len() < 1000, "{}", doc.markdown.len());
    assert_eq!(doc.warnings, ["text after 500 chars (char cap)"]);
    // A wide table that would grow large stops at the work cap. The text
    // read before it, its first rows and the page markers stay whole.
    let row = format!("<w:tr>{}</w:tr>", "<w:tc><w:tcPr><w:gridSpan w:val=\"64\"/></w:tcPr><w:p><w:r><w:t>v</w:t></w:r></w:p></w:tc>".repeat(50));
    let table = format!("<w:tbl>{}{}</w:tbl>", "<w:tr><w:tc><w:p><w:r><w:t>kept</w:t></w:r></w:p></w:tc></w:tr>", row.repeat(200));
    let body = format!("{}{table}<w:p><w:r><w:br w:type=\"page\"/><w:t>after</w:t></w:r></w:p>", p("Hello"));
    let caps = Caps { max_chars: 1000, ..Caps::default() };
    let (_, doc) = read(&docx(&body), "a.docx", &caps).unwrap();
    assert!(doc.markdown.starts_with("<!-- page 1 of 2 -->\n## Page 1\n\nHello\n\n| kept |  |"), "{:?}", &doc.markdown[..60]);
    assert!(doc.markdown.lines().filter(|l| l.starts_with("<!--")).all(|l| l.ends_with(" -->")));
    // The work cap stops the reader; the char cap then cuts the long table rows.
    assert_eq!(doc.chars, 1000);
    assert_eq!(doc.warnings, [format!("text after {MIN_WORK_BYTES} bytes of conversion work (memory cap)"), "text after 1000 chars (char cap)".to_owned()]);
    // Links that stop at the work cap stay plain text; none is cut open.
    let links = format!("<w:p>{}</w:p>", "<w:hyperlink r:id=\"rId3\"><w:r><w:t>x</w:t></w:r></w:hyperlink>".repeat(3000));
    // The reader's own output: `read` then cuts it at the char cap anywhere.
    let doc = super::docx::DocxReader.to_markdown(&docx(&links), Format::Docx, &Caps { max_chars: 4096, ..Caps::default() }).unwrap();
    let (open, closed) = (doc.markdown.matches("[x](").count(), doc.markdown.matches("[x](https://example.org/a%20b)").count());
    assert!(open > 1000 && open < 3000 && open == closed, "{open} {closed}");
    assert!(doc.markdown.starts_with("<!-- page 1 of 1 -->\n## Page 1\n\n[x]("));
}

#[test]
fn docx_metadata_at_the_work_cap_gives_the_cap_warning() {
    let caps = Caps { max_chars: 1000, ..Caps::default() };
    let styles = format!("<w:styles xmlns:w=\"w\">{}</w:styles>", (0..2000).map(|i| format!("<w:style w:styleId=\"S{i}\"><w:name w:val=\"s {i}\"/></w:style>")).collect::<String>());
    let numbering = format!(
        "<w:numbering xmlns:w=\"w\">{}</w:numbering>",
        (0..2000).map(|i| format!("<w:abstractNum w:abstractNumId=\"{i}\"><w:lvl w:ilvl=\"0\"><w:numFmt w:val=\"bullet\"/></w:lvl></w:abstractNum>")).collect::<String>()
    );
    let rels = format!(
        "<Relationships>{}</Relationships>",
        (0..2000).map(|i| format!("<Relationship Id=\"r{i}\" Type=\"t/hyperlink\" Target=\"https://example.org/{i}\" TargetMode=\"External\"/>")).collect::<String>()
    );
    // A part target longer than a ZIP entry name can be names no part.
    let n = 100_000;
    let long = format!("<Relationships><Relationship Id=\"rId1\" Type=\"t/officeDocument\" Target=\"{}{}word/document.xml\"/></Relationships>", "a/".repeat(n), "../".repeat(n));
    let err = read(&docx_with(&p("kept"), &[("_rels/.rels", &long)]), "a.docx", &caps).unwrap_err().to_string();
    assert!(err.contains("_rels/.rels: has a part path longer than 65535 bytes"), "{err}");
    let parts = [("word/_rels/document.xml.rels", &rels), ("word/styles.xml", &styles), ("word/numbering.xml", &numbering)];
    for (name, data) in parts {
        let (_, doc) = read(&docx_with(&p("kept"), &[(name, data)]), "a.docx", &caps).unwrap();
        assert_eq!(doc.warnings, [format!("text after {MIN_WORK_BYTES} bytes of conversion work (memory cap)")], "{name}");
        assert!(doc.markdown.starts_with("<!-- page 1 of 1 -->\n## Page 1\n"), "{name}");
    }
    assert_eq!(resolve("", &format!("{}{}word/document.xml", "a/".repeat(n), "../".repeat(n))), "word/document.xml");
    // Relationships are found by id; the first of two with one id wins.
    let rels = format!(
        "<Relationships>{}<Relationship Id=\"rId3\" Type=\"t/hyperlink\" Target=\"https://example.org/second\" TargetMode=\"External\"/></Relationships>",
        (0..8000).map(|i| format!("<Relationship Id=\"rId{}\" Type=\"t/hyperlink\" Target=\"https://example.org/{i}\" TargetMode=\"External\"/>", i + 3)).collect::<String>()
    );
    let body = format!("<w:p><w:hyperlink r:id=\"rId3\"><w:r><w:t>first</w:t></w:r></w:hyperlink></w:p><w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>{}", "<w:p><w:hyperlink r:id=\"rId8002\"/></w:p>".repeat(8000));
    let (_, doc) = read(&docx_with(&body, &[("word/_rels/document.xml.rels", &rels)]), "a.docx", &Caps { max_pages: 1, ..Caps::default() }).unwrap();
    assert_eq!(doc.markdown, "<!-- page 1 of 2 -->\n## Page 1\n\n[first](https://example.org/0)\n");
}

#[test]
fn docx_list_overrides_and_style_levels() {
    let numbering = r#"<w:numbering xmlns:w="w">
<w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/></w:lvl><w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/></w:lvl></w:abstractNum>
<w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/></w:lvl></w:abstractNum>
<w:num w:numId="1"><w:abstractNumId w:val="0"/><w:lvlOverride w:ilvl="0"><w:lvl w:ilvl="0"><w:start w:val="7"/></w:lvl></w:lvlOverride></w:num>
<w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>
<w:num w:numId="3"><w:abstractNumId w:val="1"/><w:lvlOverride w:ilvl="0"><w:startOverride w:val="4"/></w:lvlOverride></w:num>
</w:numbering>"#;
    // `ilvl` comes before `numId` in the schema order.
    let styles = r#"<w:styles xmlns:w="w"><w:style w:styleId="Nested"><w:name w:val="Nested"/><w:pPr><w:numPr><w:ilvl w:val="1"/><w:numId w:val="1"/></w:numPr></w:pPr></w:style></w:styles>"#;
    let body = [item(1, 0, "target"), item(2, 0, "other"), item(3, 0, "restart"), styled("Nested", "nested")].concat();
    let (_, doc) = read(&docx_with(&body, &[("word/numbering.xml", numbering), ("word/styles.xml", styles)]), "a.docx", &Caps::default()).unwrap();
    assert_eq!(doc.markdown, "<!-- page 1 of 1 -->\n## Page 1\n\n7. target\n1. other\n4. restart\n    - nested\n");
}

#[test]
fn docx_page_mode_and_properties_come_from_live_elements() {
    let hard = "<w:p><w:r><w:t>first</w:t><w:br w:type=\"page\"/><w:t>second</w:t></w:r></w:p>";
    let two = |before: &str| format!("<!-- page 1 of 2 -->\n## Page 1\n\n{before}first\n\n<!-- page 2 of 2 -->\n## Page 2\n\nsecond\n");
    // The word in a comment, in text, in deleted text or in a text box does
    // not turn off hard page breaks.
    assert_eq!(docx_md(&format!("<!-- lastRenderedPageBreak -->{hard}")), two(""));
    assert_eq!(docx_md(&format!("{}{hard}", p("lastRenderedPageBreak"))), two("lastRenderedPageBreak\n\n"));
    let deleted = "<w:p><w:del><w:r><w:lastRenderedPageBreak/><w:delText>x</w:delText></w:r></w:del></w:p>";
    assert_eq!(docx_md(&format!("{deleted}{hard}")), two(""));
    let boxed = "<w:p><w:r><w:drawing><w:txbxContent><w:p><w:r><w:lastRenderedPageBreak/></w:r></w:p></w:txbxContent></w:drawing></w:r></w:p>";
    assert_eq!(docx_md(&format!("{boxed}{hard}")), two(""));
    // A live mark gives the pages, and the hard break is not counted twice.
    let marked = "<w:p><w:r><w:t>first</w:t><w:br w:type=\"page\"/><w:lastRenderedPageBreak/><w:t>second</w:t></w:r></w:p>";
    assert_eq!(docx_md(marked), two(""));
    // The old properties of a tracked change do not apply.
    let old = "<w:p><w:pPr><w:pStyle w:val=\"Normal\"/><w:pPrChange w:id=\"1\"><w:pPr><w:pStyle w:val=\"Title\"/></w:pPr></w:pPrChange></w:pPr><w:r><w:t>current normal</w:t></w:r></w:p>";
    assert_eq!(docx_md(old), "<!-- page 1 of 1 -->\n## Page 1\n\ncurrent normal\n");
}

#[test]
fn zip_records_from_the_end_are_checked_before_use() {
    // A ZIP64 locator whose record offset is the largest number is damage.
    let mut file = b"PK\x03\x04PK\x06\x07\0\0\0\0".to_vec();
    file.extend_from_slice(&u64::MAX.to_le_bytes());
    file.extend_from_slice(&[1, 0, 0, 0]);
    file.extend_from_slice(b"PK\x05\x06\0\0\0\0\xff\xff\xff\xff\0\0\0\0\0\0\0\0\0\0");
    assert_eq!(file.len(), 46);
    assert_eq!(Zip::open(&file, Format::Docx).err().unwrap().to_string(), "damaged DOCX file: the ZIP64 record is out of the file; no text was read");
    // A comment may hold a false end record; the real one is found.
    let fakes: [&[u8]; 2] = [b"PK\x05\x06\0\0\0\0\x01\0\x01\0\x2e\0\0\0\0\0\0\0\0\0", b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"];
    for fake in fakes {
        let mut file = docx(&p("kept"));
        let n = file.len();
        file[n - 2..].copy_from_slice(&(fake.len() as u16).to_le_bytes());
        file.extend_from_slice(fake);
        assert_eq!(read(&file, "a.docx", &Caps::default()).unwrap().1.markdown, "<!-- page 1 of 1 -->\n## Page 1\n\nkept\n");
    }
    // A stored part keeps the bytes that the read's total cap has left.
    let file = zip_file(&[stored("a.txt", &"s".repeat(6000))]);
    let mut zip = Zip::open_with(&file, Format::Docx, usize::MAX, 5000).unwrap();
    let part = zip.part("a.txt").unwrap().unwrap();
    assert_eq!((part.data.len(), part.cut), (5000, true));
    assert_eq!(zip.warnings, ["a.txt after 5000 bytes (unzip cap)"]);
    // A part cut inside a reference keeps the text before it.
    let file = zip_file(&[stored("a.xml", "<a>kept &amp; more</a>")]);
    let mut zip = Zip::open_with(&file, Format::Docx, 11, 1 << 20).unwrap();
    let (text, cut) = zip.text("a.xml").unwrap().unwrap();
    assert_eq!((text.as_str(), cut), ("<a>kept &am", true));
    let mut scan = Scanner::new(&text, cut);
    let mut texts = Vec::new();
    while let Some(event) = scan.next().unwrap() {
        if let Event::Text(t) = event {
            texts.push(unescape(t).unwrap().into_owned());
        }
    }
    assert!(scan.finish(&mut Budget::new(&Caps::default())).is_ok());
    assert_eq!(texts, ["kept "]);
    // Only the start of a text that can give the bytes kept is decoded.
    assert_eq!(unescape_prefix(&"&amp;".repeat(100), 2).unwrap(), "&&&&&&");
    assert_eq!(unescape_prefix("a &lt; b", 100).unwrap(), "a < b");
}

#[test]
fn docx_scans_each_byte_a_fixed_number_of_times() {
    let scanned = |paragraphs: usize| {
        let body: String = (0..paragraphs).map(|i| format!("<w:p><w:r><w:t>p{i}</w:t></w:r><w:del><w:r><w:delText>d</w:delText></w:r></w:del></w:p>")).collect();
        let file = docx(&body);
        SCANNED.with(|n| n.set(0));
        read(&file, "a.docx", &Caps::default()).unwrap();
        (SCANNED.with(|n| n.get()), body.len())
    };
    let (small, n) = scanned(1_000);
    let (large, m) = scanned(10_000);
    assert!(small <= 4 * (n + 4096) && large <= 4 * (m + 4096), "{small} for {n}, {large} for {m}");
}

#[test]
fn docx_damaged_encrypted_and_legacy_files_give_one_line() {
    let err = |bytes: &[u8]| read(bytes, "a.docx", &Caps::default()).unwrap_err().to_string();
    assert_eq!(err(b"not a zip at all"), "damaged DOCX file: the file is not a ZIP archive; no text was read");
    let mut ole = super::OLE.to_vec();
    ole.extend_from_slice(&[0; 64]);
    assert_eq!(err(&ole), "Vade does not read legacy Office (OLE) files; no text was read");
    ole.extend("EncryptedPackage".encode_utf16().flat_map(u16::to_le_bytes));
    assert_eq!(err(&ole), "this DOCX file is encrypted; Vade cannot read encrypted files");
    let bad = docx("<w:p><w:r><w:t>x</w:r></w:p>");
    assert_eq!(err(&bad), "damaged DOCX file: word/document.xml the close tag </w:r> does not match <w:t> at byte 98; no text was read");
    let empty = zip_file(&[stored("_rels/.rels", RELS)]);
    assert_eq!(err(&empty), "damaged DOCX file: the archive has no word/document.xml; no text was read");
    // No cut of a valid file panics.
    let file = docx(&p("hello"));
    for n in (0..file.len()).step_by(7) {
        let _ = read(&file[..n], "a.docx", &Caps::default());
    }
    // A name that is not .docx keeps the ZIP archive line.
    assert_eq!(read(&file, "a.zip", &Caps::default()).unwrap_err().to_string(), "Vade does not read ZIP archive files; no text was read");
}

const BOOK_RELS: &str = r#"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#;

const WORKBOOK_RELS: &str = r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="/xl/worksheets/sheet2.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"#;

const SHARED: &str = r#"<sst xmlns="x" count="4"><si><t>Item</t></si><si><t>Price</t></si><si><r><t>pen</t></r><r><t xml:space="preserve"> | ink</t></r><rPh><t>PHON</t></rPh></si><si><t>When</t></si></sst>"#;

const XL_STYLES: &str = r#"<styleSheet><numFmts count="1"><numFmt numFmtId="164" formatCode="yyyy\-mm\-dd&quot; at &quot;hh:mm"/></numFmts>
<cellStyleXfs count="1"><xf numFmtId="14"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0"/><xf numFmtId="14"/><xf numFmtId="164"/></cellXfs></styleSheet>"#;

/// An XLSX file with two sheets: `sheet1` and `sheet2` are the sheetData contents.
pub(super) fn xlsx(sheet1: &str, sheet2: &str) -> Vec<u8> {
    let sheet = |rows: &str| format!(r#"<?xml version="1.0"?><worksheet xmlns="x" xmlns:r="r"><dimension ref="A1"/><sheetData>{rows}</sheetData></worksheet>"#);
    zip_file(&[
        stored("[Content_Types].xml", "<Types/>"),
        stored("_rels/.rels", BOOK_RELS),
        stored("xl/workbook.xml", r#"<workbook xmlns:r="r"><workbookPr/><sheets><sheet name="Sales &amp; costs" sheetId="1" r:id="rId1"/><sheet name="Old" sheetId="2" state="hidden" r:id="rId2"/></sheets></workbook>"#),
        stored("xl/_rels/workbook.xml.rels", WORKBOOK_RELS),
        deflated("xl/sharedStrings.xml", SHARED),
        stored("xl/styles.xml", XL_STYLES),
        deflated("xl/worksheets/sheet1.xml", &sheet(sheet1)),
        stored("xl/worksheets/sheet2.xml", &sheet(sheet2)),
    ])
}

const SALES: &str = r#"<row r="1"><c r="A1" t="s"><v>0</v></c><c r="C1" t="s"><v>1</v></c><c r="D1" t="s"><v>3</v></c></row>
<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2" t="b"><v>1</v></c><c r="C2"><v>1.1000000000000001</v></c><c r="D2" s="1"><v>44927</v></c></row>
<row r="3"><c r="A3" s="1"/></row>
<row r="5"><c r="A5" t="inlineStr"><is>
<t>inline</t>
</is></c><c r="B5" t="e"><v>#DIV/0!</v></c><c r="C5" t="str"><f>A1&amp;"x"</f><v>Itemx</v></c><c r="D5" s="2"><v>44927.75</v></c></row>"#;

#[test]
fn xlsx_reads_each_sheet_as_a_table() {
    let (format, doc) = read(&xlsx(SALES, ""), "book.xlsx", &Caps::default()).unwrap();
    assert_eq!(format, Format::Xlsx);
    assert_eq!(
        doc.markdown,
        "<!-- sheet 1 of 2 -->\n## Sheet Sales & costs\n\n\
         | Item |  | Price | When |\n| --- | --- | --- | --- |\n| pen \\| ink | TRUE | 1.1 | 2023-01-01 |\n| inline | #DIV/0! | Itemx | 2023-01-01 18:00:00 |\n\
         \n<!-- sheet 2 of 2 -->\n## Sheet Old (hidden)\n\n(empty sheet)\n"
    );
    assert_eq!((doc.pages, doc.warnings), (2, vec![]));
}

#[test]
fn xlsx_dates_and_formats() {
    use super::xlsx::{test_date, test_format};
    for (serial, date1904, want) in [(44927.0, false, "2023-01-01"), (1.0, false, "1900-01-01"), (60.0, false, "1900-02-29"), (61.0, false, "1900-03-01"), (0.0, true, "1904-01-01"), (45230.5, false, "2023-10-31 12:00:00")] {
        assert_eq!(test_date(serial, date1904), want, "{serial}");
    }
    for (code, want) in [
        ("yyyy-mm-dd", "date"),
        ("h:mm AM/PM", "time"),
        ("[h]:mm:ss", "elapsed"),
        ("mm:ss", "time"),
        ("mmm-yy", "date"),
        ("d/m/yyyy h:mm", "datetime"),
        ("0.00", "no"),
        ("\"day\"0", "no"),
        ("[Red]#,##0;[Blue]\\d0", "no"),
        ("General", "no"),
    ] {
        assert_eq!(test_format(code), want, "{code}");
    }
}

#[test]
fn xlsx_caps_rows_sheets_and_work() {
    let rows: String = (1..=6).map(|r| format!("<row r=\"{r}\"><c r=\"A{r}\"><v>{r}</v></c></row>")).collect();
    let caps = Caps { max_rows: 2, max_pages: 1, ..Caps::default() };
    let (_, doc) = read(&xlsx(&rows, ""), "a.xlsx", &caps).unwrap();
    assert_eq!(doc.markdown, "<!-- sheet 1 of 2 -->\n## Sheet Sales & costs\n\n| 1 |\n| --- |\n| 2 |\n| 3 |\n");
    assert_eq!(doc.warnings, ["rows after 2 of 5 in sheet Sales & costs (row cap)", "sheets after 1 of 2 (sheets cap)"]);
    // One far cell makes every row wide; the work cap stops the table. The
    // first row keeps both of its values, and the marker stays whole.
    let rows: String = (1..=300).map(|r| format!("<row r=\"{r}\"><c r=\"A{r}\"><v>1</v></c><c r=\"XFD{r}\"><v>2</v></c></row>")).collect();
    let caps = Caps { max_chars: 1000, ..Caps::default() };
    let own = super::xlsx::XlsxReader.to_markdown(&xlsx(&rows, ""), Format::Xlsx, &caps).unwrap();
    let first = own.markdown.lines().nth(3).unwrap();
    assert!(first.starts_with("| 1 |  |") && first.ends_with("|  | 2 |"), "{}", &first[..40]);
    assert!(own.markdown.len() <= MIN_WORK_BYTES + 200, "{}", own.markdown.len());
    assert_eq!(own.warnings, [format!("text after {MIN_WORK_BYTES} bytes of conversion work (memory cap)")]);
    let (_, doc) = read(&xlsx(&rows, ""), "a.xlsx", &caps).unwrap();
    assert!(doc.markdown.starts_with("<!-- sheet 1 of 2 -->\n## Sheet Sales & costs\n\n| 1 |  |"));
    assert_eq!(doc.warnings, [format!("text after {MIN_WORK_BYTES} bytes of conversion work (memory cap)"), "text after 1000 chars (char cap)".to_owned()]);
    // The char cap stops the cells.
    let rows: String = (1..=5000).map(|r| format!("<row r=\"{r}\"><c r=\"A{r}\" t=\"inlineStr\"><is><t>row {r}</t></is></c></row>")).collect();
    let caps = Caps { max_chars: 300, ..Caps::default() };
    let (_, doc) = read(&xlsx(&rows, ""), "a.xlsx", &caps).unwrap();
    assert!(doc.chars <= 300 && doc.markdown.len() < 2000, "{}", doc.markdown.len());
    assert_eq!(doc.warnings, ["text after 300 chars (char cap)"]);
}

#[test]
fn xlsx_damaged_files_give_one_line() {
    let err = |sheet: &str| read(&xlsx(sheet, ""), "a.xlsx", &Caps::default()).unwrap_err().to_string();
    assert_eq!(err("<row><c r=\"A0B\"><v>1</v></c></row>"), "damaged XLSX file: xl/worksheets/sheet1.xml has a cell reference that is not valid: A0B; no text was read");
    assert_eq!(err("<row><c r=\"A1\" t=\"s\"><v>9</v></c></row>"), "damaged XLSX file: xl/worksheets/sheet1.xml refers to shared string 9; the table has 4; no text was read");
    assert_eq!(err("<row r=\"0\"/>"), "damaged XLSX file: xl/worksheets/sheet1.xml has a row number that is not valid: 0; no text was read");
    assert_eq!(err("<row><c>"), "damaged XLSX file: xl/worksheets/sheet1.xml the close tag </sheetData> does not match <c> at byte 94; no text was read");
    let file = xlsx(SALES, "");
    for n in (0..file.len()).step_by(11) {
        let _ = read(&file[..n], "a.xlsx", &Caps::default());
    }
}

const DECK_RELS: &str = r#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>"#;

const PRESENTATION_RELS: &str = r#"<Relationships>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide2.xml"/>
<Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
</Relationships>"#;

const SLIDE_RELS: &str = r#"<Relationships>
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image1.png"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/>
</Relationships>"#;

fn shape(placeholder: &str, paragraphs: &str) -> String {
    format!("<p:sp><p:nvSpPr><p:cNvPr id=\"2\" name=\"S\"/><p:cNvSpPr/><p:nvPr>{placeholder}</p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>{paragraphs}</p:txBody></p:sp>")
}

fn ap(level: u32, text: &str) -> String {
    format!("<a:p><a:pPr lvl=\"{level}\"/><a:r><a:rPr lang=\"en-US\"/><a:t>{text}</a:t></a:r></a:p>")
}

/// A deck whose presentation lists slide 2 first, then slide 1.
pub(super) fn pptx(slide1: &str) -> Vec<u8> {
    let slide = |show: &str, tree: &str| format!(r#"<?xml version="1.0"?><p:sld xmlns:p="p" xmlns:a="a" xmlns:r="r"{show}><p:cSld><p:spTree>{tree}</p:spTree></p:cSld></p:sld>"#);
    let notes = slide("", &[shape("<p:ph type=\"sldImg\"/>", ""), shape("<p:ph type=\"body\" idx=\"1\"/>", &ap(0, "Say hello first.")), shape("<p:ph type=\"sldNum\"/>", &ap(0, "1"))].concat());
    zip_file(&[
        stored("[Content_Types].xml", "<Types/>"),
        stored("_rels/.rels", DECK_RELS),
        stored("ppt/presentation.xml", r#"<p:presentation xmlns:p="p" xmlns:r="r"><p:sldIdLst><p:sldId id="257" r:id="rId3"/><p:sldId id="256" r:id="rId2"/></p:sldIdLst></p:presentation>"#),
        stored("ppt/_rels/presentation.xml.rels", PRESENTATION_RELS),
        deflated("ppt/slides/slide1.xml", &slide("", slide1)),
        stored("ppt/slides/_rels/slide1.xml.rels", SLIDE_RELS),
        stored("ppt/slides/slide2.xml", &slide(" show=\"0\"", "")),
        stored("ppt/notesSlides/notesSlide1.xml", &notes),
    ])
}

#[test]
fn pptx_reads_slides_in_order_with_titles_bullets_tables_and_notes() {
    let tree = [
        shape("<p:ph type=\"title\"/>", &[ap(0, "Plan &amp; goals"), ap(0, "for Q4")].concat()),
        shape("<p:ph idx=\"1\"/>", &[ap(0, "Ship"), ap(1, "readers"), ap(0, "Test")].concat()),
        shape("", &["<a:p><a:r><a:t>Free text</a:t></a:r><a:br/><a:r><a:t>line two</a:t></a:r></a:p>".to_owned(), "<a:p><a:pPr><a:buChar char=\"*\"/></a:pPr><a:r><a:t>starred</a:t></a:r></a:p>".into()].concat()),
        shape("<p:ph type=\"sldNum\" idx=\"12\"/>", &ap(0, "7")),
        "<p:graphicFrame><a:graphic><a:graphicData><a:tbl><a:tblPr/><a:tr><a:tc><a:txBody><a:p><a:r><a:t>k</a:t></a:r></a:p></a:txBody><a:tcPr/></a:tc><a:tc><a:txBody><a:p><a:r><a:t>v</a:t></a:r></a:p></a:txBody></a:tc></a:tr><a:tr><a:tc><a:txBody><a:p><a:r><a:t>1</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>a|b</a:t></a:r></a:p></a:txBody></a:tc></a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>".into(),
        "<p:pic><p:nvPicPr><p:cNvPr id=\"5\" name=\"Picture 4\" descr=\"Sales chart\"/></p:nvPicPr><p:blipFill><a:blip r:embed=\"rId2\"/></p:blipFill></p:pic>".into(),
        "<mc:AlternateContent><mc:Choice>".to_owned() + &shape("", &ap(0, "new")) + "</mc:Choice><mc:Fallback>" + &shape("", &ap(0, "old copy")) + "</mc:Fallback></mc:AlternateContent>",
    ]
    .concat();
    let (format, doc) = read(&pptx(&tree), "deck.pptx", &Caps::default()).unwrap();
    assert_eq!(format, Format::Pptx);
    assert_eq!(
        doc.markdown,
        "<!-- slide 1 of 2 -->\n## Slide 1\n\n(hidden slide)\n\
         \n<!-- slide 2 of 2 -->\n## Slide 2\n\n### Plan & goals for Q4\n\n- Ship\n    - readers\n- Test\n\nFree text\nline two\n\n- starred\n\n\
         | k | v |\n| --- | --- |\n| 1 | a\\|b |\n\n[image: Sales chart (ppt/media/image1.png)]\n\nnew\n\n#### Notes\n\nSay hello first.\n"
    );
    assert_eq!((doc.pages, doc.warnings), (2, vec!["1 images (listed, not read)".to_owned()]));
    let caps = Caps { max_pages: 1, ..Caps::default() };
    let (_, doc) = read(&pptx(&tree), "deck.pptx", &caps).unwrap();
    assert_eq!(doc.warnings, ["slides after 1 of 2 (slides cap)"]);
    let err = read(&pptx("<p:sp>"), "deck.pptx", &Caps::default()).unwrap_err().to_string();
    assert_eq!(err, "damaged PPTX file: ppt/slides/slide1.xml the close tag </p:spTree> does not match <p:sp> at byte 88; no text was read");
    let file = pptx(&tree);
    for n in (0..file.len()).step_by(13) {
        let _ = read(&file[..n], "deck.pptx", &Caps::default());
    }
}

#[test]
fn images_give_one_line_with_size_and_path() {
    let line = |bytes: &[u8], name: &str| read(bytes, name, &Caps::default()).map(|(f, d)| (f.label(), d.markdown, d.warnings));
    let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
    png.extend_from_slice(&[0, 0, 2, 128, 0, 0, 1, 224, 8, 6, 0, 0, 0, 1, 2, 3, 4]);
    let (label, md, warnings) = line(&png, "pics/a.png").unwrap();
    assert_eq!(label, "PNG image");
    assert_eq!(md, "PNG image: 640 x 480 pixels, 8-bit RGBA. The file is pics/a.png. Vade does not run text recognition (OCR), so no text was read from the pixels.\n");
    assert_eq!(warnings, ["image pixels (no text recognition)"]);
    // JPEG: an APP0 segment, fill bytes, then a progressive frame header.
    let jpeg = b"\xff\xd8\xff\xe0\x00\x04ab\xff\xff\xc2\x00\x11\x08\x01\x2c\x01\x90\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01";
    assert!(line(jpeg, "a.jpg").unwrap().1.starts_with("JPEG image: 400 x 300 pixels, color, progressive. The file is a.jpg."));
    // GIF: a global color table, two frames and an extension.
    let mut gif = b"GIF89a\x0a\x00\x05\x00\x80\x00\x00".to_vec();
    gif.extend_from_slice(&[0; 6]);
    for _ in 0..2 {
        gif.extend_from_slice(b"\x21\xf9\x04\x00\x00\x00\x00\x00\x2c\0\0\0\0\x0a\x00\x05\x00\x00\x02\x02\x44\x01\x00");
    }
    gif.push(0x3b);
    assert!(line(&gif, "a.gif").unwrap().1.starts_with("GIF image: 10 x 5 pixels, animated, 2 frames."));
    let webp = |chunk: &[u8], body: &[u8]| [b"RIFF", &(12 + body.len() as u32).to_le_bytes()[..], b"WEBP", chunk, &(body.len() as u32).to_le_bytes(), body].concat();
    let lossy = webp(b"VP8 ", b"\0\0\0\x9d\x01\x2a\x20\x03\x58\x02");
    assert!(line(&lossy, "a.webp").unwrap().1.starts_with("WebP image: 800 x 600 pixels, lossy."));
    let lossless = webp(b"VP8L", &[0x2f, 0x3f, 0x40, 0x06, 0x00]);
    assert!(line(&lossless, "a.webp").unwrap().1.starts_with("WebP image: 64 x 26 pixels, lossless."), "{:?}", line(&lossless, "a.webp"));
    let extended = webp(b"VP8X", &[0x02, 0, 0, 0, 0x1f, 0, 0, 0x0f, 0, 0]);
    assert!(line(&extended, "a.webp").unwrap().1.starts_with("WebP image: 32 x 16 pixels, animated."));
    // Cut and broken headers are damaged files.
    for (bytes, why) in [
        (&png[..20], "damaged PNG image file: the PNG header is cut short; no text was read"),
        (&jpeg[..12], "damaged JPEG image file: the JPEG data ends before the frame header; no text was read"),
        (b"\xff\xd8\xff\xda\x00\x02", "damaged JPEG image file: the JPEG data has no frame header; no text was read"),
        (&gif[..30], "damaged GIF image file: the GIF data is cut short; no text was read"),
        (&lossy[..24], "damaged WebP image file: the WebP header is cut short; no text was read"),
        // The size must come from inside the declared header.
        (&png[..29], "damaged PNG image file: the PNG header is cut short; no text was read"),
        (b"\xff\xd8\xff\xc0\x00\x02\x08\x01\x2c\x01\x90\x03\x01\x22\x00", "damaged JPEG image file: the JPEG frame header is shorter than its fields; no text was read"),
        (&jpeg[..27], "damaged JPEG image file: the JPEG data ends before the frame header; no text was read"),
        (&[&extended[..16], &[0; 4], &extended[20..]].concat(), "damaged WebP image file: the first WebP chunk is shorter than its header; no text was read"),
        (&[b"RIFF\x04\0\0\0", &extended[8..]].concat(), "damaged WebP image file: the first WebP chunk ends after the RIFF data; no text was read"),
    ] {
        assert_eq!(line(bytes, "x").unwrap_err().to_string(), why);
    }
    for n in 0..gif.len() {
        let _ = line(&gif[..n], "a.gif");
        let _ = line(&jpeg[..n.min(jpeg.len())], "a.jpg");
    }
}

const CONTAINER: &str = r#"<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"#;

const OPF: &str = r#"<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>A  Short &amp; Small Book</dc:title><dc:creator>Ann Author</dc:creator><dc:creator>Bo Writer</dc:creator></metadata>
<manifest><item id="c1" href="text/one.xhtml" media-type="application/xhtml+xml"/><item id="c2" href="text/two%20b.xhtml" media-type="application/xhtml+xml"/>
<item id="cover" href="images/cover.jpg" media-type="image/jpeg"/><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/></manifest>
<spine><itemref idref="c2"/><itemref idref="cover"/><itemref idref="c1" linear="no"/></spine></package>"#;

/// An EPUB file with the package above; `encryption` adds `META-INF/encryption.xml`.
pub(super) fn epub(encryption: Option<&str>) -> Vec<u8> {
    let xhtml = |body: &str| format!("<?xml version=\"1.0\"?><html xmlns=\"http://www.w3.org/1999/xhtml\"><head><title>t</title><style>p{{}}</style></head><body>{body}</body></html>");
    let mut entries = vec![
        stored("mimetype", "application/epub+zip"),
        stored("META-INF/container.xml", CONTAINER),
        deflated("OEBPS/content.opf", OPF),
        deflated("OEBPS/text/one.xhtml", &xhtml("<h1>One</h1><p>First &amp; <em>last</em>.</p>")),
        deflated("OEBPS/text/two b.xhtml", &xhtml("<h2>Two</h2><ul><li>a</li><li>b</li></ul><img alt=\"map\" src=\"m.png\"/>")),
        stored("OEBPS/nav.xhtml", &xhtml("<nav>contents</nav>")),
    ];
    if let Some(e) = encryption {
        entries.push(stored("META-INF/encryption.xml", e));
    }
    zip_file(&entries)
}

#[test]
fn epub_reads_the_spine_in_order_with_chapter_markers() {
    // The mimetype entry decides the format, whatever the name.
    let (format, doc) = read(&epub(None), "book.zip", &Caps::default()).unwrap();
    assert_eq!(format, Format::Epub);
    assert_eq!(
        doc.markdown,
        "# A Short & Small Book\n\nby Ann Author, Bo Writer\n\n<!-- chapter 1 of 3 -->\n## Chapter 1\n\n## Two\n\n- a\n- b\n\n![map](m.png)\n\
         \n<!-- chapter 2 of 3 -->\n## Chapter 2\n\n(no text in this chapter)\n\
         \n<!-- chapter 3 of 3 -->\n## Chapter 3\n\n# One\n\nFirst & *last*.\n"
    );
    assert_eq!((doc.pages, doc.warnings), (3, vec![]));
    let caps = Caps { max_pages: 1, ..Caps::default() };
    assert_eq!(read(&epub(None), "b.epub", &caps).unwrap().1.warnings, ["chapters after 1 of 3 (chapters cap)"]);
    let caps = Caps { max_chars: 40, ..Caps::default() };
    let (_, doc) = read(&epub(None), "b.epub", &caps).unwrap();
    assert_eq!((doc.chars, doc.warnings), (40, vec!["text after 40 chars (char cap)".to_owned()]));
}

#[test]
fn epub_encryption_and_damage_give_one_line() {
    let fonts = r#"<encryption><EncryptedData><EncryptionMethod Algorithm="http://www.idpf.org/2008/embedding"/></EncryptedData></encryption>"#;
    assert!(read(&epub(Some(fonts)), "b.epub", &Caps::default()).is_ok());
    let drm = r#"<encryption><EncryptedData><EncryptionMethod Algorithm="http://www.w3.org/2001/04/xmlenc#aes128-cbc"/></EncryptedData></encryption>"#;
    assert_eq!(read(&epub(Some(drm)), "b.epub", &Caps::default()).unwrap_err().to_string(), "this EPUB file is encrypted; Vade cannot read encrypted files");
    let err = |bytes: &[u8]| read(bytes, "b.epub", &Caps::default()).unwrap_err().to_string();
    assert_eq!(err(&zip_file(&[stored("mimetype", "application/epub+zip")])), "damaged EPUB file: the archive has no META-INF/container.xml; no text was read");
    assert_eq!(err(b"plain"), "damaged EPUB file: the file is not a ZIP archive; no text was read");
    let file = epub(None);
    for n in (0..file.len()).step_by(9) {
        let _ = read(&file[..n], "b.epub", &Caps::default());
    }
}

/// An OpenDocument file of `kind` (`text`, `spreadsheet`, `presentation`) with `body` in its office body.
pub(super) fn odf(kind: &str, body: &str) -> Vec<u8> {
    let content = format!(
        r#"<?xml version="1.0"?><office:document-content xmlns:office="o" xmlns:text="t" xmlns:table="ta" xmlns:draw="d" xmlns:xlink="x" xmlns:presentation="p"><office:automatic-styles><style:style style:name="P1"><text:p>style text</text:p></style:style></office:automatic-styles><office:body><office:{kind}>{body}</office:{kind}></office:body></office:document-content>"#
    );
    zip_file(&[
        stored("mimetype", &format!("application/vnd.oasis.opendocument.{kind}")),
        stored("META-INF/manifest.xml", "<manifest:manifest xmlns:manifest=\"m\"><manifest:file-entry manifest:full-path=\"/\"/></manifest:manifest>"),
        deflated("content.xml", &content),
    ])
}

#[test]
fn odt_reads_headings_lists_tables_links_and_soft_pages() {
    let body = "<text:h text:outline-level=\"2\">Intro</text:h><text:p>One<text:s text:c=\"2\"/>two<text:tab/>three<text:line-break/>four \
        <text:a xlink:href=\"https://example.org\">site</text:a><office:annotation><text:p>comment</text:p></office:annotation><text:note><text:note-body><text:p>foot</text:p></text:note-body></text:note></text:p>\
        <text:list><text:list-item><text:p>a</text:p><text:list><text:list-item><text:p>a1</text:p></text:list-item></text:list></text:list-item><text:list-item><text:p>b</text:p></text:list-item></text:list>\
        <table:table><table:table-row><table:table-cell table:number-columns-repeated=\"2\"><text:p>h</text:p></table:table-cell></table:table-row><table:table-row><table:table-cell><text:p>x|y</text:p></table:table-cell><table:covered-table-cell/></table:table-row></table:table>\
        <text:p>end <text:soft-page-break/>next page</text:p><text:p><draw:frame draw:name=\"Logo\"><draw:image xlink:href=\"Pictures/logo.png\"/></draw:frame></text:p>";
    let (format, doc) = read(&odf("text", body), "a.odt", &Caps::default()).unwrap();
    assert_eq!(format, Format::Odt);
    assert_eq!(
        doc.markdown,
        "<!-- page 1 of 2 -->\n## Page 1\n\n## Intro\n\nOne  two\tthree\nfour [site](https://example.org)\n\n- a\n    - a1\n- b\n\n| h | h |\n| --- | --- |\n| x\\|y |  |\n\nend\n\
         \n<!-- page 2 of 2 -->\n## Page 2\n\nnext page\n\n[image: Logo (Pictures/logo.png)]\n"
    );
    assert_eq!((doc.pages, doc.warnings), (2, vec!["1 images (listed, not read)".to_owned()]));
}

#[test]
fn ods_reads_sheets_and_expands_repeats_only_between_values() {
    let body = "<table:table table:name=\"Q1\"><table:table-row><table:table-cell><text:p>Item</text:p></table:table-cell><table:table-cell table:number-columns-repeated=\"2\"/><table:table-cell office:value-type=\"float\" office:value=\"3\"><text:p>3.00</text:p></table:table-cell><table:table-cell table:number-columns-repeated=\"16380\"/></table:table-row>\
        <table:table-row table:number-rows-repeated=\"2\"><table:table-cell><text:p>same</text:p></table:table-cell><table:table-cell office:value-type=\"boolean\" office:boolean-value=\"true\"/></table:table-row>\
        <table:table-row table:number-rows-repeated=\"1048570\"><table:table-cell table:number-columns-repeated=\"16384\"/></table:table-row></table:table>\
        <table:table table:name=\"Empty\"><table:table-row><table:table-cell/></table:table-row></table:table>";
    let (format, doc) = read(&odf("spreadsheet", body), "a.ods", &Caps::default()).unwrap();
    assert_eq!(format, Format::Ods);
    assert_eq!(
        doc.markdown,
        "<!-- sheet 1 of 2 -->\n## Sheet Q1\n\n| Item |  |  | 3.00 |\n| --- | --- | --- | --- |\n| same | true |  |  |\n| same | true |  |  |\n\n<!-- sheet 2 of 2 -->\n## Sheet Empty\n\n(empty sheet)\n"
    );
    let caps = Caps { max_rows: 1, ..Caps::default() };
    assert_eq!(read(&odf("spreadsheet", body), "a.ods", &caps).unwrap().1.warnings, ["rows after 1 of 2 in sheet Q1 (row cap)"]);
    let many = "<table:table table:name=\"W\"><table:table-row table:number-rows-repeated=\"1000000\"><table:table-cell table:number-columns-repeated=\"16384\"><text:p>v</text:p></table:table-cell></table:table-row></table:table>";
    let caps = Caps { max_chars: 1000, ..Caps::default() };
    let (_, doc) = read(&odf("spreadsheet", many), "a.ods", &caps).unwrap();
    assert!(doc.markdown.len() <= MIN_WORK_BYTES + 200, "{}", doc.markdown.len());
    assert!(doc.warnings.contains(&format!("text after {MIN_WORK_BYTES} bytes of conversion work (memory cap)")), "{:?}", doc.warnings);
}

#[test]
fn odp_reads_pages_titles_frames_and_notes() {
    let frame = |class: &str, text: &str| format!("<draw:frame presentation:class=\"{class}\"><draw:text-box>{text}</draw:text-box></draw:frame>");
    let body = [
        "<draw:page draw:name=\"p1\">".to_owned(),
        frame("title", "<text:p>Welcome</text:p><text:p>all</text:p>"),
        frame("outline", "<text:list><text:list-item><text:p>point</text:p></text:list-item><text:list-item><text:p>more</text:p></text:list-item></text:list>"),
        frame("page-number", "<text:p>1</text:p>"),
        "<presentation:notes><draw:page-thumbnail/>".into(),
        frame("notes", "<text:p>Say hi.</text:p>"),
        "</presentation:notes></draw:page><draw:page draw:name=\"p2\"></draw:page>".into(),
    ]
    .concat();
    let (format, doc) = read(&odf("presentation", &body), "a.odp", &Caps::default()).unwrap();
    assert_eq!(format, Format::Odp);
    assert_eq!(
        doc.markdown,
        "<!-- slide 1 of 2 -->\n## Slide 1\n\n### Welcome all\n\n- point\n- more\n\n#### Notes\n\nSay hi.\n\n<!-- slide 2 of 2 -->\n## Slide 2\n\n(no text on this slide)\n"
    );
    assert_eq!(doc.pages, 2);
}

#[test]
fn odf_encrypted_and_damaged_files_give_one_line() {
    let file = odf("text", "<text:p>x</text:p>");
    let manifest = "<manifest:manifest><manifest:file-entry manifest:full-path=\"content.xml\"><manifest:encryption-data/></manifest:file-entry></manifest:manifest>";
    let locked = zip_file(&[stored("mimetype", "application/vnd.oasis.opendocument.text"), stored("META-INF/manifest.xml", manifest), stored("content.xml", "garbage")]);
    assert_eq!(read(&locked, "a.odt", &Caps::default()).unwrap_err().to_string(), "this ODT file is encrypted; Vade cannot read encrypted files");
    let broken = odf("text", "<text:p>x</text:span>");
    assert!(read(&broken, "a.odt", &Caps::default()).unwrap_err().to_string().starts_with("damaged ODT file: content.xml the close tag </text:span> does not match <text:p>"));
    // The mimetype entry decides the format under any name; a name alone does too.
    assert_eq!(sniff(&file, "notes.zip").map(|(f, _)| f), Some(Format::Odt));
    assert_eq!(read(b"plain", "a.ods", &Caps::default()).unwrap_err().to_string(), "damaged ODS file: the file is not a ZIP archive; no text was read");
    for n in (0..file.len()).step_by(5) {
        let _ = read(&file[..n], "a.odt", &Caps::default());
    }
}

#[test]
fn parts_that_stop_at_the_work_cap_are_not_damaged() {
    // A shared string table larger than the work cap stops with the warning.
    let strings: String = (0..5000).map(|i| format!("<si><t>string number {i}</t></si>")).collect();
    let shared = format!("<sst>{strings}</sst>");
    let mut file = xlsx("<row><c r=\"A1\" t=\"s\"><v>4000</v></c></row>", "");
    file = {
        let mut entries = Vec::new();
        let mut zip = Zip::open(&file, Format::Xlsx).unwrap();
        for name in ["[Content_Types].xml", "_rels/.rels", "xl/workbook.xml", "xl/_rels/workbook.xml.rels", "xl/styles.xml", "xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"] {
            entries.push((name, String::from_utf8(zip.part(name).unwrap().unwrap().data).unwrap()));
        }
        entries.push(("xl/sharedStrings.xml", shared));
        zip_file(&entries.iter().map(|(n, d)| deflated(n, d)).collect::<Vec<_>>())
    };
    let caps = Caps { max_chars: 1000, ..Caps::default() };
    let (_, doc) = read(&file, "a.xlsx", &caps).unwrap();
    assert!(doc.warnings.contains(&format!("text after {MIN_WORK_BYTES} bytes of conversion work (memory cap)")), "{:?}", doc.warnings);
}

#[test]
fn pdf_reads_with_a_marker_before_each_page() {
    let file = crate::pdf::tests::sample(&["first page", "page 2 of 9 footer", "third"]);
    let (format, doc) = read(&file, "r.pdf", &Caps::default()).unwrap();
    assert_eq!(format, Format::Pdf);
    assert_eq!(
        doc.markdown,
        "<!-- page 1 of 3 -->\n## Page 1\n\nfirst page\n\n<!-- page 2 of 3 -->\n## Page 2\n\npage 2 of 9 footer\n\n<!-- page 3 of 3 -->\n## Page 3\n\nthird\n"
    );
    assert_eq!((doc.pages, doc.warnings), (3, vec![]));
    // The agent tool's PDF text is unchanged.
    let text = crate::pdf::text(&file, &std::sync::atomic::AtomicBool::new(false)).unwrap();
    assert!(text.starts_with("PDF text layer extracted by Vade: 3 pages.") && text.contains("page 2 of 3\npage 2 of 9 footer\n\npage 3 of 3\nthird"), "{text}");
    let caps = Caps { max_pages: 1, ..Caps::default() };
    assert_eq!(read(&file, "r.pdf", &caps).unwrap().1.warnings, ["pages after 1 of 3 (pages cap)"]);
    let blank = crate::pdf::tests::sample(&[""]);
    assert_eq!(read(&blank, "r.pdf", &Caps::default()).unwrap_err(), ReadError::NoText(Format::Pdf, 1));
    assert_eq!(read(b"%PDF-1.4\n", "a.pdf", &Caps::default()).unwrap_err().to_string().split(':').next(), Some("damaged PDF file"));
    assert_eq!(read(b"just text", "a.pdf", &Caps::default()).unwrap_err().to_string(), "damaged PDF file: the file has no PDF header (%PDF-) in its first 1024 bytes; no text was read");
}

#[test]
fn pptx_and_odt_tables_at_the_work_cap_keep_text_and_markers() {
    let caps = Caps { max_chars: 4096, ..Caps::default() };
    let cap = format!("text after {MIN_WORK_BYTES} bytes of conversion work (memory cap)");
    // The readers' own output: `read` then cuts it at the char cap anywhere.
    let cell = "<a:tc><a:txBody><a:p><a:r><a:t>v</a:t></a:r></a:p></a:txBody></a:tc>";
    let rows = format!("<a:tr>{}</a:tr>", cell.repeat(20)).repeat(3000);
    let table = format!("<p:graphicFrame><a:graphic><a:graphicData><a:tbl><a:tr>{}</a:tr>{rows}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>", cell.replace(">v<", ">kept<"));
    let doc = super::pptx::PptxReader.to_markdown(&pptx(&format!("{}{table}", shape("", &ap(0, "Hello")))), Format::Pptx, &caps).unwrap();
    // The builder's first slide is hidden; the table is on slide 2.
    assert!(doc.markdown.contains("\n<!-- slide 2 of 2 -->\n## Slide 2\n\nHello\n\n| kept |  |"), "{:?}", &doc.markdown[..120]);
    assert!(doc.markdown.contains("\n| v | v |"));
    assert_eq!(doc.warnings, [cap.clone()]);
    let cell = "<table:table-cell><text:p>v</text:p></table:table-cell>";
    let rows = format!("<table:table-row>{}</table:table-row>", cell.repeat(20)).repeat(3000);
    let body = format!("<text:p>Hello</text:p><table:table><table:table-row>{}</table:table-row>{rows}</table:table><text:p>after<text:soft-page-break/>next</text:p>", cell.replace(">v<", ">kept<"));
    let doc = super::odf::OdfReader.to_markdown(&odf("text", &body), Format::Odt, &caps).unwrap();
    assert!(doc.markdown.starts_with("<!-- page 1 of 2 -->\n## Page 1\n\nHello\n\n| kept |  |"), "{:?}", &doc.markdown[..80]);
    assert!(doc.markdown.contains("\n| v | v |"));
    assert!(doc.markdown.lines().filter(|l| l.starts_with("<!--")).all(|l| l.ends_with(" -->")));
    assert_eq!(doc.warnings, [cap]);
}

/// `file` rebuilt from its parts `names`, stored, with `parts` replaced.
fn rebuilt(file: &[u8], names: &[&str], parts: &[(&str, &str)]) -> Vec<u8> {
    let mut zip = Zip::open(file, Format::Docx).unwrap();
    let data: Vec<(&str, String)> = names
        .iter()
        .map(|&n| (n, parts.iter().find(|p| p.0 == n).map_or_else(|| String::from_utf8(zip.part(n).unwrap().unwrap().data).unwrap(), |p| p.1.to_owned())))
        .collect();
    zip_file(&data.iter().map(|(n, d)| stored(n, d)).collect::<Vec<_>>())
}

#[test]
fn xlsx_and_ods_sheets_at_the_work_cap_keep_their_tables() {
    let caps = Caps { max_chars: 4096, ..Caps::default() };
    let cap = format!("text after {MIN_WORK_BYTES} bytes of conversion work (memory cap)");
    // The readers' own output: `read` then cuts it at the char cap anywhere.
    let rows: String = (1..=3000).map(|r| format!("<row r=\"{r}\">{}</row>", (b'A'..=b'T').map(|c| format!("<c r=\"{}{r}\"><v>1</v></c>", c as char)).collect::<String>())).collect();
    let doc = super::xlsx::XlsxReader.to_markdown(&xlsx(&rows, ""), Format::Xlsx, &caps).unwrap();
    assert!(doc.markdown.starts_with("<!-- sheet 1 of 2 -->\n## Sheet Sales & costs\n\n| 1 | 1 |"), "{:?}", &doc.markdown[..80.min(doc.markdown.len())]);
    assert!(doc.markdown.contains("\n| 1 | 1 |") && doc.markdown.lines().filter(|l| l.starts_with("<!--")).all(|l| l.ends_with(" -->")));
    assert_eq!(doc.warnings, [cap.clone()]);
    let cell = "<table:table-cell office:value-type=\"float\" office:value=\"1\"><text:p>1</text:p></table:table-cell>";
    let body = format!("<table:table table:name=\"S\">{}</table:table>", format!("<table:table-row>{}</table:table-row>", cell.repeat(20)).repeat(3000));
    let doc = super::odf::OdfReader.to_markdown(&odf("spreadsheet", &body), Format::Ods, &caps).unwrap();
    assert!(doc.markdown.starts_with("<!-- sheet 1 of 1 -->\n## Sheet S\n\n| 1 | 1 |"), "{:?}", &doc.markdown[..80.min(doc.markdown.len())]);
    assert!(doc.markdown.contains("\n| 1 | 1 |"));
    assert_eq!(doc.warnings, [cap]);
}

#[test]
fn rows_past_the_row_cap_are_counted_not_read() {
    // A long value in a row past the cap is counted and gives no cap warning.
    let caps = Caps { max_chars: 1000, max_rows: 1, ..Caps::default() };
    let long = "x".repeat(70_000);
    let rows = format!("<row r=\"1\"><c r=\"A1\"><v>1</v></c></row><row r=\"2\"><c r=\"A2\"><v>2</v></c></row><row r=\"3\"><c r=\"A3\" t=\"inlineStr\"><is><t>{long}</t></is></c></row><row r=\"4\"><c r=\"A4\"><v>4</v></c></row>");
    let (_, doc) = read(&xlsx(&rows, ""), "a.xlsx", &caps).unwrap();
    assert_eq!(doc.warnings, ["rows after 1 of 3 in sheet Sales & costs (row cap)"]);
    let cell = |t: &str| format!("<table:table-row><table:table-cell><text:p>{t}</text:p></table:table-cell></table:table-row>");
    let body = format!("<table:table table:name=\"S\">{}{}{}{}</table:table>", cell("1"), cell("2"), cell(&long), cell("4"));
    let (_, doc) = read(&odf("spreadsheet", &body), "a.ods", &caps).unwrap();
    assert_eq!(doc.warnings, ["rows after 1 of 3 in sheet S (row cap)"]);
}

#[test]
fn odf_frames_and_images_hold_their_place() {
    // An image stays where it is in its paragraph, in a table cell too, and
    // the innermost frame names it.
    let image = |name: &str| format!("<draw:frame draw:name=\"{name}\"><draw:image xlink:href=\"Pictures/a.png\"/></draw:frame>");
    let body = format!(
        "<text:p>Hello {} world</text:p><table:table><table:table-row><table:table-cell><text:p>in {}</text:p></table:table-cell><table:table-cell><text:p>b</text:p></table:table-cell></table:table-row></table:table>\
         <text:p><draw:frame draw:name=\"Outer\"><draw:text-box><text:p>{}</text:p></draw:text-box></draw:frame></text:p>",
        image("Pic"),
        image("Cell"),
        image("Inner")
    );
    let (_, doc) = read(&odf("text", &body), "a.odt", &Caps::default()).unwrap();
    assert_eq!(
        doc.markdown,
        "<!-- page 1 of 1 -->\n## Page 1\n\nHello [image: Pic (Pictures/a.png)] world\n\n| in [image: Cell (Pictures/a.png)] | b |\n| --- | --- |\n\n[image: Inner (Pictures/a.png)]\n"
    );
    assert_eq!(doc.warnings, ["3 images (listed, not read)"]);
    // Frames that open after the work cap hold nothing and give nothing back.
    let caps = Caps { max_chars: 4096, ..Caps::default() };
    let cell = "<table:table-cell><text:p>v</text:p></table:table-cell>";
    let table = format!("<table:table>{}</table:table>", format!("<table:table-row>{}</table:table-row>", cell.repeat(20)).repeat(3000));
    let body = format!("{table}{}<text:p>a{}</text:p>", "<draw:frame presentation:class=\"graphic\"/>".repeat(3000), "<text:soft-page-break/>".repeat(2000));
    let doc = super::odf::OdfReader.to_markdown(&odf("text", &body), Format::Odt, &caps).unwrap();
    assert!(doc.markdown.len() <= MIN_WORK_BYTES, "{}", doc.markdown.len());
}

#[test]
fn slide_lists_and_spines_at_the_work_cap_give_the_cap_warning() {
    let caps = Caps { max_chars: 1000, ..Caps::default() };
    let cap = format!("text after {MIN_WORK_BYTES} bytes of conversion work (memory cap)");
    let ids: String = (0..5000).map(|i| format!("<p:sldId id=\"{}\" r:id=\"rIdX{i}\"/>", 300 + i)).collect();
    let presentation = format!("<p:presentation xmlns:p=\"p\" xmlns:r=\"r\"><p:sldIdLst>{ids}<p:sldId id=\"257\" r:id=\"rId3\"/></p:sldIdLst></p:presentation>");
    let names = ["[Content_Types].xml", "_rels/.rels", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels", "ppt/slides/slide1.xml", "ppt/slides/_rels/slide1.xml.rels", "ppt/slides/slide2.xml", "ppt/notesSlides/notesSlide1.xml"];
    let file = rebuilt(&pptx(&shape("", &ap(0, "x"))), &names, &[("ppt/presentation.xml", &presentation)]);
    let (_, doc) = read(&file, "a.pptx", &caps).unwrap();
    assert_eq!(doc.warnings, [cap.clone()]);
    let items: String = (0..5000).map(|i| format!("<item id=\"x{i}\" href=\"x{i}.xhtml\" media-type=\"application/xhtml+xml\"/>")).collect();
    let names = ["mimetype", "META-INF/container.xml", "OEBPS/content.opf", "OEBPS/text/one.xhtml", "OEBPS/text/two b.xhtml", "OEBPS/nav.xhtml"];
    let opf = OPF.replace("<manifest>", &format!("<manifest>{items}"));
    let (_, doc) = read(&rebuilt(&epub(None), &names, &[("OEBPS/content.opf", &opf)]), "a.epub", &caps).unwrap();
    assert_eq!(doc.warnings, [cap]);
}

#[test]
fn cut_parts_keep_their_text() {
    // A stored part after a decoder stop is cut, not damaged.
    let bomb = Entry { name: "b.xml", data: vec![b'a'; 1 + 258 * 100], deflate: true, raw: Some(deflate_bomb(100)) };
    let file = zip_file(&[bomb, stored("s.txt", "stored")]);
    let mut zip = Zip::open_with(&file, Format::Docx, usize::MAX, 5000).unwrap();
    assert!(zip.part("b.xml").unwrap().unwrap().cut);
    let part = zip.part("s.txt").unwrap().unwrap();
    assert!(part.cut && part.data.len() < 6, "{}", part.data.len());
    // A UTF-8 text cut inside a character loses only that character.
    let mut data = "caf\u{e9} na\u{ef}ve".as_bytes()[..9].to_vec();
    super::zip::drop_cut_char(&mut data);
    assert_eq!(data, "caf\u{e9} na".as_bytes());
    assert_eq!(text::decode(&data).unwrap().0, "caf\u{e9} na");
}

#[test]
fn sheets_keep_accepted_text_and_stop_at_their_caps() {
    // A value larger than the room keeps its start.
    let long = "y".repeat(70_000);
    let rows = format!("<row r=\"1\"><c r=\"A1\" t=\"inlineStr\"><is><t>{long}</t></is></c></row>");
    let caps = Caps { max_chars: 1000, ..Caps::default() };
    let doc = super::xlsx::XlsxReader.to_markdown(&xlsx(&rows, ""), Format::Xlsx, &caps).unwrap();
    assert!(doc.markdown.contains(&format!("\n\n| {} |", "y".repeat(1000))), "{:?}", &doc.markdown[..80]);
    assert_eq!(doc.warnings, [super::char_cap_warning(1000)]);
    // A sheet past the sheet cap is not read, so it takes no room.
    let cell = "<table:table-cell><text:p>v</text:p></table:table-cell>";
    let big = format!("<table:table table:name=\"Big\">{}</table:table>", format!("<table:table-row>{}</table:table-row>", cell.repeat(20)).repeat(3000));
    let body = format!("<table:table table:name=\"S\"><table:table-row><table:table-cell><text:p><![CDATA[kept]]></text:p></table:table-cell></table:table-row></table:table>{big}");
    let caps = Caps { max_chars: 1000, max_pages: 1, ..Caps::default() };
    let (_, doc) = read(&odf("spreadsheet", &body), "a.ods", &caps).unwrap();
    assert_eq!(doc.markdown, "<!-- sheet 1 of 2 -->\n## Sheet S\n\n| kept |\n| --- |\n");
    assert_eq!(doc.warnings, ["sheets after 1 of 2 (sheets cap)"]);
    // Many sheets: every marker the reader writes is whole.
    let many: String = (0..400).map(|i| format!("<table:table table:name=\"S{i}\"><table:table-row>{}</table:table-row></table:table>", cell.repeat(30))).collect();
    let doc = super::odf::OdfReader.to_markdown(&odf("spreadsheet", &many), Format::Ods, &Caps { max_chars: 1000, ..Caps::default() }).unwrap();
    assert!(doc.markdown.starts_with("<!-- sheet 1 of 400 -->\n## Sheet S0\n\n| v |"));
    assert!(doc.markdown.lines().filter(|l| l.starts_with("<!--")).all(|l| l.ends_with(" -->")));
    // Spaces and repeated rows grow within the cap.
    let spaces = format!("<table:table table:name=\"S\"><table:table-row><table:table-cell><text:p>a{}</text:p></table:table-cell></table:table-row></table:table>", "<text:s text:c=\"64\"/>".repeat(10_000));
    let doc = super::odf::OdfReader.to_markdown(&odf("spreadsheet", &spaces), Format::Ods, &Caps { max_chars: 1000, ..Caps::default() }).unwrap();
    assert!(doc.markdown.contains("| a |") && doc.warnings == [super::char_cap_warning(1000)], "{} {:?}", doc.markdown.len(), doc.warnings);
    let value = "z".repeat(20_000);
    let clones = format!("<table:table table:name=\"S\"><table:table-row table:number-rows-repeated=\"1000\"><table:table-cell><text:p>{value}</text:p></table:table-cell></table:table-row></table:table>");
    let doc = super::odf::OdfReader.to_markdown(&odf("spreadsheet", &clones), Format::Ods, &Caps { max_chars: 20_000, ..Caps::default() }).unwrap();
    let cap = "text after 320000 bytes of conversion work (memory cap)";
    assert!(doc.markdown.len() <= 20_000 * 16 && doc.warnings == [cap], "{} {:?}", doc.markdown.len(), doc.warnings);
}

#[test]
fn metadata_text_and_references_stay_in_bounds() {
    let caps = Caps { max_chars: 1000, ..Caps::default() };
    // An attribute that no reader asks for is checked, not decoded.
    let ignored = format!("<w:p w:x=\"{}\"><w:r><w:t>kept</w:t></w:r></w:p>", "&amp;".repeat(1_000_000));
    assert_eq!(docx_md(&ignored), "<!-- page 1 of 1 -->\n## Page 1\n\nkept\n");
    assert!(events("<a x=\"&bad;\"/>").unwrap_err().contains("&bad;"));
    // Long attribute values are left out, not decoded or copied: the text
    // stays and the attribute cap says so.
    let long = "b".repeat(1_000_000);
    let page = "<!-- page 1 of 1 -->\n## Page 1\n\nkept\n";
    let based = format!("<w:styles xmlns:w=\"w\"><w:style w:styleId=\"S\"><w:basedOn w:val=\"{long}\"/></w:style></w:styles>");
    let style = format!("<w:p><w:pPr><w:pStyle w:val=\"{long}\"/></w:pPr><w:r><w:t>kept</w:t></w:r></w:p>");
    for file in [docx_with(&p("kept"), &[("word/styles.xml", &based)]), docx(&style)] {
        let (_, doc) = read(&file, "a.docx", &caps).unwrap();
        assert_eq!((doc.markdown.as_str(), doc.warnings), (page, vec![ATTR_CAP.to_owned()]));
    }
    let link = format!("<text:p><text:a xlink:href=\"https://example.org/{long}\">kept</text:a></text:p>");
    let image = format!("<text:p>kept<draw:frame draw:name=\"{long}\"><draw:image xlink:href=\"{long}\"/></draw:frame></text:p>");
    for body in [link, image] {
        let (_, doc) = read(&odf("text", &body), "a.odt", &caps).unwrap();
        assert_eq!((doc.markdown.as_str(), doc.warnings), (page, vec![ATTR_CAP.to_owned()]));
    }
    let fallback = format!("<table:table table:name=\"S\"><table:table-row><table:table-cell office:string-value=\"{long}\"/><table:table-cell><text:p>kept</text:p></table:table-cell></table:table-row></table:table>");
    let (_, doc) = read(&odf("spreadsheet", &fallback), "a.ods", &caps).unwrap();
    assert_eq!((doc.markdown.as_str(), doc.warnings), ("<!-- sheet 1 of 1 -->\n## Sheet S\n\n| kept |\n| --- |\n", vec![ATTR_CAP.to_owned()]));
    // A requested alt text decodes only its start: 200 chars, then `...`.
    let descr = format!("<p:sld><p:cSld><p:spTree><p:pic><p:nvPicPr><p:cNvPr id=\"1\" descr=\"{}\"/></p:nvPicPr><p:blipFill><a:blip r:embed=\"rId2\"/></p:blipFill></p:pic></p:spTree></p:cSld></p:sld>", "&amp;".repeat(1_000_000));
    let tag = events_tag(&descr, "cNvPr");
    let (start, whole) = tag.0.unwrap();
    assert_eq!((start.len(), whole), (804, false));
    assert_eq!(tag.1, format!("{}...", "&".repeat(200)));
    // 8,000 custom formats and 8,000 cell formats: each lookup is by id.
    let formats: String = (0..8000).map(|i| format!("<numFmt numFmtId=\"{}\" formatCode=\"0.0\"/>", 200 + i)).collect();
    let xfs: String = (0..8000).map(|_| "<xf numFmtId=\"9\"/>").collect();
    let styles = format!("<styleSheet><numFmts>{formats}</numFmts><cellXfs>{xfs}</cellXfs></styleSheet>");
    let names = ["[Content_Types].xml", "_rels/.rels", "xl/workbook.xml", "xl/_rels/workbook.xml.rels", "xl/sharedStrings.xml", "xl/styles.xml", "xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"];
    let file = rebuilt(&xlsx("<row r=\"1\"><c r=\"A1\" s=\"7999\"><v>5</v></c></row>", ""), &names, &[("xl/styles.xml", &styles)]);
    assert!(read(&file, "a.xlsx", &Caps::default()).unwrap().1.markdown.contains("| 5 |"));
    // CDATA text counts in EPUB metadata.
    let opf = OPF.replace("A  Short &amp; Small Book", "<![CDATA[Cdata & Book]]>");
    let names = ["mimetype", "META-INF/container.xml", "OEBPS/content.opf", "OEBPS/text/one.xhtml", "OEBPS/text/two b.xhtml", "OEBPS/nav.xhtml"];
    let (_, doc) = read(&rebuilt(&epub(None), &names, &[("OEBPS/content.opf", &opf)]), "a.epub", &Caps::default()).unwrap();
    assert!(doc.markdown.starts_with("# Cdata & Book\n\nby Ann Author, Bo Writer\n"), "{:?}", &doc.markdown[..40]);
}

#[test]
fn formats_lists_and_headers_read_their_exact_values() {
    // A list that starts at 0 counts 0, 1, 2.
    let numbering = r#"<w:numbering xmlns:w="w"><w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="0"/><w:numFmt w:val="decimal"/></w:lvl></w:abstractNum><w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>"#;
    let body = [item(1, 0, "a"), item(1, 0, "b"), item(1, 0, "c")].concat();
    let (_, doc) = read(&docx_with(&body, &[("word/numbering.xml", numbering)]), "a.docx", &Caps::default()).unwrap();
    assert_eq!(doc.markdown, "<!-- page 1 of 1 -->\n## Page 1\n\n0. a\n1. b\n2. c\n");
    // Elapsed time keeps whole days.
    assert_eq!(super::xlsx::test_show("1.5", "[h]:mm:ss"), "36:00:00");
    assert_eq!(super::xlsx::test_show("0.5", "hh:mm"), "12:00:00");
    // A PNG header chunk must have 13 bytes; the size is not read past it.
    let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\0IHDR".to_vec();
    png.extend_from_slice(&[0, 0, 2, 128, 0, 0, 1, 224, 8, 6, 0, 0, 0]);
    assert_eq!(read(&png, "a.png", &Caps::default()).unwrap_err().to_string(), "damaged PNG image file: the PNG header chunk does not have 13 bytes; no text was read");
    // EPUB part names keep their case; OPC part names do not.
    let file = zip_file(&[stored("A.xhtml", "upper"), stored("a.xhtml", "lower")]);
    let mut zip = Zip::open(&file, Format::Epub).unwrap();
    assert_eq!(zip.part("a.xhtml").unwrap().unwrap().data, b"lower");
    assert!(Zip::open(&file, Format::Docx).err().unwrap().to_string().contains("two entries are named a.xhtml"));
    // A false empty end record in the comment that names its own place.
    let mut file = docx(&p("kept"));
    let n = file.len();
    let mut fake = b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0".to_vec();
    fake.extend_from_slice(&(n as u32).to_le_bytes());
    fake.extend_from_slice(&[0, 0, b'x']);
    file[n - 2..].copy_from_slice(&(fake.len() as u16).to_le_bytes());
    file.extend_from_slice(&fake);
    assert_eq!(read(&file, "a.docx", &Caps::default()).unwrap().1.markdown, "<!-- page 1 of 1 -->\n## Page 1\n\nkept\n");
}

#[test]
fn caps_keep_accepted_text_and_say_so() {
    let caps = Caps { max_chars: 1000, ..Caps::default() };
    let cap = format!("text after {MIN_WORK_BYTES} bytes of conversion work (memory cap)");
    // EPUB: a manifest that the cap stops keeps the title and authors.
    let items: String = (0..5000).map(|i| format!("<item id=\"x{i}\" href=\"x{i}.xhtml\" media-type=\"application/xhtml+xml\"/>")).collect();
    let names = ["mimetype", "META-INF/container.xml", "OEBPS/content.opf", "OEBPS/text/one.xhtml", "OEBPS/text/two b.xhtml", "OEBPS/nav.xhtml"];
    let opf = OPF.replace("<manifest>", &format!("<manifest>{items}"));
    let doc = super::epub::EpubReader.to_markdown(&rebuilt(&epub(None), &names, &[("OEBPS/content.opf", &opf)]), Format::Epub, &caps).unwrap();
    assert_eq!((doc.markdown.as_str(), doc.warnings), ("# A Short & Small Book\n\nby Ann Author, Bo Writer\n", vec![cap.clone()]));
    // EPUB: more than 20 authors say how many more there are.
    let many: String = (0..23).map(|i| format!("<dc:creator>A{i}</dc:creator>")).collect();
    let opf = OPF.replace("<dc:creator>Ann Author</dc:creator><dc:creator>Bo Writer</dc:creator>", &many);
    let (_, doc) = read(&rebuilt(&epub(None), &names, &[("OEBPS/content.opf", &opf)]), "a.epub", &Caps::default()).unwrap();
    assert!(doc.markdown.contains(", A19 and 3 more\n"), "{}", doc.markdown);
    // ODT: a cell repeated over 64 columns keeps the copies that fit.
    let value = format!("kept{}", "v".repeat(996));
    let table = format!("<table:table><table:table-row><table:table-cell table:number-columns-repeated=\"64\"><text:p>{value}</text:p></table:table-cell></table:table-row></table:table>");
    let doc = super::odf::OdfReader.to_markdown(&odf("text", &table), Format::Odt, &caps).unwrap();
    assert!(doc.markdown.contains(&format!("| {value} |")) && doc.warnings == [cap.clone()], "{:?}", doc.warnings);
    // PPTX: a slide part that a cap cut keeps its open paragraph.
    let slide = format!("<p:sld><p:cSld><p:spTree>{}", shape("", &ap(0, "kept")).replace("</a:t></a:r></a:p></p:txBody></p:sp>", ""));
    assert_eq!(super::pptx::test_slide(&slide, true).unwrap(), "kept");
    assert!(super::pptx::test_slide(&slide, false).is_err());
    // PPTX: relationships the slides do not use leave the output room.
    let rels: String = (0..560).map(|i| format!("<Relationship Id=\"rL{i}\" Type=\"t/hyperlink\" Target=\"https://example.org/{i:04}\" TargetMode=\"External\"/>")).collect();
    let names = ["[Content_Types].xml", "_rels/.rels", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels", "ppt/slides/slide1.xml", "ppt/slides/_rels/slide1.xml.rels", "ppt/slides/slide2.xml", "ppt/notesSlides/notesSlide1.xml"];
    let file = rebuilt(&pptx(&shape("", &ap(0, "kept"))), &names, &[("ppt/_rels/presentation.xml.rels", &PRESENTATION_RELS.replace("</Relationships>", &format!("{rels}</Relationships>")))]);
    let doc = super::pptx::PptxReader.to_markdown(&file, Format::Pptx, &caps).unwrap();
    assert!(doc.markdown.contains("\n\nkept\n") && doc.warnings.is_empty(), "{:?} {:?}", doc.markdown, doc.warnings);
    // PPTX: the text of a slide number does not count against the char cap.
    let tree = [shape("<p:ph type=\"sldNum\"/>", &ap(0, &"9".repeat(2000))), shape("", &ap(0, "kept"))].concat();
    let doc = super::pptx::PptxReader.to_markdown(&pptx(&tree), Format::Pptx, &caps).unwrap();
    assert!(doc.markdown.contains("\n\nkept\n") && !doc.markdown.contains('9') && doc.warnings.is_empty(), "{:?}", doc.warnings);
    // Sheets and slides that start after a stop are counted, not shown empty.
    let big = format!("<table:table table:name=\"Big\"><table:table-row><table:table-cell><text:p>{}</text:p></table:table-cell></table:table-row></table:table>", "b".repeat(2000));
    let small = "<table:table table:name=\"Small\"><table:table-row><table:table-cell><text:p>s</text:p></table:table-cell></table:table-row></table:table>";
    let doc = super::odf::OdfReader.to_markdown(&odf("spreadsheet", &format!("{big}{small}")), Format::Ods, &caps).unwrap();
    assert!(doc.markdown.starts_with("<!-- sheet 1 of 2 -->\n## Sheet Big\n\n| bbb") && !doc.markdown.contains("Small"), "{}", doc.markdown);
    let page = |text: &str| format!("<draw:page><draw:frame><draw:text-box><text:p>{text}</text:p></draw:text-box></draw:frame></draw:page>");
    let doc = super::odf::OdfReader.to_markdown(&odf("presentation", &format!("{}{}", page(&"b".repeat(2000)), page("s"))), Format::Odp, &caps).unwrap();
    assert!(doc.markdown.starts_with("<!-- slide 1 of 2 -->\n") && !doc.markdown.contains("slide 2") && !doc.markdown.contains("(no text"), "{}", doc.markdown);
    // XLSX: a sheet list after relationships that the cap stopped is a cap.
    let rels: String = (0..2000).map(|i| format!("<Relationship Id=\"rL{i}\" Type=\"t/hyperlink\" Target=\"https://example.org/{i}\" TargetMode=\"External\"/>")).collect();
    let names = ["[Content_Types].xml", "_rels/.rels", "xl/workbook.xml", "xl/_rels/workbook.xml.rels", "xl/sharedStrings.xml", "xl/styles.xml", "xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"];
    let file = rebuilt(&xlsx("", ""), &names, &[("xl/_rels/workbook.xml.rels", &WORKBOOK_RELS.replace("<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">", &format!("<Relationships>{rels}")))]);
    let doc = super::xlsx::XlsxReader.to_markdown(&file, Format::Xlsx, &caps).unwrap();
    assert!(!doc.markdown.contains("not found") && doc.warnings == [cap], "{:?} {:?}", doc.markdown, doc.warnings);
}

#[test]
fn list_levels_pdf_caps_and_zip_records_are_exact() {
    // A paragraph level wins over the level of the style that gives the list.
    let numbering = r#"<w:numbering xmlns:w="w"><w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:numFmt w:val="decimal"/></w:lvl><w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/></w:lvl></w:abstractNum><w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>"#;
    let styles = r#"<w:styles xmlns:w="w"><w:style w:styleId="L"><w:pPr><w:numPr><w:numId w:val="1"/><w:ilvl w:val="0"/></w:numPr></w:pPr></w:style></w:styles>"#;
    let body = "<w:p><w:pPr><w:pStyle w:val=\"L\"/><w:numPr><w:ilvl w:val=\"1\"/></w:numPr></w:pPr><w:r><w:t>kept</w:t></w:r></w:p>";
    let (_, doc) = read(&docx_with(body, &[("word/numbering.xml", numbering), ("word/styles.xml", styles)]), "a.docx", &Caps::default()).unwrap();
    assert_eq!(doc.markdown, "<!-- page 1 of 1 -->\n## Page 1\n\n    - kept\n");
    // PDF: the caller's caps limit the extraction, not only the output.
    let file = crate::pdf::tests::sample(&["first page", "second page"]);
    let limits = crate::pdf::Limits { pages: 1, text_bytes: MIN_WORK_BYTES, ..crate::pdf::Limits::DEFAULT };
    let never = std::sync::atomic::AtomicBool::new(false);
    let work = crate::pdf::budget::Budget::new(&limits, &never);
    let x = crate::pdf::extract(&file, &limits, &work).unwrap();
    assert_eq!((x.total, x.pages.len()), (2, 1));
    assert!(x.body.contains("first page") && !x.body.contains("second"));
    let (_, doc) = read(&file, "a.pdf", &Caps { max_pages: 1, ..Caps::default() }).unwrap();
    assert_eq!((doc.markdown.as_str(), doc.warnings), ("<!-- page 1 of 2 -->\n## Page 1\n\nfirst page\n", vec!["pages after 1 of 2 (pages cap)".to_owned()]));
    let long = crate::pdf::tests::sample(&[&"word ".repeat(40_000)]);
    let doc = super::pdf::PdfReader.to_markdown(&long, Format::Pdf, &Caps { max_chars: 1000, ..Caps::default() }).unwrap();
    assert!(doc.markdown.len() < 2000 && doc.warnings == [super::char_cap_warning(1000)], "{} {:?}", doc.markdown.len(), doc.warnings);
    // ZIP: a false end record in the comment that names only the first
    // directory entry is not the archive's end record.
    let mut file = docx(&p("kept"));
    let n = file.len();
    let (size, start) = (u32::from_le_bytes(file[n - 10..n - 6].try_into().unwrap()), u32::from_le_bytes(file[n - 6..n - 2].try_into().unwrap()));
    let at = start as usize;
    let first = 46 + [28, 30, 32].iter().map(|&k| u16::from_le_bytes([file[at + k], file[at + k + 1]]) as u32).sum::<u32>();
    assert!(first < size);
    let mut fake = b"PK\x05\x06\0\0\0\0\x01\0\x01\0".to_vec();
    fake.extend_from_slice(&first.to_le_bytes());
    fake.extend_from_slice(&start.to_le_bytes());
    fake.extend_from_slice(&[0, 0, b'x']);
    let mut comment = b"x".to_vec();
    comment.extend_from_slice(&fake[..fake.len() - 1]);
    file[n - 2..].copy_from_slice(&(comment.len() as u16).to_le_bytes());
    file.extend_from_slice(&comment);
    assert_eq!(read(&file, "a.docx", &Caps::default()).unwrap().1.markdown, "<!-- page 1 of 1 -->\n## Page 1\n\nkept\n");
}

#[test]
fn cut_values_slots_paths_and_scopes_read_exactly() {
    let caps = Caps { max_chars: 1000, ..Caps::default() };
    // XLSX: under metadata pressure a cell holds its slot, then keeps the
    // start of its text; a note shows only whole.
    let formats: String = (0..2005).map(|i| format!("<numFmt numFmtId=\"{}\" formatCode=\"0.0\"/>", 200 + i)).collect();
    let styles = format!("<styleSheet><numFmts>{formats}</numFmts><cellXfs><xf numFmtId=\"0\"/></cellXfs></styleSheet>");
    let value = format!("kept{}", "v".repeat(996));
    let names = ["[Content_Types].xml", "_rels/.rels", "xl/workbook.xml", "xl/_rels/workbook.xml.rels", "xl/sharedStrings.xml", "xl/styles.xml", "xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"];
    let sheet = format!("<row r=\"1\"><c r=\"A1\" t=\"inlineStr\"><is><t>{value}</t></is></c></row>");
    let file = rebuilt(&xlsx(&sheet, ""), &names, &[("xl/styles.xml", &styles)]);
    let doc = super::xlsx::XlsxReader.to_markdown(&file, Format::Xlsx, &caps).unwrap();
    assert!(doc.markdown.contains("\n\n| kept") && !doc.markdown.contains("(emp"), "{:?}", doc.markdown);
    assert_eq!(doc.warnings.len(), 1, "{:?}", doc.warnings);
    // XML: a cut part keeps the start of its last CDATA section and says the
    // text was cut; a whole section is not cut.
    let mut scan = Scanner::new("<a><![CDATA[kept  ]", true);
    assert!(matches!(scan.next().unwrap(), Some(Event::Open(_))));
    assert!(matches!(scan.next().unwrap(), Some(Event::Cdata("kept  "))) && scan.text_cut());
    assert!(matches!(scan.next().unwrap(), Some(Event::Close("a"))));
    let mut scan = Scanner::new("<a><![CDATA[kept]]></a>", true);
    scan.next().unwrap();
    assert!(matches!(scan.next().unwrap(), Some(Event::Cdata("kept"))) && !scan.text_cut());
    // XLSX: a string index or number that a cut ends inside is not read.
    let strings = ["zero", "one"];
    let cut = "<row r=\"1\"><c r=\"A1\" t=\"s\"><v>1</v></c><c r=\"B1\"><v>7</v></c><c r=\"C1\" t=\"s\"><v>0";
    assert_eq!(super::xlsx::test_sheet(cut, true, &strings), "| one | 7 |\n| --- | --- |");
    let cut = "<row r=\"1\"><c r=\"A1\" t=\"s\"><v>1</v></c><c r=\"B1\"><v>123";
    assert_eq!(super::xlsx::test_sheet(cut, true, &strings), "| one |\n| --- |");
    // ZIP: false end records that name the real directory walk it once.
    let entries: Vec<Entry> = (0..45_000).map(|i| Entry { name: Box::leak(format!("e{i}").into_boxed_str()), data: Vec::new(), deflate: false, raw: None }).collect();
    let mut file = zip_file(&entries);
    let n = file.len();
    let start = u32::from_le_bytes(file[n - 6..n - 2].try_into().unwrap());
    let fakes = 7;
    let mut comment = Vec::new();
    for k in 0..fakes {
        let at = (n + 22 * k) as u32;
        comment.extend_from_slice(b"PK\x05\x06\0\0\0\0\x10\x27\x10\x27");
        comment.extend_from_slice(&(at - start).to_le_bytes());
        comment.extend_from_slice(&start.to_le_bytes());
        comment.extend_from_slice(&((22 * (fakes - k - 1)) as u16).to_le_bytes());
    }
    file[n - 2..].copy_from_slice(&(comment.len() as u16).to_le_bytes());
    file.extend_from_slice(&comment);
    assert_eq!(Zip::open(&file, Format::Docx).unwrap().part("e44999").unwrap().unwrap().data, b"");
    // EPUB: a package path longer than the attribute cap still reads; one
    // longer than a ZIP name names its limit.
    let dir = "d/".repeat(2050);
    let opf = r#"<package><metadata><dc:title>T</dc:title></metadata><manifest><item id="c" href="c.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c"/></spine></package>"#;
    let book = |path: &str| {
        let container = CONTAINER.replace("OEBPS/content.opf", path);
        zip_file(&[stored("mimetype", "application/epub+zip"), stored("META-INF/container.xml", &container), stored(&format!("{dir}content.opf"), opf), stored(&format!("{dir}c.xhtml"), "<html><body><p>kept</p></body></html>")])
    };
    let (_, doc) = read(&book(&format!("{dir}content.opf")), "a.epub", &Caps::default()).unwrap();
    assert_eq!((doc.markdown.as_str(), doc.warnings.is_empty()), ("# T\n\n<!-- chapter 1 of 1 -->\n## Chapter 1\n\nkept\n", true));
    let err = read(&book(&"p/".repeat(40_000)), "a.epub", &Caps::default()).unwrap_err().to_string();
    assert!(err.contains("has a part path longer than 65535 bytes"), "{err}");
    // PDF: no text in the pages that the page cap let it read is not a file
    // with no text layer.
    let file = crate::pdf::tests::sample(&["", "kept"]);
    let (_, doc) = read(&file, "a.pdf", &Caps { max_pages: 1, ..Caps::default() }).unwrap();
    assert!(doc.markdown.starts_with("no text layer in the first 1 of 2 pages"), "{}", doc.markdown);
    assert_eq!(doc.warnings, ["pages after 1 of 2 (pages cap)"]);
    // DOCX: an explicit body text outline level wins over a heading style;
    // the list number of the style stays.
    let body = "<w:p><w:pPr><w:pStyle w:val=\"Heading2\"/><w:outlineLvl w:val=\"9\"/></w:pPr><w:r><w:t>kept</w:t></w:r></w:p>";
    assert_eq!(docx_md(body), "<!-- page 1 of 1 -->\n## Page 1\n\n3. kept\n");
    assert!(docx_md(&styled("Heading2", "head")).contains("\n## head\n"), "{}", docx_md(&styled("Heading2", "head")));
}
