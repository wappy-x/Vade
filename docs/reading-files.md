# Reading files

The engine reads only text. Vade turns other files into Markdown before the
model sees them. One reader layer in the harness
([`harness/src/readers/`](../harness/src/readers/mod.rs)) does this work for
both of its users:

- `vade read <path>` prints a local file as Markdown. It needs no engine, no
  network and no config.
- The agent's `workspace_read` tool reads documents as Markdown. UTF-8 text
  files are not changed: the tool returns their exact bytes, as before.

The readers use only the Rust standard library.

## Formats

| Format | Extensions | Status | What you get |
| --- | --- | --- | --- |
| Plain text | `.txt`, `.log` and any file that decodes as text | Supported | The text. UTF-8, UTF-8 with a byte order mark, UTF-16 LE and BE (with or without a byte order mark) and Windows-1252 (a Latin-1 superset) are detected. A note names any encoding that is not UTF-8. A file with a UTF-8 byte order mark and bytes that are not UTF-8, a sequence cut at the end of the file included, shows those bytes as U+FFFD, with a note. An odd last byte of UTF-16 text shows as U+FFFD. A NUL byte anywhere makes the file binary |
| Markdown | `.md`, `.markdown`, `.mdx` | Supported | The file as written |
| HTML, XHTML | `.html`, `.htm`, `.xhtml` | Supported | Headings, paragraphs, lists, links, emphasis, inline code, code blocks, quotes, tables and image alt text. Scripts, styles, templates and the document head are dropped. A template, SVG or MathML element ends at its own close tag, also when an element of the same name is inside it, and a prefixed one such as `svg:svg` ends at its prefixed close tag. Tags in its comments, CDATA sections, quoted attribute values, scripts, styles, text areas and titles do not count. XHTML empty elements such as `<blockquote/>` read as empty. Quotes, tables and code blocks nest at most 32 levels deep; a deeper one reads as text in the block around it. A link inside a link ends the first link. Lists indent at most 32 levels; deeper items are list items at that indent. List numbers stop at 999,999,999, the Markdown limit |
| CSV, TSV | `.csv`, `.tsv`, `.tab` | Supported | One GitHub Markdown table. The first row is the header. Quoted fields follow RFC 4180. The CSV delimiter (comma, semicolon, tab or pipe) is detected. A quoted field with no closing quote is an error |
| JSON | `.json`, `.jsonl`, `.ipynb` and others | Supported | A `json` code block. Minified JSON up to 256 KiB is indented by two spaces; other JSON is kept as written |
| YAML, TOML, XML | `.yaml`, `.yml`, `.toml`, `.xml`, `.svg`, `.plist` and others | Supported | A code block in the language |
| Source code | `.rs`, `.py`, `.c`, `.ts`, `.go`, `Makefile` and about 120 more | Supported | A code block in the language of the extension |
| RTF | `.rtf` | Supported | Plain text. `\par` and `\line` end a line. Formatting, fonts, pictures, field codes and `\binN` data are dropped; table cells are separated by ` \| `. `\'hh` bytes decode in the code page of the font charset (`\fcharsetN`), or else in the document code page (`\ansicpgN` or `\mac`): Windows-1252, Windows-1250, Windows-1251 or Mac Roman. `\plain` goes back to the default font (`\deffN`). In another code page, for example the double-byte Japanese, Chinese and Korean ones, they show as U+FFFD and `skipped:` names the code page |
| PDF | `.pdf` | Agent tool only | `vade read` prints one line: `PDF files are not supported yet; no text was read`. The agent tool `workspace_read` reads the text layer of a PDF (see [The agent read tool](#the-agent-read-tool)) |
| DOCX, XLSX, PPTX, EPUB | `.docx`, `.xlsx`, `.pptx`, `.epub` | Not yet | One line that names the format |
| ODT, ODS, ODP | `.odt`, `.ods`, `.odp` | Not yet | One line that names the format |
| Images, scanned PDFs | `.png`, `.jpg`, `.gif`, `.tif`, `.webp` | Not yet | One line. Vade has no text recognition (OCR) |
| Archives, executables, databases | ZIP, gzip, 7z, RAR, ELF, Mach-O, SQLite, WebAssembly, binary property lists, legacy Office | Not read | One line that names the format |

Vade finds the format from the file contents first. It uses the extension only
when the contents do not decide. For example, a file named `notes.txt` that
starts with `%PDF-` is read as a PDF. ZIP magic bytes are content too. DOCX,
XLSX, PPTX, EPUB, ODF and plain ZIP files start with the same bytes, so the
extension then only picks one of these ZIP-based formats. A ZIP file named
`notes.txt` is a ZIP archive, not text. A file that matches no format and does
not decode as text is refused as binary.

## Output conventions

All readers follow these rules. A new reader must follow them too.

**Header line.** The first line names the file, the format, the page count,
the character count of the body and what was skipped:

```text
<!-- vade read: report.pdf | PDF | pages 12 | chars 34567 | skipped: 3 images -->
```

A file with no pages shows `pages 1`. `skipped: none` means nothing was left
out. The header is an HTML comment, so a Markdown viewer hides it.

**Page markers.** Paged formats (PDF and office files, when their readers are
present) put a marker before each page. No reader in this release has pages:
they print no markers and the header shows `pages 1`. The form is:

```text
<!-- page 2 of 12 -->
## Page 2
```

Slides use the same form with `slide` and `Slide`. Sheets use
`<!-- sheet N of M -->` and then `## Sheet <name>`.

**Tables** are GitHub Markdown tables. A `|` in a cell becomes `\|`, and a line
break in a cell becomes `<br>`.

**Code and data** are fenced code blocks with the language. The fence is
longer than any run of backticks in the file.

**Caps.** Each read has these limits. The readers apply them while they
convert, so a small file cannot build a large result before the cut:

| Cap | Value | When it is hit |
| --- | --- | --- |
| File size | 50 MiB (52,428,800 bytes) | The file is refused with one line |
| Pages, slides or sheets | 2,000 | Paged formats only (none in this release). The reader stops and adds `pages after 2000 of N (pages cap)` to `skipped:` |
| Characters of Markdown | 2,000,000 | The text is cut and `text after N chars (char cap)` goes in `skipped:` |
| Table rows (CSV, one sheet) | 10,000 data rows | The table stops and `rows after 10000 of N (row cap)` goes in `skipped:` |
| Conversion work | 16 bytes for each character of the character cap, at least 64 KiB (32,000,000 bytes by default) | The reader stops and `text after N bytes of conversion work (memory cap)` goes in `skipped:`. Table padding, list indents, quote prefixes and text copied out of nested blocks count against it. Parser storage counts too: open RTF groups and fonts, HTML table rows and cells, link targets and the JSON indent. A CSV cell or an HTML attribute value that does not fit in the room left also stops the reader with this warning |

When something is skipped, a closing comment repeats the `skipped:` list after
the text, so a reader of the end of the output also sees it.

**Errors.** An unsupported, encrypted, damaged or image-only file gives one
clear line and no partial text:

```text
PDF files are not supported yet; no text was read
this PDF file is encrypted; Vade cannot read encrypted files
this PDF file has no text layer (3 pages; scanned images?); Vade does not read images yet
Vade does not read PNG image files; no text was read
binary file with no known format; no text was read
damaged CSV file: the quoted field in record 2 has no closing quote; no text was read
file is 60000000 bytes; the reader limit is 52428800 bytes
```

## `vade read`

```sh
vade read <PATH> [--max-chars N]
```

`--max-chars` sets the character cap, from 1 to 2,000,000 (the default). The
exit code is 0 when Markdown was printed, 1 when the file cannot be read and 2
for a usage error. Errors go to standard error. When the reader of the output
closes it early, as `| head` does, the exit code is 0.

`PATH` must be a regular file or a symbolic link to one. A directory, named
pipe, socket or device is refused with one line before it is opened, so a pipe
with no writer cannot make `vade read` wait.

A CSV file:

```text
$ vade read cities.csv
<!-- vade read: cities.csv | CSV | pages 1 | chars 79 | skipped: none -->
| city | population |
| --- | --- |
| Austin | 979,882 |
| Paris, TX | 24476 |
```

An HTML file. The style block in the head is dropped:

````text
$ vade read notes.html
<!-- vade read: notes.html | HTML | pages 1 | chars 84 | skipped: none -->
# Release notes

Version **2** adds [readers](https://example.org/r).

- HTML
- CSV
````

Minified JSON:

````text
$ vade read meta.json
<!-- vade read: meta.json | JSON | pages 1 | chars 47 | skipped: none -->
```json
{
  "name": "vade",
  "ok": true
}
```
````

A character cap:

```text
$ vade read --max-chars 20 notes.html
<!-- vade read: notes.html | HTML | pages 1 | chars 20 | skipped: text after 20 chars (char cap) -->
# Release notes

Ver

<!-- vade read: skipped: text after 20 chars (char cap) -->
```

## The agent read tool

`workspace_read` keeps the exact bytes of every UTF-8 text file, including
HTML, CSV and JSON, so the agent can edit them. These files are read as
Markdown instead:

- files whose contents or extension name a document format (RTF and the
  office formats when their readers are present);
- text files that are not UTF-8 (UTF-16 or Windows-1252). UTF-16 with no
  byte order mark is found from its zero bytes, because these bytes can also
  be valid UTF-8.

A PDF file (by its `%PDF-` bytes, up to 50 MiB) reads as its extracted text
layer, page by page, not through these readers. `offset` and `limit` then count
bytes of that text.
A file named `.pdf` that this test does not take reads as text when it is UTF-8
text; otherwise it is refused with one line. Files over 50 MiB are refused
before any read.

The encoding selects the path before a size limit applies. UTF-8 text over
4 MiB is refused, as before. The source limit for the other files is 50 MiB.
The tool's `offset` and `limit` then count bytes of the Markdown, header line
included. A partial result starts with a line such as
`Read Markdown bytes 0..4096 of 9120, converted from RTF (4096 bytes; range
limit 4096; next offset 4096).`

The chat's `/read` preview and `@file` attachments still accept UTF-8 text
only.

## Add a reader

A reader is one type that implements the `Reader` trait in
[`harness/src/readers/mod.rs`](../harness/src/readers/mod.rs):

```rust
pub trait Reader: Sync {
    fn sniff(&self, bytes: &[u8], name: &str) -> Option<Format>;
    fn to_markdown(&self, bytes: &[u8], format: Format, caps: &Caps) -> Result<Doc, ReadError>;
}
```

- `sniff` returns the format for magic bytes. The registry calls every reader
  with an empty `name` first and with the file name second, so the contents
  win over the extension.
- `to_markdown` returns a `Doc`: the Markdown body without the header, the
  page count and the warnings for `skipped:`. Use `page_marker`,
  `slide_marker`, `sheet_marker`, `table`, `cell` and `fence` from the module,
  and `html_to_markdown` for XHTML parts (`html_to_markdown_in` with your
  budget). Respect `caps.max_pages` and
  `caps.max_rows`.
- Make one `Budget` from `caps` (`Budget::new`) and take every piece of
  output through it (`push`, `text`, or `work` for markup). Stop when
  `full()` is true and add `warning()` to the warnings. Build tables with
  `table_in` and the same budget. The registry still cuts the result to the
  character cap, but it cannot undo memory that a reader already used.
- Count storage that grows with the input (open groups, rows, fonts) with
  `reserve` before you allocate it, and give it back with `release` when you
  free it. Keep no copy of source text that is larger than `room()`, and
  stop with a warning (not in silence) when a cut leaves text out. Scan each
  source byte once: a search for the end of a run stops at the end of the
  part you convert.
- Return `ReadError::Encrypted`, `ReadError::NoText` or `ReadError::Malformed`
  instead of partial text.
- Add the reader to `READERS` before the text reader, and remove its format
  from the fallback table `known()`.

Test each format with small inline fixtures, and include cut and corrupted
input: a reader must never panic. Also test a small input that would grow
large, and assert the output size with a small character cap.

## Known limits

- HTML emphasis markers stay where the tags are, so `<b>bold </b>word` gives
  `**bold **word`, which some Markdown viewers do not show as bold.
- A table inside a table cell becomes the text of that cell.
- RTF tables become lines with cells separated by ` | `, not Markdown tables.
- RTF `\'hh` bytes in double-byte code pages (Japanese, Chinese, Korean) and
  in single-byte code pages other than Windows-1250, 1251 and 1252 and Mac
  Roman show as U+FFFD. Text written as `\uN`, which most current writers
  use, is not affected.
- Plain text that is not UTF-8 or UTF-16 is read as Windows-1252. Other legacy
  code pages are not detected.
