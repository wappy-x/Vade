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
| HTML, XHTML | `.html`, `.htm`, `.xhtml` | Supported | Headings, paragraphs, lists, links, emphasis, inline code, code blocks, quotes, tables and image alt text. Scripts, styles, templates and the document head are dropped. A template, SVG or MathML element ends at its own close tag, also when an element of the same name is inside it, and a prefixed one such as `svg:svg` ends at its prefixed close tag. Tags in its comments, CDATA sections, quoted attribute values, scripts, styles, text areas and titles do not count. XHTML empty elements such as `<blockquote/>` read as empty. Quotes, tables and code blocks nest at most 32 levels deep; a deeper one reads as text in the block around it. A link inside a link ends the first link. A paragraph that starts a list item stays on the marker line (`- text`). A nested list lines up under the text of its item, so it is three spaces in under `1. `. A list that a table cell, table or quote leaves open ends with it. A cell with `colspan` keeps the columns it spans, at most 64. An empty `<title/>` or `<script/>` inside an SVG, MathML or template element holds no text, and a `>` in a quoted value of a close tag does not end the tag. Lists indent at most 32 levels; deeper items are list items at that indent. List numbers stop at 999,999,999, the Markdown limit |
| CSV, TSV | `.csv`, `.tsv`, `.tab` | Supported | One GitHub Markdown table. The first row is the header. Quoted fields follow RFC 4180. The CSV delimiter (comma, semicolon, tab or pipe) is detected. A quoted field with no closing quote is an error |
| JSON | `.json`, `.jsonl`, `.ipynb` and others | Supported | A `json` code block. Minified JSON up to 256 KiB is indented by two spaces; other JSON is kept as written |
| YAML, TOML, XML | `.yaml`, `.yml`, `.toml`, `.xml`, `.svg`, `.plist` and others | Supported | A code block in the language |
| Source code | `.rs`, `.py`, `.c`, `.ts`, `.go`, `Makefile` and about 120 more | Supported | A code block in the language of the extension |
| RTF | `.rtf` | Supported | Plain text. `\par` and `\line` end a line. Formatting, fonts, pictures, field codes and `\binN` data are dropped; table cells are separated by ` \| `. `\'hh` bytes decode in the code page of the font charset (`\fcharsetN`), or else in the document code page (`\ansicpgN` or `\mac`): Windows-1252, Windows-1250, Windows-1251 or Mac Roman. `\plain` goes back to the default font (`\deffN`). In another code page, for example the double-byte Japanese, Chinese and Korean ones, they show as U+FFFD and `skipped:` names the code page |
| PDF | `.pdf` | Supported | The text layer, page by page, with a page marker before each page. Text follows the order of the content streams, so the layout is approximate. A file with no text layer (scanned pages) gives one line; Vade runs no text recognition (OCR). The notes of the PDF reader on its limits go in `skipped:`. The agent tool `workspace_read` reads PDF files through its own path (see [The agent read tool](#the-agent-read-tool)) |
| DOCX | `.docx`, `.docm`, `.dotx`, `.dotm` | Supported | A page marker before each page (see [Page markers](#output-conventions)). Headings come from the paragraph style (`Title` and `heading 1` to `heading 9`, also through `basedOn`) or the outline level; outline level 9 (body text) wins over a heading style. Bullet and numbered lists keep their level (four spaces for each level) and their numbers. Tables become Markdown tables; a merged cell keeps its columns and a table in a cell becomes text in that cell. Links keep their targets. Line breaks and tabs stay. Each image shows where it is in the text as `[image: <alt text> (<part name>)]`, or `[image: <part name>]` when it has no alt text, and `skipped:` counts the images. Tracked deletions, the properties before a tracked change, field codes, the fallback copy of alternate content, headers, footers, footnotes, comments and character formatting are dropped. A text box reads as text in its paragraph |
| XLSX | `.xlsx`, `.xlsm`, `.xltx`, `.xltm` | Supported | A sheet marker and a Markdown table for each sheet, in workbook order. The name of a hidden sheet ends in `(hidden)`. Cells show their saved values: text, numbers in their shortest form, `TRUE` and `FALSE`, error codes and the last result of a formula. A number in a date or time format shows as `2023-01-31`, `13:45:00` or `2023-01-31 13:45:00`, in the 1900 or 1904 date system of the file. The table spans the used columns. Empty rows are left out, and the first row with values is the header. The shared string table may use at most half of the conversion work cap, so the sheets keep room for their rows: past that, a cell that uses a later string is empty, and `skipped:` says `shared strings after N (memory cap; later ones show as empty cells)`. Formulas, charts, images, comments and formatting are dropped |
| PPTX | `.pptx`, `.pptm`, `.potx`, `.potm`, `.ppsx`, `.ppsm` | Supported | A slide marker for each slide, in show order. Then the slide title as a `###` heading, the text of each shape (body placeholders as bullet lists with their levels), tables, images as `[image: <alt text> (<part name>)]` and the speaker notes under `#### Notes`. A hidden slide says `(hidden slide)`. Slide numbers, dates, footers, charts, SmartArt, layouts and formatting are dropped |
| EPUB | `.epub` (also found by its `mimetype` entry) | Supported | The title and authors (the first 20, then `and N more`), then a chapter marker for each item of the spine (the reading order), each read as HTML. An item with no text, such as a cover image, says `(no text in this chapter)`. A book with encrypted content (DRM) is refused; font obfuscation is not encryption of the text |
| ODT | `.odt`, `.ott` (also found by its `mimetype` entry) | Supported | Headings by outline level, paragraphs, bullet and numbered lists with their levels (four spaces for each level), tables, links, line breaks and images as `[image: <name> (<part name>)]` where they are in the text (also in a table cell). A page marker before each page: LibreOffice marks where each page started in its last layout (soft page breaks); a file with no marks is one page. Annotations, footnotes, tracked changes and formatting are dropped |
| ODS | `.ods`, `.ots` | Supported | A sheet marker and a Markdown table for each sheet, with the text each cell shows (as formatted in the file). Repeated rows and columns expand only between cells with values. The first row with values is the header |
| ODP | `.odp`, `.otp` | Supported | A slide marker for each page, the title as a `###` heading, the text frames (lists as bullets), tables, images and the speaker notes under `#### Notes`. Page numbers, dates, headers and footers are dropped |
| PNG, JPEG, GIF, WebP images | `.png`, `.jpg`, `.gif`, `.webp` (found by contents) | Supported | One line with the format, the size in pixels and details such as the bit depth or the frame count, then the path and a note that Vade runs no text recognition (OCR). `skipped:` says `image pixels (no text recognition)` |
| ZIP, tar and gzip archives | `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.gz` (found by contents; an old tar file with no `ustar` magic by its name) | Supported | A line with the entry count and the unpacked size, then a table of the entries: path, size in bytes and type. Then a file marker before each file, and the file read by the reader for its format (see [Archives](#archives)) |
| TIFF images, scanned PDFs | `.tif`, `.tiff` | Not read | One line. Vade has no text recognition (OCR) |
| Other archives, executables, databases | 7z, RAR, xz, bzip2, zstd, ELF, Mach-O, SQLite, WebAssembly, binary property lists, legacy Office | Not read | One line that names the format |

Vade finds the format from the file contents first. It uses the extension only
when the contents do not decide. For example, a file named `notes.txt` that
starts with `%PDF-` is read as a PDF. ZIP magic bytes are content too. DOCX,
XLSX, PPTX, EPUB, ODF and plain ZIP files start with the same bytes, so the
extension then only picks one of these ZIP-based formats. EPUB and OpenDocument files name
their format in their first entry, `mimetype`, so they are found under any
name. A ZIP file named
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

**Page markers.** Paged formats put a marker before each page, and the header
shows the page count. Other readers print no markers, and the header shows
`pages 1`. The form is:

```text
<!-- page 2 of 12 -->
## Page 2
```

Slides use the same form with `slide` and `Slide`. EPUB chapters use `chapter` and
`Chapter`. Sheets use
`<!-- sheet N of M -->` and then `## Sheet <name>`. The files in an archive use
`<!-- file N of M -->` and then `## File <path>`.

A DOCX file has no fixed pages. When Word saves a file, it marks where each
page started in its last layout, and the DOCX reader uses those marks. Only
marks in the document text count: not marks in deleted text or text boxes.
A file with no marks, for example one that another program wrote, starts a
page at each hard page break, `pageBreakBefore` paragraph and section break
that starts a new page. A page that starts inside a table starts after the
table. When the memory cap stops a reader, the markers of the pages it kept
stay whole. The char cap cuts the finished Markdown at its last char, so a
very small `--max-chars` can cut a marker too.

**Tables** are GitHub Markdown tables. A `|` in a cell becomes `\|`, and a line
break in a cell becomes `<br>`.

**Code and data** are fenced code blocks with the language. The fence is
longer than any run of backticks in the file.

**Caps.** Each read has these limits. The readers apply them while they
convert, so a small file cannot build a large result before the cut:

| Cap | Value | When it is hit |
| --- | --- | --- |
| File size | 50 MiB (52,428,800 bytes) | The file is refused with one line |
| Pages, slides, sheets or chapters | 2,000 | Paged formats only. The reader stops and adds `pages after 2000 of N (pages cap)` to `skipped:` |
| Characters of Markdown | 2,000,000 | The text is cut and `text after N chars (char cap)` goes in `skipped:` |
| Table rows (CSV, one sheet) | 10,000 data rows | The table stops and `rows after 10000 of N (row cap)` goes in `skipped:`. For a sheet, the phrase ends in `in sheet <name> (row cap)` |
| Conversion work | 16 bytes for each character of the character cap, at least 64 KiB (32,000,000 bytes by default) | The reader stops and `text after N bytes of conversion work (memory cap)` goes in `skipped:`. Table padding, list indents, quote prefixes and text copied out of nested blocks count against it. Parser storage counts too: open RTF groups and fonts, HTML table rows and cells, link targets and the JSON indent. A CSV cell or an HTML attribute value that does not fit in the room left also stops the reader with this warning |
| Unzipped data (ZIP-based files: DOCX, XLSX, PPTX, EPUB, ODT, ODS, ODP) | 64 MiB (67,108,864 bytes) for one part, 256 MiB (268,435,456 bytes) for one read | The part keeps its first bytes (stored and compressed parts alike), and `<part> after N bytes (unzip cap)` goes in `skipped:`. A character reference or a UTF-8 character that the cap cuts is left out. The paragraphs, cells, tables and CDATA sections that the cut leaves open keep their text. A spreadsheet cell value in `<v>` (a number, a shared string index, a formula result or an error code) that the cut ends inside is left out, because its value is not known. A ZIP file with more than 65,535 entries is refused |
| XML attribute value (DOCX, XLSX, PPTX, EPUB, ODT, ODS, ODP) | 4,096 bytes | The value is not read: a style, list, link target or image reads as if it had none, and `attribute values over 4096 bytes left out (attribute cap)` goes in `skipped:`. The text stays. Image alt text keeps its first 200 characters and then ends in `...`. A part path (an internal link target, an EPUB package or chapter path) is read up to 65,535 bytes, the ZIP name limit. A longer internal link target is left out with this warning. A longer path to the main document part, the EPUB package or a chapter refuses the file |

After the character cap or the memory cap stops a reader, the pages, slides,
sheets and chapters that start later get no marker; the page count in the
header still counts them. For PDF, the page cap and the memory cap also limit
the pages and text that the PDF reader extracts.

When something is skipped, a closing comment repeats the `skipped:` list after
the text, so a reader of the end of the output also sees it.

**Errors.** An unsupported, encrypted, damaged or image-only file gives one
clear line and no partial text:

```text
this PDF file is encrypted; Vade cannot read encrypted files
this PDF file has no text layer (3 pages; scanned images?); Vade does not read images yet
Vade does not read TIFF image files; no text was read
binary file with no known format; no text was read
damaged CSV file: the quoted field in record 2 has no closing quote; no text was read
damaged DOCX file: the entry word/document.xml fails its CRC-32 check; no text was read
this DOCX file is encrypted; Vade cannot read encrypted files
file is 60000000 bytes; the reader limit is 52428800 bytes
```

## Archives

The archive reader lists every entry of a ZIP, tar or gzip archive, then reads
each file in it with the reader for that file's format. A `.tar.gz` or `.tgz`
file is one tar archive: the gzip data starts with a tar header, or the name
in the gzip header ends in `.tar`. A gzip file that holds no tar archive has
one entry: the file named in its header. A tar file that ends inside a
header, or after a long name with no entry, is damaged. ZIP entries with a name that is not UTF-8 are
not listed.

- Folders are listed. Symbolic links and hard links are listed and never
  followed. Encrypted ZIP entries are listed and not read.
- The files share the caps of the read: the character cap and the work cap,
  and the pages cap for the files read from one archive (`files after 2000 of
  N (files cap)`). The decoded bytes of all files together, at all levels,
  are at most the file size cap (50 MiB), also the bytes of gzip data that
  fails its check. A file in a tar archive is part of bytes already counted. A ZIP entry larger than the bytes
  left is not decoded, and its section says so. A ZIP entry that decodes to more
  than its declared size is damaged. Gzip data stops at the bytes left, with
  `<name> after N bytes (archive bytes cap)` in `skipped:`. The parts that an
  office or EPUB file in the archive unzips draw from the same bytes, with
  `<file>: <part> after N bytes (unzip cap)` in `skipped:`. So a compression
  bomb stops at the size of a file that Vade would read. A PDF file in the
  archive decodes its streams within the limits of the PDF reader.
- An archive in an archive is read, two levels deep in all. A deeper archive
  is listed in its parent but not opened, with
  `<path> (archive past 2 levels; depth cap)` in `skipped:`.
- A file in a nested archive shows as `<archive path>/<file path>`. A
  warning of a file reader starts with the file path.
- A file that cannot be read gives one line in its section, for example
  `(not read: this DOCX file is encrypted; Vade cannot read encrypted files)`.
  The other files still read.

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
- A ZIP-based reader opens the file with `office_zip` or `zip::Zip` and reads
  each part with `xml::Scanner`. The scanner checks that the XML is well
  formed and never expands declared entities. Unzipped parts stop at the unzip
  caps.
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
- DOCX bold, italic and other character formatting is not kept. Footnotes,
  comments, headers and footers are not read.
- XLSX merged cells show their value in the first cell only. A number shows
  its saved value, not its number format, except for dates and times; for
  example, a percentage shows as `0.25`.
- ODT list numbers show as digits, also for letter and Roman number formats.
  A list that continues its numbering continues only the last top-level list
  of the same style. ODT pages come only from soft page breaks.
- Archives: 7z, RAR, xz, bzip2 and zstd are not read (Vade has no decoders
  for them). Only the first member of a gzip file is read. Tar entries other
  than files, folders and links (devices, pipes) are listed and not read.
- PPTX shapes come in the order of the slide file, which is often, but not
  always, the reading order.
