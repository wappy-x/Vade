//! PPTX (PresentationML) to Markdown: one slide marker for each slide, in
//! presentation order, then the slide title as a heading, the text of each
//! shape (body placeholders as bullet lists, with their levels), tables,
//! images by alt text and part name, and the speaker notes.
//!
//! Skipped: slide numbers, dates, footers, the fallback copy of alternate
//! content, charts, SmartArt, layouts, masters, comments and formatting.
use super::xml::{self, Event, Scanner};
use super::zip::{self, Rels};
use super::{Budget, Caps, Doc, Format, ReadError, Reader, cell, extension, page_cap_warning, slide_marker, table_in};

pub(super) struct PptxReader;

/// Elements whose content is not slide text.
const SKIP: &[&str] = &["Fallback", "extLst", "rPr", "endParaRPr", "spPr", "style", "tblPr", "tcPr", "bodyPr", "lstStyle"];
/// Placeholders that hold slide furniture, not content.
const FURNITURE: &[&str] = &["sldNum", "dt", "ftr", "hdr", "sldImg"];
/// Storage of one table row or cell, for the work cap.
const SLOT: usize = 32;

impl Reader for PptxReader {
    fn sniff(&self, _: &[u8], name: &str) -> Option<Format> {
        matches!(extension(name).as_str(), "pptx" | "pptm" | "potx" | "potm" | "ppsx" | "ppsm").then_some(Format::Pptx)
    }

    fn to_markdown(&self, bytes: &[u8], format: Format, caps: &Caps) -> Result<Doc, ReadError> {
        let mut budget = Budget::new(caps);
        let mut zip = super::office_zip(bytes, format)?;
        let main = zip::main_part(&mut zip, &mut budget)?.unwrap_or_else(|| "ppt/presentation.xml".to_owned());
        let bad = |name: &str, why: String| ReadError::Malformed(format, format!("{name} {why}"));
        let rels = zip.rels(&main, &mut budget)?;
        let (text, cut) = zip.text(&main)?.ok_or_else(|| ReadError::Malformed(format, format!("the archive has no {main}")))?;
        let (ids, ids_held) = slide_ids(&text, cut, &mut budget).map_err(|why| bad(&main, why))?;
        let mut slides = Vec::new();
        for r in ids.iter().filter_map(|id| rels.get(id)).filter(|r| r.kind == "slide" && !r.external) {
            if !budget.reserve(SLOT + r.target.len()) {
                break;
            }
            slides.push(r.target.clone());
        }
        // The ids and relationships are not needed after this: their bytes go back.
        budget.release(ids_held + rels.held);
        drop((ids, rels));
        if slides.is_empty() {
            // A slide list that the work cap stopped, or whose ids the
            // attribute cap left out, is not damage.
            if budget.full() || budget.long_attr {
                let mut doc = Doc::new(String::new(), 0);
                doc.warnings.extend(budget.warning());
                return Ok(doc);
            }
            return Err(ReadError::Malformed(format, format!("{main} lists no slides")));
        }
        let total = slides.len();
        let mut doc = Doc::new(String::new(), total);
        let mut images = 0;
        for (n, name) in slides.iter().enumerate() {
            if n == caps.max_pages {
                doc.warnings.push(page_cap_warning(caps.max_pages, total, "slides"));
                break;
            }
            if budget.full() {
                break;
            }
            let marker = slide_marker(n + 1, total);
            let marker = if n == 0 { marker.trim_start() } else { &marker };
            if !budget.fits(marker.len()) {
                budget.cut();
                break;
            }
            doc.markdown.push_str(budget.work(marker));
            let (text, cut) = zip.text(name)?.ok_or_else(|| ReadError::Malformed(format, format!("the archive has no {name}")))?;
            let slide_rels = zip.rels(name, &mut budget)?;
            let slide = Slide::read(&text, cut, &slide_rels, false, &mut budget).map_err(|why| bad(name, why))?;
            images += slide.images;
            let mut body = String::new();
            if slide.hidden {
                body.push_str("(hidden slide)\n\n");
            }
            if !slide.title.is_empty() {
                body.push_str(&format!("### {}\n\n", slide.title));
            }
            body.push_str(&slide.body);
            let notes = slide_rels.first("notesSlide").map(str::to_owned);
            if let Some(notes) = notes
                && let Some((text, cut)) = zip.text(&notes)?
            {
                let note = Slide::read(&text, cut, &Rels::default(), true, &mut budget).map_err(|why| bad(&notes, why))?;
                budget.release(note.body.len());
                if !note.body.trim().is_empty() {
                    body.push_str(&format!("\n\n#### Notes\n\n{}", note.body.trim()));
                }
            }
            // The slide's text moves into the output: its bytes, and those of
            // its relationships, go back first.
            budget.release(slide.title.len() + slide.body.len() + slide_rels.held + SLOT + name.len());
            let body = body.trim();
            if body.is_empty() {
                budget.whole(&mut doc.markdown, "(no text on this slide)");
            } else {
                doc.markdown.push_str(budget.work(body));
            }
            doc.markdown.push('\n');
        }
        if images > 0 {
            doc.warnings.push(format!("{images} images (listed, not read)"));
        }
        doc.warnings.append(&mut zip.warnings);
        doc.warnings.extend(budget.warning());
        Ok(doc)
    }
}

/// The relationship ids of the slides, in presentation order, and the work
/// bytes they hold. Each id counts against the work cap before it is stored.
fn slide_ids(text: &str, cut: bool, budget: &mut Budget) -> Result<(Vec<String>, usize), String> {
    let mut scan = Scanner::new(text, cut);
    let (mut out, mut held) = (Vec::new(), 0);
    while let Some(event) = scan.next()? {
        if let Event::Open(tag) = event
            && tag.local() == "sldId"
            && let Some(id) = tag.attr("r:id")?
        {
            if !budget.reserve(SLOT + id.len()) {
                break;
            }
            held += SLOT + id.len();
            out.push(id.into_owned());
        }
    }
    scan.finish(budget)?;
    Ok((out, held))
}

#[derive(Default)]
struct Shape {
    /// The placeholder type; `obj` for a placeholder with no type.
    placeholder: Option<String>,
    paras: String,
    /// The previous paragraph was a bullet item.
    in_list: bool,
}

#[derive(Default)]
struct Para {
    text: String,
    level: usize,
    bullet: Option<bool>,
}

#[derive(Default)]
struct Table {
    rows: Vec<Vec<String>>,
    cell: String,
    /// Work bytes that the table's storage and rows hold, not its open cell.
    held: usize,
}

#[derive(Default)]
struct Slide {
    title: String,
    body: String,
    hidden: bool,
    images: usize,
}

impl Slide {
    /// The title, body and image count of a slide part. For a notes part
    /// (`notes`), only the body placeholder counts.
    fn read(text: &str, cut: bool, rels: &Rels, notes: bool, budget: &mut Budget) -> Result<Self, String> {
        let mut scan = Scanner::new(text, cut);
        let mut out = Slide::default();
        let (mut skip, mut shape, mut para, mut table) = (0usize, None::<Shape>, None::<Para>, None::<Table>);
        let mut in_text = false;
        let mut image: Option<(String, Option<String>)> = None;
        while let Some(event) = scan.next()? {
            match event {
                Event::Open(tag) => {
                    if skip > 0 || SKIP.contains(&tag.local()) {
                        skip += 1;
                        continue;
                    }
                    match tag.local() {
                        "sld" => out.hidden = tag.flag("show")? == Some(false),
                        "sp" => shape = Some(Shape::default()),
                        "ph" => {
                            if let Some(s) = shape.as_mut() {
                                s.placeholder = Some(tag.attr("type")?.map_or_else(|| "obj".to_owned(), |t| t.into_owned()));
                            }
                        }
                        "p" => para = Some(Para::default()),
                        "pPr" => {
                            if let Some(p) = para.as_mut() {
                                p.level = tag.attr("lvl")?.and_then(|l| l.parse().ok()).unwrap_or(0).min(8);
                            }
                        }
                        "buNone" | "buChar" | "buAutoNum" | "buBlip" => {
                            if let Some(p) = para.as_mut() {
                                p.bullet = Some(tag.local() != "buNone");
                            }
                        }
                        "t" => in_text = true,
                        "br" if !dropped(&shape, notes) => push(&mut para, "\n", budget),
                        "tbl" => {
                            if table.is_none() && budget.reserve(SLOT) {
                                table = Some(Table { held: SLOT, ..Table::default() });
                            }
                        }
                        "tr" => {
                            if let Some(t) = table.as_mut()
                                && !budget.full()
                                && budget.reserve(SLOT)
                            {
                                t.held += SLOT;
                                t.rows.push(Vec::new());
                            }
                        }
                        "tc" => {
                            if let Some(t) = table.as_mut() {
                                t.cell.clear();
                            }
                        }
                        "pic" => image = Some((String::new(), None)),
                        "cNvPr" => {
                            if let Some((alt, _)) = image.as_mut().filter(|i| i.0.is_empty()) {
                                *alt = tag.alt_text(&["descr", "name"])?;
                            }
                        }
                        "blip" => {
                            if let Some((_, slot)) = image.as_mut().filter(|i| i.1.is_none() && !budget.full()) {
                                *slot = tag.attr("r:embed")?.and_then(|id| rels.get(&id)).map(|r| r.target.clone());
                            }
                        }
                        _ => {}
                    }
                }
                Event::Close(name) => {
                    if skip > 0 {
                        skip -= 1;
                        continue;
                    }
                    match xml::local(name) {
                        "t" => in_text = false,
                        "p" => {
                            let Some(p) = para.take() else { continue };
                            let text = p.text.trim().to_owned();
                            budget.release(p.text.len());
                            if text.is_empty() {
                                continue;
                            }
                            if let Some(t) = table.as_mut() {
                                let sep = if t.cell.is_empty() { "" } else { "\n" };
                                t.cell.push_str(budget.work(&format!("{sep}{text}")));
                            } else if let Some(s) = shape.as_mut() {
                                let body = matches!(s.placeholder.as_deref(), Some("obj" | "body"));
                                let item = p.bullet.unwrap_or(body && !notes);
                                let line = if item { format!("{}- {text}", "    ".repeat(p.level)) } else { text };
                                let sep = if s.paras.is_empty() {
                                    ""
                                } else if item && s.in_list {
                                    "\n"
                                } else {
                                    "\n\n"
                                };
                                s.paras.push_str(budget.work(&format!("{sep}{line}")));
                                s.in_list = item;
                            }
                        }
                        "tc" => {
                            if let Some(t) = table.as_mut() {
                                // The cell's text moves into its row: its bytes go back first.
                                budget.release(t.cell.len());
                                let value = cell(&std::mem::take(&mut t.cell));
                                if !t.rows.is_empty() && budget.reserve(SLOT) {
                                    let value = budget.work(&value).to_owned();
                                    t.held += SLOT + value.len();
                                    if let Some(row) = t.rows.last_mut() {
                                        row.push(value);
                                    }
                                }
                            }
                        }
                        "tbl" => {
                            if let Some(t) = table.take() {
                                // The rows move into the slide: their bytes go back first.
                                budget.release(t.held + t.cell.len());
                                let rows: Vec<Vec<String>> = t.rows.into_iter().filter(|r| r.iter().any(|c| !c.is_empty())).collect();
                                if !rows.is_empty() {
                                    if !out.body.is_empty() {
                                        out.body.push_str(budget.work("\n\n"));
                                    }
                                    // The cells were counted as text once; the table counts as work.
                                    let counts = std::mem::replace(&mut budget.text_counts, false);
                                    let md = table_in(&rows, budget);
                                    budget.text_counts = counts;
                                    out.body.push_str(md.trim_end());
                                }
                            }
                        }
                        "sp" => {
                            let Some(s) = shape.take() else { continue };
                            let kind = s.placeholder.as_deref();
                            budget.release(s.paras.len());
                            if notes {
                                if kind == Some("body") {
                                    add_block(&mut out.body, &s.paras, budget);
                                }
                            } else if matches!(kind, Some("title" | "ctrTitle")) && out.title.is_empty() {
                                let title = s.paras.split_whitespace().collect::<Vec<_>>().join(" ");
                                out.title = budget.work(&title).to_owned();
                            } else if !kind.is_some_and(|k| FURNITURE.contains(&k)) {
                                add_block(&mut out.body, &s.paras, budget);
                            }
                        }
                        "pic" => {
                            if let Some((alt, Some(target))) = image.take()
                                && !notes
                            {
                                out.images += 1;
                                let alt = alt.replace(['[', ']', '\n'], " ");
                                let line = if alt.trim().is_empty() { format!("[image: {target}]") } else { format!("[image: {} ({target})]", alt.trim()) };
                                let line = budget.text(&line).to_owned();
                                add_block(&mut out.body, &line, budget);
                            }
                        }
                        _ => {}
                    }
                }
                // Only the text that can be kept is decoded; the text of a
                // shape that is left out is not counted.
                Event::Text(raw) if in_text && skip == 0 && !budget.full() && !dropped(&shape, notes) => {
                    push(&mut para, &xml::unescape_prefix(raw, budget.room())?, budget)
                }
                Event::Cdata(raw) if in_text && skip == 0 && !budget.full() && !dropped(&shape, notes) => push(&mut para, raw, budget),
                _ => {}
            }
        }
        scan.finish(budget)?;
        Ok(out)
    }
}

/// The body of the slide part `text` (cut by a cap when `cut`).
#[cfg(test)]
pub(super) fn test_slide(text: &str, cut: bool) -> Result<String, String> {
    Slide::read(text, cut, &Rels::default(), false, &mut Budget::new(&Caps::default())).map(|s| s.body)
}

/// True when the open shape is left out: slide furniture (a slide number,
/// date, header or footer), or on a notes page any shape but the notes body.
fn dropped(shape: &Option<Shape>, notes: bool) -> bool {
    let Some(kind) = shape.as_ref().map(|s| s.placeholder.as_deref()) else { return false };
    if notes { kind != Some("body") } else { kind.is_some_and(|k| FURNITURE.contains(&k)) }
}

fn push(para: &mut Option<Para>, text: &str, budget: &mut Budget) {
    if let Some(p) = para.as_mut() {
        p.text.push_str(budget.text(text));
    }
}

/// Appends a block to `body`, a blank line after the previous one.
fn add_block(body: &mut String, block: &str, budget: &mut Budget) {
    let block = block.trim_matches('\n');
    if block.trim().is_empty() {
        return;
    }
    if !body.is_empty() {
        body.push_str(budget.work("\n\n"));
    }
    body.push_str(budget.work(block));
}
