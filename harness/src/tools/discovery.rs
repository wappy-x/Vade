//! Literal, bounded discovery over the same directory handles as reads/edits.
use super::{FileStamp, MAX_TEXT_BYTES, Workspace, error, safe_parts};
use crate::json::{self, Value};
use std::io::{self, Read};
use std::sync::atomic::{AtomicBool, Ordering};

const SEARCH_FILE_BYTES: usize = 512 * 1024;
const SEARCH_BYTES: usize = 8 * 1024 * 1024;
const SEARCH_FILES: usize = 512;
const SEARCH_DIRECTORIES: usize = 128;
const SEARCH_ENTRIES: usize = 4096;
const SEARCH_DEPTH: usize = 32;

fn number(value: usize) -> Value {
    Value::Number(value.to_string())
}
fn clipped(text: &str, limit: usize) -> &str {
    let mut end = text.len().min(limit);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}
/// Rows shown from the start of each range. The smallest range takes first:
/// each range gets an equal part of the bytes left, and a range that needs
/// less leaves the rest to the larger ranges. One range, or ranges that fit
/// together, show the same rows as one shared budget filled in order.
fn share_budget(rows: &[Vec<(usize, Value, usize)>], budget: usize) -> Vec<usize> {
    let need = |rows: &[(usize, Value, usize)]| rows.iter().fold(0usize, |sum, row| sum.saturating_add(row.2));
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&range| (need(&rows[range]), range));
    let mut shown = vec![0; rows.len()];
    let mut left = budget;
    for (done, &range) in order.iter().enumerate() {
        let share = left / (rows.len() - done);
        let mut used = 0usize;
        for row in &rows[range] {
            if row.2 > share - used {
                break;
            }
            used += row.2;
            shown[range] += 1;
        }
        left -= used;
    }
    shown
}
fn cancelled(cancel: &AtomicBool) -> io::Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(io::ErrorKind::Interrupted.into())
    } else {
        Ok(())
    }
}

impl Workspace {
    fn discovery_text(
        &self,
        path: &str,
        cap: usize,
        expected: Option<&FileStamp>,
    ) -> io::Result<String> {
        let (mut file, stamp) = self
            .open_snapshot(path)?
            .ok_or_else(|| error("No file exists at this path."))?;
        if expected.is_some_and(|expected| *expected != stamp) {
            return Err(error("source changed before discovery"));
        }
        if stamp.directory || stamp.bytes > cap as u64 {
            return Err(error(
                "Search and line reads need a regular file within the source size limit. Use workspace_read with offset and limit for a large file.",
            ));
        }
        let mut bytes = Vec::new();
        (&mut file).take((cap + 1) as u64).read_to_end(&mut bytes)?;
        if bytes.len() as u64 != stamp.bytes || FileStamp::from_metadata(&file.metadata()?) != stamp
        {
            return Err(error("source changed during discovery"));
        }
        // Check the header first: an ASCII-only PDF is valid UTF-8 too.
        if crate::pdf::is_pdf(&bytes[..bytes.len().min(1024)]) {
            return Err(error("This file is a PDF. Use workspace_read to read its text."));
        }
        String::from_utf8(bytes).map_err(|_| error("This file is not UTF-8 text, so the agent cannot search it."))
    }

    /// One-based inclusive lines, bounded in source bytes, lines and wire bytes.
    /// Long individual lines are left for byte-range reads, never silently lost.
    pub fn read_lines(
        &self,
        path: &str,
        start: usize,
        end: usize,
        cap: usize,
        cancel: &AtomicBool,
    ) -> io::Result<String> {
        self.read_line_ranges(path, &[(start, end)], cap, cancel)
    }

    /// Read disjoint ranges from one guarded source snapshot. At most eight
    /// ordered ranges and 200 selected lines share one output budget.
    pub fn read_line_ranges(
        &self,
        path: &str,
        ranges: &[(usize, usize)],
        cap: usize,
        cancel: &AtomicBool,
    ) -> io::Result<String> {
        self.read_line_ranges_capture(path, ranges, cap, cancel).map(|(output, _)| output)
    }
    pub(crate) fn read_line_ranges_capture(&self, path: &str, ranges: &[(usize, usize)], cap: usize, cancel: &AtomicBool) -> io::Result<(String, String)> {
        let mut selected = 0usize;
        let mut previous_end = 0;
        if ranges.is_empty() || ranges.len() > 8 || cap < 512 {
            return Err(error(
                "read needs 1–8 ordered disjoint ranges and output budget ≥512 bytes",
            ));
        }
        for &(start, end) in ranges {
            if start == 0 || start <= previous_end || end < start || end - start >= 200 {
                return Err(error(
                    "ranges must be one-based, ordered and disjoint, selecting ≤200 total lines",
                ));
            }
            selected += end - start + 1;
            previous_end = end;
        }
        if selected > 200 {
            return Err(error("ranges select more than 200 total lines"));
        }
        cancelled(cancel)?;
        let text = self.discovery_text(path, MAX_TEXT_BYTES, None)?;
        let cap = cap.min(super::MAX_READ_BYTES);
        let total = text.split_inclusive('\n').count();
        let requested = Value::Array(
            ranges
                .iter()
                .map(|&(start, end)| {
                    Value::Object(vec![
                        ("start_line".into(), number(start)),
                        ("end_line".into(), number(end)),
                    ])
                })
                .collect(),
        );
        let render = |lines: &[Value], next: Option<(usize, usize)>| {
            json::stringify(&Value::Object(vec![
                ("path".into(), Value::String(path.into())),
                ("edit_tag".into(), Value::String(crate::line_edit::tag(path, &text))),
                ("requested_ranges".into(), requested.clone()),
                ("lines".into(), Value::Array(lines.to_vec())),
                ("total_lines".into(), number(total)),
                ("next_range".into(), next.map_or(Value::Null, |(range, _)| number(range))),
                ("next_line".into(), next.map_or(Value::Null, |(_, line)| number(line))),
                ("truncated".into(), Value::Bool(next.is_some())),
                ("hint".into(), Value::String("Read only needed ranges. next_range is a zero-based requested-range index; next_line is the first omitted selected line. For oversized lines use workspace_read bytes.".into())),
            ]))
        };
        let mut lines = Vec::new();
        // Reserve the largest possible continuation envelope before adding
        // rows, including JSON-escaped paths and range provenance.
        let overhead = render(&lines, Some((ranges.len() - 1, previous_end)))?.len() + 16;
        if overhead > cap {
            return Err(error(
                "path/range provenance exceeds output budget; narrow path or ranges",
            ));
        }
        // Each row costs its bytes plus one comma; the first row has none.
        let budget = cap - overhead + 1;
        let mut rows: Vec<Vec<(usize, Value, usize)>> = vec![Vec::new(); ranges.len()];
        let mut range_index = 0;
        for (index, line) in text.split_inclusive('\n').enumerate() {
            cancelled(cancel)?;
            let line_number = index + 1;
            while range_index < ranges.len() && line_number > ranges[range_index].1 {
                range_index += 1;
            }
            if range_index == ranges.len() {
                break;
            }
            if line_number < ranges[range_index].0 {
                continue;
            }
            // A line over the whole budget can never be shown; stop its range.
            if line.len() >= budget {
                rows[range_index].push((line_number, Value::Null, usize::MAX));
                continue;
            }
            let row = Value::Object(vec![
                ("line".into(), number(line_number)),
                ("text".into(), Value::String(line.into())),
            ]);
            let size = json::stringify(&row)?.len() + 1;
            rows[range_index].push((line_number, row, size));
        }
        let shown = share_budget(&rows, budget);
        let mut next = None;
        for (range, (rows, &count)) in rows.into_iter().zip(&shown).enumerate() {
            if next.is_none() && let Some(&(line, _, _)) = rows.get(count) {
                next = Some((range, line));
            }
            lines.extend(rows.into_iter().take(count).map(|(_, row, _)| row));
        }
        Ok((render(&lines, next)?, text))
    }

    /// Deterministic path/line order within a bounded live view. Offsets count
    /// matching lines, not occurrences. Changes between pages can shift offsets.
    pub fn search_literal(
        &self,
        path: &str,
        literal: &str,
        offset: usize,
        limit: usize,
        cap: usize,
        cancel: &AtomicBool,
    ) -> io::Result<String> {
        self.search_literals(path, &[literal], offset, limit, cap, cancel)
    }

    /// Any-of literal search scans each file once, counting a matching line
    /// once even when several needles match. Literal order does not affect pages.
    pub fn search_literals(
        &self,
        path: &str,
        literals: &[&str],
        offset: usize,
        limit: usize,
        cap: usize,
        cancel: &AtomicBool,
    ) -> io::Result<String> {
        if literals.is_empty()
            || literals.len() > 8
            || literals.iter().any(|literal| {
                literal.is_empty() || literal.len() > 1024 || literal.contains(['\r', '\n'])
            })
            || literals.iter().map(|literal| literal.len()).sum::<usize>() > 4096
            || offset > 1_000_000
            || !(1..=64).contains(&limit)
            || cap < 512
        {
            return Err(error(
                "search needs 1–8 single-line literals ≤1024 bytes each/4096 total, offset ≤1000000 and limit 1–64",
            ));
        }
        let parts = safe_parts(path)?;
        let excluded = |name: &str| {
            ["target", "node_modules", "vendor"]
                .iter()
                .any(|excluded| name.eq_ignore_ascii_case(excluded))
        };
        if parts.iter().any(|name| excluded(name)) {
            return Err(error("search excludes dependency/build paths"));
        }
        let mut pending = vec![(parts.join("/"), 0)];
        let (mut files, mut directories, mut entries, mut scanned_bytes, mut skipped) =
            (0, 0, 0, 0, 0);
        let mut matches = Vec::new();
        let mut matched = 0;
        let mut output_bytes = 384;
        let cap = cap.min(super::MAX_READ_BYTES);
        let mut scan_truncated = false;
        let mut more = false;
        while let Some((path, depth)) = pending.pop() {
            cancelled(cancel)?;
            if entries == SEARCH_ENTRIES {
                scan_truncated = true;
                break;
            }
            entries += 1;
            let stamp = match self.stamp(if path.is_empty() { "." } else { &path }) {
                Ok(Some(stamp)) => stamp,
                _ => {
                    skipped += 1;
                    continue;
                }
            };
            if stamp.directory {
                if directories == SEARCH_DIRECTORIES || depth == SEARCH_DEPTH {
                    scan_truncated = true;
                    continue;
                }
                directories += 1;
                let mut names = match self.list(if path.is_empty() { "." } else { &path }) {
                    Ok(names) => names,
                    Err(_) => {
                        skipped += 1;
                        continue;
                    }
                };
                names.retain(|name| {
                    if name == "[listing limit reached]" {
                        scan_truncated = true;
                        return false;
                    }
                    !excluded(name)
                });
                for name in names.into_iter().rev() {
                    if entries + pending.len() >= SEARCH_ENTRIES {
                        scan_truncated = true;
                        break;
                    }
                    pending.push((
                        if path.is_empty() {
                            name
                        } else {
                            format!("{path}/{name}")
                        },
                        depth + 1,
                    ));
                }
                continue;
            }
            if stamp.bytes > SEARCH_FILE_BYTES as u64 {
                skipped += 1;
                continue;
            }
            if files == SEARCH_FILES || scanned_bytes + stamp.bytes as usize > SEARCH_BYTES {
                scan_truncated = true;
                break;
            }
            files += 1;
            scanned_bytes += stamp.bytes as usize;
            let text = match self.discovery_text(&path, SEARCH_FILE_BYTES, Some(&stamp)) {
                Ok(text) if !text.contains('\0') => text,
                _ => {
                    skipped += 1;
                    continue;
                }
            };
            for (index, line) in text.lines().enumerate() {
                cancelled(cancel)?;
                let Some(match_start) = literals
                    .iter()
                    .filter_map(|literal| line.find(literal))
                    .min()
                else {
                    continue;
                };
                if matched < offset {
                    matched += 1;
                    continue;
                }
                // Show the matched region even when a line has a long prefix.
                let mut snippet_start = match_start.saturating_sub(60);
                while !line.is_char_boundary(snippet_start) {
                    snippet_start -= 1;
                }
                let snippet = clipped(&line[snippet_start..], 240);
                let row = Value::Object(vec![
                    ("path".into(), Value::String(path.clone())),
                    ("line".into(), number(index + 1)),
                    ("text".into(), Value::String(snippet.into())),
                    ("byte_column".into(), number(snippet_start + 1)),
                    (
                        "text_truncated".into(),
                        Value::Bool(snippet_start != 0 || snippet.len() < line.len()),
                    ),
                ]);
                let size = json::stringify(&row)?.len() + 1;
                if matches.len() == limit || output_bytes + size > cap {
                    if matches.is_empty() {
                        return Err(error(
                            "matching path/snippet exceeds output budget; narrow path or use line/byte reads",
                        ));
                    }
                    more = true;
                    break;
                }
                output_bytes += size;
                matches.push(row);
                matched += 1;
            }
            if more {
                break;
            }
        }
        json::stringify(&Value::Object(vec![
            ("matches".into(), Value::Array(matches)),
            ("next_offset".into(), if more { number(matched) } else { Value::Null }),
            ("scan_truncated".into(), Value::Bool(scan_truncated)),
            ("files_scanned".into(), number(files)),
            ("bytes_scanned".into(), number(scanned_bytes)),
            ("skipped".into(), number(skipped)),
            ("hint".into(), Value::String("Copy next_offset for more matches; narrow path when scan_truncated or skipped is nonzero. Live files can change between pages.".into())),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_workspace::Fixture;
    fn sizes(rows: &[&[usize]]) -> Vec<Vec<(usize, Value, usize)>> {
        rows.iter().map(|range| range.iter().enumerate().map(|(i, &size)| (i + 1, Value::Null, size)).collect()).collect()
    }
    #[test]
    fn ranges_share_the_byte_budget_smallest_first() {
        // Filled in order, the first range took all 1000 bytes: [10, 0, 0].
        // C (50) takes 50 of 333, B (300) takes 300 of 950/2 = 475, A gets 650.
        assert_eq!(share_budget(&sizes(&[&[100; 10], &[100; 3], &[50]]), 1000), [6, 3, 1]);
        // One range: the same rows as one budget filled in order.
        assert_eq!(share_budget(&sizes(&[&[100; 10]]), 999), [9]);
        // Ranges that fit together are all shown.
        assert_eq!(share_budget(&sizes(&[&[100; 4], &[100; 3], &[50]]), 750), [4, 3, 1]);
        // Equal needs split evenly; integer shares round down, slack moves on.
        assert_eq!(share_budget(&sizes(&[&[100; 5], &[100; 5]]), 701), [3, 4]);
        // A line over the budget stops its range only.
        assert_eq!(share_budget(&sizes(&[&[10, usize::MAX, 10], &[10, 10]]), 100), [1, 2]);
    }
    #[test]
    fn a_long_first_range_cannot_starve_a_later_range() {
        let f = Fixture::new();
        let text: String = (1..=100).map(|n| format!("{n:03} {}\n", "x".repeat(35))).collect();
        f.write("long.rs", &text);
        let w = f.workspace();
        let cancel = AtomicBool::new(false);
        let output = w.read_line_ranges("long.rs", &[(1, 80), (90, 95)], 4096, &cancel).unwrap();
        assert!(output.len() <= 4096, "{}", output.len());
        let value = json::parse(&output).unwrap();
        let shown: Vec<usize> = value.get("lines").and_then(Value::as_array).unwrap().iter()
            .map(|row| match row.get("line") { Some(Value::Number(n)) => n.parse().unwrap(), _ => panic!() }).collect();
        assert_eq!(shown[shown.len() - 6..], [90, 91, 92, 93, 94, 95]);
        let first = shown.len() - 6;
        assert_eq!(shown[..first], (1..=first).collect::<Vec<_>>());
        assert_eq!(value.get("next_range"), Some(&number(0)));
        assert_eq!(value.get("next_line"), Some(&number(first + 1)));
        assert_eq!(value.get("truncated"), Some(&Value::Bool(true)));
        // One range keeps the fill-in-order result: the same first rows.
        let alone = json::parse(&w.read_line_ranges("long.rs", &[(1, 80)], 4096, &cancel).unwrap()).unwrap();
        assert!(alone.get("lines").and_then(Value::as_array).unwrap().len() > first);
    }
}
