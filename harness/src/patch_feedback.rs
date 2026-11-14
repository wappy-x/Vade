//! Bounded syntax guidance; it never prepares, translates or applies an edit.
use crate::edit::{BEGIN, END, FILE_HEADERS};

/// A syntax example, never a translated proposal or an executable fallback.
/// Keep the caller's path/content; ambiguous dialect changes need a new call.
pub(crate) fn refusal_guidance(patch: &str, error: &str) -> String {
    if !(error.contains("patch line ") || error.contains("patch must start") || error.contains("mixes '+'")) {
        return String::new();
    }
    let lines: Vec<&str> = patch.strip_suffix('\n').unwrap_or(patch).split('\n').collect();
    let inserted_begin = lines.first().is_some_and(|line| {
        FILE_HEADERS.iter().any(|header| line.starts_with(header))
    }) && !lines.contains(&BEGIN);
    let framed_line = error
        .rsplit("patch line ")
        .next()
        .and_then(|text| text.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|number| number.parse::<usize>().ok());
    let wrong = framed_line
        .unwrap_or(1)
        .saturating_sub(usize::from(inserted_begin))
        .max(1);
    let found: String = lines.get(wrong - 1).unwrap_or(&"").chars().take(160).collect();
    let canonical = lines.iter().enumerate().find_map(|(at, line)| {
        FILE_HEADERS.iter().enumerate().find_map(|(kind, header)| {
            line.strip_prefix(header).map(|path| (at, kind, path))
        })
    });
    let inferred = lines.iter().enumerate().find_map(|(at, line)| {
        line.strip_prefix("+++ ")
            .filter(|path| !path.is_empty())
            .map(|path| {
                let add = lines.iter().any(|line| matches!(*line, "--- " | "--- /dev/null"));
                (at, if add { 0 } else { 1 }, path.strip_prefix("b/").unwrap_or(path))
            })
    });
    let example = canonical
        .or(inferred)
        .filter(|(_, _, path)| path.len() <= 160)
        .map(|(at, kind, path)| {
            let mut body = Vec::new();
            let mut started = false;
            for &line in &lines[at + 1..] {
                if line.starts_with("*** ") {
                    break;
                }
                if line.starts_with("@@") {
                    if started {
                        break;
                    }
                    continue;
                }
                if line.starts_with("--- ") || line.starts_with("+++ ") || line == "+++" {
                    continue;
                }
                started = true;
                if kind == 0 {
                    body.push(format!("+{}", line.strip_prefix('+').unwrap_or(line)));
                } else if line.is_empty() {
                    body.push(" ".into());
                } else if line.starts_with([' ', '-', '+']) {
                    body.push(line.to_string());
                }
            }
            if kind == 1 && !body.iter().any(|line| line.starts_with(['+', '-'])) {
                body = vec!["-exact old line from workspace_read_lines".into(), "+replacement line".into()];
            }
            let change = body
                .iter()
                .position(|line| line.starts_with('+'))
                .unwrap_or(0);
            let start = if kind == 0 { 0 } else { change.saturating_sub(2) };
            let mut sample = format!("{BEGIN}\n{}{path}\n", FILE_HEADERS[kind]);
            if kind == 1 {
                sample.push_str("@@\n");
            }
            if kind != 2 {
                let mut bytes = 0;
                for line in body.iter().skip(start).take(5) {
                    if bytes + line.len() > 384 {
                        break;
                    }
                    sample.push_str(line);
                    sample.push('\n');
                    bytes += line.len();
                }
            }
            sample.push_str(END);
            format!("\nMinimal syntax example (illustrative; substitute exact old text and your new text): {{\"patch\":{}}}", crate::json::quote(&sample))
        })
        .unwrap_or_default();
    format!("\nFirst wrong line {wrong} (up to 160 characters): {}.\nExpected patch string: *** Begin Patch, then Add (+content), Update (@@ then exact old context/-old/+new), or Delete; then *** End Patch. Do not send raw replacement text or ---/+++ headers.{example}", crate::json::quote(&found))
}
