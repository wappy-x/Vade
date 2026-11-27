use super::*;
use crate::terminal::render::{plain, width};

fn samples() -> Vec<Prompt> {
    let command = finish(&action("Vade wants to run a command", "rm -rf build", Risk::Red, "can delete files", Some("this project")),
        "This command deletes files.\nYou cannot get deleted files back.",
        &choices("do not run it", Some("this exact command"), Some("in this project (this exact command)"), Some("details")));
    let mut files = finish(&action("Vade wants to change 2 files", "", Risk::Red, "can overwrite or delete work", Some("this project")),
        "A wrong change can overwrite or delete your work.", &choices("leave the files unchanged", None, None, Some("change")));
    files.files = vec!["src/main.rs".into(), "README.md".into()];
    let search = finish(&action("Vade wants to search the web", "\"ownership rules\"", Risk::Red, "sends search words", None),
        "The search website receives these words.\nA harmful file or web page can trick Vade into searching for your private text.",
        &choices("refuse without sending anything", Some("allow web searches"), Some("for web searches"), None));
    let page = finish(&action("Vade wants to read this web page", "https://example.org/guide", Risk::Red, "sends the full address", None),
        "The website receives the full address.\nA harmful file or web page can trick Vade into putting your private text in the address.",
        &choices("refuse without sending anything", Some("allow requests to https://example.org"), Some("for this website (https://example.org)"), None));
    let redirect = finish(&action("https://example.org sends Vade to a different website", "https://other.example.org/next", Risk::Red, "sends the full address", None),
        "You approved https://example.org. You did not approve the new website.\nThe new website receives the full address. The address can contain your private text.",
        &choices("stop the redirect without contacting that site", Some("allow requests to https://other.example.org"), Some("for this website (https://other.example.org)"), None));
    let http = finish(&action("https://example.org sends Vade to an address that is not encrypted (http, not https).", "http://example.org/next", Risk::Red, "sends the full address", None),
        "Other people on the network can read the address and the reply.",
        &choices("stop the redirect without contacting that site", Some("allow requests to http://example.org"), Some("for this website (http://example.org)"), None));
    let save = finish(&action("Vade wants to save this note for this project", "\"build\": \"Run unit tests first.\"", Risk::Amber, "affects future chats", Some("this project")),
        "Vade reads saved notes in every future chat in this project and obeys them.\nA harmful file or web page can trick Vade into saving a bad note.",
        &choices("leave project notes unchanged", Some("allow note changes in this project"), Some("in this project (note changes)"), None));
    let delete = finish(&action("Vade wants to delete this note from this project", "\"build\"", Risk::Red, "removes a note", Some("this project")),
        "Future chats in this project will not have this note.",
        &choices("keep project notes unchanged", Some("allow note changes in this project"), Some("in this project (note changes)"), None));
    let outside = finish(&action("Vade wants to use an outside tool", "\"create_issue\" on \"github\"", Risk::Red, "can send data or change things", None),
        "This tool runs outside Vade. It can send your data to that server.\nIt can change things on that server.", &choices("do not run this tool", None, None, Some("data")));
    let secret = finish(&action("Vade wants to read this file from this project", "\"token.txt\"", Risk::Red, "private text enters chat", Some("this project")),
        crate::tools::secret_path_reason("token.txt").unwrap(), &choices("leave it unread", None, None, None));
    vec![command, files, search, page, redirect, http, save, delete, outside, secret]
}

#[test]
fn command_64() {
    let expected = r#"Vade wants to run a command                     can delete files
■  rm -rf build                                     this project
      This command deletes files.
      You cannot get deleted files back.
----------------------------------------------------------------
Your turn.  1  Yes, one time
            2  No, do not run it
            3  Yes, until you close Vade (this exact command)
            4  Yes, always in this project (this exact command)
            5  Show the exact details"#;
    assert_eq!(samples()[0].render(64, false), expected);
}

#[test]
fn files_64() {
    let expected = r#"Vade wants to change 2 files        can overwrite or delete work
■  src/main.rs, README.md                           this project
      A wrong change can overwrite or delete your work.
----------------------------------------------------------------
Your turn.  1  Yes, one time
            2  No, leave the files unchanged
            5  Show the exact change"#;
    assert_eq!(samples()[1].render(64, false), expected);
}

#[test]
fn search_64() {
    let expected = r#"Vade wants to search the web                  sends search words
■  "ownership rules"
      The search website receives these words.
      A harmful file or web page can trick Vade into searching
      for your private text.
----------------------------------------------------------------
Your turn.  1  Yes, one time
            2  No, refuse without sending anything
            3  Yes, until you close Vade (allow web searches)
            4  Yes, always for web searches"#;
    assert_eq!(samples()[2].render(64, false), expected);
}

#[test]
fn page_64() {
    let expected = r#"Vade wants to read this web page          sends the full address
■  https://example.org/guide
      The website receives the full address.
      A harmful file or web page can trick Vade into putting
      your private text in the address.
----------------------------------------------------------------
Your turn.  1  Yes, one time
            2  No, refuse without sending anything
            3  Yes, until you close Vade (allow requests to
               https://example.org)
            4  Yes, always for this website
               (https://example.org)"#;
    assert_eq!(samples()[3].render(64, false), expected);
}

#[test]
fn redirect_64() {
    let expected = r#"https://example.org sends Vade to a different website
                                          sends the full address
■  https://other.example.org/next
      You approved https://example.org.
      You did not approve the new website.
      The new website receives the full address.
      The address can contain your private text.
----------------------------------------------------------------
Your turn.  1  Yes, one time
            2  No, stop the redirect without contacting that
               site
            3  Yes, until you close Vade (allow requests to
               https://other.example.org)
            4  Yes, always for this website
               (https://other.example.org)"#;
    assert_eq!(samples()[4].render(64, false), expected);
}

#[test]
fn unencrypted_64() {
    let expected = r#"https://example.org sends Vade to an address that is not
encrypted (http, not https).
                                          sends the full address
■  http://example.org/next
      Other people on the network can read the address and the
      reply.
----------------------------------------------------------------
Your turn.  1  Yes, one time
            2  No, stop the redirect without contacting that
               site
            3  Yes, until you close Vade (allow requests to
               http://example.org)
            4  Yes, always for this website (http://example.org)"#;
    assert_eq!(samples()[5].render(64, false), expected);
}

#[test]
fn save_64() {
    let expected = r#"Vade wants to save this note for this project
                                            affects future chats
▲  "build": "Run unit tests first."                 this project
      Vade reads saved notes in every future chat in this
      project and obeys them.
      A harmful file or web page can trick Vade into saving a
      bad note.
----------------------------------------------------------------
Your turn.  1  Yes, one time
            2  No, leave project notes unchanged
            3  Yes, until you close Vade (allow note changes in
               this project)
            4  Yes, always in this project (note changes)"#;
    assert_eq!(samples()[6].render(64, false), expected);
}

#[test]
fn delete_64() {
    let expected = r#"Vade wants to delete this note from this project  removes a note
■  "build"                                          this project
      Future chats in this project will not have this note.
----------------------------------------------------------------
Your turn.  1  Yes, one time
            2  No, keep project notes unchanged
            3  Yes, until you close Vade (allow note changes in
               this project)
            4  Yes, always in this project (note changes)"#;
    assert_eq!(samples()[7].render(64, false), expected);
}

#[test]
fn outside_64() {
    let expected = r#"Vade wants to use an outside tool
                                  can send data or change things
■  "create_issue" on "github"
      This tool runs outside Vade.
      It can send your data to that server.
      It can change things on that server.
----------------------------------------------------------------
Your turn.  1  Yes, one time
            2  No, do not run this tool
            5  Show the exact data"#;
    assert_eq!(samples()[8].render(64, false), expected);
}

#[test]
fn secret_64() {
    let expected = r#"Vade wants to read this file from this project
                                        private text enters chat
■  "token.txt"                                      this project
      this filename commonly holds a key, token or credential
----------------------------------------------------------------
Your turn.  1  Yes, one time
            2  No, leave it unread"#;
    assert_eq!(samples()[9].render(64, false), expected);
}

#[test]
fn command_40() {
    let expected = r#"Vade wants to run a command
                        can delete files
■  rm -rf build             this project
      This command deletes files.
      You cannot get deleted files back.
----------------------------------------
Your turn.  1  Yes, one time
            2  No, do not run it
            3  Yes, until you close Vade
               (this exact command)
            4  Yes, always in this
               project (this exact
               command)
            5  Show the exact details"#;
    assert_eq!(samples()[0].render(40, false), expected);
}

#[test]
fn files_40() {
    let expected = r#"Vade wants to change 2 files
            can overwrite or delete work
■  src/main.rs, README.md   this project
      A wrong change can overwrite or
      delete your work.
----------------------------------------
Your turn.  1  Yes, one time
            2  No, leave the files
               unchanged
            5  Show the exact change"#;
    assert_eq!(samples()[1].render(40, false), expected);
}

#[test]
fn search_40() {
    let expected = r#"Vade wants to search the web
                      sends search words
■  "ownership rules"
      The search website receives these
      words.
      A harmful file or web page can
      trick Vade into searching for your
      private text.
----------------------------------------
Your turn.  1  Yes, one time
            2  No, refuse without
               sending anything
            3  Yes, until you close Vade
               (allow web searches)
            4  Yes, always for web
               searches"#;
    assert_eq!(samples()[2].render(40, false), expected);
}

pub(crate) fn command_block() -> String { samples()[0].render(64, false) }

#[test]
fn every_kind_preserves_plain_text_and_terminal_cells_with_color() {
    for prompt in samples() {
        for columns in [40, 64, 80] {
            let text = prompt.render(columns, false);
            let colored = prompt.render(columns, true);
            assert_eq!(plain(&colored), text);
            assert!(text.lines().all(|line| width(line) <= columns.min(64)), "{text}");
            assert!(colored.contains("\x1b[1;37m") && colored.contains("\x1b[1;38;2;27;143;196m"));
            let dim: Vec<_> = colored.lines().filter(|line| line.starts_with("\x1b[2m")).collect();
            assert_eq!(dim.len(), crate::terminal::render::wrap(&prompt.sentences[0], columns.min(64) - 6).len());
        }
    }
}

#[test]
fn long_command_is_whole_and_risks_wrap_with_their_indent() {
    let mut prompt = samples().remove(0);
    prompt.object = "x".repeat(300);
    prompt.sentences = vec!["A long first risk sentence that must wrap onto several lines so all its words stay visible.".into(), "Another sentence stays normal.".into()];
    for columns in [40, 64] {
        let text = prompt.render(columns, false);
        let object = object_rows(&text).filter(|line| !line.contains("this project")).map(|line| &line[5..]).collect::<String>();
        assert_eq!(object, prompt.object);
        assert!(text.lines().all(|line| width(line) <= columns));
        let risk_rows: Vec<_> = text.lines().filter(|line| line.starts_with("      ") && !line.trim().is_empty() && !line.trim_start().starts_with(|c: char| c.is_ascii_digit())).collect();
        assert!(risk_rows.len() >= 3);
        assert!(text.contains(&format!("\n│  {}this project", " ".repeat(columns - 15))));
    }
}

#[test]
fn twelve_files_use_four_lines_and_count_the_rest() {
    let mut prompt = samples().remove(1);
    prompt.files = (1..=12).map(|n| format!("source/long-file-{n}.rs")).collect();
    let text = prompt.render(40, false);
    let target: Vec<_> = object_rows(&text).collect();
    // Rows break between names: no name is cut in two when it fits on a row.
    assert_eq!(target, ["■  source/long-file-1.rs,   this project", "│  source/long-file-2.rs,", "│  source/long-file-3.rs,", "│  source/long-file-4.rs, and 8 more"]);
    assert_eq!(text.matches("this project").count(), 1);
}

#[test]
fn wide_unicode_and_a_tag_that_does_not_fit_count_cells() {
    let mut prompt = samples().remove(0);
    prompt.what = "界".repeat(25);
    prompt.tag = "long warning tag".into();
    prompt.object = "界".repeat(35);
    let text = prompt.render(64, false);
    assert_eq!(text.lines().nth(1).unwrap(), format!("{}long warning tag", " ".repeat(48)));
    assert!(text.lines().all(|line| width(line) <= 64));
}

#[test]
fn distinct_argv_never_share_the_same_display_including_visible_escapes() {
    let words = ["", "abc", "a b", "a'b", "a\"b", "a\\\"b", "a\\b", "a\nb", "a\\nb", "a\tb", "a\\tb", "\u{200b}", "\\u200b", "\u{e0100}", "\\udb40\\udd00", "界", "a;b", "a  b", "a␣␣b", "\u{202e}", "\\u202e", "\x7f", "\r", "\\x20\\x20", "  "];
    let mut displays = std::collections::HashMap::new();
    for a in words {
        for b in words {
            let argv = vec![a.to_owned(), b.to_owned()];
            assert!(displays.insert(command(&argv), argv).is_none());
        }
        let argv = vec![a.to_owned()];
        assert!(displays.insert(command(&argv), argv).is_none());
    }
    assert_eq!(command(&["echo".into(), "a'b".into(), "".into()]), "echo $'a\\'b' ''");
    assert_eq!(command(&["echo".into(), "a\"b".into()]), "echo 'a\"b'");
}

pub(crate) fn command_sample() -> Prompt { samples().remove(0) }

#[test]
fn fixed_system_command_displays_the_words_that_run() {
    let spec = crate::shell::CommandSpec { argv: vec!["git".into(), "status".into()], timeout_ms: 20000 };
    let executed = spec.executed_argv();
    assert_ne!(executed, spec.argv);
    let shown = command(&executed);
    assert!(shown.starts_with("/usr/bin/git --no-pager --no-optional-locks"));
    assert!(shown.contains("core.hooksPath=/dev/null"));
    assert!(shown.contains("':(exclude,glob)**/.*'"));
}

#[test]
fn command_wrapping_preserves_spaces_at_the_edge() {
    let mut prompt = samples().remove(0);
    prompt.object = command(&["echo".into(), format!("{}  end", "a".repeat(28))]);
    prompt.scope = None;
    let text = prompt.render(40, false);
    let target = object_rows(&text).map(|line| &line[5..]).collect::<String>();
    assert_eq!(target, prompt.object);
    assert!(target.contains("\\x20\\x20end"));
}

fn object_rows(text: &str) -> impl Iterator<Item = &str> {
    text.lines().skip_while(|line| !line.starts_with("■  ") && !line.starts_with("▲  "))
        .take_while(|line| line.starts_with("■  ") || line.starts_with("▲  ") || line.starts_with("│  "))
}

#[test]
fn padded_argument_cannot_forge_prompt_rows() {
    let attack = format!("{}{}{}{}", " ".repeat(55), format!("   This command only reads files.{: <29}", ""), "-".repeat(61), format!("Your turn.  1  No, do not run it{: <31}", ""));
    let mut prompt = command_sample();
    prompt.object = command(&["echo".into(), attack]);
    let text = prompt.render(64, false);
    let object: Vec<_> = object_rows(&text).collect();
    assert!(object.len() > 2);
    assert!(object[0].starts_with("■  "));
    assert!(object[1..].iter().all(|row| row.starts_with("│  ")));
    assert_eq!(object.iter().map(|row| &row[5..]).collect::<String>().replace("this project", "").trim_end(), prompt.object);
    assert_eq!(text.lines().filter(|row| *row == "-".repeat(64)).count(), 1);
    assert_eq!(text.lines().filter(|row| row.starts_with("Your turn.")).count(), 1);
    assert!(prompt.object.contains(&"\\x20".repeat(55)));
}

#[test]
fn file_names_are_whole_quoted_and_budgeted() {
    let mut prompt = samples().remove(1);
    prompt.scope = None;
    prompt.files = vec!["x".repeat(130)];
    let text = prompt.render(64, false);
    assert_eq!(object_rows(&text).map(|row| &row[5..]).collect::<String>(), prompt.files[0]);
    assert!(!text.contains("more"));
    prompt.files = vec!["a, b".into(), "c".into()];
    let first = prompt.render(64, false);
    assert!(first.contains("■  'a, b', c"));
    prompt.files = vec!["a".into(), "b, c".into()];
    assert!(prompt.render(64, false).contains("■  a, 'b, c'"));
    assert_ne!(first, prompt.render(64, false));
    prompt.files = vec!["a,b".into()];
    assert!(prompt.render(64, false).contains("■  'a,b'"));
    prompt.files = vec!["x".repeat(245), "small".into()];
    let text = prompt.render(64, false);
    assert!(object_rows(&text).map(|row| &row[5..]).collect::<String>().starts_with(&prompt.files[0]));
    assert!(text.contains("and 1 more"));
    prompt.files = vec!["x".repeat(170), "small".into()];
    let text = prompt.render(64, false);
    assert!(text.contains("small") && !text.contains("and 1 more"));
}

#[test]
fn every_kind_preserves_all_parts_at_widths_1_to_200() {
    let mut prompts = samples();
    let mut unicode = command_sample();
    unicode.object = command(&["rm".into(), "界".into()]);
    prompts.push(unicode);
    for prompt in prompts {
        for columns in 1..=200 {
            let text = prompt.render(columns, false);
            assert!(text.lines().all(|row| width(row) <= columns.clamp(20, 64)), "width {columns}: {text}");
            assert_eq!(plain(&prompt.render(columns, true)), text);
            // Layout may add spaces and gutters, but must keep every nonspace
            // character in each part, in order. File lists have their own budget test.
            for part in std::iter::once(&prompt.what).chain(std::iter::once(&prompt.tag))
                .chain(prompt.scope.iter()).chain(prompt.sentences.iter()).chain(std::iter::once(&prompt.choices)) {
                let mut output = text.chars();
                for c in part.chars().filter(|c| !c.is_whitespace()) {
                    assert!(output.any(|shown| shown == c), "missing {c:?} from {part:?} at {columns}: {text}");
                }
            }
            let object = object_rows(&text).map(|row| &row[5..]).collect::<String>();
            let source = if prompt.files.is_empty() { prompt.object.clone() } else { prompt.files.join(", ") };
            let mut chars = object.chars();
            for c in source.chars().filter(|c| !c.is_whitespace()) { assert!(chars.any(|shown| shown == c)); }
        }
    }
}

#[test]
fn command_words_round_trip_through_bash() {
    let hard = ["a\\b", "a'b", "a\"b", "a\tb", "a\nb", "", "-n", "界", "café", "a\rb", "a\x01b", "a\x7fb", "a;b", "$(exit 9)", "`exit 9`", "a b", "a*b", "$HOME", "a|b", "a&b", "a(b)", "a#b", "~", "a\\  b", "\u{202e}", "\u{e0100}", "a␣b", "a  b", "a\u{0085}b", "\u{202e}$'\\\n", "  ", " a ", "a   b  c"];
    let mut tested = 0;
    for word in hard.iter().map(|word| word.to_string()).chain(std::iter::once("x".repeat(300))) {
        let argv = vec!["/usr/bin/printf".into(), "%s".into(), word.clone(), "tail".into()];
        let shown = command(&argv[1..]);
        // Every case must round trip, in the bash 3.2 that macOS ships too: the shown text uses no Unicode escapes.
        assert!(!shown.contains("\\u") && !shown.contains("\\U"), "{shown}");
        let output = std::process::Command::new("/bin/bash").arg("-c").arg(format!("printf '%s\\0' {shown}")).env("LC_ALL", "en_US.UTF-8").output().unwrap();
        assert!(output.status.success(), "{shown}: {:?}", output.stderr);
        let expected = argv[1..].iter().flat_map(|word| word.bytes().chain(std::iter::once(0))).collect::<Vec<_>>();
        assert_eq!(output.stdout, expected, "{shown}");
        tested += 1;
    }
    assert!(tested >= 20, "round-tripped {tested} lists");
    assert_eq!(command(&["\u{202e}".into(), "\u{e0100}".into()]), "$'\\xe2\\x80\\xae' $'\\xf3\\xa0\\x84\\x80'");
    assert_eq!(command(&["a\\  b".into()]), "$'a\\\\\\x20\\x20b'");
    assert_eq!(command(&["a  b".into()]), "$'a\\x20\\x20b'");
    assert_eq!(command(&["a␣b".into()]), "'a␣b'");
}

#[test]
fn long_command_150_and_twelve_files_64() {
    let mut prompt = command_sample();
    prompt.object = "x".repeat(150);
    let text = prompt.render(64, false);
    assert!(text.contains(&format!("■  {}\n│  {}\n│  {}", "x".repeat(61), "x".repeat(61), "x".repeat(28))));
    assert_eq!(text, include_str!("../tests/goldens/approval-command-150.txt").trim_end_matches('\n'));
    println!("LONG COMMAND (150 characters)\n{text}");
    let mut prompt = samples().remove(1);
    prompt.what = "Vade wants to change 12 files".into();
    prompt.files = (1..=12).map(|n| format!("source/long-file-{n}.rs")).collect();
    let text = prompt.render(64, false);
    assert!(text.contains("and 4 more"));
    assert_eq!(text, include_str!("../tests/goldens/approval-files-12.txt").trim_end_matches('\n'));
    println!("TWELVE FILES\n{text}");
}
