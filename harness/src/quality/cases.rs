//! The fixed quality prompt set, version 1. Every check is an exact check on
//! the answer text, on parsed JSON values, or on unit tests run against the
//! returned code. No model judges another model's answer here.
use super::*;

/// One exact check on a plain-text or JSON answer.
#[derive(Clone, Copy, Debug)]
pub enum Rule {
    /// The trimmed answer contains this exact text.
    Contains(&'static str),
    /// The trimmed answer does not contain this exact text (case-sensitive).
    NotContains(&'static str),
    /// The trimmed answer starts with this exact text.
    StartsWith(&'static str),
    /// The trimmed answer ends with this exact text.
    EndsWith(&'static str),
    /// Word count (whitespace split) inside this inclusive range.
    Words(usize, usize),
    /// Sentence count, counted by '.', '!' or '?' terminators.
    Sentences(usize),
    /// Non-empty line count equals this number.
    Lines(usize),
    /// Every non-empty line starts with this exact text.
    EveryLineStartsWith(&'static str),
    /// The trimmed answer equals this exact text.
    EqualsExact(&'static str),
    /// Non-empty line `index` (zero-based) contains this exact text.
    LineContains(usize, &'static str),
    /// Every non-empty line holds at most this many words after an
    /// optional leading `- ` list marker.
    EveryLineWordsAtMost(usize),
    /// A top-level JSON field equals this exact JSON literal.
    JsonField(&'static str, &'static str),
    /// A top-level JSON field is an array of this length.
    JsonArrayLen(&'static str, usize),
    /// Element `index` of a top-level JSON array field, at `key`, equals this
    /// exact JSON literal.
    JsonItemField(&'static str, usize, &'static str, &'static str),
}

impl Rule {
    pub fn name(self) -> String {
        match self {
            Self::Contains(text) => format!("contains {text:?}"),
            Self::NotContains(text) => format!("does not contain {text:?}"),
            Self::StartsWith(text) => format!("starts with {text:?}"),
            Self::EndsWith(text) => format!("ends with {text:?}"),
            Self::Words(low, high) => format!("word count in {low}..={high}"),
            Self::Sentences(count) => format!("exactly {count} sentences"),
            Self::Lines(count) => format!("exactly {count} non-empty lines"),
            Self::EveryLineStartsWith(text) => format!("every line starts with {text:?}"),
            Self::EqualsExact(_) => "answer equals the exact required text".into(),
            Self::LineContains(index, text) => format!("line {} contains {text:?}", index + 1),
            Self::EveryLineWordsAtMost(max) => format!("every line has at most {max} words"),
            Self::JsonField(key, _) => format!("json field {key} exact"),
            Self::JsonArrayLen(key, len) => format!("json array {key} has {len} items"),
            Self::JsonItemField(key, index, field, _) => {
                format!("json {key}[{index}].{field} exact")
            }
        }
    }
}

/// How one case is scored.
#[derive(Clone, Copy, Debug)]
pub enum Scoring {
    /// Exact rules on the answer text.
    Rules(&'static [Rule]),
    /// The answer must be one JSON document; rules read its fields. A parse
    /// failure fails every rule and is reported as its own check.
    Json(&'static [Rule]),
    /// The answer must hold one fenced Rust block with `signature` and its
    /// own tests. The scorer appends `hidden_tests` and runs `rustc --test`.
    Code {
        signature: &'static str,
        hidden_tests: &'static str,
    },
}

/// One fixed prompt and its scoring.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub id: &'static str,
    pub class: Class,
    pub prompt: &'static str,
    pub max_tokens: usize,
    pub scoring: Scoring,
}

/// The prompt set version printed on every card and receipt.
pub const SET_VERSION: &str = "quality-set-1";

/// Appended to every code prompt when the request is built.
pub const CODE_TAIL: &str = " Return only one fenced rust code block with the function and a #[cfg(test)] module that holds at least two unit tests. Return no prose.";

pub const CASES: &[Case] = &[
    Case {
        id: "code-clamp",
        class: Class::Code,
        prompt: "Write a Rust function with this exact signature: pub fn clamp_i32(value: i32, low: i32, high: i32) -> i32. It returns value limited to the range low..=high. If low is greater than high, it returns low.",
        max_tokens: 700,
        scoring: Scoring::Code {
            signature: "pub fn clamp_i32(value: i32, low: i32, high: i32) -> i32",
            hidden_tests: r#"
#[cfg(test)] mod quality_hidden { use super::*;
#[test] fn inside() { assert_eq!(clamp_i32(5, 0, 10), 5); }
#[test] fn below() { assert_eq!(clamp_i32(-3, 0, 10), 0); }
#[test] fn above() { assert_eq!(clamp_i32(42, 0, 10), 10); }
#[test] fn edges() { assert_eq!(clamp_i32(0, 0, 10), 0); assert_eq!(clamp_i32(10, 0, 10), 10); }
#[test] fn swapped() { assert_eq!(clamp_i32(5, 10, 0), 10); }
}
"#,
        },
    },
    Case {
        id: "code-dedupe",
        class: Class::Code,
        prompt: "Write a Rust function with this exact signature: pub fn dedupe_sorted(values: &[i32]) -> Vec<i32>. The input is sorted in ascending order. Return each distinct value once, in ascending order. An empty input returns an empty vector.",
        max_tokens: 700,
        scoring: Scoring::Code {
            signature: "pub fn dedupe_sorted(values: &[i32]) -> Vec<i32>",
            hidden_tests: r#"
#[cfg(test)] mod quality_hidden { use super::*;
#[test] fn empty() { assert_eq!(dedupe_sorted(&[]), Vec::<i32>::new()); }
#[test] fn single() { assert_eq!(dedupe_sorted(&[7]), vec![7]); }
#[test] fn runs() { assert_eq!(dedupe_sorted(&[1, 1, 2, 2, 2, 3]), vec![1, 2, 3]); }
#[test] fn distinct() { assert_eq!(dedupe_sorted(&[-2, 0, 4, 9]), vec![-2, 0, 4, 9]); }
#[test] fn all_same() { assert_eq!(dedupe_sorted(&[3, 3, 3, 3]), vec![3]); }
}
"#,
        },
    },
    Case {
        id: "code-parse-list",
        class: Class::Code,
        prompt: "Write a Rust function with this exact signature: pub fn parse_u32_list(text: &str) -> Result<Vec<u32>, String>. It parses a comma-separated list of unsigned integers. Spaces around a number are allowed. An empty or all-space input returns an empty vector. Any part that is not a valid u32 makes the function return an error.",
        max_tokens: 700,
        scoring: Scoring::Code {
            signature: "pub fn parse_u32_list(text: &str) -> Result<Vec<u32>, String>",
            hidden_tests: r#"
#[cfg(test)] mod quality_hidden { use super::*;
#[test] fn basic() { assert_eq!(parse_u32_list("1,2,3").unwrap(), vec![1, 2, 3]); }
#[test] fn spaces() { assert_eq!(parse_u32_list(" 4 , 50 ,600 ").unwrap(), vec![4, 50, 600]); }
#[test] fn empty() { assert_eq!(parse_u32_list("").unwrap(), Vec::<u32>::new()); assert_eq!(parse_u32_list("   ").unwrap(), Vec::<u32>::new()); }
#[test] fn bad_word() { assert!(parse_u32_list("1,two,3").is_err()); }
#[test] fn negative() { assert!(parse_u32_list("-1").is_err()); }
}
"#,
        },
    },
    Case {
        id: "code-word-count",
        class: Class::Code,
        prompt: "Write a Rust function with this exact signature: pub fn count_word(text: &str, word: &str) -> usize. It counts whole-word occurrences of word in text. Words are maximal runs of ASCII letters and digits. The comparison ignores ASCII case. An empty word counts zero.",
        max_tokens: 700,
        scoring: Scoring::Code {
            signature: "pub fn count_word(text: &str, word: &str) -> usize",
            hidden_tests: r#"
#[cfg(test)] mod quality_hidden { use super::*;
#[test] fn basic() { assert_eq!(count_word("the cat sat", "cat"), 1); }
#[test] fn case_and_repeat() { assert_eq!(count_word("Cat cat CAT", "cat"), 3); }
#[test] fn whole_word_only() { assert_eq!(count_word("cats scatter cat", "cat"), 1); }
#[test] fn punctuation() { assert_eq!(count_word("cat, cat. (cat)", "cat"), 3); }
#[test] fn empty_word() { assert_eq!(count_word("some text", ""), 0); }
}
"#,
        },
    },
    Case {
        id: "json-order",
        class: Class::Structured,
        prompt: "Read this order note: Order A-1042 from Dana Ruiz holds 3 items. The total is 4599 cents. The priority is high. The order is not paid. Return only one JSON object with exactly these fields: order_id (string), items (integer), total_cents (integer), priority (string), paid (boolean). Return no other text.",
        max_tokens: 300,
        scoring: Scoring::Json(&[
            Rule::JsonField("order_id", "\"A-1042\""),
            Rule::JsonField("items", "3"),
            Rule::JsonField("total_cents", "4599"),
            Rule::JsonField("priority", "\"high\""),
            Rule::JsonField("paid", "false"),
        ]),
    },
    Case {
        id: "json-person",
        class: Class::Structured,
        prompt: "Read this note: Marta Klein was born in 1988. She lives in Porto. Her hobbies are rowing, chess and baking, in that order. Return only one JSON object with exactly these fields: name (string), birth_year (integer), city (string), hobbies (array of strings, in the order given). Return no other text.",
        max_tokens: 300,
        scoring: Scoring::Json(&[
            Rule::JsonField("name", "\"Marta Klein\""),
            Rule::JsonField("birth_year", "1988"),
            Rule::JsonField("city", "\"Porto\""),
            Rule::JsonArrayLen("hobbies", 3),
            Rule::JsonField("hobbies", "[\"rowing\",\"chess\",\"baking\"]"),
        ]),
    },
    Case {
        id: "json-classify",
        class: Class::Structured,
        prompt: "Classify each support ticket. Use only these labels: billing, bug, question. Ticket 1: My invoice shows two charges for March. Ticket 2: The export button stops with an error every time. Ticket 3: How do I change the report language? Ticket 4: The app closes when I open a saved file. Return only one JSON object with one field tickets: an array of 4 objects in ticket order, each with id (integer) and label (string). Return no other text.",
        max_tokens: 300,
        scoring: Scoring::Json(&[
            Rule::JsonArrayLen("tickets", 4),
            Rule::JsonItemField("tickets", 0, "id", "1"),
            Rule::JsonItemField("tickets", 0, "label", "\"billing\""),
            Rule::JsonItemField("tickets", 1, "id", "2"),
            Rule::JsonItemField("tickets", 1, "label", "\"bug\""),
            Rule::JsonItemField("tickets", 2, "id", "3"),
            Rule::JsonItemField("tickets", 2, "label", "\"question\""),
            Rule::JsonItemField("tickets", 3, "id", "4"),
            Rule::JsonItemField("tickets", 3, "label", "\"bug\""),
        ]),
    },
    Case {
        id: "json-table",
        class: Class::Structured,
        prompt: "Read this table of name and score rows: Ana 120, Ben 85, Cy 210, Dee 99. Return only one JSON object with exactly these fields: rows (integer count of rows), sum (integer total of scores), max (integer highest score), over_100 (array of names with a score over 100, in table order). Return no other text.",
        max_tokens: 300,
        scoring: Scoring::Json(&[
            Rule::JsonField("rows", "4"),
            Rule::JsonField("sum", "514"),
            Rule::JsonField("max", "210"),
            Rule::JsonField("over_100", "[\"Ana\",\"Cy\"]"),
        ]),
    },
    Case {
        id: "prose-open",
        class: Class::Prose,
        prompt: "Write exactly 3 sentences about a rain barrel for a small garden. Start with this exact sentence: A rain barrel stores roof water for dry days. One later sentence must contain the exact words overflow pipe. Do not use the word very. Use between 35 and 80 words in total.",
        max_tokens: 400,
        scoring: Scoring::Rules(&[
            Rule::StartsWith("A rain barrel stores roof water for dry days."),
            Rule::Sentences(3),
            Rule::Contains("overflow pipe"),
            Rule::NotContains(" very "),
            Rule::Words(35, 80),
        ]),
    },
    Case {
        id: "prose-list",
        class: Class::Prose,
        prompt: "Return exactly 4 lines and nothing else. Each line starts with - and one space. Line 1 names spring, line 2 names summer, line 3 names autumn and line 4 names winter. Each line gives one garden task for that season in at most 8 words after the dash.",
        max_tokens: 300,
        scoring: Scoring::Rules(&[
            Rule::Lines(4),
            Rule::EveryLineStartsWith("- "),
            Rule::LineContains(0, "spring"),
            Rule::LineContains(1, "summer"),
            Rule::LineContains(2, "autumn"),
            Rule::LineContains(3, "winter"),
            Rule::EveryLineWordsAtMost(8),
        ]),
    },
    Case {
        id: "prose-rewrite",
        class: Class::Prose,
        prompt: "Rewrite this sentence in your own words as one sentence of at most 20 words: The technician replaced the worn valve on line 17 during the 2049 maintenance window. Keep the exact numbers 17 and 2049. Keep the word valve. Return only the rewritten sentence.",
        max_tokens: 300,
        scoring: Scoring::Rules(&[
            Rule::Sentences(1),
            Rule::Words(1, 20),
            Rule::Contains("17"),
            Rule::Contains("2049"),
            Rule::Contains("valve"),
        ]),
    },
    Case {
        id: "prose-form",
        class: Class::Prose,
        prompt: "Read this note: The north shelf holds 26 blue binders. Return exactly 3 lines in this form and nothing else: line 1 is Location: north shelf; line 2 is Count: 26; line 3 is Color: blue. Copy those three lines exactly.",
        max_tokens: 200,
        scoring: Scoring::Rules(&[
            Rule::Lines(3),
            Rule::EqualsExact("Location: north shelf\nCount: 26\nColor: blue"),
        ]),
    },
];
