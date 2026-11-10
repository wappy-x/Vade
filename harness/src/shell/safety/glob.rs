//! Whether a path pattern (`/u*`, `/dev/sd?`, `/[!a]sr`) can match a
//! protected name. The answer leans to yes: a form this does not read
//! (`[:class:]`, a long pattern), and any pattern met after the work limit
//! of one classification, matches anything.
use std::cell::Cell;

/// Patterns longer than this match anything.
const LONG: usize = 4096;

/// Work units (bytes scanned, pattern steps) for one classification.
const WORK_LIMIT: u64 = 20_000_000;

thread_local! {
    static WORK: Cell<u64> = const { Cell::new(0) };
}

/// Start a classification's work count.
pub(super) fn reset_work() {
    WORK.with(|work| work.set(0));
}

/// The work counted so far in this classification.
#[cfg(test)]
pub(super) fn work_used() -> u64 {
    WORK.with(Cell::get)
}

/// Count `units` of work. False once the limit is passed: the caller then
/// gives its most careful answer.
pub(super) fn charge(units: usize) -> bool {
    WORK.with(|work| {
        let used = work.get().saturating_add(units as u64);
        work.set(used);
        used <= WORK_LIMIT
    })
}

enum Part {
    /// `*`: any run of characters but `/`.
    Star,
    /// `?` or a bracket: one character but `/`.
    One(Set),
    Byte(u8),
}

struct Set {
    any: bool,
    negated: bool,
    ranges: Vec<(u8, u8)>,
}

impl Set {
    fn has(&self, c: u8) -> bool {
        if self.any {
            return true;
        }
        // Names match in any case (macOS file names ignore it): the set
        // matches when either case of `c` passes it, so `[!U]` matches `u`.
        let listed = |c: u8| self.ranges.iter().any(|&(low, high)| low <= c && c <= high);
        [c.to_ascii_lowercase(), c.to_ascii_uppercase()].into_iter().any(|c| listed(c) != self.negated)
    }
}

/// True when `text` has a pattern character that a shell can expand.
pub(super) fn pattern(text: &str) -> bool {
    text.contains(['*', '?', '['])
}

/// A pattern read once, for many names.
pub(super) struct Glob {
    parts: Vec<Part>,
    /// Matches anything: too long, or past the work limit.
    any: bool,
}

/// The parts of the pattern that can come next: index `i` when the first
/// `i` parts match the text so far. Sorted, no repeats. `ANY` alone: the
/// pattern can match anything from here.
pub(super) type States = Vec<usize>;
const ANY: usize = usize::MAX;

impl Glob {
    pub(super) fn new(pattern: &str) -> Self {
        match (pattern.len() <= LONG && charge(pattern.len())).then(|| parts(pattern)).flatten() {
            Some(parts) => Self { parts, any: false },
            None => Self { parts: Vec::new(), any: true },
        }
    }
    /// The states before any text.
    pub(super) fn start(&self) -> States {
        if self.any { vec![ANY] } else { self.close(vec![0]) }
    }
    /// The states after `text`, from `states`. Empty when nothing can match.
    pub(super) fn run(&self, mut states: States, text: &str) -> States {
        for &c in text.as_bytes() {
            if states.is_empty() || states.contains(&ANY) {
                break;
            }
            if !charge(states.len() + 1) {
                return vec![ANY];
            }
            let mut next = Vec::new();
            for &at in &states {
                let Some(part) = self.parts.get(at) else { continue };
                if c == b'/' && !matches!(part, Part::Byte(_)) {
                    continue;
                }
                match part {
                    Part::Star => next.push(at),
                    Part::One(set) if set.has(c) => next.push(at + 1),
                    Part::Byte(b) if b.eq_ignore_ascii_case(&c) => next.push(at + 1),
                    _ => {}
                }
            }
            next.sort_unstable();
            next.dedup();
            states = self.close(next);
        }
        states
    }
    /// True when the pattern can match a name that starts with the text
    /// that led to `states`.
    pub(super) fn open(states: &States) -> bool {
        !states.is_empty()
    }
    /// True when the pattern can match all of `name`.
    pub(super) fn matches(&self, name: &str) -> bool {
        let states = self.run(self.start(), name);
        states.contains(&ANY) || states.contains(&self.parts.len())
    }
    /// A `*` can match nothing.
    fn close(&self, mut states: States) -> States {
        let mut at = 0;
        while at < states.len() {
            if matches!(self.parts.get(states[at]), Some(Part::Star)) && !states.contains(&(states[at] + 1)) {
                states.push(states[at] + 1);
            }
            at += 1;
        }
        states.sort_unstable();
        states
    }
}

/// The parts of a pattern, or None past the work limit. A `[` with no `]`
/// after it is a plain `[` at once; every bracket scan counts as work.
fn parts(pattern: &str) -> Option<Vec<Part>> {
    let bytes = pattern.as_bytes();
    let last_close = bytes.iter().rposition(|&b| b == b']');
    let mut parts = Vec::new();
    let mut at = 0;
    while let Some(&b) = bytes.get(at) {
        at += 1;
        match b {
            b'*' => {
                if !matches!(parts.last(), Some(Part::Star)) {
                    parts.push(Part::Star);
                }
            }
            b'?' => parts.push(Part::One(Set { any: true, negated: false, ranges: Vec::new() })),
            b'[' if last_close.is_none_or(|close| close <= at) => parts.push(Part::Byte(b'[')),
            b'[' => match set(&bytes[at..]) {
                Some((set, used)) => {
                    if !charge(used) {
                        return None;
                    }
                    parts.push(Part::One(set));
                    at += used;
                }
                None => {
                    if !charge(bytes.len() - at) {
                        return None;
                    }
                    parts.push(Part::Byte(b'['));
                }
            },
            b'\\' if at < bytes.len() => {
                parts.push(Part::Byte(bytes[at]));
                at += 1;
            }
            _ => parts.push(Part::Byte(b)),
        }
    }
    Some(parts)
}

/// The bracket after `[`, and the bytes it uses with its `]`. A `]` first
/// is a member. None when it does not close.
fn set(text: &[u8]) -> Option<(Set, usize)> {
    let negated = matches!(text.first(), Some(b'!' | b'^'));
    let mut set = Set { any: false, negated, ranges: Vec::new() };
    let mut at = usize::from(negated);
    let first = at;
    loop {
        let c = *text.get(at)?;
        if c == b']' && at > first {
            return Some((set, at + 1));
        }
        // `[:alpha:]`, `[=a=]` and `[.a.]`: any character.
        if c == b'[' && matches!(text.get(at + 1), Some(b':' | b'=' | b'.')) {
            let kind = text[at + 1];
            let end = text[at + 2..].windows(2).position(|w| w == [kind, b']'])?;
            set.any = true;
            at += end + 4;
            continue;
        }
        let low = if c == b'\\' {
            at += 1;
            *text.get(at)?
        } else {
            c
        };
        if text.get(at + 1) == Some(&b'-') && text.get(at + 2).is_some_and(|&high| high != b']') {
            set.ranges.push((low, text[at + 2]));
            at += 3;
        } else {
            set.ranges.push((low, low));
            at += 1;
        }
    }
}
