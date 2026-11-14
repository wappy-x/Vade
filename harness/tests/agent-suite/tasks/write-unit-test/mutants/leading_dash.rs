//! URL slugs.

/// Turns a title into a URL slug.
///
/// ASCII letters become lower case and ASCII digits stay. Every run of other
/// characters (spaces, punctuation, non-ASCII) becomes one `-`. The slug
/// never starts or ends with `-`.
pub fn slugify(title: &str) -> String {
    let mut out = String::new();
    let mut gap = false;
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            if gap {
                out.push('-');
            }
            gap = false;
            out.push(c.to_ascii_lowercase());
        } else {
            gap = true;
        }
    }
    out
}
