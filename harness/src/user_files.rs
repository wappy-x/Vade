//! Bounded reads of user settings through existing directory handles.
use std::{fs::File, io::{self, Read}};

pub(crate) fn text(file: File, cap: usize) -> io::Result<String> {
    let meta = file.metadata()?;
    if !meta.is_file() { return Err(io::Error::other("settings must be regular files")); }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if meta.nlink() != 1 { return Err(io::Error::other("settings must not have hard links")); }
    }
    let mut text = String::new();
    file.take(cap as u64 + 1).read_to_string(&mut text)?;
    if text.len() > cap { return Err(io::Error::other("settings file exceeds its size limit")); }
    Ok(text)
}

pub(crate) fn prefix(text: &str, cap: usize) -> &str {
    let mut end = text.len().min(cap);
    while !text.is_char_boundary(end) { end -= 1; }
    &text[..end]
}
