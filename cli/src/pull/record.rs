//! The check record that stays beside the model: `.vade-verified` in the model
//! folder, one row per file with its SHA-256 and what identifies it on this disk.
//! A pull's own state is private, so without this file a new Vade home or a
//! second account on the Spark would hash the checkpoint again. A row counts only
//! while the file has the same size, inode, modification time and change time.
//! A copy on another disk has other inodes, so its rows count for nothing there.
use super::*;

/// The record's name in the model folder.
pub(crate) const RECORD: &str = ".vade-verified";

/// A file that was checked and has not changed since: its SHA-256, and the
/// metadata that this check compared.
pub(crate) struct Verified { pub sha256: String, pub meta: fs::Metadata }

/// Whether a record with this owner and mode can speak for the files of a folder
/// with this owner. The owner of the folder can also change the checkpoint and
/// the checksum list beside it, so this gives that account no new power. In a
/// shared folder, every account that uses the folder trusts its owner's checks.
pub(crate) fn trusted(owner: u32, mode: u32, folder_owner: u32) -> bool { owner == folder_owner && mode & 0o002 == 0 }

fn same(row: &Checked, meta: &fs::Metadata) -> bool {
    meta.is_file() && (row.size, row.inode, row.modified, row.nanos, row.changed, row.changed_nanos) == identity(meta)
}

/// The rows of the record that can still speak for `folder`, with each file's
/// metadata: the owner of the folder wrote the record, and the file has not
/// changed. Empty with no record.
fn kept(folder: &Path) -> BTreeMap<String, (Checked, fs::Metadata)> {
    let path = folder.join(RECORD);
    let owned = fs::symlink_metadata(&path).ok().zip(fs::metadata(folder).ok())
        .is_some_and(|(record, folder)| record.is_file() && trusted(record.uid(), record.mode(), folder.uid()));
    if !owned { return BTreeMap::new(); }
    let text = read_small(&path, 16 << 20, false).map(|bytes| String::from_utf8_lossy(&bytes).into_owned()).unwrap_or_default();
    parse_checked(&text).into_iter().filter_map(|(name, row)| {
        let file = folder.join(&name);
        let usable = hub::safe_path(&name) && no_links(folder, &file).is_ok() && crate::text::is_lower_hex(&row.sha256, 64);
        let meta = fs::symlink_metadata(&file).ok().filter(|meta| usable && same(&row, meta))?;
        Some((name, (row, meta)))
    }).collect()
}

/// The files of `folder` that its record covers and that have not changed, by
/// path in the folder. Empty with no record, or while a pull replaces files.
pub(crate) fn verified(folder: &Path) -> BTreeMap<String, Verified> {
    if fs::symlink_metadata(folder.join(STATE).join(UPDATING)).is_ok() { return BTreeMap::new(); }
    kept(folder).into_iter().map(|(name, (row, meta))| (name, Verified { sha256: row.sha256, meta })).collect()
}

/// What identifies each named file now, before a check of its `(name, sha256)`.
pub(crate) struct Snapshot(BTreeMap<String, Checked>);

pub(crate) fn snapshot(folder: &Path, rows: &[(String, String)]) -> Snapshot {
    Snapshot(rows.iter().filter(|(name, _)| hub::safe_path(name) && no_links(folder, &folder.join(name)).is_ok()).filter_map(|(name, sha256)| {
        let meta = fs::symlink_metadata(folder.join(name)).ok().filter(fs::Metadata::is_file)?;
        let (size, inode, modified, nanos, changed, changed_nanos) = identity(&meta);
        Some((name.clone(), Checked { size, inode, modified, nanos, changed, changed_nanos, expected: sha256.clone(), sha256: sha256.clone() }))
    }).collect())
}

/// Add the files of `checked` that did not change during their check to the
/// rows that still count, as `how` found them. A row that no longer counts goes:
/// this save must not sign the rows of a record that nobody trusts. Only a missed
/// hash follows a failure here, so this account may have no write access to the
/// folder.
pub(crate) fn save(folder: &Path, how: &str, checked: Snapshot) {
    let path = folder.join(RECORD);
    let mut rows: BTreeMap<String, Checked> = kept(folder).into_iter().map(|(name, (row, _))| (name, row)).collect();
    rows.extend(checked.0.into_iter().filter(|(name, row)| fs::symlink_metadata(folder.join(name)).is_ok_and(|meta| same(row, &meta))));
    let _ = publish(&path, &format!("# verified by {how} {}\n{}", utc(std::time::SystemTime::now()), checked_text(&rows)));
}

/// The record of a finished pull: the rows this pull checked, for its files.
pub(super) fn after_pull(folder: &Path, scratch: &Path, commit: &str, records: &[Record]) {
    let mut rows = read_checked(scratch);
    rows.retain(|name, _| records.iter().any(|record| record.path == *name));
    save(folder, &format!("vade pull {commit}"), Snapshot(rows));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn only_the_owner_of_the_folder_can_speak_for_its_files() {
        assert!(trusted(1000, 0o100644, 1000));
        assert!(trusted(1000, 0o100664, 1000), "umask 002 on Ubuntu gives a group bit");
        assert!(!trusted(1001, 0o100644, 1000), "another account wrote this file");
        assert!(!trusted(1000, 0o100666, 1000), "any account can change this file");
    }

    #[test]
    fn a_row_counts_only_for_the_same_file_and_the_record_keeps_its_rows() {
        let folder = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-record-{}", crate::text::random_hex(8).unwrap()));
        fs::create_dir_all(&folder).unwrap();
        for name in ["a", "b"] { fs::write(folder.join(name), name).unwrap(); }
        let hash = "a".repeat(64);
        // A file that changes during its check gets no row.
        let before = snapshot(&folder, &[("a".into(), hash.clone()), ("b".into(), hash.clone()), ("../c".into(), hash.clone()), ("gone".into(), hash.clone())]);
        fs::write(folder.join("b"), "longer").unwrap();
        save(&folder, "vade install", before);
        let text = fs::read_to_string(folder.join(RECORD)).unwrap();
        assert!(text.starts_with("# verified by vade install 20"), "{text}");
        assert_eq!(verified(&folder).keys().collect::<Vec<_>>(), ["a"]);
        // A second save adds its rows and keeps the first.
        let second = snapshot(&folder, &[("b".into(), "b".repeat(64))]);
        save(&folder, "vade pull 0123", second);
        let found = verified(&folder);
        assert_eq!((found["a"].sha256.as_str(), found["b"].sha256.len()), (hash.as_str(), 64));
        assert!(fs::read_to_string(folder.join(RECORD)).unwrap().starts_with("# verified by vade pull 0123 "));
        // A rewrite with the same size is a new file.
        std::thread::sleep(Duration::from_millis(20));
        fs::write(folder.join("a"), "z").unwrap();
        assert!(!verified(&folder).contains_key("a"));
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn a_save_signs_only_the_rows_that_still_count() {
        let folder = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-record-{}", crate::text::random_hex(8).unwrap()));
        fs::create_dir_all(&folder).unwrap();
        for name in ["a", "b", "c"] { fs::write(folder.join(name), name).unwrap(); }
        let record = folder.join(RECORD);
        let rows = |names: &[&str]| names.iter().map(|name| (name.to_string(), name.repeat(64))).collect::<Vec<_>>();
        // A record that any account can change speaks for nothing, so the next
        // save does not copy its rows into a record that counts.
        save(&folder, "vade install", snapshot(&folder, &rows(&["a", "b"])));
        fs::set_permissions(&record, fs::Permissions::from_mode(0o666)).unwrap();
        assert!(verified(&folder).is_empty());
        save(&folder, "vade install", snapshot(&folder, &rows(&["c"])));
        assert_eq!(verified(&folder).keys().collect::<Vec<_>>(), ["c"]);
        assert!(!fs::read_to_string(&record).unwrap().contains(&"a".repeat(64)));
        // A row goes when its file goes.
        save(&folder, "vade install", snapshot(&folder, &rows(&["a"])));
        fs::remove_file(folder.join("c")).unwrap();
        save(&folder, "vade install", snapshot(&folder, &rows(&["b"])));
        assert_eq!(verified(&folder).keys().collect::<Vec<_>>(), ["a", "b"]);
        assert!(!fs::read_to_string(&record).unwrap().contains(&"c".repeat(64)));
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn a_save_during_a_pull_keeps_the_rows_that_still_count() {
        let folder = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("vade-record-{}", crate::text::random_hex(8).unwrap()));
        fs::create_dir_all(folder.join(STATE)).unwrap();
        for name in ["a", "b"] { fs::write(folder.join(name), name).unwrap(); }
        save(&folder, "vade install", snapshot(&folder, &[("a".into(), "a".repeat(64))]));
        fs::write(folder.join(STATE).join(UPDATING), "commit\n").unwrap();
        save(&folder, "vade pull", snapshot(&folder, &[("b".into(), "b".repeat(64))]));
        fs::remove_file(folder.join(STATE).join(UPDATING)).unwrap();
        assert_eq!(verified(&folder).keys().collect::<Vec<_>>(), ["a", "b"]);
        fs::remove_dir_all(folder).unwrap();
    }
}
