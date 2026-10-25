//! Raw file flags for the CLI's supported targets; no guessed Unix fallback.
//!
//! ABI table (hex for O_*; decimal for the remaining constants):
//!
//! ```text
//! target          O_NOFOLLOW O_DIRECTORY O_CLOEXEC O_NONBLOCK O_CREAT O_EXCL
//! Linux aarch64   0x8000     0x4000      0x80000   0x800      0x40    0x80
//! Linux x86_64    0x20000    0x10000     0x80000   0x800      0x40    0x80
//! macOS           0x100      0x100000    0x1000000 0x4        0x200   0x800
//! All targets: O_RDONLY = 0, O_RDWR = 2, F_GETFL = 3, F_SETFL = 4,
//! LOCK_EX = 2, LOCK_NB = 4.
//! ```
//!
//! Linux ARM64 overrides O_NOFOLLOW and O_DIRECTORY (0x20000 is O_LARGEFILE):
//! <https://github.com/torvalds/linux/blob/v6.12/arch/arm64/include/uapi/asm/fcntl.h>
//! Linux x86_64 uses the generic file flags, as do ARM64's other flags here:
//! <https://github.com/torvalds/linux/blob/v6.12/include/uapi/asm-generic/fcntl.h>
//! macOS (both architectures):
//! <https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/fcntl.h>

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub const O_NOFOLLOW: i32 = 0x8000;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub const O_NOFOLLOW: i32 = 0x20000;
#[cfg(target_os = "macos")]
pub const O_NOFOLLOW: i32 = 0x100;

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub const O_DIRECTORY: i32 = 0x4000;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub const O_DIRECTORY: i32 = 0x10000;
#[cfg(target_os = "macos")]
pub const O_DIRECTORY: i32 = 0x100000;

#[cfg(target_os = "linux")]
pub const O_CLOEXEC: i32 = 0x80000;
#[cfg(target_os = "macos")]
pub const O_CLOEXEC: i32 = 0x1000000;

#[cfg(target_os = "linux")]
pub const O_CREAT: i32 = 0x40;
#[cfg(target_os = "macos")]
pub const O_CREAT: i32 = 0x200;
#[cfg(target_os = "linux")]
pub const O_EXCL: i32 = 0x80;
#[cfg(target_os = "macos")]
pub const O_EXCL: i32 = 0x800;

pub const O_RDONLY: i32 = 0;
pub const O_RDWR: i32 = 2;

#[cfg(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
pub const O_NONBLOCK: i32 = 0x800;
#[cfg(target_os = "macos")]
pub const O_NONBLOCK: i32 = 0x4;

#[cfg(not(any(
    all(
        target_os = "linux",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ),
    target_os = "macos"
)))]
compile_error!(
    "Raw file flags need an audited ABI for this target; supported: Linux aarch64/x86_64 and macOS."
);

pub const F_GETFL: i32 = 3;
pub const F_SETFL: i32 = 4;
pub const LOCK_EX: i32 = 2;
pub const LOCK_NB: i32 = 4;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, OpenOptions};
    use std::os::unix::fs::{OpenOptionsExt, symlink};
    use std::path::PathBuf;

    struct TestDir(PathBuf);
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn constants_match_host_system_headers() {
        let _process_guard = crate::runner::process_test_lock();
        use std::process::Command;
        let suffix = crate::text::random_hex(12).unwrap();
        let root = TestDir(std::env::temp_dir().join(format!("vade-flags-abi-{suffix}")));
        fs::create_dir(&root.0).unwrap();
        let source = root.0.join("flags.c");
        let binary = root.0.join("flags");
        fs::write(&source, r#"#include <fcntl.h>
#include <sys/file.h>
#include <stdio.h>
int main(void) { printf("%d %d %d %d %d %d %d %d %d %d %d %d\n", O_NOFOLLOW, O_DIRECTORY, O_CLOEXEC, O_NONBLOCK, O_CREAT, O_EXCL, O_RDONLY, O_RDWR, F_GETFL, F_SETFL, LOCK_EX, LOCK_NB); return 0; }
"#).unwrap();
        let compiled = Command::new("cc")
            .arg(source)
            .arg("-o")
            .arg(&binary)
            .status()
            .unwrap();
        assert!(compiled.success());
        let result = Command::new(&binary).output().unwrap();
        assert!(result.status.success());
        let actual: Vec<i32> = String::from_utf8(result.stdout)
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let mut expected = vec![
            O_NOFOLLOW,
            O_DIRECTORY,
            O_CLOEXEC,
            O_NONBLOCK,
            O_CREAT,
            O_EXCL,
        ];
        expected.extend_from_slice(&[O_RDONLY, O_RDWR, F_GETFL, F_SETFL, LOCK_EX, LOCK_NB]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn nofollow_opens_regular_files_and_refuses_symlinks() {
        let root = TestDir(
            fs::canonicalize(std::env::temp_dir())
                .unwrap()
                .join(format!(
                    "vade-nofollow-{}",
                    crate::text::random_hex(12).unwrap()
                )),
        );
        fs::create_dir(&root.0).unwrap();
        let target = root.0.join("file");
        fs::write(&target, "keep").unwrap();
        OpenOptions::new()
            .read(true)
            .custom_flags(O_NOFOLLOW)
            .open(&target)
            .expect("O_NOFOLLOW must still allow a regular file");

        fs::create_dir(root.0.join("directory")).unwrap();
        // A directory link also catches accidentally using O_DIRECTORY.
        for name in ["file", "directory"] {
            let link = root.0.join(format!("{name}-link"));
            symlink(root.0.join(name), &link).unwrap();
            fs::File::open(&link).expect("the control symlink must be valid");
            // Exercise the actual flag, with no preliminary metadata check.
            let error = OpenOptions::new()
                .read(true)
                .custom_flags(O_NOFOLLOW)
                .open(&link)
                .expect_err("O_NOFOLLOW must refuse the symlink");
            // ELOOP: Linux asm-generic/errno.h; macOS bsd/sys/errno.h.
            #[cfg(target_os = "linux")]
            let eloop = 40;
            #[cfg(target_os = "macos")]
            let eloop = 62;
            assert_eq!(error.raw_os_error(), Some(eloop), "{error}");
        }
        assert_eq!(fs::read_to_string(target).unwrap(), "keep");
    }
}
