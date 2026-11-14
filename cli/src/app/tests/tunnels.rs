//! The tunnel keeps its control socket and logs in the private state folder.
use super::*;

#[test]
fn every_tunnel_operation_uses_the_same_private_directory_and_short_name() {
    let f = Fixture::new(2, false);
    let manifest = Manifest::load(&f.app.config.build_manifest).unwrap();
    let session = Session::new(&f.app.config, &fp8_options(), &manifest, "a".repeat(64)).unwrap();
    for operation in [None, Some("check"), Some("exit")] {
        let exec = f.app.tunnel_at(&session, 0, operation);
        assert_eq!(exec.cwd.as_ref(), Some(&f.app.config.state_dir));
        assert_eq!(exec.detached_log, operation.is_none().then(|| f.app.config.state_dir.join("logs/tunnel-0.log")));
        assert_eq!(exec.args.iter().any(|arg| arg == "-fNT"), operation.is_none());
        assert!(exec.stdin.is_empty());
        let path = exec.args.windows(2).find(|pair| pair[0] == "-S").unwrap()[1].clone();
        assert_eq!(Path::new(&path).file_name(), session.socket(&f.app.config.state_dir).file_name());
        assert!(Path::new(&path).is_relative() && path.len() + 18 <= 104);
    }
}
