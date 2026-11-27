use super::*;

#[test]
fn deep_text_depth_survives_session_storage_and_launch_arguments() {
    let f=Fixture::new(2, true);let manifest=f.app.manifest().unwrap();
    for depth in 4..=7 {
        let options=UpOptions {diagnostic_unqualified_text:true,mtp:depth,mtp_explicit:true,..fp8_options()};
        let mut session=Session::new(&f.app.config,&options,&manifest,"a".repeat(64)).unwrap();
        assert!(session.resident_text && !session.diagnostic_unqualified_text);
        session.save(&f.app.config.state_dir).unwrap();
        let loaded=Session::load(&f.app.config.state_dir).unwrap().unwrap();
        assert_eq!((loaded.mtp, loaded.resident_text), (depth, true));
        assert!(loaded.engine_args(0).windows(2).any(|pair|pair==["--mtp",&depth.to_string()]));
        assert!(!loaded.engine_args(0).iter().any(|arg| arg == "--diagnostic-unqualified-text"));
        session.resident_text=false;session.diagnostic_unqualified_text=false;session.save(&f.app.config.state_dir).unwrap();
        assert!(Session::load(&f.app.config.state_dir).is_err());
    }
}
