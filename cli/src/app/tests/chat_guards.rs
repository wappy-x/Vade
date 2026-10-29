//! vade chat with a controller record opens the harness only after every
//! rank, the tunnel and the engine's health check agree.
use super::*;

fn refused(f: &mut Fixture, expected: &str) {
    f.app.runner.calls.clear();
    assert_eq!(f.app.execute(Command::Chat(Default::default())).unwrap_err(), expected);
    assert!(!f.app.runner.calls.iter().any(|e| e.label == "Vade chat"), "chat opened despite: {expected}");
}

#[test]
fn chat_refuses_each_unhealthy_state_before_opening_the_harness() {
    let mut f = Fixture::new(2, false);
    refused(&mut f, "Vade is stopped. Run vade up --fp8, then vade chat.");
    f.app.up(&fp8_options()).unwrap();
    f.app.runner.calls.clear();
    f.app.execute(Command::Chat(Default::default())).unwrap();
    assert!(f.app.runner.calls.iter().any(|e| e.label == "Vade chat"));

    let mut session = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    session.phase = "starting".into();
    session.save(&f.app.config.state_dir).unwrap();
    refused(&mut f, "Engine startup or cleanup is unfinished. Run vade status before chat.");
    session.phase = "running".into();
    session.save(&f.app.config.state_dir).unwrap();

    f.app.runner.started.remove("b");
    refused(&mut f, "An engine rank has stopped. Run vade status, then vade down.");
    f.app.runner.started.insert("b".into());

    f.app.runner.tunnel_bad = true;
    refused(&mut f, "The owned controller tunnel is down. Run vade status before chatting.");
    f.app.runner.tunnel_bad = false;

    f.app.runner.ready_model = Some("another-model".into());
    refused(&mut f, "The engine is not ready. Run vade status or vade doctor.");
    f.app.runner.ready_model = None;

    f.app.execute(Command::Chat(Default::default())).unwrap();
    f.app.execute(Command::Down).unwrap();
    assert!(!f.recorded());
}
