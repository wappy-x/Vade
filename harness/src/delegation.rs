//! A delegated backend owns an entire external turn. It is deliberately not a
//! native ModelBackend, and never participates in Vade's tool/model loop.
use std::io;
use std::sync::{atomic::AtomicBool, mpsc::Receiver};
use vade_harness::app_server::{self, hermes};

pub enum Options {
    Codex(app_server::Options),
    Hermes(hermes::Options),
}
pub enum Backend {
    Codex(app_server::Client),
    Hermes(hermes::Client),
}
impl Backend {
    pub fn connect(options: Options, cancel: &AtomicBool) -> io::Result<Self> {
        match options {
            Options::Codex(options) => {
                app_server::Client::connect_cancellable(options, cancel).map(Self::Codex)
            }
            Options::Hermes(options) => {
                hermes::Client::connect_cancellable(options, cancel).map(Self::Hermes)
            }
        }
    }
    pub fn matches(&self, options: &Options) -> bool {
        matches!(
            (self, options),
            (Self::Codex(_), Options::Codex(_)) | (Self::Hermes(_), Options::Hermes(_))
        )
    }
    pub fn session_id(&self) -> &str {
        match self {
            Self::Codex(client) => client.thread_id(),
            Self::Hermes(client) => client.session_id(),
        }
    }
    pub fn completed_text(&self) -> Option<&str> {
        match self {
            Self::Codex(client) => client.completed_text(),
            Self::Hermes(client) => client.completed_text(),
        }
    }
    pub fn turn(
        &mut self,
        text: &str,
        cancel: &AtomicBool,
        controls: &Receiver<app_server::Control>,
        on_event: &mut dyn FnMut(app_server::Event),
    ) -> io::Result<app_server::TurnOutcome> {
        match self {
            Self::Codex(client) => client.turn_with_control(text, cancel, controls, on_event),
            Self::Hermes(client) => client.turn(text, cancel, on_event),
        }
    }
}
