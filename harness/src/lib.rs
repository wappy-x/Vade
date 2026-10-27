//! Vade's terminal harness. Inference and KV state belong to the upstream engine.
pub mod agent;
pub mod agent_wire;
pub mod app_server;
pub mod child_agents;
pub mod contract;
pub mod doctor;
pub mod diagnostics;
pub mod gateway;
pub mod json;
pub mod prompt;
pub mod protocol;
pub mod session;
pub mod setup;
pub mod terminal;
pub mod tools;
pub mod transport;

pub mod artifacts;
pub mod branch;
pub mod coding_tools;
pub mod context;
pub mod edit;
mod patch_feedback;
mod submission;
pub(crate) use edit::line_edit;
pub mod journal;
pub mod kv;
pub mod project;
pub mod shell;
#[cfg(test)]
mod test_workspace;
pub mod quality;
pub mod wake;
pub mod workload;

pub mod auth;
