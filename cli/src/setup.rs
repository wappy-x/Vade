//! `vade config`: which setup the lifecycle commands will use, and what it
//! holds. Reads only; the selection order matches `main`'s dispatch.
use crate::{
    evals::json::{self, Value},
    output,
    style,
};

/// Print the selected setup as an aligned table, or one `vade.config.v1`
/// object with `json`. A setup that cannot be read is still named, then its
/// error ends the command (in the object's `error` field with `json`).
pub fn show(json: bool) -> Result<(), String> {
    let Selected { setup, path, rows, error } = select();
    // A known failure stays one even if the reader closes stdout first.
    if error.is_some() { output::operation_failed(); }
    if json {
        let settings = rows.as_ref().map_or(Value::Null, |rows| {
            json::object(rows.iter().map(|(_, _, key, value)| (*key, value.clone())).collect())
        });
        let value = json::object(vec![
            ("schema", "vade.config.v1".into()),
            ("setup", setup.into()),
            ("path", path.as_ref().map_or(Value::Null, |path| path.as_str().into())),
            ("settings", settings),
            ("error", error.as_ref().map_or(Value::Null, |error| style::split(error).0.as_str().into())),
            ("next", error.as_ref().and_then(|error| style::split(error).1).map_or(Value::Null, Into::into)),
        ]);
        crate::checks::print_json(&value)?;
        return if error.is_some() { Err(crate::checks::REPORTED.into()) } else { Ok(()) };
    }
    let mut table = vec![("Setup", describe(setup).to_owned())];
    if let Some(path) = &path { table.push(("File", crate::text::terminal_text(path).replace('\n', "\\n"))); }
    for (label, text, _, _) in rows.iter().flatten() { table.push((label, crate::text::terminal_text(text).into_owned())); }
    let mut out = style::table(&table, 0);
    if setup == "none" {
        out += &crate::status::next_steps(&[("On a Spark", "vade launch"), ("From a controller", "set VADE_CONFIG (INSTALL.md)")]);
    }
    let written = output::text(format_args!("{out}"));
    error.map_or(written, Err)
}

fn describe(setup: &str) -> &'static str {
    match setup {
        "local" => "local (saved defaults on this Spark)",
        "controller" => "controller (drives Sparks over SSH)",
        "legacy" => "legacy (vade run qwen3-8b)",
        _ => "none yet",
    }
}

type Rows = Vec<(&'static str, String, &'static str, Value)>;

/// The selected setup, its file, its rows, and why it cannot be read.
struct Selected { setup: &'static str, path: Option<String>, rows: Option<Rows>, error: Option<String> }
impl Selected {
    fn failed(setup: &'static str, path: Option<String>, error: String) -> Self { Self { setup, path, rows: None, error: Some(error) } }
}

/// The setup lifecycle commands will use, in `main`'s selection order:
/// explicit VADE_CONFIG, the local owner (saved defaults or an owned-engine
/// record, as [`crate::launch::handles`] decides), the legacy home, then the
/// user config file.
fn select() -> Selected {
    if std::env::var_os("VADE_CONFIG").is_none() {
        if crate::launch::owner_configured() {
            let path = crate::launch::root().ok().map(|root| root.join("config").display().to_string());
            return match crate::launch::saved_settings() {
                Ok(Some((path, rows))) => Selected { setup: "local", path: Some(path.display().to_string()), rows: Some(rows), error: None },
                // The engine record keeps this setup local even without its config.
                Ok(None) => Selected::failed("local", path, "An owned engine has no saved config; preserve the state folder and run vade down.".into()),
                Err(error) => Selected::failed("local", path, error),
            };
        }
        if crate::launch::legacy_configured() {
            return match crate::run::home() {
                Ok(home) => Selected { setup: "legacy", path: Some(home.display().to_string()), rows: None, error: None },
                Err(error) => Selected::failed("legacy", None, error),
            };
        }
        if !crate::config::config_path().is_ok_and(|path| path.is_file()) {
            return Selected { setup: "none", path: None, rows: None, error: None };
        }
    }
    let path = match crate::config::config_path() {
        Ok(path) => path.display().to_string(),
        Err(error) => return Selected::failed("controller", None, error),
    };
    let config = match crate::config::Config::load() {
        Ok(config) => config,
        Err(error) => return Selected::failed("controller", Some(path), error),
    };
    let names: Vec<&str> = config.machines.iter().map(|machine| machine.name.as_str()).collect();
    let rows = vec![
        ("Model", format!("{} ({} at {})", config.model, config.model_repo, config.model_revision), "model", config.model.as_str().into()),
        ("Sparks", format!("{}: {}", names.len(), names.join(", ")), "machines", Value::Array(names.iter().map(|name| (*name).into()).collect())),
        ("Port", format!("{} on this computer", config.local_port), "local_port", Value::Int(config.local_port.into())),
        ("Build", config.prebuilt_dir.as_ref().map_or_else(|| config.engine_source.display().to_string(), |dir| format!("prebuilt {}", dir.display())), "prebuilt_dir",
            config.prebuilt_dir.as_ref().map_or(Value::Null, |dir| dir.to_string_lossy().as_ref().into())),
        ("State", config.state_dir.display().to_string(), "state_dir", config.state_dir.to_string_lossy().as_ref().into()),
    ];
    Selected { setup: "controller", path: Some(path), rows: Some(rows), error: None }
}
