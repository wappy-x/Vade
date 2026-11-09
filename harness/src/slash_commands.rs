//! User prompt templates. Built-in dispatch stays in the chat CLI.
use std::{collections::BTreeMap, io, path::PathBuf};
use crate::{session::MAX_PROMPT_BYTES, tools::Workspace, user_files::text};

#[derive(Clone, Debug)]
pub struct Template { pub name: String, pub description: String, body: String }
impl Template {
    fn parse(name: &str, source: String) -> Self {
        let (first, rest) = source.split_once('\n').unwrap_or((&source, ""));
        let (description, body) = match first.strip_prefix("description:") {
            Some(description) => (description.trim().to_owned(), rest.to_owned()),
            None => (String::new(), source),
        };
        Self { name: name.into(), description, body }
    }
    pub fn expand(&self, arguments: &str) -> io::Result<String> {
        let count = self.body.matches("$ARGUMENTS").count();
        let size = self.body.len().saturating_sub(count * "$ARGUMENTS".len())
            .saturating_add(count.saturating_mul(arguments.len()));
        if size > MAX_PROMPT_BYTES { return Err(io::Error::other("command prompt exceeds the input limit")); }
        Ok(self.body.replace("$ARGUMENTS", arguments))
    }
}
pub fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 64 && name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}

/// The supplied home follows --vade-home, VADE_HOME, then HOME/.vade.
/// An absent directory is empty. An invalid entry is an explicit error.
pub fn load(workspace: &Workspace, home: Option<&std::path::Path>) -> io::Result<Vec<Template>> {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        use crate::tools::anchored::Dir;
        let mut commands = BTreeMap::new();
        fn add(dir: Dir, commands: &mut BTreeMap<String, Template>) -> io::Result<()> {
            let Some(dir) = dir.optional_workspace_child("commands")? else { return Ok(()); };
            let names = dir.names(258)?;
            if names.iter().any(|n| n == "[listing limit reached]") { return Err(io::Error::other("too many user command files")); }
            let mut total = 0usize;
            for file in names {
                let Some(name) = file.strip_suffix(".md").filter(|name| valid_name(name)) else { continue; };
                if commands.contains_key(name) { continue; }
                if let Some(file) = dir.workspace_file(&file)? {
                    let source = text(file, MAX_PROMPT_BYTES)?;
                    total += source.len();
                    if total > 1024 * 1024 { return Err(io::Error::other("user commands exceed the total size limit")); }
                    commands.insert(name.into(), Template::parse(name, source));
                }
            }
            Ok(())
        }
        add(workspace.state_dir()?, &mut commands)?;
        if let Some(home) = home {
            match std::fs::symlink_metadata(home) {
                Ok(_) => add(Dir::open_root(home)?, &mut commands)?,
                Err(e) if e.kind() == io::ErrorKind::NotFound => (),
                Err(e) => return Err(e),
            }
        }
        Ok(commands.into_values().collect())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    { let _ = (workspace, home); Ok(Vec::new()) }
}

pub fn home(state: Option<&crate::tools::StateLocation>) -> Option<PathBuf> {
    use crate::tools::StateLocation;
    match state {
        Some(StateLocation::Home(path)) => Some(path.clone()),
        _ => match StateLocation::from_env().ok()? { StateLocation::Home(path) => Some(path), StateLocation::Project => None },
    }
}

#[cfg(test)]
mod tests;
