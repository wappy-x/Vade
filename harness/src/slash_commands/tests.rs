use super::*;
use crate::test_workspace::Fixture;

#[test]
fn project_wins_and_descriptions_do_not_enter_the_prompt() {
    let project = Fixture::new();
    let home = Fixture::new();
    let workspace = project.workspace();
    workspace.state_dir().unwrap();
    std::fs::create_dir(project.0.join(".vade/commands")).unwrap();
    std::fs::create_dir(home.0.join("commands")).unwrap();
    home.write("commands/review.md", "global $ARGUMENTS");
    home.write("commands/shared.md", "shared");
    project.write(".vade/commands/review.md", "description: Review the diff\nproject $ARGUMENTS\nAgain: $ARGUMENTS");
    let commands = load(&workspace, Some(&home.0)).unwrap();
    assert_eq!(commands.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["review", "shared"]);
    assert_eq!(commands[0].description, "Review the diff");
    assert_eq!(commands[0].expand("a $HOME /help").unwrap(), "project a $HOME /help\nAgain: a $HOME /help");
    assert_eq!(commands[0].expand("").unwrap(), "project \nAgain: ");
    project.write(".vade/commands/review.md", "changed");
    assert_eq!(load(&workspace, Some(&home.0)).unwrap()[0].expand("").unwrap(), "changed");
    // An overridden global file cannot prevent use of the project command.
    home.write("commands/review.md", &"a".repeat(MAX_PROMPT_BYTES + 1));
    assert_eq!(load(&workspace, Some(&home.0)).unwrap()[0].expand("").unwrap(), "changed");
}

#[test]
fn template_limits_names_and_links_are_checked() {
    let template = Template::parse("name", "$ARGUMENTS$ARGUMENTS".into());
    assert!(template.expand(&"a".repeat(MAX_PROMPT_BYTES)).is_err());
    assert!(valid_name("review-2_a"));
    for name in ["", "../escape", "a/b", "name.md", "a b", "é"] { assert!(!valid_name(name)); }
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    assert!(load(&workspace, None).unwrap().is_empty());
    std::fs::create_dir(fixture.0.join(".vade/commands")).unwrap();
    fixture.write(".vade/commands/huge.md", &"a".repeat(MAX_PROMPT_BYTES + 1));
    assert!(load(&workspace, None).is_err());
    std::fs::remove_file(fixture.0.join(".vade/commands/huge.md")).unwrap();
    #[cfg(unix)]
    {
        fixture.write("outside.md", "secret");
        std::os::unix::fs::symlink("../../outside.md", fixture.0.join(".vade/commands/link.md")).unwrap();
        assert!(load(&workspace, None).is_err());
    }
}
