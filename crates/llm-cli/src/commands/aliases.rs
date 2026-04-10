use clap::Subcommand;
use llm_core::{Config, LlmError, Paths, Result};

#[derive(Subcommand)]
pub enum AliasesCommand {
    /// Set an alias for a model
    Set {
        /// Alias name
        alias: String,
        /// Model ID the alias points to
        model: String,
    },
    /// Show what model an alias points to
    Show {
        /// Alias name
        alias: String,
    },
    /// List all aliases
    List,
    /// Remove an alias
    Remove {
        /// Alias name
        alias: String,
    },
    /// Show the path to the config file
    Path,
}

pub fn run(cmd: &AliasesCommand) -> Result<()> {
    let paths = Paths::resolve()?;
    let config_path = paths.config_file();

    match cmd {
        AliasesCommand::Set { alias, model } => {
            let mut config = Config::load(&config_path)?;
            config.set_alias(alias, model);
            config.save(&config_path)?;
        }
        AliasesCommand::Show { alias } => {
            let config = Config::load(&config_path)?;
            match config.aliases.get(alias.as_str()) {
                Some(model) => println!("{alias}: {model}"),
                None => {
                    return Err(LlmError::Config(format!("alias '{alias}' not found")));
                }
            }
        }
        AliasesCommand::List => {
            let config = Config::load(&config_path)?;
            if config.aliases.is_empty() {
                println!("No aliases set");
            } else {
                let mut entries: Vec<_> = config.aliases.iter().collect();
                entries.sort_by_key(|(k, _)| *k);
                for (alias, model) in entries {
                    println!("{alias}: {model}");
                }
            }
        }
        AliasesCommand::Remove { alias } => {
            let mut config = Config::load(&config_path)?;
            if !config.remove_alias(alias) {
                return Err(LlmError::Config(format!("alias '{alias}' not found")));
            }
            config.save(&config_path)?;
        }
        AliasesCommand::Path => {
            println!("{}", config_path.display());
        }
    }
    Ok(())
}
