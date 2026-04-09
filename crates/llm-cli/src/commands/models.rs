use std::io::Write;

use clap::Subcommand;
use llm_core::{Config, LlmError, Paths, Result};

use super::providers;

#[derive(Subcommand)]
pub enum ModelsCommand {
    /// List available models
    List,
    /// Show or set the default model
    Default {
        /// Model ID to set as default (omit to show current)
        model: Option<String>,
    },
}

pub async fn run(cmd: &ModelsCommand) -> Result<()> {
    let paths = Paths::resolve()?;

    match cmd {
        ModelsCommand::List => {
            let providers = providers().await;
            for provider in &providers {
                for model in provider.models() {
                    println!("{} ({})", model.id, provider.id());
                }
            }
        }
        ModelsCommand::Default { model: None } => {
            let config = Config::load(&paths.config_file())?;
            println!("{}", config.effective_default_model());
        }
        ModelsCommand::Default { model: Some(model) } => {
            set_default_model(&paths.config_file(), model)?;
        }
    }
    Ok(())
}

fn set_default_model(config_path: &std::path::Path, model: &str) -> Result<()> {
    // Read existing config as a raw TOML table to preserve other fields
    let mut table: toml::Table = match std::fs::read_to_string(config_path) {
        Ok(contents) => toml::from_str(&contents)
            .map_err(|e| LlmError::Config(format!("invalid config.toml: {e}")))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => toml::Table::new(),
        Err(e) => return Err(LlmError::Io(e)),
    };

    table.insert(
        "default_model".to_string(),
        toml::Value::String(model.to_string()),
    );

    let contents = toml::to_string(&table)
        .map_err(|e| LlmError::Config(format!("failed to serialize config: {e}")))?;

    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = std::fs::File::create(config_path)?;
    file.write_all(contents.as_bytes())?;

    Ok(())
}
