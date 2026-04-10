use clap::Subcommand;
use llm_core::{Config, LlmError, Paths, Result, parse_option_value};

#[derive(Subcommand)]
pub enum OptionsCommand {
    /// Set a model option
    Set {
        /// Model name (e.g. "gpt-4o-mini")
        model: String,
        /// Option key (e.g. "temperature")
        key: String,
        /// Option value (e.g. "0.7")
        value: String,
    },
    /// Get option(s) for a model
    Get {
        /// Model name
        model: String,
        /// Option key (omit to show all options for the model)
        key: Option<String>,
    },
    /// List all model options
    List,
    /// Clear option(s) for a model
    Clear {
        /// Model name
        model: String,
        /// Option key (omit to clear all options for the model)
        key: Option<String>,
    },
}

pub fn run(cmd: &OptionsCommand) -> Result<()> {
    let paths = Paths::resolve()?;
    let config_path = paths.config_file();

    match cmd {
        OptionsCommand::Set { model, key, value } => {
            let mut config = Config::load(&config_path)?;
            config.set_option(model, key, parse_option_value(value));
            config.save(&config_path)?;
        }
        OptionsCommand::Get { model, key } => {
            let config = Config::load(&config_path)?;
            if let Some(key) = key {
                let opts = config.model_options(model);
                match opts.get(key.as_str()) {
                    Some(v) => println!("{key}: {v}"),
                    None => {
                        return Err(LlmError::Config(format!(
                            "no option '{key}' set for model '{model}'"
                        )));
                    }
                }
            } else {
                let opts = config.model_options(model);
                if opts.is_empty() {
                    println!("No options set for {model}");
                } else {
                    let mut keys: Vec<_> = opts.keys().collect();
                    keys.sort();
                    for k in keys {
                        println!("{k}: {}", opts[k]);
                    }
                }
            }
        }
        OptionsCommand::List => {
            let config = Config::load(&config_path)?;
            if config.options.is_empty() {
                println!("No options set");
            } else {
                let mut models: Vec<_> = config.options.keys().collect();
                models.sort();
                for model in models {
                    println!("{model}:");
                    let opts = &config.options[model];
                    let mut keys: Vec<_> = opts.keys().collect();
                    keys.sort();
                    for k in keys {
                        println!("  {k}: {}", opts[k]);
                    }
                }
            }
        }
        OptionsCommand::Clear { model, key } => {
            let mut config = Config::load(&config_path)?;
            if let Some(key) = key {
                if !config.clear_option(model, key) {
                    return Err(LlmError::Config(format!(
                        "no option '{key}' set for model '{model}'"
                    )));
                }
            } else {
                if !config.clear_model_options(model) {
                    return Err(LlmError::Config(format!(
                        "no options set for model '{model}'"
                    )));
                }
            }
            config.save(&config_path)?;
        }
    }
    Ok(())
}
