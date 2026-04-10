use clap::{Parser, Subcommand};

use crate::commands;

#[derive(Parser)]
#[command(name = "llm", version, about = "Access large language models from the command line")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Send a prompt to a language model
    Prompt(commands::prompt::PromptArgs),
    /// Start an interactive chat session
    Chat(commands::chat::ChatArgs),
    /// Manage API keys
    Keys {
        #[command(subcommand)]
        command: commands::keys::KeysCommand,
    },
    /// List and manage models
    Models {
        #[command(subcommand)]
        command: commands::models::ModelsCommand,
    },
    /// View and manage conversation logs
    Logs {
        #[command(subcommand)]
        command: commands::logs::LogsCommand,
    },
    /// List and manage tools
    Tools {
        #[command(subcommand)]
        command: commands::tools::ToolsCommand,
    },
    /// Manage schemas for structured output
    Schemas {
        #[command(subcommand)]
        command: commands::schemas::SchemasCommand,
    },
    /// List providers and tools (compiled and external)
    Plugins {
        #[command(subcommand)]
        command: commands::plugins::PluginsCommand,
    },
    /// Manage model options
    Options {
        #[command(subcommand)]
        command: commands::options::OptionsCommand,
    },
}
