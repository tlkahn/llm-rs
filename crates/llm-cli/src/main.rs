mod app;
mod commands;
mod subprocess;

use std::ffi::OsString;
use std::io::IsTerminal;

use clap::Parser;
use llm_core::LlmError;

use app::{Cli, Commands};

#[tokio::main]
async fn main() {
    let args = rewrite_args();
    let cli = Cli::parse_from(args);

    let result = match cli.command {
        Some(Commands::Prompt(args)) => commands::prompt::run(&args).await,
        Some(Commands::Chat(args)) => commands::chat::run(&args).await,
        Some(Commands::Keys { command }) => commands::keys::run(&command),
        Some(Commands::Models { command }) => commands::models::run(&command).await,
        Some(Commands::Logs { command }) => commands::logs::run(&command),
        Some(Commands::Tools { command }) => commands::tools::run(&command).await,
        Some(Commands::Schemas { command }) => commands::schemas::run(&command),
        Some(Commands::Plugins { command }) => commands::plugins::run(&command).await,
        Some(Commands::Options { command }) => commands::options::run(&command),
        None => Ok(()),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(exit_code(&e));
    }
}

/// Rewrite args to insert "prompt" as the default subcommand when appropriate.
///
/// Rules:
/// - If no args and stdin is piped: insert "prompt"
/// - If first real arg is not a known subcommand or global flag: insert "prompt"
fn rewrite_args() -> Vec<OsString> {
    let mut args: Vec<OsString> = std::env::args_os().collect();

    if args.len() <= 1 {
        // Just the binary name, no args
        if !std::io::stdin().is_terminal() {
            args.insert(1, "prompt".into());
        }
        return args;
    }

    let first = args[1].to_str().unwrap_or("");
    if should_insert_prompt(first) {
        args.insert(1, "prompt".into());
    }

    args
}

fn should_insert_prompt(first_arg: &str) -> bool {
    let known = [
        "prompt", "keys", "models", "logs", "tools", "schemas", "chat", "plugins",
        "options", "help", "--help", "-h", "--version", "-V",
    ];
    !known.contains(&first_arg)
}

fn exit_code(err: &LlmError) -> i32 {
    match err {
        LlmError::Io(_) | LlmError::Store(_) => 1,
        LlmError::Model(_) | LlmError::NeedsKey(_) | LlmError::Config(_) => 2,
        LlmError::Provider(_) => 3,
    }
}
