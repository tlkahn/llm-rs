use clap::Subcommand;
use llm_core::{Config, Paths, Result};
use llm_store::{ListOptions, LogStore, latest_conversation_id, list_conversations_filtered};

#[derive(Subcommand)]
pub enum LogsCommand {
    /// List recent conversations
    List {
        /// Output as JSON lines
        #[arg(long)]
        json: bool,
        /// Output the response text of the most recent conversation
        #[arg(short, long)]
        response: bool,
        /// Number of conversations to show
        #[arg(short = 'n', long, default_value = "10")]
        count: usize,
        /// Filter by model
        #[arg(short, long)]
        model: Option<String>,
        /// Full-text search
        #[arg(short, long)]
        query: Option<String>,
        /// Show token usage
        #[arg(short, long)]
        usage: bool,
    },
    /// Print the logs directory path
    Path,
    /// Show logging status
    Status,
    /// Enable logging
    On,
    /// Disable logging
    Off,
}

pub fn run(cmd: &LogsCommand) -> Result<()> {
    let paths = Paths::resolve()?;
    let logs_dir = paths.logs_dir();

    match cmd {
        LogsCommand::List {
            json,
            response,
            count,
            model,
            query,
            usage,
        } => {
            if *response {
                return show_latest_response(&logs_dir);
            }

            let options = ListOptions {
                model: model.as_deref(),
                query: query.as_deref(),
            };
            let summaries = list_conversations_filtered(&logs_dir, *count, &options)?;

            if *usage {
                // Need to read full conversations for usage data
                let store = LogStore::open(&logs_dir)?;
                for summary in &summaries {
                    if *json {
                        // Include usage in JSON output
                        let (_, responses) = store.read_conversation(&summary.id)?;
                        let total_input: u64 = responses
                            .iter()
                            .filter_map(|r| r.usage.as_ref())
                            .filter_map(|u| u.input)
                            .sum();
                        let total_output: u64 = responses
                            .iter()
                            .filter_map(|r| r.usage.as_ref())
                            .filter_map(|u| u.output)
                            .sum();
                        let mut val = serde_json::to_value(summary)
                            .map_err(|e| llm_core::LlmError::Store(e.to_string()))?;
                        val["usage"] = serde_json::json!({
                            "input": total_input,
                            "output": total_output,
                        });
                        let line = serde_json::to_string(&val)
                            .map_err(|e| llm_core::LlmError::Store(e.to_string()))?;
                        println!("{line}");
                    } else {
                        let (_, responses) = store.read_conversation(&summary.id)?;
                        let total_input: u64 = responses
                            .iter()
                            .filter_map(|r| r.usage.as_ref())
                            .filter_map(|u| u.input)
                            .sum();
                        let total_output: u64 = responses
                            .iter()
                            .filter_map(|r| r.usage.as_ref())
                            .filter_map(|u| u.output)
                            .sum();
                        let name = summary.name.as_deref().unwrap_or("-");
                        println!(
                            "{} {} {} ({}) [tokens: {}/{}]",
                            summary.id, summary.model, name, summary.created,
                            total_input, total_output
                        );
                    }
                }
            } else {
                for summary in &summaries {
                    if *json {
                        let line = serde_json::to_string(summary)
                            .map_err(|e| llm_core::LlmError::Store(e.to_string()))?;
                        println!("{line}");
                    } else {
                        let name = summary.name.as_deref().unwrap_or("-");
                        println!(
                            "{} {} {} ({})",
                            summary.id, summary.model, name, summary.created
                        );
                    }
                }
            }
        }
        LogsCommand::Path => {
            println!("{}", logs_dir.display());
        }
        LogsCommand::Status => {
            let config = Config::load(&paths.config_file())?;
            if config.logging {
                println!("Logging is enabled");
            } else {
                println!("Logging is disabled");
            }
        }
        LogsCommand::On => {
            let mut config = Config::load(&paths.config_file())?;
            config.logging = true;
            config.save(&paths.config_file())?;
            println!("Logging enabled");
        }
        LogsCommand::Off => {
            let mut config = Config::load(&paths.config_file())?;
            config.logging = false;
            config.save(&paths.config_file())?;
            println!("Logging disabled");
        }
    }
    Ok(())
}

fn show_latest_response(logs_dir: &std::path::Path) -> Result<()> {
    let latest_id = latest_conversation_id(logs_dir)?;
    let Some(id) = latest_id else {
        return Ok(());
    };

    let store = LogStore::open(logs_dir)?;
    let (_, responses) = store.read_conversation(&id)?;
    if let Some(last) = responses.last() {
        print!("{}", last.response);
    }
    Ok(())
}
