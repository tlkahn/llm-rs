use clap::Subcommand;
use llm_core::{Paths, Result};
use llm_store::{list_conversations, latest_conversation_id, LogStore};

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
    },
}

pub fn run(cmd: &LogsCommand) -> Result<()> {
    let paths = Paths::resolve()?;
    let logs_dir = paths.logs_dir();

    match cmd {
        LogsCommand::List { json, response, count } => {
            if *response {
                return show_latest_response(&logs_dir);
            }

            let summaries = list_conversations(&logs_dir, *count)?;
            for summary in &summaries {
                if *json {
                    let line = serde_json::to_string(summary)
                        .map_err(|e| llm_core::LlmError::Store(e.to_string()))?;
                    println!("{line}");
                } else {
                    let name = summary.name.as_deref().unwrap_or("-");
                    println!("{} {} {} ({})", summary.id, summary.model, name, summary.created);
                }
            }
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
