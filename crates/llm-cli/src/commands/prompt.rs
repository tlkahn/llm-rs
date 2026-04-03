use std::io::{IsTerminal, Write};

use clap::Args;
use futures::StreamExt;
use llm_core::{
    Chunk, Config, KeyStore, Paths, Prompt, Provider, Response,
    collect_text, collect_tool_calls, collect_usage, resolve_key,
};

use super::providers;

#[derive(Args)]
pub struct PromptArgs {
    /// Prompt text
    pub text: Option<String>,

    /// Model to use
    #[arg(short, long)]
    pub model: Option<String>,

    /// System prompt
    #[arg(short, long)]
    pub system: Option<String>,

    /// Disable streaming
    #[arg(long)]
    pub no_stream: bool,

    /// Don't log this prompt/response
    #[arg(short = 'n', long)]
    pub no_log: bool,

    /// API key to use
    #[arg(long)]
    pub key: Option<String>,

    /// Show token usage on stderr
    #[arg(short, long)]
    pub usage: bool,
}

pub async fn run(args: &PromptArgs) -> llm_core::Result<()> {
    let paths = Paths::resolve()?;
    let config = Config::load(&paths.config_file())?;
    let key_store = KeyStore::load(&paths.keys_file())?;

    // Resolve prompt text: from arg or stdin
    let text = resolve_prompt_text(&args.text)?;

    // Resolve model
    let effective_default = config.effective_default_model();
    let model_input = args.model.as_deref().unwrap_or(&effective_default);
    let model_id = config.resolve_model(model_input).to_string();

    // Find the provider for this model
    let all_providers = providers();
    let (provider, _model_info) = find_provider(&all_providers, &model_id)?;

    // Resolve key
    let key = resolve_key(
        args.key.as_deref(),
        &key_store,
        provider.needs_key().unwrap_or(""),
        provider.key_env_var(),
    )?;

    // Build prompt
    let mut prompt = Prompt::new(&text);
    if let Some(system) = &args.system {
        prompt = prompt.with_system(system);
    }

    let stream_mode = !args.no_stream;
    let start = std::time::Instant::now();

    // Execute
    let response_stream = provider.execute(&model_id, &prompt, Some(&key), stream_mode).await?;

    // Collect chunks, printing text as it arrives
    let mut chunks = Vec::new();
    let mut stream = std::pin::pin!(response_stream);
    let mut stdout = std::io::stdout().lock();

    while let Some(result) = stream.next().await {
        let chunk = result?;
        match &chunk {
            Chunk::Text(t) => {
                write!(stdout, "{t}").ok();
                stdout.flush().ok();
            }
            _ => {}
        }
        chunks.push(chunk);
    }

    let duration_ms = start.elapsed().as_millis() as u64;

    // Show usage on stderr if requested
    let usage_data = collect_usage(&chunks);
    if args.usage {
        if let Some(u) = &usage_data {
            let input = u.input.unwrap_or(0);
            let output = u.output.unwrap_or(0);
            eprintln!("Token usage: {input} input, {output} output");
        }
    }

    // Log if enabled
    if !args.no_log && config.logging {
        let response = Response {
            id: ulid::Ulid::new().to_string().to_lowercase(),
            model: model_id.clone(),
            prompt: text.clone(),
            system: args.system.clone(),
            response: collect_text(&chunks),
            options: Default::default(),
            usage: usage_data,
            tool_calls: collect_tool_calls(&chunks),
            tool_results: Vec::new(),
            attachments: Vec::new(),
            schema: None,
            schema_id: None,
            duration_ms,
            datetime: chrono::Utc::now().to_rfc3339(),
        };
        let store = llm_store::LogStore::open(&paths.logs_dir())?;
        store.log_response(None, &model_id, &response)?;
    }

    Ok(())
}

fn resolve_prompt_text(arg_text: &Option<String>) -> llm_core::Result<String> {
    let stdin_text = if !std::io::stdin().is_terminal() {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        if buf.is_empty() { None } else { Some(buf) }
    } else {
        None
    };

    match (arg_text, stdin_text) {
        (Some(arg), Some(stdin)) => Ok(format!("{stdin}{arg}")),
        (Some(arg), None) => Ok(arg.clone()),
        (None, Some(stdin)) => Ok(stdin),
        (None, None) => Err(llm_core::LlmError::Config(
            "no prompt text provided — pass text as an argument or pipe via stdin".into(),
        )),
    }
}

fn find_provider<'a>(
    providers: &'a [Box<dyn Provider>],
    model_id: &str,
) -> llm_core::Result<(&'a dyn Provider, llm_core::ModelInfo)> {
    for provider in providers {
        for model in provider.models() {
            if model.id == model_id {
                return Ok((provider.as_ref(), model));
            }
        }
    }
    Err(llm_core::LlmError::Model(format!(
        "unknown model: {model_id}"
    )))
}
