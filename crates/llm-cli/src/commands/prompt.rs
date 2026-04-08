use std::io::{IsTerminal, Write};

use clap::{ArgAction, Args};
use futures::StreamExt;
use llm_core::{
    Chunk, Config, KeyStore, Paths, Prompt, Provider, Response,
    collect_text, collect_tool_calls, collect_usage, resolve_key,
};

use super::providers;
use super::schemas::make_schema_id;
use super::tools::{BuiltinToolRegistry, CliToolExecutor};

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

    /// Enable a built-in tool (repeatable)
    #[arg(short = 'T', long = "tool", action = ArgAction::Append)]
    pub tool: Vec<String>,

    /// Maximum number of tool call chain iterations
    #[arg(long, default_value = "5")]
    pub chain_limit: usize,

    /// Print tool calls and results to stderr
    #[arg(long)]
    pub tools_debug: bool,

    /// Prompt for approval before each tool execution
    #[arg(long)]
    pub tools_approve: bool,

    /// Schema for structured output (JSON, file path, or DSL)
    #[arg(long)]
    pub schema: Option<String>,

    /// Wrap schema in array structure for multiple items
    #[arg(long)]
    pub schema_multi: bool,
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

    // Resolve tools if specified
    let mut tools = Vec::new();
    if !args.tool.is_empty() {
        let registry = BuiltinToolRegistry::new();
        for name in &args.tool {
            match registry.get(name) {
                Some(tool) => tools.push(tool.clone()),
                None => {
                    return Err(llm_core::LlmError::Config(format!(
                        "unknown tool: {name}"
                    )));
                }
            }
        }
    }

    // Resolve schema if specified
    let mut schema: Option<serde_json::Value> = None;
    let mut schema_id: Option<String> = None;
    if let Some(schema_input) = &args.schema {
        let resolved = resolve_schema(schema_input)?;
        let resolved = if args.schema_multi {
            llm_core::multi_schema(resolved)
        } else {
            resolved
        };
        schema_id = Some(make_schema_id(&resolved));
        schema = Some(resolved);
    }

    // Build prompt
    let mut prompt = Prompt::new(&text);
    if let Some(system) = &args.system {
        prompt = prompt.with_system(system);
    }
    if !tools.is_empty() {
        prompt = prompt.with_tools(tools);
    }
    if let Some(s) = &schema {
        prompt = prompt.with_schema(s.clone());
    }

    let stream_mode = !args.no_stream;
    let start = std::time::Instant::now();

    let (chunks, chain_tool_results) = if !args.tool.is_empty() {
        // Tool chain mode
        let executor = CliToolExecutor::new(args.tools_debug, args.tools_approve);
        let mut stdout = std::io::stdout().lock();

        let result = llm_core::chain(
            provider,
            &model_id,
            prompt,
            Some(&key),
            stream_mode,
            &executor,
            args.chain_limit,
            &mut |chunk| {
                if let Chunk::Text(t) = chunk {
                    write!(stdout, "{t}").ok();
                    stdout.flush().ok();
                }
            },
        )
        .await?;
        (result.chunks, result.tool_results)
    } else {
        // Normal mode (no tools)
        let response_stream =
            provider
                .execute(&model_id, &prompt, Some(&key), stream_mode)
                .await?;

        let mut chunks = Vec::new();
        let mut stream = std::pin::pin!(response_stream);
        let mut stdout = std::io::stdout().lock();

        while let Some(result) = stream.next().await {
            let chunk = result?;
            if let Chunk::Text(t) = &chunk {
                write!(stdout, "{t}").ok();
                stdout.flush().ok();
            }
            chunks.push(chunk);
        }
        (chunks, Vec::new())
    };

    let duration_ms = start.elapsed().as_millis() as u64;

    // Show usage on stderr if requested
    let usage_data = collect_usage(&chunks);
    if args.usage
        && let Some(u) = &usage_data
    {
        let input = u.input.unwrap_or(0);
        let output = u.output.unwrap_or(0);
        eprintln!("Token usage: {input} input, {output} output");
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
            tool_results: chain_tool_results,
            attachments: Vec::new(),
            schema,
            schema_id,
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

/// Resolve schema input: try JSON literal, then file path, then DSL.
fn resolve_schema(input: &str) -> llm_core::Result<serde_json::Value> {
    // 1. Try JSON literal
    if let Ok(schema) = serde_json::from_str(input) {
        return Ok(schema);
    }
    // 2. Try file path
    let path = std::path::Path::new(input);
    if path.exists() {
        let content = std::fs::read_to_string(path)?;
        return serde_json::from_str(&content)
            .map_err(|e| llm_core::LlmError::Config(format!("invalid JSON in schema file: {e}")));
    }
    // 3. Try DSL
    llm_core::parse_schema_dsl(input)
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
