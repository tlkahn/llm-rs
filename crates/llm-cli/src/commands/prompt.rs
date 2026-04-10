use std::io::{IsTerminal, Write};

use clap::{ArgAction, Args};
use futures::StreamExt;
use llm_core::{
    ChainEvent, Chunk, Config, KeyStore, Message, Paths, Prompt, Provider, Response, RetryConfig,
    Role, collect_text, collect_tool_calls, collect_usage, resolve_key,
};

use super::providers;
use super::schemas::make_schema_id;
use super::tools::{BuiltinToolRegistry, CliToolExecutor};
use crate::retry::RetryProvider;
use crate::subprocess::tool::ExternalToolExecutor;

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

    /// Continue the most recent conversation
    #[arg(short = 'c', long = "continue")]
    pub continue_last: bool,

    /// Continue a specific conversation by ID
    #[arg(long)]
    pub cid: Option<String>,

    /// Load messages from a JSON file (or - for stdin)
    #[arg(long)]
    pub messages: Option<String>,

    /// Output response as a JSON envelope
    #[arg(long)]
    pub json: bool,

    /// Set a model option (repeatable): -o temperature 0.7
    #[arg(short = 'o', long = "option", num_args = 2, value_names = ["KEY", "VALUE"], action = ArgAction::Append)]
    pub option: Vec<String>,

    /// Verbose chain loop output (-v summary, -vv full messages). Implies --tools-debug.
    #[arg(short, long, action = ArgAction::Count)]
    pub verbose: u8,

    /// Maximum number of retries for transient HTTP errors (429, 5xx)
    #[arg(long)]
    pub retries: Option<u32>,
}

pub async fn run(args: &PromptArgs) -> llm_core::Result<()> {
    let paths = Paths::resolve()?;
    let config = Config::load(&paths.config_file())?;
    let key_store = KeyStore::load(&paths.keys_file())?;

    // Resolve prompt text: from arg or stdin (allow empty when continuing or using --messages)
    let has_messages_input = args.messages.is_some();
    let messages_from_stdin = args.messages.as_deref() == Some("-");
    let allow_empty = args.continue_last || args.cid.is_some() || has_messages_input;
    let text = resolve_prompt_text(&args.text, allow_empty, messages_from_stdin)?;

    // Resolve model
    let effective_default = config.effective_default_model();
    let model_input = args.model.as_deref().unwrap_or(&effective_default);
    let model_id = config.resolve_model(model_input).to_string();

    // Build options (config defaults + CLI -o overrides)
    let options = super::build_options(&config, &model_id, &args.option);

    // Find the provider for this model
    let all_providers = providers().await;
    let (provider, _model_info) = find_provider(&all_providers, &model_id)?;

    // Wrap provider with retry logic if --retries is set
    let retry_provider;
    let provider: &dyn Provider = if let Some(retries) = args.retries {
        retry_provider = RetryProvider::new(
            provider,
            RetryConfig { max_retries: retries, ..Default::default() },
        );
        &retry_provider
    } else {
        provider
    };

    // Resolve key (skip if provider doesn't need one and none explicitly given)
    let key = if provider.needs_key().is_some() || args.key.is_some() {
        Some(resolve_key(
            args.key.as_deref(),
            &key_store,
            provider.needs_key().unwrap_or(""),
            provider.key_env_var(),
        )?)
    } else {
        None
    };

    // Resolve tools if specified (check builtins first, then external)
    let mut tools = Vec::new();
    let mut need_external: Vec<String> = Vec::new();
    if !args.tool.is_empty() {
        let registry = BuiltinToolRegistry::new();
        for name in &args.tool {
            match registry.get(name) {
                Some(tool) => tools.push(tool.clone()),
                None => need_external.push(name.clone()),
            }
        }
    }

    let external_executor = if !need_external.is_empty() || !args.tool.is_empty() {
        let ext = ExternalToolExecutor::discover().await?;
        // Resolve any tools not found in builtins
        for name in &need_external {
            match ext.get_tool(name) {
                Some((_, tool)) => tools.push(tool.clone()),
                None => {
                    return Err(llm_core::LlmError::Config(format!(
                        "unknown tool: {name}"
                    )));
                }
            }
        }
        Some(ext)
    } else {
        None
    };

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

    // Load conversation history if continuing
    let mut conversation_id: Option<String> = None;
    let mut history_messages = Vec::new();

    if args.continue_last || args.cid.is_some() {
        let logs_dir = paths.logs_dir();
        let cid = if let Some(id) = &args.cid {
            id.clone()
        } else {
            llm_store::latest_conversation_id(&logs_dir)?
                .ok_or_else(|| llm_core::LlmError::Store("no conversations found".into()))?
        };

        let store = llm_store::LogStore::open(&logs_dir)?;
        let (_, responses) = store.read_conversation(&cid)?;
        history_messages = llm_store::reconstruct_messages(&responses);
        conversation_id = Some(cid);
    }

    // Load messages from --messages flag
    if let Some(messages_src) = &args.messages {
        if args.continue_last || args.cid.is_some() {
            return Err(llm_core::LlmError::Config(
                "--messages cannot be combined with -c/--cid".into(),
            ));
        }
        let json_str = if messages_src == "-" {
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
            buf
        } else {
            std::fs::read_to_string(messages_src)?
        };
        history_messages = serde_json::from_str::<Vec<llm_core::Message>>(&json_str)
            .map_err(|e| llm_core::LlmError::Config(format!("invalid messages JSON: {e}")))?;
    }

    // Build prompt
    let mut prompt = Prompt::new(&text);
    if let Some(system) = &args.system {
        prompt = prompt.with_system(system);
    }
    for (k, v) in &options {
        prompt = prompt.with_option(k, v.clone());
    }
    if !tools.is_empty() {
        prompt = prompt.with_tools(tools);
    }
    if let Some(s) = &schema {
        prompt = prompt.with_schema(s.clone());
    }

    // Append current user message to history and set on prompt
    if !history_messages.is_empty() {
        if !text.is_empty() {
            history_messages.push(llm_core::Message::user(&text));
        }
        prompt = prompt.with_messages(history_messages);
    }

    let stream_mode = !args.no_stream && !args.json;
    let start = std::time::Instant::now();
    let json_output = args.json;

    let (chunks, chain_tool_results, chain_total_usage) = if !args.tool.is_empty() {
        // Tool chain mode — verbose > 0 implies tools-debug
        let debug = args.tools_debug || args.verbose > 0;
        let mut executor = CliToolExecutor::new(debug, args.tools_approve);
        if let Some(ext) = external_executor {
            executor = executor.with_external(ext);
        }
        let executor = executor;
        let mut stdout = std::io::stdout().lock();

        let verbose = args.verbose;
        let chain_limit = args.chain_limit;
        let mut on_event_fn = move |event: &ChainEvent| {
            format_chain_event(event, verbose, chain_limit);
        };
        let on_event: Option<&mut dyn FnMut(&ChainEvent)> = if verbose > 0 {
            Some(&mut on_event_fn)
        } else {
            None
        };

        let result = llm_core::chain(
            provider,
            &model_id,
            prompt,
            key.as_deref(),
            stream_mode,
            &executor,
            args.chain_limit,
            &mut |chunk| {
                if !json_output
                    && let Chunk::Text(t) = chunk
                {
                    write!(stdout, "{t}").ok();
                    stdout.flush().ok();
                }
            },
            on_event,
            None,
        )
        .await?;
        (result.chunks, result.tool_results, result.total_usage)
    } else {
        // Normal mode (no tools)
        let response_stream =
            provider
                .execute(&model_id, &prompt, key.as_deref(), stream_mode)
                .await?;

        let mut chunks = Vec::new();
        let mut stream = std::pin::pin!(response_stream);
        let mut stdout = std::io::stdout().lock();

        while let Some(result) = stream.next().await {
            let chunk = result?;
            if !json_output
                && let Chunk::Text(t) = &chunk
            {
                write!(stdout, "{t}").ok();
                stdout.flush().ok();
            }
            chunks.push(chunk);
        }
        let usage_data = collect_usage(&chunks);
        (chunks, Vec::new(), usage_data)
    };

    let duration_ms = start.elapsed().as_millis() as u64;

    let response_text = collect_text(&chunks);
    let tool_calls_data = collect_tool_calls(&chunks);

    // Show usage on stderr if requested (total_usage for chain, last usage for non-chain)
    let usage_data = chain_total_usage.or_else(|| collect_usage(&chunks));
    if args.usage
        && let Some(u) = &usage_data
    {
        let input = u.input.unwrap_or(0);
        let output = u.output.unwrap_or(0);
        eprintln!("Token usage: {input} input, {output} output");
    }

    // Log if enabled
    let logged_conv_id = if !args.no_log && config.logging {
        let response = Response {
            id: ulid::Ulid::new().to_string().to_lowercase(),
            model: model_id.clone(),
            prompt: text.clone(),
            system: args.system.clone(),
            response: response_text.clone(),
            options: options.clone(),
            usage: usage_data.clone(),
            tool_calls: tool_calls_data.clone(),
            tool_results: chain_tool_results,
            attachments: Vec::new(),
            schema,
            schema_id,
            duration_ms,
            datetime: chrono::Utc::now().to_rfc3339(),
        };
        let store = llm_store::LogStore::open(&paths.logs_dir())?;
        let cid = store.log_response(conversation_id.as_deref(), &model_id, &response)?;
        Some(cid)
    } else {
        conversation_id
    };

    // JSON output envelope
    if json_output {
        let mut envelope = serde_json::json!({
            "model": model_id,
            "content": response_text,
        });
        if let Some(cid) = &logged_conv_id {
            envelope["conversation_id"] = serde_json::json!(cid);
        }
        if !tool_calls_data.is_empty() {
            envelope["tool_calls"] = serde_json::json!(tool_calls_data);
        }
        if let Some(u) = &usage_data {
            envelope["usage"] = serde_json::json!({
                "input": u.input.unwrap_or(0),
                "output": u.output.unwrap_or(0),
            });
        }
        envelope["duration_ms"] = serde_json::json!(duration_ms);
        println!(
            "{}",
            serde_json::to_string_pretty(&envelope)
                .map_err(|e| llm_core::LlmError::Store(e.to_string()))?
        );
    }

    Ok(())
}

pub(crate) fn resolve_prompt_text(
    arg_text: &Option<String>,
    allow_empty: bool,
    skip_stdin: bool,
) -> llm_core::Result<String> {
    let stdin_text = if !skip_stdin && !std::io::stdin().is_terminal() {
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
        (None, None) if allow_empty => Ok(String::new()),
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

/// Format a ChainEvent for stderr output.
pub fn format_chain_event(event: &ChainEvent, verbose: u8, _chain_limit: usize) {
    match event {
        ChainEvent::IterationStart { iteration, limit, messages } => {
            let summary = format_message_summary(messages);
            eprintln!(
                "[chain] Iteration {iteration}/{limit} | {} message{} [{summary}]",
                messages.len(),
                if messages.len() == 1 { "" } else { "s" },
            );
            if verbose >= 2 {
                eprintln!("[chain] Messages:");
                let json = serde_json::to_string_pretty(messages).unwrap_or_default();
                eprintln!("{json}");
            }
        }
        ChainEvent::IterationEnd { iteration, usage, cumulative_usage, tool_calls } => {
            let usage_str = if let Some(u) = usage {
                let base = format!(
                    "usage: {} input, {} output",
                    u.input.unwrap_or(0),
                    u.output.unwrap_or(0),
                );
                if let Some(cum) = cumulative_usage {
                    format!(
                        "{base} (cumulative: {} input, {} output)",
                        cum.input.unwrap_or(0),
                        cum.output.unwrap_or(0),
                    )
                } else {
                    base
                }
            } else {
                "no usage data".into()
            };
            eprintln!(
                "[chain] Iteration {iteration} complete | {usage_str} | {} tool call(s)",
                tool_calls.len(),
            );
        }
        ChainEvent::BudgetExhausted { cumulative_usage, budget } => {
            eprintln!(
                "[budget] Budget exhausted: {}/{budget} tokens used",
                cumulative_usage.total(),
            );
        }
    }
}

/// Summarize a message list as e.g. "user, assistant+tools(1), tool(1)"
pub(crate) fn format_message_summary(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|m| match m.role {
            Role::User => "user".to_string(),
            Role::Assistant => {
                if m.tool_calls.is_empty() {
                    "assistant".to_string()
                } else {
                    format!("assistant+tools({})", m.tool_calls.len())
                }
            }
            Role::Tool => format!("tool({})", m.tool_results.len()),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn find_provider<'a>(
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
