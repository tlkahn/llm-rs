use std::io::Write;

use clap::{ArgAction, Args};
use futures::StreamExt;
use llm_core::{
    ChainEvent, Chunk, Config, KeyStore, Message, Paths, Prompt, Provider, Response, RetryConfig,
    Usage, collect_text, collect_tool_calls, collect_usage, resolve_key,
};

use super::prompt::find_provider;
use super::providers;
use super::tools::{BuiltinToolRegistry, CliToolExecutor};
use crate::retry::RetryProvider;
use crate::subprocess::tool::ExternalToolExecutor;

#[derive(Args)]
pub struct ChatArgs {
    /// Model to use
    #[arg(short, long)]
    pub model: Option<String>,

    /// System prompt
    #[arg(short, long)]
    pub system: Option<String>,

    /// Enable a tool (repeatable, built-in or external)
    #[arg(short = 'T', long = "tool", action = ArgAction::Append)]
    pub tool: Vec<String>,

    /// Maximum number of tool call chain iterations per turn
    #[arg(long, default_value = "5")]
    pub chain_limit: usize,

    /// Set a model option (repeatable): -o temperature 0.7
    #[arg(short = 'o', long = "option", num_args = 2, value_names = ["KEY", "VALUE"], action = ArgAction::Append)]
    pub option: Vec<String>,

    /// Verbose chain loop output (-v summary, -vv full messages). Implies --tools-debug.
    #[arg(short, long, action = ArgAction::Count)]
    pub verbose: u8,

    /// Maximum number of retries for transient HTTP errors (429, 5xx)
    #[arg(long)]
    pub retries: Option<u32>,

    /// Force sequential tool dispatch (default: parallel within a turn).
    #[arg(long)]
    pub sequential_tools: bool,

    /// Cap parallel tool dispatch concurrency. `None` = unlimited.
    #[arg(long)]
    pub max_parallel_tools: Option<usize>,
}

pub async fn run(args: &ChatArgs) -> llm_core::Result<()> {
    let paths = Paths::resolve()?;
    let config = Config::load(&paths.config_file())?;
    let key_store = KeyStore::load(&paths.keys_file())?;

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

    // Resolve key (skip if provider doesn't need one)
    let key = if provider.needs_key().is_some() {
        Some(resolve_key(
            None,
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

    // Create executor once, reuse across all turns (verbose implies debug)
    let debug = args.verbose > 0;
    let executor = {
        let e = CliToolExecutor::new(debug, false);
        match external_executor {
            Some(ext) => e.with_external(ext),
            None => e,
        }
    };

    // Resolve parallel tool dispatch config for this chat session.
    let parallel_config = llm_core::ParallelConfig {
        enabled: !args.sequential_tools,
        max_concurrent: args.max_parallel_tools,
    };

    eprintln!("Chatting with {model_id} (Ctrl-D to exit)");

    let mut editor = rustyline::DefaultEditor::new()
        .map_err(|e| llm_core::LlmError::Io(std::io::Error::other(e)))?;
    let mut messages: Vec<Message> = Vec::new();

    // Create log store and conversation ID
    let store = if config.logging {
        Some(llm_store::LogStore::open(&paths.logs_dir())?)
    } else {
        None
    };
    let mut conversation_id: Option<String> = None;
    let mut session_usage: Option<Usage> = None;

    loop {
        let input = match editor.readline("> ") {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(rustyline::error::ReadlineError::Interrupted) => break,
            Err(e) => {
                return Err(llm_core::LlmError::Io(std::io::Error::other(e)));
            }
        };

        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }
        if input == "/exit" {
            break;
        }

        let _ = editor.add_history_entry(&input);

        // Add user message to history
        messages.push(Message::user(&input));

        // Build prompt
        let mut prompt = Prompt::new(&input).with_messages(messages.clone());
        if let Some(system) = &args.system {
            prompt = prompt.with_system(system);
        }
        for (k, v) in &options {
            prompt = prompt.with_option(k, v.clone());
        }
        if !tools.is_empty() {
            prompt = prompt.with_tools(tools.clone());
        }

        let start = std::time::Instant::now();

        let (chunks, chain_tool_results, turn_total_usage) = if !tools.is_empty() {
            let mut stdout = std::io::stdout().lock();

            let verbose = args.verbose;
            let chain_limit = args.chain_limit;
            let mut on_event_fn = move |event: &ChainEvent| {
                super::prompt::format_chain_event(event, verbose, chain_limit);
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
                true,
                &executor,
                args.chain_limit,
                &mut |chunk| {
                    if let Chunk::Text(t) = chunk {
                        write!(stdout, "{t}").ok();
                        stdout.flush().ok();
                    }
                },
                on_event,
                None,
                parallel_config.clone(),
            )
            .await?;
            (result.chunks, result.tool_results, result.total_usage)
        } else {
            let response_stream = provider
                .execute(&model_id, &prompt, key.as_deref(), true)
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
            let usage_data = collect_usage(&chunks);
            (chunks, Vec::new(), usage_data)
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        let response_text = collect_text(&chunks);

        // Print newline after response
        println!();

        // Add assistant message to history
        let assistant_tool_calls = collect_tool_calls(&chunks);
        if assistant_tool_calls.is_empty() {
            messages.push(Message::assistant(&response_text));
        } else {
            messages.push(Message::assistant_with_tool_calls(
                &response_text,
                assistant_tool_calls.clone(),
            ));
            if !chain_tool_results.is_empty() {
                messages.push(Message::tool_results(chain_tool_results.clone()));
            }
        }

        // Accumulate session-wide usage
        let turn_usage = turn_total_usage.clone().or_else(|| collect_usage(&chunks));
        if let Some(tu) = &turn_usage {
            session_usage = Some(match &session_usage {
                Some(s) => s.add(tu),
                None => tu.clone(),
            });
        }

        // Log turn
        if let Some(store) = &store {
            let response = Response {
                id: ulid::Ulid::new().to_string().to_lowercase(),
                model: model_id.clone(),
                prompt: input.clone(),
                system: args.system.clone(),
                response: response_text,
                options: options.clone(),
                usage: turn_usage.clone(),
                tool_calls: assistant_tool_calls,
                tool_results: chain_tool_results,
                attachments: Vec::new(),
                schema: None,
                schema_id: None,
                duration_ms,
                datetime: chrono::Utc::now().to_rfc3339(),
            };
            match store.log_response(conversation_id.as_deref(), &model_id, &response) {
                Ok(cid) => conversation_id = Some(cid),
                Err(e) => eprintln!("Warning: failed to log: {e}"),
            }
        }
    }

    // Print session usage summary on exit
    if let Some(u) = &session_usage {
        let input = u.input.unwrap_or(0);
        let output = u.output.unwrap_or(0);
        eprintln!("Session usage: {input} input, {output} output, {} total", input + output);
    }

    Ok(())
}

