use std::io::Write;

use clap::{ArgAction, Args, Subcommand};
use llm_core::{
    AgentConfig, ChainEvent, Chunk, Config, KeyStore, Paths, Prompt, Response,
    collect_text, collect_tool_calls, collect_usage, discover_agents, resolve_agent, resolve_key,
};

use super::prompt::{find_provider, format_chain_event, resolve_prompt_text};
use super::providers;
use super::tools::{BuiltinToolRegistry, CliToolExecutor};
use crate::subprocess::tool::ExternalToolExecutor;

#[derive(Subcommand)]
pub enum AgentCommand {
    /// Run an agent
    Run(AgentRunArgs),
    /// List discovered agents
    List,
    /// Show an agent's configuration
    Show {
        /// Agent name
        name: String,
    },
    /// Scaffold a new agent TOML template
    Init(AgentInitArgs),
    /// Show agent directory paths
    Path,
}

#[derive(Args)]
pub struct AgentRunArgs {
    /// Agent name
    pub name: String,

    /// Prompt text
    pub text: Option<String>,

    /// Override agent model
    #[arg(short, long)]
    pub model: Option<String>,

    /// Override agent system prompt
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

    /// Override chain limit
    #[arg(long)]
    pub chain_limit: Option<usize>,

    /// Print tool calls and results to stderr
    #[arg(long)]
    pub tools_debug: bool,

    /// Prompt for approval before each tool execution
    #[arg(long)]
    pub tools_approve: bool,

    /// Output response as a JSON envelope
    #[arg(long)]
    pub json: bool,

    /// Verbose chain loop output (-v summary, -vv full messages). Implies --tools-debug.
    #[arg(short, long, action = ArgAction::Count)]
    pub verbose: u8,
}

#[derive(Args)]
pub struct AgentInitArgs {
    /// Agent name
    pub name: String,

    /// Create in global directory instead of local
    #[arg(long)]
    pub global: bool,
}

pub async fn run(cmd: &AgentCommand) -> llm_core::Result<()> {
    match cmd {
        AgentCommand::Run(args) => run_agent(args).await,
        AgentCommand::List => list_agents(),
        AgentCommand::Show { name } => show_agent(name),
        AgentCommand::Init(args) => init_agent(args),
        AgentCommand::Path => agent_path(),
    }
}

fn agent_path() -> llm_core::Result<()> {
    let paths = Paths::resolve()?;
    println!("Global: {}", paths.agents_dir().display());

    let local = std::env::current_dir()
        .map(|cwd| cwd.join(".llm").join("agents"))
        .ok();
    if let Some(local) = local {
        println!("Local:  {}", local.display());
    }
    Ok(())
}

fn list_agents() -> llm_core::Result<()> {
    let paths = Paths::resolve()?;
    let local_dir = std::env::current_dir()
        .map(|cwd| cwd.join(".llm").join("agents"))
        .ok();

    let agents = discover_agents(
        &paths.agents_dir(),
        local_dir.as_deref(),
    )?;

    if agents.is_empty() {
        println!("No agents found");
        return Ok(());
    }

    for agent in &agents {
        // Load config to show model
        let model = match AgentConfig::load(&agent.path) {
            Ok(config) => config.model.unwrap_or_else(|| "(default)".into()),
            Err(_) => "(error loading)".into(),
        };
        println!("{}: {} ({})", agent.name, model, agent.source);
    }
    Ok(())
}

fn show_agent(name: &str) -> llm_core::Result<()> {
    let paths = Paths::resolve()?;
    let local_dir = std::env::current_dir()
        .map(|cwd| cwd.join(".llm").join("agents"))
        .ok();

    let (config, path) = resolve_agent(
        name,
        &paths.agents_dir(),
        local_dir.as_deref(),
    )?;

    println!("Agent: {name}");
    println!("Path:  {}", path.display());
    if let Some(model) = &config.model {
        println!("Model: {model}");
    }
    if let Some(system) = &config.system_prompt {
        println!("System: {system}");
    }
    if !config.tools.is_empty() {
        println!("Tools: {}", config.tools.join(", "));
    }
    println!("Chain limit: {}", config.chain_limit);
    if !config.options.is_empty() {
        println!("Options:");
        let mut entries: Vec<_> = config.options.iter().collect();
        entries.sort_by_key(|(k, _)| *k);
        for (k, v) in entries {
            println!("  {k}: {v}");
        }
    }
    if !config.sub_agents.is_empty() {
        println!("Sub-agents: {}", config.sub_agents.join(", "));
    }
    if let Some(mem) = &config.memory {
        println!("Memory: enabled={}, last_n={:?}", mem.enabled, mem.last_n);
    }
    if let Some(budget) = &config.budget {
        println!("Budget: max_tokens={:?}", budget.max_tokens);
    }
    Ok(())
}

fn init_agent(args: &AgentInitArgs) -> llm_core::Result<()> {
    let dir = if args.global {
        let paths = Paths::resolve()?;
        paths.agents_dir()
    } else {
        std::env::current_dir()
            .map_err(llm_core::LlmError::Io)?
            .join(".llm")
            .join("agents")
    };

    let path = dir.join(format!("{}.toml", args.name));
    if path.exists() {
        return Err(llm_core::LlmError::Config(format!(
            "agent already exists: {}",
            path.display()
        )));
    }

    std::fs::create_dir_all(&dir)?;

    let template = format!(
        r#"# Agent: {}
# model = "gpt-4o-mini"
# system_prompt = "You are a helpful assistant."
# tools = []
# chain_limit = 10

# [options]
# temperature = 0.7
"#,
        args.name
    );

    std::fs::write(&path, template)?;
    println!("Created {}", path.display());
    Ok(())
}

async fn run_agent(args: &AgentRunArgs) -> llm_core::Result<()> {
    let paths = Paths::resolve()?;
    let config = Config::load(&paths.config_file())?;
    let key_store = KeyStore::load(&paths.keys_file())?;

    // Resolve agent config
    let local_dir = std::env::current_dir()
        .map(|cwd| cwd.join(".llm").join("agents"))
        .ok();
    let (agent_config, _agent_path) = resolve_agent(
        &args.name,
        &paths.agents_dir(),
        local_dir.as_deref(),
    )?;

    // Resolve prompt text (allow empty since agent might just use system prompt)
    let text = resolve_prompt_text(&args.text, false, false)?;

    // Resolve model: CLI -m > agent TOML model > global default
    let effective_default = config.effective_default_model();
    let model_from_agent = agent_config.model.as_deref();
    let model_input = args
        .model
        .as_deref()
        .or(model_from_agent)
        .unwrap_or(&effective_default);
    let model_id = config.resolve_model(model_input).to_string();

    // Build options from agent TOML [options]
    let options = agent_config.options.clone();

    // Find the provider for this model
    let all_providers = providers().await;
    let (provider, _model_info) = find_provider(&all_providers, &model_id)?;

    // Resolve key
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

    // Resolve tools from agent config
    let mut tools = Vec::new();
    let mut need_external: Vec<String> = Vec::new();
    if !agent_config.tools.is_empty() {
        let registry = BuiltinToolRegistry::new();
        for name in &agent_config.tools {
            match registry.get(name) {
                Some(tool) => tools.push(tool.clone()),
                None => need_external.push(name.clone()),
            }
        }
    }

    let external_executor = if !need_external.is_empty() || !agent_config.tools.is_empty() {
        let ext = ExternalToolExecutor::discover().await?;
        for name in &need_external {
            match ext.get_tool(name) {
                Some((_, tool)) => tools.push(tool.clone()),
                None => {
                    return Err(llm_core::LlmError::Config(format!(
                        "unknown tool in agent config: {name}"
                    )));
                }
            }
        }
        Some(ext)
    } else {
        None
    };

    // Build prompt
    let system = args
        .system
        .as_deref()
        .or(agent_config.system_prompt.as_deref());
    let mut prompt = Prompt::new(&text);
    if let Some(system) = system {
        prompt = prompt.with_system(system);
    }
    for (k, v) in &options {
        prompt = prompt.with_option(k, v.clone());
    }
    if !tools.is_empty() {
        prompt = prompt.with_tools(tools);
    }

    let chain_limit = args.chain_limit.unwrap_or(agent_config.chain_limit);
    let stream_mode = !args.no_stream && !args.json;
    let start = std::time::Instant::now();
    let json_output = args.json;

    // Agent always uses chain mode
    let debug = args.tools_debug || args.verbose > 0;
    let mut executor = CliToolExecutor::new(debug, args.tools_approve);
    if let Some(ext) = external_executor {
        executor = executor.with_external(ext);
    }
    let executor = executor;
    let mut stdout = std::io::stdout().lock();

    let verbose = args.verbose;
    let mut on_event_fn = move |event: &ChainEvent| {
        format_chain_event(event, verbose, chain_limit);
    };
    let on_event: Option<&mut dyn FnMut(&ChainEvent)> = if verbose > 0 {
        Some(&mut on_event_fn)
    } else {
        None
    };

    let agent_budget = agent_config.budget.as_ref().and_then(|b| b.max_tokens);

    let result = llm_core::chain(
        provider,
        &model_id,
        prompt,
        key.as_deref(),
        stream_mode,
        &executor,
        chain_limit,
        &mut |chunk| {
            if !json_output
                && let Chunk::Text(t) = chunk
            {
                write!(stdout, "{t}").ok();
                stdout.flush().ok();
            }
        },
        on_event,
        agent_budget,
    )
    .await?;

    if result.budget_exhausted
        && let (Some(u), Some(b)) = (&result.total_usage, agent_budget)
    {
        eprintln!("[budget] Budget exhausted: {}/{b} tokens used", u.total());
    }

    let duration_ms = start.elapsed().as_millis() as u64;

    let response_text = collect_text(&result.chunks);
    let usage_data = result.total_usage.or_else(|| collect_usage(&result.chunks));
    let tool_calls_data = collect_tool_calls(&result.chunks);

    // Show usage on stderr if requested
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
            system: system.map(|s| s.to_string()),
            response: response_text.clone(),
            options: options.clone(),
            usage: usage_data.clone(),
            tool_calls: tool_calls_data.clone(),
            tool_results: result.tool_results,
            attachments: Vec::new(),
            schema: None,
            schema_id: None,
            duration_ms,
            datetime: chrono::Utc::now().to_rfc3339(),
        };
        let store = llm_store::LogStore::open(&paths.logs_dir())?;
        let cid = store.log_response(None, &model_id, &response)?;
        Some(cid)
    } else {
        None
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
