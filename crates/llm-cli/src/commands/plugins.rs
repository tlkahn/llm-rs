use clap::Subcommand;
use llm_core::Provider;

use super::compiled_providers;
use crate::subprocess::discovery;
use crate::subprocess::tool::ExternalToolExecutor;

#[derive(Subcommand)]
pub enum PluginsCommand {
    /// List all providers and tools (compiled and external)
    List,
}

pub async fn run(command: &PluginsCommand) -> llm_core::Result<()> {
    match command {
        PluginsCommand::List => list().await,
    }
}

async fn list() -> llm_core::Result<()> {
    // Compiled providers
    let compiled = compiled_providers();
    if !compiled.is_empty() {
        println!("Compiled providers:");
        for provider in &compiled {
            let models = provider.models();
            let model_names: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
            println!(
                "  {} ({} models: {})",
                provider.id(),
                models.len(),
                model_names.join(", ")
            );
        }
    }

    // External providers
    let ext_provider_paths = discovery::discover_providers();
    if !ext_provider_paths.is_empty() {
        println!();
        println!("External providers:");
        for path in &ext_provider_paths {
            match crate::subprocess::provider::SubprocessProvider::from_binary(path.clone()).await {
                Ok(p) => {
                    let models = p.models();
                    let model_names: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
                    println!(
                        "  {} ({}) ({} models: {})",
                        p.id(),
                        path.display(),
                        models.len(),
                        model_names.join(", ")
                    );
                }
                Err(e) => {
                    println!("  {} (error: {e})", path.display());
                }
            }
        }
    }

    // External tools
    let external = ExternalToolExecutor::discover().await?;
    let mut ext_tools = external.list_tools();
    ext_tools.sort_by_key(|(name, _, _)| name.to_string());
    if !ext_tools.is_empty() {
        println!();
        println!("External tools:");
        for (name, path, tool) in &ext_tools {
            println!("  {name} ({}) — {}", path.display(), tool.description);
        }
    }

    if compiled.is_empty() && ext_provider_paths.is_empty() && ext_tools.is_empty() {
        println!("No providers or tools found.");
    }

    Ok(())
}
