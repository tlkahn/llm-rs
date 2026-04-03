use std::io::{BufRead, IsTerminal};

use clap::Subcommand;
use llm_core::{KeyStore, LlmError, Paths, Result};

#[derive(Subcommand)]
pub enum KeysCommand {
    /// Print the path to the keys file
    Path,
    /// Set an API key (reads value from stdin)
    Set {
        /// Key name (e.g. "openai", "anthropic")
        name: String,
    },
    /// Get an API key value
    Get {
        /// Key name
        name: String,
    },
    /// List all stored key names
    List,
}

pub fn run(cmd: &KeysCommand) -> Result<()> {
    let paths = Paths::resolve()?;
    match cmd {
        KeysCommand::Path => {
            println!("{}", paths.keys_file().display());
        }
        KeysCommand::Set { name } => {
            let value = read_key_value()?;
            let mut store = KeyStore::load(&paths.keys_file())?;
            store.set(name, &value)?;
        }
        KeysCommand::Get { name } => {
            let store = KeyStore::load(&paths.keys_file())?;
            match store.get(name) {
                Some(value) => println!("{value}"),
                None => {
                    return Err(LlmError::Config(format!("key not found: {name}")));
                }
            }
        }
        KeysCommand::List => {
            let store = KeyStore::load(&paths.keys_file())?;
            for name in store.list() {
                println!("{name}");
            }
        }
    }
    Ok(())
}

fn read_key_value() -> Result<String> {
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        eprint!("Enter key: ");
        let value = rpassword::read_password().map_err(|e| LlmError::Io(e))?;
        Ok(value.trim().to_string())
    } else {
        let mut value = String::new();
        stdin.lock().read_line(&mut value)?;
        Ok(value.trim().to_string())
    }
}
