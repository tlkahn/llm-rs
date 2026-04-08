use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};

use clap::Subcommand;
use llm_core::Paths;

#[derive(Subcommand)]
pub enum SchemasCommand {
    /// Parse schema DSL and print as JSON
    Dsl {
        /// Schema DSL input (e.g. "name str, age int")
        input: String,
    },
    /// List schemas used in logged conversations
    List,
    /// Show a schema by its ID
    Show {
        /// Schema ID (hex prefix match)
        id: String,
    },
}

pub fn run(command: &SchemasCommand) -> llm_core::Result<()> {
    match command {
        SchemasCommand::Dsl { input } => {
            let schema = llm_core::parse_schema_dsl(input)?;
            println!("{}", serde_json::to_string_pretty(&schema).unwrap());
            Ok(())
        }
        SchemasCommand::List => {
            let paths = Paths::resolve()?;
            let schemas = scan_schemas(&paths.logs_dir())?;
            if schemas.is_empty() {
                println!("No schemas found in logs.");
            } else {
                for (id, schema) in &schemas {
                    println!("{id}: {schema}");
                }
            }
            Ok(())
        }
        SchemasCommand::Show { id } => {
            let paths = Paths::resolve()?;
            let schemas = scan_schemas(&paths.logs_dir())?;
            let matches: Vec<_> = schemas
                .iter()
                .filter(|(sid, _)| sid.starts_with(id.as_str()))
                .collect();
            match matches.len() {
                0 => Err(llm_core::LlmError::Config(format!(
                    "no schema found with id: {id}"
                ))),
                1 => {
                    let (_, schema) = matches[0];
                    let parsed: serde_json::Value = serde_json::from_str(schema)
                        .unwrap_or_else(|_| serde_json::Value::String(schema.clone()));
                    println!("{}", serde_json::to_string_pretty(&parsed).unwrap());
                    Ok(())
                }
                _ => Err(llm_core::LlmError::Config(format!(
                    "ambiguous schema id: {id} (matches {} schemas)",
                    matches.len()
                ))),
            }
        }
    }
}

/// Generate a schema ID from JSON bytes using a hash.
pub fn make_schema_id(schema: &serde_json::Value) -> String {
    let json_bytes = serde_json::to_vec(schema).unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    json_bytes.hash(&mut hasher);
    let hash = hasher.finish();
    format!("{hash:016x}")
}

/// Scan log files for unique schema_ids and their schemas.
fn scan_schemas(
    logs_dir: &std::path::Path,
) -> llm_core::Result<Vec<(String, String)>> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    if !logs_dir.exists() {
        return Ok(result);
    }

    let entries = std::fs::read_dir(logs_dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let content = std::fs::read_to_string(&path)?;
        for line in content.lines() {
            if let Ok(record) = serde_json::from_str::<serde_json::Value>(line)
                && record.get("type").and_then(|v| v.as_str()) == Some("response")
                && let Some(schema_id) = record.get("schema_id").and_then(|v| v.as_str())
                && !schema_id.is_empty()
                && seen.insert(schema_id.to_string())
            {
                let schema_str = record
                    .get("schema")
                    .map(|v| v.to_string())
                    .unwrap_or_default();
                result.push((schema_id.to_string(), schema_str));
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_schema_id_deterministic() {
        let schema = serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}});
        let id1 = make_schema_id(&schema);
        let id2 = make_schema_id(&schema);
        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 16);
    }

    #[test]
    fn make_schema_id_differs_for_different_schemas() {
        let s1 = serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}});
        let s2 = serde_json::json!({"type": "object", "properties": {"age": {"type": "integer"}}});
        assert_ne!(make_schema_id(&s1), make_schema_id(&s2));
    }

    #[test]
    fn scan_schemas_empty_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        let result = scan_schemas(dir.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn scan_schemas_nonexistent_dir() {
        let result = scan_schemas(std::path::Path::new("/nonexistent/path")).unwrap();
        assert!(result.is_empty());
    }
}
