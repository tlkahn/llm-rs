use crate::LlmError;

/// Parse a schema DSL string into a JSON Schema object.
///
/// Format: comma-separated (or newline-separated) fields.
/// Each field: `name [type][:description]`
/// Types: str (default), int, float, bool
pub fn parse_schema_dsl(input: &str) -> Result<serde_json::Value, LlmError> {
    let input = input.trim();
    if input.is_empty() {
        return Err(LlmError::Config("empty schema DSL input".into()));
    }

    let fields: Vec<&str> = if input.contains('\n') {
        input.split('\n').collect()
    } else {
        input.split(',').collect()
    };

    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();

    for field in fields {
        let field = field.trim();
        if field.is_empty() {
            continue;
        }

        let mut parts = field.splitn(2, char::is_whitespace);
        let name = parts
            .next()
            .ok_or_else(|| LlmError::Config(format!("invalid field: {field}")))?
            .trim();

        let mut json_type = "string".to_string();
        let mut description: Option<String> = None;

        if let Some(rest) = parts.next() {
            let rest = rest.trim();
            if !rest.is_empty() {
                // Check for type:description or just type or just :description
                if let Some((type_part, desc_part)) = rest.split_once(':') {
                    let type_part = type_part.trim();
                    if !type_part.is_empty() {
                        json_type = map_type(type_part)?;
                    }
                    let desc_part = desc_part.trim();
                    if !desc_part.is_empty() {
                        description = Some(desc_part.to_string());
                    }
                } else {
                    json_type = map_type(rest)?;
                }
            }
        }

        let mut prop = serde_json::Map::new();
        prop.insert("type".into(), serde_json::Value::String(json_type));
        if let Some(desc) = description {
            prop.insert("description".into(), serde_json::Value::String(desc));
        }
        properties.insert(name.to_string(), serde_json::Value::Object(prop));
        required.push(serde_json::Value::String(name.to_string()));
    }

    if properties.is_empty() {
        return Err(LlmError::Config("no valid fields in schema DSL".into()));
    }

    Ok(serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    }))
}

/// Wrap a schema in an array structure for --schema-multi.
pub fn multi_schema(schema: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": schema,
            }
        },
        "required": ["items"],
    })
}

fn map_type(t: &str) -> Result<String, LlmError> {
    match t {
        "str" => Ok("string".into()),
        "int" => Ok("integer".into()),
        "float" => Ok("number".into()),
        "bool" => Ok("boolean".into()),
        other => Err(LlmError::Config(format!("unknown type: {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_string_field() {
        let result = parse_schema_dsl("name str").unwrap();
        assert_eq!(result["type"], "object");
        assert_eq!(result["properties"]["name"]["type"], "string");
        assert_eq!(result["required"][0], "name");
    }

    #[test]
    fn parse_each_type() {
        let result = parse_schema_dsl("a int, b float, c bool, d str").unwrap();
        assert_eq!(result["properties"]["a"]["type"], "integer");
        assert_eq!(result["properties"]["b"]["type"], "number");
        assert_eq!(result["properties"]["c"]["type"], "boolean");
        assert_eq!(result["properties"]["d"]["type"], "string");
    }

    #[test]
    fn parse_multiple_fields() {
        let result = parse_schema_dsl("name str, age int, active bool").unwrap();
        let props = result["properties"].as_object().unwrap();
        assert_eq!(props.len(), 3);
        let required = result["required"].as_array().unwrap();
        assert_eq!(required.len(), 3);
    }

    #[test]
    fn parse_field_with_description() {
        let result = parse_schema_dsl("age int:The person's age").unwrap();
        assert_eq!(result["properties"]["age"]["type"], "integer");
        assert_eq!(
            result["properties"]["age"]["description"],
            "The person's age"
        );
    }

    #[test]
    fn parse_mixed_descriptions() {
        let result = parse_schema_dsl("name str, age int:The age, active bool").unwrap();
        assert!(result["properties"]["name"]["description"].is_null());
        assert_eq!(result["properties"]["age"]["description"], "The age");
        assert!(result["properties"]["active"]["description"].is_null());
    }

    #[test]
    fn parse_default_type_is_string() {
        let result = parse_schema_dsl("name").unwrap();
        assert_eq!(result["properties"]["name"]["type"], "string");
    }

    #[test]
    fn parse_whitespace_tolerance() {
        let result = parse_schema_dsl("  name   str  ,  age   int  ").unwrap();
        assert_eq!(result["properties"]["name"]["type"], "string");
        assert_eq!(result["properties"]["age"]["type"], "integer");
    }

    #[test]
    fn parse_newline_separated() {
        let result = parse_schema_dsl("name str\nage int\nactive bool").unwrap();
        let props = result["properties"].as_object().unwrap();
        assert_eq!(props.len(), 3);
    }

    #[test]
    fn parse_empty_string_error() {
        let result = parse_schema_dsl("");
        assert!(result.is_err());
    }

    #[test]
    fn parse_invalid_type_error() {
        let result = parse_schema_dsl("name xyz");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unknown type"));
    }

    #[test]
    fn multi_schema_wraps_in_array() {
        let schema = parse_schema_dsl("name str, age int").unwrap();
        let multi = multi_schema(schema.clone());
        assert_eq!(multi["type"], "object");
        assert_eq!(multi["properties"]["items"]["type"], "array");
        assert_eq!(multi["properties"]["items"]["items"], schema);
        assert_eq!(multi["required"][0], "items");
    }
}
