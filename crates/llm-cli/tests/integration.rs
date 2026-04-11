use std::fs;
use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;
use wiremock::matchers::{body_string_contains, method, path as match_path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn llm() -> Command {
    Command::cargo_bin("llm").unwrap()
}

/// Create a command with `LLM_USER_PATH` pointing to a tmpdir for isolation.
fn llm_with_dir(dir: &TempDir) -> Command {
    let mut cmd = llm();
    cmd.env("LLM_USER_PATH", dir.path());
    cmd
}

// ==========================================================================
// Cycle 1: Scaffold — --version, --help
// ==========================================================================

#[test]
fn version_flag() {
    llm()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("0.1.0"));
}

#[test]
fn help_flag() {
    llm()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("prompt"))
        .stdout(predicate::str::contains("keys"))
        .stdout(predicate::str::contains("models"))
        .stdout(predicate::str::contains("logs"));
}

// ==========================================================================
// Cycle 2: llm keys path
// ==========================================================================

#[test]
fn keys_path() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["keys", "path"])
        .assert()
        .success()
        .stdout(predicate::str::contains("keys.toml"));
}

// ==========================================================================
// Cycle 3: llm keys set/get/list
// ==========================================================================

#[test]
fn keys_set_and_get() {
    let dir = TempDir::new().unwrap();

    // Set a key (pipe value via stdin)
    llm_with_dir(&dir)
        .args(["keys", "set", "openai"])
        .write_stdin("sk-test-key\n")
        .assert()
        .success();

    // Get it back
    llm_with_dir(&dir)
        .args(["keys", "get", "openai"])
        .assert()
        .success()
        .stdout(predicate::str::contains("sk-test-key"));
}

#[test]
fn keys_list() {
    let dir = TempDir::new().unwrap();

    // Set two keys
    llm_with_dir(&dir)
        .args(["keys", "set", "openai"])
        .write_stdin("sk-1\n")
        .assert()
        .success();
    llm_with_dir(&dir)
        .args(["keys", "set", "anthropic"])
        .write_stdin("sk-2\n")
        .assert()
        .success();

    // List should show both (sorted)
    llm_with_dir(&dir)
        .args(["keys", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("anthropic"))
        .stdout(predicate::str::contains("openai"));
}

#[test]
fn keys_get_missing() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["keys", "get", "nonexistent"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("nonexistent"));
}

// ==========================================================================
// Cycle 4: llm models list
// ==========================================================================

#[test]
fn models_list() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["models", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("gpt-4o"))
        .stdout(predicate::str::contains("gpt-4o-mini"));
}

// ==========================================================================
// Cycle 5: llm models default
// ==========================================================================

#[test]
fn models_default_show() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["models", "default"])
        .env_remove("LLM_DEFAULT_MODEL")
        .assert()
        .success()
        .stdout(predicate::str::contains("gpt-4o-mini"));
}

#[test]
fn models_default_set() {
    let dir = TempDir::new().unwrap();

    // Set default
    llm_with_dir(&dir)
        .args(["models", "default", "gpt-4o"])
        .assert()
        .success();

    // Verify it changed
    llm_with_dir(&dir)
        .args(["models", "default"])
        .env_remove("LLM_DEFAULT_MODEL")
        .assert()
        .success()
        .stdout(predicate::str::contains("gpt-4o\n"));
}

// ==========================================================================
// Cycle 6: llm logs list
// ==========================================================================

/// Write a minimal JSONL conversation file for testing.
fn write_test_conversation(logs_dir: &Path, id: &str, model: &str, name: &str, response_text: &str) {
    fs::create_dir_all(logs_dir).unwrap();
    let header = serde_json::json!({
        "type": "conversation",
        "v": 1,
        "id": id,
        "model": model,
        "name": name,
        "created": "2026-04-03T12:00:00Z"
    });
    let response = serde_json::json!({
        "type": "response",
        "id": format!("{id}-r1"),
        "model": model,
        "prompt": name,
        "system": null,
        "response": response_text,
        "options": {},
        "usage": {"input": 5, "output": 8, "details": null},
        "tool_calls": [],
        "tool_results": [],
        "attachments": [],
        "schema": null,
        "schema_id": null,
        "duration_ms": 230,
        "datetime": "2026-04-03T12:00:01Z"
    });
    let content = format!("{}\n{}\n", header, response);
    fs::write(logs_dir.join(format!("{id}.jsonl")), content).unwrap();
}

#[test]
fn logs_list_empty() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["logs", "list"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty());
}

#[test]
fn logs_list_populated() {
    let dir = TempDir::new().unwrap();
    let logs_dir = dir.path().join("logs");
    write_test_conversation(&logs_dir, "conv001", "gpt-4o", "Hello world", "Hi!");

    llm_with_dir(&dir)
        .args(["logs", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("conv001"))
        .stdout(predicate::str::contains("gpt-4o"))
        .stdout(predicate::str::contains("Hello world"));
}

#[test]
fn logs_list_json() {
    let dir = TempDir::new().unwrap();
    let logs_dir = dir.path().join("logs");
    write_test_conversation(&logs_dir, "conv001", "gpt-4o", "Hello world", "Hi!");

    let output = llm_with_dir(&dir)
        .args(["logs", "list", "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    // Should be valid JSON
    let parsed: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(parsed["id"], "conv001");
}

#[test]
fn logs_list_response() {
    let dir = TempDir::new().unwrap();
    let logs_dir = dir.path().join("logs");
    write_test_conversation(&logs_dir, "conv001", "gpt-4o", "Hello", "Hi there!");

    llm_with_dir(&dir)
        .args(["logs", "list", "-r"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Hi there!"));
}

// ==========================================================================
// Helper: wiremock-backed CLI command
// ==========================================================================

/// Returns a non-streaming OpenAI chat completion JSON body.
fn openai_non_streaming_body(content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    })
}

/// Returns SSE streaming body for OpenAI chat completion.
fn openai_streaming_body(content: &str) -> String {
    // Split content into 2 chunks for realism
    let mid = content.len() / 2;
    let (first, second) = content.split_at(mid);
    format!(
        "data: {{\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"\"}},\"finish_reason\":null}}]}}\n\n\
         data: {{\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{first}\"}},\"finish_reason\":null}}]}}\n\n\
         data: {{\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{second}\"}},\"finish_reason\":null}}]}}\n\n\
         data: {{\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n\
         data: {{\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}}}\n\n\
         data: [DONE]\n\n"
    )
}

// ==========================================================================
// Cycle 7: llm prompt non-streaming
// ==========================================================================

#[tokio::test]
async fn prompt_non_streaming() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Hi there!")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Hi there!"));
}

// ==========================================================================
// Cycle 8: llm prompt streaming
// ==========================================================================

#[tokio::test]
async fn prompt_streaming() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(openai_streaming_body("Hello world")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Hello world"));
}

// ==========================================================================
// Cycle 9: Prompt flags
// ==========================================================================

#[tokio::test]
async fn prompt_model_flag() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    // Mock expects any model — we just verify the command succeeds with -m
    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("OK")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "-m", "gpt-4o", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("OK"));
}

#[tokio::test]
async fn prompt_system_flag() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Brief response")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "-s", "be brief", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Brief response"));
}

#[tokio::test]
async fn prompt_key_flag() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("OK")),
        )
        .mount(&server)
        .await;

    // Use --key instead of env var
    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "--key", "sk-explicit", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env_remove("OPENAI_API_KEY")
        .assert()
        .success();
}

#[tokio::test]
async fn prompt_usage_flag() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Hi")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "-u", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stderr(predicate::str::contains("Token usage:"))
        .stderr(predicate::str::contains("10 input"))
        .stderr(predicate::str::contains("5 output"));
}

// ==========================================================================
// Cycle 10: Stdin piping + default subcommand
// ==========================================================================

#[tokio::test]
async fn prompt_stdin_pipe() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("World")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--no-stream"])
        .write_stdin("Hello\n")
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("World"));
}

#[tokio::test]
async fn prompt_stdin_with_arg() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Combined")),
        )
        .mount(&server)
        .await;

    // Both stdin and positional arg — should combine
    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "question"])
        .write_stdin("context\n")
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Combined"));
}

#[tokio::test]
async fn default_subcommand() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Hi")),
        )
        .mount(&server)
        .await;

    // `llm "hello"` without "prompt" subcommand
    llm_with_dir(&dir)
        .args(["--no-stream", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Hi"));
}

#[tokio::test]
async fn default_subcommand_stdin() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Piped")),
        )
        .mount(&server)
        .await;

    // `echo "hello" | llm` — no subcommand, stdin piped
    llm_with_dir(&dir)
        .args(["--no-stream"])
        .write_stdin("hello\n")
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Piped"));
}

// ==========================================================================
// Cycle 11: Exit codes
// ==========================================================================

#[test]
fn exit_code_missing_key() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "hello"])
        .env("OPENAI_BASE_URL", "http://localhost:1")
        .env_remove("OPENAI_API_KEY")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("key"));
}

#[tokio::test]
async fn exit_code_api_error() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_json(&serde_json::json!({
                    "error": {
                        "message": "Invalid API key",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key"
                    }
                })),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-bad")
        .assert()
        .code(3)
        .stderr(predicate::str::contains("Invalid API key"));
}

#[test]
fn exit_code_unknown_model() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "-m", "nonexistent-model", "hello"])
        .env("OPENAI_BASE_URL", "http://localhost:1")
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("unknown model"));
}

// ==========================================================================
// Anthropic provider tests
// ==========================================================================

#[test]
fn models_list_includes_anthropic() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["models", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("claude-opus-4-6"))
        .stdout(predicate::str::contains("claude-sonnet-4-6"))
        .stdout(predicate::str::contains("claude-haiku-4-5"));
}

/// Returns a non-streaming Anthropic Messages API JSON body.
fn anthropic_non_streaming_body(content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5
        }
    })
}

/// Returns SSE streaming body for Anthropic Messages API.
fn anthropic_streaming_body(content: &str) -> String {
    let mid = content.len() / 2;
    let (first, second) = content.split_at(mid);
    format!(
        "\
event: message_start\n\
data: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"usage\":{{\"input_tokens\":10,\"output_tokens\":0}}}}}}\n\n\
event: content_block_start\n\
data: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
event: content_block_delta\n\
data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{first}\"}}}}\n\n\
event: content_block_delta\n\
data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{second}\"}}}}\n\n\
event: content_block_stop\n\
data: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
event: message_delta\n\
data: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}},\"usage\":{{\"output_tokens\":5}}}}\n\n\
event: message_stop\n\
data: {{\"type\":\"message_stop\"}}\n\n"
    )
}

#[tokio::test]
async fn prompt_anthropic_non_streaming() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&anthropic_non_streaming_body("Bonjour!")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "-m", "claude-sonnet-4-6", "hello"])
        .env("ANTHROPIC_BASE_URL", server.uri())
        .env("ANTHROPIC_API_KEY", "sk-ant-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Bonjour!"));
}

#[tokio::test]
async fn prompt_anthropic_streaming() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(anthropic_streaming_body("Hello world")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "-m", "claude-sonnet-4-6", "hello"])
        .env("ANTHROPIC_BASE_URL", server.uri())
        .env("ANTHROPIC_API_KEY", "sk-ant-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Hello world"));
}

// ==========================================================================
// Cycle 12: Logging integration
// ==========================================================================

#[tokio::test]
async fn prompt_creates_log() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Logged response")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success();

    // Check that a JSONL log file was created
    let logs_dir = dir.path().join("logs");
    assert!(logs_dir.exists(), "logs dir should exist");
    let entries: Vec<_> = fs::read_dir(&logs_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "jsonl").unwrap_or(false))
        .collect();
    assert_eq!(entries.len(), 1, "should have exactly one log file");

    // Verify the log content
    let content = fs::read_to_string(entries[0].path()).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2, "should have header + response");
    assert!(lines[0].contains("\"type\":\"conversation\""));
    assert!(lines[1].contains("Logged response"));
}

#[tokio::test]
async fn prompt_no_log_flag() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Not logged")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "-n", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success();

    // logs dir should not exist or be empty
    let logs_dir = dir.path().join("logs");
    if logs_dir.exists() {
        let entries: Vec<_> = fs::read_dir(&logs_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(entries.is_empty(), "no log files should be created with -n");
    }
}

#[tokio::test]
async fn prompt_logging_disabled_in_config() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    // Write config with logging = false
    fs::write(dir.path().join("config.toml"), "logging = false\n").unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Not logged")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success();

    // logs dir should not exist or be empty
    let logs_dir = dir.path().join("logs");
    if logs_dir.exists() {
        let entries: Vec<_> = fs::read_dir(&logs_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(entries.is_empty(), "no log files when config.logging=false");
    }
}

// ==========================================================================
// Tools
// ==========================================================================

#[test]
fn tools_list_shows_builtins() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["tools", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("llm_version"))
        .stdout(predicate::str::contains("llm_time"));
}

#[test]
fn prompt_with_unknown_tool_error() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "-T", "nonexistent", "hello"])
        .env("OPENAI_BASE_URL", "http://localhost:1")
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("unknown tool: nonexistent"));
}

#[tokio::test]
async fn prompt_with_tool_flag() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    // First response: tool call
    let tool_call_body = serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "llm_version",
                        "arguments": "{}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}
    });

    // Second response: final text
    let text_body = serde_json::json!({
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "The version is 0.1.0"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48}
    });

    // Register second response first (default), then first with limit 1
    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&text_body),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&tool_call_body),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args([
            "prompt", "--no-stream", "-n", "-T", "llm_version", "What version?",
        ])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("version"));
}

#[tokio::test]
async fn prompt_chain_limit_respected() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    // Always return tool call - chain should stop at limit
    let tool_call_body = serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "llm_version",
                        "arguments": "{}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}
    });

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&tool_call_body),
        )
        .mount(&server)
        .await;

    // With chain-limit 2, should stop after 2 iterations without error
    llm_with_dir(&dir)
        .args([
            "prompt",
            "--no-stream",
            "-n",
            "-T",
            "llm_version",
            "--chain-limit",
            "2",
            "Loop test",
        ])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success();
}

// ==========================================================================
// Schemas
// ==========================================================================

#[test]
fn schemas_dsl_command() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["schemas", "dsl", "name str, age int"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"type\": \"object\""))
        .stdout(predicate::str::contains("\"name\""))
        .stdout(predicate::str::contains("\"string\""))
        .stdout(predicate::str::contains("\"integer\""));
}

#[test]
fn schemas_list_empty() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["schemas", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No schemas"));
}

#[tokio::test]
async fn prompt_with_schema_dsl() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    let body = serde_json::json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"name\":\"John\",\"age\":30}"
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    });

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&body),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args([
            "prompt",
            "--no-stream",
            "--schema",
            "name str, age int",
            "Extract from: John is 30",
        ])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("John"));
}

#[tokio::test]
async fn prompt_with_schema_json_literal() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    let body = serde_json::json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"name\":\"Jane\"}"
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    });

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&body),
        )
        .mount(&server)
        .await;

    let schema_json = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#;
    llm_with_dir(&dir)
        .args([
            "prompt",
            "--no-stream",
            "-n",
            "--schema",
            schema_json,
            "Extract name",
        ])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Jane"));
}

#[tokio::test]
async fn prompt_with_schema_multi() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    let body = serde_json::json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"items\":[{\"name\":\"A\"},{\"name\":\"B\"}]}"
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    });

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&body),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args([
            "prompt",
            "--no-stream",
            "-n",
            "--schema",
            "name str",
            "--schema-multi",
            "List names",
        ])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("items"));
}

// ==========================================================================
// Phase 3: Conversation continuation
// ==========================================================================

#[tokio::test]
async fn continue_conversation_appends_to_same_file() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("First answer")),
        )
        .mount(&server)
        .await;

    // First prompt: creates a new conversation
    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "Hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success();

    let logs_dir = dir.path().join("logs");
    let entries: Vec<_> = fs::read_dir(&logs_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "jsonl").unwrap_or(false))
        .collect();
    assert_eq!(entries.len(), 1);
    let log_path = entries[0].path();
    let lines_before = fs::read_to_string(&log_path).unwrap().lines().count();
    assert_eq!(lines_before, 2); // header + response

    // Continue with -c
    server.reset().await;
    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Second answer")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "-c", "Follow up"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Second answer"));

    // Should still be one file, now with 3 lines
    let entries_after: Vec<_> = fs::read_dir(&logs_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "jsonl").unwrap_or(false))
        .collect();
    assert_eq!(entries_after.len(), 1);
    let content = fs::read_to_string(&log_path).unwrap();
    assert_eq!(content.lines().count(), 3); // header + 2 responses
}

#[tokio::test]
async fn cid_continues_specific_conversation() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Reply")),
        )
        .mount(&server)
        .await;

    // First conversation
    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "First conv"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success();

    // Find the conversation ID
    let logs_dir = dir.path().join("logs");
    let entries: Vec<_> = fs::read_dir(&logs_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "jsonl").unwrap_or(false))
        .collect();
    let conv_id = entries[0]
        .path()
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    // Continue by --cid
    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "--cid", &conv_id, "Continue"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success();

    // Original file should now have 3 lines
    let content = fs::read_to_string(entries[0].path()).unwrap();
    assert_eq!(content.lines().count(), 3);
}

// ==========================================================================
// Phase 3: --messages & --json
// ==========================================================================

#[tokio::test]
async fn messages_from_file() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Continued")),
        )
        .mount(&server)
        .await;

    // Write messages file
    let messages_file = dir.path().join("msgs.json");
    fs::write(
        &messages_file,
        r#"[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi!"},{"role":"user","content":"Follow up"}]"#,
    ).unwrap();

    llm_with_dir(&dir)
        .args([
            "prompt",
            "--no-stream",
            "-n",
            "--messages",
            messages_file.to_str().unwrap(),
        ])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Continued"));
}

#[tokio::test]
async fn json_output_envelope() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("The answer is 42")),
        )
        .mount(&server)
        .await;

    let output = llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "-n", "--json", "What is 6x7?"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let json: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(json["content"], "The answer is 42");
    assert_eq!(json["model"], "gpt-4o-mini");
    assert!(json.get("duration_ms").is_some());
}

#[tokio::test]
async fn messages_stdin_with_json_output() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Roundtrip")),
        )
        .mount(&server)
        .await;

    let output = llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "-n", "--messages", "-", "--json"])
        .write_stdin(r#"[{"role":"user","content":"hi"}]"#)
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let json: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(json["content"], "Roundtrip");
}

// ==========================================================================
// Phase 3: llm logs subcommands
// ==========================================================================

#[test]
fn logs_path() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["logs", "path"])
        .assert()
        .success()
        .stdout(predicate::str::contains("logs"));
}

#[test]
fn logs_status_default() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["logs", "status"])
        .assert()
        .success()
        .stdout(predicate::str::contains("enabled"));
}

#[test]
fn logs_on_off_toggle() {
    let dir = TempDir::new().unwrap();

    // Turn off
    llm_with_dir(&dir)
        .args(["logs", "off"])
        .assert()
        .success()
        .stdout(predicate::str::contains("disabled"));

    // Status should show disabled
    llm_with_dir(&dir)
        .args(["logs", "status"])
        .assert()
        .success()
        .stdout(predicate::str::contains("disabled"));

    // Turn back on
    llm_with_dir(&dir)
        .args(["logs", "on"])
        .assert()
        .success()
        .stdout(predicate::str::contains("enabled"));

    llm_with_dir(&dir)
        .args(["logs", "status"])
        .assert()
        .success()
        .stdout(predicate::str::contains("enabled"));
}

// ==========================================================================
// Phase 4: Subprocess extensibility — external tools and providers
// ==========================================================================

/// Return the absolute path to tests/fixtures/bin/ so we can prepend it to PATH.
fn fixtures_bin() -> String {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .join("tests/fixtures/bin")
        .to_string_lossy()
        .to_string()
}

/// Build a PATH that has our fixtures dir first, then the system PATH.
fn path_with_fixtures() -> String {
    let sys_path = std::env::var("PATH").unwrap_or_default();
    format!("{}:{sys_path}", fixtures_bin())
}

#[test]
fn tools_list_shows_external_tool() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["tools", "list"])
        .env("PATH", path_with_fixtures())
        .assert()
        .success()
        .stdout(predicate::str::contains("upper"))
        .stdout(predicate::str::contains("Uppercase text"));
}

#[test]
fn tools_list_shows_builtin_and_external() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["tools", "list"])
        .env("PATH", path_with_fixtures())
        .assert()
        .success()
        .stdout(predicate::str::contains("llm_version"))
        .stdout(predicate::str::contains("upper"));
}

#[test]
fn plugins_list_shows_compiled_providers() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["plugins", "list"])
        .env("PATH", path_with_fixtures())
        .assert()
        .success()
        .stdout(predicate::str::contains("Compiled providers:"))
        .stdout(predicate::str::contains("openai"))
        .stdout(predicate::str::contains("anthropic"));
}

#[test]
fn plugins_list_shows_external_provider() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["plugins", "list"])
        .env("PATH", path_with_fixtures())
        .assert()
        .success()
        .stdout(predicate::str::contains("External providers:"))
        .stdout(predicate::str::contains("echo"))
        .stdout(predicate::str::contains("echo-model"));
}

#[test]
fn plugins_list_shows_external_tool() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["plugins", "list"])
        .env("PATH", path_with_fixtures())
        .assert()
        .success()
        .stdout(predicate::str::contains("External tools:"))
        .stdout(predicate::str::contains("upper"))
        .stdout(predicate::str::contains("Uppercase text"));
}

#[test]
fn models_list_includes_subprocess_provider_models() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["models", "list"])
        .env("PATH", path_with_fixtures())
        .assert()
        .success()
        .stdout(predicate::str::contains("echo-model (echo)"));
}

#[test]
fn prompt_with_subprocess_provider() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "-m", "echo-model", "-n", "hello world"])
        .env("PATH", path_with_fixtures())
        .assert()
        .success()
        .stdout(predicate::str::contains("echo: hello world"));
}

#[tokio::test]
async fn prompt_with_external_tool_in_chain() {
    // Set up wiremock to simulate an LLM that calls the "upper" tool
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    // First call: LLM returns a tool call to "upper"
    let tool_call_response = serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "upper",
                        "arguments": "{\"text\":\"hello\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });

    // Second call: LLM returns final text
    let final_response = serde_json::json!({
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The uppercased version is: HELLO"
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    });

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&tool_call_response),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&final_response),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args([
            "prompt", "--no-stream", "--no-log",
            "-m", "gpt-4o-mini",
            "-T", "upper",
            "make this loud: hello",
        ])
        .env("PATH", path_with_fixtures())
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("HELLO"));
}

#[test]
fn help_shows_plugins_subcommand() {
    llm()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("plugins"));
}

// ==========================================================================
// Verbose chain loop observability (-v / -vv)
// ==========================================================================

/// Helper: build OpenAI-style JSON responses for tool chain wiremock tests.
fn openai_tool_call_response(tool_name: &str, tool_id: &str, args: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-v1",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": args
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    })
}

fn openai_text_response(text: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-v2",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    })
}

#[tokio::test]
async fn verbose_shows_chain_iteration_summary() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_tool_call_response("upper", "call_1", r#"{"text":"hello"}"#)),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_text_response("HELLO")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args([
            "prompt", "--no-stream", "--no-log",
            "-m", "gpt-4o-mini",
            "-T", "upper",
            "--verbose",
            "make this loud: hello",
        ])
        .env("PATH", path_with_fixtures())
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stderr(predicate::str::contains("[chain] Iteration 1/"))
        .stderr(predicate::str::contains("[chain] Iteration 2/"))
        .stderr(predicate::str::contains("[chain] Iteration 1 complete"))
        .stderr(predicate::str::contains("tool call(s)"));
}

#[tokio::test]
async fn verbose_vv_shows_messages_json() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_tool_call_response("upper", "call_1", r#"{"text":"hello"}"#)),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_text_response("HELLO")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args([
            "prompt", "--no-stream", "--no-log",
            "-m", "gpt-4o-mini",
            "-T", "upper",
            "-vv",
            "make this loud: hello",
        ])
        .env("PATH", path_with_fixtures())
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stderr(predicate::str::contains("[chain] Messages:"))
        .stderr(predicate::str::contains("\"role\": \"user\""));
}

#[tokio::test]
async fn verbose_implies_tools_debug() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_tool_call_response("upper", "call_1", r#"{"text":"hello"}"#)),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_text_response("HELLO")),
        )
        .mount(&server)
        .await;

    // Use --verbose without --tools-debug; should still see tool debug output
    llm_with_dir(&dir)
        .args([
            "prompt", "--no-stream", "--no-log",
            "-m", "gpt-4o-mini",
            "-T", "upper",
            "--verbose",
            "make this loud: hello",
        ])
        .env("PATH", path_with_fixtures())
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stderr(predicate::str::contains("Tool call:"))
        .stderr(predicate::str::contains("Tool result:"));
}

#[test]
fn verbose_flag_parsing() {
    // -v should be parsed (help text is sufficient validation since clap validates)
    llm()
        .args(["prompt", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--verbose"));
}

// ==========================================================================
// Options subcommand
// ==========================================================================

#[test]
fn options_set_and_get() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["options", "set", "gpt-4o-mini", "temperature", "0.7"])
        .assert()
        .success();

    llm_with_dir(&dir)
        .args(["options", "get", "gpt-4o-mini", "temperature"])
        .assert()
        .success()
        .stdout(predicate::str::contains("temperature: 0.7"));
}

#[test]
fn options_set_overwrites() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["options", "set", "gpt-4o-mini", "temperature", "0.5"])
        .assert()
        .success();
    llm_with_dir(&dir)
        .args(["options", "set", "gpt-4o-mini", "temperature", "0.9"])
        .assert()
        .success();

    llm_with_dir(&dir)
        .args(["options", "get", "gpt-4o-mini", "temperature"])
        .assert()
        .success()
        .stdout(predicate::str::contains("temperature: 0.9"));
}

#[test]
fn options_get_missing() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["options", "get", "gpt-4o-mini", "temperature"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("no option"));
}

#[test]
fn options_get_all() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["options", "set", "gpt-4o-mini", "temperature", "0.7"])
        .assert()
        .success();
    llm_with_dir(&dir)
        .args(["options", "set", "gpt-4o-mini", "max_tokens", "200"])
        .assert()
        .success();

    llm_with_dir(&dir)
        .args(["options", "get", "gpt-4o-mini"])
        .assert()
        .success()
        .stdout(predicate::str::contains("temperature: 0.7"))
        .stdout(predicate::str::contains("max_tokens: 200"));
}

#[test]
fn options_list() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["options", "set", "gpt-4o-mini", "temperature", "0.7"])
        .assert()
        .success();

    llm_with_dir(&dir)
        .args(["options", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("gpt-4o-mini:"))
        .stdout(predicate::str::contains("temperature: 0.7"));
}

#[test]
fn options_list_empty() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["options", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No options set"));
}

#[test]
fn options_clear_key() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["options", "set", "gpt-4o-mini", "temperature", "0.7"])
        .assert()
        .success();

    llm_with_dir(&dir)
        .args(["options", "clear", "gpt-4o-mini", "temperature"])
        .assert()
        .success();

    llm_with_dir(&dir)
        .args(["options", "get", "gpt-4o-mini"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No options set"));
}

#[test]
fn options_clear_all() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["options", "set", "gpt-4o-mini", "temperature", "0.7"])
        .assert()
        .success();
    llm_with_dir(&dir)
        .args(["options", "set", "gpt-4o-mini", "max_tokens", "200"])
        .assert()
        .success();

    llm_with_dir(&dir)
        .args(["options", "clear", "gpt-4o-mini"])
        .assert()
        .success();

    llm_with_dir(&dir)
        .args(["options", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No options set"));
}

#[test]
fn options_clear_missing() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["options", "clear", "gpt-4o-mini", "temperature"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("no option"));
}

#[tokio::test]
async fn prompt_with_option_flag() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .and(body_string_contains("\"temperature\":0.7"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("temp response")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args([
            "prompt", "--no-stream", "-n",
            "-o", "temperature", "0.7",
            "hello",
        ])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("temp response"));
}

#[tokio::test]
async fn prompt_config_options_applied() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    // Write config with options for gpt-4o-mini
    fs::write(
        dir.path().join("config.toml"),
        "[options.gpt-4o-mini]\ntemperature = 0.3\n",
    )
    .unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .and(body_string_contains("\"temperature\":0.3"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("config response")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--no-stream", "-n", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("config response"));
}

#[tokio::test]
async fn prompt_cli_overrides_config() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    // Config sets temperature=0.3
    fs::write(
        dir.path().join("config.toml"),
        "[options.gpt-4o-mini]\ntemperature = 0.3\n",
    )
    .unwrap();

    // CLI -o temperature 1.0 should override
    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .and(body_string_contains("\"temperature\":1.0"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("override response")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args([
            "prompt", "--no-stream", "-n",
            "-o", "temperature", "1.0",
            "hello",
        ])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("override response"));
}

#[test]
fn help_shows_options_subcommand() {
    llm()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("options"));
}

// ==========================================================================
// Aliases
// ==========================================================================

#[test]
fn aliases_path() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["aliases", "path"])
        .assert()
        .success()
        .stdout(predicate::str::contains("config.toml"));
}

#[test]
fn aliases_set_and_list() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["aliases", "set", "claude", "claude-sonnet-4-20250514"])
        .assert()
        .success();
    llm_with_dir(&dir)
        .args(["aliases", "set", "fast", "gpt-4o-mini"])
        .assert()
        .success();

    llm_with_dir(&dir)
        .args(["aliases", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("claude: claude-sonnet-4-20250514"))
        .stdout(predicate::str::contains("fast: gpt-4o-mini"));
}

#[test]
fn aliases_set_overwrites() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["aliases", "set", "claude", "claude-sonnet-4-20250514"])
        .assert()
        .success();
    llm_with_dir(&dir)
        .args(["aliases", "set", "claude", "claude-opus-4-20250514"])
        .assert()
        .success();

    llm_with_dir(&dir)
        .args(["aliases", "show", "claude"])
        .assert()
        .success()
        .stdout(predicate::str::contains("claude: claude-opus-4-20250514"));
}

#[test]
fn aliases_show() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["aliases", "set", "claude", "claude-sonnet-4-20250514"])
        .assert()
        .success();

    llm_with_dir(&dir)
        .args(["aliases", "show", "claude"])
        .assert()
        .success()
        .stdout(predicate::str::contains("claude: claude-sonnet-4-20250514"));
}

#[test]
fn aliases_show_missing() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["aliases", "show", "nonexistent"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("alias 'nonexistent' not found"));
}

#[test]
fn aliases_remove() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["aliases", "set", "claude", "claude-sonnet-4-20250514"])
        .assert()
        .success();

    llm_with_dir(&dir)
        .args(["aliases", "remove", "claude"])
        .assert()
        .success();

    llm_with_dir(&dir)
        .args(["aliases", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No aliases set"));
}

#[test]
fn aliases_remove_missing() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["aliases", "remove", "nonexistent"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("alias 'nonexistent' not found"));
}

#[test]
fn aliases_list_empty() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["aliases", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No aliases set"));
}

#[test]
fn help_shows_aliases_subcommand() {
    llm()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("aliases"));
}

// ==========================================================================
// Phase 5: Agent commands
// ==========================================================================

#[test]
fn help_shows_agent_subcommand() {
    llm()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("agent"));
}

// --- agent path ---

#[test]
fn agent_path_shows_directories() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["agent", "path"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Global:"))
        .stdout(predicate::str::contains("Local:"));
}

// --- agent list ---

#[test]
fn agent_list_empty() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["agent", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No agents found"));
}

#[test]
fn agent_list_with_agents() {
    let dir = TempDir::new().unwrap();
    let agents_dir = dir.path().join("agents");
    fs::create_dir_all(&agents_dir).unwrap();
    fs::write(
        agents_dir.join("reviewer.toml"),
        "model = \"gpt-4o\"\nsystem_prompt = \"Review code.\"\n",
    )
    .unwrap();
    fs::write(
        agents_dir.join("helper.toml"),
        "model = \"gpt-4o-mini\"\n",
    )
    .unwrap();

    llm_with_dir(&dir)
        .args(["agent", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("helper"))
        .stdout(predicate::str::contains("reviewer"))
        .stdout(predicate::str::contains("gpt-4o"));
}

// --- agent show ---

#[test]
fn agent_show_existing() {
    let dir = TempDir::new().unwrap();
    let agents_dir = dir.path().join("agents");
    fs::create_dir_all(&agents_dir).unwrap();
    fs::write(
        agents_dir.join("reviewer.toml"),
        "model = \"gpt-4o\"\nsystem_prompt = \"Review code.\"\ntools = [\"llm_time\"]\nchain_limit = 15\n",
    )
    .unwrap();

    llm_with_dir(&dir)
        .args(["agent", "show", "reviewer"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Agent: reviewer"))
        .stdout(predicate::str::contains("Model: gpt-4o"))
        .stdout(predicate::str::contains("System: Review code."))
        .stdout(predicate::str::contains("Tools: llm_time"))
        .stdout(predicate::str::contains("Chain limit: 15"));
}

#[test]
fn agent_show_nonexistent() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["agent", "show", "nonexistent"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("agent not found"));
}

// --- agent init ---

#[test]
fn agent_init_local() {
    let dir = TempDir::new().unwrap();
    let cwd = dir.path().join("project");
    fs::create_dir_all(&cwd).unwrap();

    llm_with_dir(&dir)
        .args(["agent", "init", "myagent"])
        .current_dir(&cwd)
        .assert()
        .success()
        .stdout(predicate::str::contains("Created"));

    let agent_file = cwd.join(".llm").join("agents").join("myagent.toml");
    assert!(agent_file.exists());
    let content = fs::read_to_string(agent_file).unwrap();
    assert!(content.contains("# Agent: myagent"));
}

#[test]
fn agent_init_global() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["agent", "init", "myagent", "--global"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Created"));

    let agent_file = dir.path().join("agents").join("myagent.toml");
    assert!(agent_file.exists());
}

#[test]
fn agent_init_already_exists() {
    let dir = TempDir::new().unwrap();
    let cwd = dir.path().join("project");
    let agents_dir = cwd.join(".llm").join("agents");
    fs::create_dir_all(&agents_dir).unwrap();
    fs::write(agents_dir.join("myagent.toml"), "model = \"gpt-4o\"\n").unwrap();

    llm_with_dir(&dir)
        .args(["agent", "init", "myagent"])
        .current_dir(&cwd)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("agent already exists"));
}

// --- agent run ---

#[tokio::test]
async fn agent_run_basic() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();
    let agents_dir = dir.path().join("agents");
    fs::create_dir_all(&agents_dir).unwrap();
    fs::write(
        agents_dir.join("greeter.toml"),
        "model = \"gpt-4o-mini\"\nsystem_prompt = \"You greet people.\"\n",
    )
    .unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .and(body_string_contains("You greet people"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Hello there!")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["agent", "run", "greeter", "--no-stream", "Hi!"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Hello there!"));
}

#[tokio::test]
async fn agent_run_stdin() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();
    let agents_dir = dir.path().join("agents");
    fs::create_dir_all(&agents_dir).unwrap();
    fs::write(
        agents_dir.join("greeter.toml"),
        "model = \"gpt-4o-mini\"\n",
    )
    .unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Response!")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["agent", "run", "greeter", "--no-stream"])
        .write_stdin("Hello from stdin\n")
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Response!"));
}

#[tokio::test]
async fn agent_run_model_override() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();
    let agents_dir = dir.path().join("agents");
    fs::create_dir_all(&agents_dir).unwrap();
    fs::write(
        agents_dir.join("greeter.toml"),
        "model = \"gpt-4o-mini\"\n",
    )
    .unwrap();

    // Mock expects gpt-4o model (overridden from agent's gpt-4o-mini)
    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .and(body_string_contains("gpt-4o"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("Overridden!")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["agent", "run", "greeter", "--no-stream", "-m", "gpt-4o", "Hi"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("Overridden!"));
}

#[tokio::test]
async fn agent_run_system_override() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();
    let agents_dir = dir.path().join("agents");
    fs::create_dir_all(&agents_dir).unwrap();
    fs::write(
        agents_dir.join("greeter.toml"),
        "model = \"gpt-4o-mini\"\nsystem_prompt = \"Original system.\"\n",
    )
    .unwrap();

    // Verify the overridden system prompt is sent
    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .and(body_string_contains("Overridden system"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("OK")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["agent", "run", "greeter", "--no-stream", "-s", "Overridden system", "Hi"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("OK"));
}

#[tokio::test]
async fn agent_run_json_output() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();
    let agents_dir = dir.path().join("agents");
    fs::create_dir_all(&agents_dir).unwrap();
    fs::write(
        agents_dir.join("greeter.toml"),
        "model = \"gpt-4o-mini\"\n",
    )
    .unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_non_streaming_body("JSON output")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["agent", "run", "greeter", "--json", "Hi"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"model\""))
        .stdout(predicate::str::contains("\"content\""))
        .stdout(predicate::str::contains("JSON output"));
}

#[test]
fn agent_run_nonexistent() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["agent", "run", "nonexistent", "Hi"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("agent not found"));
}

#[test]
fn agent_run_unknown_tool() {
    let dir = TempDir::new().unwrap();
    let agents_dir = dir.path().join("agents");
    fs::create_dir_all(&agents_dir).unwrap();
    fs::write(
        agents_dir.join("bad.toml"),
        "model = \"gpt-4o-mini\"\ntools = [\"nonexistent_tool_xyz\"]\n",
    )
    .unwrap();

    llm_with_dir(&dir)
        .args(["agent", "run", "bad", "Hi"])
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("unknown tool"));
}

// ==========================================================================
// Phase 8: --dry-run on `llm agent run`
// ==========================================================================

fn write_dry_run_agent(dir: &TempDir, name: &str, body: &str) {
    let agents_dir = dir.path().join("agents");
    fs::create_dir_all(&agents_dir).unwrap();
    fs::write(agents_dir.join(format!("{name}.toml")), body).unwrap();
}

#[test]
fn agent_run_dry_run_prints_resolved_config() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(
        &dir,
        "researcher",
        "model = \"gpt-4o-mini\"\n\
         system_prompt = \"You are helpful.\"\n\
         tools = [\"llm_version\"]\n\
         chain_limit = 7\n\
         \n\
         [options]\n\
         temperature = 0.7\n",
    );

    llm_with_dir(&dir)
        .args(["agent", "run", "researcher", "--dry-run", "hello world"])
        .env_remove("OPENAI_API_KEY")
        .assert()
        .success()
        .stdout(predicate::str::contains("Agent:       researcher"))
        .stdout(predicate::str::contains("Model:       gpt-4o-mini"))
        .stdout(predicate::str::contains("source: agent"))
        .stdout(predicate::str::contains("Provider:    openai"))
        .stdout(predicate::str::contains("System:      You are helpful."))
        .stdout(predicate::str::contains("Prompt:      hello world"))
        .stdout(predicate::str::contains("llm_version (builtin)"))
        .stdout(predicate::str::contains("temperature: 0.7"))
        .stdout(predicate::str::contains("Chain limit: 7"));
}

#[test]
fn agent_run_dry_run_succeeds_without_api_key() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(&dir, "greeter", "model = \"gpt-4o-mini\"\n");

    llm_with_dir(&dir)
        .args(["agent", "run", "greeter", "--dry-run", "Hi"])
        .env_remove("OPENAI_API_KEY")
        .assert()
        .success();
}

#[test]
fn agent_run_dry_run_model_source_cli() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(&dir, "greeter", "model = \"gpt-4o-mini\"\n");

    llm_with_dir(&dir)
        .args([
            "agent", "run", "greeter", "--dry-run", "-m", "gpt-4o", "Hi",
        ])
        .env_remove("OPENAI_API_KEY")
        .assert()
        .success()
        .stdout(predicate::str::contains("Model:       gpt-4o"))
        .stdout(predicate::str::contains("source: cli"));
}

#[test]
fn agent_run_dry_run_json_envelope() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(
        &dir,
        "researcher",
        "model = \"gpt-4o-mini\"\ntools = [\"llm_version\"]\n",
    );

    let output = llm_with_dir(&dir)
        .args(["agent", "run", "researcher", "--dry-run", "--json", "hi"])
        .env_remove("OPENAI_API_KEY")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(parsed["agent_name"], "researcher");
    assert_eq!(parsed["model"], "gpt-4o-mini");
    assert_eq!(parsed["provider"], "openai");
    assert_eq!(parsed["prompt_text"], "hi");
    assert_eq!(parsed["tools"][0]["name"], "llm_version");
    assert_eq!(parsed["tools"][0]["source"], "builtin");
    assert!(parsed.get("system_prompt").is_none());
    assert!(parsed.get("prompt").is_none());
}

#[test]
fn agent_run_dry_run_verbose_includes_prompt_json() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(
        &dir,
        "greeter",
        "model = \"gpt-4o-mini\"\nsystem_prompt = \"sys\"\n",
    );

    llm_with_dir(&dir)
        .args(["agent", "run", "greeter", "--dry-run", "-v", "hello"])
        .env_remove("OPENAI_API_KEY")
        .assert()
        .success()
        .stdout(predicate::str::contains("Prompt (full JSON):"))
        .stdout(predicate::str::contains("\"system\""))
        .stdout(predicate::str::contains("\"text\""));
}

#[test]
fn agent_run_dry_run_verbose_vv_also_dumps_prompt_json() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(&dir, "greeter", "model = \"gpt-4o-mini\"\n");

    llm_with_dir(&dir)
        .args(["agent", "run", "greeter", "--dry-run", "-vv", "hello"])
        .env_remove("OPENAI_API_KEY")
        .assert()
        .success()
        .stdout(predicate::str::contains("Prompt (full JSON):"));
}

#[test]
fn agent_run_dry_run_json_verbose_includes_prompt_field() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(&dir, "greeter", "model = \"gpt-4o-mini\"\n");

    let output = llm_with_dir(&dir)
        .args(["agent", "run", "greeter", "--dry-run", "--json", "-v", "hi"])
        .env_remove("OPENAI_API_KEY")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(parsed.get("prompt").is_some(), "expected prompt field");
    assert_eq!(parsed["prompt"]["text"], "hi");
}

#[test]
fn agent_run_dry_run_unknown_agent() {
    let dir = TempDir::new().unwrap();
    llm_with_dir(&dir)
        .args(["agent", "run", "nonexistent", "--dry-run", "hi"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("agent not found"));
}

#[test]
fn agent_run_dry_run_unknown_tool() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(
        &dir,
        "bad",
        "model = \"gpt-4o-mini\"\ntools = [\"nonexistent_tool_xyz\"]\n",
    );

    llm_with_dir(&dir)
        .args(["agent", "run", "bad", "--dry-run", "hi"])
        .env_remove("OPENAI_API_KEY")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("unknown tool"));
}

#[test]
fn agent_run_dry_run_unknown_model() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(&dir, "bogus", "model = \"not-a-real-model-xyz\"\n");

    llm_with_dir(&dir)
        .args(["agent", "run", "bogus", "--dry-run", "hi"])
        .env_remove("OPENAI_API_KEY")
        .assert()
        .code(2);
}

#[test]
fn agent_run_dry_run_does_not_log() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(&dir, "greeter", "model = \"gpt-4o-mini\"\n");

    llm_with_dir(&dir)
        .args(["agent", "run", "greeter", "--dry-run", "hi"])
        .env_remove("OPENAI_API_KEY")
        .assert()
        .success();

    let logs_dir = dir.path().join("logs");
    if logs_dir.exists() {
        let entries: Vec<_> = fs::read_dir(&logs_dir).unwrap().collect();
        assert!(
            entries.is_empty(),
            "dry-run must not write any log files, found {entries:?}"
        );
    }
}

#[tokio::test]
async fn agent_run_dry_run_does_not_call_provider() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(&dir, "greeter", "model = \"gpt-4o-mini\"\n");

    // Mount no mocks. If any HTTP request happens, wiremock will log
    // an unmatched request and the test harness will observe it.
    llm_with_dir(&dir)
        .args(["agent", "run", "greeter", "--dry-run", "hi"])
        .env("OPENAI_BASE_URL", server.uri())
        .env_remove("OPENAI_API_KEY")
        .assert()
        .success();

    let received = server.received_requests().await.unwrap();
    assert!(
        received.is_empty(),
        "dry-run must not make any HTTP calls, got {} request(s)",
        received.len()
    );
}

// ==========================================================================
// Phase 6: Budget tracking
// ==========================================================================

#[tokio::test]
async fn usage_flag_shows_total_across_chain() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    // First call: tool call (usage: 10 input, 5 output)
    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_tool_call_response("upper", "call_1", r#"{"text":"hello"}"#)),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // Second call: text response (usage: 20 input, 10 output)
    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_text_response("HELLO")),
        )
        .mount(&server)
        .await;

    // -u should show cumulative: 30 input, 15 output
    llm_with_dir(&dir)
        .args([
            "prompt", "--no-stream", "--no-log",
            "-m", "gpt-4o-mini",
            "-T", "upper",
            "-u",
            "hello",
        ])
        .env("PATH", path_with_fixtures())
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stderr(predicate::str::contains("30 input"))
        .stderr(predicate::str::contains("15 output"));
}

#[tokio::test]
async fn verbose_shows_cumulative_usage() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_tool_call_response("upper", "call_1", r#"{"text":"hello"}"#)),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_text_response("HELLO")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args([
            "prompt", "--no-stream", "--no-log",
            "-m", "gpt-4o-mini",
            "-T", "upper",
            "--verbose",
            "hello",
        ])
        .env("PATH", path_with_fixtures())
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stderr(predicate::str::contains("cumulative: 30 input, 15 output"));
}

#[tokio::test]
async fn agent_budget_exhausted_output() {
    let server = MockServer::start().await;
    let dir = TempDir::new().unwrap();
    let agents_dir = dir.path().join("agents");
    fs::create_dir_all(&agents_dir).unwrap();
    fs::write(
        agents_dir.join("budgeted.toml"),
        "model = \"gpt-4o-mini\"\ntools = [\"llm_time\"]\n\n[budget]\nmax_tokens = 10\n",
    )
    .unwrap();

    // Tool call response (10+5=15 tokens total > budget 10)
    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_tool_call_response("llm_time", "call_1", "{}")),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // Second response should not be reached (budget exhausted after iter 1)
    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_text_response("Done!")),
        )
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["agent", "run", "budgeted", "--no-stream", "What time is it?"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stderr(predicate::str::contains("[budget] Budget exhausted: 15/10 tokens used"));
}

// ==========================================================================
// Phase 7: Retry / Backoff
// ==========================================================================

#[test]
fn retries_flag_accepted() {
    let dir = TempDir::new().unwrap();
    // --retries flag parses without error (fails for a different reason: no key)
    llm_with_dir(&dir)
        .args(["prompt", "--retries", "3", "--no-stream", "hello"])
        .env("OPENAI_BASE_URL", "http://127.0.0.1:1")
        .assert()
        .failure();
}

#[tokio::test]
async fn retries_on_429() {
    let dir = TempDir::new().unwrap();
    let server = MockServer::start().await;

    // First request returns 429, second returns success
    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&openai_text_response("OK")),
        )
        .expect(1)
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--retries", "2", "--no-stream", "-n", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .success()
        .stdout(predicate::str::contains("OK"))
        .stderr(predicate::str::contains("[retry]"));
}

#[tokio::test]
async fn retries_exhausted_returns_error() {
    let dir = TempDir::new().unwrap();
    let server = MockServer::start().await;

    // All requests return 500
    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
        .expect(3) // 1 original + 2 retries
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--retries", "2", "--no-stream", "-n", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .failure()
        .stderr(predicate::str::contains("HTTP error 500"));
}

#[tokio::test]
async fn no_retry_on_401() {
    let dir = TempDir::new().unwrap();
    let server = MockServer::start().await;

    // 401 is not retryable — should fail immediately with only 1 call
    Mock::given(method("POST"))
        .and(match_path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .expect(1)
        .mount(&server)
        .await;

    llm_with_dir(&dir)
        .args(["prompt", "--retries", "3", "--no-stream", "-n", "hello"])
        .env("OPENAI_BASE_URL", server.uri())
        .env("OPENAI_API_KEY", "sk-test")
        .assert()
        .failure()
        .stderr(predicate::str::contains("HTTP error 401"));
}

// ==========================================================================
// Phase 9: Parallel tool execution — CLI plumbing via dry-run JSON
// ==========================================================================

fn dry_run_json(cmd: &mut Command) -> serde_json::Value {
    let output = cmd
        .env_remove("OPENAI_API_KEY")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).unwrap();
    serde_json::from_str(&stdout).unwrap()
}

#[test]
fn agent_run_parallel_defaults_are_parallel() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(&dir, "greeter", "model = \"gpt-4o-mini\"\n");

    let parsed = dry_run_json(llm_with_dir(&dir).args([
        "agent", "run", "greeter", "--dry-run", "--json", "hi",
    ]));
    assert_eq!(parsed["parallel"]["enabled"], true);
    assert!(parsed["parallel"]["max_concurrent"].is_null());
}

#[test]
fn agent_run_parallel_tools_toml_disables() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(
        &dir,
        "serial",
        "model = \"gpt-4o-mini\"\nparallel_tools = false\nmax_parallel_tools = 2\n",
    );

    let parsed = dry_run_json(llm_with_dir(&dir).args([
        "agent", "run", "serial", "--dry-run", "--json", "hi",
    ]));
    assert_eq!(parsed["parallel"]["enabled"], false);
    assert_eq!(parsed["parallel"]["max_concurrent"], 2);
}

#[test]
fn agent_run_parallel_tools_cli_overrides_toml() {
    let dir = TempDir::new().unwrap();
    // TOML says parallel=true; --sequential-tools should flip it.
    write_dry_run_agent(
        &dir,
        "greeter",
        "model = \"gpt-4o-mini\"\nparallel_tools = true\n",
    );

    let parsed = dry_run_json(llm_with_dir(&dir).args([
        "agent",
        "run",
        "greeter",
        "--dry-run",
        "--json",
        "--sequential-tools",
        "hi",
    ]));
    assert_eq!(parsed["parallel"]["enabled"], false);
}

#[test]
fn agent_run_max_parallel_tools_cli_overrides_toml() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(
        &dir,
        "greeter",
        "model = \"gpt-4o-mini\"\nmax_parallel_tools = 2\n",
    );

    let parsed = dry_run_json(llm_with_dir(&dir).args([
        "agent",
        "run",
        "greeter",
        "--dry-run",
        "--json",
        "--max-parallel-tools",
        "8",
        "hi",
    ]));
    assert_eq!(parsed["parallel"]["max_concurrent"], 8);
}

#[test]
fn agent_run_tools_approve_forces_sequential_even_with_max_parallel() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(&dir, "greeter", "model = \"gpt-4o-mini\"\n");

    let parsed = dry_run_json(llm_with_dir(&dir).args([
        "agent",
        "run",
        "greeter",
        "--dry-run",
        "--json",
        "--tools-approve",
        "--max-parallel-tools",
        "8",
        "hi",
    ]));
    assert_eq!(
        parsed["parallel"]["enabled"], false,
        "--tools-approve must force sequential dispatch"
    );
}

#[test]
fn prompt_sequential_tools_flag_parses() {
    // Flag should be accepted by clap and appear in --help.
    llm()
        .args(["prompt", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--sequential-tools"))
        .stdout(predicate::str::contains("--max-parallel-tools"));
}

#[test]
fn chat_sequential_tools_flag_parses() {
    llm()
        .args(["chat", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--sequential-tools"))
        .stdout(predicate::str::contains("--max-parallel-tools"));
}

#[test]
fn agent_run_plain_dry_run_shows_parallel_config() {
    let dir = TempDir::new().unwrap();
    write_dry_run_agent(
        &dir,
        "serial",
        "model = \"gpt-4o-mini\"\nparallel_tools = false\nmax_parallel_tools = 4\n",
    );

    llm_with_dir(&dir)
        .args(["agent", "run", "serial", "--dry-run", "hi"])
        .env_remove("OPENAI_API_KEY")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Parallel:    enabled=false, max_concurrent=4",
        ));
}
