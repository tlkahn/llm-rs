use std::fs;
use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;
use wiremock::matchers::{method, path as match_path};
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
