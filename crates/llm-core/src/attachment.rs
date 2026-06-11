use std::borrow::Cow;

use base64::Engine;

use crate::error::{LlmError, Result};
use crate::types::{Attachment, AttachmentSource, Prompt};

/// A resolved attachment with base64-encoded data, ready for embedding
/// in an API request body.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAttachment {
    /// MIME type, e.g. "image/png". Falls back to "application/octet-stream".
    pub media_type: String,
    /// Base64-encoded file content (standard alphabet, with padding).
    pub base64_data: String,
}

/// Resolve an [`Attachment`] into a base64-encoded [`ResolvedAttachment`].
///
/// - `AttachmentSource::Path(p)` -- reads the file at `p` and base64-encodes it.
/// - `AttachmentSource::Bytes(b)` -- base64-encodes the bytes directly.
/// - `AttachmentSource::Url(_)` -- returns `Err(LlmError::Provider(...))` because
///   providers handle URL attachments natively without base64 encoding.
pub fn resolve_to_base64(att: &Attachment) -> Result<ResolvedAttachment> {
    let media_type = att
        .mime_type
        .clone()
        .unwrap_or_else(|| "application/octet-stream".to_string());

    let bytes: Cow<'_, [u8]> = match &att.source {
        AttachmentSource::Path(p) => Cow::Owned(std::fs::read(p)?),
        AttachmentSource::Bytes(b) => Cow::Borrowed(b),
        AttachmentSource::Url(url) => {
            return Err(LlmError::Provider(format!(
                "URL attachments cannot be resolved to base64 \
                 (use provider-native URL support): {url}"
            )));
        }
    };

    let base64_data = base64::engine::general_purpose::STANDARD.encode(&*bytes);

    Ok(ResolvedAttachment {
        media_type,
        base64_data,
    })
}

/// A fully-resolved image attachment ready for provider wire-format mapping.
///
/// Providers iterate over a `Vec<ResolvedImageBlock>` and map each variant
/// to their specific wire struct (Anthropic `ContentBlock`, OpenAI `ContentPart`).
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedImageBlock {
    /// Base64-encoded image data with MIME type.
    Base64 {
        media_type: String,
        base64_data: String,
    },
    /// URL passthrough -- provider sends the URL directly.
    Url {
        url: String,
    },
}

/// Resolve a slice of [`Attachment`]s into [`ResolvedImageBlock`]s.
///
/// - `AttachmentSource::Url` becomes `ResolvedImageBlock::Url` (passthrough).
/// - `AttachmentSource::Path` and `AttachmentSource::Bytes` are base64-encoded
///   via [`resolve_to_base64`].
///
/// Returns an error on the first attachment that fails (e.g. file not found).
pub fn resolve_attachments(attachments: &[Attachment]) -> Result<Vec<ResolvedImageBlock>> {
    attachments
        .iter()
        .map(|att| match &att.source {
            AttachmentSource::Url(url) => Ok(ResolvedImageBlock::Url {
                url: url.clone(),
            }),
            _ => {
                let resolved = resolve_to_base64(att)?;
                Ok(ResolvedImageBlock::Base64 {
                    media_type: resolved.media_type,
                    base64_data: resolved.base64_data,
                })
            }
        })
        .collect()
}

/// Pre-resolve all [`AttachmentSource::Path`] attachments in a [`Prompt`] to
/// [`AttachmentSource::Bytes`], reading each file exactly once.
///
/// Returns `Cow::Borrowed(prompt)` when there are no `Path` attachments (zero-cost).
/// Returns `Cow::Owned(...)` with a cloned prompt whose paths are resolved otherwise.
///
/// Call this **before** entering a retry loop so that file I/O happens once
/// and downstream message-builders never perform blocking filesystem reads.
pub fn resolve_prompt_paths(prompt: &Prompt) -> Result<Cow<'_, Prompt>> {
    let has_path = prompt
        .attachments
        .iter()
        .any(|a| matches!(a.source, AttachmentSource::Path(_)));
    if !has_path {
        return Ok(Cow::Borrowed(prompt));
    }

    let mut resolved = prompt.clone();
    for att in &mut resolved.attachments {
        if let AttachmentSource::Path(p) = &att.source {
            let bytes = std::fs::read(p)?;
            att.source = AttachmentSource::Bytes(bytes);
        }
    }
    Ok(Cow::Owned(resolved))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Attachment, AttachmentSource};
    use std::io::Write;

    #[test]
    fn resolve_bytes() {
        let data = vec![0x89, 0x50, 0x4e, 0x47]; // PNG magic bytes
        let att = Attachment {
            mime_type: Some("image/png".to_string()),
            source: AttachmentSource::Bytes(data.clone()),
        };
        let resolved = resolve_to_base64(&att).unwrap();
        assert_eq!(resolved.media_type, "image/png");
        // Verify round-trip: decode and compare
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&resolved.base64_data)
            .unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn resolve_bytes_no_mime_type_falls_back() {
        let att = Attachment {
            mime_type: None,
            source: AttachmentSource::Bytes(vec![1, 2, 3]),
        };
        let resolved = resolve_to_base64(&att).unwrap();
        assert_eq!(resolved.media_type, "application/octet-stream");
    }

    #[test]
    fn resolve_path() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let content = b"fake image data for testing";
        tmp.write_all(content).unwrap();
        tmp.flush().unwrap();

        let att = Attachment {
            mime_type: Some("image/jpeg".to_string()),
            source: AttachmentSource::Path(tmp.path().to_path_buf()),
        };
        let resolved = resolve_to_base64(&att).unwrap();
        assert_eq!(resolved.media_type, "image/jpeg");
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&resolved.base64_data)
            .unwrap();
        assert_eq!(decoded, content);
    }

    #[test]
    fn resolve_path_nonexistent_returns_io_error() {
        let att = Attachment {
            mime_type: Some("image/png".to_string()),
            source: AttachmentSource::Path("/nonexistent/path/to/file.png".into()),
        };
        let err = resolve_to_base64(&att).unwrap_err();
        assert!(matches!(err, LlmError::Io(_)));
    }

    #[test]
    fn resolve_url_errors() {
        let att = Attachment {
            mime_type: Some("image/png".to_string()),
            source: AttachmentSource::Url("https://example.com/img.png".to_string()),
        };
        let err = resolve_to_base64(&att).unwrap_err();
        assert!(matches!(err, LlmError::Provider(_)));
        let msg = err.to_string();
        assert!(msg.contains("URL attachments cannot be resolved to base64"));
        assert!(msg.contains("https://example.com/img.png"));
    }

    // --- resolve_prompt_paths tests ---

    #[test]
    fn resolve_prompt_paths_no_paths_returns_borrowed() {
        let prompt = Prompt::new("hello").with_attachments(vec![Attachment {
            mime_type: Some("image/png".into()),
            source: AttachmentSource::Bytes(vec![1, 2, 3]),
        }]);
        let result = resolve_prompt_paths(&prompt).unwrap();
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn resolve_prompt_paths_empty_attachments_returns_borrowed() {
        let prompt = Prompt::new("hello");
        let result = resolve_prompt_paths(&prompt).unwrap();
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn resolve_prompt_paths_resolves_file_to_bytes() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let content = b"fake image data";
        tmp.write_all(content).unwrap();
        tmp.flush().unwrap();

        let prompt = Prompt::new("describe").with_attachments(vec![Attachment {
            mime_type: Some("image/jpeg".into()),
            source: AttachmentSource::Path(tmp.path().to_path_buf()),
        }]);

        let result = resolve_prompt_paths(&prompt).unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        let resolved = result.into_owned();
        assert_eq!(resolved.attachments.len(), 1);
        match &resolved.attachments[0].source {
            AttachmentSource::Bytes(b) => assert_eq!(b, content),
            other => panic!("expected Bytes, got {:?}", other),
        }
        assert_eq!(
            resolved.attachments[0].mime_type.as_deref(),
            Some("image/jpeg")
        );
    }

    #[test]
    fn resolve_prompt_paths_mixed_sources() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"file content").unwrap();
        tmp.flush().unwrap();

        let prompt = Prompt::new("describe").with_attachments(vec![
            Attachment {
                mime_type: Some("image/png".into()),
                source: AttachmentSource::Bytes(vec![0x89]),
            },
            Attachment {
                mime_type: Some("image/jpeg".into()),
                source: AttachmentSource::Path(tmp.path().to_path_buf()),
            },
            Attachment {
                mime_type: None,
                source: AttachmentSource::Url("https://example.com/img.png".into()),
            },
        ]);

        let result = resolve_prompt_paths(&prompt).unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        let resolved = result.into_owned();
        // Bytes unchanged
        assert!(
            matches!(&resolved.attachments[0].source, AttachmentSource::Bytes(b) if b == &[0x89])
        );
        // Path resolved to Bytes
        assert!(
            matches!(&resolved.attachments[1].source, AttachmentSource::Bytes(b) if b == b"file content")
        );
        // Url unchanged
        assert!(
            matches!(&resolved.attachments[2].source, AttachmentSource::Url(u) if u == "https://example.com/img.png")
        );
    }

    #[test]
    fn resolve_prompt_paths_nonexistent_file_errors() {
        let prompt = Prompt::new("describe").with_attachments(vec![Attachment {
            mime_type: Some("image/png".into()),
            source: AttachmentSource::Path("/nonexistent/file.png".into()),
        }]);
        let err = resolve_prompt_paths(&prompt).unwrap_err();
        assert!(matches!(err, LlmError::Io(_)));
    }

    // --- resolve_attachments tests ---

    #[test]
    fn resolve_attachments_empty_returns_empty() {
        let result = resolve_attachments(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn resolve_attachments_url_passthrough() {
        let attachments = vec![Attachment {
            mime_type: Some("image/png".into()),
            source: AttachmentSource::Url("https://example.com/a.png".into()),
        }];
        let result = resolve_attachments(&attachments).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0],
            ResolvedImageBlock::Url {
                url: "https://example.com/a.png".into(),
            }
        );
    }

    #[test]
    fn resolve_attachments_bytes_to_base64() {
        let attachments = vec![Attachment {
            mime_type: Some("image/png".into()),
            source: AttachmentSource::Bytes(vec![1, 2, 3]),
        }];
        let result = resolve_attachments(&attachments).unwrap();
        assert_eq!(result.len(), 1);
        let expected_b64 = base64::engine::general_purpose::STANDARD.encode(&[1u8, 2, 3]);
        assert_eq!(
            result[0],
            ResolvedImageBlock::Base64 {
                media_type: "image/png".into(),
                base64_data: expected_b64,
            }
        );
    }

    #[test]
    fn resolve_attachments_path_to_base64() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let content = b"hello";
        tmp.write_all(content).unwrap();
        tmp.flush().unwrap();

        let attachments = vec![Attachment {
            mime_type: Some("image/jpeg".into()),
            source: AttachmentSource::Path(tmp.path().to_path_buf()),
        }];
        let result = resolve_attachments(&attachments).unwrap();
        assert_eq!(result.len(), 1);
        let expected_b64 = base64::engine::general_purpose::STANDARD.encode(content);
        assert_eq!(
            result[0],
            ResolvedImageBlock::Base64 {
                media_type: "image/jpeg".into(),
                base64_data: expected_b64,
            }
        );
    }

    #[test]
    fn resolve_attachments_bad_path_errors() {
        let attachments = vec![Attachment {
            mime_type: Some("image/png".into()),
            source: AttachmentSource::Path("/nonexistent/path/to/file.png".into()),
        }];
        let err = resolve_attachments(&attachments).unwrap_err();
        assert!(matches!(err, LlmError::Io(_)));
    }

    #[test]
    fn resolve_attachments_mixed_sources() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"file data").unwrap();
        tmp.flush().unwrap();

        let attachments = vec![
            Attachment {
                mime_type: Some("image/png".into()),
                source: AttachmentSource::Url("https://example.com/img.png".into()),
            },
            Attachment {
                mime_type: Some("image/jpeg".into()),
                source: AttachmentSource::Bytes(vec![0xFF, 0xD8]),
            },
            Attachment {
                mime_type: Some("image/gif".into()),
                source: AttachmentSource::Path(tmp.path().to_path_buf()),
            },
        ];
        let result = resolve_attachments(&attachments).unwrap();
        assert_eq!(result.len(), 3);
        assert!(matches!(&result[0], ResolvedImageBlock::Url { url } if url == "https://example.com/img.png"));
        assert!(matches!(&result[1], ResolvedImageBlock::Base64 { media_type, .. } if media_type == "image/jpeg"));
        assert!(matches!(&result[2], ResolvedImageBlock::Base64 { media_type, .. } if media_type == "image/gif"));
    }

    #[test]
    fn resolve_to_base64_bytes_does_not_consume_source() {
        let data = vec![0x89, 0x50, 0x4e, 0x47];
        let att = Attachment {
            mime_type: Some("image/png".to_string()),
            source: AttachmentSource::Bytes(data.clone()),
        };
        let _resolved = resolve_to_base64(&att).unwrap();
        // att is still accessible -- function borrows, not moves
        match &att.source {
            AttachmentSource::Bytes(b) => assert_eq!(b, &data),
            _ => panic!("source should still be Bytes"),
        }
    }

    #[test]
    fn resolve_attachments_no_mime_fallback() {
        let attachments = vec![Attachment {
            mime_type: None,
            source: AttachmentSource::Bytes(vec![1, 2, 3]),
        }];
        let result = resolve_attachments(&attachments).unwrap();
        assert_eq!(result.len(), 1);
        assert!(matches!(&result[0], ResolvedImageBlock::Base64 { media_type, .. } if media_type == "application/octet-stream"));
    }

    #[test]
    fn resolve_prompt_paths_preserves_other_fields() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"data").unwrap();
        tmp.flush().unwrap();

        let prompt = Prompt::new("text")
            .with_system("system")
            .with_option("temperature", serde_json::json!(0.5))
            .with_attachments(vec![Attachment {
                mime_type: Some("image/png".into()),
                source: AttachmentSource::Path(tmp.path().to_path_buf()),
            }]);

        let result = resolve_prompt_paths(&prompt).unwrap();
        let resolved = result.as_ref();
        assert_eq!(resolved.text, "text");
        assert_eq!(resolved.system.as_deref(), Some("system"));
        assert_eq!(resolved.options["temperature"], 0.5);
    }
}
