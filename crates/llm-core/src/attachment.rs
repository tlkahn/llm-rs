use base64::Engine;

use crate::error::{LlmError, Result};
use crate::types::{Attachment, AttachmentSource};

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

    let bytes = match &att.source {
        AttachmentSource::Path(p) => std::fs::read(p)?,
        AttachmentSource::Bytes(b) => b.clone(),
        AttachmentSource::Url(url) => {
            return Err(LlmError::Provider(format!(
                "URL attachments cannot be resolved to base64 \
                 (use provider-native URL support): {url}"
            )));
        }
    };

    let base64_data = base64::engine::general_purpose::STANDARD.encode(&bytes);

    Ok(ResolvedAttachment {
        media_type,
        base64_data,
    })
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
}
