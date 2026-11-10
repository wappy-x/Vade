//! Bounded transport recovery for read-only Engine metadata.
use super::*;

impl Contract {
    /// Only these two routes are read-only: retry one broken transport,
    /// including a truncated body. Never replay generation, HTTP refusals or
    /// malformed metadata. The owner's task cancellation bounds both attempts.
    pub(super) fn metadata<'a>(
        &mut self,
        method: &str,
        endpoint: &Endpoint,
        body: &[u8],
        pool: &ConnectionPool,
        cancel: Option<&'a AtomicBool>,
    ) -> io::Result<(transport::Response<'a>, Vec<u8>)> {
        for attempt in 0..2 {
            if cancel.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed)) {
                return Err(io::Error::new(io::ErrorKind::ConnectionAborted, "metadata canceled"));
            }
            if method == "GET" { self.context.requests.capabilities += 1; }
            else { self.context.requests.admissions += 1; }
            self.context.last_http_status = None;
            let result = (|| {
                let mut response = if method == "POST" {
                    transport::request_pooled(endpoint, body, None, cancel, pool)?
                } else {
                    transport::request_method_pooled(method, endpoint, body, None, cancel, pool)?
                };
                self.context.last_http_status = Some(response.status);
                self.context.requests.status(response.status);
                let mut bytes = Vec::new();
                response.body.by_ref().take((MAX_METADATA + 1) as u64).read_to_end(&mut bytes)?;
                if bytes.len() > MAX_METADATA { return Err(invalid("metadata exceeds limit")); }
                Ok((response, bytes))
            })();
            match result {
                Err(error) if attempt == 0 && self.context.last_http_status.is_none_or(|status| status == 200)
                    && matches!(error.kind(),
                    io::ErrorKind::UnexpectedEof | io::ErrorKind::BrokenPipe
                    | io::ErrorKind::ConnectionReset | io::ErrorKind::ConnectionAborted
                    | io::ErrorKind::ConnectionRefused | io::ErrorKind::TimedOut) => continue,
                result => return result,
            }
        }
        unreachable!("second metadata attempt returns")
    }
}
