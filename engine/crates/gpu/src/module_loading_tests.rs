use super::*;

#[derive(Debug, PartialEq, Eq)]
enum Call { Query, Cubin(Vec<u8>), Ptx(Vec<u8>, bool), Unload(usize) }
struct Fake {
    calls: Vec<Call>, notices: Vec<String>, version: Result<((i32, i32), i32), String>,
    cubin_result: (i32, Option<usize>), ptx_error: Option<(Option<usize>, String)>,
    unload_error: Option<String>, ptx_address: usize,
}
impl Default for Fake {
    fn default() -> Self {
        Self { calls: vec![], notices: vec![], version: Ok(((12, 1), 13000)),
            cubin_result: (0, Some(10)), ptx_error: None, unload_error: None, ptx_address: 0 }
    }
}
impl Driver for Fake {
    type Handle = usize;
    fn device_version(&mut self) -> Result<((i32, i32), i32), String> {
        self.calls.push(Call::Query); self.version.clone()
    }
    fn cubin(&mut self, image: &[u8]) -> (i32, Option<usize>) {
        self.calls.push(Call::Cubin(image.to_vec())); self.cubin_result
    }
    fn ptx(&mut self, ptx: &CStr, receipt: bool) -> Result<Loaded<usize>, Failure<usize>> {
        self.calls.push(Call::Ptx(ptx.to_bytes_with_nul().to_vec(), receipt));
        self.ptx_address = ptx.as_ptr() as usize;
        match &self.ptx_error {
            Some((handle, error)) => Err(Failure { handle: *handle, error: error.clone() }),
            None => Ok(Loaded { handle: 20, receipt: receipt.then_some(3) }),
        }
    }
    fn unload(&mut self, handle: usize) -> Result<(), String> {
        self.calls.push(Call::Unload(handle));
        self.unload_error.clone().map_or(Ok(()), Err)
    }
    fn error(&self, code: i32) -> String { format!("prebuilt cubin load: {code}") }
    fn notice(&mut self, message: &str) { self.notices.push(message.into()); }
}
const ORIGINAL: &CStr = c"// exact original PTX\n.version 8.8\n.target sm_121\n";
const CUBIN: &[u8] = b"compiler image";
fn image() -> Option<Image<'static>> {
    Some(Image { bytes: CUBIN, ptx: ORIGINAL.to_bytes(), minimum_driver: 13000 })
}
fn ptx_call(receipt: bool) -> Call { Call::Ptx(ORIGINAL.to_bytes_with_nul().to_vec(), receipt) }

#[test]
fn successful_cubin_load_transfers_only_the_compiled_handle() {
    let mut driver = Fake::default();
    assert_eq!(load(&mut driver, ORIGINAL, image(), false, false).unwrap(), Loaded { handle: 10, receipt: None });
    assert_eq!(driver.calls, [Call::Query, Call::Cubin(CUBIN.to_vec())]);
    assert_eq!(driver.notices, ["gpu_backend module=prebuilt_cubin target=sm_121"]);
}

#[test]
fn forced_ptx_and_receipts_bypass_cubin_and_queries() {
    for (receipt, forced) in [(false, true), (true, false), (true, true)] {
        let mut driver = Fake { version: Err("must not query".into()), ..Fake::default() };
        assert_eq!(load(&mut driver, ORIGINAL, image(), receipt, forced).unwrap(),
            Loaded { handle: 20, receipt: receipt.then_some(3) });
        assert_eq!(driver.calls, [ptx_call(receipt)]);
        assert_eq!(driver.ptx_address, ORIGINAL.as_ptr() as usize);
        assert!(driver.notices.is_empty());
    }
}

#[test]
fn each_incompatible_code_unloads_partial_image_then_passes_exact_original_ptx() {
    for code in [200, 209, 218, 222] {
        for partial in [None, Some(10)] {
            let mut driver = Fake { cubin_result: (code, partial), ..Fake::default() };
            assert_eq!(load(&mut driver, ORIGINAL, image(), false, false).unwrap(), Loaded { handle: 20, receipt: None });
            let mut calls = vec![Call::Query, Call::Cubin(CUBIN.to_vec())];
            if let Some(handle) = partial { calls.push(Call::Unload(handle)); }
            calls.push(ptx_call(false));
            assert_eq!(driver.calls, calls, "code {code}");
            assert_eq!(driver.ptx_address, ORIGINAL.as_ptr() as usize);
            assert_eq!(driver.notices, [format!("gpu_backend module=ptx reason=incompatible_prebuilt_cubin code={code}")]);
        }
    }
}

#[test]
fn resource_context_and_other_errors_cleanup_without_retry() {
    for code in [1, 2, 201, 700, 999] {
        for partial in [None, Some(10)] {
            let mut driver = Fake { cubin_result: (code, partial), ..Fake::default() };
            assert_eq!(load(&mut driver, ORIGINAL, image(), false, false).unwrap_err(), format!("prebuilt cubin load: {code}"));
            let mut calls = vec![Call::Query, Call::Cubin(CUBIN.to_vec())];
            if let Some(handle) = partial { calls.push(Call::Unload(handle)); }
            assert_eq!(driver.calls, calls);
            assert!(driver.notices.is_empty());
        }
    }
}

#[test]
fn failed_cleanup_stops_fallback_and_preserves_load_error() {
    let mut driver = Fake { cubin_result: (209, Some(10)), unload_error: Some("unload failed".into()), ..Fake::default() };
    assert_eq!(load(&mut driver, ORIGINAL, image(), false, false).unwrap_err(), "prebuilt cubin load: 209; unload failed");
    assert_eq!(driver.calls, [Call::Query, Call::Cubin(CUBIN.to_vec()), Call::Unload(10)]);
    assert!(driver.notices.is_empty());
}

#[test]
fn ptx_and_receipt_load_failures_cleanup_partial_handles() {
    for receipt in [false, true] {
        for partial in [None, Some(20)] {
            for cleanup_error in [None, Some("unload failed".to_string())] {
                let mut driver = Fake { ptx_error: Some((partial, "PTX failed".into())),
                    unload_error: cleanup_error.clone(), ..Fake::default() };
                let error = load(&mut driver, ORIGINAL, None, receipt, false).unwrap_err();
                let mut expected = "PTX failed".to_string();
                let mut calls = vec![ptx_call(receipt)];
                if let Some(handle) = partial {
                    calls.push(Call::Unload(handle));
                    if let Some(cleanup) = cleanup_error { expected.push_str(&format!("; {cleanup}")); }
                }
                assert_eq!(error, expected);
                assert_eq!(driver.calls, calls);
            }
        }
    }
}

#[test]
fn missing_images_and_every_selection_mismatch_use_original_ptx() {
    for (bytes, embedded, device, version, minimum) in [
        (&[][..], ORIGINAL.to_bytes(), (12, 1), 13000, 13000),
        (CUBIN, b"different PTX".as_slice(), (12, 1), 13000, 13000),
        (CUBIN, ORIGINAL.to_bytes(), (12, 0), 13000, 13000),
        (CUBIN, ORIGINAL.to_bytes(), (12, 1), 12999, 13000),
        (CUBIN, ORIGINAL.to_bytes(), (12, 1), 13000, 0),
    ] {
        let mut driver = Fake { version: Ok((device, version)), ..Fake::default() };
        let image = Some(Image { bytes, ptx: embedded, minimum_driver: minimum });
        assert_eq!(load(&mut driver, ORIGINAL, image, false, false).unwrap().handle, 20);
        let expected = if bytes.is_empty() { vec![ptx_call(false)] } else { vec![Call::Query, ptx_call(false)] };
        assert_eq!(driver.calls, expected);
        assert_eq!(driver.ptx_address, ORIGINAL.as_ptr() as usize);
        assert!(driver.notices.is_empty());
    }
    let mut driver = Fake::default();
    load(&mut driver, ORIGINAL, None, false, false).unwrap();
    assert_eq!(driver.calls, [ptx_call(false)]);
}

#[test]
fn query_failure_and_missing_success_handle_are_not_hidden_by_ptx() {
    let mut driver = Fake { version: Err("query failed".into()), ..Fake::default() };
    assert_eq!(load(&mut driver, ORIGINAL, image(), false, false).unwrap_err(), "query failed");
    assert_eq!(driver.calls, [Call::Query]);
    let mut driver = Fake { cubin_result: (0, None), ..Fake::default() };
    assert_eq!(load(&mut driver, ORIGINAL, image(), false, false).unwrap_err(), "prebuilt cubin load returned no module");
    assert_eq!(driver.calls, [Call::Query, Call::Cubin(CUBIN.to_vec())]);
}
