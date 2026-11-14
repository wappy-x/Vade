//! Check tar headers and manifests without extracting archive members.
use std::{collections::BTreeMap, fs::{self,File}, io::Read, path::{Path,PathBuf}, process::{Command,Stdio}};
use crate::{Result,is_lower_hex,sha256};
const REQUIRED: &[&str] = &["LICENSE","NOTICE","flash-serve","flash.ptx","flash.cubin.conf","vade","vade-harness","launcher-bundle.conf",
    "engine/LICENSE","engine/NOTICE","engine/crates/tokenizer/UNICODE-NOTICE.txt","engine/crates/tokenizer/QWEN-NOTICE.txt"];
const ARTIFACTS: &[(&str,&str)] = &[("engine","flash-serve"),("ptx","flash.ptx"),("harness","vade-harness"),("vade","vade")];
struct Member { mode: u64, regular: bool, digest: String, data: Vec<u8>, retained: bool }
struct Audit<'a, R> { input: R, data: Vec<u8>, tail: Vec<u8>, forbidden: &'a [String], manifest: bool, retained: bool }
impl<R: Read> Read for Audit<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.input.read(buf)?;
        if self.manifest {
            if self.data.len()+n > 1024*1024 { return Err(std::io::Error::other("manifest too large")); }
            self.data.extend_from_slice(&buf[..n]);
        }
        self.tail.extend_from_slice(&buf[..n]);
        self.retained |= self.forbidden.iter().filter(|s| !s.is_empty()).any(|s| self.tail.windows(s.len()).any(|w| w == s.as_bytes()));
        let keep = self.tail.len().saturating_sub(self.forbidden.iter().map(String::len).max().unwrap_or(0));
        self.tail.drain(..keep); Ok(n)
    }
}
fn field(bytes: &[u8]) -> Result<&str> { Ok(std::str::from_utf8(bytes.split(|&b| b == 0).next().unwrap())?) }
fn octal(bytes: &[u8]) -> Result<u64> { Ok(u64::from_str_radix(field(bytes)?.trim(),8)?) }
fn header(input: &mut impl Read) -> Result<Option<[u8;512]>> {
    let mut h = [0u8;512]; input.read_exact(&mut h)?;
    if h == [0;512] { return Ok(None); }
    let sum: u64 = h.iter().enumerate().map(|(i,&b)| if (148..156).contains(&i) { 32 } else { u64::from(b) }).sum();
    if sum != octal(&h[148..156])? { return Err("invalid tar header checksum".into()); } Ok(Some(h))
}
pub fn check_tar(mut input: impl Read, short: &str, forbidden: &[String]) -> Result<()> {
    let mut members = BTreeMap::new(); let mut long_name = None;
    while let Some(h) = header(&mut input)? {
        let size = octal(&h[124..136])?; let kind = h[156];
        // GNU tar uses a preceding long-name header for paths over 100 bytes.
        if kind == b'L' {
            if size > 1024*1024 || long_name.is_some() { return Err("invalid GNU long name".into()); }
            let mut data = vec![0;size as usize]; input.read_exact(&mut data)?;
            long_name = Some(field(&data)?.to_owned());
        } else {
            let mut name = long_name.take().unwrap_or(field(&h[..100])?.into());
            if &h[257..263] == b"ustar\0" { let prefix = field(&h[345..500])?; if !prefix.is_empty() { name = format!("{prefix}/{name}"); } }
            if kind != b'5' && name.ends_with('/') { return Err(format!("regular archive member ends in '/': {name}").into()); }
            let name = if kind == b'5' { name.trim_end_matches('/').to_owned() } else { name };
            if members.contains_key(&name) { return Err("duplicate archive member".into()); }
            if name.starts_with('/') || name.split('/').any(|s| s == "..") || !matches!(kind,0|b'0'|b'5') {
                return Err(format!("unsafe archive member: {name}").into());
            }
            let mut audit = Audit { input: (&mut input).take(size), data: Vec::new(), tail: Vec::new(), forbidden,
                manifest: matches!(name.as_str(),"launcher-bundle.conf"|"flash.cubin.conf"), retained: false };
            let digest = sha256::digest_reader(&mut audit)?;
            if audit.input.limit() != 0 { return Err("truncated tar member".into()); }
            members.insert(name, Member { mode: octal(&h[100..108])?, regular: matches!(kind,0|b'0'), digest,
                data: audit.data, retained: audit.retained });
        }
        let mut padding = [0;512]; input.read_exact(&mut padding[..((512-size%512)%512) as usize])?;
    }
    // Tar record padding can be larger than the pipe buffer. Consume it before
    // dropping the decompressor's stdout and checking its exit status.
    std::io::copy(&mut input,&mut std::io::sink())?;
    if long_name.is_some() { return Err("missing GNU long-name member".into()); }
    if REQUIRED.iter().any(|name| !members.contains_key(*name)) { return Err("missing bundle member".into()); }
    if REQUIRED.iter().any(|name| !members[*name].regular) { return Err("bundle member is not a regular file".into()); }
    if !members.iter().any(|(name,m)| name.starts_with("engine/licenses/") && m.regular) { return Err("missing engine licenses".into()); }
    if members.keys().any(|n| !REQUIRED.contains(&n.as_str()) && n != "engine" && !n.starts_with("engine/")) { return Err("unexpected bundle member".into()); }
    let manifest = std::str::from_utf8(&members["launcher-bundle.conf"].data)?;
    let mut fields = BTreeMap::new();
    for line in manifest.lines() {
        let (k,v) = line.split_once('=').ok_or("invalid manifest fields")?;
        if fields.insert(k,v).is_some() { return Err("invalid manifest fields".into()); }
    }
    let expected = ["format","target","engine_sha","cli_sha","harness_sha","engine_sha256","ptx_sha256","harness_sha256","vade_sha256"];
    if fields.len() != expected.len() || expected.iter().any(|k| !fields.contains_key(k)) { return Err("invalid manifest fields".into()); }
    if fields["format"] != "vade-launch-bundle-v1" || fields["target"] != "aarch64-unknown-linux-gnu" { return Err("wrong bundle format or target".into()); }
    let revision = fields["engine_sha"];
    if !is_lower_hex(revision,40) || !revision.starts_with(short) { return Err("invalid source revision".into()); }
    if fields["cli_sha"] != revision || fields["harness_sha"] != revision { return Err("source revisions differ".into()); }
    for (key,name) in ARTIFACTS {
        let m = &members[*name];
        if m.digest != fields[format!("{key}_sha256").as_str()] { return Err(format!("manifest hash mismatch: {name}").into()); }
        if *name != "flash.ptx" && m.mode & 0o111 != 0o111 { return Err(format!("missing executable mode: {name}").into()); }
        if m.retained || members["launcher-bundle.conf"].retained { return Err(format!("producer path retained: {name}").into()); }
    }
    if members["flash.cubin.conf"].retained { return Err("producer path retained: flash.cubin.conf".into()); }
    check_cubin(std::str::from_utf8(&members["flash.cubin.conf"].data)?,fields["ptx_sha256"])
}
/// The engine's build script checks the image itself. Here the archive must name the same PTX.
fn check_cubin(conf: &str, ptx: &str) -> Result<()> {
    let mut fields = BTreeMap::new();
    for line in conf.lines() {
        let (k,v) = line.split_once('=').ok_or("invalid cubin fields")?;
        if fields.insert(k,v).is_some() { return Err("invalid cubin fields".into()); }
    }
    let expected = ["format","target","ptx_sha256","cubin_sha256","min_driver_api","ptxas_version","ptxas_sha256","ptxas_args"];
    if fields.len() != expected.len() || expected.iter().any(|k| !fields.contains_key(k)) { return Err("invalid cubin fields".into()); }
    if fields["format"] != "vade-cubin-v1" || fields["target"] != "sm_121" { return Err("wrong cubin format or target".into()); }
    if fields["ptx_sha256"] != ptx { return Err("cubin names other PTX bytes".into()); }
    if !is_lower_hex(fields["cubin_sha256"],64) || !is_lower_hex(fields["ptxas_sha256"],64) { return Err("invalid cubin hash".into()); }
    Ok(())
}
pub fn checksum(text: &str) -> Result<(&str,&str,&str)> {
    let invalid = || "invalid SHA256SUMS";
    let line = text.strip_suffix('\n').ok_or_else(invalid)?;
    let (digest,name) = line.split_once("  ").ok_or_else(invalid)?;
    let short = name.strip_prefix("vade-launch-bundle-v1-").and_then(|s| s.strip_suffix("-aarch64.tar.zst")).ok_or_else(invalid)?;
    if !is_lower_hex(digest,64) || !is_lower_hex(short,7) { return Err(invalid().into()); } Ok((digest,name,short))
}
pub fn check(directory: &Path, forbidden: &[String]) -> Result<PathBuf> {
    let text = fs::read_to_string(directory.join("SHA256SUMS"))?;
    let (digest,name,short) = checksum(&text)?; let archive = directory.join(name);
    if sha256::digest_reader(File::open(&archive)?)? != digest { return Err("compressed archive hash mismatch".into()); }
    let mut child = Command::new("zstd").args(["-q","-d","-c"]).arg(&archive).stdout(Stdio::piped()).spawn()?;
    let checked = check_tar(child.stdout.take().unwrap(),short,forbidden);
    if checked.is_err() { let _ = child.kill(); }
    let status = child.wait()?; checked?;
    if !status.success() { return Err(format!("zstd failed: {status}").into()); } Ok(archive)
}
