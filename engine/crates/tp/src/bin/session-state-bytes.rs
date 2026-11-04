//! Device allocation and opaque state-copy gate. No checkpoint or model arithmetic.
fn main() {
    if let Err(e)=run() { eprintln!("FAIL session-state-bytes: {e}"); std::process::exit(1); }
}
fn run() -> Result<(),String> {
    let args=std::env::args_os().skip(1).collect::<Vec<_>>();
    let [command,destination]=args.as_slice() else {return Err("usage: session-state-bytes <expected|device> <new-path>".into());};
    if command=="expected" {
        use std::io::Write;
        let bytes=fixture::oracle()?;
        let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(destination).map_err(|e| e.to_string())?;
        file.write_all(&bytes).map_err(|e| e.to_string())?;
        println!("PASS expected-state bytes={} scope=opaque-storage",bytes.len()); return Ok(());
    }
    if command!="device" {return Err("unknown state gate command".into());}
    #[cfg(target_os="linux")]
    {device::run(std::path::Path::new(destination))}
    #[cfg(not(target_os="linux"))]
    {Err("device state gate requires a CUDA Linux host".into())}
}
#[path="session-state-bytes/fixture.rs"]
mod fixture;
#[cfg(target_os="linux")]
mod device {
    use spark_tp::session_state::{*, device::DeviceStore};
    use super::fixture::{layout, next};
    use spark_gpu_probe::driver::with_backend;
    use std::{fs, path::Path};
    type Result<T> = std::result::Result<T,String>;
    fn need(ok: bool,why: &str) -> Result<()> { if ok {Ok(())} else {Err(why.into())} }
    fn contents<S: Store>(s: &Session<S>) -> Result<Vec<Vec<u8>>> {
        (0..s.layout().planes().len()).map(|i| s.read(i)).collect()
    }
    fn advance<S: Store>(s: &mut Session<S>,token: u8) -> Result<()> {
        let old=contents(s)?; let next=next(s.layout(),&old,s.context()+1,token)?;
        let mut a=s.begin_append()?;
        // These copies use resident destinations. The store must preserve all
        // bits, including partial FP32 words. No arithmetic kernel is substituted.
        a.run(&mut ComputeOnly,|store,work| {
            need(store.read(work.scratch)?.iter().all(|&b| b==0),"stream scratch was not reset")?;
            for (a,b) in work.committed.iter().zip(work.staged.iter()) {
                need(store.read(a)?==store.read(b)?,"append copy differs from committed bank")?;
            }
            store.write(work.scratch,&[0xff;258])?;
            for (buffer,bytes) in work.staged.iter_mut().zip(&next) {store.zero(buffer)?; store.write(buffer,bytes)?;}
            Ok((0..next.len()).collect())
        })?;
        a.commit()?; need(contents(s)?==next,"device state differs from host bytes")
    }
    fn trace<S: Store>(s: &Session<S>,out: &mut Vec<u8>) -> Result<()> {
        super::fixture::record(s.context(),&contents(s)?,out);
        Ok(())
    }
    fn bit_patterns<S: Store>(pool: &Pool<S>) -> Result<()> {
        let mut session=pool.session()?; let mut append=session.begin_append()?;
        let pattern=[0,0,0x80,0x7f,0,0,0xc0,0x7f,1,0,0x80,0xff,0,0,0,0x80];
        let bytes=append.layout().planes().iter().map(|p| p.used_bytes(1).map(|n| (0..n).map(|i|pattern[i%pattern.len()]).collect::<Vec<_>>())).collect::<Result<Vec<_>>>()?;
        for (i,b) in bytes.iter().enumerate() {append.write(i,b)?;} append.commit()?;
        need(contents(&session)?==bytes,"raw state bit pattern changed")?;
        let saved=session.save()?; session.restore(&saved)?;
        need(contents(&session)?==bytes,"saved state bit pattern changed")
    }
    fn rejected_replacement<S: Store>(s: &mut Session<S>) -> Result<()> {
        let before=contents(s)?; let context=s.context();
        for invalid_plane in [false,true] {
            let mut append=s.begin_append()?;
            let bytes=next(append.layout(),&before,append.context(),0xfe)?;
            for (i,b) in bytes.iter().enumerate() {append.write(i,b)?;}
            let rejected=if invalid_plane {append.write(bytes.len(),&[])} else {append.write(0,&[])};
            need(rejected.is_err(),"invalid replacement write was accepted")?;
            need(append.commit().is_err(),"rejected replacement committed an earlier staged bank")?;
            need(s.context()==context && contents(s)?==before,"rejected replacement changed committed state")?;
        }
        Ok(())
    }
    fn cell<S: Store>(pool: &Pool<S>,out: &mut Vec<u8>) -> Result<()> {
        let tokens=[1,3,2,7,5,9,11];
        let mut alone=pool.session()?; let mut chunked=pool.session()?; let mut interleaved=pool.session()?;
        need(contents(&alone)?.iter().flatten().all(|&b| b==0),"new state was not cleared")?;
        for &token in &tokens {advance(&mut alone,token)?; trace(&alone,out)?;}
        for chunk in tokens.chunks(2) {for &token in chunk {advance(&mut chunked,token)?; advance(&mut interleaved,token)?;}}
        need(contents(&alone)?==contents(&chunked)? && contents(&alone)?==contents(&interleaved)?,"lane or chunk state differs")?;
        drop(chunked); drop(interleaved);
        let mut cached=pool.session()?;
        for &token in &tokens[..3] {advance(&mut cached,token)?;}
        let saved=cached.save()?; let prefix=contents(&cached)?;
        rejected_replacement(&mut cached)?;
        advance(&mut cached,99)?; advance(&mut cached,98)?; cached.truncate(&saved)?;
        need(contents(&cached)?==prefix,"truncate state differs")?;
        {let mut append=cached.begin_append()?; append.write(0,&[0xff;128])?;}
        need(contents(&cached)?==prefix,"dropped append changed state")?;
        for &token in &tokens[3..] {advance(&mut cached,token)?;}
        need(contents(&cached)?==contents(&alone)?,"cached continuation differs")?;
        trace(&cached,out)?;
        need(cached.begin_append().is_err(),"capacity limit was not enforced")?;
        // Failed partial producer must drain before the next transaction.
        cached.restore(&saved)?;
        {let mut append=cached.begin_append()?;
            need(append.run(&mut ComputeOnly,|store,work| {store.write(&mut work.staged[0],&[0x7f;128])?; Err("injected producer error".into())}).is_err(),"producer failure was ignored")?;
            need(append.commit().is_err(),"failed producer committed")?;
        }
        need(contents(&cached)?==prefix,"failed producer changed state")?;
        drop(cached); let mut fresh=pool.session()?; fresh.restore(&saved)?;
        need(contents(&fresh)?==prefix,"saved bank did not retain independent bytes")?;
        trace(&fresh,out)?; Ok(())
    }
    pub fn run(path: &Path) -> Result<()> {
        fs::create_dir(path).map_err(|e| e.to_string())?;
        let revision=include_str!("../../../../REVISION").trim();
        println!("SESSION_STATE_BYTES revision={revision} scope=opaque-device-storage model_ids=false transport=false");
        let mut reference=None;
        for repeat in 0..3 {
            let mut bytes=Vec::new();
            with_backend(0,spark_gpu_probe::layer_ptx()?,64*1024*1024,|backend| {
                let layout=layout()?;
                let budget=Budget {weights:0,runtime:backend.allocated_bytes(),reserve:0,lanes:3,saved:1,limit:backend.allocation_budget()};
                println!("SESSION_CAPACITY repeat={repeat} bank={} scratch={} required={}",layout.bank_bytes(),layout.scratch_bytes(),budget.required(&layout)?);
                let before=backend.allocated_bytes();
                {let pool=DeviceStore::new(backend).admit(layout,budget)?;
                    bit_patterns(&pool)?;
                    cell(&pool,&mut bytes)?; need(pool.used_slots()==(0,0),"session credits leaked")?;
                }
                backend.synchronize()?;
                need(backend.allocated_bytes()==before,"session device allocations leaked")
            })?;
            fs::write(path.join(format!("state-{repeat}.bin")),&bytes).map_err(|e| e.to_string())?;
            need(bytes==super::fixture::oracle()?,"state trace differs from independent host replay")?;
            if let Some(old)=&reference {need(old==&bytes,"fresh-context state bytes differ")?;} else {reference=Some(bytes.clone());}
            println!("PASS session-state-bytes repeat={repeat} bytes={} alone interleaved chunks cache rollback cleanup",bytes.len());
        }
        fs::write(path.join("SOURCE"),format!("{revision}\n")).map_err(|e| e.to_string())?;
        println!("PASS session-state-bytes repeats=3 model_ids=false"); Ok(())
    }
}
