//! One box, no model: the mailbox put at every message size in the exchange census, one-block and wide, direct and recorded.
//! The slot must hold the source words, length and sequence must publish, and a nonfinite word in any block must stop
//! the publish. It also prints the median launch-to-sync time of each form on this box.
use std::{fs,path::Path,sync::atomic::Ordering,time::Instant};
use spark_gpu_probe::{driver::{Backend,set_put_narrow},tp_mailbox_host::{SEND_LEN,SEND_SEQ}};
use spark_tp::Result;
const CENSUS:[usize;10]=[1168,2048,2192,8192,10240,12288,12448,16384,18432,77440];
fn payload(words:usize,seed:u32)->Vec<f32> {
    let bits=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x7f7fffff,0xff7fffff,0x3f800000,0xbf800000];
    (0..words).map(|at|f32::from_bits(if at%3==0 {bits[(at+seed as usize)%bits.len()]} else {0x3f000000|((at as u32).wrapping_mul(2654435761+seed)>>9)})).collect()
}
fn bytes(backend:&Backend<'_>,tally:&mut (usize,usize),lines:&mut String,queued:bool)->Result<()> {
    for (wide_flag,ctas) in [(true,6u32),(false,6),(true,1),(true,96)] {
        let mailbox=backend.tp_mailbox(1<<20)?;let mut host=mailbox.take_host().ok_or("put fixture host already taken")?;host.set_wide(wide_flag);host.set_take_ctas(ctas);
        let layout=mailbox.layout();let mut seq=0u64;
        for narrow in [false,true] {for words in CENSUS {for own_first in [true,false] {
            let row=payload(words*2,words as u32);let source=backend.upload_vector(&row)?;let own=if own_first {0} else {words};
            backend.synchronize()?;let queue=if queued {let queue=backend.resident_queue()?;backend.enable_resident_launches()?;Some(queue)} else {None};
            set_put_narrow(narrow);seq+=1;let sent=backend.tp_mailbox_put_finite(&mailbox,&source,1,words*2,words,own,seq);set_put_narrow(false);sent?;
            drop(queue);backend.synchronize()?;mailbox.require_success()?;
            let slot=seq as usize%layout.slots;let all=host.words();
            if all[SEND_SEQ+slot].load(Ordering::Acquire)!=seq as u32 || all[SEND_LEN+slot].load(Ordering::Acquire)!=(words*4) as u32 {return Err(format!("PUT WIDE did not publish words={words} narrow={narrow}"));}
            let start=layout.send_slot(slot)+layout.frame;
            for at in 0..words {if all[start+at].load(Ordering::Relaxed)!=row[own+at].to_bits() {return Err(format!("PUT WIDE BYTE FAIL words={words} own={own} narrow={narrow} wide_flag={wide_flag} ctas={ctas} word={at}"));}}
            tally.0+=1;tally.1+=words;
        }}}
        lines.push_str(&format!("PUT WIDE BYTES PASS queued={queued} four_word_flag={wide_flag} take_ctas={ctas} sizes={}\n",CENSUS.len()));
    }
    Ok(())
}
fn nonfinite(backend:&Backend<'_>,lines:&mut String)->Result<()> {
    for words in [8192usize,8196,16384] {for at in [0,1,255,256,1535,1536,words/2,words-1] {for bad in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY] {for wide_flag in [true,false] {
        let mailbox=backend.tp_mailbox(1<<20)?;let mut host=mailbox.take_host().ok_or("put fixture host already taken")?;host.set_wide(wide_flag);host.set_take_ctas(6);
        let mut row=payload(words,at as u32);row[at]=bad;let source=backend.upload_vector(&row)?;
        backend.tp_mailbox_put_finite(&mailbox,&source,1,words,words,0,1)?;backend.synchronize()?;
        if !mailbox.failed() || host.words()[SEND_SEQ+1%mailbox.layout().slots].load(Ordering::Acquire)!=0 {return Err(format!("PUT WIDE published a nonfinite row words={words} at={at}"));}
    }}}}
    lines.push_str("PUT WIDE NONFINITE PASS sizes=3 positions=8 values=3 publish_refused=true\n");Ok(())
}
fn timing(backend:&Backend<'_>,lines:&mut String)->Result<()> {
    let mailbox=backend.tp_mailbox(1<<20)?;let mut host=mailbox.take_host().ok_or("put fixture host already taken")?;host.set_wide(true);host.set_take_ctas(6);let mut seq=0u64;
    for words in CENSUS {
        let source=backend.upload_vector(&payload(words,7))?;backend.synchronize()?;let mut medians=[0.0;2];
        for repeat in 0..2 {for (form,narrow) in [false,true].into_iter().enumerate() {
            set_put_narrow(narrow);let mut samples=Vec::with_capacity(400);
            for _ in 0..400 {seq+=1;let started=Instant::now();let sent=backend.tp_mailbox_put_finite(&mailbox,&source,1,words,words,0,seq);let done=backend.synchronize();samples.push(started.elapsed().as_secs_f64()*1e6);sent?;done?;}
            set_put_narrow(false);samples.sort_by(f64::total_cmp);if repeat==1 {medians[form]=samples[200];}
        }}
        lines.push_str(&format!("PUT WIDE TIME words={words} bytes={} direct_median_us narrow_form={:.2} default_form={:.2} saved_us={:.2} wide_form={}\n",words*4,medians[1],medians[0],medians[1]-medians[0],spark_gpu_probe::tp_put_wide::admitted(1,words)));
    }
    Ok(())
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let (mut tally,mut lines)=((0usize,0usize),String::new());
    spark_gpu_probe::driver::with_backend(0,spark_gpu_probe::layer_ptx()?,1<<31,|backend| {
        bytes(backend,&mut tally,&mut lines,false)?;bytes(backend,&mut tally,&mut lines,true)?;
        nonfinite(backend,&mut lines)?;timing(backend,&mut lines)
    })?;
    let text=format!("{lines}PUT WIDE COMPLETE messages={} compared_words={} differing_words=0\n",tally.0,tally.1);
    super::write(&out.join("COMPLETE"),text.as_bytes())?;print!("{text}");Ok(())
}
