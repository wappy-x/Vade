//! CHAT ROWS premise. The absorbed attention of several chats, one decode row each: today one chat after another on
//! the decode kernels, against one row group of the verify pass's kernels over one joined latent value (what the
//! batch step now runs). Both arms read the same rows; the Gathers, the selections and the exchange are not timed.
use super::*;
/// The many-chat card row's contexts: c15, c127, c1040, c4158, then four prompts of 1,026 to 1,382 tokens (HOUR-ROWS
/// CX8; the four middle values are stand-ins inside that range).
const CONTEXTS:[usize;8]=[15,127,1040,4158,1026,1145,1264,1382];
/// Rows the selection keeps at this context: GLM's pool of 4, budget of 2,048 and tail.
fn live(slots:usize,context:usize)->usize {
    let kept=spark_tp::bounded_decoder::selected_count(4,2048,true,context as u64) as usize;kept.min(slots)
}
struct Chats<'a> {queries:DeviceVector<'a>,counts:Vec<DeviceVector<'a>>,joined:DeviceVector<'a>,table:DeviceVector<'a>,lives:Vec<usize>}
fn prepare<'a>(b:&Backend<'a>,case:&Case<'a>,chats:usize)->Result<Chats<'a>> {
    let (h,d,stride)=(case.shape.heads as usize,case.shape.dim as usize,case.shape.rows as usize);
    let lives=CONTEXTS[..chats].iter().map(|&context|live(stride,context)).collect::<Vec<_>>();
    let queries=b.upload_vector(&data(chats*h*d,case.layer*3+5))?.into_resident_shared()?;
    let counts=lives.iter().map(|&n|b.upload_vector(&[n as f32])).collect::<std::result::Result<Vec<_>,_>>()?;
    // Chat i's rows at i x stride: the same rows the decode arm reads from the layer's latent.
    let joined=(0..chats).flat_map(|_|case.raw_latent.iter().copied()).collect::<Vec<_>>();
    let mut table=vec![0.0f32;chats*(stride+1)];
    for (chat,(row,&n)) in table.chunks_exact_mut(stride+1).zip(&lives).enumerate() {
        for (slot,id) in row[..stride].iter_mut().enumerate() {*id=(chat*stride+slot) as f32;}row[stride]=n as f32;
    }
    Ok(Chats {queries,counts,joined:b.upload_vector(&joined)?,table:b.upload_vector(&table)?,lives})
}
/// Today: one chat after another, the decode step's five launches a chat.
fn today<'a>(b:&Backend<'a>,case:&Case<'a>,c:&Chats<'a>)->Result<Vec<Vec<DeviceVector<'a>>>> {
    let (h,d,k)=(case.shape.heads as usize,case.shape.dim as usize,case.shape.cols as usize);let mut outputs=Vec::new();
    for (chat,(&n,count)) in c.lives.iter().zip(&c.counts).enumerate() {
        let s=Shape {rows:n as u32,..case.shape};let query=c.queries.resident_range(chat*h*d..(chat+1)*h*d)?;
        let latent=case.latent.resident_range(0..n*k)?;
        let q=b.causal_operation(s,[Some(&query),None,None,None],Some(&case.weight))?;
        let scores=b.causal_operation(Shape {op:Op::AbsorbScores,..s},[Some(&q),Some(&latent),Some(count),None],None)?;
        let p=b.causal_operation(Shape {op:Op::Softmax,taps:0,cols:0,..s},[Some(&scores),Some(count),None,None],None)?;
        let u=b.causal_operation(Shape {op:Op::AbsorbValues,..s},[Some(&p),Some(&latent),Some(count),None],None)?;
        let out=b.causal_operation(Shape {op:Op::AbsorbProject,..s},[Some(&u),None,None,None],Some(&case.weight))?;
        outputs.push(vec![q,u,out]);
    }
    Ok(outputs)
}
/// CHAT ROWS: one row group of the five grouped launches for all chats.
fn rows<'a>(b:&Backend<'a>,case:&Case<'a>,c:&Chats<'a>)->Result<Vec<DeviceVector<'a>>> {
    let chats=c.lives.len();let stride=case.shape.rows as usize;
    let s=Rows {heads:case.shape.heads,dim:case.shape.dim,taps:case.shape.taps,rows:(chats*stride) as u32,cols:case.shape.cols,first:stride as u32,
        slots:stride as u32,..Rows::new(Row::AbsorbQuery,chats).ok_or("chat rows shape")?};
    let q=b.prompt_operation(s,[Some(&c.queries),None,None,None,None],Some(&case.weight))?;
    let scores=b.prompt_operation(Rows {op:Row::AbsorbScores,..s},[Some(&q),Some(&c.joined),Some(&c.table),None,None],None)?;
    let p=b.prompt_operation(Rows {op:Row::AbsorbSoftmax,cols:s.first,..s},[Some(&scores),Some(&c.table),None,None,None],None)?;
    let u=b.prompt_operation(Rows {op:Row::AbsorbValues,..s},[Some(&p),Some(&c.joined),Some(&c.table),None,None],None)?;
    let out=b.prompt_operation(Rows {op:Row::AbsorbProject,..s},[Some(&u),Some(&c.table),None,None,None],Some(&case.weight))?;
    Ok(vec![q,u,out])
}
pub(super) fn run<'a>(b:&Backend<'a>,cases:&[Case<'a>],samples:usize)->Result<()> {
    for chats in 2usize..=8 {
        let prepared=cases.iter().map(|case|prepare(b,case,chats)).collect::<Result<Vec<_>>>()?;b.synchronize()?;
        // Twin: every chat's query, values and output planes of the row group equal its own decode launches, bit for bit.
        for (case,c) in cases.iter().zip(&prepared) {
            let one=today(b,case,c)?;let group=rows(b,case,c)?;
            for (chat,planes) in one.iter().enumerate() {for (plane,(alone,grouped)) in planes.iter().zip(&group).enumerate() {
                let alone=bits(b,alone)?;let grouped=bits(b,grouped)?;let at=chat*alone.len();
                let differing=alone.iter().zip(&grouped[at..at+alone.len()]).filter(|(a,g)|a!=g).count();
                println!("CHATS_TWIN chats={chats} layer={} chat={chat} live={} plane={plane} words={} differing={differing}",case.layer,c.lives[chat],alone.len());
                if differing!=0 {return Err("chat rows differ from the chat alone".into());}
            }}
        }
        let mut wall=[Vec::new(),Vec::new()];let mut event=[Vec::new(),Vec::new()];
        for sample in 0..samples+2 {for arm in if sample%2==0 {[0,1]} else {[1,0]} {
            b.synchronize()?;let time=Instant::now();let start=b.gpu_mark()?;let mut kept=Vec::new();
            for (case,c) in cases.iter().zip(&prepared) {if arm==0 {kept.extend(today(b,case,c)?.into_iter().flatten());} else {kept.extend(rows(b,case,c)?);}}
            let end=b.gpu_mark()?;b.synchronize()?;let elapsed=time.elapsed().as_nanos();let gpu=start.ns_until(&end)?;std::hint::black_box(kept);
            println!("CHATS chats={chats} sample={sample} arm={} wall_ns={elapsed} event_ns={gpu}",["today","rows"][arm]);
            if sample>=2 {wall[arm].push(elapsed);event[arm].push(gpu);}
        }}
        let (old,new,og,ng)=(median(&mut wall[0]),median(&mut wall[1]),median(&mut event[0]),median(&mut event[1]));
        println!("CHATS_RESULT chats={chats} layers={} samples={samples} today_event_ns={og} rows_event_ns={ng} saving_event_ms={:.3} today_wall_ns={old} rows_wall_ns={new} gathers_selection_exchange=not_timed",
            cases.len(),(og as f64-ng as f64)/1e6);
    }
    Ok(())
}
