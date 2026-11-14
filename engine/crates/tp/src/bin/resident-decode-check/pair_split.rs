//! Pair split device gates on ONE device (no model, no RDMA: this tool plays the wire). Every device word is held to
//! the CPU twin (`tc_twin`, `tc_twin_fold`).
//! REDUCE, on both mailbox forms (the inline wire: rows end at the send slot's trailer; the engine's two-write
//! mailbox: rows at the slot's start): the take adds the two partial sums in owner order, with and without the scale
//! and the BF16 round, on both owner roles, at every message size, with the take queued before and after the message
//! arrives, across slot reuse and the sequence wrap. Refusals: a nonfinite partial sum, a timeout, a gather launch
//! on an inline mailbox, a partial sum outside the slot.
//! SPLIT: routed expert down matrices stored whole (one owner), as two column halves (two owners) and as four
//! column quarters (four owners). Raw quarter accumulators, the fold inside each quarter, the owners' partial sums
//! and the finished words: all equal the twin, and one, two and four owners give the same words.
//! FP8 QUARTERS and MX QUARTERS: a block-FP8 and an MXFP8 second matrix held by one, two and four owners: one job
//! (FP8, the tile kernel's raw quarter form) or one projection (MXFP8, today's kernel) a quarter, and the join of an
//! owner's quarters in the quarter-major layout, against `tc_twin_quarter`.
//! RECORDED: the SPLIT and the two QUARTERS gates again with recorded launches; each pair split launch has its kind.
use std::{fs,path::Path,sync::atomic::Ordering};
use spark_tp::Result;
use spark_gpu_probe::{driver::{Backend,DeviceVector,TpMailboxHost,with_backend},expert_ops::{self,Op,Shape,FP8_QUARTER,NV_SECOND,NV_TILE_ROWS},
    mxfp8_projection::Shape as MxShape,norm_boundary::WeightType,tc_twin::{self,Rng},tc_twin_fold as fold,tc_twin_quarter as quarter,tp_mailbox_host::{SEND_LEN,SEND_SEQ,TRAILER},tp_route::Channels};

fn bits(b:&Backend<'_>,v:&DeviceVector<'_>)->Result<Vec<u32>> {Ok(b.read_vector_values(v)?.iter().map(|x|x.to_bits()).collect())}
fn upload_bits<'a>(b:&Backend<'a>,words:&[u32])->Result<DeviceVector<'a>> {b.upload_vector(&words.iter().map(|&w|f32::from_bits(w)).collect::<Vec<_>>())}
fn upload_bytes<'a>(b:&Backend<'a>,bytes:&[u8])->Result<DeviceVector<'a>> {
    b.upload_vector(&bytes.chunks(4).map(|chunk|{let mut word=[0;4];word[..chunk.len()].copy_from_slice(chunk);f32::from_le_bytes(word)}).collect::<Vec<_>>())
}
/// The peer's message as the wire leaves it. Inline wire: rows that end at the receive slot's trailer, then the
/// trailer's sequence word. Two-write wire (the engine's mailbox): rows at the slot's first row word, then RECV_SEQ.
fn deliver(host:&TpMailboxHost<'_>,l:Channels,inline:bool,seq:u64,rows:&[u32]) {
    let words=host.words();let slot=seq as usize%l.slots;let end=l.recv_slot(0,slot)+l.slot_words;
    let (start,seq_word)=if inline {(end-TRAILER-rows.len(),end-1)} else {(l.recv_slot(0,slot)+l.frame,l.recv_seq(0,slot))};
    for (word,&value) in words[start..start+rows.len()].iter().zip(rows) {word.store(value,Ordering::Relaxed);}
    words[seq_word].store(seq as u32,Ordering::Release);
}
/// The rows the put published for message `seq`: its sequence word, its byte length and the words before the trailer.
fn sent(host:&TpMailboxHost<'_>,l:Channels,inline:bool,seq:u64,count:usize)->Result<Vec<u32>> {
    let words=host.words();let slot=seq as usize%l.slots;let end=l.send_slot(slot)+l.slot_words;
    if words[SEND_SEQ+slot].load(Ordering::Acquire)!=seq as u32 || words[SEND_LEN+slot].load(Ordering::Relaxed) as usize!=count*4 {return Err(format!("REDUCE put {seq} did not publish {count} words"));}
    let start=if inline {end-TRAILER-count} else {l.send_slot(slot)+l.frame};
    Ok(words[start..start+count].iter().map(|word|word.load(Ordering::Relaxed)).collect())
}
/// Finite partial sums of the two owners from the fold's hard classes, with a few pairs set by hand: signed zeros,
/// subnormals, exact and near cancellation, a BF16 tie, the largest finite values (their sum is infinite).
fn partials(words:usize,seed:u64)->(Vec<u32>,Vec<u32>) {
    let cases=fold::hard_folds(words,seed);let mut r=Rng(seed^0x5eed);
    let fixed=[(0,0),(0x8000_0000,0x8000_0000),(0,0x8000_0000),(1,0x8000_0001),(0x007f_ffff,1),(0x3f80_0000,0xbf80_0000),(0x3f80_0000,0xbf80_0001),
        (0x3f80_8000,0x3300_0000),(0x7f7f_ffff,0x7f7f_ffff),(0x7f7f_ffff,0xff7f_ffff),(0x7f7f_ffff,0x7300_0000),(0x4170_0000,0x0000_0001)];
    let finite=|word:u32,r:&mut Rng|if word&0x7f80_0000==0x7f80_0000 {(r.next_u64() as u32&0x80ff_ffff)|0x3f00_0000} else {word};
    (0..words).map(|i|{
        if i%7==3 {return fixed[(i/7)%fixed.len()];}
        let q=fold::quarter_fold(&cases[i].experts,cases[i].shared).q;
        (finite(fold::partial(q,0,2),&mut r),finite(fold::partial(q,1,2),&mut r))
    }).unzip()
}
fn reduce_gate(b:&Backend<'_>,receipt:&mut String)->Result<()> {
    let (mut cases,mut checked,mut early)=(0usize,0usize,0usize);
    let scale=0.158_203_13f32;
    // Both mailbox forms: the inline wire (one write a message) and the engine's two-write mailbox.
    for (inline,words) in [true,false].into_iter().flat_map(|inline|[4usize,16,256,4096,8192,16384].into_iter().map(move |words|(inline,words))) {
        let mailbox=if inline {b.tp_mailbox_inline(words*4)?} else {b.tp_mailbox(words*4)?};
        let host=mailbox.take_host().ok_or("REDUCE gate: the mailbox endpoint is taken")?;let l=mailbox.layout();
        // One rising sequence a mailbox, as a wire has: it starts 30 messages before the 32-bit wrap and crosses it.
        // (A sequence that started again at 1 would meet its own low word in a slot's trailer.)
        let mut seq=u32::MAX as u64-30;
        for (mode,(lower,scaled,round)) in [(true,false,false),(false,false,false),(true,false,true),(false,false,true),(true,true,true),(false,true,true),(true,true,false)].into_iter().enumerate() {
            for turn in 0..12usize {
                seq+=1;
                let (own,peer)=partials(words,(words*131+mode*17+turn) as u64);
                let source=upload_bits(b,&own)?;let mut output=upload_bits(b,&vec![0xdead_beef;words])?;
                b.tp_reduce_put(&mailbox,&source,seq)?;b.synchronize()?;
                if sent(&host,l,inline,seq,words)?!=own {return Err(format!("REDUCE put changed its rows inline={inline} words={words} seq={seq}"));}
                let scale=scaled.then_some(scale);
                if turn%2==0 {
                    // The take is queued first and waits for the message.
                    b.tp_reduce_take(&mailbox,&mut output,&source,lower,scale,round,0,seq,2_000_000_000)?;
                    std::thread::scope(|scope|->Result<()> {
                        let arrived=scope.spawn(||{std::thread::sleep(std::time::Duration::from_micros(300));deliver(&host,l,inline,seq,&peer);});
                        b.synchronize()?;arrived.join().map_err(|_|"REDUCE delivery thread failed".to_string())
                    })?;early+=1;
                } else {
                    deliver(&host,l,inline,seq,&peer);
                    b.tp_reduce_take(&mailbox,&mut output,&source,lower,scale,round,0,seq,2_000_000_000)?;b.synchronize()?;
                }
                let got=bits(b,&output)?;
                for i in 0..words {
                    let (low,high)=if lower {(own[i],peer[i])} else {(peer[i],own[i])};
                    let want=fold::reduce_word(low,high,scale,round);
                    if got[i]!=want {return Err(format!("REDUCE word differs inline={inline} words={words} lower={lower} scaled={scaled} round={round} seq={seq} word={i}: device {:08x} twin {want:08x} (low {low:08x} high {high:08x})",got[i]));}
                }
                if bits(b,&source)?!=own {return Err("REDUCE changed this owner's partial sum".into());}
                cases+=1;checked+=words;
            }
        }
        if mailbox.failed() {return Err(format!("REDUCE gate: the mailbox failed at inline={inline} words={words}"));}
    }
    receipt.push_str(&format!("REDUCE_BYTES PASS cases={cases} words={checked} differing_words=0 sizes=4,16,256,4096,8192,16384 mailbox_forms=inline,two_write roles=lower,higher modes=sum,round,scale+round,scale takes_queued_before_arrival={early} slot_reuse=true sequence_wrap=true\n"));
    // A nonfinite partial sum is not published and fails the wire (sticky).
    for bad in [0x7f80_0000u32,0xff80_0000,0x7fc0_0000,0x7f80_0001] {
        let mailbox=b.tp_mailbox_inline(64)?;let host=mailbox.take_host().ok_or("REDUCE gate: the mailbox endpoint is taken")?;
        let mut words=vec![0x3f80_0000u32;16];words[5]=bad;let source=upload_bits(b,&words)?;
        b.tp_reduce_put(&mailbox,&source,1)?;b.synchronize()?;
        if !mailbox.failed() || host.words()[SEND_SEQ+1].load(Ordering::Acquire)!=0 {return Err(format!("REDUCE put published a nonfinite partial sum {bad:08x}"));}
    }
    // A take that times out, and a take on a failed wire, write no output word.
    for failed in [false,true] {
        let mailbox=b.tp_mailbox_inline(64)?;
        if failed {mailbox.fail();}
        let source=upload_bits(b,&[0x3f80_0000;16])?;let mut output=upload_bits(b,&[0xdead_beef;16])?;
        b.tp_reduce_take(&mailbox,&mut output,&source,true,None,true,0,3,1_000)?;b.synchronize()?;
        if !mailbox.failed() || bits(b,&output)?!=vec![0xdead_beef;16] {return Err("a failed REDUCE take changed its output or lost its error".into());}
    }
    // A gather's launches place a message at its slot's start: they refuse an inline mailbox, before anything is queued.
    // (The REDUCE takes both forms.)
    let (inline,plain)=(b.tp_mailbox_inline(64)?,b.tp_mailbox(64)?);
    let source=upload_bits(b,&[0x3f80_0000;16])?;let mut output=upload_bits(b,&[0;16])?;
    if b.tp_mailbox_put_finite(&inline,&source,1,16,16,0,1).is_ok() || b.tp_mailbox_take(&inline,&mut output,1,16,16,0,1,1_000).is_ok() {return Err("a gather launch took an inline mailbox".into());}
    // Shapes outside the slot or the rule are refused.
    let odd=upload_bits(b,&[0;6])?;let long=upload_bits(b,&[0;32])?;
    for mailbox in [&inline,&plain] {
        let taken=b.tp_reduce_put(mailbox,&odd,1).is_ok() || b.tp_reduce_put(mailbox,&long,1).is_ok() || b.tp_reduce_take(mailbox,&mut output,&long,true,None,true,0,1,1_000).is_ok();
        if taken {return Err("REDUCE took a partial sum outside its slot".into());}
    }
    b.synchronize()?;if inline.failed() || plain.failed() {return Err("a refused launch changed a mailbox".into());}
    receipt.push_str("REDUCE_REFUSALS PASS nonfinite_partial=4 timeout=1 failed_wire=1 gather_on_inline=2 outside_slot=6 output_unchanged=true\n");Ok(())
}

/// One second matrix: codes one a byte `[row][col]` and scale codes `[row][block]`, with its epilogue scales.
struct Matrix {codes:Vec<u8>,scales:Vec<u8>,global:f32,activation:f32}
/// Columns `[from, from + cols)` of the matrix as device column tiles (the storage of a column owner).
fn owner_tiles<'a>(b:&Backend<'a>,m:&Matrix,rows:usize,all:usize,from:usize,cols:usize)->Result<(DeviceVector<'a>,DeviceVector<'a>)> {
    let mut packed=vec![0u8;rows*cols/2];let mut scales=vec![0u8;rows*cols/16];
    for row in 0..rows {
        for pair in 0..cols/2 {let at=row*all+from+2*pair;packed[row*(cols/2)+pair]=m.codes[at]|m.codes[at+1]<<4;}
        scales[row*(cols/16)..(row+1)*(cols/16)].copy_from_slice(&m.scales[row*(all/16)+from/16..][..cols/16]);
    }
    Ok((upload_bytes(b,&expert_ops::nv_tile_codes(&packed,rows,cols))?,upload_bytes(b,&expert_ops::nv_tile_scales(&scales,rows,cols))?))
}
/// Activation columns `[from, from + cols)` as the decode kernel reads them: 16 decoded values and the decoded scale a block.
fn owner_input<'a>(b:&Backend<'a>,codes:&[u8],scales:&[u8],from:usize,cols:usize)->Result<DeviceVector<'a>> {
    b.upload_vector(&(from/16..(from+cols)/16).flat_map(|block|{
        let mut values=codes[block*16..block*16+16].iter().map(|&code|expert_ops::e2m1(code)).collect::<Vec<_>>();values.push(expert_ops::e4m3(scales[block]));values
    }).collect::<Vec<_>>())
}
fn shape(rows:usize,cols:usize,m:&Matrix)->Shape {
    Shape {op:Op::NvProject,rows:rows as u32,cols:cols as u32,block_rows:NV_TILE_ROWS,block_cols:16,tokens:1,global:m.global,activation:m.activation,bound:0.0,role:NV_SECOND}
}
fn first_difference(what:&str,got:&[u32],want:&[u32])->Result<()> {
    if got.len()!=want.len() {return Err(format!("{what}: {} device words, {} twin words",got.len(),want.len()));}
    match got.iter().zip(want).position(|(a,b)|a!=b) {Some(at)=>Err(format!("{what}: word {at} differs: device {:08x} twin {:08x}",got[at],want[at])),None=>Ok(())}
}
fn split_gate(b:&Backend<'_>,receipt:&mut String)->Result<()> {
    let (rows,cols,pool)=(4096usize,2048usize,8usize);let quarter=cols/4;
    let mut r=Rng(0xd2_5911);
    // The pool of down matrices. Every seventh row is a row of the twin's hard classes; scale codes stay below 0x7f.
    let matrices=(0..pool).map(|e|{
        let mut codes=vec![0u8;rows*cols];let mut scales=vec![0u8;rows*cols/16];
        for row in 0..rows {
            if row%7==e%7 {let hard=tc_twin::hard_row(cols,r.next_u64());codes[row*cols..(row+1)*cols].copy_from_slice(&hard.w);scales[row*cols/16..(row+1)*cols/16].copy_from_slice(&hard.sw);}
            else {for code in &mut codes[row*cols..(row+1)*cols] {*code=r.below(16) as u8;}
                for scale in &mut scales[row*cols/16..(row+1)*cols/16] {*scale=if r.below(37)==0 {0} else {0x20+r.below(64) as u8};}}
        }
        Matrix {codes,scales,global:0.5+r.below(65536) as f32/65536.0,activation:0.125+e as f32/32.0}
    }).collect::<Vec<_>>();
    // Device storage: whole (one owner), halves (two owners), quarters (four owners).
    let mut stored=Vec::new();
    for parts in [1usize,2,4] {let local=cols/parts;let mut owners=Vec::new();
        for owner in 0..parts {owners.push(matrices.iter().map(|m|owner_tiles(b,m,rows,cols,owner*local,local)).collect::<Result<Vec<_>>>()?);}
        stored.push(owners);
    }
    let (mut tokens,mut raw_words,mut partial_words,mut out_words)=(0usize,0usize,0usize,0usize);
    let pool_scales=b.upload_vector(&matrices.iter().map(|m|m.global*m.activation).collect::<Vec<_>>())?;
    // (routed experts, shared expert, one id in two slots)
    for (token,(experts,with_shared,tie)) in [(8usize,true,false),(8,false,false),(1,true,false),(3,true,false),(8,true,true),(5,false,false)].into_iter().enumerate() {
        // The token's middle vector: hard classes of the twin, then sparse (about a quarter of the values are +-0).
        let hard=tc_twin::hard_row(cols,r.next_u64());let (mut a,sa)=(hard.a,hard.sa.iter().map(|&code|code&0x7f).map(|code|if code==0x7f {0x7e} else {code}).collect::<Vec<_>>());
        for code in &mut a {if r.below(4)==0 {*code&=8;}}
        // The route: distinct ids in a mixed slot order (or one id twice), positive weights.
        let mut ids=(0..pool).collect::<Vec<_>>();for slot in (1..pool).rev() {ids.swap(slot,r.below(slot as u64+1) as usize);}ids.truncate(experts);
        if tie && experts>2 {ids[experts-1]=ids[1];}
        let weights=(0..experts).map(|_|0.03125+r.below(1<<20) as f32/(1u32<<22) as f32).collect::<Vec<_>>();
        let scales=ids.iter().map(|&id|matrices[id].global*matrices[id].activation).collect::<Vec<_>>();
        let shared=with_shared.then(||(0..rows*4).map(|i|if i%97==0 {0} else {(r.next_u64() as u32&0x807f_ffff)|((120+r.below(14) as u32)<<23)}).collect::<Vec<u32>>());
        // The twin: quarter chains, then QUARTER-FOLD v1, for every output row.
        let mut twin_raw=vec![0u32;experts*rows*4];let mut twin=Vec::with_capacity(rows);
        for row in 0..rows {
            let folded=ids.iter().enumerate().map(|(slot,&id)|{let m=&matrices[id];
                let acc:[u32;4]=std::array::from_fn(|q|tc_twin::chain(&m.codes[row*cols+q*quarter..][..quarter],&a[q*quarter..][..quarter],&m.scales[row*cols/16+q*quarter/16..][..quarter/16],&sa[q*quarter/16..][..quarter/16]));
                twin_raw[(slot*rows+row)*4..][..4].copy_from_slice(&acc);
                fold::FoldExpert {id:id as u32,acc,k:scales[slot],w:weights[slot]}
            }).collect::<Vec<_>>();
            twin.push(fold::quarter_fold(&folded,shared.as_ref().map(|s|[s[row*4],s[row*4+1],s[row*4+2],s[row*4+3]])));
        }
        let word=twin.iter().map(|f|f.word).collect::<Vec<_>>();
        let (id_vector,weight_vector,scale_vector)=(b.upload_vector(&ids.iter().map(|&id|id as f32).collect::<Vec<_>>())?,b.upload_vector(&weights)?,b.upload_vector(&scales)?);
        let mut owners_words=Vec::new();
        for (level,parts) in [1usize,2,4].into_iter().enumerate() {
            let (local,quarters)=(cols/parts,4/parts);let mut sums=Vec::new();
            for owner in 0..parts {
                let input=owner_input(b,&a,&sa,owner*local,local)?;
                let jobs=ids.iter().map(|&id|(shape(rows,local,&matrices[id]),&input,&stored[level][owner][id].0,&stored[level][owner][id].1)).collect::<Vec<_>>();
                let acc=b.nv_quarters(&jobs,quarters)?;
                let want=(0..experts*rows).flat_map(|at|twin_raw[at*4+owner*quarters..][..quarters].to_vec()).collect::<Vec<_>>();
                first_difference(&format!("raw quarter accumulators token={token} owners={parts} owner={owner}"),&bits(b,&acc)?,&want)?;raw_words+=want.len();
                let local_shared=shared.as_ref().map(|s|upload_bits(b,&(0..rows).flat_map(|row|s[row*4+owner*quarters..][..quarters].to_vec()).collect::<Vec<_>>())).transpose()?;
                let sum=b.nv_quarter_fold(&id_vector,&weight_vector,&scale_vector,false,&acc,local_shared.as_ref(),false,rows,quarters,parts==1)?;
                let want=twin.iter().map(|f|if parts==1 {f.word} else {fold::partial(f.q,owner,parts)}).collect::<Vec<_>>();
                first_difference(&format!("fold token={token} owners={parts} owner={owner}"),&bits(b,&sum)?,&want)?;partial_words+=rows;
                // The engine's form: the scale of every expert of the layer, picked by the device's ids.
                let by_id=b.nv_quarter_fold(&id_vector,&weight_vector,&pool_scales,true,&acc,local_shared.as_ref(),false,rows,quarters,parts==1)?;
                first_difference(&format!("fold with scales by id token={token} owners={parts} owner={owner}"),&bits(b,&by_id)?,&want)?;partial_words+=rows;
                sums.push(sum);
            }
            let finished=match parts {
                1=>bits(b,&sums[0])?,
                2=>{
                    // The pair's REDUCE: each owner puts its partial sum, this tool carries it to the other, each takes.
                    let mut both=Vec::new();let seq=token as u64+1;
                    let ends=[b.tp_mailbox_inline(rows*4)?,b.tp_mailbox_inline(rows*4)?];
                    let hosts=[ends[0].take_host().ok_or("split gate: endpoint taken")?,ends[1].take_host().ok_or("split gate: endpoint taken")?];
                    for owner in 0..2 {b.tp_reduce_put(&ends[owner],&sums[owner],seq)?;}b.synchronize()?;
                    let moved=[sent(&hosts[0],ends[0].layout(),true,seq,rows)?,sent(&hosts[1],ends[1].layout(),true,seq,rows)?];
                    for owner in 0..2 {
                        deliver(&hosts[owner],ends[owner].layout(),true,seq,&moved[1-owner]);let mut output=b.upload_vector(&vec![0.0;rows])?;
                        b.tp_reduce_take(&ends[owner],&mut output,&sums[owner],owner==0,None,true,0,seq,2_000_000_000)?;b.synchronize()?;both.push(bits(b,&output)?);
                    }
                    if ends[0].failed() || ends[1].failed() {return Err("split gate: a REDUCE mailbox failed".into());}
                    if both[0]!=both[1] {return Err(format!("the two owners hold different words after the REDUCE, token={token}"));}
                    both.swap_remove(0)
                },
                _=>{
                    // Four owners: the exchange of four partial sums is not built. The join here is the twin's.
                    let sent=sums.iter().map(|sum|bits(b,sum)).collect::<Result<Vec<_>>>()?;
                    (0..rows).map(|row|fold::bf16_word(f32::from_bits(fold::reduce(&[sent[0][row],sent[1][row],sent[2][row],sent[3][row]])))).collect()
                },
            };
            first_difference(&format!("finished words token={token} owners={parts}"),&finished,&word)?;out_words+=rows;owners_words.push(finished);
        }
        if owners_words[0]!=owners_words[1] || owners_words[0]!=owners_words[2] {return Err(format!("owner counts give different words, token={token}"));}
        // A second matrix with no fold (an o projection, a dense down): today's one-owner kernel against the pair.
        let id=ids[0];let m=&matrices[id];let k=m.global*m.activation;
        let whole=owner_input(b,&a,&sa,0,cols)?;
        let one=bits(b,&b.expert_projection_batch(&[(shape(rows,cols,m),&whole,&stored[0][0][id].0,&stored[0][0][id].1)])?)?;
        let want=(0..rows).map(|row|fold::single([twin_raw[row*4],twin_raw[row*4+1],twin_raw[row*4+2],twin_raw[row*4+3]],k).1).collect::<Vec<_>>();
        first_difference(&format!("one-owner second matrix (nv_cols) token={token}"),&one,&want)?;
        let mut halves=Vec::new();
        for (owner,held) in stored[1].iter().enumerate() {
            let input=owner_input(b,&a,&sa,owner*cols/2,cols/2)?;
            let acc=b.nv_quarters(&[(shape(rows,cols/2,m),&input,&held[id].0,&held[id].1)],2)?;
            halves.push(b.nv_quarter_join(&acc,false,rows,2)?);
        }
        let ends=[b.tp_mailbox_inline(rows*4)?,b.tp_mailbox_inline(rows*4)?];
        let hosts=[ends[0].take_host().ok_or("split gate: endpoint taken")?,ends[1].take_host().ok_or("split gate: endpoint taken")?];
        for owner in 0..2 {b.tp_reduce_put(&ends[owner],&halves[owner],1)?;}b.synchronize()?;
        let moved=[sent(&hosts[0],ends[0].layout(),true,1,rows)?,sent(&hosts[1],ends[1].layout(),true,1,rows)?];
        for owner in 0..2 {
            let want_partial=(0..rows).map(|row|fold::partial([twin_raw[row*4],twin_raw[row*4+1],twin_raw[row*4+2],twin_raw[row*4+3]],owner,2)).collect::<Vec<_>>();
            first_difference(&format!("plain join token={token} owner={owner}"),&moved[owner],&want_partial)?;
            deliver(&hosts[owner],ends[owner].layout(),true,1,&moved[1-owner]);let mut output=b.upload_vector(&vec![0.0;rows])?;
            b.tp_reduce_take(&ends[owner],&mut output,&halves[owner],owner==0,Some(k),true,0,1,2_000_000_000)?;b.synchronize()?;
            first_difference(&format!("pair second matrix with no fold token={token} owner={owner}"),&bits(b,&output)?,&one)?;out_words+=rows;
        }
        tokens+=1;
    }
    let takes=spark_gpu_probe::tp_mailbox_host::REDUCE_TAKES.load(Ordering::Relaxed);
    receipt.push_str(&format!("SPLIT_BYTES PASS tokens={tokens} rows={rows} cols={cols} raw_quarter_words={raw_words} fold_words={partial_words} finished_words={out_words} differing_words=0 owners=1,2,4 one_owner_equals_pair=true pair_owners_equal=true four_owner_join=twin no_fold_matrix_equals_nv_cols=true scales_by_id=true reduce_takes={takes}\n"));Ok(())
}
/// One token of a quarter gate. `launch(own quarters)` gives an owner's raw quarter accumulators, quarter-major. For
/// one, two and four owners: they, the join of the owner's quarters and the finished words must equal the twin.
/// Returns the raw words and the finished words that were checked.
fn quarter_owners<'d>(b:&Backend<'d>,what:&str,rows:usize,twin:&[[u32;4]],launch:&dyn Fn(std::ops::Range<usize>)->Result<DeviceVector<'d>>)->Result<(usize,usize)> {
    let word=twin.iter().map(|&acc|quarter::second_word(acc).1).collect::<Vec<u32>>();let (mut raw_words,mut out_words)=(0usize,0usize);
    for parts in [1usize,2,4] {
        let quarters=4/parts;let mut sent_words=Vec::new();
        for owner in 0..parts {
            let own=owner*quarters..(owner+1)*quarters;
            let acc=launch(own.clone())?;
            let want=own.flat_map(|q|twin.iter().map(move |acc|acc[q])).collect::<Vec<u32>>();
            first_difference(&format!("{what} raw quarter accumulators owners={parts} owner={owner}"),&bits(b,&acc)?,&want)?;raw_words+=want.len();
            let sum=bits(b,&b.nv_quarter_join(&acc,true,rows,quarters)?)?;
            let want=twin.iter().map(|&acc|fold::partial(acc,owner,parts)).collect::<Vec<u32>>();
            first_difference(&format!("{what} quarter join owners={parts} owner={owner}"),&sum,&want)?;
            sent_words.push(sum);
        }
        // The join of the owners' partial sums and the one BF16 round are the twin's here: the REDUCE gate holds the device exchange to that function.
        let finished=(0..rows).map(|row|fold::bf16_word(f32::from_bits(fold::reduce(&sent_words.iter().map(|words|words[row]).collect::<Vec<_>>())))).collect::<Vec<u32>>();
        first_difference(&format!("{what} finished words owners={parts}"),&finished,&word)?;out_words+=rows;
    }
    Ok((raw_words,out_words))
}
/// The token rows of the quarter gates: finite inputs only, as the REDUCE takes them (ordinary values, signed zeros, subnormals).
fn quarter_tokens(cols:usize,seed:u64)->Vec<quarter::QuarterRow> {
    quarter::hard_rows(quarter::QUARTER_CLASSES*2,cols,128,seed).into_iter().filter(|hard|matches!(hard.class,"ordinary values"|"signed zero inputs"|"subnormal inputs")).collect()
}
fn finite_code(r:&mut Rng)->u8 {let code=r.next_u64() as u8;if code&127==127 {code&0x80|0x38} else {code}}
/// FP8 QUARTERS: a block-FP8 second matrix in the shape of GLM's shared expert down (4,096 x 2,048, [128,128] scale
/// blocks), held by one, two and four owners. One decode job a quarter with the role FP8_QUARTER (the tile kernel's
/// raw form), then the join of the owner's quarters in the quarter-major layout. Every raw quarter accumulator, every
/// partial sum and every finished word equals the twin, and one, two and four owners give the same words.
fn fp8_gate(b:&Backend<'_>,receipt:&mut String)->Result<()> {
    let (rows,cols,block)=(4096usize,2048usize,128usize);let (width,blocks,bands)=(cols/4,cols/4/block,cols/block);
    let mut r=Rng(0xf8d2);
    let codes=(0..rows*cols).map(|_|finite_code(&mut r)).collect::<Vec<u8>>();
    let scales=(0..rows/block*bands).map(|_|f32::from_bits(((121+r.below(9) as u32)<<23)|(r.next_u64() as u32&0x7f_ffff))).collect::<Vec<f32>>();
    // Each quarter as its own matrix: its columns of every row and its scale blocks of every row band.
    let mut held=Vec::new();
    for q in 0..4 {
        let weight=(0..rows).flat_map(|row|codes[row*cols+q*width..][..width].iter().copied()).collect::<Vec<u8>>();
        let scale=(0..rows/block).flat_map(|band|scales[band*bands+q*blocks..][..blocks].iter().copied()).collect::<Vec<f32>>();
        held.push((upload_bytes(b,&weight)?,b.upload_vector(&scale)?));
    }
    let shape=Shape {op:Op::Fp8Serial,rows:rows as u32,cols:width as u32,block_rows:block as u32,block_cols:block as u32,tokens:1,global:1.0,activation:1.0,bound:0.0,role:FP8_QUARTER};
    let (mut tokens,mut raw_words,mut out_words)=(0usize,0usize,0usize);
    for hard in quarter_tokens(cols,0x51de) {
        let x=&hard.x;
        let twin=(0..rows).map(|row|std::array::from_fn::<u32,4,_>(|q|{
            quarter::fp8_chain(&codes[row*cols+q*width..][..width],&scales[row/block*bands+q*blocks..][..blocks],block,&x[q*width..][..width])
        })).collect::<Vec<[u32;4]>>();
        let inputs=(0..4).map(|q|b.upload_vector(&x[q*width..][..width])).collect::<Result<Vec<_>>>()?;
        let checked=quarter_owners(b,&format!("FP8 class={}",hard.class),rows,&twin,&|own|{
            b.fp8_quarters(&own.map(|q|(shape,&inputs[q],&held[q].0,&held[q].1)).collect::<Vec<_>>())
        })?;
        raw_words+=checked.0;out_words+=checked.1;tokens+=1;
    }
    receipt.push_str(&format!("FP8_QUARTER_BYTES PASS tokens={tokens} rows={rows} cols={cols} raw_quarter_words={raw_words} finished_words={out_words} \
        differing_words=0 owners=1,2,4 layout=quarter_major\n"));
    Ok(())
}
/// MX QUARTERS: an MXFP8 second matrix in the shape of GLM's KDA out_proj (4,096 x 8,192), held by one, two and four
/// owners. No new kernel: one projection a column quarter in one launch of today's one-token kernel, raw FP32 sums,
/// quarter-major; then the join of the owner's quarters. Every word equals the twin (`tc_twin_quarter::mx_chain`).
fn mx_gate(b:&Backend<'_>,out:&Path,receipt:&mut String)->Result<()> {
    let (rows,cols)=(4096usize,8192usize);let width=cols/4;
    let mut r=Rng(0x3bd2);
    let codes=(0..rows*cols).map(|_|finite_code(&mut r)).collect::<Vec<u8>>();
    let scales=(0..rows*cols/32).map(|_|(119+r.below(9)) as u8).collect::<Vec<u8>>();
    let shape=MxShape::new(rows,width,WeightType::F32)?;
    // Each quarter as its own matrix: its columns of every row, then their scale bytes, as an arena holds them.
    let mut held=Vec::new();
    for q in 0..4 {
        let mut bytes=(0..rows).flat_map(|row|codes[row*cols+q*width..][..width].iter().copied()).collect::<Vec<u8>>();
        bytes.extend((0..rows).flat_map(|row|scales[(row*cols+q*width)/32..][..width/32].iter().copied()));
        let path=out.join(format!("mx-quarter{q}.raw"));super::write(&path,&bytes)?;let file=fs::File::open(&path).map_err(|error|error.to_string())?;
        let ranges=[spark_gpu_probe::load::WeightRange {file_offset:0,destination:0,bytes:bytes.len(),row_bytes:0,row_stride:0}];
        let arena=b.load_weight_arena(bytes.len(),&[spark_gpu_probe::load::WeightFile {file:&file,ranges:&ranges}])?;
        held.push(b.resident_mxfp8_projection(&arena,shape)?);
    }
    let (mut tokens,mut raw_words,mut out_words)=(0usize,0usize,0usize);
    for hard in quarter_tokens(cols,0x3b5e) {
        let x=&hard.x;
        let twin=(0..rows).map(|row|std::array::from_fn::<u32,4,_>(|q|{
            quarter::mx_chain(&codes[row*cols+q*width..][..width],&scales[(row*cols+q*width)/32..][..width/32],&x[q*width..][..width])
        })).collect::<Vec<[u32;4]>>();
        let inputs=(0..4).map(|q|b.upload_vector(&x[q*width..][..width])).collect::<Result<Vec<_>>>()?;
        let checked=quarter_owners(b,&format!("MXFP8 class={}",hard.class),rows,&twin,&|own|b.mxfp8_quarters(&own.map(|q|(&held[q],&inputs[q])).collect::<Vec<_>>()))?;
        raw_words+=checked.0;out_words+=checked.1;tokens+=1;
    }
    receipt.push_str(&format!("MX_QUARTER_BYTES PASS tokens={tokens} rows={rows} cols={cols} raw_quarter_words={raw_words} finished_words={out_words} \
        differing_words=0 owners=1,2,4 layout=quarter_major\n"));
    Ok(())
}
/// RECORDED: the SPLIT, FP8 QUARTERS and MX QUARTERS gates once more with recorded launches (graph pieces), as the
/// engine runs a decode step. Every word is held to the twin again, and each launch of the pair split must have been
/// recorded under its own kind: a launch with no kind ends the piece it is in.
fn recorded_gate(b:&Backend<'_>,out:&Path,receipt:&mut String)->Result<()> {
    use spark_gpu_probe::resident_launch::Kind;
    let folder=out.join("recorded");fs::create_dir(&folder).map_err(|error|error.to_string())?;
    let (before,mut lines)=(b.stats(),String::new());
    {
        let _queue=b.resident_queue()?;b.enable_resident_launches()?;
        split_gate(b,&mut lines)?;fp8_gate(b,&mut lines)?;mx_gate(b,&folder,&mut lines)?;b.synchronize()?;
    }
    let counts=b.stats().since(before).recorded_kinds;
    let kinds=[Kind::NvQuarters,Kind::QuarterFold,Kind::ReduceTake,Kind::Expert,Kind::Mx];
    if let Some(kind)=kinds.into_iter().find(|&kind|counts[kind as usize]==0) {return Err(format!("the recorded gates recorded no {}",kind.kernel()));}
    let recorded=kinds.map(|kind|format!("{}={}",kind.kernel(),counts[kind as usize])).join(" ");
    receipt.push_str(&format!("RECORDED PASS gates=split,fp8,mx differing_words=0 {recorded}\n"));Ok(())
}
pub fn run(out:&Path)->Result<()> {
    fs::create_dir(out).map_err(|error|error.to_string())?;let mut receipt=String::new();
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b|{
        reduce_gate(b,&mut receipt)?;split_gate(b,&mut receipt)?;fp8_gate(b,&mut receipt)?;mx_gate(b,out,&mut receipt)?;recorded_gate(b,out,&mut receipt)
    })?;
    receipt.push_str("PAIR SPLIT STAGE PASS\n");super::write(&out.join("COMPLETE"),receipt.as_bytes())?;print!("{receipt}");Ok(())
}

/// One token of the chain: the route, this owner's inputs on the device, and the twin's words.
struct ChainToken<'d> {jobs:Vec<usize>,input:DeviceVector<'d>,ids:DeviceVector<'d>,weights:DeviceVector<'d>,scales:DeviceVector<'d>,shared:DeviceVector<'d>,partials:[Vec<u32>;2],word:Vec<u32>}
fn median(values:&mut [u128])->f64 {values.sort_unstable();values[values.len()/2] as f64/1000.0}
fn at(values:&[u128],percent:usize)->f64 {values[(values.len()-1)*percent/100] as f64/1000.0}
/// P-D2b: the REDUCE exchange of two owners on the real wire, in a chain. Each rank runs this with its own rank number;
/// rank 0 connects to `next`, rank 1 listens on `listen`. One exchange of the chain: the down projection of 8 routed
/// experts on this owner's column half (16.8 MB of weights), the quarter fold, then the exchange. `arm` = `inline`
/// (REDUCE put, one write, take-and-add with the BF16 round) or `gather` (today's finite put, two writes, and the
/// plain take of the peer's partial sum: the cost of today's exchange for the same 4,096 words in this rig).
/// Every launch is queued ahead (a run of weight launches goes first), not recorded as graph pieces. The exchange
/// time is the stream time from the event before the put to the event after the take.
pub fn chain(rank:usize,listen:std::net::SocketAddr,next:std::net::SocketAddr,out:&Path,arm:&str,exchanges:usize,repeats:usize)->Result<()> {
    use std::sync::atomic::AtomicBool;
    // The lower owner connects out and its partial sum is the first operand; the higher owner listens.
    let lower=*[true,false].get(rank).ok_or("pair-split-chain: rank 0 or 1")?;
    if exchanges==0 || repeats==0 || !(arm=="inline" || arm=="gather") {return Err("pair-split-chain: arm inline or gather, counts above zero".into());}
    fs::create_dir(out).map_err(|error|error.to_string())?;
    let (rows,cols,pool,kinds)=(4096usize,2048usize,8usize,7usize);let (half,quarter)=(cols/2,cols/4);
    // Both ranks make the same matrices and tokens from one seed; each keeps its own column half on the device.
    let mut r=Rng(0xd2_c4a1);
    let matrices=(0..pool).map(|e|{
        let mut codes=vec![0u8;rows*cols];let mut scales=vec![0u8;rows*cols/16];
        for code in &mut codes {*code=r.below(16) as u8;}
        for scale in &mut scales {*scale=if r.below(37)==0 {0} else {0x20+r.below(64) as u8};}
        Matrix {codes,scales,global:0.5+r.below(65536) as f32/65536.0,activation:0.125+e as f32/32.0}
    }).collect::<Vec<_>>();
    with_backend(0,spark_gpu_probe::layer_ptx()?,1<<30,|b|{
        let inline=arm=="inline";
        let mailbox=if inline {b.tp_mailbox_inline(rows*4)?} else {b.tp_mailbox(rows*4)?};
        let (incoming,outgoing)=spark_seqpar::control::connect_peers((!lower).then_some(listen),lower.then_some(next),std::time::Instant::now()+std::time::Duration::from_secs(120))?;
        let link=incoming.or(outgoing).ok_or("pair-split-chain has no peer")?;
        let wire=spark_tp::tensor_wire::Wire::Write(spark_tp::tensor_wire::WriteLink::new(&link)?);
        let worker=wire.bind(mailbox.take_host().ok_or("pair-split-chain: the mailbox endpoint is taken")?,lower,mailbox.layout())?;
        let (stop,attached,failed)=(AtomicBool::new(false),AtomicBool::new(false),AtomicBool::new(false));
        std::thread::scope(|scope| {
            let handle=spark_tp::tensor_wire::spawn_on(scope,worker,&stop,&attached,&failed,None)?;
            let result=(|| {
                spark_tp::tensor_wire::wait_attached(&attached,&handle)?;
                let stored=matrices.iter().map(|m|owner_tiles(b,m,rows,cols,rank*half,half)).collect::<Result<Vec<_>>>()?;
                let mut tokens=Vec::new();
                for _ in 0..kinds {
                    let hard=tc_twin::hard_row(cols,r.next_u64());let (mut a,sa)=(hard.a,hard.sa.iter().map(|&code|code&0x7f).map(|code|if code==0x7f {0x7e} else {code}).collect::<Vec<_>>());
                    for code in &mut a {if r.below(4)==0 {*code&=8;}}
                    let mut ids=(0..pool).collect::<Vec<_>>();for slot in (1..pool).rev() {ids.swap(slot,r.below(slot as u64+1) as usize);}
                    let weights=(0..pool).map(|_|0.03125+r.below(1<<20) as f32/(1u32<<22) as f32).collect::<Vec<_>>();
                    let scales=ids.iter().map(|&id|matrices[id].global*matrices[id].activation).collect::<Vec<_>>();
                    let shared=(0..rows*4).map(|_|(r.next_u64() as u32&0x807f_ffff)|((120+r.below(14) as u32)<<23)).collect::<Vec<u32>>();
                    let twin=(0..rows).map(|row|{
                        let folded=ids.iter().enumerate().map(|(slot,&id)|{let m=&matrices[id];
                            let acc:[u32;4]=std::array::from_fn(|q|tc_twin::chain(&m.codes[row*cols+q*quarter..][..quarter],&a[q*quarter..][..quarter],&m.scales[row*cols/16+q*quarter/16..][..quarter/16],&sa[q*quarter/16..][..quarter/16]));
                            fold::FoldExpert {id:id as u32,acc,k:scales[slot],w:weights[slot]}
                        }).collect::<Vec<_>>();
                        fold::quarter_fold(&folded,Some([shared[row*4],shared[row*4+1],shared[row*4+2],shared[row*4+3]]))
                    }).collect::<Vec<_>>();
                    tokens.push(ChainToken {input:owner_input(b,&a,&sa,rank*half,half)?,ids:b.upload_vector(&ids.iter().map(|&id|id as f32).collect::<Vec<_>>())?,weights:b.upload_vector(&weights)?,scales:b.upload_vector(&scales)?,
                        shared:upload_bits(b,&(0..rows).flat_map(|row|shared[row*4+rank*2..][..2].to_vec()).collect::<Vec<_>>())?,
                        partials:[0,1].map(|owner|twin.iter().map(|f|fold::partial(f.q,owner,2)).collect()),word:twin.iter().map(|f|f.word).collect(),jobs:ids});
                }
                b.synchronize()?;
                let project=|token:&ChainToken<'_>|->Result<DeviceVector<'_>> {
                    let jobs=token.jobs.iter().map(|&id|(shape(rows,half,&matrices[id]),&token.input,&stored[id].0,&stored[id].1)).collect::<Vec<_>>();
                    let acc=b.nv_quarters(&jobs,2)?;b.nv_quarter_fold(&token.ids,&token.weights,&token.scales,false,&acc,Some(&token.shared),false,rows,2,false)
                };
                let exchange=|partial:&DeviceVector<'_>,output:&mut DeviceVector<'_>,seq:u64|->Result<()> {
                    if inline {b.tp_reduce_put(&mailbox,partial,seq)?;b.tp_reduce_take(&mailbox,output,partial,lower,None,true,0,seq,10_000_000_000)}
                    else {b.tp_mailbox_put_finite(&mailbox,partial,1,rows,rows,0,seq)?;b.tp_mailbox_take(&mailbox,output,1,rows,rows,0,seq,10_000_000_000)}
                };
                let (mut seq,mut spans,mut weights_ns,mut checked,mut report)=(0u64,Vec::new(),Vec::new(),0usize,String::new());
                for repeat in 0..repeats {
                    let mut outputs=(0..=exchanges).map(|_|b.upload_vector(&vec![0.0;rows])).collect::<Result<Vec<_>>>()?;b.synchronize()?;
                    let queue=b.resident_queue()?;
                    // A head start for the host: weight launches that the stream runs while the chain is queued behind them.
                    let mut warm=Vec::new();for lap in 0..48 {warm.push(project(&tokens[lap%kinds])?);}
                    // The first exchange lines the two ranks up. It is not timed.
                    let first=project(&tokens[0])?;seq+=1;exchange(&first,&mut outputs[exchanges],seq)?;
                    let mut marks=Vec::with_capacity(exchanges);let mut kept=Vec::with_capacity(exchanges);
                    for (lap,output) in outputs[..exchanges].iter_mut().enumerate() {
                        let start=b.gpu_mark()?;let partial=project(&tokens[lap%kinds])?;
                        let middle=b.gpu_mark()?;seq+=1;exchange(&partial,output,seq)?;
                        marks.push([start,middle,b.gpu_mark()?]);kept.push(partial);
                    }
                    drop(queue);b.synchronize()?;mailbox.require_success()?;drop(warm);
                    let (mut lap_spans,mut lap_weights)=(Vec::new(),Vec::new());
                    for (lap,output) in outputs[..exchanges].iter().enumerate() {
                        let token=&tokens[lap%kinds];
                        // inline: both ranks hold the finished word. gather: this rank holds the peer's partial sum.
                        first_difference(&format!("chain repeat={repeat} exchange={lap} rank={rank} arm={arm}"),&bits(b,output)?,if inline {&token.word} else {&token.partials[1-rank]})?;checked+=rows;
                        first_difference(&format!("chain partial repeat={repeat} exchange={lap} rank={rank}"),&bits(b,&kept[lap])?,&token.partials[rank])?;
                        lap_weights.push(marks[lap][0].ns_until(&marks[lap][1])?);lap_spans.push(marks[lap][1].ns_until(&marks[lap][2])?);
                    }
                    let line=format!("CHAIN_REPEAT arm={arm} rank={rank} repeat={repeat} exchanges={exchanges} exchange_us median={:.2} p10={:.2} p90={:.2} max={:.2} weight_and_fold_us median={:.2}\n",
                        median(&mut lap_spans),at(&lap_spans,10),at(&lap_spans,90),at(&lap_spans,100),median(&mut lap_weights));
                    print!("{line}");report.push_str(&line);
                    // The first repeat warms the path. It is reported and not counted.
                    if repeat>0 || repeats==1 {spans.extend(lap_spans);weights_ns.extend(lap_weights);}
                }
                let takes=spark_gpu_probe::tp_mailbox_host::REDUCE_TAKES.load(Ordering::Relaxed);
                // Every counted span in time order (ns). The event clock steps by about 2 us here, so the mean of the
                // lowest 80 percent is printed beside the median: it is finer than one step and leaves the tail out.
                super::write(&out.join("SPANS"),spans.iter().map(|ns|format!("{ns}\n")).collect::<String>().as_bytes())?;
                let mut ordered=spans.clone();ordered.sort_unstable();let low=&ordered[..(ordered.len()*4/5).max(1)];
                let low_mean=low.iter().sum::<u128>() as f64/low.len() as f64/1000.0;
                let line=format!("CHAIN_RESULT arm={arm} rank={rank} exchanges_a_chain={exchanges} repeats={repeats} counted_exchanges={} exchange_us mean_low80={low_mean:.2} median={:.2} p10={:.2} p90={:.2} max={:.2} weight_and_fold_us median={:.2} payload_words={rows} weight_mb_between_exchanges={:.1} checked_words={checked} differing_words=0 reduce_takes={takes} queued_ahead=true recorded=false\n",
                    spans.len(),median(&mut spans),at(&spans,10),at(&spans,90),at(&spans,100),median(&mut weights_ns),(pool*rows*half/2+pool*rows*half/16) as f64/1e6);
                print!("{line}");report.push_str(&line);
                super::write(&out.join("COMPLETE"),report.as_bytes())
            })();
            let drained=b.synchronize();let result=result.and(drained);
            if result.is_err() {mailbox.fail();}
            stop.store(true,Ordering::Release);handle.thread().unpark();
            let joined=handle.join().map_err(|_|"pair-split-chain wire panicked".to_owned()).and_then(|v|v);
            result.and(joined)
        })
    })
}
/// `pair-split-chain RANK LISTEN NEXT OUT ARM EXCHANGES REPEATS`.
pub fn chain_arguments(a:[&String;7])->Result<()> {
    chain(a[0].parse().map_err(|_|"pair-split-chain: invalid rank")?,a[1].parse().map_err(|_|"pair-split-chain: invalid listen address")?,a[2].parse().map_err(|_|"pair-split-chain: invalid next address")?,
        Path::new(a[3]),a[4],a[5].parse().map_err(|_|"pair-split-chain: invalid exchange count")?,a[6].parse().map_err(|_|"pair-split-chain: invalid repeat count")?)
}

/// The pair lineup of a checkpoint (no device, no weights: headers only). For every SECOND matrix
/// (`matrix_role::second_matrix`) and every owner of 2 and 4 equal owners: the columns the owner would hold (its
/// whole column quarters) and the FIRST matrices of the same module whose output rows are exactly those columns
/// under today's row split (`Rows::new`). A module lines up when at least one first matrix does, and every first
/// matrix with as many rows as the second matrix has columns does. Also: a quarter must be whole steps of the
/// second matrix's own storage (64 columns for NVFP4, 32 for MXFP8, the block width for block FP8).
pub fn lineup(root:&Path,out:&Path)->Result<()> {
    use engine_model::plan::{layer::Topology,storage::WeightStorage};
    use spark_tp::{matrix_role::second_matrix,resident_weights::Rows};
    fs::create_dir(out).map_err(|error|error.to_string())?;
    let checkpoint=super::open_checkpoint(root)?;let weights=checkpoint.headers().weights();
    let (mut report,mut classes)=(String::new(),std::collections::BTreeMap::<String,[usize;3]>::new());
    for (name,view) in weights {
        let second=view.logical();if second.shape.len()!=2 || !second_matrix(name) {continue;}
        let module=name.strip_suffix(".weight").unwrap_or(name).rsplit_once('.').map_or("",|(module,_)|module);
        let (rows,cols)=(second.shape[0],second.shape[1]);
        let step=match second.storage {WeightStorage::Nvfp4 {..}=>64,WeightStorage::Mxfp8=>32,WeightStorage::Fp8Block {block}=>block[1],WeightStorage::Plain=>16};
        let whole=cols.is_multiple_of(4) && (cols/4).is_multiple_of(step);
        // The first matrices of the module: matrices with the module's prefix that are not second matrices.
        let firsts=weights.iter().filter(|(other,v)|v.logical().shape.len()==2 && !second_matrix(other) && other.strip_suffix(".weight").unwrap_or(other).rsplit_once('.').is_some_and(|(m,_)|m==module)).collect::<Vec<_>>();
        let (mut matched,mut missed,mut names)=(0usize,0usize,Vec::new());
        for (other,v) in &firsts {
            let first=v.logical();if first.shape[0]!=cols {continue;}
            let mut lined=true;
            for parts in [2u32,4] {for rank in 0..parts {
                let held=Rows::new(&first.shape,first.storage,Topology {rank,parts})?;
                let (from,to)=(u64::from(rank)*cols/u64::from(parts),u64::from(rank+1)*cols/u64::from(parts));
                lined&=held.split && held.start as u64==from && (held.start+held.count) as u64==to;
            }}
            if lined {matched+=1;} else {missed+=1;}
            names.push(format!("{}{}",other.rsplit('.').nth(1).unwrap_or(other),if lined {""} else {"(NOT LINED UP)"}));
        }
        let verdict=if !whole {"QUARTER_NOT_WHOLE_STEPS"} else if missed>0 {"FIRST_ROWS_DIFFER"} else if matched==0 {"NO_FIRST_MATRIX_WITH_THESE_ROWS"} else {"LINED_UP"};
        // One line a class of module (the layer and expert numbers replaced), with its count.
        let class=module.split('.').map(|part|if part.chars().all(|c|c.is_ascii_digit()) {"N"} else {part}).collect::<Vec<_>>().join(".");
        let key=format!("{verdict} {class}.{} storage={:?} rows={rows} cols={cols} quarter={} first_with_these_rows=[{}] other_first=[{}]",name.strip_suffix(".weight").unwrap_or(name).rsplit('.').next().unwrap_or(""),second.storage,cols/4,names.join(","),
            firsts.iter().filter(|(_,v)|v.logical().shape[0]!=cols).map(|(other,v)|format!("{}:{}x{}",other.rsplit('.').nth(1).unwrap_or(other),v.logical().shape[0],v.logical().shape[1])).collect::<Vec<_>>().join(","));
        let entry=classes.entry(key).or_insert([0;3]);entry[0]+=1;
    }
    let (mut lined,mut open)=(0usize,0usize);
    for (key,count) in &classes {report.push_str(&format!("PAIR_LINEUP count={} {key}\n",count[0]));if key.starts_with("LINED_UP") {lined+=count[0];} else {open+=count[0];}}
    report.push_str(&format!("PAIR_LINEUP_SUMMARY second_matrices={} lined_up={lined} open={open} owners_checked=2,4\n",lined+open));
    super::write(&out.join("COMPLETE"),report.as_bytes())?;print!("{report}");Ok(())
}
