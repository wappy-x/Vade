//! Checked borrowed expert operands. Ordinary calls complete; an owned queue retains dropped allocations until its drain.
use super::*;
use crate::expert_ops::{Shape,Arguments,GroupArguments,GroupEntry,LaunchArguments,Op};
impl<'a> Backend<'a> {
    pub fn expert_operation(&self,shape:Shape,input:[Option<&DeviceVector<'_>>;4],weight:Option<&DeviceVector<'_>>,scales:Option<&DeviceVector<'_>>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let mut arguments=self.expert_arguments(shape,input,weight,scales)?;
        let (_,_,_,units,words)=shape.lengths().ok_or("invalid expert operation shape")?;
        let mut output=self.vector(units*words)?;arguments.output=output.device_address();
        if crate::expert_ops::quantize_lanes_admitted(&shape) {
            // No fill: the region is fully written by nv_quantize::lanes before any read.
            self.skip_fill(&mut output)?;
            // One lane per element: blocks of QUANTIZE_LANES_THREADS threads, eight 16-value blocks each (same bytes, see nv_quantize).
            self.launch_experts(LaunchArguments {single:arguments,batch:0,count:1},(units as u32).div_ceil(crate::expert_ops::QUANTIZE_LANES_THREADS/16),crate::expert_ops::QUANTIZE_LANES_THREADS,0)?;
            self.stats.borrow_mut().projection_dispatches[7]+=1;return Ok(output);
        }
        let rows=if shape.op==Op::NvProject {crate::expert_ops::nv_rows_per_warp(shape.rows,shape.cols,1)} else {32};
        let (grid,threads,shared)=crate::expert_ops::launch_geometry(&[arguments],(units as u32).div_ceil(rows),1)?;
        // No fill on these two bodies: the region is fully written by nv_quantize::lanes or fp8_tile before any read.
        if crate::fill_free::expert_operation(shape.op,threads) {self.skip_fill(&mut output)?;} else {self.zero_vector(&mut output)?;}
        self.launch_experts(LaunchArguments {single:arguments,batch:0,count:1},grid,threads,shared)?;
        if shape.op==Op::Quantize {self.stats.borrow_mut().projection_dispatches[9]+=1;}Ok(output)
    }
    pub fn expert_projection_batch(&self,jobs:&[(Shape,&DeviceVector<'_>,&DeviceVector<'_>,&DeviceVector<'_>)])->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let first=jobs.first().ok_or("expert projection batch is empty")?.0;
        if !matches!(first.op,Op::NvProject|Op::Fp8Project|Op::Fp8Serial|Op::PlainSerial) || jobs.iter().any(|(shape,_,_,_)|shape.op!=first.op || shape.rows!=first.rows || shape.cols!=first.cols) {
            return Err("expert projection batch requires equal complete projection rows".into());
        }
        let count=u32::try_from(jobs.len()).map_err(|_|"expert projection batch exceeds ABI")?;
        let rows=if first.op==Op::NvProject {crate::expert_ops::nv_rows_per_warp(first.rows,first.cols,count)} else if matches!(first.op,Op::Fp8Serial|Op::PlainSerial) {32} else {8};
        let legacy_grid=first.rows.div_ceil(rows).checked_mul(count).ok_or("expert projection batch grid overflow")?;
        let words=(first.rows as usize).checked_mul(jobs.len()).filter(|&words|words<=u32::MAX as usize && words<=isize::MAX as usize/4).ok_or("expert projection batch output overflow")?;
        let mut descriptors=jobs.iter().map(|(shape,input,weight,scales)|self.expert_arguments(*shape,[Some(input),None,None,None],Some(weight),if shape.op==Op::PlainSerial {None} else {Some(scales)})).collect::<Result<Vec<_>,_>>()?;
        let (stream,grid,threads,shared)=crate::expert_ops::batch_launch_geometry(&descriptors,legacy_grid,count)?;
        let (kind,entry)=if stream && shared>0 {(crate::resident_launch::Kind::NvTile,"nv_tile")} else if stream {(crate::resident_launch::Kind::NvStream,"nv_rows_stream")}
            else {(crate::resident_launch::Kind::Expert,"expert_operation")};
        // A needed zero fill runs on the device in stream order. No host zeros are uploaded.
        // No fill for a tile kind: the region is fully written by nv_tile or fp8_tile before any read.
        let mut output=self.vector(words)?;if crate::fill_free::projection_batch(first.op,kind,threads) {self.skip_fill(&mut output)?;} else {self.zero_vector(&mut output)?;}
        for (index,arguments) in descriptors.iter_mut().enumerate() {arguments.output=output.device_address()+(index*first.rows as usize*4) as u64;}
        let words=descriptors.iter().flat_map(|arguments|arguments.descriptor_words()).collect::<Vec<u32>>();
        // A recorded batch keeps its descriptors in the recorder's private area:
        // the same kernel, grid and descriptor bytes, with no upload of its own.
        let bytes=words.iter().flat_map(|word|word.to_ne_bytes()).collect::<Vec<u8>>();self.require_arm_kernels(&[entry])?;
        if self.record_resident_tabled(kind,[grid,1,1,threads,shared],&bytes,|address|{
            let p=LaunchArguments {single:descriptors[0],batch:address,count};match kind {crate::resident_launch::Kind::NvTile=>crate::resident_launch_encode::nv_tile(p),crate::resident_launch::Kind::NvStream=>crate::resident_launch_encode::nv_stream(p),_=>crate::resident_launch_encode::expert(p)}
        })? {
            self.resident_completion()?;if stream {self.stats.borrow_mut().projection_dispatches[if shared>0 {6} else {0}]+=1;}
            if !stream && first.op==Op::Fp8Serial && threads==crate::expert_ops::FP8_TILE_THREADS {self.stats.borrow_mut().projection_dispatches[1]+=1;}return Ok(output);}
        let table=self.upload_vector(&words.into_iter().map(f32::from_bits).collect::<Vec<_>>())?;
        let mut arguments=LaunchArguments {single:descriptors[0],batch:table.device_address(),count};
        if stream {let launched=self.launch_checked_expert_payload(entry,grid,1,threads,shared,&mut arguments);let completed=self.resident_completion();launched.and(completed)?;
            self.stats.borrow_mut().projection_dispatches[if shared>0 {6} else {0}]+=1;
        } else {self.launch_experts(arguments,grid,threads,shared)?;}Ok(output)
    }
    /// Quantize several equal-width activations in ONE launch (the lanes body, a job table of Quantize descriptors).
    /// Each job's output is its own `cols/16*17` word range of the returned vector; the caller takes views of it.
    /// The widths must be admitted by `quantize_lanes_admitted`; activation scales may differ per job.
    pub fn expert_quantize_batch(&self,jobs:&[(Shape,&DeviceVector<'_>)])->Result<(DeviceVector<'a>,usize),String> {
        self.require_resident_stream()?;
        let first=jobs.first().ok_or("expert quantize batch is empty")?.0;
        if jobs.iter().any(|(shape,_)|shape.cols!=first.cols || !crate::expert_ops::quantize_lanes_admitted(shape)) || !crate::quantize_sweep::batch_admitted(first.cols,jobs.len()) {
            return Err("expert quantize batch requires 2 to 9 equal admitted Quantize widths".into());
        }
        let count=u32::try_from(jobs.len()).map_err(|_|"expert quantize batch exceeds ABI")?;
        let (_,_,_,units,words)=first.lengths().ok_or("invalid expert operation shape")?;
        let each=units*words;
        let total=each.checked_mul(jobs.len()).filter(|&n|n<=u32::MAX as usize && n<=isize::MAX as usize/4).ok_or("expert quantize batch output overflow")?;
        let grid=(units as u32).div_ceil(crate::expert_ops::QUANTIZE_LANES_THREADS/16).checked_mul(count).ok_or("expert quantize batch grid overflow")?;
        let mut descriptors=jobs.iter().map(|(shape,input)|self.expert_arguments(*shape,[Some(input),None,None,None],None,None)).collect::<Result<Vec<_>,_>>()?;
        // No fill: the region is fully written by nv_quantize::lanes before any read.
        let mut output=self.vector(total)?;self.skip_fill(&mut output)?;
        for (index,arguments) in descriptors.iter_mut().enumerate() {arguments.output=output.device_address()+(index*each*4) as u64;}
        let words=descriptors.iter().flat_map(|arguments|arguments.descriptor_words()).collect::<Vec<u32>>();
        let bytes=words.iter().flat_map(|word|word.to_ne_bytes()).collect::<Vec<u8>>();self.require_arm_kernels(&["expert_operation"])?;
        if self.record_resident_tabled(crate::resident_launch::Kind::Expert,[grid,1,1,crate::expert_ops::QUANTIZE_LANES_THREADS,0],&bytes,|address|{
            crate::resident_launch_encode::expert(LaunchArguments {single:descriptors[0],batch:address,count})
        })? {self.resident_completion()?;self.stats.borrow_mut().projection_dispatches[8]+=1;return Ok((output,each));}
        let table=self.upload_vector(&words.into_iter().map(f32::from_bits).collect::<Vec<_>>())?;
        let mut arguments=LaunchArguments {single:descriptors[0],batch:table.device_address(),count};
        let launched=self.launch_checked_expert_payload("expert_operation",grid,1,crate::expert_ops::QUANTIZE_LANES_THREADS,0,&mut arguments);
        let completed=self.resident_completion();launched.and(completed)?;self.stats.borrow_mut().projection_dispatches[8]+=1;Ok((output,each))
    }
    pub(super) fn expert_arguments(&self,shape:Shape,input:[Option<&DeviceVector<'_>>;4],weight:Option<&DeviceVector<'_>>,scales:Option<&DeviceVector<'_>>)->Result<Arguments,String> {
        let (lengths,wb,sb,_,_)=shape.lengths().ok_or("invalid expert operation shape")?;
        let mut addresses=[4u64;4];
        for (i,(v,n)) in input.into_iter().zip(lengths).enumerate() {match v {
            Some(v) if n>0=>{self.check_vector(v,n)?;if v.is_segmented() {return Err("expert input must be contiguous".into());}addresses[i]=v.device_address();},
            None if n==0=>{},_=>return Err("expert input count differs".into()),
        }}
        let mut raw=[4u64;2];
        for (i,(v,n)) in [weight,scales].into_iter().zip([wb,sb]).enumerate() {match v {
            Some(v) if n>0=>{self.check_vector(v,n.div_ceil(4))?;if v.is_segmented() {return Err("expert bytes must be contiguous".into());}raw[i]=v.device_address();},
            None if n==0=>{},_=>return Err("expert raw byte count differs".into()),
        }}
        Ok(Arguments {input:addresses,weight:raw[0],scales:raw[1],output:0,shape})
    }
    fn launch_experts(&self,mut arguments:LaunchArguments,grid:u32,threads:u32,shared:u32)->Result<(),String> {
        self.require_arm_kernels(&["expert_operation"])?;
        // SAFETY: Shape checks all slice extents and disjoint output units.
        // check_vector checks exact lengths and context. All buffers are
        // contiguous. Raw owners fund each byte. Output is retained, aligned, private storage. On a
        // fill_free path the region is fully written by nv_quantize::lanes, fp8_tile or nv_tile before
        // any read. On every other path it is initialized before any typed slice or read. By-value arguments are copied at
        // submission. Equal row extents and the checked grid give each batch job a disjoint output span. A single
        // launch keeps lane A's queued completion: queued owner drops defer allocations
        // through the real drain, including a batch's local descriptor table.
        // Ordinary calls complete before return.
        let launched=self.launch_checked_expert_payload("expert_operation",grid,1,threads,shared,&mut arguments);
        let completed=self.resident_completion();launched.and(completed)?;
        if arguments.single.shape.op==Op::Fp8Serial && threads==crate::expert_ops::FP8_TILE_THREADS {self.stats.borrow_mut().projection_dispatches[1]+=1;}Ok(())
    }
    pub(super) fn launch_checked_expert_payload<T>(&self,entry:&str,blocks:u32,rows:u32,threads:u32,shared:u32,arguments:&mut T)->Result<(),String> {
        // SAFETY: Only checked expert calls and the feature-gated referee plans call this
        // boundary. Each caller checks native ABI extents, context, contiguous retained
        // owners, complete warp geometry and disjoint output spans. Each span is fully written by nv_quantize::lanes, fp8_tile or nv_tile before any read on a fill_free path, and initialized otherwise. The
        // payload is copied at submission and all pointed-to owners survive completion.
        unsafe {self.launch_flash_rows(entry,blocks,rows,threads,shared,&mut [parameter(arguments)])}
    }
    /// One launch for one layer's experts. Each part is (shape, raw weight and
    /// scale owners, first input row). Parts own contiguous output units.
    pub fn expert_group(&self,parts:&[(Shape,Option<(&DeviceVector<'_>,&DeviceVector<'_>)>,usize)],input:&DeviceVector<'_>,stride:usize,lut:&DeviceVector<'_>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;self.require_arm_kernels(&["expert_group"])?;
        self.check_vector(lut,crate::expert_ops::LUT_WORDS)?;if lut.is_segmented() {return Err("expert decode table must be contiguous".into());}
        if parts.is_empty() || input.is_segmented() || stride==0 {return Err("expert group is empty or its input is segmented".into());}
        self.check_vector(input,input.len())?;
        let mut table=Vec::with_capacity(parts.len());let mut units=0usize;let mut word_size=None;let mut end_row=0usize;
        for &(shape,raw,first_row) in parts {
            let (lengths,wb,sb,count,words)=shape.lengths().ok_or("invalid expert group shape")?;
            if !matches!(shape.op,crate::expert_ops::Op::QuantizeRows|crate::expert_ops::Op::NvRows|crate::expert_ops::Op::Fp8Rows) {return Err("expert group accepts token rows only".into());}
            if *word_size.get_or_insert(words)!=words || lengths[1..].iter().any(|&n|n!=0) {return Err("expert group word size differs".into());}
            let (weight,scales)=match raw {
                Some((w,s))=>{self.check_vector(w,wb.div_ceil(4))?;self.check_vector(s,sb.div_ceil(4))?;if w.is_segmented() || s.is_segmented() {return Err("expert bytes must be contiguous".into());}(w.device_address(),s.device_address())},
                None if wb==0 && sb==0=>(4,4),_=>return Err("expert group raw byte count differs".into()),
            };
            let tokens=shape.tokens as usize;
            if first_row.checked_mul(stride).and_then(|n|n.checked_add(lengths[0])).is_none_or(|n|n>input.len()) || lengths[0]!=tokens*stride {return Err("expert group input span differs".into());}
            end_row=end_row.max(first_row+tokens);
            table.push(GroupEntry {shape,weight,scales,first:u32::try_from(units).map_err(|_|"expert group units exceed ABI")?,first_row:first_row as u32,
                input_words:lengths[0] as u32,weight_bytes:wb as u32,scale_bytes:sb as u32,pad:0});
            units=units.checked_add(count).ok_or("expert group units overflow")?;
        }
        let words=word_size.ok_or("expert group is empty")?;let total=units.checked_mul(words).ok_or("expert group output overflow")?;
        if u32::try_from(total).is_err() || end_row*stride>input.len() {return Err("expert group output exceeds ABI".into());}
        let bytes=table.iter().flat_map(|e|{let mut raw=Vec::with_capacity(std::mem::size_of::<GroupEntry>());
            raw.extend_from_slice(&(e.shape.op as u32).to_le_bytes());for v in [e.shape.rows,e.shape.cols,e.shape.block_rows,e.shape.block_cols,e.shape.tokens] {raw.extend_from_slice(&v.to_le_bytes());}
            for v in [e.shape.global,e.shape.activation,e.shape.bound] {raw.extend_from_slice(&v.to_le_bytes());}raw.extend_from_slice(&0u32.to_le_bytes());
            raw.extend_from_slice(&e.weight.to_le_bytes());raw.extend_from_slice(&e.scales.to_le_bytes());
            for v in [e.first,e.first_row,e.input_words,e.weight_bytes,e.scale_bytes,e.pad] {raw.extend_from_slice(&v.to_le_bytes());}raw}).collect::<Vec<_>>();
        if bytes.len()!=table.len()*std::mem::size_of::<GroupEntry>() {return Err("expert group table layout differs".into());}
        let mut owner=self.vector(bytes.len().div_ceil(4))?;self.write_vector_bytes(&mut owner,&bytes)?;
        // A direct clear before a direct launch, as in prompt_operation: no piece of one command.
        let mut output=self.vector(total)?;self.unrecorded(||self.zero_vector(&mut output))?;
        let mut args=GroupArguments {table:owner.device_address(),input:input.device_address(),output:output.device_address(),lut:lut.device_address(),count:table.len() as u32,units:units as u32,words:words as u32,stride:stride as u32};
        // NVFP4 rows of one layer run in shared-memory tiles when lane K's predicate admits them, same bytes.
        let nv:Vec<_>=table.iter().map(|e|crate::nv_rows::Part {rows:e.shape.rows as usize,cols:e.shape.cols as usize,tokens:e.shape.tokens as usize,weight:e.weight,scales:e.scales}).collect();
        // Parts of mostly one token waste a block's tile: 15-token prompts measured slower, 127 faster.
        let dense=nv.iter().map(|p|p.tokens).sum::<usize>()>=2*nv.len();
        let tiles=if dense && table.iter().all(|e|e.shape.op==crate::expert_ops::Op::NvRows && e.shape.block_cols==16) && words==1 && crate::nv_rows::admitted(&nv,input.device_address(),lut.device_address()) {crate::nv_rows::grid(&nv)} else {None};
        // FP8 rows likewise, behind lane K's own predicate, same bytes.
        let fp8:Vec<_>=table.iter().map(|e|crate::fp8_rows::Part {rows:e.shape.rows as usize,cols:e.shape.cols as usize,block_rows:e.shape.block_rows as usize,block_cols:e.shape.block_cols as usize,tokens:e.shape.tokens as usize,weight:e.weight,scales:e.scales}).collect();
        let fp8_tiles=if table.iter().all(|e|e.shape.op==crate::expert_ops::Op::Fp8Rows) && words==1 && crate::fp8_rows::admitted(&fp8,input.device_address()) {crate::fp8_rows::grid(&fp8)} else {None};
        // The small-count kernels of a draft verify pass go first (their order and tables are tested in small_choice.rs); then lane K's tiles; then the old kernel.
        let cumulative=crate::nv_small::contiguous(table.iter().map(|e|(e.first_row as usize,e.shape.tokens as usize)));
        let choice=crate::small_choice::group(&crate::small_choice::Group {nv:&nv,fp8:&fp8,words:words as u32,cumulative,input:input.device_address(),
            all_nv:table.iter().all(|e|e.shape.op==crate::expert_ops::Op::NvRows && e.shape.block_cols==16),all_fp8:table.iter().all(|e|e.shape.op==crate::expert_ops::Op::Fp8Rows)});
        let (kernel,tile,threads,shared)=match choice {
            Some((kernel,blocks,threads,shared))=>(kernel,Some([blocks,1,1]),threads,shared),
            None=>match (tiles,fp8_tiles) {
                (Some(grid),_)=>("nv_rows_tile",Some(grid),crate::nv_rows::THREADS,0),
                (_,Some(grid))=>("fp8_rows_tile",Some(grid),crate::fp8_rows::THREADS,0),
                _=>("expert_group",None,32,0),
            },
        };
        if tile.is_some() {self.require_arm_kernels(&[kernel])?;}
        crate::prompt_launches::record(kernel);
        // SAFETY: Each part's shape, raw owners, context, extents and input span
        // are checked above. The table bytes follow the repr(C) layout, checked
        // by size. Buffer::drop drains the stream before any owner is freed. The
        // private output is zeroed earlier on this ordered stream and disjoint.
        // The tile kernels take the same arguments and table. Each traps unless its block size and
        // the alignment and equal shapes its predicate checks all hold.
        let launched=unsafe {match tile {
            Some([tokens,rows,parts])=>self.launch_flash_grid(kernel,tokens,rows,parts,threads,shared,&mut [parameter(&mut args)]),
            None=>self.launch_flash(kernel,(units as u32).div_ceil(threads),threads,&mut [parameter(&mut args)]),
        }};
        launched?;drop(owner);Ok(output)
    }
}
