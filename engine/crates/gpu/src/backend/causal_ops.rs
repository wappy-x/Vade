//! Owned single-token operation outputs. Borrowed inputs live through completion.
use super::*;
use crate::causal_ops::{Shape,Arguments};
pub struct CausalInputs<'s,'d> {slots:[Option<&'s DeviceVector<'d>>;4],checked_rows:bool,bf16:bool}
impl<'s,'d> From<[Option<&'s DeviceVector<'d>>;4]> for CausalInputs<'s,'d> {
    fn from(slots:[Option<&'s DeviceVector<'d>>;4])->Self {Self {slots,checked_rows:false,bf16:false}}
}
impl<'a> Backend<'a> {
    #[cfg(feature="kernel-bench")]
    pub fn fixture_head_attention(&self,shape:Shape,parts:&[&DeviceVector<'_>],query:&DeviceVector<'_>)->Result<[DeviceVector<'a>;3],String> {
        use crate::causal_ops::Op;
        self.require_resident_stream()?;
        if shape.op!=Op::ScoresIndirect || !crate::causal_ops::attention_table_proven(Shape {op:Op::ScoresTable,..shape}) || shape.cols!=256 {
            return Err("fixture head attention shape differs".into());
        }
        self.check_vector(query,shape.heads as usize*shape.dim as usize)?;
        if query.device_address()%16!=0 {return Err("fixture query is unaligned".into());}
        let width=shape.heads as usize*(shape.dim+shape.cols) as usize;
        let mut keys=Vec::new();let mut values=Vec::new();
        for part in parts {
            self.check_vector(part,part.len())?;
            if part.is_segmented() || part.len()%width!=0 || part.device_address()%16!=0 {return Err("fixture row owner differs".into());}
            for row in 0..part.len()/width {
                let address=part.device_address()+(row*width*4) as u64;let value=address+shape.dim as u64*4;
                keys.extend([f32::from_bits(address as u32),f32::from_bits((address>>32) as u32),f32::from_bits(shape.dim+shape.cols),0.0]);
                values.extend([f32::from_bits(value as u32),f32::from_bits((value>>32) as u32),f32::from_bits(shape.dim+shape.cols),0.0]);
            }
        }
        if keys.len()!=shape.taps as usize*4 {return Err("fixture live rows differ".into());}
        let keys=self.upload_vector(&keys)?;let values=self.upload_vector(&values)?;
        let mut scores=self.vector(shape.heads as usize*shape.rows as usize)?;
        self.resident_causal_into(Shape {op:Op::ScoresTable,..shape},CausalInputs {slots:[Some(query),Some(&keys),None,None],checked_rows:true,bf16:false},None,&mut scores)?;
        let count=self.upload_vector(&[shape.taps as f32])?;
        let probabilities=self.causal_operation(Shape {op:Op::Softmax,taps:0,cols:0,..shape},[Some(&scores),Some(&count),None,None],None)?;
        let mut output=self.vector(shape.heads as usize*shape.cols as usize)?;
        self.resident_causal_into(Shape {op:Op::ValuesTable,dim:shape.cols,cols:shape.dim,..shape},CausalInputs {slots:[Some(&probabilities),Some(&values),None,None],checked_rows:true,bf16:false},None,&mut output)?;
        Ok([scores,probabilities,output])
    }
    pub fn causal_operation(&self,shape:Shape,input:[Option<&DeviceVector<'_>>;4],weight:Option<&DeviceVector<'_>>)->Result<DeviceVector<'a>,String> {
        let (_,_,units,words)=shape.lengths().ok_or("invalid causal operation shape")?;
        let mut output=self.vector(units*words)?;self.resident_causal_into(shape,input,weight,&mut output)?;Ok(output)
    }
    /// Caller-owned stable storage for an ordered graph piece. Inputs and
    /// weights may alias each other, but the private output cannot alias either.
    pub fn resident_causal_into<'s,'d:'s>(&self,shape:Shape,input:impl Into<CausalInputs<'s,'d>>,weight:Option<&DeviceVector<'_>>,output:&mut DeviceVector<'_>)->Result<(),String> {
        // `bf16` (only `project_bf16_into` sets it): the Project weight is rows x cols BF16 patterns packed two to a word (half the words) and
        // `project_bf16_ordered` runs instead of `project_f32_ordered`.
        let CausalInputs {slots:input,checked_rows,bf16}=input.into();
        self.require_resident_stream()?;
        use crate::causal_ops::Op;
        let table_form=matches!(shape.op,Op::ScoresTable|Op::ValuesTable|Op::ScoresPacked|Op::ValuesPacked);
        if matches!(shape.op,Op::ScoresIndirect|Op::ValuesIndirect|Op::ScoresTable|Op::ValuesTable|Op::ScoresPacked|Op::ValuesPacked) && !checked_rows {return Err("indirect attention requires checked row owners".into());}
        if table_form && !crate::causal_ops::attention_table_proven(shape) {return Err("attention table shape is outside the byte-proven family".into());}
        let (lengths,weights,units,words)=shape.lengths().ok_or("invalid causal operation shape")?;
        if bf16 && (shape.op!=Op::Project || weights%2!=0) {return Err("the BF16 form is the plain Project only".into());}
        let weights=if bf16 {weights/2} else {weights};
        let mut addresses=[4u64;4];
        for (i,(vector,count)) in input.into_iter().zip(lengths).enumerate() {
            match vector {Some(v) if count>0=>{self.check_vector(v,count)?;
                if v.is_segmented() {return Err("causal operation input must be contiguous".into());}addresses[i]=v.device_address();},
                None if count==0=>{},_=>return Err("causal operation input count differs".into())}
        }
        let address=match weight {Some(w) if weights>0=>{self.check_vector(w,weights)?;
            if w.is_segmented() {return Err("causal operation weight must be contiguous".into());}w.device_address()},
            None if weights==0=>4,_=>return Err("causal operation weight count differs".into())};
        let fast=crate::causal_ops::fast_admitted(shape,crate::causal_ops::FAST_THREADS);
        if matches!(shape.op,Op::StepPlanar|Op::ConvParts) && !fast {return Err("planar causal operation is outside the byte-swept family".into());}
        self.require_arm_kernels(&[if shape.op==Op::AbsorbScores {"latent_absorption_scores"} else if shape.op==Op::AbsorbValues {"latent_absorption_values"} else if crate::latent_absorb::is_absorb(shape.op) {"latent_absorption"} else if fast {"causal_fast"} else {"causal_operation"},if bf16 {"project_bf16_ordered"} else {"project_f32_ordered"}])?;
        self.check_vector(output,units*words)?;
        if output.is_segmented() || !output.writable_from(0) {return Err("resident output requires private contiguous storage".into());}
        let start=output.device_address();let end=start+output.len() as u64*4;
        for source in input.into_iter().flatten().chain(weight) {let at=source.device_address();let last=at+source.len() as u64*4;
            if start<last && at<end {return Err("resident output overlaps an input".into());}}
        if !crate::causal_ops::writes_every_word(shape.op) {self.zero_vector(output)?;}
        self.resident_completion()?;
        let mut args=Arguments {input:addresses,weight:address,output:output.device_address(),shape};
        let (mut w,mut x,mut y)=(address,addresses[0],output.device_address());
        let (mut rows,mut cols)=(shape.rows,shape.cols);
        // SAFETY: Shape bounds every indexed read and output unit. check_vector
        // checks all nonempty owners, contexts, and exact lengths. Inputs are
        // read-only. The private output is initialized, disjoint, and contiguous. Empty spans
        // use aligned nonnull addresses. Owners live through stream completion.
        // Indirect rows are accepted only from resident_row_attention below. It
        // builds the immutable address table from checked same-context row owners.
        // Each raw row covers heads*(key_dim+value_dim) initialized words.
        // Plane offsets and head strides stay inside that row. The queue retains
        // dropped owners until completion, or completion waits here.
        // The table form has separate key/value entries with a checked head
        // stride. Its query and row bases are 16-byte aligned. The proven
        // shape supplies each kernel's complete native launch geometry.
        let launched=unsafe {if shape.op==crate::causal_ops::Op::Project {
            // This resident-only operation keeps separate RN FP32 products and
            // ascending adds. The complete row and weight extents are checked above.
            // The BF16 entry keeps the geometry it was swept in; the FP32 entry takes the fold and short-row rules of plain_geometry.
            {let (blocks,threads)=if bf16 {crate::plain_admission::bf16_geometry(units)} else {crate::plain_admission::plain_geometry(units,shape.cols as usize)};self.launch_flash(if bf16 {"project_bf16_ordered"} else {"project_f32_ordered"},blocks,threads,
                &mut [parameter(&mut w),parameter(&mut x),parameter(&mut y),parameter(&mut rows),parameter(&mut cols)])}
        } else if table_form {
            let threads=if matches!(shape.op,Op::ScoresTable|Op::ScoresPacked) {crate::causal_ops::ATTENTION_TABLE_SCORES_THREADS} else {crate::causal_ops::ATTENTION_TABLE_VALUES_THREADS};
            self.launch_flash("causal_operation",crate::causal_ops::attention_table_blocks(shape),threads,&mut [parameter(&mut args)])
        } else {let (blocks,threads)=if shape.op==Op::AbsorbScores {(crate::causal_ops::absorb_scores_blocks(shape),128)} else if shape.op==Op::AbsorbValues {(shape.heads.div_ceil(4)*shape.cols.div_ceil(32),1024)} else if crate::latent_absorb::is_absorb(shape.op) {(units.div_ceil(4) as u32,128)} else {crate::causal_ops::geometry(shape,units)};self.launch_flash(if shape.op==Op::AbsorbScores {"latent_absorption_scores"} else if shape.op==Op::AbsorbValues {"latent_absorption_values"} else if crate::latent_absorb::is_absorb(shape.op) {"latent_absorption"} else if fast {"causal_fast"} else {"causal_operation"},blocks,threads,&mut [parameter(&mut args)])}};
        let completed=self.resident_completion();let result=launched.and(completed);
        #[cfg(feature="kernel-bench")]
        if result.is_ok() && table_form {let slot=usize::from(matches!(shape.op,Op::ValuesTable|Op::ValuesPacked));self.stats.borrow_mut().attention_dispatches[slot]+=1;}
        result
    }
    /// The plain single-token projection with BF16 weights (`project_bf16_ordered`). `weight` holds rows x cols BF16 patterns packed two to a
    /// word (rows * cols / 2 words, the checkpoint's own bytes); the f32 the chain multiplies is the pattern shifted left 16 bits, the loader's
    /// widening, so the output equals `causal_operation` on the widened FP32 copy bit for bit.
    pub fn project_bf16(&self,rows:usize,cols:usize,input:&DeviceVector<'_>,weight:&DeviceVector<'_>)->Result<DeviceVector<'a>,String> {
        let mut output=self.vector(rows)?;self.project_bf16_into(rows,cols,input,weight,&mut output)?;Ok(output)
    }
    pub fn project_bf16_into(&self,rows:usize,cols:usize,input:&DeviceVector<'_>,weight:&DeviceVector<'_>,output:&mut DeviceVector<'_>)->Result<(),String> {
        use crate::causal_ops::Op;
        if rows==0 || cols==0 || !cols.is_multiple_of(8) {return Err("BF16 projection needs rows and columns that are a multiple of 8".into());}
        if !crate::plain_admission::bf16_listed(rows,cols) {return Err("BF16 projection shape is outside the listed step shapes".into());}
        let shape=Shape {op:Op::Project,heads:0,dim:0,taps:0,rows:u32::try_from(rows).map_err(|_|"BF16 projection rows exceed the ABI")?,cols:u32::try_from(cols).map_err(|_|"BF16 projection columns exceed the ABI")?,epsilon:0.0,bound:0.0};
        self.resident_causal_into(shape,CausalInputs {slots:[Some(input),None,None,None],checked_rows:false,bf16:true},Some(weight),output)
    }
    pub fn resident_row_attention(&self,shape:Shape,parts:&[&DeviceVector<'_>],query:&DeviceVector<'_>)->Result<[DeviceVector<'a>;3],String> {
        use crate::causal_ops::Op;
        self.require_resident_stream()?;
        if shape.op!=Op::ScoresIndirect {return Err("indirect attention requires a scores shape".into());}
        if !crate::causal_ops::indirect_attention_proven(shape) {return Err("indirect attention shape is outside the byte-proven family".into());}
        if shape.taps>1<<24 {return Err("indirect live count exceeds exact FP32 integers".into());}
        let (_,_,units,_)=shape.lengths().ok_or("invalid indirect attention shape")?;
        let width=(shape.heads as usize).checked_mul((shape.dim as usize).checked_add(shape.cols as usize).ok_or("indirect row width overflow")?).ok_or("indirect row width overflow")?;
        self.check_vector(query,shape.heads as usize*shape.dim as usize)?;
        let table_form=query.device_address().is_multiple_of(16) && parts.iter().all(|part|part.device_address()%16==0);
        // One table entry a live row: the extent is known, so neither table grows while it is filled.
        let entries=shape.taps as usize*if table_form {4} else {2};
        let mut addresses=Vec::with_capacity(entries);let mut value_addresses=Vec::with_capacity(if table_form {entries} else {0});
        for part in parts {
            self.check_vector(part,part.len())?;
            if part.is_segmented() || part.len()%width!=0 {return Err("indirect row owner extent differs".into());}
            for row in 0..part.len()/width {let address=part.device_address().checked_add((row*width*4) as u64).ok_or("indirect address overflow")?;
                addresses.push(f32::from_bits(address as u32));addresses.push(f32::from_bits((address>>32) as u32));
                if table_form {
                    let stride=shape.dim+shape.cols;addresses.extend([f32::from_bits(stride),0.0]);
                    let value=address.checked_add(shape.dim as u64*4).ok_or("indirect value address overflow")?;
                    value_addresses.extend([f32::from_bits(value as u32),f32::from_bits((value>>32) as u32),f32::from_bits(stride),0.0]);
                }
            }
        }
        let entry_words=if table_form {4} else {2};
        if addresses.len()!=shape.taps as usize*entry_words {return Err("indirect live rows differ".into());}
        let key_words=addresses.len();let value_words=value_addresses.len();addresses.extend_from_slice(&value_addresses);let count_at=addresses.len();addresses.push(shape.taps as f32);
        let tables=self.upload_vector(&addresses)?.into_resident_shared()?;
        let table=if key_words==0 {None} else {Some(tables.resident_range(0..key_words)?)};
        let value_table=if table_form {Some(tables.resident_range(key_words..key_words+value_words)?)} else {None};
        let count=tables.resident_range(count_at..count_at+1)?;
        let mut scores=self.vector(units)?;
        let scores_shape=Shape {op:if table_form {Op::ScoresTable} else {Op::ScoresIndirect},..shape};
        self.resident_causal_into(scores_shape,CausalInputs {slots:[Some(query),table.as_ref(),None,None],checked_rows:true,bf16:false},None,&mut scores)?;
        let softmax=Shape {op:Op::Softmax,taps:0,cols:0,..shape};
        let probabilities=self.causal_operation(softmax,[Some(&scores),Some(&count),None,None],None)?;
        let values=Shape {op:if table_form {Op::ValuesTable} else {Op::ValuesIndirect},dim:shape.cols,cols:shape.dim,..shape};
        let mut output=self.vector(shape.heads as usize*shape.cols as usize)?;
        self.resident_causal_into(values,CausalInputs {slots:[Some(&probabilities),value_table.as_ref().or(table.as_ref()),None,None],checked_rows:true,bf16:false},None,&mut output)?;
        Ok([scores,probabilities,output])
    }
    /// Attention over expanded rows stored packed two BF16 patterns a word (`Op::PackBf16` at the producer). Each
    /// part holds whole packed rows of heads*(dim+cols)/2 words. The packed table kernels widen each half to the
    /// exact f32 the unpacked row held, so all three outputs equal `resident_row_attention` on the unpacked rows.
    /// Operands that are not 16-byte aligned take that path on rows unpacked here, never a packed launch.
    pub fn resident_packed_attention(&self,shape:Shape,parts:&[&DeviceVector<'_>],query:&DeviceVector<'_>)->Result<[DeviceVector<'a>;3],String> {
        use crate::causal_ops::Op;
        self.require_resident_stream()?;
        if shape.op!=Op::ScoresIndirect {return Err("packed attention requires a scores shape".into());}
        if !crate::causal_ops::indirect_attention_proven(shape) {return Err("packed attention shape is outside the byte-proven family".into());}
        if shape.taps>1<<24 {return Err("packed live count exceeds exact FP32 integers".into());}
        let (_,_,units,_)=shape.lengths().ok_or("invalid packed attention shape")?;
        let width=(shape.heads as usize).checked_mul((shape.dim as usize).checked_add(shape.cols as usize).ok_or("packed row width overflow")?).ok_or("packed row width overflow")?/2;
        self.check_vector(query,shape.heads as usize*shape.dim as usize)?;
        for part in parts {self.check_vector(part,part.len())?;
            if part.is_segmented() || part.len()%width!=0 {return Err("packed row owner extent differs".into());}}
        if parts.iter().map(|part|part.len()/width).sum::<usize>()!=shape.taps as usize {return Err("packed live rows differ".into());}
        let table_form=query.device_address().is_multiple_of(16) && parts.iter().all(|part|part.device_address().is_multiple_of(16));
        if !table_form {
            let mut plain=Vec::with_capacity(parts.len());
            for part in parts {let unpack=Shape {op:Op::UnpackBf16,heads:0,dim:0,taps:0,rows:u32::try_from(part.len()).map_err(|_|"packed row owner exceeds the ABI")?,cols:0,epsilon:0.0,bound:0.0};
                plain.push(self.causal_operation(unpack,[Some(part),None,None,None],None)?);}
            return self.resident_row_attention(shape,&plain.iter().collect::<Vec<_>>(),query);
        }
        let stride=(shape.dim+shape.cols)/2;let mut addresses=Vec::new();
        for part in parts {for row in 0..part.len()/width {let address=part.device_address().checked_add((row*width*4) as u64).ok_or("packed address overflow")?;
            addresses.extend([f32::from_bits(address as u32),f32::from_bits((address>>32) as u32),f32::from_bits(stride),0.0]);}}
        let count_at=addresses.len();addresses.push(shape.taps as f32);
        let tables=self.upload_vector(&addresses)?.into_resident_shared()?;
        let table=Some(tables.resident_range(0..count_at)?);let count=tables.resident_range(count_at..count_at+1)?;
        let mut scores=self.vector(units)?;
        self.resident_causal_into(Shape {op:Op::ScoresPacked,..shape},CausalInputs {slots:[Some(query),table.as_ref(),None,None],checked_rows:true,bf16:false},None,&mut scores)?;
        let probabilities=self.causal_operation(Shape {op:Op::Softmax,taps:0,cols:0,..shape},[Some(&scores),Some(&count),None,None],None)?;
        let mut output=self.vector(shape.heads as usize*shape.cols as usize)?;
        self.resident_causal_into(Shape {op:Op::ValuesPacked,dim:shape.cols,cols:shape.dim,..shape},CausalInputs {slots:[Some(&probabilities),table.as_ref(),None,None],checked_rows:true,bf16:false},None,&mut output)?;
        Ok([scores,probabilities,output])
    }
}
