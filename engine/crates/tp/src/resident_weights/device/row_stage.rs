//! ROW STAGE: the MXFP8 projections of a row stage that read the same rows (q, k and v of a recurrent layer), for a group
//! of 2 to 4 rows (`ROW_STAGE_ROWS`: a verify pass, a batch of chats, the short tail group of a prompt), as ONE merged launch: one
//! quantize of the rows and one launch that reads every weight once for all rows. The plain decode step runs the same
//! matrices this way for its one row (`apply_mixed_batch`); the row path ran quantize, launch, round and copy for each
//! matrix. Every output word keeps its one-row arithmetic and order (`flash_resident_projection_bf16_rows`;
//! kernel-bench --rows-p1 compares every byte with the one-row entry), so a row's words are the words `matrix_rows`
//! gives for it.
use super::*;
use crate::bounded_decoder::ROW_STAGE_ROWS;
// The stage's bound stays inside the entry's contract (2 to 4 source rows).
const _:()=assert!(ROW_STAGE_ROWS>=2 && ROW_STAGE_ROWS<=4);
impl<'b,'d> Weights<'b,'d> {
    /// The stage's MXFP8 jobs by the rows they read. Each group of two or more is one merged launch. Chosen by
    /// storage, shape and the identity of the input alone, so every rank makes the same groups.
    pub(super) fn mx_row_groups(&self,jobs:&[(&str,&DeviceVector<'_>,bool)],m:usize)->Vec<Vec<usize>> {
        if !(2..=ROW_STAGE_ROWS).contains(&m) {return Vec::new();}
        let mut groups:Vec<((u64,usize),Vec<usize>)>=Vec::new();
        for (job,(name,input,_)) in jobs.iter().enumerate() {
            let Some(matrix)=self.matrices.get(*name) else {continue;};
            if !matches!(&matrix.storage,Storage::Mx(_)) || !matrix.cols.is_multiple_of(32) || input.is_segmented() {continue;}
            let key=(input.device_address(),input.len());
            match groups.iter_mut().find(|(prior,held)|*prior==key && held.len()<9) {Some((_,held))=>held.push(job),None=>groups.push((key,vec![job]))}
        }
        groups.into_iter().map(|(_,held)|held).filter(|held|held.len()>=2).collect()
    }
    /// QUANTIZE SHARE: the MXFP8 jobs of a group past the row stage (a prompt group) that read the same rows and whose
    /// rows stay on this rank (not split, or the KDA head split: q, k and v of a recurrent layer). Each group of two or
    /// more quantizes its rows ONCE; `matrix_rows` quantized the same rows again for every matrix. A job whose rows are
    /// exchanged keeps its single call, so a stage's exchanges are the ones it had. Chosen by storage, shape and the
    /// identity of the input alone, so every rank makes the same groups.
    pub(super) fn mx_share_groups(&self,jobs:&[(&str,&DeviceVector<'_>,bool)],m:usize)->Vec<Vec<usize>> {
        if m<=ROW_STAGE_ROWS {return Vec::new();}
        let mut groups:Vec<((u64,usize),Vec<usize>)>=Vec::new();
        for (job,(name,input,_)) in jobs.iter().enumerate() {
            let Some(matrix)=self.matrices.get(*name) else {continue;};
            if !matches!(&matrix.storage,Storage::Mx(_)) || input.is_segmented() || matrix.rows.split && !self.kda_local.contains(*name) {continue;}
            let key=(input.device_address(),input.len());
            match groups.iter_mut().find(|(prior,_)|*prior==key) {Some((_,held))=>held.push(job),None=>groups.push((key,vec![job]))}
        }
        groups.into_iter().map(|(_,held)|held).filter(|held|held.len()>=2).collect()
    }
    /// One group of `mx_share_groups`: this rank's rows of each job, `[m,local rows]`, rounded: the calls of
    /// `matrix_rows` for an MXFP8 matrix (`project_rows` = quantize, projection, round), with the one
    /// quantize in front. The quantized rows depend on the rows and the width only, and the jobs of a group have one
    /// width (equal input lengths at one `m`), so each matrix reads the words its own quantize would have made.
    pub(super) fn mx_rows_shared(&mut self,b:&Backend<'d>,jobs:&[(&str,&DeviceVector<'_>,bool)],group:&[usize],m:usize,made:&mut [Option<(DeviceVector<'d>,bool)>])->Result<()> {
        let mut decoded=None;
        for &job in group {
            let stamp=self.cost.start(b);let (name,input,_)=jobs[job];
            let matrix=self.matrices.get(name).ok_or("quantize share matrix is absent")?;
            let Storage::Mx(projection)=&matrix.storage else {return Err("quantize share MXFP8 storage differs".into());};
            if input.len()!=m.checked_mul(matrix.cols).ok_or("quantize share extent overflow")? {return Err("quantize share feature extent differs".into());}
            let rows=match &decoded {Some(rows)=>rows,None=>decoded.insert(projection.quantize_rows(input,m)?)};
            let local=matrix.rows.count;let output=projection.project_quantized_rows(rows,m)?.into_vector();
            if output.len()!=m*local {return Err("quantize share rows extent differs".into());}made[job]=Some((output,false));
            let bytes=local*(matrix.cols+matrix.cols/32);
            self.cost.settle(b)?;self.cost.record(||"projection_0".into(),stamp,b,bytes);
        }
        Ok(())
    }
    /// One group of `mx_row_groups`: this rank's rows of each job, `[m,local rows]`, rounded, as `matrix_rows` returns
    /// them for an MXFP8 matrix. The quantizer owns complete 32-lane groups, so concatenated rows quantize independently
    /// (the same call as `project_rows`).
    pub(super) fn mx_rows_launch(&mut self,b:&Backend<'d>,jobs:&[(&str,&DeviceVector<'_>,bool)],group:&[usize],m:usize,made:&mut [Option<(DeviceVector<'d>,bool)>])->Result<()> {
        use spark_gpu_probe::{driver::ResidentProjection,causal_ops::{Shape as CausalShape,Op as CausalOp}};
        let stamp=self.cost.start(b);let input=jobs[*group.first().ok_or("row stage group is empty")?].1;
        let cols=input.len().try_into().map_err(|_|"row stage columns exceed ABI")?;
        let quantize=CausalShape {op:CausalOp::Mxfp8Quantize,heads:0,dim:0,taps:0,rows:0,cols,epsilon:0.0,bound:0.0};
        let decoded=b.causal_operation(quantize,[Some(input),None,None,None],None)?;
        let mut batch=Vec::with_capacity(group.len());let (mut words,mut bytes)=(0usize,0usize);
        for &job in group {
            let matrix=self.matrices.get(jobs[job].0).ok_or("row stage matrix is absent")?;
            let Storage::Mx(projection)=&matrix.storage else {return Err("row stage MXFP8 storage differs".into());};
            if jobs[job].1.len()!=m*matrix.cols {return Err("row stage feature extent differs".into());}
            batch.push(ResidentProjection::Mxfp8 {matrix:projection.resident(),input:&decoded});
            words+=matrix.rows.count*m;bytes+=matrix.rows.count*(matrix.cols+matrix.cols/32);
        }
        let mut output=b.vector(words)?;b.resident_projection_batch_rows(&batch,&mut output,true,m)?;drop(batch);
        // A matrix's rows are one descriptor: [m,local rows] is one range of the launch's output.
        let output=output.into_resident_shared()?;let mut start=0;
        for &job in group {let extent=self.matrices[jobs[job].0].rows.count*m;made[job]=Some((output.resident_range(start..start+extent)?,false));start+=extent;}
        self.cost.record(||"projection_rows_stage".into(),stamp,b,bytes);Ok(())
    }
}
