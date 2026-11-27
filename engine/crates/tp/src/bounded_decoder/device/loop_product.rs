//! The product of a matrix that stays on the device with one input row, for the driver of the block family with a loop.
//!
//! The matrix is a range of words of a resident vector: `rows x columns` BF16 values, two a word, row-major. The
//! picked expert of a stack is such a range, so no expert is copied. The output is `rows` 32-bit words: one chain a row
//! over the columns in rising order from 0.0, each product rounded before its add, the sum not rounded.
//!
//! Two launches give these words. A matrix of whole 64-column stages takes the staged product, an entry of its own
//! (`Backend::project_staged_into`), while launches are recorded and its operands are aligned. Any other case takes
//! `Backend::project_bf16_chain_into` (kernel `project_bf16_ordered`, for such a shape one thread a row).
//!
//! A matrix above the envelope of the staged product (`staged_product::part_rows`) runs in parts of whole
//! blocks, one launch a part, into one output. A row is one chain, so the parts give the words of one launch.
//!
//! Many input rows (`resident_product_rows`) use the few-row MMA entry for small decode batches and the
//! multi-row MMA entry for prefill and larger batches. Both accumulate in ascending K order.
//! Widened expert runs that admit MMA use the gathered matrix path. The indexed scalar tile remains available
//! for shapes outside the MMA envelope.
use super::*;

impl<'b,'d> Device<'b,'d> {
    /// Prepare the one-row expert panels within `room` bytes (the space left past the admitted ledger): for each narrow
    /// expert stack, a copy of the same length whose NVFP4 code bytes and scale pairs are in MMA lane order
    /// (`decode_panels::pack_expert`, the original units unchanged). A one-row picked product reads it in coalesced
    /// loads. The dense one-row products read the lossless BF16 panels that the few-row products also read
    /// (`Backend::register_bf16_panels`): there is one copy of those.
    pub fn prepare_expert_panels(&mut self, operands: &crate::loop_decoder::operands::Operands<DeviceVector<'d>>, room: u64) -> Result<u64> {
        let mut bytes = 0;
        let mut expert_panels = BTreeMap::new();
        for operand in operands.all() {
            let Some(unit) = operand.tensor.unit() else { continue };
            let raw = &operand.value;
            if unit.codes == engine_model::plan::loop_load::ExpertCodes::Mxfp8
                || !unit.rows.is_multiple_of(64) || !unit.columns.is_multiple_of(64)
                || unit.columns > spark_gpu_probe::packed_product::PANEL_INPUT_COLS
                || raw.is_segmented() || raw.len() as u64 * 4 > room - bytes { continue; }
            let unit_words = unit.layout().map_err(|e| e.to_string())?.bytes / 4;
            if unit_words == 0 || !raw.len().is_multiple_of(unit_words) { continue; }
            let packed = self.backend.vector(raw.len())?.into_resident_shared()?;
            let source = raw.resident_range(0..raw.len())?;
            for start in (0..raw.len()).step_by(unit_words) {
                let end = start + unit_words;
                let words = self.backend.read_vector_values(&source.resident_range(start..end)?)?;
                let input = words.iter().flat_map(|w| w.to_bits().to_le_bytes()).collect::<Vec<_>>();
                let panels = spark_gpu_probe::decode_panels::pack_expert(&input, unit.rows, unit.columns);
                self.backend.write_vector_bytes(&mut packed.resident_range(start..end)?, &panels)?;
            }
            bytes += raw.len() as u64 * 4;
            expert_panels.insert(raw.device_address(),packed);
        }
        self.expert_panels = Rc::new(expert_panels);
        Ok(bytes)
    }

    pub(super) fn resident_row_dots(&self,queries:&[f32],stack:&Value<'b,'d>,count:usize,width:usize)->Result<Value<'b,'d>> {
        let shape=super::super::row_dots_shape(queries.len(),count,width)?;
        let asked=queries.len()/width;
        if self.loop_prefill.get() && self.backend.resident_launches_active() && stack.words==count*width && !stack.vector.is_segmented()
            && spark_gpu_probe::fast_ops::row_dots_admitted(1,asked,count,width,spark_gpu_probe::fast_ops::ROW_DOTS_THREADS)
        {
            self.ready(stack)?;
            let weight=self.parameters.borrow_mut().resolve(queries,None,|values|self.backend.upload_vector(values)?.into_resident_shared())?.ok_or("row dots have no query")?;
            if !weight.is_segmented() {
                let stamp=self.cost.start(self.backend);
                let mut out=self.backend.vector(asked*count)?;
                self.backend.fast_row_dots_into(1,asked,count,width,&stack.vector,&weight,&mut out)?;
                let value=self.value(out,asked*count)?;
                self.cost.record(||"fast_row_dots".into(),stamp,self.backend,0);
                return Ok(value);
            }
        }
        // FOLD DOTS of several query rows (the attention mix of a layer with three inputs, the mix of the messages): one
        // fold launch with the query rows as its weight rows gives `[count][asked]`, each word the chain of its own
        // (query row, stack row) from +0.0 in column order (the one-row fold's chain); one gather of single words then
        // lays them out as `[asked][count]`, the words of `RowDots`. The staged arm took ~96 us for 3 x 2 dots.
        if (2..=spark_gpu_probe::plain_admission::FOLD_ROWS_MAX_TOKENS).contains(&count) && stack.words==count*width && queries.len()==asked*width {
            self.ready(stack)?;
            let weight=self.parameters.borrow_mut().resolve(queries,None,|values|self.backend.upload_vector(values)?.into_resident_shared())?.ok_or("row dots have no query")?;
            if !weight.is_segmented() && !stack.vector.is_segmented()
                && spark_gpu_probe::plain_admission::fold_rows_admitted(asked,width,weight.device_address() as usize,stack.vector.device_address() as usize,count) {
                let stamp=self.cost.start(self.backend);let mut out=self.backend.vector(asked*count)?;
                // The old flag comes back when the guard drops: after the launch, and on its error.
                let pass=super::super::Restore::new(self.row_pass.get(),true,|on:bool|self.set_row_pass(if on {2} else {1}));
                let done=self.backend.resident_projection_fold_rows(&[spark_gpu_probe::driver::ResidentProjection::Plain {rows:asked,cols:width,weight:&weight,input:&stack.vector}],&mut out,false,count);
                drop(pass);done?;
                let mut value=self.value(out,asked*count)?;
                if asked>1 {
                    // Word q*count + c of the result is word c*asked + q of the fold.
                    let order=(0..asked).flat_map(|q|(0..count).map(move |c|(c*asked+q) as f32)).collect::<Vec<_>>();
                    let table=self.parameters.borrow_mut().resolve(&order,None,|values|self.backend.upload_vector(values)?.into_resident_shared())?.ok_or("row dots have no order")?;
                    let table=self.shared_value(table,order.len())?;
                    let mut p=spark_gpu_probe::prompt_ops::Shape::new(spark_gpu_probe::prompt_ops::Op::PermuteRows,1).ok_or("row dots: a gather exceeds the launch format")?;
                    p.rows=u32::try_from(asked*count).map_err(|_|"row dots exceed the gather")?;p.cols=1;p.first=p.rows;
                    value=self.prompt(p,[Some(&value),Some(&table),None,None,None],&[])?;
                }
                self.cost.record(||"dot_rows_fold".into(),stamp,self.backend,0);return Ok(value);
            }
        }
        self.operation(shape,[Some(stack),None,None,None],queries)
    }

    /// One launch: `rows` chains of one BF16 matrix with one input row. The narrow staged product for a matrix of few
    /// rows (`staged_narrow`) when it runs now, else the staged product when it runs now, else the
    /// chain of `project_bf16_ordered`. A split matrix keeps the same K order in every part.
    fn row_product(&self,rows:usize,columns:usize,total_rows:usize,input:&DeviceVector<'d>,matrix:&DeviceVector<'d>,output:&mut DeviceVector<'d>)->Result<()> {
        if rows==total_rows && self.backend.staged_narrow_ready(rows,columns,input,matrix) {self.backend.project_staged_narrow_into(rows,columns,input,matrix,output)?;}
        else if self.backend.staged_product_ready(rows,columns,input,matrix) {self.backend.project_staged_into(rows,columns,input,matrix,output)?;}
        else {self.backend.project_bf16_chain_into(rows,columns,input,matrix,output)?;}
        Ok(())
    }
    /// `resident_product` of each matrix of `matrices` (all `rows x columns`) on its own input row, in ONE launch of
    /// the grouped narrow product (`staged_narrow::check_group`): the results back to back, `[groups][rows]`, each
    /// part the words of `resident_product` on its matrix (the same chain). None when that launch does not run now;
    /// the caller takes one matrix at a time.
    pub(crate) fn resident_group(&self,matrices:&[(&DeviceVector<'d>,std::ops::Range<usize>)],rows:usize,columns:usize,xs:&[&Value<'b,'d>])->Result<Option<Value<'b,'d>>> {
        need(matrices.len()==xs.len() && !matrices.is_empty(),"grouped product: the matrices differ from the inputs")?;
        for (_,words) in matrices {need(rows>0 && columns>0 && rows.checked_mul(columns)==words.len().checked_mul(2),"grouped product: the weight words differ from rows x columns of BF16 values")?;}
        need(xs.iter().all(|x|x.words==columns),"grouped product: an input differs from the columns of the weight")?;
        if !spark_gpu_probe::staged_narrow::groups_admitted(matrices.len()) || spark_gpu_probe::staged_product::part_rows(rows,columns)<rows {return Ok(None);}
        let weights=matrices.iter().map(|(weight,words)|weight.resident_range(words.clone())).collect::<std::result::Result<Vec<_>,_>>()?;
        let weight_refs=weights.iter().collect::<Vec<_>>();
        let inputs=xs.iter().map(|x|&*x.vector).collect::<Vec<_>>();
        if !self.backend.staged_narrow_group_ready(rows,columns,&inputs,&weight_refs) {return Ok(None);}
        for x in xs {self.ready(x)?;}
        let mut output=self.backend.vector(matrices.len()*rows)?;
        self.backend.project_staged_narrow_group_into(rows,columns,&inputs,&weight_refs,&mut output)?;
        self.value(output,matrices.len()*rows).map(Some)
    }
    pub(crate) fn resident_product(&self,weight:&DeviceVector<'d>,words:std::ops::Range<usize>,rows:usize,columns:usize,x:&Value<'b,'d>)->Result<Value<'b,'d>> {
        need(rows>0 && columns>0 && rows.checked_mul(columns)==words.len().checked_mul(2),"resident product: the weight words differ from rows x columns of BF16 values")?;
        need(x.words==columns,"resident product: the input differs from the columns of the weight")?;
        self.ready(x)?;
        if self.backend.resident_launches_active() && !weight.is_segmented()
            && !x.vector.is_segmented() && x.vector.device_address().is_multiple_of(16)
            && spark_gpu_probe::few_product::few_shape_admitted(1,columns,1)
            && self.backend.staged_few_has_panels(&weight.resident_range(words.clone())?, columns)
        {
            return self.resident_product_rows(weight,words,rows,columns,x,1,true);
        }
        let part=spark_gpu_probe::staged_product::part_rows(rows,columns);
        if part>=rows {
            let matrix=weight.resident_range(words)?;
            let mut output=self.backend.vector(rows)?;
            self.row_product(rows,columns,rows,&x.vector,&matrix,&mut output)?;
            return self.value(output,rows);
        }
        // Two BF16 values a word: a row is columns / 2 words (the columns are whole stages here).
        let (whole,row_words)=(self.backend.vector(rows)?.into_resident_shared()?,columns/2);
        let mut row=0;
        while row<rows {
            let count=part.min(rows-row);
            let matrix=weight.resident_range(words.start+row*row_words..words.start+(row+count)*row_words)?;
            let mut output=whole.resident_range(row..row+count)?;
            self.row_product(count,columns,rows,&x.vector,&matrix,&mut output)?;
            row+=count;
        }
        self.shared_value(Rc::new(whole),rows)
    }
    /// `resident_product` for `inputs` rows of `x` (`[inputs][columns]`): the result is `[inputs][rows]`, row-major.
    pub(crate) fn resident_product_rows(&self,weight:&DeviceVector<'d>,words:std::ops::Range<usize>,rows:usize,columns:usize,x:&Value<'b,'d>,inputs:usize,decode:bool)->Result<Value<'b,'d>> {
        need(rows>0 && columns>0 && rows.checked_mul(columns)==words.len().checked_mul(2),"resident product: the weight words differ from rows x columns of BF16 values")?;
        need(inputs>0 && inputs.checked_mul(columns)==Some(x.words),"resident product: the input differs from the input rows x the columns of the weight")?;
        self.ready(x)?;
        let total=inputs.checked_mul(rows).ok_or("resident product: the output of the input rows is too large")?;
        use spark_gpu_probe::{few_product,tc_rows};
        let tc_shape=tc_rows::tc_shape_admitted(1,columns,1);
        let few_shape=few_product::few_shape_admitted(1,columns,1);
        if !weight.is_segmented() && !x.vector.is_segmented() && (tc_shape || few_shape) {
            let limit=if tc_shape {tc_rows::TC_INPUTS_MAX} else {few_product::FEW_MAX};
            if inputs>limit {
                let mut parts=Vec::with_capacity(inputs.div_ceil(limit));
                for first in (0..inputs).step_by(limit) {
                    let count=(inputs-first).min(limit);
                    let part=self.range(x,first*columns,count*columns)?;
                    parts.push(self.resident_product_rows(weight,words.clone(),rows,columns,&part,count,decode)?);
                }
                return self.join(&parts.iter().collect::<Vec<_>>());
            }
        }
        // The few-row body shares the multi-row entry's ascending-K BF16 arithmetic.
        // For `M <= ONE_PANEL_INPUTS` (1..=8), `PLAIN_STAGED_ONE`/`PLAIN_STAGED_FEW` read compressed lossless BF16 panels.
        // For `M = 9..=16` (`C16`) when `PLAIN_TC_ROWS` admits the shape, route to wave-matched `PLAIN_TC_ROWS`.
        let few_cap=if tc_shape {few_product::ONE_PANEL_INPUTS} else {few_product::FEW_MAX};
        if (decode || !tc_shape) && few_shape && inputs<=few_cap && !weight.is_segmented() {
            let part=spark_gpu_probe::few_product::few_part_rows(rows,columns);
            if inputs <= few_product::ONE_PANEL_INPUTS && rows > part {
                if let Ok(matrix) = weight.resident_range(words.clone()) {
                    if self.backend.staged_one_panel_ready(rows, columns, inputs, &x.vector, &matrix) {
                        let mut output = self.backend.vector(total)?;
                        self.backend.project_staged_panels_rows_into(rows, columns, inputs, rows, None, &x.vector, &matrix, &mut output)?;
                        return self.value(output, total);
                    }
                }
            }
            let ready=(0..rows).step_by(part.max(1)).all(|row|{
                let count=part.min(rows-row);
                weight.resident_range(words.start+row*columns/2..words.start+(row+count)*columns/2).is_ok_and(|matrix|self.backend.staged_few_ready(count,columns,inputs,&x.vector,&matrix))
            });
            if ready {
                if part>=rows {
                    let mut output=self.backend.vector(total)?;
                    self.backend.project_staged_few_into(rows,columns,inputs,rows,&x.vector,&weight.resident_range(words)?,&mut output)?;
                    return self.value(output,total);
                }
                let whole=self.backend.vector(total)?.into_resident_shared()?;
                let mut row=0;
                while row<rows {
                    let count=part.min(rows-row);
                    // Two BF16 values a word; a part ends on a whole row.
                    let matrix=weight.resident_range(words.start+row*columns/2..words.start+(row+count)*columns/2)?;
                    let mut output=whole.resident_range(row..row+(inputs-1)*rows+count)?;
                    self.backend.project_staged_few_into(count,columns,inputs,rows,&x.vector,&matrix,&mut output)?;
                    row+=count;
                }
                return self.shared_value(Rc::new(whole),total);
            }
        }
        // The same multi-row BF16 MMA entry also covers decode above the few-row limit.
        if let Some(part)=tc_rows::tc_tile_part(rows,columns,inputs).filter(|_|!weight.is_segmented()) {
            let ready=(0..rows).step_by(part).all(|row|{
                let count=part.min(rows-row);
                weight.resident_range(words.start+row*columns/2..words.start+(row+count)*columns/2).is_ok_and(|matrix|self.backend.tc_rows_ready(count,columns,inputs,&x.vector,&matrix))
            });
            if ready {
                if part>=rows {
                    let mut output=self.backend.vector(total)?;
                    self.backend.project_tc_rows_into(rows,columns,inputs,&x.vector,&weight.resident_range(words)?,&mut output)?;
                    return self.value(output,total);
                }
                let whole=self.backend.vector(total)?.into_resident_shared()?;
                let mut row=0;
                while row<rows {
                    let count=part.min(rows-row);
                    let matrix=weight.resident_range(words.start+row*columns/2..words.start+(row+count)*columns/2)?;
                    let mut output=whole.resident_range(row..row+(inputs-1)*rows+count)?;
                    self.backend.project_tc_rows_strided_into(count,columns,inputs,rows,&x.vector,&matrix,&mut output)?;
                    row+=count;
                }
                return self.shared_value(Rc::new(whole),total);
            }
        }
        if inputs==1 {return self.resident_product(weight,words,rows,columns,x);}
        // The rows tile: one launch for all input rows, the output is `[inputs][rows]` at once.
        // A matrix above the rows of the tile (the head) runs in parts, each writing its columns of the output in place.
        if let Some(part)=spark_gpu_probe::staged_product::rows_tile_part(rows,columns,inputs).filter(|_|!weight.is_segmented()) {
            let ready=(0..rows).step_by(part).all(|row|{
                let count=part.min(rows-row);
                weight.resident_range(words.start+row*columns/2..words.start+(row+count)*columns/2).is_ok_and(|matrix|self.backend.staged_rows_ready(count,columns,inputs,&x.vector,&matrix))
            });
            if ready {
                if part>=rows {
                    let mut output=self.backend.vector(total)?;
                    self.backend.project_staged_rows_into(rows,columns,inputs,&x.vector,&weight.resident_range(words)?,&mut output)?;
                    return self.value(output,total);
                }
                let whole=self.backend.vector(total)?.into_resident_shared()?;
                let mut row=0;
                while row<rows {
                    let count=part.min(rows-row);
                    // Two BF16 values a word; a part ends on a whole row.
                    let matrix=weight.resident_range(words.start+row*columns/2..words.start+(row+count)*columns/2)?;
                    let mut output=whole.resident_range(row..row+(inputs-1)*rows+count)?;
                    self.backend.project_staged_rows_strided_into(count,columns,inputs,rows,&x.vector,&matrix,&mut output)?;
                    row+=count;
                }
                return self.shared_value(Rc::new(whole),total);
            }
        }
        if !x.vector.is_segmented() && !weight.is_segmented() && spark_gpu_probe::staged_product::rows_one_launch(rows,columns,inputs) {
            use spark_gpu_probe::prompt_ops::{Op,Shape};
            let shape=|op:Op,tokens:usize,rows:usize,cols:usize|->Result<Shape> {
                let mut shape=Shape::new(op,tokens).ok_or("resident product: no shape for the rows product")?;
                (shape.rows,shape.cols)=(u32::try_from(rows).map_err(|_|"resident product: the rows product is too large")?,u32::try_from(cols).map_err(|_|"resident product: the rows product is too large")?);
                Ok(shape)
            };
            let matrix=weight.resident_range(words)?;
            // The rows kernel gives `[rows][inputs]`; the turn gives `[inputs][rows]`.
            let by_matrix_row=self.backend.prompt_operation(shape(Op::ProjectRowsBf16,inputs,rows,columns)?,[Some(&*x.vector),None,None,None,None],Some(&matrix))?;
            let output=self.backend.prompt_operation(shape(Op::Transpose,1,rows,inputs)?,[Some(&by_matrix_row),None,None,None,None],None)?;
            return self.value(output,total);
        }
        let whole=self.backend.vector(total)?.into_resident_shared()?;
        // A matrix above the envelope of the warp tile runs in parts, as in `resident_product`.
        let part=spark_gpu_probe::staged_product::part_rows(rows,columns).min(rows);
        for input in 0..inputs {
            let source=x.vector.resident_range(input*columns..(input+1)*columns)?;
            let mut row=0;
            while row<rows {
                let count=part.min(rows-row);
                // Two BF16 values a word. A part ends on a whole row; with one part the range is the whole matrix.
                let matrix=weight.resident_range(words.start+row*columns/2..words.start+(row+count)*columns/2)?;
                let mut output=whole.resident_range(input*rows+row..input*rows+row+count)?;
                self.row_product(count,columns,rows,&source,&matrix,&mut output)?;
                row+=count;
            }
        }
        self.shared_value(Rc::new(whole),total)
    }
    /// The products of the expert runs of a step of rows with no gather (`ResidentProduct::resident_product_runs`):
    /// `runs` holds (matrix, words, rows, launch rows) of each run in launch order. None when a run takes neither
    /// form here (a segmented operand, mixed shapes, a matrix above the envelope of the row product); then nothing
    /// was launched and the caller gathers.
    pub(crate) fn resident_product_runs(&self,runs:&[(&DeviceVector<'d>,std::ops::Range<usize>,usize,usize)],x:&Value<'b,'d>,columns:usize,input:Option<&[usize]>,output:Option<&[usize]>)->Result<Option<Value<'b,'d>>> {
        use spark_gpu_probe::staged_product as staged;
        let Some(&(_,_,rows,_))=runs.first() else {return Ok(None)};
        let n=runs.iter().map(|run|run.3).sum::<usize>();
        if rows==0 || columns==0 || n==0 || !x.words.is_multiple_of(columns) || x.vector.len()!=x.words || x.vector.is_segmented() || staged::part_rows(rows,columns)<rows
            || runs.iter().any(|run|run.2!=rows || run.3==0 || run.0.is_segmented() || run.1.len().checked_mul(2)!=rows.checked_mul(columns)) {return Ok(None);}
        // Indexed scalar launches retain FP32 activations. Let the gathered caller use the same BF16 MMA as
        // a contiguous product whenever the matrix shape admits it, at every run length.
        if spark_gpu_probe::few_product::few_shape_admitted(1,columns,1) {return Ok(None);}
        self.ready(x)?;
        let sources=x.words/columns;
        // The input row and the result row of each launch row (the result table is the inverse of `output`).
        let word=|row:usize|u32::try_from(row).map_err(|_|"resident product: a row of the runs exceeds the ABI".to_owned());
        let in_words=match input {Some(rows)=>rows.iter().map(|&row|word(row)).collect::<Result<Vec<_>>>()?,None=>(0..n).map(word).collect::<Result<Vec<_>>>()?};
        let mut out_words=(0..n).map(word).collect::<Result<Vec<_>>>()?;
        if let Some(rows)=output {for (j,&i) in rows.iter().enumerate() {*out_words.get_mut(i).ok_or("resident product: an output row of the runs is outside the launch rows")?=word(j)?;}}
        need(in_words.len()==n && output.is_none_or(|rows|rows.len()==n) && in_words.iter().all(|&row|(row as usize)<sources),"resident product: a table of the runs differs from the launch rows")?;
        // The form of each run, all asked before the first launch.
        // A run of more launch rows than one launch of the tile runs in launches of at most `ROWS_INPUTS_MAX`.
        let tile=|count:usize|count>=staged::ROWS_TILE_INPUTS_MIN && staged::rows_shape_admitted(rows,columns,1);
        let mut matrices=Vec::with_capacity(runs.len());
        for run in runs {
            let matrix=run.0.resident_range(run.1.clone())?;
            if tile(run.3) && !self.backend.staged_rows_at_ready(rows,columns,run.3.min(staged::ROWS_INPUTS_MAX),&x.vector,&matrix) {return Ok(None);}
            matrices.push(matrix);
        }
        let total=n.checked_mul(rows).ok_or("resident product: the result of the runs is too large")?;
        let whole=self.backend.vector(total)?.into_resident_shared()?;
        let mut at=0;
        for (run,matrix) in runs.iter().zip(&matrices) {
            let count=run.3;
            if tile(count) {
                for first in (at..at+count).step_by(staged::ROWS_INPUTS_MAX) {
                    let span=first..(first+staged::ROWS_INPUTS_MAX).min(at+count);
                    self.backend.project_staged_rows_at_into(rows,columns,span.len(),rows,&x.vector,Some(&in_words[span.clone()]),matrix,&mut whole.resident_range(0..total)?,Some(&out_words[span]))?;
                }
            } else {
                for (&source,&target) in in_words[at..at+count].iter().zip(&out_words[at..at+count]) {
                    let (source,target)=(source as usize,target as usize);
                    self.row_product(rows,columns,rows,&x.vector.resident_range(source*columns..(source+1)*columns)?,matrix,&mut whole.resident_range(target*rows..(target+1)*rows)?)?;
                }
            }
            at+=count;
        }
        Ok(Some(self.shared_value(Rc::new(whole),total)?))
    }
    /// Rows `ids` of a BF16 table that stays on the device (`ResidentProduct::resident_rows`): `words` of `table` hold
    /// `rows x columns` BF16 values, two a word. One launch of the embedding rows entry (`Backend::gather_embed_rows_into`)
    /// gives `[ids][columns]` 32-bit words, each the value that a host read of the stored bytes gives. None when the
    /// launch cannot run now (launches not recorded, a segmented table); then nothing was launched.
    pub(crate) fn resident_rows(&self,table:&DeviceVector<'d>,words:std::ops::Range<usize>,rows:usize,columns:usize,ids:&[u32])->Result<Option<Value<'b,'d>>> {
        need(rows>0 && columns>0 && columns.is_multiple_of(2) && rows.checked_mul(columns)==words.len().checked_mul(2),"resident rows: the table words differ from rows x columns of BF16 values")?;
        if ids.is_empty() || table.is_segmented() || !self.backend.embed_rows_ready(ids.len()) {return Ok(None);}
        let table=table.resident_range(words)?;
        let total=ids.len().checked_mul(columns).ok_or("resident rows: the rows are too large")?;
        let mut output=self.backend.vector(total)?;
        self.backend.gather_embed_rows_into(rows,columns,&table,ids,&mut output)?;
        self.value(output,total).map(Some)
    }
    /// Fused multi-output BF16 product: writes directly into separate `[inputs][part]` buffers with zero copy.
    /// Uses `PLAIN_STAGED_FEW` for decode batches of 2 to 16 rows and `PLAIN_TC_ROWS` for prefill/large batches.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resident_product_rows_split(&self,weight:&DeviceVector<'d>,words:std::ops::Range<usize>,parts:&[usize],columns:usize,x:&Value<'b,'d>,inputs:usize,decode:bool)->Result<Option<Vec<Value<'b,'d>>>> {
        use spark_gpu_probe::{few_product,tc_rows};
        let rows:usize=parts.iter().sum();
        if parts.is_empty() || parts.len()>few_product::FEW_SPLIT_MAX || parts.contains(&0) || weight.is_segmented() || x.vector.is_segmented() {return Ok(None);}
        need(rows>0 && columns>0 && rows.checked_mul(columns)==words.len().checked_mul(2),"resident split product: weight words differ from rows x columns")?;
        need(inputs>0 && inputs.checked_mul(columns)==Some(x.words),"resident split product: input differs from inputs x columns")?;
        self.ready(x)?;
        let matrix=weight.resident_range(words)?;
        let tc_split_ok=inputs>=tc_rows::TC_ROWS_INPUTS_MIN && parts.len()<=tc_rows::TC_SPLIT_MAX
            && !parts.iter().any(|&w|!w.is_multiple_of(2)) && self.backend.tc_rows_ready(rows,columns,inputs,&x.vector,&matrix);
        let few_cap=if tc_split_ok {few_product::ONE_PANEL_INPUTS} else {few_product::FEW_MAX};
        if decode && (2..=few_cap).contains(&inputs) && self.backend.staged_few_ready(rows,columns,inputs,&x.vector,&matrix) {
            let mut buffers=Vec::with_capacity(parts.len());
            for &w in parts {buffers.push(self.backend.vector(inputs*w)?);}
            {
                let mut refs:Vec<(&mut DeviceVector<'d>,usize)>=buffers.iter_mut().zip(parts.iter().copied()).collect();
                self.backend.project_staged_few_split_into(rows,columns,inputs,&x.vector,&matrix,&mut refs)?;
            }
            let mut out=Vec::with_capacity(parts.len());
            for (buf,&w) in buffers.into_iter().zip(parts) {out.push(self.value(buf,inputs*w)?);}
            return Ok(Some(out));
        }
        if (decode && inputs<=few_cap) || !tc_split_ok {return Ok(None);}
        let mut buffers=Vec::with_capacity(parts.len());
        for &w in parts {buffers.push(self.backend.vector(inputs*w)?);}
        {
            let mut refs:Vec<(&mut DeviceVector<'d>,usize)>=buffers.iter_mut().zip(parts.iter().copied()).collect();
            self.backend.project_tc_rows_split_into(rows,columns,inputs,&x.vector,&matrix,&mut refs)?;
        }
        let mut out=Vec::with_capacity(parts.len());
        for (buf,&w) in buffers.into_iter().zip(parts) {out.push(self.value(buf,inputs*w)?);}
        Ok(Some(out))
    }
    /// Dual-input BF16 product over `[x1[t], x2[t]]` with optional BF16 output rounding, eliminating
    /// `join` + `swap(2, inputs, half)` (and `round` when `round_out` is set) on `GateDown`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resident_product_rows_pair(&self,weight:&DeviceVector<'d>,words:std::ops::Range<usize>,rows:usize,columns:usize,x1:&Value<'b,'d>,x2:&Value<'b,'d>,inputs:usize,decode:bool,round_out:bool)->Result<Option<Value<'b,'d>>> {
        use spark_gpu_probe::{few_product,tc_rows};
        if !columns.is_multiple_of(2) || weight.is_segmented() || x1.vector.is_segmented() || x2.vector.is_segmented() {return Ok(None);}
        let half=columns/2;
        if x1.words!=inputs*half || x2.words!=inputs*half {return Ok(None);}
        need(rows>0 && columns>0 && rows.checked_mul(columns)==words.len().checked_mul(2),"resident pair product: weight words differ from rows x columns")?;
        self.ready(x1)?;self.ready(x2)?;
        let matrix=weight.resident_range(words)?;
        if decode && inputs==1 && self.backend.staged_narrow_pair_ready(rows,columns,&x1.vector,&x2.vector,&matrix) {
            let mut output=self.backend.vector(rows)?;
            self.backend.project_staged_narrow_pair_into(rows,columns,round_out,&x1.vector,&x2.vector,&matrix,&mut output)?;
            return Ok(Some(self.value(output,rows)?));
        }
        let tc_pair_ok=inputs>=tc_rows::TC_ROWS_INPUTS_MIN
            && (!round_out || tc_rows::tc_fp32_direct_wanted(rows,columns,inputs,false,true))
            && self.backend.tc_rows_pair_ready(rows,columns,inputs,&x1.vector,&x2.vector,&matrix);
        let few_cap=if tc_pair_ok {few_product::ONE_PANEL_INPUTS} else {few_product::FEW_MAX};
        if decode && (2..=few_cap).contains(&inputs) && self.backend.staged_few_pair_ready(rows,columns,inputs,&x1.vector,&x2.vector,&matrix) {
            let total=inputs.checked_mul(rows).ok_or("resident pair product: output too large")?;
            let mut output=self.backend.vector(total)?;
            self.backend.project_staged_few_pair_into(rows,columns,inputs,round_out,&x1.vector,&x2.vector,&matrix,&mut output)?;
            return Ok(Some(self.value(output,total)?));
        }
        if (decode && inputs<=few_cap) || !tc_pair_ok {return Ok(None);}
        let total=inputs.checked_mul(rows).ok_or("resident pair product: output too large")?;
        let mut output=self.backend.vector(total)?;
        self.backend.project_tc_rows_pair_form_into(rows,columns,inputs,round_out,&x1.vector,&x2.vector,&matrix,&mut output)?;
        Ok(Some(self.value(output,total)?))
    }
    /// Grouped dual-input narrow product for up to 3 matrices of one shape (`GateDown` Q/K/V in decode):
    /// uses `PLAIN_STAGED_FEW` for `inputs <= FEW_MAX` and direct-FP32 `PLAIN_TC_ROWS` for `inputs > FEW_MAX`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resident_group_rows_pair(&self,matrices:&[(&DeviceVector<'d>,std::ops::Range<usize>)],rows:usize,columns:usize,pairs:&[(&Value<'b,'d>,&Value<'b,'d>)],inputs:usize,round_out:bool)->Result<Option<Value<'b,'d>>> {
        if !(2..=3).contains(&matrices.len()) || matrices.len()!=pairs.len() || inputs==0 || !columns.is_multiple_of(2) {return Ok(None);}
        let half=columns/2;
        if pairs.iter().any(|(a,b)|a.words!=inputs*half || b.words!=inputs*half) {return Ok(None);}
        let weights=matrices.iter().map(|(w,words)|w.resident_range(words.clone())).collect::<std::result::Result<Vec<_>,_>>()?;
        let items=weights.iter().zip(pairs).map(|(w,(a,b))|(w,&*a.vector,&*b.vector)).collect::<Vec<_>>();
        if inputs==1 && self.backend.staged_narrow_group_pair_ready(rows,columns,&items) {
            for &(a,b) in pairs {self.ready(a)?;self.ready(b)?;}
            let total=matrices.len()*rows;
            let mut output=self.backend.vector(total)?;
            self.backend.project_staged_narrow_group_pair_into(rows,columns,round_out,&items,&mut output)?;
            return self.value(output,total).map(Some);
        }
        if inputs<=spark_gpu_probe::few_product::ONE_PANEL_INPUTS {
            if !self.backend.staged_few_group_pair_ready(rows,columns,inputs,&items) {return Ok(None);}
            for &(a,b) in pairs {self.ready(a)?;self.ready(b)?;}
            let total=matrices.len()*inputs*rows;
            let mut output=self.backend.vector(total)?;
            self.backend.project_staged_few_group_pair_into(rows,columns,inputs,round_out,&items,&mut output)?;
            return self.value(output,total).map(Some);
        }
        if !self.backend.tc_rows_group_pair_ready(rows,columns,inputs,&items) {
            if inputs<=spark_gpu_probe::few_product::FEW_MAX && self.backend.staged_few_group_pair_ready(rows,columns,inputs,&items) {
                for &(a,b) in pairs {self.ready(a)?;self.ready(b)?;}
                let total=matrices.len()*inputs*rows;
                let mut output=self.backend.vector(total)?;
                self.backend.project_staged_few_group_pair_into(rows,columns,inputs,round_out,&items,&mut output)?;
                return self.value(output,total).map(Some);
            }
            return Ok(None);
        }
        for &(a,b) in pairs {self.ready(a)?;self.ready(b)?;}
        let total=matrices.len()*inputs*rows;
        let mut output=self.backend.vector(total)?;
        self.backend.project_tc_rows_group_pair_into(rows,columns,inputs,round_out,&items,&mut output)?;
        self.value(output,total).map(Some)
    }
    /// Device-side deterministic MoE route grouping (`MOE_GROUP_ROUTE`), avoiding GPU-to-CPU sync.
    pub(crate) fn device_route_group(&self,route:&Value<'b,'d>,rows:usize,picks:usize,experts:usize)->Result<Option<(Value<'b,'d>,spark_gpu_probe::driver::MoeRoutePlan<'d>)>> {
        use spark_gpu_probe::tc_rows;
        if rows==0 || picks==0 || !route.words.is_multiple_of(rows) || route.vector.is_segmented() || !self.backend.resident_launches_active() {return Ok(None);}
        let each=route.words/rows;
        let Some(all)=rows.checked_mul(picks) else {return Ok(None)};
        let max_tiles=tc_rows::moe_max_tiles(all,experts);
        if !tc_rows::moe_route_admitted(rows,each,picks,experts,max_tiles,tc_rows::MOE_ROUTE_THREADS) {return Ok(None);}
        self.ready(route)?;
        let mut weights=self.backend.vector(all)?;
        let plan=self.backend.moe_group_route_into(rows,each,picks,experts,&route.vector,&mut weights)?;
        Ok(Some((self.value(weights,all)?,plan)))
    }
    /// Device-routed tensor-core MoE expert stack GEMM (`PLAIN_TC_ROWS` Mode 1).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn device_routed_products(&self,stack:&DeviceVector<'d>,experts:usize,rows:usize,columns:usize,route:&spark_gpu_probe::driver::MoeRoutePlan<'d>,x:&Value<'b,'d>,down:bool)->Result<Option<Value<'b,'d>>> {
        use spark_gpu_probe::tc_rows;
        if stack.is_segmented() || x.vector.is_segmented() || !self.backend.resident_launches_active() {return Ok(None);}
        if rows>tc_rows::TC_ROWS_MAX || !tc_rows::tc_shape_admitted(rows,columns,1) || route.experts()!=experts {return Ok(None);}
        if stack.len()!=experts*rows*columns/2 || (stack.device_address()|x.vector.device_address())&15!=0 || !(rows*columns*2).is_multiple_of(16) {return Ok(None);}
        self.ready(x)?;
        let total=route.picks()*rows;
        let mut output=self.backend.vector(total)?;
        self.backend.project_tc_moe_into(rows,columns,stack,route,&x.vector,&mut output,down)?;
        Ok(Some(self.value(output,total)?))
    }
    /// Device-routed tensor-core packed MoE expert stack GEMM (`PLAIN_PACKED_TC_ROWS`, kind 106).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn device_routed_packed_products(&self,stack:&DeviceVector<'d>,experts:usize,unit:engine_model::plan::loop_packed_unit::UnitShape,route:&spark_gpu_probe::driver::MoeRoutePlan<'d>,x:&Value<'b,'d>,down:bool)->Result<Option<Value<'b,'d>>> {
        use engine_model::plan::loop_load::ExpertCodes;
        use spark_gpu_probe::{packed_product,tc_rows};
        let (rows,columns,wide,pairs)=(unit.rows,unit.columns,unit.codes==ExpertCodes::Mxfp8,unit.order.is_some());
        if stack.is_segmented() || x.vector.is_segmented() || !self.backend.resident_launches_active() {return Ok(None);}
        if !tc_rows::tc_packed_moe_shape_admitted(rows,columns,wide,pairs) || route.experts()!=experts {return Ok(None);}
        let (plan,tile)=(unit.layout().map_err(|e|e.to_string())?,packed_product::layout(rows,columns,wide));
        need((plan.scales,plan.table,plan.bytes)==(tile.scales,tile.table,tile.bytes) && experts.checked_mul(plan.bytes)==stack.len().checked_mul(4),"device routed packed products: packed stack differs from unit layout")?;
        if (stack.device_address()|x.vector.device_address())&15!=0 {return Ok(None);}
        self.ready(x)?;
        let total=route.picks()*rows;
        let mut output=self.backend.vector(total)?;
        self.backend.project_tc_packed_moe_into(rows,columns,wide,pairs,stack,route,&x.vector,&mut output,down)?;
        Ok(Some(self.value(output,total)?))
    }
    /// The gated gate-and-up form of `device_routed_packed_products` (`PLAIN_PACKED_TC_ROWS` with `CappedGate` in its
    /// last step): `[picks][rows / 2]`, the words of the product followed by `fast_capped_gate_into`. None when the
    /// stack has no mixed gate and up rows or its shape is outside the launch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn device_routed_packed_gate(&self,stack:&DeviceVector<'d>,experts:usize,unit:engine_model::plan::loop_packed_unit::UnitShape,route:&spark_gpu_probe::driver::MoeRoutePlan<'d>,x:&Value<'b,'d>,caps:[f32;4])->Result<Option<Value<'b,'d>>> {
        use engine_model::plan::loop_load::ExpertCodes;
        use spark_gpu_probe::{packed_product,tc_rows};
        let (rows,columns,wide,pairs)=(unit.rows,unit.columns,unit.codes==ExpertCodes::Mxfp8,unit.order.is_some());
        if !pairs || stack.is_segmented() || x.vector.is_segmented() || !self.backend.resident_launches_active() {return Ok(None);}
        if !tc_rows::tc_packed_moe_shape_admitted(rows,columns,wide,pairs) || route.experts()!=experts {return Ok(None);}
        let (plan,tile)=(unit.layout().map_err(|e|e.to_string())?,packed_product::layout(rows,columns,wide));
        need((plan.scales,plan.table,plan.bytes)==(tile.scales,tile.table,tile.bytes) && experts.checked_mul(plan.bytes)==stack.len().checked_mul(4),"device routed packed gate: packed stack differs from unit layout")?;
        if (stack.device_address()|x.vector.device_address())&15!=0 {return Ok(None);}
        self.ready(x)?;
        let total=route.picks()*(rows/2);
        let mut output=self.backend.vector(total)?;
        self.backend.project_tc_packed_gate_into(rows,columns,wide,stack,route,&x.vector,caps,&mut output)?;
        Ok(Some(self.value(output,total)?))
    }
    /// The prefill `Down` product of the routed experts and its `RouteFold` with the down words in BF16: `[tokens][rows]`,
    /// the words of `device_routed_packed_products(Down)` then the fold over `weights` (`[tokens][picks]`, pick order).
    /// The fold rounds each down word with `rn`, so the product stores `rn` of each word as BF16, two a word, in a
    /// private vector (`project_tc_packed_down_half_into`), and the fold reads them (`fast_route_fold_half_into`): half
    /// the bytes of the FP32 downs written and read. None when the stack has mixed rows or a shape is outside the launches.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn device_routed_packed_down_fold(&self,stack:&DeviceVector<'d>,experts:usize,unit:engine_model::plan::loop_packed_unit::UnitShape,route:&spark_gpu_probe::driver::MoeRoutePlan<'d>,x:&Value<'b,'d>,weights:&Value<'b,'d>,picks:usize)->Result<Option<Value<'b,'d>>> {
        use engine_model::plan::loop_load::ExpertCodes;
        use spark_gpu_probe::{fast_ops,packed_product,tc_rows};
        let (rows,columns,wide,pairs)=(unit.rows,unit.columns,unit.codes==ExpertCodes::Mxfp8,unit.order.is_some());
        if pairs || !rows.is_multiple_of(2) || picks==0 || stack.is_segmented() || x.vector.is_segmented() || weights.vector.is_segmented() || !self.backend.resident_launches_active() {return Ok(None);}
        if !tc_rows::tc_packed_moe_shape_admitted(rows,columns,wide,pairs) || route.experts()!=experts {return Ok(None);}
        let all=route.picks();
        if !all.is_multiple_of(picks) || weights.words!=all {return Ok(None);}
        let tokens=all/picks;
        if !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_ROUTE_FOLD,tokens,rows,picks,fast_ops::STREAM_OPS_THREADS) {return Ok(None);}
        let (plan,tile)=(unit.layout().map_err(|e|e.to_string())?,packed_product::layout(rows,columns,wide));
        need((plan.scales,plan.table,plan.bytes)==(tile.scales,tile.table,tile.bytes) && experts.checked_mul(plan.bytes)==stack.len().checked_mul(4),"device routed down fold: packed stack differs from unit layout")?;
        if (stack.device_address()|x.vector.device_address())&15!=0 {return Ok(None);}
        self.ready(x)?;self.ready(weights)?;
        let stamp=self.cost.start(self.backend);
        // Freed after the fold is recorded, as the gated form frees its coefficient tables.
        let mut downs=self.backend.vector(all*rows/2)?;
        self.backend.project_tc_packed_down_half_into(rows,columns,wide,stack,route,&x.vector,&mut downs)?;
        let mut output=self.backend.vector(tokens*rows)?;
        self.backend.fast_route_fold_half_into(tokens,picks,rows,&weights.vector,&downs,&mut output)?;
        let value=self.value(output,tokens*rows)?;
        self.cost.record(||"routed_down_fold".to_owned(),stamp,self.backend,0);
        Ok(Some(value))
    }
    /// The prefill experts of one layer in three launches: `device_routed_packed_gate` with each gated word kept as
    /// the BF16 value that the `Down` product makes of it (`project_tc_packed_gate_half_into`), then
    /// `device_routed_packed_down_fold` on those BF16 rows (`project_tc_packed_down_half_from_half_into`): the words
    /// of the two calls, with no FP32 gated vector written, read or rounded again. `gate` and `down` are the packed
    /// stack, its expert count and its unit. None when a stack or shape is outside these launches.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn device_routed_packed_experts(&self,gate:(&DeviceVector<'d>,usize,engine_model::plan::loop_packed_unit::UnitShape),down:(&DeviceVector<'d>,usize,engine_model::plan::loop_packed_unit::UnitShape),route:&spark_gpu_probe::driver::MoeRoutePlan<'d>,x:&Value<'b,'d>,caps:[f32;4],weights:&Value<'b,'d>,picks:usize)->Result<Option<Value<'b,'d>>> {
        use engine_model::plan::loop_load::ExpertCodes;
        use spark_gpu_probe::{fast_ops,packed_product,tc_rows};
        let ((g_stack,g_experts,g_unit),(d_stack,d_experts,d_unit))=(gate,down);
        let (g_rows,g_cols,g_wide,g_pairs)=(g_unit.rows,g_unit.columns,g_unit.codes==ExpertCodes::Mxfp8,g_unit.order.is_some());
        let (d_rows,d_cols,d_wide,d_pairs)=(d_unit.rows,d_unit.columns,d_unit.codes==ExpertCodes::Mxfp8,d_unit.order.is_some());
        let inner=g_rows/2;
        // Gated BF16 rows are whole words (`inner` even); the down product reads them as its input (`d_cols == inner`).
        if !g_pairs || d_pairs || !inner.is_multiple_of(2) || d_cols!=inner || !d_rows.is_multiple_of(2) || picks==0 {return Ok(None);}
        if [g_stack,d_stack,&x.vector,&weights.vector].iter().any(|v|v.is_segmented()) || !self.backend.resident_launches_active() {return Ok(None);}
        if !tc_rows::tc_packed_moe_shape_admitted(g_rows,g_cols,g_wide,g_pairs) || !tc_rows::tc_packed_moe_shape_admitted(d_rows,d_cols,d_wide,d_pairs) {return Ok(None);}
        if route.experts()!=g_experts || route.experts()!=d_experts {return Ok(None);}
        let all=route.picks();
        if !all.is_multiple_of(picks) || weights.words!=all {return Ok(None);}
        let tokens=all/picks;
        if !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_ROUTE_FOLD,tokens,d_rows,picks,fast_ops::STREAM_OPS_THREADS) {return Ok(None);}
        for (stack,experts,unit,(rows,cols,wide)) in [(g_stack,g_experts,g_unit,(g_rows,g_cols,g_wide)),(d_stack,d_experts,d_unit,(d_rows,d_cols,d_wide))] {
            let (plan,tile)=(unit.layout().map_err(|e|e.to_string())?,packed_product::layout(rows,cols,wide));
            need((plan.scales,plan.table,plan.bytes)==(tile.scales,tile.table,tile.bytes) && experts.checked_mul(plan.bytes)==stack.len().checked_mul(4),"device routed experts: packed stack differs from unit layout")?;
        }
        if (g_stack.device_address()|d_stack.device_address()|x.vector.device_address())&15!=0 {return Ok(None);}
        self.ready(x)?;self.ready(weights)?;
        let stamp=self.cost.start(self.backend);
        if (tokens*g_cols).is_multiple_of(8) {
            let mut output=self.backend.vector(tokens*d_rows)?;
            self.backend.project_prefill_packed_experts_into(
                g_rows,g_cols,g_wide,g_stack,d_rows,d_cols,d_wide,d_stack,
                route,&x.vector,caps,&weights.vector,picks,&mut output,
            )?;
            let value=self.value(output,tokens*d_rows)?;
            self.cost.record(||"routed_experts_prefill_exact".to_owned(),stamp,self.backend,0);
            return Ok(Some(value));
        }
        // Freed after the fold is recorded, as the gated form frees its coefficient tables.
        let mut active=self.backend.vector(all*inner/2)?;
        self.backend.project_tc_packed_gate_half_into(g_rows,g_cols,g_wide,g_stack,route,&x.vector,caps,&mut active)?;
        let mut downs=self.backend.vector(all*d_rows/2)?;
        self.backend.project_tc_packed_down_half_from_half_into(d_rows,d_cols,d_wide,d_stack,route,&active,&mut downs)?;
        let mut output=self.backend.vector(tokens*d_rows)?;
        self.backend.fast_route_fold_half_into(tokens,picks,d_rows,&weights.vector,&downs,&mut output)?;
        let value=self.value(output,tokens*d_rows)?;
        self.cost.record(||"routed_experts".to_owned(),stamp,self.backend,0);
        Ok(Some(value))
    }
    /// Prefill `device_routed_packed_experts` with optional packed-BF16 input (`half_in`), skipping `TC_MODE_PACK`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn device_routed_packed_experts_form(&self,gate:(&DeviceVector<'d>,usize,engine_model::plan::loop_packed_unit::UnitShape),down:(&DeviceVector<'d>,usize,engine_model::plan::loop_packed_unit::UnitShape),route:&spark_gpu_probe::driver::MoeRoutePlan<'d>,x:&Value<'b,'d>,caps:[f32;4],weights:&Value<'b,'d>,picks:usize,half_in:bool)->Result<Option<Value<'b,'d>>> {
        use engine_model::plan::loop_load::ExpertCodes;
        use spark_gpu_probe::{fast_ops,packed_product,tc_rows};
        let ((g_stack,g_experts,g_unit),(d_stack,d_experts,d_unit))=(gate,down);
        let (g_rows,g_cols,g_wide,g_pairs)=(g_unit.rows,g_unit.columns,g_unit.codes==ExpertCodes::Mxfp8,g_unit.order.is_some());
        let (d_rows,d_cols,d_wide,d_pairs)=(d_unit.rows,d_unit.columns,d_unit.codes==ExpertCodes::Mxfp8,d_unit.order.is_some());
        let inner=g_rows/2;
        if !g_pairs || d_pairs || !inner.is_multiple_of(2) || d_cols!=inner || !d_rows.is_multiple_of(2) || picks==0 {return Ok(None);}
        if [g_stack,d_stack,&x.vector,&weights.vector].iter().any(|v|v.is_segmented()) || !self.backend.resident_launches_active() {return Ok(None);}
        if !tc_rows::tc_packed_moe_shape_admitted(g_rows,g_cols,g_wide,g_pairs) || !tc_rows::tc_packed_moe_shape_admitted(d_rows,d_cols,d_wide,d_pairs) {return Ok(None);}
        if route.experts()!=g_experts || route.experts()!=d_experts {return Ok(None);}
        let all=route.picks();
        if !all.is_multiple_of(picks) || weights.words!=all {return Ok(None);}
        let tokens=all/picks;
        if x.words!=if half_in {tokens*(g_cols/2)} else {tokens*g_cols} {return Ok(None);}
        if !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_ROUTE_FOLD,tokens,d_rows,picks,fast_ops::STREAM_OPS_THREADS) {return Ok(None);}
        for (stack,experts,unit,(rows,cols,wide)) in [(g_stack,g_experts,g_unit,(g_rows,g_cols,g_wide)),(d_stack,d_experts,d_unit,(d_rows,d_cols,d_wide))] {
            let (plan,tile)=(unit.layout().map_err(|e|e.to_string())?,packed_product::layout(rows,cols,wide));
            need((plan.scales,plan.table,plan.bytes)==(tile.scales,tile.table,tile.bytes) && experts.checked_mul(plan.bytes)==stack.len().checked_mul(4),"device routed experts: packed stack differs from unit layout")?;
        }
        if (g_stack.device_address()|d_stack.device_address()|x.vector.device_address())&15!=0 {return Ok(None);}
        self.ready(x)?;self.ready(weights)?;
        let stamp=self.cost.start(self.backend);
        if half_in || (tokens*g_cols).is_multiple_of(8) {
            let mut output=self.backend.vector(tokens*d_rows)?;
            self.backend.project_prefill_packed_experts_form_into(
                g_rows,g_cols,g_wide,g_stack,d_rows,d_cols,d_wide,d_stack,
                route,&x.vector,half_in,caps,&weights.vector,picks,&mut output,
            )?;
            let value=self.value(output,tokens*d_rows)?;
            self.cost.record(||"routed_experts_prefill_exact".to_owned(),stamp,self.backend,0);
            return Ok(Some(value));
        }
        let mut active=self.backend.vector(all*inner/2)?;
        if half_in {
            self.backend.project_tc_packed_gate_half_from_half_into(g_rows,g_cols,g_wide,g_stack,route,&x.vector,caps,&mut active)?;
        } else {
            self.backend.project_tc_packed_gate_half_into(g_rows,g_cols,g_wide,g_stack,route,&x.vector,caps,&mut active)?;
        }
        let mut downs=self.backend.vector(all*d_rows/2)?;
        self.backend.project_tc_packed_down_half_from_half_into(d_rows,d_cols,d_wide,d_stack,route,&active,&mut downs)?;
        let mut output=self.backend.vector(tokens*d_rows)?;
        self.backend.fast_route_fold_half_into(tokens,picks,d_rows,&weights.vector,&downs,&mut output)?;
        let value=self.value(output,tokens*d_rows)?;
        self.cost.record(||"routed_experts".to_owned(),stamp,self.backend,0);
        Ok(Some(value))
    }
    /// Prefill `device_routed_packed_experts` fused with `AfterExperts ShortConv` + `next_ring`:
    /// `GateUp` (`HALF_OUT`, optional `HALF_IN`) -> `Down` (`HALF_IN`, `HALF_OUT`) -> `STREAM_OP_ROUTE_FOLD_CONV_PREFILL` (`output`, `next_ring`),
    /// eliminating the `[tokens][width]` intermediate folded buffer (37.7 MB write + 4x read) and `prefill_ring_after`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn device_routed_packed_experts_close(
        &self,
        gate:(&DeviceVector<'d>,usize,engine_model::plan::loop_packed_unit::UnitShape),
        down:(&DeviceVector<'d>,usize,engine_model::plan::loop_packed_unit::UnitShape),
        route:&spark_gpu_probe::driver::MoeRoutePlan<'d>,
        x:&Value<'b,'d>,caps:[f32;4],weights:&Value<'b,'d>,picks:usize,
        residual:&Value<'b,'d>,fixed:&[f32],ring:&Value<'b,'d>,taps:usize,first_slot:usize,bound:f32,half_in:bool,
    )->Result<Option<(Value<'b,'d>,Value<'b,'d>)>> {
        use engine_model::plan::loop_load::ExpertCodes;
        use spark_gpu_probe::{fast_ops,packed_product,tc_rows};
        let ((g_stack,g_experts,g_unit),(d_stack,d_experts,d_unit))=(gate,down);
        let (g_rows,g_cols,g_wide,g_pairs)=(g_unit.rows,g_unit.columns,g_unit.codes==ExpertCodes::Mxfp8,g_unit.order.is_some());
        let (d_rows,d_cols,d_wide,d_pairs)=(d_unit.rows,d_unit.columns,d_unit.codes==ExpertCodes::Mxfp8,d_unit.order.is_some());
        let inner=g_rows/2;
        if !g_pairs || d_pairs || !inner.is_multiple_of(2) || d_cols!=inner || !d_rows.is_multiple_of(2) || picks==0 {return Ok(None);}
        if [g_stack,d_stack,&x.vector,&weights.vector,&residual.vector,&ring.vector].iter().any(|v|v.is_segmented()) || !self.backend.resident_launches_active() {return Ok(None);}
        if !tc_rows::tc_packed_moe_shape_admitted(g_rows,g_cols,g_wide,g_pairs) || !tc_rows::tc_packed_moe_shape_admitted(d_rows,d_cols,d_wide,d_pairs) {return Ok(None);}
        if route.experts()!=g_experts || route.experts()!=d_experts {return Ok(None);}
        let all=route.picks();
        if !all.is_multiple_of(picks) || weights.words!=all {return Ok(None);}
        let tokens=all/picks;
        if x.words!=if half_in {tokens*(g_cols/2)} else {tokens*g_cols} {return Ok(None);}
        if !fast_ops::stream_ops_admitted(fast_ops::STREAM_OP_ROUTE_FOLD_CONV_PREFILL,tokens,d_rows,picks,fast_ops::ROUTE_FOLD_PREFILL_THREADS)
            || !(2..=fast_ops::ROUTE_FOLD_CONV_TAPS_MAX).contains(&taps) || tokens<taps || first_slot>=taps
            || residual.words!=tokens*d_rows || ring.words!=taps*d_rows || fixed.len()!=d_rows*taps
        {
            return Ok(None);
        }
        for (stack,experts,unit,(rows,cols,wide)) in [(g_stack,g_experts,g_unit,(g_rows,g_cols,g_wide)),(d_stack,d_experts,d_unit,(d_rows,d_cols,d_wide))] {
            let (plan,tile)=(unit.layout().map_err(|e|e.to_string())?,packed_product::layout(rows,cols,wide));
            need((plan.scales,plan.table,plan.bytes)==(tile.scales,tile.table,tile.bytes) && experts.checked_mul(plan.bytes)==stack.len().checked_mul(4),"device routed experts close: packed stack differs from unit layout")?;
        }
        if (g_stack.device_address()|d_stack.device_address()|x.vector.device_address())&15!=0 {return Ok(None);}
        self.ready(x)?;self.ready(weights)?;self.ready(residual)?;self.ready(ring)?;
        let stamp=self.cost.start(self.backend);
        let fixed_dev=self.parameters.borrow_mut().resolve(fixed,None,|v|self.backend.upload_vector(v)?.into_resident_shared())?
            .ok_or("device routed experts close: missing fixed taps")?;
        if fixed_dev.is_segmented() {return Ok(None);}
        let mut active=self.backend.vector(all*inner/2)?;
        let mut downs=self.backend.vector(all*d_rows/2)?;
        if half_in || (tokens*g_cols).is_multiple_of(8) {
            self.backend.project_prefill_packed_moe_into(
                g_rows,g_cols,g_wide,true,spark_gpu_probe::prefill_moe::PREFILL_OUT_BF16_GATED,
                g_stack,route,&x.vector,half_in,&mut active,false,Some(caps),
            )?;
            self.backend.project_prefill_packed_moe_into(
                d_rows,d_cols,d_wide,false,spark_gpu_probe::prefill_moe::PREFILL_OUT_BF16,
                d_stack,route,&active,true,&mut downs,true,None,
            )?;
        } else {
            self.backend.project_tc_packed_gate_half_into(g_rows,g_cols,g_wide,g_stack,route,&x.vector,caps,&mut active)?;
            self.backend.project_tc_packed_down_half_from_half_into(d_rows,d_cols,d_wide,d_stack,route,&active,&mut downs)?;
        }
        let mut output=self.backend.vector(tokens*d_rows)?;
        let mut next_ring=self.backend.vector(taps*d_rows)?;
        self.backend.fast_route_fold_conv_prefill_into(
            tokens,picks,d_rows,taps,first_slot,bound,
            &weights.vector,&downs,&residual.vector,&fixed_dev,&ring.vector,&mut output,&mut next_ring,
        )?;
        let res=(self.value(output,tokens*d_rows)?,self.value(next_ring,taps*d_rows)?);
        self.cost.record(||"routed_experts_close".to_owned(),stamp,self.backend,fixed.len()*4);
        Ok(Some(res))
    }
    /// Prefill fused `GateOuter (GateForget+GateGain) + GateBlend + RoundedNorm` (`TC_MODE_GATE_BLEND_NORM`)
    /// across 1..=3 gates in one `PLAIN_TC_BIG` launch, with optional packed-BF16 outputs (`half_out`).
    pub(crate) fn resident_gate_outer_blend_norm(
        &self,
        entries:&[(&DeviceVector<'d>,std::ops::Range<usize>,&Value<'b,'d>,&Value<'b,'d>,&Value<'b,'d>,Option<&[f32]>)],
        rows:usize,dim:usize,k:usize,eps:f32,half_out:bool,
    )->Result<Option<Vec<Value<'b,'d>>>> {
        use spark_gpu_probe::{few_product,tc_rows};
        if (!self.loop_prefill.get() && rows<=few_product::FEW_MAX) || !self.backend.resident_launches_active()
            || !tc_rows::gate_outer_admitted(rows,dim,k,entries.len(),tc_rows::TC_THREADS)
            || !(eps>0.0 && eps.is_finite())
        {
            return Ok(None);
        }
        let (in_words,v_words,w_words)=(rows*k,rows*dim,dim*k);
        let out_words=if half_out {v_words/2} else {v_words};
        let mut w_views=Vec::with_capacity(entries.len());
        for &(w,ref range,inner,kept,msg,nw) in entries {
            if range.len()!=w_words || inner.words!=in_words || kept.words!=v_words || msg.words!=v_words
                || w.is_segmented() || inner.vector.is_segmented() || kept.vector.is_segmented() || msg.vector.is_segmented()
                || nw.is_some_and(|n|n.len()!=dim)
            {
                return Ok(None);
            }
            let view=w.resident_range(range.clone())?;
            if (view.device_address()|inner.vector.device_address())&15!=0 {return Ok(None);}
            w_views.push(view);
        }
        for &(_,_,inner,kept,msg,_) in entries {
            self.ready(inner)?;self.ready(kept)?;self.ready(msg)?;
        }
        let stamp=self.cost.start(self.backend);
        let mut nw_devs:Vec<Option<Rc<DeviceVector<'d>>>>=Vec::with_capacity(entries.len());
        for &(_,_,_,_,_,nw) in entries {
            let d=match nw {
                Some(n)=>{
                    let dev=self.parameters.borrow_mut().resolve(n,None,|v|self.backend.upload_vector(v)?.into_resident_shared())?
                        .ok_or("tc gate blend norm: missing norm weight")?;
                    if dev.is_segmented() {return Ok(None);}
                    Some(dev)
                }
                None=>None,
            };
            nw_devs.push(d);
        }
        let refs:Vec<_>=w_views.iter().zip(entries).zip(&nw_devs)
            .map(|((wv,&(_,_,inner,kept,msg,_)),nw)|(wv,&*inner.vector,&*kept.vector,&*msg.vector,nw.as_deref()))
            .collect();
        let outs=self.backend.project_tc_gate_blend_norm(rows,dim,k,eps,half_out,&refs)?;
        let values=outs.into_iter().map(|v|self.value(v,out_words)).collect::<Result<Vec<_>>>()?;
        self.cost.record(||"tc_gate_blend_norm".to_owned(),stamp,self.backend,0);
        Ok(Some(values))
    }
    /// Prefill BF16-input dense product (`HALF_IN`, skips `TC_MODE_PACK`, uses `TC_MODE_NARROW_HALF` for `rows <= 128`).
    pub(crate) fn resident_product_rows_half_in(
        &self,weight:&DeviceVector<'d>,words:std::ops::Range<usize>,rows:usize,columns:usize,x:&Value<'b,'d>,inputs:usize,
    )->Result<Option<Value<'b,'d>>> {
        use spark_gpu_probe::{few_product,tc_rows};
        if (!self.loop_prefill.get() && inputs<=few_product::FEW_MAX) || !columns.is_multiple_of(2) || weight.is_segmented() || x.vector.is_segmented() {return Ok(None);}
        if rows==0 || inputs==0 || rows.checked_mul(columns)!=words.len().checked_mul(2) || x.words!=inputs*(columns/2) {return Ok(None);}
        let Some(part)=tc_rows::tc_tile_part(rows,columns,inputs) else {return Ok(None)};
        if part<rows {return Ok(None);}
        let matrix=weight.resident_range(words)?;
        if !self.backend.tc_rows_half_ready(rows,columns,inputs,&x.vector,&matrix) {return Ok(None);}
        self.ready(x)?;
        let total=inputs.checked_mul(rows).ok_or("resident half-in product: output too large")?;
        let mut output=self.backend.vector(total)?;
        self.backend.project_tc_rows_half_in_into(rows,columns,inputs,&x.vector,&matrix,&mut output)?;
        Ok(Some(self.value(output,total)?))
    }
    /// Prefill BF16-input split multi-output product (`HALF_IN`, skips `TC_MODE_PACK`).
    pub(crate) fn resident_product_rows_split_half_in(
        &self,weight:&DeviceVector<'d>,words:std::ops::Range<usize>,parts:&[usize],columns:usize,x:&Value<'b,'d>,inputs:usize,
    )->Result<Option<Vec<Value<'b,'d>>>> {
        self.resident_product_rows_split_halves(weight,words,parts,columns,x,inputs,0)
    }
    /// `resident_product_rows_split_half_in` with the parts of `halves` (bit `i`: part `i`) as BF16 pairs
    /// (`tc_rows::TC_SPLIT_HALF_OUT`, `inputs * part / 2` words): for a reader that rounds each word it reads.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resident_product_rows_split_halves(
        &self,weight:&DeviceVector<'d>,words:std::ops::Range<usize>,parts:&[usize],columns:usize,x:&Value<'b,'d>,inputs:usize,halves:usize,
    )->Result<Option<Vec<Value<'b,'d>>>> {
        use spark_gpu_probe::{few_product,tc_rows};
        let size=|i:usize,w:usize|if (halves>>i)&1!=0 {inputs*w/2} else {inputs*w};
        let rows:usize=parts.iter().sum();
        if (!self.loop_prefill.get() && inputs<=few_product::FEW_MAX) || !columns.is_multiple_of(2) || parts.is_empty() || parts.len()>tc_rows::TC_SPLIT_MAX
            || parts.contains(&0) || parts.iter().any(|&w|!w.is_multiple_of(2))
            || weight.is_segmented() || x.vector.is_segmented() || rows==0 || inputs<tc_rows::TC_ROWS_INPUTS_MIN
            || rows.checked_mul(columns)!=words.len().checked_mul(2) || x.words!=inputs*(columns/2)
        {
            return Ok(None);
        }
        let matrix=weight.resident_range(words)?;
        if !self.backend.tc_rows_half_ready(rows,columns,inputs,&x.vector,&matrix) {return Ok(None);}
        if halves!=0 && (halves>>parts.len()!=0 || !self.backend.tc_split_halves_ready(rows,columns,inputs,&x.vector,&matrix)) {return Ok(None);}
        self.ready(x)?;
        let mut buffers=Vec::with_capacity(parts.len());
        for (i,&w) in parts.iter().enumerate() {buffers.push(self.backend.vector(size(i,w))?);}
        {
            let mut refs:Vec<(&mut DeviceVector<'d>,usize)>=buffers.iter_mut().zip(parts.iter().copied()).collect();
            self.backend.project_tc_rows_split_halves_into(rows,columns,inputs,halves,&x.vector,&matrix,&mut refs)?;
        }
        let mut out=Vec::with_capacity(parts.len());
        for (i,(buf,&w)) in buffers.into_iter().zip(parts).enumerate() {out.push(self.value(buf,size(i,w))?);}
        Ok(Some(out))
    }
    /// Prefill dual-input BF16 product over `[x1[t], x2[t]]` packed BF16 halves (`TC_MODE_NARROW_HALF` for `rows <= 128`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resident_product_rows_pair_half_in(
        &self,weight:&DeviceVector<'d>,words:std::ops::Range<usize>,rows:usize,columns:usize,x1:&Value<'b,'d>,x2:&Value<'b,'d>,inputs:usize,
    )->Result<Option<Value<'b,'d>>> {
        use spark_gpu_probe::{few_product,tc_rows};
        if (!self.loop_prefill.get() && inputs<=few_product::FEW_MAX) || !columns.is_multiple_of(4) || weight.is_segmented() || x1.vector.is_segmented() || x2.vector.is_segmented() {return Ok(None);}
        let quarter=columns/4;
        if rows==0 || inputs<tc_rows::TC_ROWS_INPUTS_MIN || x1.words!=inputs*quarter || x2.words!=inputs*quarter
            || rows.checked_mul(columns)!=words.len().checked_mul(2)
        {
            return Ok(None);
        }
        let matrix=weight.resident_range(words)?;
        if !self.backend.tc_rows_pair_half_ready(rows,columns,inputs,&x1.vector,&x2.vector,&matrix) {return Ok(None);}
        self.ready(x1)?;self.ready(x2)?;
        let total=inputs.checked_mul(rows).ok_or("resident pair half-in product: output too large")?;
        let mut output=self.backend.vector(total)?;
        self.backend.project_tc_rows_pair_half_in_into(rows,columns,inputs,&x1.vector,&x2.vector,&matrix,&mut output)?;
        Ok(Some(self.value(output,total)?))
    }
    /// Prefill `FLASH_WINDOW_ATTN` with `FW_CLIP_IN | FW_HALF_OUT` (packed BF16 output for `Weight::Output`).
    pub(crate) fn record_window_rows_half(&self,a:&super::WindowOne<'_,Value<'b,'d>>,first_position:u64)->Result<Option<Value<'b,'d>>> {
        self.record_window_rows_half_form(a,first_position,false)
    }
    /// `record_window_rows_half` with the form of the query and the gate: FP32 words, or with `qg_half` BF16 pairs
    /// (`flash_window::FW_QG_HALF`). The pair form has only the prefill kernel: a launch that kernel does not take
    /// gives `None` before any launch.
    pub(crate) fn record_window_rows_half_form(&self,a:&super::WindowOne<'_,Value<'b,'d>>,first_position:u64,qg_half:bool)->Result<Option<Value<'b,'d>>> {
        use spark_gpu_probe::flash_window;
        let (rows,heads,kv_heads,view,extent)=(a.rows,a.heads,a.kv_heads,a.view,a.extent);
        let (head_dim,rel_dim)=(flash_window::FW_HEAD_DIM,flash_window::FW_REL_DIM);
        let qg_words=if qg_half {rows*heads*head_dim/2} else {rows*heads*head_dim};
        if !self.loop_prefill.get() || rows<flash_window::FW_ROWS_MIN || !self.backend.resident_launches_active()
            || !flash_window::flash_window_shape_admitted(rows,heads,kv_heads,head_dim,rel_dim,view,extent)
        {
            return Ok(None);
        }
        if a.query.words!=qg_words || a.gate.words!=qg_words || a.moved.words!=rows*heads*rel_dim
            || a.sink.words!=rows*heads || a.steps.words!=2*rows || a.rel_table.len()!=rel_dim*extent
            || !(a.query_scale.is_empty() || a.query_scale.len()==heads*head_dim)
            || !(a.moved_scale.is_empty() || a.moved_scale.len()==heads*rel_dim) || a.position.is_some_and(|p|p.words!=rows)
            || !(a.query_bound>0.0 && a.moved_bound>0.0) || a.written.is_some()
        {
            return Ok(None);
        }
        let operands=[a.query,a.moved,a.sink,a.gate,a.steps];
        for v in operands.iter().chain(a.position.iter()) {if v.vector.is_segmented() {return Ok(None);}}
        let kv_words=view*kv_heads*(head_dim/2);
        if a.keys.words!=kv_words || a.values.words!=kv_words || a.keys.vector.len()<kv_words || a.values.vector.len()<kv_words {return Ok(None);}
        if (a.query.vector.device_address()|a.keys.vector.device_address()|a.values.vector.device_address()|a.moved.vector.device_address()|a.gate.vector.device_address())&15!=0 {
            return Ok(None);
        }
        for v in operands.iter().chain([a.keys,a.values].iter()).chain(a.position.iter()) {self.ready(v)?;}
        let stamp=self.cost.start(self.backend);
        let upload=|words:&[f32]|self.parameters.borrow_mut().resolve(words,None,|values|self.backend.upload_vector(values)?.into_resident_shared());
        let table=upload(a.rel_table)?.ok_or("flash window half: relative table empty")?;
        let (query_scale,moved_scale)=(upload(a.query_scale)?,upload(a.moved_scale)?);
        let clips=spark_gpu_probe::driver::OneClips {
            query_scale:query_scale.as_deref(),moved_scale:moved_scale.as_deref(),position:a.position.map(|p|&*p.vector),
            query_bound:a.query_bound,moved_bound:a.moved_bound,written_keys:None,written_values:None,
        };
        let out_words=rows*heads*head_dim/2;
        let mut output=self.backend.vector(out_words)?;
        if self.backend.prefill_flash_window_ready(
            rows,heads,kv_heads,view,extent,
            &a.query.vector,&a.keys.vector,&a.values.vector,&a.moved.vector,&table,&a.gate.vector,&output,
        ) {
            self.backend.prefill_flash_window_form_into(
                rows,heads,kv_heads,head_dim,rel_dim,view,extent,a.scale,first_position,true,qg_half,
                &a.query.vector,&a.keys.vector,&a.values.vector,&a.moved.vector,&table,&a.sink.vector,&a.gate.vector,&a.steps.vector,
                Some(clips),&mut output,
            )?;
        } else if qg_half {
            return Ok(None);
        } else {
            self.backend.flash_window_attn_half_into(
                rows,heads,kv_heads,head_dim,rel_dim,view,extent,a.scale,first_position,
                &a.query.vector,&a.keys.vector,&a.values.vector,&a.moved.vector,&table,&a.sink.vector,&a.gate.vector,&a.steps.vector,
                Some(clips),&mut output,
            )?;
        }
        let value=self.value(output,out_words)?;
        self.cost.record(||"window_rows_half".into(),stamp,self.backend,(a.rel_table.len()+a.query_scale.len()+a.moved_scale.len())*4);
        Ok(Some(value))
    }
    /// Prefill `MIX_ONE_POST_HALF_OUT`: emits packed BF16 `post` (`rows * asked * dim / 2` words).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_mix_rows_normed_half(
        &self,rows:usize,dim:usize,scale:f32,eps:f32,post_eps:f32,
        transpose:bool,keep_normed:bool,keep_mixed:bool,
        queries:&[f32],list:&[&Value<'b,'d>],stream:&Value<'b,'d>,
        post_weight:Option<&[f32]>,post_weight_per_query:bool,
    )->Result<Option<(Option<Value<'b,'d>>,Option<Value<'b,'d>>,Value<'b,'d>)>> {
        use spark_gpu_probe::{fast_ops,few_product};
        if (!self.loop_prefill.get() && rows<=few_product::FEW_MAX) || !self.backend.resident_launches_active() || dim==0 || !dim.is_multiple_of(2) || !queries.len().is_multiple_of(dim) {
            return Ok(None);
        }
        let asked=queries.len()/dim;
        let kept=list.len()+1;
        let row_words=rows.saturating_mul(dim);
        if !fast_ops::mix_one_admitted(rows,asked,kept,dim)
            || !fast_ops::mix_list_norm_admitted(dim,fast_ops::MIX_BLEND_THREADS)
            || !(post_eps>0.0 && post_eps.is_finite())
            || stream.words!=row_words || stream.vector.is_segmented() || stream.vector.len()!=row_words || stream.vector.device_address()&15!=0
            || list.iter().any(|v|v.words!=row_words || v.vector.is_segmented() || v.vector.len()!=row_words || v.vector.device_address()&15!=0)
        {
            return Ok(None);
        }
        if let Some(pw)=post_weight {
            let need_w=if post_weight_per_query {asked.saturating_mul(dim)} else {dim};
            if pw.len()!=need_w {return Ok(None);}
        }
        for value in list {self.ready(value)?;}
        self.ready(stream)?;
        let stamp=self.cost.start(self.backend);
        let q_dev=self.parameters.borrow_mut().resolve(queries,None,|v|self.backend.upload_vector(v)?.into_resident_shared())?.ok_or("fast mix one half has no queries")?;
        if q_dev.is_segmented() || q_dev.device_address()&15!=0 {return Ok(None);}
        let pw_dev=match post_weight {
            Some(pw)=>{
                let d=self.parameters.borrow_mut().resolve(pw,None,|v|self.backend.upload_vector(v)?.into_resident_shared())?.ok_or("fast mix one half has no post weight")?;
                if d.is_segmented() {return Ok(None);}
                Some(d)
            }
            None=>None,
        };
        let refs:Vec<&DeviceVector<'d>>=list.iter().map(|v|&*v.vector).collect();
        let (normed,mixed,post)=self.backend.fast_mix_rows_normed_half(
            rows,dim,scale,eps,post_eps,transpose,keep_normed,keep_mixed,
            &q_dev,&refs,&stream.vector,pw_dev.as_deref(),post_weight_per_query,
        )?;
        let normed=match normed {Some(v)=>Some(self.value(v,row_words)?),None=>None};
        let fp_words=rows*asked*dim;
        let mixed=match mixed {Some(v)=>Some(self.value(v,fp_words)?),None=>None};
        let post=self.value(post,fp_words/2)?;
        self.cost.record(||"fast_mix_one_normed_half".into(),stamp,self.backend,0);
        Ok(Some((normed,mixed,post)))
    }
    /// Prefill `MIX_LIST_NORM_HALF`: emits FP32 `messages` and packed BF16 `normed` (`rows * asked * dim / 2` words).
    pub(crate) fn try_mix_prefill_normed_half(
        &self,rows:usize,dim:usize,scale:f32,eps:f32,queries:&[f32],list:&[&Value<'b,'d>],
    )->Result<Option<(Value<'b,'d>,Value<'b,'d>)>> {
        use spark_gpu_probe::fast_ops;
        if !self.loop_prefill.get() || rows<2 || !self.backend.resident_launches_active() || dim==0 || !dim.is_multiple_of(2) || dim>fast_ops::MIX_ONE_DIM_MAX {
            return Ok(None);
        }
        let Some(asked)=queries.len().checked_div(dim) else {return Ok(None)};
        let kept=list.len();
        if !queries.len().is_multiple_of(dim)
            || !fast_ops::row_dots_admitted(rows,asked,kept,dim,fast_ops::ROW_DOTS_THREADS)
            || !fast_ops::mix_blend_admitted(rows,asked,kept,dim,fast_ops::MIX_BLEND_THREADS)
            || !fast_ops::mix_list_norm_admitted(dim,fast_ops::MIX_BLEND_THREADS)
            || !fast_ops::norm_admitted(rows*asked,dim,true,fast_ops::NORM_THREADS)
            || !(eps>0.0 && eps.is_finite())
            || list.iter().any(|v|v.vector.is_segmented() || v.vector.len()!=rows*dim)
        {
            return Ok(None);
        }
        for value in list {self.ready(value)?;}
        let stamp=self.cost.start(self.backend);
        let queries=self.parameters.borrow_mut().resolve(queries,None,|v|self.backend.upload_vector(v)?.into_resident_shared())?.ok_or("prefill mix half has no queries")?;
        let list=list.iter().map(|v|&*v.vector).collect::<Vec<_>>();
        let (out,normed)=self.backend.fast_mix_list_normed_half(rows,dim,scale,eps,&queries,&list)?;
        let (out_words,norm_words)=(out.len(),normed.len());
        let both=(self.value(out,out_words)?,self.value(normed,norm_words)?);
        self.cost.record(||"fast_mix_list_normed_half".into(),stamp,self.backend,0);
        Ok(Some(both))
    }
    /// Prefill `PG_OP_MSG_META` (kind 123, sub-op 0): compact lazy `(msg_weights, msg_nscales)` replacing `MIX_LIST_NORM_HALF`.
    pub(crate) fn try_prefill_msg_meta(
        &self,rows:usize,dim:usize,scale:f32,eps:f32,queries:&[f32],list:&[&Value<'b,'d>],
    )->Result<Option<(Value<'b,'d>,Value<'b,'d>)>> {
        use spark_gpu_probe::prefill_gate;
        if !self.loop_prefill.get() || rows<2 || !self.backend.resident_launches_active() || dim==0 || !queries.len().is_multiple_of(dim) {
            return Ok(None);
        }
        let asked=queries.len()/dim;
        let kept=list.len();
        if !prefill_gate::msg_meta_admitted(rows,asked,kept,dim,prefill_gate::PG_META_THREADS)
            || !scale.is_finite() || !(eps>0.0 && eps.is_finite())
            || list.iter().any(|v|v.vector.is_segmented() || v.words!=rows*dim || v.vector.device_address()&15!=0)
        {
            return Ok(None);
        }
        for value in list {self.ready(value)?;}
        let stamp=self.cost.start(self.backend);
        let q_dev=self.parameters.borrow_mut().resolve(queries,None,|v|self.backend.upload_vector(v)?.into_resident_shared())?.ok_or("prefill msg meta has no queries")?;
        if q_dev.is_segmented() || q_dev.device_address()&15!=0 {return Ok(None);}
        let refs:Vec<&DeviceVector<'d>>=list.iter().map(|v|&*v.vector).collect();
        if !self.backend.prefill_msg_meta_ready(rows,asked,dim,&q_dev,&refs) {return Ok(None);}
        let (weights,nscales)=self.backend.prefill_msg_meta(rows,dim,scale,eps,&q_dev,&refs)?;
        let (w_len,n_len)=(weights.len(),nscales.len());
        let both=(self.value(weights,w_len)?,self.value(nscales,n_len)?);
        self.cost.record(||"prefill_msg_meta".into(),stamp,self.backend,0);
        Ok(Some(both))
    }
    /// Prefill `PG_OP_FULL` (kind 123, sub-op 1): fused `GateDown + GateOuter + GateBlend + RoundedNorm` over lazy sources.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn resident_prefill_gate_fuse(
        &self,
        sources:&[&Value<'b,'d>],
        entries:&[(&DeviceVector<'d>,std::ops::Range<usize>,&DeviceVector<'d>,std::ops::Range<usize>,&Value<'b,'d>,&Value<'b,'d>,&Value<'b,'d>,&Value<'b,'d>,Option<&[f32]>)],
        rows:usize,dim:usize,k:usize,eps:f32,half_out:bool,
    )->Result<Option<Vec<Value<'b,'d>>>> {
        use spark_gpu_probe::prefill_gate;
        let (count,kept)=(entries.len(),sources.len());
        if !self.loop_prefill.get() || !self.backend.resident_launches_active()
            || !prefill_gate::full_admitted(rows,dim,k,count,kept,prefill_gate::PG_FULL_THREADS)
            || !(eps>0.0 && eps.is_finite())
        {
            return Ok(None);
        }
        if sources.iter().any(|s|s.vector.is_segmented() || s.words!=rows*dim || s.vector.device_address()&15!=0) {
            return Ok(None);
        }
        let (wd_words,wo_words,v_words,kh_words,mw_words,mn_words)=(k*dim,dim*k,rows*dim,rows*(dim/2),rows*kept,rows);
        let out_words=if half_out {kh_words} else {v_words};
        let mut wd_views=Vec::with_capacity(count);
        let mut wo_views=Vec::with_capacity(count);
        for &(wd,ref wd_r,wo,ref wo_r,kv,kh,mw,mn,nw) in entries {
            if wd_r.len()!=wd_words || wo_r.len()!=wo_words
                || kv.words!=v_words || kh.words!=kh_words || mw.words!=mw_words || mn.words!=mn_words
                || wd.is_segmented() || wo.is_segmented() || kv.vector.is_segmented() || kh.vector.is_segmented()
                || mw.vector.is_segmented() || mn.vector.is_segmented()
                || nw.is_some_and(|n|n.len()!=dim)
            {
                return Ok(None);
            }
            let wdv=wd.resident_range(wd_r.clone())?;
            let wov=wo.resident_range(wo_r.clone())?;
            if (wdv.device_address()|wov.device_address()|kv.vector.device_address()|kh.vector.device_address())&15!=0 {
                return Ok(None);
            }
            wd_views.push(wdv);
            wo_views.push(wov);
        }
        for s in sources {self.ready(s)?;}
        for &(_,_,_,_,kv,kh,mw,mn,_) in entries {
            self.ready(kv)?;self.ready(kh)?;self.ready(mw)?;self.ready(mn)?;
        }
        let stamp=self.cost.start(self.backend);
        let mut nw_devs:Vec<Option<Rc<DeviceVector<'d>>>>=Vec::with_capacity(count);
        for &(_,_,_,_,_,_,_,_,nw) in entries {
            let d=match nw {
                Some(n)=>{
                    let dev=self.parameters.borrow_mut().resolve(n,None,|v|self.backend.upload_vector(v)?.into_resident_shared())?
                        .ok_or("prefill gate fuse: missing norm weight")?;
                    if dev.is_segmented() || dev.device_address()&15!=0 {return Ok(None);}
                    Some(dev)
                }
                None=>None,
            };
            nw_devs.push(d);
        }
        let src_refs:Vec<&DeviceVector<'d>>=sources.iter().map(|s|&*s.vector).collect();
        let gate_refs:Vec<_>=wd_views.iter().zip(&wo_views).zip(entries).zip(&nw_devs)
            .map(|(((wdv,wov),&(_,_,_,_,kv,kh,mw,mn,_)),nw)|(wdv,wov,&*kv.vector,&*kh.vector,&*mw.vector,&*mn.vector,nw.as_deref()))
            .collect();
        let outs=self.backend.prefill_gate_fuse(rows,dim,k,eps,half_out,&src_refs,&gate_refs)?;
        let values=outs.into_iter().map(|v|self.value(v,out_words)).collect::<Result<Vec<_>>>()?;
        self.cost.record(||"prefill_gate_fuse".into(),stamp,self.backend,0);
        Ok(Some(values))
    }
}
