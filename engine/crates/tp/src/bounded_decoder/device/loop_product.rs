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
//! Many input rows (`resident_product_rows`), result `[inputs][rows]`: each word is the word of `resident_product`
//! on its input row. A large matrix of whole stages takes the rows tile, an entry of its own
//! (`Backend::project_staged_rows_into`), from 8 input rows (`staged_product::rows_tile_part`; a matrix above the
//! rows of the tile, the head, runs in parts that write their columns in place) while launches are
//! recorded. For the other cases two forms with the same words, picked by cost (`staged_product::rows_one_launch`):
//! one launch of the general rows kernel for all input rows (`prompt_ops::Op::ProjectRowsBf16`, then
//! `Op::Transpose`), or one launch of the row product for each input row into one output, with no join.
use super::*;

impl<'b,'d> Device<'b,'d> {
    /// One launch: `rows` chains of one BF16 matrix with one input row. The staged product when it runs now, else the
    /// chain of `project_bf16_ordered`. Both give the same words.
    fn row_product(&self,rows:usize,columns:usize,input:&DeviceVector<'d>,matrix:&DeviceVector<'d>,output:&mut DeviceVector<'d>)->Result<()> {
        if self.backend.staged_product_ready(rows,columns,input,matrix) {self.backend.project_staged_into(rows,columns,input,matrix,output)?;}
        else {self.backend.project_bf16_chain_into(rows,columns,input,matrix,output)?;}
        Ok(())
    }
    pub(crate) fn resident_product(&self,weight:&DeviceVector<'d>,words:std::ops::Range<usize>,rows:usize,columns:usize,x:&Value<'b,'d>)->Result<Value<'b,'d>> {
        need(rows>0 && columns>0 && rows.checked_mul(columns)==words.len().checked_mul(2),"resident product: the weight words differ from rows x columns of BF16 values")?;
        need(x.words==columns,"resident product: the input differs from the columns of the weight")?;
        self.ready(x)?;
        let part=spark_gpu_probe::staged_product::part_rows(rows,columns);
        if part>=rows {
            let matrix=weight.resident_range(words)?;
            let mut output=self.backend.vector(rows)?;
            self.row_product(rows,columns,&x.vector,&matrix,&mut output)?;
            return self.value(output,rows);
        }
        // Two BF16 values a word: a row is columns / 2 words (the columns are whole stages here).
        let (whole,row_words)=(self.backend.vector(rows)?.into_resident_shared()?,columns/2);
        let mut row=0;
        while row<rows {
            let count=part.min(rows-row);
            let matrix=weight.resident_range(words.start+row*row_words..words.start+(row+count)*row_words)?;
            let mut output=whole.resident_range(row..row+count)?;
            self.row_product(count,columns,&x.vector,&matrix,&mut output)?;
            row+=count;
        }
        self.shared_value(Rc::new(whole),rows)
    }
    /// `resident_product` for `inputs` rows of `x` (`[inputs][columns]`): the result is `[inputs][rows]`, row-major.
    pub(crate) fn resident_product_rows(&self,weight:&DeviceVector<'d>,words:std::ops::Range<usize>,rows:usize,columns:usize,x:&Value<'b,'d>,inputs:usize)->Result<Value<'b,'d>> {
        if inputs==1 {return self.resident_product(weight,words,rows,columns,x);}
        need(rows>0 && columns>0 && rows.checked_mul(columns)==words.len().checked_mul(2),"resident product: the weight words differ from rows x columns of BF16 values")?;
        need(inputs>0 && inputs.checked_mul(columns)==Some(x.words),"resident product: the input differs from the input rows x the columns of the weight")?;
        self.ready(x)?;
        let total=inputs.checked_mul(rows).ok_or("resident product: the output of the input rows is too large")?;
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
                self.row_product(count,columns,&source,&matrix,&mut output)?;
                row+=count;
            }
        }
        self.shared_value(Rc::new(whole),total)
    }
}
