//! Native MXFP8 decode ABI. Row arithmetic is independent of launch geometry.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MxProjection {
    pub weights: u64,
    pub scales: u64,
    pub input: u64,
    pub output: u64,
    pub rows: u32,
    pub cols: u32,
    pub tokens: u32,
    /// Input is packed BF16 when 1, otherwise FP32 (rounded only for the GDN contract).
    pub input_bf16: u32,
    /// GDN projection boundaries round to BF16 represented in FP32.
    pub round: u32,
}
impl MxProjection {
    /// Decode a loaded FP32 operand. Only rounded GDN projections may discard
    /// its residual; ordinary projections consume the same FP32 input as PLAIN.
    #[inline(always)]
    pub fn f32_input(&self, bits: u32) -> f32 {
        let value = f32::from_bits(bits);
        if self.round != 0 {
            #[cfg(target_arch = "nvptx64")]
            {
                return crate::flash_gdn::rounded(value);
            }
            #[cfg(not(target_arch = "nvptx64"))]
            {
                return crate::bf16::rounded(value);
            }
        }
        value
    }
    pub fn kernel(&self) -> &'static str {
        if self.tokens == 1 {
            "flash_next_mxfp8_m1"
        } else if self.tokens <= 8 || self.round != 0 {
            "flash_next_mxfp8_pair"
        } else {
            "flash_next_mxfp8_bits"
        }
    }
    pub fn tile(&self) -> u32 {
        if self.round == 0 && self.tokens > 1 && self.tokens <= 8 {
            8 // port-mxfp8-gemm retained SIMT schedule, independent of N/K
        } else if self.tokens == 1 || self.rows < 128 {
            1
        } else if self.rows <= 640 || self.cols >= 4096 {
            if self.tokens <= 2 {
                2
            } else {
                4
            }
        } else {
            self.tokens
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MxLayer {
    pub projections: [MxProjection; 9],
    pub ends: [u32; 10],
    pub count: u32,
}
impl MxLayer {
    pub fn push(&mut self, p: MxProjection) -> Result<(), &'static str> {
        if self.count == 9
            || p.rows == 0
            || p.cols == 0
            || p.cols % 32 != 0
            || p.tokens == 0
            || p.tokens > 16
            || p.input_bf16 > 1
            || p.round > 1
            || [p.weights, p.scales, p.input, p.output].contains(&0)
            || p.weights % 16 != 0
            || p.input % 16 != 0
            || p.output % 4 != 0
        {
            return Err("invalid MXFP8 descriptor");
        }
        let blocks = p
            .rows
            .div_ceil(4)
            .checked_mul(p.tokens.div_ceil(p.tile()))
            .ok_or("MXFP8 grid overflow")?;
        let i = self.count as usize;
        self.ends[i + 1] = self.ends[i]
            .checked_add(blocks)
            .ok_or("MXFP8 grid overflow")?;
        self.projections[i] = p;
        self.count += 1;
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone,Copy,Default)]
pub struct ResidentProjectionBatch {pub layer:MxLayer,pub plain:u32,pub round:u32}
impl ResidentProjectionBatch {
    pub fn push(&mut self,projection:MxProjection,plain:bool)->Result<(),&'static str> {
        if projection.tokens!=1 || projection.round!=0 || projection.input_bf16!=0 {return Err("resident projection batch requires one FP32 source row");}
        let index=self.layer.count as usize;let start=self.layer.ends[index];
        let blocks=if plain {projection.rows.div_ceil(128)} else {projection.rows.div_ceil(4)};
        let end=start.checked_add(blocks).ok_or("resident projection batch grid overflow")?;
        self.layer.push(projection)?;self.layer.ends[index+1]=end;
        if plain {self.plain|=1<<index;}Ok(())
    }
    /// ROW STAGE (`flash_resident_projection_bf16_rows`): a descriptor of 2 to 4 FP32 source rows. An MXFP8 descriptor keeps its
    /// one-row blocks (a warp owns one weight row and all its source rows); a plain descriptor takes `tokens` times its one-row
    /// blocks, source row major (one thread a (weight row, source row)).
    pub fn push_rows(&mut self,projection:MxProjection,plain:bool)->Result<(),&'static str> {
        if !(2..=4).contains(&projection.tokens) || projection.round!=0 || projection.input_bf16!=0 {return Err("resident projection row batch requires 2 to 4 FP32 source rows");}
        let index=self.layer.count as usize;if index>=9 {return Err("resident projection row batch is full");}let start=self.layer.ends[index];
        let blocks=if plain {projection.rows.div_ceil(128).checked_mul(projection.tokens).ok_or("resident projection row batch grid overflow")?} else {projection.rows.div_ceil(4)};
        let end=start.checked_add(blocks).ok_or("resident projection row batch grid overflow")?;
        self.layer.push(projection)?;self.layer.ends[index+1]=end;
        if plain {self.plain|=1<<index;}Ok(())
    }
}

#[cfg(test)]
mod resident_batch_tests {
    use super::*;
    fn projection(rows:u32)->MxProjection {MxProjection {weights:16,scales:4,input:16,output:4,rows,cols:32,tokens:1,input_bf16:0,round:0}}
    #[test]
    fn mixed_rows_have_original_complete_warp_and_plain_grids() {
        use std::mem::{size_of,align_of,offset_of};
        assert_eq!((size_of::<MxProjection>(),size_of::<MxLayer>(),size_of::<ResidentProjectionBatch>(),align_of::<ResidentProjectionBatch>()),(56,552,560,8));
        assert_eq!((offset_of!(ResidentProjectionBatch,layer),offset_of!(ResidentProjectionBatch,plain),offset_of!(ResidentProjectionBatch,round)),(0,552,556));
        let mut batch=ResidentProjectionBatch::default();
        for (rows,plain) in [(17,false),(129,true),(1,false),(32,true),(33,false)] {batch.push(projection(rows),plain).unwrap();}
        assert_eq!(batch.layer.count,5);assert_eq!(&batch.layer.ends[..6],&[0,5,7,8,9,18]);assert_eq!(batch.plain,10);
        for job in 0..5 {let p=batch.layer.projections[job];let plain=batch.plain&(1<<job)!=0;let rows=if plain {128} else {4};
            let mut visits=vec![0;p.rows as usize];
            for block in batch.layer.ends[job]..batch.layer.ends[job+1] {for lane in 0..rows {let row=(block-batch.layer.ends[job])*rows+lane;if row<p.rows {visits[row as usize]+=1;}}}
            assert!(visits.iter().all(|&count|count==1));
        }
    }
    #[test]
    fn row_batches_give_plain_rows_one_block_group_a_source_row_and_refuse_other_row_counts() {
        for tokens in 2..=4u32 {
            let mut batch=ResidentProjectionBatch::default();
            for (rows,plain) in [(17,false),(129,true),(4096,false),(32,true)] {let mut p=projection(rows);p.tokens=tokens;batch.push_rows(p,plain).unwrap();}
            // MXFP8: one warp a weight row for all source rows. Plain: one thread a (weight row, source row), source row major.
            assert_eq!(&batch.layer.ends[..5],&[0,5,5+2*tokens,5+2*tokens+1024,5+2*tokens+1024+tokens]);assert_eq!(batch.plain,10);
            assert!(batch.layer.projections[..4].iter().all(|p|p.tokens==tokens));
            let (job,rest)=crate::plain_admission::batch_block(batch.layer.ends,batch.plain,4,2*tokens);assert_eq!((job,rest),(3,0));
        }
        for tokens in [0,1,5,16] {let mut batch=ResidentProjectionBatch::default();let mut p=projection(4);p.tokens=tokens;
            assert!(batch.push_rows(p,false).is_err());assert_eq!((batch.layer.count,batch.plain),(0,0));}
        let mut batch=ResidentProjectionBatch::default();let mut p=projection(4);p.tokens=2;p.cols=48;assert!(batch.push_rows(p,true).is_err());assert_eq!(batch.layer.count,0);
        let mut p=projection(4);p.tokens=2;for _ in 0..9 {batch.push_rows(p,false).unwrap();}assert!(batch.push_rows(p,true).is_err());assert_eq!((batch.layer.count,batch.plain),(9,0));
    }
    #[test]
    fn invalid_and_overfull_batches_refuse_before_mutation() {
        let mut batch=ResidentProjectionBatch::default();let mut invalid=projection(4);invalid.tokens=2;assert!(batch.push(invalid,false).is_err());assert_eq!(batch.layer.count,0);
        invalid=projection(4);invalid.cols=31;assert!(batch.push(invalid,true).is_err());assert_eq!(batch.layer.count,0);
        for _ in 0..9 {batch.push(projection(4),false).unwrap();}
        assert!(batch.push(projection(4),false).is_err());assert_eq!(batch.layer.count,9);
    }
}
