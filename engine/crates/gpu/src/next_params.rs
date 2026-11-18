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
    /// The word function of this projection: `MX_SCALAR` (the lane chains and the shuffle tree of `next_decode::row`) or
    /// `MX_TC` (the tensor-core word of `docs/tc-dense-contract.md`, section MXFP8; kernels `mx_tc`, CPU twin
    /// `tc_twin_mx`). It sits in the four bytes that were padding, so no size or offset of the ABI moves. Every entry
    /// that takes a descriptor reads it; the block grid of a descriptor follows it (`mx_blocks`).
    pub mode: u32,
}
/// `MxProjection::mode`: today's scalar word.
pub const MX_SCALAR: u32 = 0;
/// `MxProjection::mode`: the tensor-core word. FP32 input, `round` 0.
pub const MX_TC: u32 = 1;
/// Weight rows of a block of the tensor-core row body (`mx_tc::ROWS_BLOCK`: 4 warps, one a range, 16 rows) and the tokens of
/// one of its token tiles.
pub const MX_TC_ROWS: u32 = 16;
pub const MX_TC_TOKENS: u32 = 8;
/// Blocks of 128 threads of one tensor-core descriptor: row tiles times token tiles, row tile fastest.
pub fn mx_tc_blocks(rows: u32, tokens: u32) -> Option<u32> { rows.div_ceil(MX_TC_ROWS).checked_mul(tokens.div_ceil(MX_TC_TOKENS)) }
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
            || p.mode > MX_TC
            || (p.mode == MX_TC && (p.input_bf16 != 0 || p.round != 0))
            || [p.weights, p.scales, p.input, p.output].contains(&0)
            || p.weights % 16 != 0
            || p.input % 16 != 0
            || p.output % 4 != 0
        {
            return Err("invalid MXFP8 descriptor");
        }
        let blocks = if p.mode == MX_TC {
            mx_tc_blocks(p.rows, p.tokens)
        } else {
            p.rows.div_ceil(4).checked_mul(p.tokens.div_ceil(p.tile()))
        }
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
        if projection.tokens!=1 || projection.round!=0 || projection.input_bf16!=0 || (plain && projection.mode!=MX_SCALAR) {return Err("resident projection batch requires one FP32 source row");}
        let index=self.layer.count as usize;let start=self.layer.ends[index];
        let blocks=if plain {projection.rows.div_ceil(128)} else if projection.mode==MX_TC {projection.rows.div_ceil(MX_TC_ROWS)} else {projection.rows.div_ceil(4)};
        let end=start.checked_add(blocks).ok_or("resident projection batch grid overflow")?;
        self.layer.push(projection)?;self.layer.ends[index+1]=end;
        if plain {self.plain|=1<<index;}Ok(())
    }
    /// ROW STAGE (`flash_resident_projection_bf16_rows`): a descriptor of 2 to 8 FP32 source rows (STAGE EIGHT: was 2 to 4). An MXFP8 descriptor keeps its
    /// one-row blocks (scalar word: a warp owns one weight row and all its source rows; tensor-core word: a block owns 16
    /// weight rows and all its source rows); a plain descriptor takes `tokens` times its one-row blocks, source row major (one
    /// thread a (weight row, source row)).
    pub fn push_rows(&mut self,projection:MxProjection,plain:bool)->Result<(),&'static str> {
        if !(2..=8).contains(&projection.tokens) || projection.round!=0 || projection.input_bf16!=0 || (plain && projection.mode!=MX_SCALAR) {return Err("resident projection row batch requires 2 to 8 FP32 source rows");}
        let index=self.layer.count as usize;if index>=9 {return Err("resident projection row batch is full");}let start=self.layer.ends[index];
        let blocks=if plain {projection.rows.div_ceil(128).checked_mul(projection.tokens).ok_or("resident projection row batch grid overflow")?}
            else if projection.mode==MX_TC {projection.rows.div_ceil(MX_TC_ROWS)} else {projection.rows.div_ceil(4)};
        let end=start.checked_add(blocks).ok_or("resident projection row batch grid overflow")?;
        self.layer.push(projection)?;self.layer.ends[index+1]=end;
        if plain {self.plain|=1<<index;}Ok(())
    }
    /// FOLD ROWS (`flash_resident_projection_fold_rows`, `_f32`): a plain descriptor of 2 to 8 FP32 source rows in the fold
    /// envelope. It keeps its one-row fold blocks (FOLD_ROWS weight rows a block); a block runs all its source rows.
    pub fn push_fold_rows(&mut self,projection:MxProjection)->Result<(),&'static str> {
        if !(2..=8).contains(&projection.tokens) || projection.round!=0 || projection.input_bf16!=0 || projection.mode!=MX_SCALAR {return Err("resident projection fold rows require 2 to 8 FP32 source rows");}
        let index=self.layer.count as usize;if index>=9 {return Err("resident projection fold row batch is full");}let start=self.layer.ends[index];
        let end=start.checked_add(projection.rows.div_ceil(4)).ok_or("resident projection fold row batch grid overflow")?;
        self.layer.push(projection)?;self.layer.ends[index+1]=end;self.plain|=1<<index;Ok(())
    }
}

#[cfg(test)]
mod resident_batch_tests {
    use super::*;
    fn projection(rows:u32)->MxProjection {MxProjection {weights:16,scales:4,input:16,output:4,rows,cols:32,tokens:1,input_bf16:0,round:0,mode:MX_SCALAR}}
    /// A tensor-core descriptor has 16 weight rows a block (and 8 token rows a token tile); a scalar one keeps its grid. The
    /// word function rides in the four bytes that were padding. A plain or fold job, BF16 input, a rounded boundary and an
    /// unknown mode are refused.
    #[test]
    fn tensor_core_descriptors_take_16_rows_a_block_and_refuse_other_inputs() {
        assert_eq!((std::mem::offset_of!(MxProjection,mode),std::mem::size_of::<MxProjection>()),(52,56));
        let tc=|rows:u32,tokens:u32|MxProjection {tokens,mode:MX_TC,..projection(rows)};
        let mut batch=ResidentProjectionBatch::default();
        for (p,plain) in [(tc(17,1),false),(projection(129),true),(tc(4096,1),false),(projection(33),false)] {batch.push(p,plain).unwrap();}
        assert_eq!(&batch.layer.ends[..5],&[0,2,4,260,269]);assert_eq!(batch.plain,2);
        // The kernel's mapping (`mx_tc::layer_block`): block `local` of a job owns the weight rows 16*local .. 16*local+16.
        for job in [0,2] {let p=batch.layer.projections[job];let mut visits=vec![0;p.rows as usize];
            for local in 0..batch.layer.ends[job+1]-batch.layer.ends[job] {for row in local*MX_TC_ROWS..(local+1)*MX_TC_ROWS {if row<p.rows {visits[row as usize]+=1;}}}
            assert!(visits.iter().all(|&count|count==1));}
        for tokens in 2..=8 {let mut rows=ResidentProjectionBatch::default();
            rows.push_rows(tc(4096,tokens),false).unwrap();rows.push_rows(MxProjection {tokens,..projection(4096)},false).unwrap();
            assert_eq!(&rows.layer.ends[..3],&[0,256,1280]);}
        assert!(ResidentProjectionBatch::default().push(tc(32,1),true).is_err());
        assert!(ResidentProjectionBatch::default().push_rows(tc(32,2),true).is_err());
        assert!(ResidentProjectionBatch::default().push_fold_rows(tc(32,2)).is_err());
        let mut layer=MxLayer::default();for p in [tc(4096,1),tc(100,9),tc(100,16),tc(100,8)] {layer.push(p).unwrap();}
        assert_eq!(&layer.ends[..5],&[0,256,270,284,291]);
        assert!(MxLayer::default().push(MxProjection {mode:2,..projection(4)}).is_err());
        assert!(MxLayer::default().push(MxProjection {input_bf16:1,..tc(4,1)}).is_err());
        assert!(MxLayer::default().push(MxProjection {round:1,..tc(4,1)}).is_err());
        assert_eq!(mx_tc_blocks(u32::MAX,u32::MAX),None);
    }
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
        for tokens in 2..=8u32 {
            let mut batch=ResidentProjectionBatch::default();
            for (rows,plain) in [(17,false),(129,true),(4096,false),(32,true)] {let mut p=projection(rows);p.tokens=tokens;batch.push_rows(p,plain).unwrap();}
            // MXFP8: one warp a weight row for all source rows. Plain: one thread a (weight row, source row), source row major.
            assert_eq!(&batch.layer.ends[..5],&[0,5,5+2*tokens,5+2*tokens+1024,5+2*tokens+1024+tokens]);assert_eq!(batch.plain,10);
            assert!(batch.layer.projections[..4].iter().all(|p|p.tokens==tokens));
            let (job,rest)=crate::plain_admission::batch_block(batch.layer.ends,batch.plain,4,2*tokens);assert_eq!((job,rest),(3,0));
        }
        for tokens in [0,1,9,16] {let mut batch=ResidentProjectionBatch::default();let mut p=projection(4);p.tokens=tokens;
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
