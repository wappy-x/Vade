//! DRAFT CONFIDENCE: the mailbox gather of the draft head's four-word picks (`logits_argmax_mass`).
use super::*;
use super::super::argmax_mass::merge_argmax_mass;
/// DRAFT CONFIDENCE: `gather_argmax` of the four-word parts of `logits_argmax_mass`: the pick and its probability.
pub fn gather_argmax_mass<'d>(b:&Backend<'d>,mailbox:&spark_gpu_probe::driver::TpMailbox<'d>,sequence:&mut u64,topology:Topology,local:&DeviceVector<'d>,owners:&[Rows])->Result<(u32,f32)> {
    let parts=topology.parts as usize;let total=parts.checked_mul(4).ok_or("resident argmax exchange overflow")?;
    if local.len()!=4 || owners.len()!=parts {return Err("resident argmax part extent differs".into());}
    let rows=Rows {start:topology.rank as usize*4,count:4,total,split:true};
    let gathered=gather_device(b,mailbox,sequence,topology,&[(local,false)],&[rows])?;
    merge_argmax_mass(&b.read_vector_values(&gathered[0])?,owners)
}
impl<'b,'d> Weights<'b,'d> {
    /// The rows of a head split by rows that each rank owns, in rank order (`apply_greedy`, `apply_greedy_rows`):
    /// refused unless the split is the even one, this rank's rows at rank x count. It selects no arithmetic path.
    pub(super) fn argmax_owners(&self,rows:Rows)->Result<Vec<Rows>> {
        let parts=self.topology.parts as usize;
        if rows.count.checked_mul(parts)!=Some(rows.total) || rows.start!=rows.count*self.topology.rank as usize {return Err("resident argmax ownership differs".into());}
        Ok((0..parts).map(|rank|Rows {start:rank*rows.count,..rows}).collect())
    }
    /// ONE PASS: `apply_greedy` with `mass` for `m` rows of `input` (row-major) at once. One row projection makes this
    /// rank's rows of the head for every row (`apply_matrix_local`: the words of each row's own projection). Each row is
    /// then picked as `apply_greedy` picks it alone: its own `logits_argmax_mass` launches on its own words, and its own
    /// four-word parts merged with the other rank's (`merge_argmax_mass`). Only the exchange is shared: ONE gather carries
    /// every row's parts. None, before any work: a head that `apply_greedy` gives no probability (not split by rows, no
    /// mailbox), or one row; the caller then picks each row alone.
    pub fn apply_greedy_rows(&mut self,name:&str,b:&Backend<'d>,input:&DeviceVector<'_>,m:usize)->Result<Option<Vec<(u32,f32)>>> {
        let Some(matrix)=self.matrices.get(name) else {return Ok(None);};let rows=matrix.rows;
        let Some(mailbox)=self.mailbox else {return Ok(None);};
        if m<2 || !rows.split || rows.count>1<<24 || input.len()!=m*matrix.cols {return Ok(None);}
        let owners=self.argmax_owners(rows)?;let parts=owners.len();
        let Some(local)=self.apply_matrix_local(name,b,input,m,true)? else {return Ok(None);};
        let local=local.into_resident_shared()?;
        let words=4;let mut partials=b.vector(rows.count.div_ceil(1024)*words)?;let mut result=b.vector(m*words)?;
        // Each row's launches read only its own words; the partials are reused in stream order.
        for row in 0..m {let logits=local.resident_range(row*rows.count..(row+1)*rows.count)?;
            b.enqueue_argmax_device(&logits,&mut partials,&mut result,row|spark_gpu_probe::driver::ARGMAX_MASS)?;}
        let stamp=self.cost.start(b);
        let all=Rows {start:self.topology.rank as usize*m*words,count:m*words,total:parts*m*words,split:true};
        let gathered=gather_device(b,mailbox,&mut self.sequence,self.topology,&[(&result,false)],&[all])?;
        let values=b.read_vector_values(&gathered[0])?;self.cost.record(||"exchange_argmax".into(),stamp,b,0);
        if values.len()!=parts*m*words {return Err("resident argmax rows extent differs".into());}
        // Row `row` of part `part` is at part*m*4 + row*4: the four words its own pick sends alone.
        (0..m).map(|row|{let picks=(0..parts).flat_map(|part|values[(part*m+row)*words..(part*m+row+1)*words].iter().copied()).collect::<Vec<_>>();
            merge_argmax_mass(&picks,&owners)}).collect::<Result<Vec<_>>>().map(Some)
    }
}
