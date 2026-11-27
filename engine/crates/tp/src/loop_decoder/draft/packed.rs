//! In-memory draft matrices in the packed unit layout used by expert projections.
use super::{ResidentProduct, Result, Store};
use engine_model::{flash::nvfp4::{bf16_to_f32,e2m1,e4m3},plan::{loop_load::ExpertCodes,loop_packed_unit::UnitShape}};
use spark_gpu_probe::{packed_product,tc_rows};

pub(super) enum Packed<V> { Panel{unit:UnitShape,weight:V}, Units(Vec<(UnitShape,V)>) }

pub(super) fn panel(rows:usize,columns:usize,format:Option<ExpertCodes>)->Option<UnitShape> {
    if rows<64 || format!=Some(ExpertCodes::Nvfp4) {return None;}
    spark_gpu_probe::draft_ops::panel_words(rows,columns)?;
    Some(UnitShape{rows,columns,codes:ExpertCodes::Nvfp4,order:None})
}

pub(super) fn upload_panel<S:Store>(store:&mut S,unit:UnitShape,bits:&[u16],bias:&[f32])->Result<S::Value> {
    if unit.rows.checked_mul(unit.columns)!=Some(bits.len()) || (!bias.is_empty() && bias.len()!=unit.rows) {return Err("draft panel shape".into());}
    let bytes=panel_bytes(unit,bits,band(unit.rows))?;
    let words=bytes.len()/4;
    let mut weight=store.create(words+bias.len())?;
    store.write(&mut weight,0,&bytes)?;
    if !bias.is_empty() {store.write(&mut weight,words,&bias.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;}
    Ok(weight)
}

fn nearest(value:f32, table:&[f32])->u8 {
    let right=table.partition_point(|&v|v<value);
    if right==0 {return 0;}
    if right==table.len() {return (right-1) as u8;}
    let lo=value-table[right-1];let hi=table[right]-value;
    if lo<hi || (lo==hi && (right-1)%2==0) {(right-1) as u8} else {right as u8}
}

/// Rows of one packing band: two bands a host thread, in whole 64-row blocks.
fn band(rows:usize)->usize {
    let threads=std::thread::available_parallelism().map_or(1,|n|n.get());
    rows.next_multiple_of(64).div_ceil(64).div_ceil(threads*2).max(1)*64
}

/// Sibling panels of one input stored back to back in one allocation, each with the bytes and factor
/// of its own panel: one `DraftOp::Panels` launch writes all of their outputs.
pub(super) struct Siblings<V> {weight:V,rows:[usize;3],columns:usize}
/// Two or three NVFP4 panels of `columns` columns, each of whole 64-row blocks; otherwise none.
pub(super) fn siblings<S:Store>(store:&mut S,parts:&[(&[u16],usize)],columns:usize,format:Option<ExpertCodes>)->Result<Option<Siblings<S::Value>>> {
    if !(2..=3).contains(&parts.len()) || parts.iter().any(|&(bits,rows)|!rows.is_multiple_of(64) || panel(rows,columns,format).is_none() || rows.checked_mul(columns)!=Some(bits.len())) {return Ok(None);}
    let mut bytes=Vec::new();
    let mut rows=[0;3];
    for (slot,&(bits,count)) in rows.iter_mut().zip(parts) {
        bytes.extend(panel_bytes(UnitShape{rows:count,columns,codes:ExpertCodes::Nvfp4,order:None},bits,band(count))?);
        *slot=count;
    }
    let mut weight=store.create(bytes.len()/4)?;
    store.write(&mut weight,0,&bytes)?;
    Ok(Some(Siblings{weight,rows,columns}))
}
impl<V> Siblings<V> {
    /// The rows of each panel.
    pub(super) fn rows(&self)->[usize;3] {self.rows}
    /// The input width of each panel.
    pub(super) fn columns(&self)->usize {self.columns}
    pub(super) fn project<C:ResidentProduct<V>+super::Lanes>(&self,c:&C,x:&C::Value,n:usize)->Result<C::Value> {
        use spark_gpu_probe::draft_ops::DraftOp;
        let input=c.lanes(DraftOp::PackActivations(c.words(x)),x,None)?;
        c.draft_panel(DraftOp::Panels{rows:self.rows,columns:self.columns,inputs:n},&self.weight,&input,None)
    }
}

fn largest(bits:&[u16])->f32 {bits.iter().map(|&b|bf16_to_f32(b).abs()).fold(0.0f32,f32::max)}
fn factor_of(largest:f32)->f32 {(largest/(448.0*6.0)).max(f32::MIN_POSITIVE)}
fn panel_factor(bits:&[u16])->f32 {factor_of(largest(bits))}

/// The bytes of `pack_draft(encode(unit padded to 64 rows))`, made from bands of `band` stored rows
/// (a multiple of 64) on separate threads. A band runs the same encoder and lane packer with the
/// factor of the whole matrix. Codes and scales of 64-row blocks are contiguous in the panel, and the
/// table holds only that factor, so joining the bands in row order gives the same bytes (host test).
fn panel_bytes(unit:UnitShape,bits:&[u16],band:usize)->Result<Vec<u8>> {
    use spark_gpu_probe::decode_panels::pack_draft;
    let (rows,columns)=(unit.rows,unit.columns);
    let padded=rows.next_multiple_of(64);
    if band==0 || !band.is_multiple_of(64) || rows.checked_mul(columns)!=Some(bits.len()) {return Err("draft panel band".into());}
    // f32::max ignores order on these words; a word that is not finite fails the encoder first.
    let span=(band*columns).max(1);
    let maxima=std::thread::scope(|scope| {
        let parts=bits.chunks(span).map(|part|scope.spawn(move||largest(part))).collect::<Vec<_>>();
        parts.into_iter().map(|part|part.join().map_err(|_|"draft panel thread failed".to_string())).collect::<Result<Vec<_>>>()
    })?;
    let factor=factor_of(maxima.into_iter().fold(0.0f32,f32::max));
    let bands=std::thread::scope(|scope| {
        let parts=(0..padded).step_by(band).map(|start| {
            let count=band.min(padded-start);
            let real=rows.saturating_sub(start).min(count);
            let source=&bits[start*columns..(start+real)*columns];
            scope.spawn(move||-> Result<Vec<u8>> {
                let shape=UnitShape{rows:count,columns,..unit};
                Ok(pack_draft(&encode_with(shape,source,factor)?,count,columns,factor))
            })
        }).collect::<Vec<_>>();
        parts.into_iter().map(|part|part.join().map_err(|_|"draft panel thread failed".to_string())?).collect::<Result<Vec<_>>>()
    })?;
    let total=spark_gpu_probe::draft_ops::panel_words(rows,columns).ok_or("draft panel size")?*4;
    let mut out=Vec::with_capacity(total);
    let codes=|count:usize|count*columns/2;
    for (part,start) in bands.iter().zip((0..padded).step_by(band)) {out.extend_from_slice(&part[..codes(band.min(padded-start))]);}
    for (part,start) in bands.iter().zip((0..padded).step_by(band)) {
        let count=band.min(padded-start);
        out.extend_from_slice(&part[codes(count)..codes(count)+count*columns/16]);
    }
    let first=band.min(padded);
    out.extend_from_slice(&bands[0][codes(first)+first*columns/16..]);
    if out.len()!=total {return Err("draft panel bands differ from the panel".into());}
    Ok(out)
}

fn encode(shape:UnitShape,bits:&[u16])->Result<Vec<u8>> {encode_with(shape,bits,panel_factor(bits))}
fn encode_with(shape:UnitShape,bits:&[u16],panel:f32)->Result<Vec<u8>> {
    if bits.iter().any(|&b|!bf16_to_f32(b).is_finite()) {return Err("nonfinite draft projection weight".into());}
    let wide=shape.codes==ExpertCodes::Mxfp8;
    let fp8:Vec<_>=(0..127).map(e4m3).collect();
    let fp4:Vec<_>=(0..8).map(e2m1).collect();
    let table=if wide {&fp8} else {&fp4};
    let max=|bits:&[u16]|bits.iter().map(|&b|bf16_to_f32(b).abs()).fold(0.0f32,f32::max);
    let factor=if wide {1.0} else {panel};
    let mut codes=Vec::with_capacity(shape.rows*shape.code_stride());
    let mut scales=Vec::with_capacity(shape.rows*shape.scale_stride());
    for group in bits.chunks_exact(shape.block()) {
        let target=max(group)/table.last().unwrap();
        let (scale,byte)=if wide {
            let exp=if target==0.0 {0} else {target.log2().ceil().clamp(-126.0,118.0) as i32};
            (2.0f32.powi(exp),(exp+127) as u8)
        } else {
            let nearest=nearest(target/factor,&fp8).max(u8::from(target>0.0));
            (fp8[nearest as usize]*factor,nearest)
        };
        scales.push(byte);
        let code=|b:u16| {
            let value=bf16_to_f32(b);
            nearest(if scale>0.0 {value.abs()/scale} else {0.0},table)
                | if value.is_sign_negative() {if wide {128} else {8}} else {0}
        };
        if wide {codes.extend(group.iter().map(|&b|code(b)));}
        else {codes.extend(group.as_chunks::<2>().0.iter().map(|p|code(p[0])|(code(p[1])<<4)));}
    }
    codes.resize(shape.rows*shape.code_stride(),0);
    scales.resize(shape.rows*shape.scale_stride(),0);
    let mut out=vec![0;shape.layout().map_err(|e|e.to_string())?.bytes];
    shape.fill(&codes,&scales,(!wide).then_some(factor),&mut out).map_err(|e|e.to_string())?;
    Ok(out)
}

/// The same row chunks drive admission and upload. Unsupported column counts use BF16.
pub(super) fn units(rows:usize,columns:usize,format:ExpertCodes)->Option<impl Iterator<Item=UnitShape>> {
    let wide=format==ExpertCodes::Mxfp8;
    if rows==0 || !tc_rows::tc_packed_moe_shape_admitted(1,columns,wide,false) {return None;}
    let per=(packed_product::ELEMENTS_MAX/columns).min(tc_rows::TC_ROWS_MAX)/32*32;
    Some((0..rows).step_by(per).map(move |start|UnitShape {rows:per.min(rows-start),columns,codes:format,order:None}))
}
pub(super) fn storage(rows:usize,columns:usize,format:Option<ExpertCodes>)->Result<Option<usize>> {
    if panel(rows,columns,format).is_some() {return Ok(spark_gpu_probe::draft_ops::panel_words(rows,columns).and_then(|w|w.checked_mul(4)));}
    format.and_then(|f|units(rows,columns,f)).map(|mut parts|parts.try_fold(0usize,|sum,part| {
        sum.checked_add(part.layout().map_err(|e|e.to_string())?.bytes).ok_or_else(||"packed draft size overflow".into())
    })).transpose()
}
impl<V> Packed<V> {
    pub(super) fn new<S:Store<Value=V>>(store:&mut S,bits:&[u16],rows:usize,columns:usize,format:ExpertCodes)->Result<Option<Self>> {
        if let Some(unit)=panel(rows,columns,Some(format)) {
            return Ok(Some(Self::Panel{unit,weight:upload_panel(store,unit,bits,&[])?}));
        }
        let Some(units)=units(rows,columns,format) else {return Ok(None);};
        if rows.checked_mul(columns)!=Some(bits.len()) {return Err("draft packed matrix shape".into());}
        let mut parts=Vec::new();
        let mut rest=bits;
        for unit in units {
            let (source,next)=rest.split_at(unit.rows*columns);rest=next;
            let bytes=encode(unit,source)?;
            let mut value=store.create(bytes.len()/4)?;store.write(&mut value,0,&bytes)?;parts.push((unit,value));
        }
        Ok(Some(Self::Units(parts)))
    }
    pub(super) fn dot<C:ResidentProduct<V>+super::Lanes>(&self,c:&C,x:&C::Value,n:usize)->Result<C::Value> {
        let (limit,columns)=match self {Self::Panel{unit,..}=>(spark_gpu_probe::draft_ops::PANEL_INPUTS_MAX,unit.columns),Self::Units(parts)=>(tc_rows::MOE_ROWS_MAX,parts[0].0.columns)};
        if n>limit {
            let mut batches=Vec::new();
            for start in (0..n).step_by(limit) {
                let count=(n-start).min(limit);
                batches.push(self.dot(c,&c.range(x,start*columns,count*columns)?,count)?);
            }
            return c.join(&batches.iter().collect::<Vec<_>>());
        }
        let parts=match self {
            Self::Panel{unit,weight}=>{
                let input=c.lanes(spark_gpu_probe::draft_ops::DraftOp::PackActivations(c.words(x)),x,None)?;
                return c.draft_panel(spark_gpu_probe::draft_ops::DraftOp::Panel{rows:unit.rows,columns,inputs:n},weight,&input,None);
            }
            Self::Units(parts)=>parts,
        };
        // Every row uses the same matrix. Expert zero gives the packed tensor-core kernel its row table.
        let route=if n>1 {
            let ids=c.zeros(2*n)?;
            Some(c.device_route_group(&ids,n,1,1)?.ok_or("draft packed route is unavailable")?)
        } else {None};
        let mut values=Vec::new();
        for (unit,weight) in parts {
            let out=if let Some((_,plan))=&route {c.device_routed_packed_products(weight,1,*unit,plan,x,false)?}
                else {c.packed_product(weight,0..unit.layout().map_err(|e|e.to_string())?.bytes/4,*unit,x)?};
            values.push(out.ok_or("draft packed projection is unavailable")?);
        }
        if values.len()==1 {return Ok(values.pop().unwrap());}
        let mut rows=Vec::new();
        for row in 0..n {
            for ((unit,_),out) in parts.iter().zip(&values) {rows.push(c.range(out,row*unit.rows,unit.rows)?);}
        }
        c.join(&rows.iter().collect::<Vec<_>>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_model::flash::nvfp4::f32_to_bf16;

    #[test]
    fn packed_draft_units_preserve_zero_and_both_signs() {
        for format in [ExpertCodes::Mxfp8,ExpertCodes::Nvfp4] {
            let shape=UnitShape {codes:format,rows:3,columns:128,order:None};
            let mut bits=vec![0u16;384];
            bits[128]=f32_to_bf16(-1.0);bits[256]=f32_to_bf16(1.0);
            let packed=encode(shape,&bits).unwrap();
            assert_eq!(Some(packed.len()),storage(3,128,Some(format)).unwrap());
            let mut rows=[0.0;3];shape.product(&packed,&[1.0;128],&mut rows).unwrap();
            assert_eq!(rows,[0.0,-1.0,1.0]);
            bits[0]=f32_to_bf16(f32::NAN);
            assert!(encode(shape,&bits).is_err());
        }
    }

    #[test]
    fn banded_panels_match_the_serial_panel() {
        let mut state=0x9e3779b97f4a7c15u64;
        for (rows,columns) in [(64,64),(130,128),(64*5+7,256),(1000,192)] {
            let bits=(0..rows*columns).map(|_| {
                state=state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                f32_to_bf16(((state>>40) as f32/16_777_216.0-0.5)*if state&1==0 {3.0} else {0.01})
            }).collect::<Vec<_>>();
            let unit=UnitShape{rows,columns,codes:ExpertCodes::Nvfp4,order:None};
            let padded=UnitShape{rows:rows.next_multiple_of(64),..unit};
            let serial=spark_gpu_probe::decode_panels::pack_draft(&encode(padded,&bits).unwrap(),padded.rows,columns,panel_factor(&bits));
            for band in [64,128,192,padded.rows] {assert_eq!(panel_bytes(unit,&bits,band).unwrap(),serial,"{rows}x{columns} band {band}");}
        }
        let mut bad=vec![0u16;64*64];bad[4000]=f32_to_bf16(f32::INFINITY);
        assert!(panel_bytes(UnitShape{rows:64,columns:64,codes:ExpertCodes::Nvfp4,order:None},&bad,64).is_err());
    }

    #[test]
    fn large_vocabularies_account_for_each_tile_and_table() {
        for format in [ExpertCodes::Mxfp8,ExpertCodes::Nvfp4] {
            let units=units(tc_rows::TC_ROWS_MAX+1,128,format).unwrap().collect::<Vec<_>>();
            assert_eq!(units.len(),2);
            assert_eq!(units.last().unwrap().rows,1);
            let bytes=if format==ExpertCodes::Nvfp4 {spark_gpu_probe::draft_ops::panel_words(tc_rows::TC_ROWS_MAX+1,128).unwrap()*4}
                else {units.iter().map(|u|u.layout().unwrap().bytes).sum()};
            assert_eq!(storage(tc_rows::TC_ROWS_MAX+1,128,Some(format)).unwrap(),Some(bytes));
            assert_eq!(storage(12,66,Some(format)).unwrap(),None);
        }
    }
}
