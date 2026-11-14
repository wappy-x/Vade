//! Bounded checkpoint reads for the scalar layer tool. No device conversion.
use engine_model::{Result,safetensors::{self,Dtype,Index,Limits,TensorFile},
    plan::{layer::reference::{Tensor,WeightSource},storage::{WeightStorage,ScaleEncoding}},flash::{nvfp4,mxfp8},csf::{Csf32,Quantization}};
use std::{cell::RefCell,collections::BTreeMap,path::{Path,PathBuf}};
use super::invalid;

pub struct Checkpoint {
    root: PathBuf, index: Index, files: RefCell<BTreeMap<String,TensorFile>>,
    storage: engine_model::plan::storage::StorageMap,
    pub receipts: RefCell<BTreeMap<String,String>>,
    pub bands: RefCell<BTreeMap<String,String>>,
    pub index_sha256: String,
}
impl Checkpoint {
    pub fn open(root: &Path, storage: engine_model::plan::storage::StorageMap) -> Result<Self> {
        let bytes=safetensors::read_bounded(&root.join("model.safetensors.index.json"),64<<20)?;
        let limits=Limits {header_bytes:64<<20,tensors:1_000_000,..Limits::default()};
        Ok(Self { root:root.canonicalize()?,index:Index::parse_mixed(&bytes,limits)?,files:RefCell::new(BTreeMap::new()),
            storage,receipts:RefCell::new(BTreeMap::new()),bands:RefCell::new(BTreeMap::new()),index_sha256:engine_hash::digest(&bytes) })
    }
    fn with_file<T>(&self,name: &str,f: impl FnOnce(&TensorFile)->Result<T>) -> Result<T> {
        let shard=self.index.weight_map.get(name).ok_or_else(|| invalid(format!("missing indexed tensor {name}")))?;
        let mut files=self.files.borrow_mut();
        if !files.contains_key(shard) {
            let path=self.root.join(shard).canonicalize()?;
            if !path.starts_with(&self.root) { return Err(invalid("shard is outside checkpoint")); }
            files.insert(shard.clone(),TensorFile::open(&path,Limits::default())?);
        }
        f(&files[shard])
    }
    pub fn raw(&self,name: &str) -> Result<(safetensors::Tensor,Vec<u8>)> {
        self.with_file(name,|file| {
            let t=file.header().tensors.get(name).ok_or_else(|| invalid("index and header differ"))?.clone();
            if t.byte_len()>1<<30 { return Err(invalid("tensor exceeds reference read cap")); }
            let mut bytes=vec![0;t.byte_len() as usize]; file.read_range(name,0,&mut bytes)?;
            let hash=engine_hash::digest(&bytes);
            let row=format!("{name}\t{:?}\t{:?}\t{}\t{}\t{}\t{}",t.dtype,t.shape,t.start,t.end,self.index.weight_map[name],hash);
            if self.receipts.borrow().get(name).is_some_and(|old| old!=&row) { return Err(invalid("tensor changed between reads")); }
            self.receipts.borrow_mut().insert(name.into(),row);
            Ok((t,bytes))
        })
    }
    pub fn embedding(&self,id: u32) -> Result<Vec<f32>> {
        let name="model.language_model.embed_tokens.weight";
        self.with_file(name,|file| {
            let t=file.header().tensors.get(name).ok_or_else(|| invalid("index and embedding header differ"))?;
            if t.dtype!=Dtype::Bf16 || t.shape.len()!=2 || id as u64>=t.shape[0] { return Err(invalid("invalid embedding row")); }
            let size=t.shape[1]*2; let mut bytes=vec![0;size as usize]; file.read_range(name,id as u64*size,&mut bytes)?;
            let key=format!("{name}[{id}]"); let row=format!("{key}\tBF16\t[{}]\t{}\t{}\t{}\t{}",
                t.shape[1],t.start+id as u64*size,t.start+(id as u64+1)*size,self.index.weight_map[name],engine_hash::digest(&bytes));
            if self.receipts.borrow().get(&key).is_some_and(|old| old!=&row) { return Err(invalid("embedding row changed between reads")); }
            self.receipts.borrow_mut().insert(key,row);
            decode_plain(Dtype::Bf16,&bytes)
        })
    }
}
fn decode_plain(dtype: Dtype,bytes: &[u8]) -> Result<Vec<f32>> {
    let values=match dtype {
        Dtype::Bf16 => bytes.as_chunks::<2>().0.iter().map(|v| f32::from_bits((u16::from_le_bytes(*v) as u32)<<16)).collect(),
        Dtype::F32 => bytes.as_chunks::<4>().0.iter().map(|v| f32::from_le_bytes(*v)).collect(),
        _ => return Err(invalid("reference requires BF16 or FP32 plain values")),
    }; Ok(values)
}
impl WeightSource for Checkpoint {
    fn profile(&self) -> engine_model::plan::layer::profile::Profile {
        engine_model::plan::layer::profile::Profile::LogicalF32V1
    }
    fn row(&self,name: &str,index: u32) -> Result<Tensor> {
        if name!="model.language_model.embed_tokens.weight" { return Err(invalid("checkpoint row reader requires the embedding role")); }
        let values=self.embedding(index)?; Tensor::f32(&[values.len() as u64],values)
    }
    fn shape(&self,name: &str) -> Result<Vec<u64>> {
        self.with_file(name,|f| {
            let mut shape=f.header().tensors.get(name).ok_or_else(|| invalid("missing weight header"))?.shape.clone();
            let module=name.strip_suffix(".weight").unwrap_or(name);
            if matches!(self.storage.storage(module),WeightStorage::Nvfp4 {..}) { *shape.last_mut().ok_or_else(|| invalid("empty packed shape"))?*=2; }
            // Conv storage has a singleton group dimension. The logical ABI omits it.
            if name.ends_with("_conv1d.weight") && shape.len()==3 && shape[1]==1 { shape.remove(1); }
            Ok(shape)
        })
    }
    fn load(&self,name: &str) -> Result<Tensor> {
        let (header,bytes)=self.raw(name)?; let shape=self.shape(name)?;
        let count=shape.iter().try_fold(1usize,|n,&d| n.checked_mul(d as usize).ok_or_else(|| invalid("weight size overflow")))?;
        let module=name.strip_suffix(".weight").unwrap_or(name);
        let values=match self.storage.storage(module) {
            WeightStorage::Plain => decode_plain(header.dtype,&bytes)?,
            WeightStorage::Mxfp8 => {
                if header.dtype!=Dtype::F8E4M3 { return Err(invalid("invalid MXFP8 weight dtype")); }
                let (sh,scales)=self.raw(&format!("{module}.weight_scale"))?;
                if sh.dtype!=Dtype::U8 || shape.len()!=2 || sh.shape!=[shape[0],shape[1]/32] { return Err(invalid("invalid MXFP8 scale plane")); }
                let mut out=vec![0.0;count]; mxfp8::dequantize(&bytes,&scales,&mut out)?; out
            },
            WeightStorage::Fp8Block {block} => {
                if shape.len()!=2 || header.dtype!=Dtype::F8E4M3 { return Err(invalid("invalid block FP8 weight")); }
                let (sh,scales)=self.raw(&format!("{module}.weight_scale"))?;
                let scale=decode_plain(sh.dtype,&scales)?; let (rows,cols)=(shape[0] as usize,shape[1] as usize);
                if scale.iter().any(|v| !v.is_finite() || *v<0.0) { return Err(invalid("invalid block FP8 scale value")); }
                let (br,bc)=(block[0] as usize,block[1] as usize);
                let grid=[rows.div_ceil(br) as u64,cols.div_ceil(bc) as u64];
                if sh.shape!=grid && sh.shape!=[grid[0],1,grid[1],1] { return Err(invalid(format!("block scale shape differs: {module} weight={shape:?} scale={:?} block={block:?}",sh.shape))); }
                bytes.iter().enumerate().map(|(j,&v)| nvfp4::e4m3(v)*scale[(j/cols/br)*cols.div_ceil(bc)+(j%cols/bc)]).collect()
            },
            WeightStorage::Nvfp4 {encoding,activation_scale} => {
                if shape.len()!=2 || header.dtype!=Dtype::U8 || shape[1]%16!=0 { return Err(invalid("invalid packed FP4 weight")); }
                let (gh,global)=self.raw(&format!("{module}.weight_scale_2"))?;
                let global=decode_plain(gh.dtype,&global)?;
                if global.len()!=1 { return Err(invalid("invalid global scale extent")); }
                if activation_scale {
                    let (ih,input)=self.raw(&format!("{module}.input_scale"))?;
                    let input=decode_plain(ih.dtype,&input)?;
                    if input.len()!=1 || !input[0].is_finite() || input[0]<=0.0 { return Err(invalid("invalid activation calibration scalar")); }
                    // Retain its identity. The logical FP32 profile has no activation quantizer.
                }
                let scale_name=format!("{module}.weight_scale");
                let scales=match encoding {
                    ScaleEncoding::Csf32 => {
                        let load=|suffix: &str| -> Result<Vec<u8>> { Ok(self.raw(&format!("{scale_name}.{suffix}"))?.1) };
                        let (bases,codes,pages,offsets,exceptions)=(load("csf32_bases")?,load("csf32_codes")?,load("csf32_page_bases")?,load("csf32_offsets")?,load("csf32_exceptions")?);
                        Csf32 {rows:shape[0] as usize,columns:shape[1] as usize/16,quantization:Quantization::Nvfp4,
                            bases:&bases,codes:&codes,page_bases:&pages,offsets:&offsets,exceptions:&exceptions}.expand()?
                    },
                    ScaleEncoding::Direct => {
                        let (sh,scales)=self.raw(&scale_name)?;
                        if sh.dtype!=Dtype::F8E4M3 || sh.shape!=[shape[0],shape[1]/16] { return Err(invalid("invalid direct FP4 scale plane")); }
                        scales
                    },
                    ScaleEncoding::Csf => return Err(invalid("legacy CSF is outside this reference tool")),
                };
                let mut out=vec![0.0;count]; nvfp4::dequantize(&bytes,&scales,global[0],&mut out)?;
                let mut max_error=0f64; let mut max_relative=0f64; let mut max_ulp=0f64;
                for (j,&value) in out.iter().enumerate() {
                    let packed=bytes[j/2]; let nibble=if j%2==0 {packed&15} else {packed>>4};
                    let exact=nvfp4::e2m1(nibble) as f64*nvfp4::e4m3(scales[j/16]) as f64*global[0] as f64;
                    let error=(value as f64-exact).abs(); max_error=max_error.max(error);
                    if exact!=0.0 { max_relative=max_relative.max(error/exact.abs()); }
                    let magnitude=value.abs(); let ulp=f32::from_bits(magnitude.to_bits()+1) as f64-magnitude as f64;
                    if ulp.is_finite() && ulp>0.0 { max_ulp=max_ulp.max(error/ulp); }
                }
                self.bands.borrow_mut().insert(name.into(),format!("{name}\t{count}\t{max_error:.17e}\t{max_relative:.17e}\t{max_ulp:.17e}")); out
            },
        };
        Tensor::f32(&shape,values)
    }
}

#[cfg(test)] mod tests {
    use super::*;
    use std::{fs,sync::atomic::{AtomicU64,Ordering}};
    static NEXT: AtomicU64=AtomicU64::new(0);
    fn fixture(entries: Vec<(&str,&str,Vec<u64>,Vec<u8>)>,storage: WeightStorage) -> Checkpoint {
        let root=std::env::temp_dir().join(format!("vade-reference-weights-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        fs::create_dir(&root).unwrap(); let mut data=Vec::new(); let mut header=Vec::new(); let mut names=Vec::new();
        for (name,dtype,shape,bytes) in entries {
            let start=data.len(); data.extend(bytes);
            header.push(format!("\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}",data.len()));
            names.push(format!("\"{name}\":\"weights.safetensors\""));
        }
        let header=format!("{{{}}}",header.join(",")); let mut file=(header.len() as u64).to_le_bytes().to_vec(); file.extend(header.as_bytes()); file.extend(&data);
        fs::write(root.join("weights.safetensors"),file).unwrap();
        fs::write(root.join("model.safetensors.index.json"),format!("{{\"metadata\":{{\"total_size\":{}}},\"weight_map\":{{{}}}}}",data.len(),names.join(","))).unwrap();
        Checkpoint::open(&root,engine_model::plan::storage::StorageMap {excluded:Vec::new(),layers:BTreeMap::from([("linear".into(),storage)])}).unwrap()
    }
    #[test] fn plain_bf16_widening_preserves_bits_and_conv_shape() {
        let bits=[0x0000u16,0x8000,0x3f80,0xc000]; let bytes=bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let cp=fixture(vec![("linear.q_conv1d.weight","BF16",vec![1,1,4],bytes)],WeightStorage::Plain);
        let t=cp.load("linear.q_conv1d.weight").unwrap(); assert_eq!(t.shape,[1,4]);
        assert_eq!(t.floats().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),bits.iter().map(|&v| (v as u32)<<16).collect::<Vec<_>>());
    }
    #[test] fn mxfp8_scales_use_unsigned_exponents_and_original_rows() {
        let cp=fixture(vec![("linear.weight","F8_E4M3",vec![2,32],vec![0x38;64]),
            ("linear.weight_scale","U8",vec![2,1],vec![127,128])],WeightStorage::Mxfp8);
        let t=cp.load("linear.weight").unwrap(); assert_eq!(&t.floats().unwrap()[..32],&[1.0;32]); assert_eq!(&t.floats().unwrap()[32..],&[2.0;32]);
    }
    #[test] fn block_fp8_accepts_explicit_singleton_scale_axes() {
        let scale=[1f32,2.0,4.0,8.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let cp=fixture(vec![("linear.weight","F8_E4M3",vec![4,4],vec![0x38;16]),
            ("linear.weight_scale","F32",vec![2,1,2,1],scale)],WeightStorage::Fp8Block {block:[2,2]});
        assert_eq!(cp.load("linear.weight").unwrap().floats().unwrap(),[1.,1.,2.,2.,1.,1.,2.,2.,4.,4.,8.,8.,4.,4.,8.,8.]);
    }
    #[test] fn direct_fp4_has_one_final_round_and_keeps_calibration_identity() {
        let global=1.0000001f32;
        let cp=fixture(vec![("linear.weight","U8",vec![1,8],vec![0x77;8]),
            ("linear.weight_scale","F8_E4M3",vec![1,1],vec![0x3c]),
            ("linear.weight_scale_2","F32",vec![],global.to_le_bytes().to_vec()),
            ("linear.input_scale","F32",vec![],2f32.to_le_bytes().to_vec())],WeightStorage::Nvfp4 {encoding:ScaleEncoding::Direct,activation_scale:true});
        let t=cp.load("linear.weight").unwrap();
        assert!(t.floats().unwrap().iter().all(|v| v.to_bits()==(9.0*global).to_bits()));
        assert!(cp.receipts.borrow().contains_key("linear.input_scale")); assert!(cp.bands.borrow().contains_key("linear.weight"));
    }
    #[test] fn direct_fp4_rejects_wrong_scale_axes_dtype_and_incomplete_blocks() {
        for (dtype,shape) in [("U8",vec![2,1]),("F8_E4M3",vec![1,2])] {
            let cp=fixture(vec![("linear.weight","U8",vec![2,8],vec![0x22;16]),
                ("linear.weight_scale",dtype,shape,vec![0x38;2]),("linear.weight_scale_2","F32",vec![],1f32.to_le_bytes().to_vec())],
                WeightStorage::Nvfp4 {encoding:ScaleEncoding::Direct,activation_scale:false});
            assert_eq!(cp.load("linear.weight").unwrap_err().to_string(),"invalid direct FP4 scale plane");
        }
        for encoding in [ScaleEncoding::Direct,ScaleEncoding::Csf32] {
            let cp=fixture(vec![("linear.weight","U8",vec![1,9],vec![0x22;9])],WeightStorage::Nvfp4 {encoding,activation_scale:false});
            assert_eq!(cp.load("linear.weight").unwrap_err().to_string(),"invalid packed FP4 weight");
        }
    }
    #[test] fn csf32_checkpoint_decode_matches_known_scale_bytes() {
        let cp=fixture(vec![("linear.weight","U8",vec![32,8],vec![0x22;256]),
            ("linear.weight_scale_2","F32",vec![],2f32.to_le_bytes().to_vec()),
            ("linear.weight_scale.csf32_bases","U8",vec![32],vec![0x38;32]),
            ("linear.weight_scale.csf32_codes","U8",vec![16],vec![0;16]),
            ("linear.weight_scale.csf32_page_bases","U32",vec![1],vec![0;4]),
            ("linear.weight_scale.csf32_offsets","U16",vec![2],vec![0;4]),
            ("linear.weight_scale.csf32_exceptions","U16",vec![0],Vec::new())],
            WeightStorage::Nvfp4 {encoding:ScaleEncoding::Csf32,activation_scale:false});
        assert_eq!(cp.load("linear.weight").unwrap().floats().unwrap(),vec![2.0;512]);
    }
    #[test] fn nonconv_axes_stay_intact_and_missing_embedding_header_returns_error() {
        let mut cp=fixture(vec![("linear.weight","F32",vec![1,1,1],1f32.to_le_bytes().to_vec())],WeightStorage::Plain);
        assert_eq!(cp.shape("linear.weight").unwrap(),[1,1,1]);
        cp.index.weight_map.insert("model.language_model.embed_tokens.weight".into(),"weights.safetensors".into());
        assert_eq!(cp.embedding(0).unwrap_err().to_string(),"index and embedding header differ");
    }
}
