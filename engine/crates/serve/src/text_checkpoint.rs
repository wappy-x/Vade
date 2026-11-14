//! Bind admitted checkpoint files to the resident text edges without a host mirror.
//! This does not bind a decoder or qualify the candidate norm and projection.
use engine_model::{plan::{admission::CheckpointAdmission,glm_index::TensorRole,text_edges::TextEdges},safetensors::{Dtype,TensorFile}};
use spark_gpu_probe::text_edges::{Layout,NormStorage,Shape};
use std::{collections::BTreeMap,fs::File,ops::Range};

struct CopyRange {file_offset:u64,destination:usize,bytes:usize}
/// A complete original tensor span. No row packing or dtype conversion occurs.
pub struct StoredCopy<'a> {pub file:&'a File,pub file_offset:u64,pub destination:usize,pub bytes:usize}

pub struct CheckpointTextEdges<'c> {
    checkpoint: &'c CheckpointAdmission, edges: TextEdges, layout: Layout, bytes: usize,
    files: [&'c TensorFile;4], copies: [CopyRange;4],
}
impl<'c> CheckpointTextEdges<'c> {
    /// Use the description and retained files from the same complete admission.
    /// The arena stores each original byte once. All edges are replicated.
    pub fn new(checkpoint: &'c CheckpointAdmission) -> Result<Self,String> {
        let mut headers = BTreeMap::new();
        for weight in checkpoint.headers().weights().values() {
            if matches!(weight.binding().operation,TensorRole::Embedding|TensorRole::FinalNorm|TensorRole::Head) {
                for component in weight.components() {
                    let file = checkpoint.component_file(&component.name).map_err(|e|e.to_string())?;
                    let header = file.header().tensors.get(&component.name).ok_or("missing text component header")?;
                    headers.insert(component.name.clone(),header.clone());
                }
            }
        }
        let edges = checkpoint.description().text_edges(&headers).map_err(|e|e.to_string())?;
        let shape = Shape::new(edges.hidden,edges.vocab,edges.streams)?;
        let norm_storage = match edges.norm.tensor.dtype {Dtype::Bf16=>NormStorage::Bf16,Dtype::F32=>NormStorage::F32,
            _=>return Err("unsupported stored text norm dtype".into())};
        let tensors = [&edges.embedding,&edges.head,&edges.head_scales,&edges.norm];
        let mut bytes = 0usize;
        let mut files = Vec::new(); let mut copies = Vec::new();
        for (tensor,alignment) in tensors.into_iter().zip([2,16,1,norm_storage.bytes()]) {
            let file = checkpoint.component_file(&tensor.name).map_err(|e|e.to_string())?;
            let count = usize::try_from(tensor.tensor.byte_len()).map_err(|_|"text component exceeds address space")?;
            let destination = bytes.checked_add(alignment-1).ok_or("text arena alignment overflow")? / alignment * alignment;
            bytes = destination.checked_add(count).ok_or("text arena size overflow")?;
            let file_offset = file.header().data_start.checked_add(tensor.tensor.start).ok_or("text file offset overflow")?;
            files.push(file);
            copies.push(CopyRange {file_offset,destination,bytes:count});
        }
        let files = files.try_into().map_err(|_|"text component count differs")?;
        let copies:[CopyRange;4] = copies.try_into().map_err(|_|"text component count differs")?;
        let layout = Layout {shape,embedding:copies[0].destination,head:copies[1].destination,scales:copies[2].destination,
            norm:copies[3].destination,norm_storage,epsilon:edges.epsilon as f32};
        layout.ranges(bytes)?;
        Ok(Self {checkpoint,edges,layout,bytes,files,copies})
    }
    pub fn checkpoint(&self) -> &'c CheckpointAdmission {self.checkpoint}
    pub fn edges(&self) -> &TextEdges {&self.edges}
    pub fn layout(&self) -> Layout {self.layout}
    pub fn arena_bytes(&self) -> usize {self.bytes}
    /// Match the admitted MXFP8 uploader's reserved-scale refusal. Use bounded
    /// staging and retain the original scale bytes for the device copy.
    pub fn validate_head_scales(&self) -> Result<(),String> {
        let bytes = self.copies[2].bytes;
        let mut buffer = vec![0;bytes.min(1024*1024)]; let mut offset = 0;
        while offset < bytes {
            let count = buffer.len().min(bytes-offset);
            self.files[2].read_range(&self.edges.head_scales.name,offset as u64,&mut buffer[..count]).map_err(|e|e.to_string())?;
            if buffer[..count].contains(&255) {return Err("invalid MXFP8 scale byte".into());}
            offset += count;
        }
        Ok(())
    }
    /// These descriptors use original file offsets and complete scale planes.
    /// No path is reopened, row is packed, or dtype is converted.
    pub fn stored_copies(&self) -> [StoredCopy<'_>;4] {
        std::array::from_fn(|i| StoredCopy {file:self.files[i].file(),file_offset:self.copies[i].file_offset,
            destination:self.copies[i].destination,bytes:self.copies[i].bytes})
    }
    /// Read an arena interval for host byte checks. Gaps are zero padding.
    /// A failed read clears the caller's output. It is never a partial success.
    pub fn read_range(&self,offset:usize,output:&mut [u8]) -> Result<(),String> {
        output.fill(0);
        let end = offset.checked_add(output.len()).filter(|&end|end<=self.bytes).ok_or("text read exceeds arena")?;
        let names = [&self.edges.embedding.name,&self.edges.head.name,&self.edges.head_scales.name,&self.edges.norm.name];
        for (i,copy) in self.copies.iter().enumerate() {
            let selected = Range {start:offset.max(copy.destination),end:end.min(copy.destination+copy.bytes)};
            if selected.start < selected.end {
                let result = self.files[i].read_range(names[i],(selected.start-copy.destination) as u64,
                    &mut output[selected.start-offset..selected.end-offset]);
                if let Err(error) = result {output.fill(0); return Err(error.to_string());}
            }
        }
        Ok(())
    }
    /// Load through the existing bounded file uploader. Resident views retain
    /// the arena after this method returns. No new launch boundary is added.
    #[cfg(target_os = "linux")]
    pub fn upload<'b,'a>(&self,backend:&'b spark_gpu_probe::driver::Backend<'a>)
        -> Result<spark_gpu_probe::driver::ResidentTextEdges<'b,'a>,String> {
        use spark_gpu_probe::load::{WeightFile,WeightRange};
        self.validate_head_scales()?;
        let ranges:[WeightRange;4] = std::array::from_fn(|i| WeightRange {file_offset:self.copies[i].file_offset,
            destination:self.copies[i].destination,bytes:self.copies[i].bytes,row_bytes:0,row_stride:0});
        let files:[WeightFile<'_>;4] = std::array::from_fn(|i| WeightFile {file:self.files[i].file(),ranges:std::slice::from_ref(&ranges[i])});
        let arena = backend.load_weight_arena(self.bytes,&files)?;
        backend.resident_text_edges(&arena,self.layout)
    }
}
