//! Native benchmark ABI encoders. No production caller selects these plans.
use super::*;
#[derive(Clone,Copy)]
pub struct RefereeShape {pub format:u32,pub rows:u32,pub cols:u32,pub block_rows:u32,pub block_cols:u32,pub global:f32,pub activation:f32}
struct Call {entry:&'static str,blocks:u32,rows:u32,threads:u32,words:[u32;32]}
pub struct RefereePlan<'s,'a> {calls:Vec<Call>,tables:Vec<DeviceVector<'a>>,owners:Vec<&'s DeviceVector<'a>>,output:DeviceVector<'a>}
impl<'a> RefereePlan<'_,'a> {pub fn output(&self)->&DeviceVector<'a> {&self.output}}
fn address(words:&mut Vec<u32>,value:u64) {words.extend([value as u32,(value>>32) as u32]);}
fn payload(words:&[u32])->Result<[u32;32],String> {let mut output=[0;32];if words.len()>32 {return Err("native referee ABI exceeds payload".into());}output[..words.len()].copy_from_slice(words);Ok(output)}
impl<'a> Backend<'a> {
    pub fn fixture_referee_plan<'s>(&self,engineer:&str,new_order:bool,shape:RefereeShape,jobs:&[(&'s DeviceVector<'a>,&'s DeviceVector<'a>,Option<&'s DeviceVector<'a>>)],calibrations:&[(f32,f32)]) ->Result<RefereePlan<'s,'a>,String> {
        self.require_resident_stream()?;
        let RefereeShape {format,rows,cols,block_rows,block_cols,global,activation}=shape;
        if !matches!(engineer,"opus"|"sonnet"|"fable"|"sol"|"astra") || format>2 || rows==0 || cols==0 || rows>1<<20 || cols>1<<20 || jobs.is_empty() || jobs.len()>8 || !global.is_finite() || global<=0.0 || !activation.is_finite() || activation<=0.0 {return Err("unsupported native referee plan".into());}
        if format==0 && (block_rows==0 || block_cols==0) || format==2 && cols%16!=0 {return Err("invalid native scale blocks".into());}
        if engineer=="sonnet" && format==2 && !new_order {return Err("sonnet has no same-bytes NVFP4 kernel".into());}
        if calibrations.len()!=jobs.len() || calibrations.iter().any(|&(global,activation)|!global.is_finite() || global<=0.0 || !activation.is_finite() || activation<=0.0) {return Err("invalid per-job calibration".into());}
        let count=jobs.len() as u32;let elements=rows as usize*cols as usize;
        let input_words=if format==2 {cols as usize/16*17} else {cols as usize};
        let weight_words=match format {0=>elements.div_ceil(4),1=>elements,_=>(elements/2).div_ceil(4)};
        let scale_words=match format {0=>rows.div_ceil(block_rows) as usize*cols.div_ceil(block_cols) as usize,1=>0,_=>(elements/16).div_ceil(4)};
        for &(input,weight,scales) in jobs {
            for (vector,length) in [(input,input_words),(weight,weight_words)] {self.check_vector(vector,length)?;if vector.is_segmented() {return Err("native referee operands must be contiguous".into());}}
            match (scales,scale_words) {(Some(vector),length) if length>0=>{self.check_vector(vector,length)?;if vector.is_segmented() {return Err("native referee scale must be contiguous".into());}},(None,0)=>{},_=>return Err("native referee scale extent differs".into())}
        }
        let mut output=self.vector(rows as usize*jobs.len())?;self.zero_vector(&mut output)?;
        let owners=jobs.iter().flat_map(|&(input,weight,scales)|[Some(input),Some(weight),scales].into_iter().flatten()).collect();
        let mut plan=RefereePlan {calls:Vec::new(),tables:Vec::new(),owners,output};let mut entries=Vec::new();
        for (job,&(input,weight,scales)) in jobs.iter().enumerate() {
            let (global,activation)=calibrations[job];
            let out=plan.output.device_address()+job as u64*rows as u64*4;let scale=scales.map_or(4,DeviceVector::device_address);let mut words=Vec::new();
            if matches!(engineer,"opus"|"fable") && !(engineer=="fable" && format==1) || engineer=="sol" && format==2 {
                for value in [input.device_address(),4,4,4,weight.device_address(),scale,out] {address(&mut words,value);}
                let op=match (engineer,format,new_order) {("opus",0,false)=>8,("opus",0,true)=>10,("opus",1,false)=>9,("opus",1,true)=>11,("opus",2,false)=>12,("opus",2,true)=>13,("fable",0,false)=>9,("fable",0,true)=>8,("fable",2,false)=>11,("fable",2,true)=>10,_=>1};
                words.extend([op,rows,cols,if format==1 {0} else {block_rows},block_cols,1,global.to_bits(),activation.to_bits(),0,0]);
                if engineer=="sol" {words.extend([if new_order {0} else {2},0]);}
            } else if engineer=="fable" {
                for value in [input.device_address(),4,4,4,weight.device_address(),out] {address(&mut words,value);}
                words.extend([if new_order {30} else {31},0,0,0,rows,cols,0,0]);
            } else {
                for value in [input.device_address(),weight.device_address(),scale,out] {address(&mut words,value);}
                match engineer {
                    "sonnet"=>words.extend([rows,cols,block_rows,block_cols,(format!=1) as u32,count,(global*activation).to_bits(),0]),
                    "sol"=>words.extend([rows,cols,block_rows,block_cols,format,if new_order {3} else {2}]),
                    "astra"=>words.extend([rows,cols,1,block_rows,block_cols,format,if new_order {0} else {1},global.to_bits(),activation.to_bits(),0]),
                    _=>return Err("native referee encoder is missing".into()),
                }
            }
            entries.push(words);
        }
        let single_calls=engineer=="astra" || engineer=="fable" && format==1;
        if single_calls {
            for words in entries {let entry=if engineer=="fable" {"causal_operation"} else {match (format,new_order) {(0,false)=>"projection_fp8_serial",(0,true)=>"projection_fp8_striped",(1,false)=>"projection_f32_serial",(1,true)=>"projection_f32_striped",(2,false)=>"projection_nvfp4_serial",_=>"projection_nvfp4_striped"}};
                plan.calls.push(Call {entry,blocks:if engineer=="fable" {rows} else {rows.div_ceil(4)},rows:1,threads:if engineer=="fable" {32} else {128},words:payload(&words)?});}
        } else {
            let table=self.upload_vector(&entries.iter().flatten().copied().map(f32::from_bits).collect::<Vec<_>>())?;
            let (entry,blocks,grid_rows,threads,words)=if engineer=="sonnet" {
                let entry=match (format,new_order) {(0,false)=>"bake_fp8_chain",(0,true)=>"bake_fp8_strip",(1,false)=>"bake_plain_chain",(1,true)=>"bake_plain_strip",_=>"bake_nv_strip"};
                let mut words=Vec::new();address(&mut words,table.device_address());(entry,rows.div_ceil(if new_order {4} else {32}),count,if new_order {128} else {32},words)
            } else {
                let mut words=entries[0].clone();address(&mut words,table.device_address());words.extend([count,0]);
                let (entry,per_warp,threads)=match engineer {"opus"=>("expert_operation",if new_order {1} else if format==1 {8} else {16},32),"fable"=>{let group=if new_order {1} else if rows*count>=8192 {8} else if rows*count>=4096 {4} else if rows*count>=2048 {2} else {1};("expert_operation",group,32)},_=>{(if format==2 {"nv_projection_warp"} else if new_order {"projection_pairs"} else {"projection_warp"},4,128)}};
                (entry,rows.div_ceil(per_warp)*count,1,threads,words)
            };
            plan.calls.push(Call {entry,blocks,rows:grid_rows,threads,words:payload(&words)?});plan.tables.push(table);
        }
        for call in &plan.calls {self.require_arm_kernels(&[call.entry])?;}self.synchronize()?;Ok(plan)
    }
    pub fn fixture_referee_launch(&self,plan:&RefereePlan<'_,'a>)->Result<(),String> {
        self.check_vector(&plan.output,plan.output.len())?;
        for owner in &plan.owners {self.check_vector(owner,owner.len())?;}
        let result=(|| {for call in &plan.calls {let mut words=call.words;self.launch_checked_expert_payload(call.entry,call.blocks,call.rows,call.threads,0,&mut words)?;}Ok(())})();
        let completed=self.synchronize();result.and(completed)
    }
}
