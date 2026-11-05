//! Owned two-launch post/pre boundaries on one ordered stream.
use super::*;
use crate::{mhc_params::{Shape,Boundary},norm_boundary::{NormShape,WeightType}};

struct Parameters<'a> {weights:DeviceVector<'a>,base:DeviceVector<'a>,scale:DeviceVector<'a>,norm:DeviceVector<'a>}
pub struct ResidentMhcBoundary<'b,'a> {backend:&'b Backend<'a>,shape:Shape,norm:NormShape,parameters:Rc<Parameters<'a>>}
/// All inputs, parameter planes and prior passes survive queued execution.
pub struct MhcBoundaryPass<'b,'a> {
    backend:&'b Backend<'a>,shape:Shape,parameters:Rc<Parameters<'a>>,
    source:Rc<DeviceVector<'a>>,residual:Rc<DeviceVector<'a>>,mixing:DeviceVector<'a>,
    coefficients:DeviceVector<'a>,collapsed:DeviceVector<'a>,partials:DeviceVector<'a>,output:Rc<DeviceVector<'a>>,
    previous:Option<Box<MhcBoundaryPass<'b,'a>>>,
}
impl<'a> Backend<'a> {
    pub fn resident_mhc_boundary(&self,shape:Shape,norm:NormShape,weights:&[f32],base:&[f32],scale:&[f32],norm_weight:&[f32]) -> Result<ResidentMhcBoundary<'_,'a>,String> {
        self.mhc_ready()?;shape.validate()?;norm.check_weights(norm_weight)?;
        if norm.width()!=shape.hidden as usize || weights.len()!=shape.coefficients()*shape.wide()
            || base.len()!=shape.coefficients() || scale.len()!=3 || weights.iter().chain(base).chain(scale).any(|value| !value.is_finite()) {
            return Err("mHC boundary parameter geometry or value differs".into());
        }
        self.require_arm_kernels(&["mhc_boundary_project","mhc_boundary_finish","mhc_bf16_combine"])?;
        let parameters=Rc::new(Parameters {weights:self.upload_vector(weights)?,base:self.upload_vector(base)?,
            scale:self.upload_vector(scale)?,norm:self.upload_vector(norm_weight)?});self.synchronize()?;
        Ok(ResidentMhcBoundary {backend:self,shape,norm,parameters})
    }
}
impl<'b,'a> ResidentMhcBoundary<'b,'a> {
    pub fn shape(&self)->Shape {self.shape}
    fn with_rows(&self,rows:usize)->Result<Self,String> {
        let shape=Shape {rows:u32::try_from(rows).map_err(|_|"mHC boundary row count exceeds u32")?,..self.shape};shape.validate()?;
        Ok(Self {backend:self.backend,shape,norm:self.norm,parameters:self.parameters.clone()})
    }
    pub fn begin_rows(&self,input:Rc<DeviceVector<'a>>,rows:usize)->Result<MhcBoundaryPass<'b,'a>,String> {self.with_rows(rows)?.begin(input)}
    pub fn begin_after_rows(&self,previous:MhcBoundaryPass<'b,'a>,block:Rc<DeviceVector<'a>>,residual:Rc<DeviceVector<'a>>,rows:usize)->Result<MhcBoundaryPass<'b,'a>,String> {
        self.with_rows(rows)?.begin_after(previous,block,residual)
    }
    pub fn begin(&self,input:Rc<DeviceVector<'a>>)->Result<MhcBoundaryPass<'b,'a>,String> {
        self.backend.mhc_ready()?;self.backend.check_vector(&input,self.shape.rows as usize*self.shape.wide())?;
        let residual=Rc::new(self.backend.vector(input.len())?);self.submit(input,residual,None)
    }
    pub fn begin_after(&self,previous:MhcBoundaryPass<'b,'a>,block:Rc<DeviceVector<'a>>,residual:Rc<DeviceVector<'a>>)->Result<MhcBoundaryPass<'b,'a>,String> {
        self.backend.mhc_ready()?;let shape=self.shape;
        if !ptr::eq(self.backend,previous.backend) || (shape.rows,shape.streams,shape.hidden)!=(previous.shape.rows,previous.shape.streams,previous.shape.hidden) {
            return Err("deferred mHC boundary context or shape differs".into());
        }
        self.backend.check_vector(&block,shape.rows as usize*shape.hidden as usize)?;
        self.backend.check_vector(&residual,shape.rows as usize*shape.wide())?;
        self.submit(block,residual,Some(Box::new(previous)))
    }
    fn submit(&self,source:Rc<DeviceVector<'a>>,residual:Rc<DeviceVector<'a>>,previous:Option<Box<MhcBoundaryPass<'b,'a>>>)->Result<MhcBoundaryPass<'b,'a>,String> {
        let backend=self.backend;let shape=self.shape;let rows=shape.rows as usize;let wide=shape.wide();let count=shape.coefficients();let hidden=shape.hidden as usize;
        let pass=MhcBoundaryPass {backend,shape,parameters:self.parameters.clone(),source,residual,mixing:backend.vector(rows*wide)?,
            coefficients:backend.vector(rows*count)?,collapsed:backend.vector(rows*hidden)?,partials:backend.vector(rows*hidden.div_ceil(128))?,
            output:Rc::new(backend.vector(rows*hidden)?.into_resident_shared()?),previous};
        let mut vectors=vec![&*pass.source,&pass.parameters.weights,&pass.parameters.base,&pass.parameters.scale,&*pass.residual,&pass.mixing,
            &pass.coefficients,&pass.collapsed,&pass.parameters.norm,&pass.partials,&*pass.output];
        let mut lengths=vec![rows*if pass.previous.is_some() {hidden} else {wide},count*wide,count,3,rows*wide,rows*wide,rows*count,rows*hidden,hidden,rows*hidden.div_ceil(128),rows*hidden];
        if let Some(previous)=&pass.previous {vectors.extend([&*previous.residual,&previous.coefficients]);lengths.extend([rows*wide,rows*count]);}
        backend.mhc_planes(&vectors,&lengths,4)?;
        let mut args=Boundary {input:pass.source.device_address(),previous_input:pass.previous.as_ref().map_or(0,|previous|previous.residual.device_address()),
            previous_coefficients:pass.previous.as_ref().map_or(0,|previous|previous.coefficients.device_address()),residual:pass.residual.device_address(),
            weights:pass.parameters.weights.device_address(),base:pass.parameters.base.device_address(),scale:pass.parameters.scale.device_address(),
            mixing_normalized:pass.mixing.device_address(),coefficients:pass.coefficients.device_address(),collapsed:pass.collapsed.device_address(),
            norm_weight:pass.parameters.norm.device_address(),norm_partials:pass.partials.device_address(),output:pass.output.device_address(),shape,
            norm_epsilon:self.norm.epsilon(),norm_bf16:u32::from(self.norm.weight_type()==WeightType::Bf16)};
        backend.mhc_launch("mhc_boundary_project",shape,&mut args)?;
        backend.mhc_launch("mhc_boundary_finish",shape,&mut args)?;
        Ok(pass)
    }
}
impl<'a> MhcBoundaryPass<'_,'a> {
    pub fn previous(&self)->Option<&Self> {self.previous.as_deref()}
    pub fn shape(&self)->Shape {self.shape}
    pub fn residual(&self)->&DeviceVector<'a> {&self.residual}
    pub fn mixing_normalized(&self)->&DeviceVector<'a> {&self.mixing}
    pub fn coefficients(&self)->&DeviceVector<'a> {&self.coefficients}
    pub fn collapsed(&self)->&DeviceVector<'a> {&self.collapsed}
    pub fn normalized(&self)->Rc<DeviceVector<'a>> {self.output.clone()}
    /// A real residual consumer ends deferral. Preserve the original block cast
    /// and combine arithmetic, including completion before owners are dropped.
    pub fn materialize(self,block:&DeviceVector<'_>,output:&DeviceVector<'_>)->Result<(),String> {
        let backend=self.backend;let shape=self.shape;backend.mhc_ready()?;
        backend.check_vector(block,shape.rows as usize*shape.hidden as usize)?;
        backend.check_vector(output,shape.rows as usize*shape.wide())?;
        let block=backend.round_bf16_lanes(block)?;
        backend.mhc_combine_owned(shape,[&self.residual,&self.coefficients,block.vector(),output])
    }
}
