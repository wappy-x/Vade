//! Default-off fixture image admission. CUDA loading uses the existing driver adapter.
#![forbid(unsafe_code)]
use std::ffi::CString;
pub enum Image {Ptx(CString),Cubin}
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct Geometry {kernel:String,grid_scale:u32,threads:u32}
impl Geometry {
    pub fn kernel(&self)->&str {&self.kernel}
    pub fn new(kernel:String,grid_scale:u32,threads:u32)->Result<Self,String> {
        if kernel.is_empty() || kernel.len()>160 || !kernel.bytes().all(|byte|byte.is_ascii_alphanumeric() || byte==b'_')
            || grid_scale==0 || grid_scale>1024 || threads>1024 || threads%32!=0 {
            return Err("fixture geometry needs an entry name, grid_scale 1..1024, threads 0 or complete warps through 1024".into());
        }Ok(Self {kernel,grid_scale,threads})
    }
    pub fn dimensions(&self,blocks:u32,threads:u32)->Result<(u32,u32),String> {
        let blocks=blocks.checked_mul(self.grid_scale).filter(|&blocks|blocks>0 && blocks<=i32::MAX as u32).ok_or("fixture grid exceeds CUDA bounds")?;
        Ok((blocks,if self.threads==0 {threads} else {self.threads}))
    }
}
fn integer(bytes:&[u8],start:usize,width:usize)->Result<u64,String> {
    let span=bytes.get(start..start.checked_add(width).ok_or("image field overflow")?).ok_or("truncated image field")?;
    Ok(span.iter().enumerate().fold(0,|value,(index,byte)|value|(u64::from(*byte)<<(index*8))))
}
fn span(bytes:&[u8],start:u64,length:u64)->Result<(),String> {
    if start.checked_add(length).is_none_or(|end|end>bytes.len() as u64) {return Err("image table or payload exceeds file".into());}Ok(())
}
pub fn validate(bytes:&[u8])->Result<Image,String> {
    if bytes.is_empty() || bytes.len()>256<<20 {return Err("fixture image must be 1..268435456 bytes".into());}
    if !bytes.starts_with(b"\x7fELF") {
        let bytes=bytes.strip_suffix(&[0]).unwrap_or(bytes);
        let text=std::str::from_utf8(bytes).map_err(|_|"fixture image is not PTX or CUDA ELF")?;
        if !text.contains(".version ") || !text.contains(".target ") || !text.contains(".entry ") {return Err("fixture PTX lacks required declarations".into());}
        return CString::new(bytes).map(Image::Ptx).map_err(|_|"fixture PTX contains an interior NUL".into());
    }
    if bytes.len()<64 || bytes.get(4..7)!=Some(&[2,1,1]) || integer(bytes,18,2)?!=190 || integer(bytes,52,2)?!=64 {
        return Err("fixture cubin requires complete little-endian CUDA ELF64".into());
    }
    for (offset,size,count,minimum,section) in [(32,54,56,56,false),(40,58,60,64,true)] {
        let first=integer(bytes,offset,8)?;let stride=integer(bytes,size,2)?;let count=integer(bytes,count,2)?;
        if count==0 {if section {return Err("fixture cubin has no section table".into());}continue;}
        if stride<minimum {return Err("fixture image table stride is too short".into());}
        span(bytes,first,stride.checked_mul(count).ok_or("image table overflow")?)?;
        for index in 0..count {
            let at=usize::try_from(first+index*stride).map_err(|_|"image table exceeds address space")?;
            if section && integer(bytes,at+4,4)?==8 {continue;}
            let (start,length)=if section {(integer(bytes,at+24,8)?,integer(bytes,at+32,8)?)} else {(integer(bytes,at+8,8)?,integer(bytes,at+32,8)?)};
            span(bytes,start,length)?;
        }
    }Ok(Image::Cubin)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn geometry_checks_complete_warps_and_grid_overflow() {
        let geometry=Geometry::new("expert_group".into(),32,32).unwrap();assert_eq!(geometry.dimensions(32,128).unwrap(),(1024,32));
        assert!(geometry.dimensions(u32::MAX,32).is_err());assert!(Geometry::new("bad name".into(),1,32).is_err());
        assert!(Geometry::new("kernel".into(),0,32).is_err());assert!(Geometry::new("kernel".into(),1,33).is_err());
    }
    #[test] fn ptx_accepts_one_terminal_nul_only() {
        let ptx=b".version 8.0\n.target sm_121\n.visible .entry demo() {}\n";
        assert!(matches!(validate(ptx),Ok(Image::Ptx(_))));let mut terminal=ptx.to_vec();terminal.push(0);assert!(validate(&terminal).is_ok());
        terminal.push(0);assert!(validate(&terminal).is_err());assert!(validate(b"not a kernel").is_err());assert!(validate(&[]).is_err());
    }
    #[test] fn truncated_elf_and_out_of_file_tables_refuse() {
        let mut image=vec![0;128];image[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");image[18..20].copy_from_slice(&190u16.to_le_bytes());
        image[52..54].copy_from_slice(&64u16.to_le_bytes());image[40..48].copy_from_slice(&64u64.to_le_bytes());
        image[58..60].copy_from_slice(&64u16.to_le_bytes());image[60..62].copy_from_slice(&1u16.to_le_bytes());assert!(matches!(validate(&image),Ok(Image::Cubin)));
        image[88..96].copy_from_slice(&127u64.to_le_bytes());image[96..104].copy_from_slice(&2u64.to_le_bytes());assert!(validate(&image).is_err());
        assert!(validate(&image[..63]).is_err());image[40..48].copy_from_slice(&u64::MAX.to_le_bytes());assert!(validate(&image).is_err());
    }
}
