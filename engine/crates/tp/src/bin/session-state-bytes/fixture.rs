//! Fixed opaque byte fixture. The oracle does not allocate or call a session owner.
use spark_tp::session_state::{Layout, Plane, Length};
use engine_model::plan::state::PlaneKind::*;
type Result<T> = std::result::Result<T,String>;
pub fn layout() -> Result<Layout> {
    Layout::new(7,vec![
        Plane {layer:0,kind:KdaState,length:Length::Fixed(128)},
        Plane {layer:0,kind:KdaHistory,length:Length::Fixed(54)},
        Plane {layer:1,kind:Latent,length:Length::Rows {row_bytes:10,tokens_per_row:1}},
        Plane {layer:1,kind:IndexKeys,length:Length::Rows {row_bytes:6,tokens_per_row:1}},
        Plane {layer:1,kind:IndexGates,length:Length::Rows {row_bytes:6,tokens_per_row:1}},
        Plane {layer:1,kind:Pools,length:Length::Rows {row_bytes:6,tokens_per_row:2}},
    ],258)
}
pub fn next(layout: &Layout,old: &[Vec<u8>],context: u64,token: u8) -> Result<Vec<Vec<u8>>> {
    layout.planes().iter().enumerate().map(|(i,p)| {
        let n=p.used_bytes(context)?;
        Ok(match p.length {
            Length::Fixed(_)=>old[i].iter().enumerate().map(|(j,&v)| v.wrapping_mul(3).wrapping_add(token).wrapping_add(j as u8)).collect(),
            Length::Rows {..}=>{let mut b=old[i].clone(); b.extend((b.len()..n).map(|j| token.wrapping_add(j as u8))); b},
        })
    }).collect()
}
pub fn record(context: u64,bytes: &[Vec<u8>],out: &mut Vec<u8>) {
    out.extend(context.to_le_bytes());
    for b in bytes {out.extend((b.len() as u64).to_le_bytes()); out.extend(b);}
}
pub fn oracle() -> Result<Vec<u8>> {
    let l=layout()?;
    let mut bytes=l.planes().iter().map(|p| p.used_bytes(0).map(|n|vec![0;n])).collect::<Result<Vec<_>>>()?;
    let mut out=Vec::new(); let mut prefix=Vec::new();
    for (i,&token) in [1,3,2,7,5,9,11].iter().enumerate() {
        bytes=next(&l,&bytes,(i+1) as u64,token)?; record((i+1) as u64,&bytes,&mut out);
        if i==2 {prefix=bytes.clone();}
    }
    record(7,&bytes,&mut out); record(3,&prefix,&mut out); Ok(out)
}
