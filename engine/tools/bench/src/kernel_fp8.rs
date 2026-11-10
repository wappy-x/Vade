//! Synthetic W8A8 grouped projection fixtures: captured decode deals, prompt routing
//! counts derived from an FP4 receipt or constructed at tile boundaries, varied finite
//! operands, and output ownership guards.
use super::*;

const EDGE_COUNTS: [u32; 13] = [1,15,16,17,31,32,33,63,64,65,127,128,129];

pub(crate) fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for m in [1,5,8] { for owner in 0..2 { for down in [false,true] {
        out.push(Case { name:format!("fp8_w2_{}-c{m}-r{owner}",if down {"down"} else {"gate"}),
            kernel:if down {"fp8_grouped_down"} else {"fp8_grouped_gate_accurate"}, m,
            n:if down {1280} else {640}, k:if down {640} else {2560},
            kind:if down {72} else {71}, mats:[2,owner,0,0], ..Default::default() });
    } } }
    for m in [32,64,129] { for width in [128,256,384,640] {
        out.push(Case {name:format!("fp8_prompt_gate-edges-m{m}-w{width}"),kernel:"fp8_grouped_gate_prefill",
            m,n:width,k:2560,kind:73,mats:[1,0,width,0],..Default::default()});
    } }
    out.push(Case {name:"fp8_prompt_gate-edges-m33-w512".into(),kernel:"fp8_grouped_gate_prefill",
        m:33,n:512,k:2560,kind:73,mats:[1,0,512,0],..Default::default()});
    for m in [2048,4096] {
        for shards in [2,4] { for owner in 0..shards {
            out.push(Case {name:format!("fp8_prompt_gate-derived-m{m}-s{shards}-r{owner}"),kernel:"fp8_grouped_gate_prefill",
                m,n:640,k:2560,kind:73,real:true,mats:[shards,owner,0,0],..Default::default()});
        } }
        out.push(Case {name:format!("fp8_prompt_gate-derived-m{m}-w640"),kernel:"fp8_grouped_gate_prefill",
            m,n:640,k:2560,kind:73,real:true,mats:[1,0,640,0],..Default::default()});
    }
    out
}

fn width(c: &Case, expert: usize) -> usize {
    if c.mats[2] != 0 { return c.mats[2]; }
    let (shards, owner) = (c.mats[0],c.mats[1]);
    // Five K128 groups, with extra groups rotated by expert as in middle_expert_end.
    (5 / shards + usize::from((owner + shards - expert % shards) % shards < 5 % shards)) * 128
}

fn prompt_counts(c: &Case) -> Result<Vec<u32>> {
    let mut counts=vec![0u32;512];
    if c.real {
        if ![2048,4096].contains(&c.m) {return Err("FP8 derived counts require M2048 or M4096".into());}
        for (to,&v) in counts.iter_mut().zip(crate::floor::COLD8K_COUNTS.iter()) { *to=v*(c.m/2048) as u32; }
    } else {
        let mut remaining=c.m*10;
        let mut at=0;
        for &n in EDGE_COUNTS.iter().filter(|&&n|n as usize<=c.m) {
            if n as usize>remaining {break;}
            counts[at]=n;remaining-=n as usize;at+=1;
        }
        while remaining>0 {
            if at==counts.len() {return Err("FP8 edge count capacity".into());}
            let n=remaining.min(c.m);counts[at]=n as u32;remaining-=n;at+=1;
        }
    }
    if counts.iter().map(|&v|v as usize).sum::<usize>()!=c.m*10 || counts.iter().any(|&v|v as usize>c.m) {
        return Err("FP8 prompt counts do not cover distinct token/expert pairs".into());
    }
    Ok(counts)
}

fn prompt_pairs(c: &Case) -> Result<Vec<u32>> {
    let counts=prompt_counts(c)?;
    let mut pairs=vec![0;c.m*10];
    let mut i=0;
    for (e,&n) in counts.iter().enumerate() { for _ in 0..n {
        // Reverse row order avoids an identity sorted-row map at every token count.
        let token=c.m-1-i%c.m;
        pairs[token*10+i/c.m]=e as u32;i+=1;
    } }
    Ok(pairs)
}

struct Route {
    counts:Vec<u32>, starts:Vec<u32>, tasks:Vec<u32>, sorted:Vec<u32>, output_rows:Vec<u32>, active:Vec<usize>,
}
fn route(c: &Case, pairs: &[u32]) -> Result<Route> {
    if pairs.len()!=c.m*10 {return Err("FP8 route pair capacity".into());}
    let mut counts=vec![0u32;512];
    for ids in pairs.as_chunks::<10>().0 {
        for (i,&e) in ids.iter().enumerate() {
            if e>=512 || ids[..i].contains(&e) {return Err("FP8 route needs ten distinct experts per token".into());}
            counts[e as usize]+=1;
        }
    }
    let mut starts=vec![0u32;513];let mut tasks=vec![0u32];let mut sorted=Vec::new();let mut output_rows=Vec::new();
    let mut active=Vec::new();
    for e in 0..512 {
        starts[e]=sorted.len() as u32;
        if counts[e]>0 {active.push(e);}
        for first in (0..counts[e]).step_by(128) {tasks.extend([e as u32,first]);tasks[0]+=1;}
        for (pair,&id) in pairs.iter().enumerate().filter(|&(_,id)|*id==e as u32) {
            let token=pair/10;
            let slot=pairs[token*10..token*10+10].iter().filter(|&&other|other<id).count();
            sorted.push(token as u32);output_rows.push((token*11+slot) as u32);
        }
        let padded=sorted.len().div_ceil(16)*16;
        sorted.resize(padded,u32::MAX);output_rows.resize(padded,u32::MAX);
    }
    starts[512]=sorted.len() as u32;
    tasks.resize(1+2*(pairs.len().div_ceil(128)+pairs.len().min(512)),0);
    Ok(Route {counts,starts,tasks,sorted,output_rows,active})
}

/// Return owned FP32 bytes after checking all holes and the leading/trailing guard.
fn checked_output(bytes:&[u8],written:&[(usize,usize)]) -> Result<Vec<u8>> {
    let mut spans=written.to_vec();spans.sort_unstable();
    let mut end=0;let mut out=Vec::new();
    for &(at,len) in &spans {
        let next=at.checked_add(len).ok_or("FP8 output span overflow")?;
        if at<end || next>bytes.len() || at%4!=0 || len%4!=0 {return Err("FP8 output span overlap/extent".into());}
        if bytes[end..at].iter().any(|&b|b!=0xff) {return Err(format!("FP8 write outside owned output before byte {at}"));}
        if bytes[at..next].as_chunks::<4>().0.iter().any(|b|!f32::from_le_bytes(*b).is_finite()) {
            return Err(format!("FP8 nonfinite or unwritten output at byte {at}"));
        }
        out.extend_from_slice(&bytes[at..next]);end=next;
    }
    if bytes[end..].iter().any(|&b|b!=0xff) {return Err("FP8 write in trailing output guard".into());}
    Ok(out)
}

#[cfg(target_os="linux")]
fn code(h:u64) -> u8 {let b=(h%254) as u8;if b>=0x7f {b+1} else {b}}

#[cfg(target_os="linux")]
pub(super) fn fixture(g:&mut Gpu,c:&Case) -> Result<Fixture> {
    let down=c.kind==72;
    let pairs=if c.kind==73 {prompt_pairs(c)?} else {bw_routed::dealt_pairs(c)?.0};
    let r=route(c,&pairs)?;
    let dims=|e:usize|if down {(1280,640)} else {(width(c,e),2560)};
    let mats=if down {1} else {2};
    let matrix=|e:usize|{let (n,k)=dims(e);(n*k).div_ceil(256)*256+(n/128*(k/128)*2).div_ceil(256)*256};
    let per_copy=r.active.iter().map(|&e|mats*matrix(e)).sum::<usize>();
    let copies=rotate_bytes().div_ceil(per_copy).max(2);
    let mut image=vec![0u8;per_copy];let mut offsets=Vec::new();let mut at=0;
    for &e in &r.active {
        let (n,k)=dims(e);let mut offset=[0usize;2];
        for (m,slot) in offset.iter_mut().enumerate().take(mats) {
            let seed=(e as u64*4+if down {2} else {m as u64})<<40;
            for (j,b) in image[at..at+n*k].iter_mut().enumerate() {*b=code(mix(seed^j as u64));}
            let scales=at+(n*k).div_ceil(256)*256;
            for j in 0..n/128*(k/128) {
                let hash=mix(seed^0x8000_0000^j as u64);
                let bits=((119+hash%4)<<7|(hash>>8)&0x7f) as u16;
                image[scales+j*2..scales+j*2+2].copy_from_slice(&bits.to_le_bytes());
            }
            *slot=at;at+=matrix(e);
        }
        offsets.push(offset);
    }
    let slab=g.alloc(copies*per_copy)?;let mut ring=Vec::new();
    eprintln!("weights case={} slab={} experts={} copies={} route=synthetic",c.name,slab,r.active.len(),copies);
    for copy in 0..copies {
        let base=slab+(copy*per_copy) as u64;g.upload_at(base,&image)?;
        let mut table=vec![0u64;512*7];
        for (a,&e) in r.active.iter().enumerate() {
            let (n,k)=dims(e);let row=&mut table[e*7..e*7+7];
            for (m,&off) in offsets[a].iter().enumerate().take(mats) {
                let p=if down {4} else {2*m};row[p]=base+off as u64;
                row[p+1]=row[p]+((n*k).div_ceil(256)*256) as u64;
            }
            row[6]=width(c,e) as u64;
        }
        ring.push(g.upload(&table.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?);
    }
    let (rows,k)=if down {(r.sorted.len(),640)} else {(c.m,2560)};
    let codes=(0..rows*k).map(|j|code(mix(0x1234_5678_0000_0000^j as u64))).collect::<Vec<_>>();
    let input=g.upload(&codes)?;
    let scales=(0..rows*(k/128)).flat_map(|j|{
        let h=mix(0x2345_6789_0000_0000^j as u64);
        f32::from_bits(((120+h%4)<<23|(h>>8)&0x7f_ffff) as u32).to_le_bytes()
    }).collect::<Vec<_>>();
    let input_scales=g.upload(&scales)?;
    let guard=64;let payload=if down {c.m*11*2560*4} else {r.sorted.len()*1280*4};
    let out_bytes=guard+payload+guard;let out=g.alloc(out_bytes)?;
    let mut written=Vec::new();
    if down {
        for &row in r.output_rows.iter().filter(|&&v|v!=u32::MAX) {written.push((guard+(row as usize*2560+c.mats[1]*1280)*4,1280*4));}
    } else {
        for &e in &r.active {for row in r.starts[e]..r.starts[e]+r.counts[e] {
            written.extend([(guard+row as usize*1280*4,width(c,e)*4),(guard+(row as usize*1280+640)*4,width(c,e)*4)]);
        }}
    }
    let unique=r.active.iter().map(|&e|{let(n,k)=dims(e);mats*(n*k+n/128*(k/128)*2)}).sum::<usize>();
    let flops=r.active.iter().map(|&e|{let(n,k)=dims(e);2*r.counts[e] as usize*n*k*mats}).sum();
    let route=fp8::GroupedRouteParams {
        error:g.words(&vec![0;c.m])?,pairs:0,counts:g.words(&r.counts)?,starts:g.words(&r.starts)?,
        pair_to_row:0,token_pairs:0,token_count:0,tasks:g.words(&r.tasks)?,sorted_row:g.words(&r.sorted)?,output_row:g.words(&r.output_rows)?,
        experts:512,tokens:c.m as u32,pair_count:pairs.len() as u32,row_capacity:r.sorted.len() as u32,
    };
    let rows=params::Bf16Rows {weight:0,output:out,rows:0,stride:0,weight2:0,output2:0,rows2:0,stride2:0,
        input,cols:k as u32,tokens:c.m as u32,input_bf16:0,hc_silu:0,input_plane_words:0};
    let live_input=if down {pairs.len()*640+pairs.len()*5*4} else {codes.len()+scales.len()};
    let stored=unique+live_input+written.iter().map(|r|r.1).sum::<usize>();
    Ok(Fixture {rows,group:Some(fp8::GroupedParams {route,weights:ring[0],input,input_scales,
        output:out+(guard+if down {c.mats[1]*1280*4} else {0}) as u64,cols:2560,activation_quantization:1,output_shard:u32::from(down)}),
        ring,out,out_bytes,aux:0,aux_bytes:0,grid:[20*fp8::grouped_planes(c.m as u32,c.kind==73),r.tasks.len().saturating_sub(1) as u32/2,1],
        threads:128,shared:0,stored,flops,chunks:1,down_slots:0,raw:0,pack:null_mut(),norms:[null_mut();2],
        weight_bytes:unique,written,qsa:None,gather:None,ring_x:vec![],scales_x:vec![]})
}

#[cfg(target_os="linux")]
pub(super) fn output_hash(g:&Gpu,v:&Fixture) -> Result<String> {
    Ok(sha256::digest(&checked_output(&g.read(v.out,v.out_bytes)?,&v.written)?))
}
/// Pair only prompt gates. `1` retains the original X-only switch spelling.
fn paired_mode(kind:u8,arm:usize,mode:Option<&str>) -> bool {
    kind==73 && matches!((mode,arm),(Some("1"|"x"),1)|(Some("a"),0)|(Some("both"),0|1))
}
#[cfg(target_os="linux")]
pub(super) fn paired(c:&Case,arm:usize) -> bool {
    paired_mode(c.kind,arm,std::env::var("KBENCH_FP8_PAIRED").ok().as_deref())
}
/// Where a case's routing counts come from: captured, receipt-derived or constructed.
fn provenance(c:&Case) -> &'static str {
    match (c.kind,c.real) {
        (73,true) => "synthetic operands; prompt counts derived from the FP4 cold 8K receipt",
        (73,false) => "synthetic operands; constructed boundary prompt counts",
        _ => "synthetic operands; captured C1/C8 decode deals",
    }
}
/// `threads` = each arm's launched block size (its module's `_THREADS`, else the fixture's).
#[cfg(target_os="linux")]
pub(super) fn metadata(c:&Case,v:&Fixture,threads:[u32;2]) -> String {
    let a=paired(c,0);let x=paired(c,1);
    format!(",\"fixture_provenance\":\"{}\",\"output_guards\":true,\"route_padding_rows\":16,\"grid_x_a\":{},\"grid_x_x\":{},\"fp8_paired_a\":{},\"fp8_paired_x\":{},\"grid_y\":{},\"grid_z\":{},\"launch_threads_a\":{},\"launch_threads_x\":{}",
        provenance(c),v.grid[0]/if a {2} else {1},v.grid[0]/if x {2} else {1},a,x,v.grid[1],v.grid[2],threads[0],threads[1])
}
#[cfg(target_os="linux")]
pub(super) fn basis() -> String {
    "FP8 E4M3 weights + BF16 K128/N128 scales + E4M3 input/FP32 scales + owned FP32 output; unique weights, bandwidth-only floor; synthetic routing/operands".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn prompt_routes_cover_edges_and_permute_rows() {
        for c in cases().iter().filter(|c|c.kind==73) {
            let pairs=prompt_pairs(c).unwrap();let r=route(c,&pairs).unwrap();
            assert_eq!(r.starts[512] as usize,r.counts.iter().map(|&n|(n as usize).div_ceil(16)*16).sum::<usize>());
            assert_eq!(r.sorted.iter().filter(|&&v|v!=u32::MAX).count(),c.m*10);assert_eq!(r.output_rows.len(),r.sorted.len());
            assert!(r.sorted.windows(2).any(|a|a[0]>a[1]));
            assert!(r.tasks[0] as usize<=r.tasks.len()/2);
            for &e in &r.active {assert!(r.counts[e]>0);assert_eq!(r.starts[e+1]-r.starts[e],r.counts[e].div_ceil(16)*16);
                let start=r.starts[e] as usize;let end=(r.starts[e]+r.counts[e]) as usize;
                assert!(r.sorted[start..end].iter().all(|&v|(v as usize)<c.m));
                assert!(r.sorted[end..r.starts[e+1] as usize].iter().all(|&v|v==u32::MAX));}
            if c.m==129 {for n in EDGE_COUNTS {assert!(r.counts.contains(&n));}}
            let mut slots=r.output_rows.iter().copied().filter(|&v|v!=u32::MAX).collect::<Vec<_>>();slots.sort_unstable();slots.dedup();assert_eq!(slots.len(),c.m*10);
        }
    }
    #[test] fn ownership_widths_cover_both_topologies() {
        for shards in [2,4] {for e in 0..512 {
            let widths=(0..shards).map(|owner|width(&Case {mats:[shards,owner,0,0],..Default::default()},e)).collect::<Vec<_>>();
            assert_eq!(widths.iter().sum::<usize>(),640);
            assert!(widths.iter().all(|w|[128,256,384].contains(w)));
        }}
    }
    #[test] fn paired_modes_select_only_requested_prompt_arms() {
        for (mode,expected) in [(None,[false,false]),(Some("1"),[false,true]),(Some("x"),[false,true]),
            (Some("a"),[true,false]),(Some("both"),[true,true]),(Some("0"),[false,false])] {
            for (arm,&exp) in expected.iter().enumerate() {
                assert_eq!(paired_mode(73,arm,mode),exp);
                assert!(!paired_mode(71,arm,mode));assert!(!paired_mode(72,arm,mode));
            }
            assert!(!paired_mode(73,2,mode));
        }
    }
    #[test] fn provenance_names_each_count_source() {
        for c in cases() {
            let p=provenance(&c);
            assert_eq!(p.contains("FP4 cold 8K receipt"),c.kind==73 && c.real);
            assert_eq!(p.contains("constructed boundary"),c.kind==73 && !c.real);
            assert_eq!(p.contains("decode deals"),c.kind!=73);
        }
    }
    #[test] fn poisoned_output_checks_holes_and_bounds() {
        let mut bytes=vec![0xff;32];bytes[8..12].copy_from_slice(&1.25f32.to_le_bytes());
        assert_eq!(checked_output(&bytes,&[(8,4)]).unwrap(),1.25f32.to_le_bytes());
        bytes[16]=0;assert!(checked_output(&bytes,&[(8,4)]).is_err());bytes[16]=0xff;
        assert!(checked_output(&bytes,&[(8,8)]).is_err());assert!(checked_output(&bytes,&[(8,4),(8,4)]).is_err());
        assert!(checked_output(&bytes,&[(28,8)]).is_err());
    }
}
