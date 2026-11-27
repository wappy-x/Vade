//! Pointwise draft operations and ordered block attention, one output per lane.
use crate::fixed_math::{exp,rn,sigmoid,tanh};
use crate::tc_ptx::{shfl_xor_f32,shfl_xor_u32};

/// Keys of the draft attention walk (operation 20) whose K and V rows a warp prefetches into L1 ahead of the key it
/// reads: 8 keys of ~150-300 cycles each cover one DRAM read; 8 keys x 1 KB a warp group stays well inside L1.
const ATTN_PREFETCH_KEYS:usize=8;

/// Pure FP32 rounding helper; non-finite sums or BF16 overflows become NaN.
#[inline(always)]
fn round_checked(raw:f32)->f32 {
    if raw.is_finite() {
        let r=rn(raw);
        if r.is_finite() {r} else {f32::NAN}
    } else {f32::NAN}
}

/// # Safety
/// SAFETY: Only the checked host wrapper may launch this entry. The operands and geometry
/// must satisfy the selected operation's layout for the full lifetime of the launch.
#[no_mangle]
/// `table[index]` must be an aligned, live command with 12 initialized payload words.
pub unsafe extern "ptx-kernel" fn inkling_draft_lanes(table:*const crate::Command,index:u64) {
    // SAFETY: Backend::draft_lanes checks each input length, grid, shape and head grouping.
    // Operation 8 reads row `id` of a table of `a` rows of n BF16 words only for a whole `id` under `a`.
    // Operation 9 reads word i of each of its three inputs of n words.
    // Inputs are contiguous, live device spans that no lane writes, so operation 2 may read them through the
    // read-only path (16 bytes only at 16-byte addresses inside a row). Output is a separate allocation. Each of
    // the n lanes owns one output word; for operation 3, n is 32 lanes a row, the 32 lanes of a warp own one
    // probability row and all take part in each shuffle (n and the 128-lane blocks are whole warps).
    // Operations 10 and 11 use whole warps and bounded inputs. Each warp writes its own pick triple.
    // Operation 12 uses whole warps: warp `p` reads words `p * b` to `min(p * b + b, a)` of x (a words) and the three
    // words of the pick in y, and lane 0 writes word p of the output (`a.div_ceil(b)` words, one a warp).
    unsafe {
        let cmd=(*table.add(index as usize)).payload.as_ptr().cast::<u64>();
        let [kind,x,y,z,out,n,a,b,c,d,e,_]=cmd.cast::<[u64;12]>().read();
        let (a,b,c,d,e)=(a as usize,b as usize,c as usize,d as usize,e as usize);
        let i=core::arch::nvptx::_block_idx_x() as usize*128+core::arch::nvptx::_thread_idx_x() as usize;
        if i>=n as usize {return;}
        let (x,y)=(x as *const f32,y as *const f32);
        // SAFETY: operation 0 (`Gate`) is vectorized 4 words/lane (`n == a.div_ceil(4)`); `v4` loads/stores run only
        // when all three spans are 16-byte aligned and all 4 words lie below `a`.
        if kind==0 {
            let base=i*4;
            if base>=a {return;}
            let dst=out as *mut f32;
            if base+4<=a && ((x as usize|y as usize|dst as usize)&15)==0 {
                let gv=crate::tc_ptx::ld_global_nc_v4(x.add(base));
                let uv=crate::tc_ptx::ld_global_nc_v4(y.add(base));
                let r0=rn(rn(gv[0]*sigmoid(gv[0]))*uv[0]);
                let r1=rn(rn(gv[1]*sigmoid(gv[1]))*uv[1]);
                let r2=rn(rn(gv[2]*sigmoid(gv[2]))*uv[2]);
                let r3=rn(rn(gv[3]*sigmoid(gv[3]))*uv[3]);
                core::arch::asm!(
                    "st.global.v4.f32 [{p}], {{{r0}, {r1}, {r2}, {r3}}};",
                    p=in(reg64) dst.add(base), r0=in(reg32) r0, r1=in(reg32) r1, r2=in(reg32) r2, r3=in(reg32) r3, options(nostack)
                );
            } else {
                let end=(base+4).min(a);
                let mut k=base;
                while k<end {
                    let g=*x.add(k);
                    *dst.add(k)=rn(rn(g*sigmoid(g))**y.add(k));
                    k+=1;
                }
            }
            return;
        }
        // SAFETY: operation 29 (`RopeBf16`) writes one packed BF16 `u32` pair per lane (`width % 4 == 0`); each pair
        // shares the same half `< width/2` or `>= width/2`, matching `Rope` (`kind == 1`) bit for bit.
        if kind==29 {
            let half=a/2;
            let quarter=a/4;
            let p=i%half;
            let h=i/half;
            let row=h/b;
            let m=(p%quarter)*2;
            let trig=y.add(row*a);
            let lo0=*x.add(h*a+m);let lo1=*x.add(h*a+m+1);
            let hi0=*x.add(h*a+half+m);let hi1=*x.add(h*a+half+m+1);
            let cos0=*trig.add(m);let cos1=*trig.add(m+1);
            let sin0=*trig.add(half+m);let sin1=*trig.add(half+m+1);
            let v0=rn(if p<quarter {lo0*cos0-hi0*sin0} else {hi0*cos0+lo0*sin0});
            let v1=rn(if p<quarter {lo1*cos1-hi1*sin1} else {hi1*cos1+lo1*sin1});
            *(out as *mut u32).add(i)=(v0.to_bits()>>16)|(v1.to_bits()&0xffff_0000);
            return;
        }
        // SAFETY: operation 30 (`PackBf16`) packs `a` BF16-rounded FP32 words into `a / 2` packed BF16 `u32` words.
        if kind==30 {
            let base=i*4;
            if base>=a {return;}
            let dst=out as *mut u32;
            if base+4<=a && ((x as usize)&15)==0 && ((dst as usize)&7)==0 {
                let v=crate::tc_ptx::ld_global_nc_v4(x.add(base));
                let p0=(v[0].to_bits()>>16)|(v[1].to_bits()&0xffff_0000);
                let p1=(v[2].to_bits()>>16)|(v[3].to_bits()&0xffff_0000);
                core::arch::asm!(
                    "st.global.v2.u32 [{p}], {{{p0}, {p1}}};",
                    p=in(reg64) dst.add(i*2), p0=in(reg32) p0, p1=in(reg32) p1, options(nostack)
                );
            } else {
                let mut k=base;
                let end=(base+4).min(a);
                while k+1<end {
                    let p0=((*x.add(k)).to_bits()>>16)|((*x.add(k+1)).to_bits()&0xffff_0000);
                    *dst.add(k/2)=p0;
                    k+=2;
                }
            }
            return;
        }
        // SAFETY: operation 31 (`ScatterKvBatch`) copies packed K and copies/BF16-packs V across `b <= 32` chats into
        // host-checked writable destination planes encoded at `cmd[12..12 + 3*b]`; each lane owns 2 `u32`s of one row.
        if kind==31 {
            let half_stride=a/2;
            let row=i/half_stride;
            let col2=i%half_stride;
            let mut bi=0usize;
            let mut dk=0u64;
            let mut dv=0u64;
            let mut src_off=0usize;
            while bi<b {
                let meta=*cmd.add(12+bi*3+2);
                let off=(meta as u32) as usize;
                let cnt=(meta>>32) as usize;
                if row>=off && row<off+cnt {
                    dk=*cmd.add(12+bi*3);
                    dv=*cmd.add(12+bi*3+1);
                    src_off=off;
                    break;
                }
                bi+=1;
            }
            if dk==0 {return;}
            let local_row=row-src_off;
            let dst_off=local_row*a+col2*2;
            let src_u32=row*a+col2*2;
            let kv=crate::tc_ptx::ld_global_nc_v2_u32((x as *const u32).add(src_u32));
            core::arch::asm!(
                "st.global.v2.u32 [{p}], {{{v0}, {v1}}};",
                p=in(reg64) (dk as *mut u32).add(dst_off), v0=in(reg32) kv[0], v1=in(reg32) kv[1], options(nostack)
            );
            let (vp0,vp1)=if c!=0 {
                let vf=crate::tc_ptx::ld_global_nc_v4(y.add(row*a*2+col2*4));
                ((vf[0].to_bits()>>16)|(vf[1].to_bits()&0xffff_0000),(vf[2].to_bits()>>16)|(vf[3].to_bits()&0xffff_0000))
            } else {
                let vv=crate::tc_ptx::ld_global_nc_v2_u32((y as *const u32).add(src_u32));
                (vv[0],vv[1])
            };
            core::arch::asm!(
                "st.global.v2.u32 [{p}], {{{v0}, {v1}}};",
                p=in(reg64) (dv as *mut u32).add(dst_off), v0=in(reg32) vp0, v1=in(reg32) vp1, options(nostack)
            );
            return;
        }
        // SAFETY: operation 27 (`FlashAttnBatch`) runs whole warps across `batch <= 32` chats with host-verified
        // committed `(ck, cv)` pointers at `cmd[12..12 + 3*batch]` plus contiguous `(k_new, v_new)`. Warps are
        // grouped by `(bi, kv_head, split)` so all query heads and query rows sharing a KV head sit adjacent in the
        // same CTA and hit L1 from the single lead warp's prefetches while keeping `inkling_draft_lanes <= 64` regs.
        if kind==27 {
            let scale=f32::from_bits(a as u32);
            let i32=i as u32;
            let lane=i32&31;
            let (q0,hist_n,new_packed,k_h_ptr0,v_h_ptr0,hist_stride,lead,kv_bf16)={
                let max_splits=((a>>40)&0xff) as u32;
                let kv_bf16=((a>>48)&1)!=0;
                let heads=(b&0xffff) as u32;
                let kv=((b>>16)&0xffff) as u32;
                let q_rows=((b>>32)&0xffff) as u32;
                let new_rows=((b>>48)&0xffff) as u32;
                let causal_base=c as u32;
                let row_off=(c>>32) as u32;
                let warp=i32>>5;
                let group=heads/kv;
                let per_split=q_rows*group;
                let per_kv=max_splits*per_split;
                let per_chat=kv*per_kv;
                let bi=warp/per_chat;let rem0=warp%per_chat;
                let kh=rem0/per_kv;let rem1=rem0%per_kv;
                let split=rem1/per_split;let rem2=rem1%per_split;
                let row=rem2/group;let sub=rem2%group;
                let head=kh*group+sub;
                let cmeta=*cmd.add(12+(bi as usize)*3+2);
                let hist_keys=cmeta as u32;
                let splits_b=(cmeta>>32) as u32;
                if split>=splits_b {return;}
                let keys=hist_keys+new_rows;
                let row_keys=if causal_base>0 {
                    let r_abs=row+row_off;
                    if r_abs<8 {causal_base} else if r_abs<12 {(causal_base+4).min(keys)} else {keys}
                } else {keys};
                let chunk=keys.div_ceil(splits_b);
                let start=split*chunk;
                let end=(start+chunk).min(row_keys);
                let hist_end=end.min(hist_keys);
                let hist_n=hist_end.saturating_sub(start);
                let new_start=start.max(hist_keys);
                let new_n=end.saturating_sub(new_start);
                let new_off=new_start.saturating_sub(hist_keys);
                let q_id=((bi*q_rows+row)*heads+head) as usize;
                let q0=crate::tc_ptx::ld_global_nc_v4(x.add(q_id*128+(lane as usize)*4));
                let ck_addr=*cmd.add(12+(bi as usize)*3);
                let cv_addr=*cmd.add(12+(bi as usize)*3+1);
                let hs=kv*if kv_bf16 {64} else {128};
                let hlo=kh*if kv_bf16 {64} else {128}+lane*if kv_bf16 {2} else {4};
                let off=(start as usize)*(hs as usize)+(hlo as usize);
                let kp=if hist_n>0 {(ck_addr as *const u32).add(off)} else {ck_addr as *const u32};
                let vp=if hist_n>0 {(cv_addr as *const u32).add(off)} else {cv_addr as *const u32};
                let lead=causal_base>0 || (warp&3)==0 || (warp-1)/per_split!=warp/per_split;
                (q0,hist_n,(new_off<<16)|new_n,kp,vp,hs,lead,kv_bf16)
            };
            let mut m=f32::NEG_INFINITY;let mut l=0.0f32;let mut o=[0.0f32;4];
            // Loop 1: committed history keys (`0..hist_n`), hot L1-prefetched path with u32 counters.
            if hist_n>0 {
                let mut k_h_ptr=k_h_ptr0;
                let mut v_h_ptr=v_h_ptr0;
                let hs=hist_stride as usize;
                let ahead=ATTN_PREFETCH_KEYS*hs;
                let mut warm=0u32;
                let warm_end=hist_n.min(ATTN_PREFETCH_KEYS as u32);
                while lead && warm<warm_end {
                    let at=(warm as usize)*hs;
                    core::arch::asm!("prefetch.global.L1 [{k}];","prefetch.global.L1 [{v}];",k=in(reg64) k_h_ptr.add(at),v=in(reg64) v_h_ptr.add(at),options(readonly,nostack));
                    warm+=1;
                }
                let mut rem_k=hist_n;
                while rem_k>0 {
                    if lead && rem_k>ATTN_PREFETCH_KEYS as u32 {
                        core::arch::asm!("prefetch.global.L1 [{k}];","prefetch.global.L1 [{v}];",k=in(reg64) k_h_ptr.add(ahead),v=in(reg64) v_h_ptr.add(ahead),options(readonly,nostack));
                    }
                    let k0=if kv_bf16 {
                        let pk=crate::tc_ptx::ld_global_nc_v2_u32(k_h_ptr);
                        [f32::from_bits(pk[0]<<16),f32::from_bits(pk[0]&0xffff_0000),f32::from_bits(pk[1]<<16),f32::from_bits(pk[1]&0xffff_0000)]
                    } else {
                        crate::tc_ptx::ld_global_nc_v4(k_h_ptr as *const f32)
                    };
                    let mut dot=q0[0]*k0[0]+q0[1]*k0[1]+q0[2]*k0[2]+q0[3]*k0[3];
                    dot+=shfl_xor_f32(dot,16);dot+=shfl_xor_f32(dot,8);dot+=shfl_xor_f32(dot,4);dot+=shfl_xor_f32(dot,2);dot+=shfl_xor_f32(dot,1);
                    let sc=dot*scale;
                    let v0=if kv_bf16 {
                        let pv=crate::tc_ptx::ld_global_nc_v2_u32(v_h_ptr);
                        [f32::from_bits(pv[0]<<16),f32::from_bits(pv[0]&0xffff_0000),f32::from_bits(pv[1]<<16),f32::from_bits(pv[1]&0xffff_0000)]
                    } else {
                        crate::tc_ptx::ld_global_nc_v4(v_h_ptr as *const f32)
                    };
                    let p=if sc>m {
                        let alpha=if l==0.0 {0.0} else {exp(m-sc)};
                        l*=alpha;o[0]*=alpha;o[1]*=alpha;o[2]*=alpha;o[3]*=alpha;m=sc;1.0f32
                    } else {exp(sc-m)};
                    l+=p;o[0]+=p*v0[0];o[1]+=p*v0[1];o[2]+=p*v0[2];o[3]+=p*v0[3];
                    k_h_ptr=k_h_ptr.add(hs);
                    v_h_ptr=v_h_ptr.add(hs);
                    rem_k-=1;
                }
            }
            let a_re=*cmd.add(6);
            let b_re=*cmd.add(7);
            let max_splits=((a_re>>40)&0xff) as u32;
            let k_new_bf16=((a_re>>49)&1)!=0;
            let heads=(b_re&0xffff) as u32;
            let kv=((b_re>>16)&0xffff) as u32;
            let q_rows=((b_re>>32)&0xffff) as u32;
            let new_rows=((b_re>>48)&0xffff) as u32;
            let warp=i32>>5;
            let group=heads/kv;
            let per_split=q_rows*group;
            let per_kv=max_splits*per_split;
            let per_chat=kv*per_kv;
            let bi=warp/per_chat;let rem0=warp%per_chat;
            let kh=rem0/per_kv;let rem1=rem0%per_kv;
            let split=rem1/per_split;let rem2=rem1%per_split;
            let row=rem2/group;let sub=rem2%group;
            let head=kh*group+sub;
            let q_id=((bi*q_rows+row)*heads+head) as usize;
            let new_n=new_packed&0xffff;
            if new_n>0 {
                let yn=*cmd.add(2) as *const f32;
                let zn=*cmd.add(3) as *const f32;
                let mut r_local=new_packed>>16;
                let r_end=r_local+new_n;
                while r_local<r_end {
                    let r=(bi*new_rows+r_local) as usize;
                    let k0=if k_new_bf16 {
                        let p=crate::tc_ptx::ld_global_nc_v2_u32((yn as *const u32).add((r*(kv as usize)+(kh as usize))*64+(lane as usize)*2));
                        [f32::from_bits(p[0]<<16),f32::from_bits(p[0]&0xffff_0000),f32::from_bits(p[1]<<16),f32::from_bits(p[1]&0xffff_0000)]
                    } else {
                        crate::tc_ptx::ld_global_nc_v4(yn.add((r*(kv as usize)+(kh as usize))*128+(lane as usize)*4))
                    };
                    let mut dot=q0[0]*k0[0]+q0[1]*k0[1]+q0[2]*k0[2]+q0[3]*k0[3];
                    dot+=shfl_xor_f32(dot,16);dot+=shfl_xor_f32(dot,8);dot+=shfl_xor_f32(dot,4);dot+=shfl_xor_f32(dot,2);dot+=shfl_xor_f32(dot,1);
                    let sc=dot*scale;
                    let v0=crate::tc_ptx::ld_global_nc_v4(zn.add((r*(kv as usize)+(kh as usize))*128+(lane as usize)*4));
                    let p=if sc>m {
                        let alpha=if l==0.0 {0.0} else {exp(m-sc)};
                        l*=alpha;o[0]*=alpha;o[1]*=alpha;o[2]*=alpha;o[3]*=alpha;m=sc;1.0f32
                    } else {exp(sc-m)};
                    l+=p;o[0]+=p*v0[0];o[1]+=p*v0[1];o[2]+=p*v0[2];o[3]+=p*v0[3];
                    r_local+=1;
                }
            }
            let out_p=*cmd.add(4) as *mut f32;
            if max_splits==1 {
                let inv=1.0/l;
                let dst=out_p.add(q_id*128+(lane as usize)*4);
                *dst=rn(o[0]*inv);*dst.add(1)=rn(o[1]*inv);*dst.add(2)=rn(o[2]*inv);*dst.add(3)=rn(o[3]*inv);
            } else {
                let ob=(q_id*(max_splits as usize)+(split as usize))*132;
                let dst=out_p.add(ob+(lane as usize)*4);
                *dst=o[0];*dst.add(1)=o[1];*dst.add(2)=o[2];*dst.add(3)=o[3];
                if lane==0 {*out_p.add(ob+128)=m;*out_p.add(ob+129)=l;}
            }
            return;
        }
        // SAFETY: operation 28 (`FlashAttnMergeBatch`) merges each chat's `splits_b` partials (packed as `u8` bytes at
        // payload offset 96..96+batch) across `max_splits = c` stride into contiguous `[batch, q_rows, heads, 128]`.
        if kind==28 {
            let (warp,lane)=(i/32,i%32);
            let (heads,q_rows,max_splits)=(a,b,c);
            let bi=warp/(q_rows*heads);
            let splits_b=*(cmd.cast::<u8>().add(96+bi)) as usize;
            let mut m=f32::NEG_INFINITY;let mut l=0.0f32;let mut o=[0.0f32;4];
            let mut s=0usize;
            while s<splits_b {
                let base=(warp*max_splits+s)*132;
                let ls=*x.add(base+129);
                if ls!=0.0 {
                    let ms=*x.add(base+128);
                    let os=crate::tc_ptx::ld_global_nc_v4(x.add(base+lane*4));
                    if l==0.0 {m=ms;l=ls;o=os;}
                    else if ms>m {
                        let alpha=exp(m-ms);
                        o[0]=o[0]*alpha+os[0];o[1]=o[1]*alpha+os[1];o[2]=o[2]*alpha+os[2];o[3]=o[3]*alpha+os[3];
                        l=l*alpha+ls;m=ms;
                    } else {
                        let alpha=exp(ms-m);
                        o[0]+=os[0]*alpha;o[1]+=os[1]*alpha;o[2]+=os[2]*alpha;o[3]+=os[3]*alpha;
                        l+=ls*alpha;
                    }
                }
                s+=1;
            }
            let inv=1.0/l;
            let dst=(out as *mut f32).add(warp*128+lane*4);
            *dst=rn(o[0]*inv);*dst.add(1)=rn(o[1]*inv);*dst.add(2)=rn(o[2]*inv);*dst.add(3)=rn(o[3]*inv);
            return;
        }
        if kind==12 {
            // The softmax denominator of a slice against the greedy pick `[best, id, flag]`: `exp(logit - best)` summed
            // by each lane in ascending order, then across the warp. A pick whose flag is not 1 gives 0.
            let (lane,part)=(i%32,i/32);let (best,flag)=(*y,*y.add(2));
            let mut sum=0.0f32;
            if flag==1.0 {
                let end=(part*b+b).min(a);let mut at=part*b+lane;
                while at<end {sum+=exp(*x.add(at)-best);at+=32;}
            }
            for distance in [16,8,4,2,1] {sum+=shfl_xor_f32(sum,distance);}
            if lane==0 {*(out as *mut f32).add(part)=sum;}
            return;
        }
        // SAFETY: complete warps own contiguous words; lanes in the tail
        // participate in the shuffle but cannot read or write a residual.
        if kind==18 {
            let mut valid=1;
            if i<a {
                let raw=*x.add(i)+*y.add(i);
                let rounded=if raw.is_finite() {rn(raw)} else {raw};
                *(out as *mut f32).add(i)=rounded;
                valid=u32::from(rounded.is_finite());
            }
            for dist in [16,8,4,2,1] {valid&=shfl_xor_u32(valid,dist);}
            if i%32==0 {
                let flag=(out as *mut f32).add(a+i/32*3);
                *flag=0.0;*flag.add(1)=0.0;*flag.add(2)=valid as f32;
            }
            return;
        }
        // SAFETY: operation 24 combines `a` depth rows across `c.max(1)` chats of `b` words:
        // `out[k] = (x[k] + y[((k/b) % batch) * b + k%b]) + z[k%b]`.
        if kind==24 {
            let batch=c.max(1);
            let total=a*batch*b;
            let base=i*4;
            if base>=total {return;}
            let z=z as *const f32;
            let dst=out as *mut f32;
            let end=(base+4).min(total);
            let mut k=base;
            while k<end {
                let col=k%b;
                let feat=if batch==1 {col} else {((k/b)%batch)*b+col};
                *dst.add(k)=(*x.add(k)+*y.add(feat))+*z.add(col);
                k+=1;
            }
            return;
        }
        // SAFETY: operations 16 (`PackActivations`), 25 (`MarkovPrep`/`MarkovPrepBatch`), and 26
        // (`MarkovStep`/`MarkovStepBatch`) share one register-only 2-pass NVFP4 activation packer (`0` stack bytes).
        // For `kind == 26`, `n` is a multiple of 128 so all 128 lanes of each block participate in the parallel GRU
        // recurrence and reach `_syncthreads()` before `blk_len / 16` lanes pack the staged `latent` values into disjoint tiles.
        if kind==16 || kind==25 || kind==26 {
            use core::arch::asm;
            let (mut vals,tile_off,pack_i)=if kind==16 {
                (core::array::from_fn::<f32,16,_>(|j|*x.add(i*16+j)),0usize,i)
            } else if kind==25 {
                let from_pick=d!=0;
                if e==0 {
                    if from_pick && *y.add(4)!=0.0 {return;}
                    let half_lanes=a/16;
                    let vals=if i<half_lanes {
                        if from_pick {
                            let st=z as *const f32;
                            core::array::from_fn::<f32,16,_>(|j|*st.add(i*16+j))
                        } else {[0.0f32;16]}
                    } else {
                        let tok=if from_pick {
                            let (id,flag)=(*y.add(1),*y.add(2));
                            if flag==1.0 && id>=0.0 && id<b as f32 && (id as u32) as f32==id && (id as usize)<b {id as usize} else {usize::MAX}
                        } else {c};
                        if tok<b {
                            let w1=(x as *const u16).add(tok*a+(i-half_lanes)*16);
                            core::array::from_fn::<f32,16,_>(|j|f32::from_bits(u32::from(*w1.add(j))<<16))
                        } else {[0.0f32;16]}
                    };
                    (vals,0usize,i)
                } else {
                    let lanes_per_chat=(2*a)/16;
                    let (bi,li)=(i/lanes_per_chat,i%lanes_per_chat);
                    let half_lanes=a/16;
                    let vals=if li<half_lanes {
                        if from_pick {
                            let st=(z as *const f32).add(bi*a+li*16);
                            core::array::from_fn::<f32,16,_>(|j|*st.add(j))
                        } else {[0.0f32;16]}
                    } else {
                        let tok=if from_pick {
                            let p=y.add(bi*c);
                            let (id,flag)=(*p.add(1),*p.add(2));
                            if flag==1.0 && id>=0.0 && id<b as f32 && (id as u32) as f32==id && (id as usize)<b {id as usize} else {usize::MAX}
                        } else {
                            let id=*y.add(bi);
                            if id>=0.0 && id<b as f32 && (id as u32) as f32==id && (id as usize)<b {id as usize} else {usize::MAX}
                        };
                        if tok<b {
                            let w1=(x as *const u16).add(tok*a+(li-half_lanes)*16);
                            core::array::from_fn::<f32,16,_>(|j|f32::from_bits(u32::from(*w1.add(j))<<16))
                        } else {[0.0f32;16]}
                    };
                    (vals,0usize,i)
                }
            } else {
                // Phase 1: all 128 threads in the block run the rounded GRU recurrence in parallel (2 elems/thread
                // for rank=256) and stage `latent` in the L1-hot tail `out[batch*a + batch*tile_words + g ..]`.
                let has_state=b!=0;
                let gated=c!=0;
                let batch=d.max(1);
                let tile_words=(a/64)*18;
                let dst=out as *mut f32;
                if gated && *x.add(3*a)!=0.0 {
                    if i==0 {*dst.add(a+tile_words)=1.0;}
                    return;
                }
                let blocks_per_chat=a.div_ceil(256);
                let block=core::arch::nvptx::_block_idx_x() as usize;
                let bi=if batch==1 {0} else {block/blocks_per_chat};
                let local_block=if batch==1 {block} else {block%blocks_per_chat};
                let tid=core::arch::nvptx::_thread_idx_x() as usize;
                let x_b=x.add(bi*3*a);
                let y_b=y.add(bi*3*a);
                let st_b=(z as *const f32).add(bi*a);
                let dst_st=dst.add(bi*a);
                let lat_off=batch*a+batch*tile_words+usize::from(gated);
                let dst_lat=dst.add(lat_off+bi*a);
                let blk_start=local_block*256;
                let blk_len=a.saturating_sub(blk_start).min(256);
                let mut off=tid;
                while off<blk_len {
                    let k=blk_start+off;
                    let p0=round_checked(*x_b.add(k)+*y_b.add(k));
                    let g=rn(sigmoid(p0));
                    let prev=if has_state {*st_b.add(k)} else {0.0};
                    let term0=rn(g*prev);
                    let one_minus_g=rn(1.0-g);
                    let p1=round_checked(*x_b.add(a+k)+*y_b.add(a+k));
                    let ns=rn(term0+rn(one_minus_g*rn(tanh(p1))));
                    *dst_st.add(k)=ns;
                    let p2=round_checked(*x_b.add(2*a+k)+*y_b.add(2*a+k));
                    *dst_lat.add(k)=if p0.is_finite() && p1.is_finite() && p2.is_finite() && ns.is_finite() {rn(tanh(p2))} else {f32::NAN};
                    off+=128;
                }
                if gated && i==0 {*dst.add(a+tile_words)=0.0;}
                core::arch::nvptx::_syncthreads();
                if tid>=blk_len/16 {return;}
                let local_pack_lane=blk_start/16+tid;
                let src=(out as *const f32).add(lat_off+bi*a+local_pack_lane*16);
                (core::array::from_fn::<f32,16,_>(|j|*src.add(j)),batch*a,bi*(a/16)+local_pack_lane)
            };
            let group=pack_i%4;
            let tile=(out as *mut u32).add(tile_off+pack_i/4*18);
            for pass in 0..2 {
                let max=vals.iter().fold(0.0f32,|v,x|v.max(x.abs()));
                let scale:u32;
                asm!("{{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, {z}, {v}; cvt.u32.u16 {d}, t; }}",
                    d=out(reg32) scale,z=in(reg32) 0.0f32,v=in(reg32) max/6.0);
                let scale=if vals.iter().all(|v|v.is_finite()) {(scale&255).max(u32::from(max>0.0))} else {127};
                *tile.add(16+pass).cast::<u8>().add(group)=scale as u8;
                let unit:f32;
                asm!("{{ .reg .b16 t; .reg .b32 h; .reg .b16 lo,hi; cvt.u16.u32 t,{s}; cvt.rn.f16x2.e4m3x2 h,t; mov.b32 {{lo,hi}},h; cvt.f32.f16 {d},lo; }}",
                    d=out(reg32) unit,s=in(reg32) scale);
                let inv=if unit>0.0 {1.0/unit} else {0.0};
                for word in 0..2 {
                    let mut bits=0u32;
                    for pair in 0..4 {
                        let j=word*8+pair*2;
                        let byte:u32;
                        asm!("{{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t,{hi},{lo}; cvt.u32.u8 {d},t; }}",
                            d=out(reg32) byte,hi=in(reg32) vals[j+1]*inv,lo=in(reg32) vals[j]*inv);
                        bits|=byte<<(pair*8);
                        if pass==0 {
                            let halfs:u32;
                            asm!("{{ .reg .b8 t; cvt.u8.u32 t,{s}; cvt.rn.f16x2.e2m1x2 {d},t; }}",d=out(reg32) halfs,s=in(reg32) byte);
                            let (lo,hi):(f32,f32);
                            asm!("{{ .reg .b16 a,b; mov.b32 {{a,b}},{s}; cvt.f32.f16 {lo},a; cvt.f32.f16 {hi},b; }}",
                                lo=out(reg32) lo,hi=out(reg32) hi,s=in(reg32) halfs);
                            vals[j]-=lo*unit;vals[j+1]-=hi*unit;
                        }
                    }
                    *tile.add(pass*8+group*2+word)=bits;
                }
            }
            return;
        }
        // SAFETY: operation 20 runs whole warps (`n == rows * heads * splits * 32`, `width == 128`). The host checked
        // contiguous 16-byte aligned spans for `q` (`x`), `k` (`y`), `v` (`z`) and `out`. Each warp owns one
        // `(row, head, split)`, grouped by `(kv_head, split)` so warps in a block reuse K/V in L1. All 32 lanes of a
        // warp run the exact same key range and butterfly shuffles (no warp divergence).
        if kind==20 {
            let rows=d&0xffff;
            let splits=(d>>16)&0xff;
            let row_off=(d>>24)&0xff;
            let causal_base=d>>32;
            let (warp,lane)=(i/32,i%32);
            let group=b/c;
            let per_split=rows*group;
            let per_kv=splits*per_split;
            let (kh,rem0)=(warp/per_kv,warp%per_kv);
            let (split,rem)=(rem0/per_split,rem0%per_split);
            let (row,sub)=(rem/group,rem%group);
            let head=kh*group+sub;
            let q_idx=(row*b+head)*128+lane*4;
            let q=crate::tc_ptx::ld_global_nc_v4(x.add(q_idx));
            let scale=f32::from_bits(a as u32);
            let out_base=if splits==1 {q_idx} else {((row*b+head)*splits+split)*132};
            let row_keys=if causal_base>0 {
                let r_abs=row+row_off;
                if r_abs<8 {causal_base} else if r_abs<12 {(causal_base+4).min(e)} else {e}
            } else {e};
            let chunk=e.div_ceil(splits);
            let start=split*chunk;
            let end=(start+chunk).min(row_keys);
            let kv_stride=c*128;
            let mut k_ptr=y.add((start*c+kh)*128+lane*4);
            let mut v_ptr=(z as *const f32).add((start*c+kh)*128+lane*4);
            let mut m=f32::NEG_INFINITY;
            let mut l=0.0f32;
            let mut o=[0.0f32;4];
            // The walk over keys is serial (online softmax), so without a prefetch each key waits on its own DRAM
            // read of K and then V. Each lane asks L1 for its 16 bytes of K and V `ATTN_PREFETCH_KEYS` keys ahead
            // (keys below `end` only, inside the checked spans); a prefetch moves no value, so every bit is unchanged.
            let ahead=ATTN_PREFETCH_KEYS*kv_stride;
            // The warps of a block that share `(kv_head, split)` walk the same K and V rows in step, and L1 does not
            // join their misses on one sector: each prefetch of each warp is its own L2 read (measured: 39 MB of L2
            // reads for 1.2 MB of K and V). Only the first such warp of the block prefetches; the others find the
            // rows in L1. With block-causal rows the warps of a block end at different keys, so each one prefetches.
            let lead=causal_base>0 || warp%4==0 || (warp-warp%4)/per_split!=warp/per_split;
            let mut warm=start;
            while lead && warm<end && warm<start+ATTN_PREFETCH_KEYS {
                let at=(warm-start)*kv_stride;
                core::arch::asm!("prefetch.global.L1 [{k}];","prefetch.global.L1 [{v}];",k=in(reg64) k_ptr.add(at),v=in(reg64) v_ptr.add(at),options(readonly,nostack));
                warm+=1;
            }
            let mut step=start;
            while step<end {
                if lead && step+ATTN_PREFETCH_KEYS<end {
                    core::arch::asm!("prefetch.global.L1 [{k}];","prefetch.global.L1 [{v}];",k=in(reg64) k_ptr.add(ahead),v=in(reg64) v_ptr.add(ahead),options(readonly,nostack));
                }
                let k0=crate::tc_ptx::ld_global_nc_v4(k_ptr);
                let mut dot=q[0]*k0[0]+q[1]*k0[1]+q[2]*k0[2]+q[3]*k0[3];
                dot+=shfl_xor_f32(dot,16);
                dot+=shfl_xor_f32(dot,8);
                dot+=shfl_xor_f32(dot,4);
                dot+=shfl_xor_f32(dot,2);
                dot+=shfl_xor_f32(dot,1);
                let score=dot*scale;
                let v0=crate::tc_ptx::ld_global_nc_v4(v_ptr);
                let p=if score>m {
                    let alpha=if l==0.0 {0.0} else {exp(m-score)};
                    l*=alpha;
                    o[0]*=alpha;o[1]*=alpha;o[2]*=alpha;o[3]*=alpha;
                    m=score;
                    1.0f32
                } else {
                    exp(score-m)
                };
                l+=p;
                o[0]+=p*v0[0];o[1]+=p*v0[1];o[2]+=p*v0[2];o[3]+=p*v0[3];
                k_ptr=k_ptr.add(kv_stride);
                v_ptr=v_ptr.add(kv_stride);
                step+=1;
            }
            if splits==1 {
                let inv=1.0/l;
                let dst=(out as *mut f32).add(out_base);
                *dst=rn(o[0]*inv);*dst.add(1)=rn(o[1]*inv);*dst.add(2)=rn(o[2]*inv);*dst.add(3)=rn(o[3]*inv);
            } else {
                let dst=(out as *mut f32).add(out_base+lane*4);
                *dst=o[0];*dst.add(1)=o[1];*dst.add(2)=o[2];*dst.add(3)=o[3];
                if lane==0 {
                    *(out as *mut f32).add(out_base+128)=m;
                    *(out as *mut f32).add(out_base+129)=l;
                }
            }
            return;
        }
        // SAFETY: operation 21 runs whole warps (`n == rows * heads * 32`, `width == 128`). The host checked
        // 16-byte aligned spans of `rows * heads * splits * 132` words for `x` and `rows * heads * 128` for `out`.
        // Each warp merges its `splits` partial `(o, m, l)` records in split order; `ls == 0.0` is warp-uniform.
        if kind==21 {
            let (warp,lane)=(i/32,i%32);
            let (heads,splits)=(a,c);
            let mut m=f32::NEG_INFINITY;
            let mut l=0.0f32;
            let mut o=[0.0f32;4];
            let mut s=0usize;
            while s<splits {
                let base=(warp*splits+s)*132;
                let ls=*x.add(base+129);
                if ls!=0.0 {
                    let ms=*x.add(base+128);
                    let os=crate::tc_ptx::ld_global_nc_v4(x.add(base+lane*4));
                    if l==0.0 {
                        m=ms;l=ls;o=os;
                    } else if ms>m {
                        let alpha=exp(m-ms);
                        o[0]=o[0]*alpha+os[0];o[1]=o[1]*alpha+os[1];o[2]=o[2]*alpha+os[2];o[3]=o[3]*alpha+os[3];
                        l=l*alpha+ls;
                        m=ms;
                    } else {
                        let alpha=exp(ms-m);
                        o[0]+=os[0]*alpha;o[1]+=os[1]*alpha;o[2]+=os[2]*alpha;o[3]+=os[3]*alpha;
                        l+=ls*alpha;
                    }
                }
                s+=1;
            }
            let _=heads;
            let inv=1.0/l;
            let dst=(out as *mut f32).add(warp*128+lane*4);
            *dst=rn(o[0]*inv);*dst.add(1)=rn(o[1]*inv);*dst.add(2)=rn(o[2]*inv);*dst.add(3)=rn(o[3]*inv);
            return;
        }
        // SAFETY: operation 22 runs `e.max(1) * b` blocks of 128 lanes (`1 <= b <= 128`, `b <= a`). Each block
        // reduces its contiguous slice of its chat's `a` four-word tile records `[best, id, valid, sum]` using 64 bytes
        // of static shared memory (`4` warps x `4` words); all 128 lanes of each block reach every `_syncthreads()`
        // and warp shuffle. Thread 0 writes the block's four-word record to `out[(bi * b + block) * 4 .. +4]`.
        if kind==22 {
            if d!=0 && *y!=0.0 {return;}
            let blk=core::arch::nvptx::_block_idx_x() as usize;
            let (bi,block)=if e>1 {(blk/b,blk%b)} else {(0,blk)};
            let x=x.add(bi*a*4);
            let tid=core::arch::nvptx::_thread_idx_x() as usize;
            let (warp,lane)=(tid>>5,tid&31);
            let smem=crate::ptx::shared::<64>();
            let per=a.div_ceil(b);
            let start=block*per;
            let end=(start+per).min(a);
            let mut best=f32::NEG_INFINITY;let mut id=u32::MAX;let mut valid=1u32;
            let mut t=start+tid;
            while t<end {
                let p=x.add(t*4);
                let (vb,vi,vv)=(*p,*p.add(1) as u32,*p.add(2) as u32);
                valid&=vv;
                if vb>best || (vb==best && vi<id) {best=vb;id=vi;}
                t+=128;
            }
            let mut dist=16;
            while dist>0 {
                let v=shfl_xor_f32(best,dist);let other=shfl_xor_u32(id,dist);
                valid&=shfl_xor_u32(valid,dist);
                if v>best || (v==best && other<id) {best=v;id=other;}
                dist/=2;
            }
            if lane==0 {
                let off=(warp*16) as u32;
                crate::tc_ptx::store_u32(smem+off,best.to_bits());
                crate::tc_ptx::store_u32(smem+off+4,id);
                crate::tc_ptx::store_u32(smem+off+8,valid);
            }
            core::arch::nvptx::_syncthreads();
            best=f32::NEG_INFINITY;id=u32::MAX;valid=1u32;
            for w in 0..4usize {
                let off=(w*16) as u32;
                let v=f32::from_bits(crate::tc_ptx::load_u32(smem+off));
                let other=crate::tc_ptx::load_u32(smem+off+4);
                valid&=crate::tc_ptx::load_u32(smem+off+8);
                if v>best || (v==best && other<id) {best=v;id=other;}
            }
            let mut sum=0.0f32;
            if c!=0 {
                if best.is_finite() {
                    let mut t=start+tid;
                    while t<end {
                        let p=x.add(t*4);
                        sum+=*p.add(3)*exp(*p-best);
                        t+=128;
                    }
                }
                let mut dist=16;
                while dist>0 {sum+=shfl_xor_f32(sum,dist);dist/=2;}
                if lane==0 {crate::tc_ptx::store_u32(smem+(warp*16+12) as u32,sum.to_bits());}
                core::arch::nvptx::_syncthreads();
                if tid==0 {
                    sum=f32::from_bits(crate::tc_ptx::load_u32(smem+12))
                        +f32::from_bits(crate::tc_ptx::load_u32(smem+28))
                        +f32::from_bits(crate::tc_ptx::load_u32(smem+44))
                        +f32::from_bits(crate::tc_ptx::load_u32(smem+60));
                }
            }
            if tid==0 {
                let dst=(out as *mut f32).add((bi*b+block)*4);
                *dst=best;*dst.add(1)=id as f32;*dst.add(2)=valid as f32;*dst.add(3)=sum;
            }
            return;
        }
        // SAFETY: operation 23 adds `a` FP32 words (`n == a.div_ceil(4)`), four words per lane. Vectorized 16-byte
        // loads/stores run only when all three spans are 16-byte aligned and all four elements lie below `a`.
        if kind==23 {
            let base=i*4;
            if base>=a {return;}
            let dst=out as *mut f32;
            if base+4<=a && ((x as usize|y as usize|dst as usize)&15)==0 {
                let xv=crate::tc_ptx::ld_global_nc_v4(x.add(base));
                let yv=crate::tc_ptx::ld_global_nc_v4(y.add(base));
                let (r0,r1,r2,r3)=(round_checked(xv[0]+yv[0]),round_checked(xv[1]+yv[1]),round_checked(xv[2]+yv[2]),round_checked(xv[3]+yv[3]));
                core::arch::asm!(
                    "st.global.v4.f32 [{p}], {{{r0}, {r1}, {r2}, {r3}}};",
                    p=in(reg64) dst.add(base), r0=in(reg32) r0, r1=in(reg32) r1, r2=in(reg32) r2, r3=in(reg32) r3, options(nostack)
                );
            } else {
                let end=(base+4).min(a);
                let mut k=base;
                while k<end {
                    *dst.add(k)=round_checked(*x.add(k)+*y.add(k));
                    k+=1;
                }
            }
            return;
        }
        // SAFETY: `e.max(1)` blocks of 128 lanes each read their chat's complete table of `a` four-word partials and
        // write one pick and one denominator, its four warp picks in words 4..20 and, with `b`, the `a` exponentials
        // in words 20..20+a, before barriers that every lane reaches (uniform bounds).
        if kind==15 {
            let tid=core::arch::nvptx::_thread_idx_x() as usize;
            let bi=if e>1 {core::arch::nvptx::_block_idx_x() as usize} else {0};
            let (lane,warp)=(tid%32,tid/32);
            let stride=20+if b!=0 {a} else {0};
            let x=x.add(bi*a*4);
            let dst=(out as *mut f32).add(bi*stride);
            if d!=0 && *y!=0.0 {
                if tid==0 {*dst=0.0;*dst.add(1)=0.0;*dst.add(2)=0.0;*dst.add(3)=0.0;*dst.add(4)=1.0;}
                return;
            }
            let mut best=f32::NEG_INFINITY;let mut id=u32::MAX;let mut valid=1;
            // Tiles tid, tid+128, ... in pairs: both loads issue before either comparison (any order picks the same).
            let mut tile=tid;
            while tile+128<a {
                let (v,w)=(x.add(tile*4),x.add(tile*4+512));
                let (vb,vi,vv,wb,wi,wv)=(*v,*v.add(1) as u32,*v.add(2) as u32,*w,*w.add(1) as u32,*w.add(2) as u32);
                valid&=vv&wv;
                if vb>best || (vb==best && vi<id) {best=vb;id=vi;}
                if wb>best || (wb==best && wi<id) {best=wb;id=wi;}
                tile+=256;
            }
            if tile<a {
                let v=x.add(tile*4);let other=*v.add(1) as u32;
                valid&=*v.add(2) as u32;
                if *v>best || (*v==best && other<id) {best=*v;id=other;}
            }
            let mut dist=16;
            while dist>0 {
                let v=shfl_xor_f32(best,dist);let other=shfl_xor_u32(id,dist);
                valid&=shfl_xor_u32(valid,dist);
                if v>best || (v==best && other<id) {best=v;id=other;}
                dist/=2;
            }
            if lane==0 {let w=dst.add(4+warp*4);*w=best;*(w as *mut u32).add(1)=id;*(w as *mut u32).add(2)=valid;}
            core::arch::nvptx::_syncthreads();
            for w in 0..4 {
                let w=dst.add(4+w*4);let (v,other)=(*w,*(w as *const u32).add(1));
                valid&=*(w as *const u32).add(2);
                if v>best || (v==best && other<id) {best=v;id=other;}
            }
            let mut sum=0.0;
            if b!=0 {
                let terms=dst.add(20);
                let mut tile=tid;
                while tile+128<a {
                    let (u,v)=(*x.add(tile*4),*x.add(tile*4+512));
                    *terms.add(tile)=exp(u-best);*terms.add(tile+128)=exp(v-best);
                    tile+=256;
                }
                if tile<a {*terms.add(tile)=exp(*x.add(tile*4)-best);}
                core::arch::nvptx::_syncthreads();
                if warp==0 {
                    let mut tile=lane;
                    // Four loads of each kind ahead, then the same adds in the same order.
                    while tile+3*32<a {
                        let (s0,s1,s2,s3)=(*x.add(tile*4+3),*x.add(tile*4+131),*x.add(tile*4+259),*x.add(tile*4+387));
                        let (t0,t1,t2,t3)=(*terms.add(tile),*terms.add(tile+32),*terms.add(tile+64),*terms.add(tile+96));
                        sum+=s0*t0;sum+=s1*t1;sum+=s2*t2;sum+=s3*t3;
                        tile+=4*32;
                    }
                    while tile<a {sum+=*x.add(tile*4+3)**terms.add(tile);tile+=32;}
                    let mut dist=16;
                    while dist>0 {sum+=shfl_xor_f32(sum,dist);dist/=2;}
                }
            }
            if tid==0 {
                let stop=if c!=0 && (valid!=1 || !(sum>0.0 && sum.is_finite()) || sum>f32::from_bits(c as u32)) {1.0f32} else {0.0f32};
                *dst=best;*dst.add(1)=id as f32;*dst.add(2)=valid as f32;*dst.add(3)=sum;*dst.add(4)=stop;
            }
            return;
        }
        if kind==10 || kind==11 {
            let (warp,lane)=(i/32,i%32);
            let mut best=f32::NEG_INFINITY;
            let mut id=u32::MAX;
            let mut valid=1u32;
            if kind==10 {
                let begin=warp*1024;
                let mut at=begin+lane;
                while at<(begin+1024).min(a) {
                    let value=*x.add(at);valid&=u32::from(value.is_finite());
                    if value>best {best=value;id=at as u32;}
                    at+=32;
                }
            } else {
                let mut at=lane;
                while at<a {
                    let value=*x.add(at*3);let index=*x.add(at*3+1) as u32;
                    valid&=*x.add(at*3+2) as u32;
                    if value>best || (value==best && index<id) {best=value;id=index;}
                    at+=32;
                }
            }
            for distance in [16,8,4,2,1] {
                let value=shfl_xor_f32(best,distance);let index=shfl_xor_u32(id,distance);
                valid&=shfl_xor_u32(valid,distance);
                if value>best || (value==best && index<id) {best=value;id=index;}
            }
            if lane==0 {
                let out=(out as *mut f32).add(warp*3);
                *out=best;*out.add(1)=id as f32;*out.add(2)=valid as f32;
            }
            return;
        }
        let v=match kind {
            1=>{let j=i%a;let h=i/a;let m=j%(a/2);let row=h/b;
                let trig=y.add(row*a);let lo=*x.add(h*a+m);let hi=*x.add(h*a+a/2+m);
                let cos=*trig.add(m);let sin=*trig.add(a/2+m);
                rn(if j<a/2 {lo*cos-hi*sin} else {hi*cos+lo*sin})}
            2=>{let k=i%e;let q=i/e;let kh=(q%b)/(b/c);let mut dot=0.0;
                let (query,key)=(x.add(q*d),y.add((k*c+kh)*d));
                // The same chain over the width in rising order; with whole 16-byte groups each lane reads 4 words
                // of its query and its key row at a time.
                if d%4==0 && (query as usize|key as usize)&15==0 {
                    let mut j=0;
                    while j<d {
                        let (u,w)=(crate::tc_ptx::ld_global_nc_v4(query.add(j)),crate::tc_ptx::ld_global_nc_v4(key.add(j)));
                        dot+=u[0]*w[0];dot+=u[1]*w[1];dot+=u[2]*w[2];dot+=u[3]*w[3];j+=4;
                    }
                } else {for j in 0..d {dot+=*query.add(j) * *key.add(j);}}
                dot*f32::from_bits(a as u32)}
            // One warp a query row (n is 32 lanes a row, so a warp is whole or absent). The maximum takes any order;
            // every lane adds the exponentials of the row in rising key order, each received from the lane that made
            // it, so the sum is the one chain of a single lane, and each output is that exponential times 1/sum.
            3=>{let (lane,row)=(i%32,i/32);let scores=x.add(row*e);let dest=(out as *mut f32).add(row*e);
                let mut best=f32::NEG_INFINITY;let mut j=lane;
                while j<e {best=best.max(*scores.add(j));j+=32;}
                for m in [16,8,4,2,1] {best=best.max(crate::tc_ptx::shfl_xor_f32(best,m));}
                let mut sum=0.0f32;let mut first=0;
                while first<e {
                    let own=if first+lane<e {let v=exp(*scores.add(first+lane)-best);*dest.add(first+lane)=v;v} else {0.0};
                    let mut from=0u32;
                    while from<32 {
                        let v:f32;
                        core::arch::asm!("shfl.sync.idx.b32 {d}, {v}, {s}, 0x1f, 0xffffffff;",d=out(reg32) v,v=in(reg32) own,s=in(reg32) from,options(nomem,nostack));
                        if first+(from as usize)<e {sum+=v;}
                        from+=1;
                    }
                    first+=32;
                }
                let inv=1.0/sum;let mut j=lane;
                while j<e {*dest.add(j)=*dest.add(j)*inv;j+=32;}
                return;}
            4=>{let q=i/d;let kh=(q%b)/(b/c);let col=i%d;
                let mut sum=0.0;for j in 0..e {sum+=*x.add(q*e+j) * *y.add((j*c+kh)*d+col);}rn(sum)}
            5=>{let j=i%a;if i<a {let g=rn(sigmoid(*x.add(j)));
                rn(rn(g * *y.add(j))+rn(rn(1.0-g)*rn(tanh(*x.add(a+j)))))} else {rn(tanh(*x.add(2*a+j)))}}
            6=>{let mut mixed=0.0;for j in 0..b {mixed+=rn(*x.add(j*n as usize+i)) * *y.add(j);}mixed}
            7=>f32::from_bits(u32::from(*(x as *const u16).add(i))<<16),
            8=>{let (id,flag)=(*y.add(1),*y.add(2));
                if flag==1.0 && id>=0.0 && id<a as f32 && (id as u32) as f32==id && (id as usize)<a {
                    f32::from_bits(u32::from(*(x as *const u16).add(id as usize*n as usize+i))<<16)} else {0.0}}
            9=>{let z=z as *const f32;
                // `logits_bf16` on a sum: the BF16 round of a finite word, any other word kept.
                let keep=|v:f32|{let bits=v.to_bits();if bits&0x7f800000!=0x7f800000 {f32::from_bits(bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff0000)} else {v}};
                let bias=keep(*x.add(i)+*y.add(i));let sum=keep(*z.add(i)+bias);
                if bias.is_finite() && sum.is_finite() {tanh(rn(sum)*f32::from_bits(a as u32))*f32::from_bits(b as u32)} else {f32::NAN}}
            _=>return,
        };
        *(out as *mut f32).add(i)=v;
    }
}
