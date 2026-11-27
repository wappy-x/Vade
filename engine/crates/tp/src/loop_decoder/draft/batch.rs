//! Batched DSpark context push and multi-chat proposal sharing draft weights across `B >= 2` chats.
//! Every token ID and probability matches single-chat `push_values` + `propose_confident` bit for bit.
use super::{DraftChat,Drafter,Lanes,kv_row_words,need,packed,prepare_layer_kv_room};
use crate::{Result,loop_decoder::{mix::Weights,operands::weights::ResidentProduct,rows}};
use spark_gpu_probe::draft_ops::{DRAFT_BATCH_MAX,DraftOp,PANEL_INPUTS_MAX};

impl<'a,V,A> Drafter<'a,V,A> {
    /// Push committed target taps for multiple chats in batched weight passes (`k_ctx`, `v_ctx`, `Mix`, `Rope`,
    /// `ScatterKvBatch`, and `markov_norm`), chunking only at real kernel ABI row limits (`Mix` `u32` words and
    /// `FastNorm` 262,144 rows).
    pub fn push_values_batch<C:ResidentProduct<V,Value=A>+Lanes>(&self,c:&C,items:&mut [(&mut DraftChat<A>,&[A],usize)])->Result<()> {
        let active=items.iter().enumerate().filter_map(|(idx,(_,_,r))|(*r>0).then_some(idx)).collect::<Vec<_>>();
        if active.is_empty() {return Ok(());}
        let cfg=self.config();
        let (h,hd,heads,n_taps)=(cfg.hidden_size,cfg.head_dim,cfg.num_kv_heads,cfg.target_layer_ids.len());
        let max_batch_rows=(u32::MAX as usize/(n_taps.max(1).saturating_mul(h.max(1)))).min(262_144/heads.max(1)).max(1);
        let mut cursor=0usize;
        while cursor<active.len() {
            let first=active[cursor];
            let first_bf16=items[first].0.kv_bf16;
            let mut end=cursor+1;
            let mut total_rows=items[first].2;
            if total_rows<=max_batch_rows {
                while end<active.len() {
                    let next_idx=active[end];
                    let next_rows=items[next_idx].2;
                    if items[next_idx].0.kv_bf16!=first_bf16 || total_rows.saturating_add(next_rows)>max_batch_rows {break;}
                    total_rows+=next_rows;
                    end+=1;
                }
            }
            let group=&active[cursor..end];
            cursor=end;
            if group.len()==1 {
                let (chat,taps,rows)=&mut items[group[0]];
                self.push_values(c,chat,taps,*rows)?;
                continue;
            }
            self.push_values_batch_group(c,items,group,total_rows,first_bf16,h,hd,heads,n_taps)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn push_values_batch_group<C:ResidentProduct<V,Value=A>+Lanes>(
        &self,c:&C,items:&mut [(&mut DraftChat<A>,&[A],usize)],active:&[usize],total_rows:usize,
        all_bf16:bool,h:usize,hd:usize,heads:usize,n_taps:usize,
    )->Result<()> {
        let cfg=self.config();
        for &idx in active {
            let (chat,taps,rows)=&items[idx];
            let end=chat.len.checked_add(*rows).ok_or("draft context overflow")?;
            need(end<=chat.limit && taps.len()==n_taps && taps.iter().all(|t|c.words(t)==*rows*h),"draft context shape")?;
        }
        let mut tap_all=Vec::with_capacity(n_taps);
        for t in 0..n_taps {
            let refs=active.iter().map(|&idx|&items[idx].1[t]).collect::<Vec<_>>();
            tap_all.push(c.join(&refs)?);
        }
        let stacked=c.join(&tap_all.iter().collect::<Vec<_>>())?;
        let phase_parts=active.iter().map(|&idx|{let (chat,_,rows)=&items[idx];self.phases(c,chat.len,*rows)}).collect::<Result<Vec<_>>>()?;
        let phases=c.join(&phase_parts.iter().collect::<Vec<_>>())?;
        let mix=|layer:usize,weight:&A|->Result<A> {
            let x=c.lanes(DraftOp::Mix{words:total_rows*h,taps:n_taps},&stacked,Some(&self.alpha[layer]))?;
            c.draft_norm(&x,total_rows,h,cfg.rms_norm_eps,weight)
        };
        let kv_words=kv_row_words(cfg,all_bf16);
        let v_row_words=cfg.kv_width();
        for (i,l) in self.layers.iter().enumerate() {
            let w=&self.norms[i];let x=mix(i,&w.dflare)?;
            let [k_all,v_all]=l.kv_ctx(c,&x,total_rows)?;
            let k_all=c.draft_norm(&k_all,total_rows*heads,hd,cfg.rms_norm_eps,&w.k_ctx)?;
            let k_all=if all_bf16 {
                c.lanes(DraftOp::RopeBf16{rows:total_rows,heads,width:hd},&k_all,Some(&phases))?
            } else {
                c.lanes(DraftOp::Rope{rows:total_rows,heads,width:hd},&k_all,Some(&phases))?
            };
            let mut scattered=false;
            if all_bf16 {
                let mut dst_offs=Vec::with_capacity(active.len());
                let mut all_fit=true;
                for &idx in active {
                    let (chat,_,rows)=&mut items[idx];
                    match prepare_layer_kv_room(cfg,c,chat,i,*rows)? {
                        Some(rel)=>dst_offs.push(rel),
                        None=>{all_fit=false;break;}
                    }
                }
                if all_fit {
                    let mut plane_refs:Vec<(Option<&mut (A,A)>,usize)>=items.iter_mut().map(|(ch,_,rows)|(ch.kv[i].as_mut(),*rows)).collect();
                    let mut chunk_row_off=0usize;
                    let mut all_ok=true;
                    for (chunk_idx,chunk_dst) in active.chunks(DRAFT_BATCH_MAX).zip(dst_offs.chunks(DRAFT_BATCH_MAX)) {
                        let mut scatter_chats=Vec::with_capacity(chunk_idx.len());
                        let mut local_off=0usize;
                        for (&idx,&dst_row) in chunk_idx.iter().zip(chunk_dst) {
                            let (plane,rows)=&mut plane_refs[idx];
                            let (dk,dv)=plane.take().ok_or("missing draft cache")?;
                            scatter_chats.push((dk,dv,dst_row,local_off,*rows));
                            local_off+=*rows;
                        }
                        let k_chunk=if chunk_idx.len()==active.len() {None} else {Some(c.range(&k_all,chunk_row_off*kv_words,local_off*kv_words)?)};
                        let v_chunk=if chunk_idx.len()==active.len() {None} else {Some(c.range(&v_all,chunk_row_off*v_row_words,local_off*v_row_words)?)};
                        let ok=c.scatter_kv_batch(k_chunk.as_ref().unwrap_or(&k_all),v_chunk.as_ref().unwrap_or(&v_all),local_off,kv_words,true,&mut scatter_chats)?;
                        if !ok {all_ok=false;break;}
                        chunk_row_off+=local_off;
                    }
                    scattered=all_ok;
                }
            }
            if !scattered {
                let v_packed=if all_bf16 {Some(c.lanes(DraftOp::PackBf16(c.words(&v_all)),&v_all,None)?)} else {None};
                let v_src=v_packed.as_ref().unwrap_or(&v_all);
                let mut off=0usize;
                for &idx in active {
                    let (chat,_,rows)=&mut items[idx];
                    let k_slice=c.range(&k_all,off*kv_words,*rows*kv_words)?;
                    let v_slice=c.range(v_src,off*kv_words,*rows*kv_words)?;
                    self.push_layer_kv(c,chat,i,&k_slice,&v_slice,*rows)?;
                    off+=*rows;
                }
            }
        }
        // Only each chat's final committed row becomes its next recurrent Markov feature.
        let mut last_taps=Vec::with_capacity(n_taps);
        for t in 0..n_taps {
            let mut per_chat=Vec::with_capacity(active.len());
            for &idx in active {
                let (_,taps,rows)=&items[idx];
                per_chat.push(c.range(&taps[t],(*rows-1)*h,h)?);
            }
            last_taps.push(c.join(&per_chat.iter().collect::<Vec<_>>())?);
        }
        let tail=c.join(&last_taps.iter().collect::<Vec<_>>())?;
        let feat_x=c.lanes(DraftOp::Mix{words:active.len()*h,taps:n_taps},&tail,Some(&self.alpha[cfg.num_layers]))?;
        let feats=c.draft_norm(&feat_x,active.len(),h,cfg.rms_norm_eps,&self.markov_norm)?;
        for (pos,&idx) in active.iter().enumerate() {
            let (chat,_,rows)=&mut items[idx];
            chat.feature=Some(c.range(&feats,pos*h,h)?);
            chat.len+=*rows;
        }
        Ok(())
    }

    /// Batched DSpark proposal across `B` chats (`items = [(chat, anchor, embed, depth)]`), sharing all draft layer,
    /// head, RNN, and Markov fold weight reads. Produces bit-exact token IDs and top-1 probabilities matching
    /// single-chat `propose_confident`. `before_read` runs exactly once: on the batched Markov path after the last
    /// draft launch is queued and before the one picks read, on every other path first.
    pub fn propose_confident_batch<C:ResidentProduct<V,Value=A>+Lanes,W:Weights<C>>(
        &self,c:&C,w:&W,items:&mut [(&mut DraftChat<A>,u32,&A,usize)],before_read:&mut dyn FnMut()->Result<()>,
    )->Result<Vec<(Vec<u32>,Vec<f32>)>> {
        if items.is_empty() {return before_read().map(|()|Vec::new());}
        if items.len()==1 {
            before_read()?;
            let (chat,anchor,embed,depth)=&mut items[0];
            return Ok(vec![self.propose_confident(c,w,chat,*anchor,embed,*depth,0.0)?]);
        }
        let cfg=self.config();
        let (rows,h,hd,heads,kvheads,kv)=(cfg.block_size,cfg.hidden_size,cfg.head_dim,cfg.num_heads,cfg.num_kv_heads,cfg.kv_width());
        let b_count=items.len();
        let max_depth=items.iter().map(|(_,_,_,d)|*d).max().unwrap_or(0);
        for (chat,anchor,embed,depth) in items.iter() {
            need(*depth>0 && *depth<rows && c.words(embed)==h && chat.len+rows<=cfg.max_position_embeddings,"draft block shape")?;
            need((*anchor as usize)<cfg.vocab_size,"draft ID outside embedding")?;
            need(chat.feature.is_some(),"draft context empty")?;
        }
        let use_batched_markov=b_count<=DRAFT_BATCH_MAX
            && self.fold_bias.is_none()
            && self.rnn_split.is_some()
            && matches!(self.head,Some(packed::Packed::Panel{..}));
        let q_final:Vec<usize>=items.iter().map(|(_,_,_,d)|if use_batched_markov {max_depth} else {*d}).collect();
        // Pre-upload anchor IDs and cached swap table BEFORE any draft layer launches so the resident launch queue
        // never flushes mid-draft between the 3 transformer layers and the Markov head/fold walk.
        let (pre_anchors,pre_swap)=if use_batched_markov {
            let anchors=c.upload(&items.iter().map(|(_,a,_,_)|*a as f32).collect::<Vec<_>>())?;
            let swap_tbl=if max_depth>1 {Some(c.upload_constant(&rows::swap_table(b_count,max_depth))?)} else {None};
            (Some(anchors),swap_tbl)
        } else {(None,None)};
        let mut res_parts=Vec::with_capacity(b_count*2);
        for (_,_,embed,_) in items.iter() {res_parts.push(*embed);res_parts.push(&self.mask);}
        let mut res=c.join(&res_parts)?;
        let phase_parts=items.iter().map(|(chat,_,_,_)|self.phases(c,chat.len,rows)).collect::<Result<Vec<_>>>()?;
        let phases=c.join(&phase_parts.iter().collect::<Vec<_>>())?;
        let norm=|x:&A,lines,width,weight:&A|c.draft_norm(x,lines,width,cfg.rms_norm_eps,weight);
        let flash_attn=hd==128 && self.head.is_some();
        let total_in=b_count*rows;
        let all_same_bf16=items.iter().all(|(ch,_,_,_)|ch.kv_bf16==items[0].0.kv_bf16);
        let batch_kv_bf16=items[0].0.kv_bf16;
        for (i,l) in self.layers.iter().enumerate() {
            let last_layer=i+1==self.layers.len();
            let weights=&self.norms[i];
            let x=norm(&res,total_in,h,&weights.input)?;
            let [q_all,k_all,v_all]=l.qkv(c,&x,total_in)?;
            let k_all=norm(&k_all,total_in*kvheads,hd,&weights.k)?;
            let q_counts:Vec<usize>=if last_layer {q_final.clone()} else {vec![rows;b_count]};
            let q_total:usize=q_counts.iter().sum();
            let all_uniform_q=q_counts.iter().all(|&q|q==q_counts[0]);
            let use_batched_attn=flash_attn && all_uniform_q && all_same_bf16;
            let k_all=if use_batched_attn {
                c.lanes(DraftOp::RopeBf16{rows:total_in,heads:kvheads,width:hd},&k_all,Some(&phases))?
            } else {
                c.lanes(DraftOp::Rope{rows:total_in,heads:kvheads,width:hd},&k_all,Some(&phases))?
            };
            let (q_in,q_phases,res_in)=if last_layer {
                let (mut qp,mut pp,mut rp)=(Vec::with_capacity(b_count),Vec::with_capacity(b_count),Vec::with_capacity(b_count));
                for (b,&d) in q_counts.iter().enumerate() {
                    let row_off=b*rows+1;
                    qp.push(c.range(&q_all,row_off*heads*hd,d*heads*hd)?);
                    pp.push(c.range(&phases,row_off*hd,d*hd)?);
                    rp.push(c.range(&res,row_off*h,d*h)?);
                }
                (c.join(&qp.iter().collect::<Vec<_>>())?,c.join(&pp.iter().collect::<Vec<_>>())?,c.join(&rp.iter().collect::<Vec<_>>())?)
            } else {
                (q_all,c.range(&phases,0,total_in*hd)?,res)
            };
            let q_all=norm(&q_in,q_total*heads,hd,&weights.q)?;
            let q_all=c.lanes(DraftOp::Rope{rows:q_total,heads,width:hd},&q_all,Some(&q_phases))?;
            let a_all=if use_batched_attn {
                let q_rows=q_counts[0];
                let k_new_row=kv/2;
                let mut chunks_out:Vec<A>=Vec::with_capacity(b_count.div_ceil(32));
                for start_b in (0..b_count).step_by(32) {
                    let cnt=(b_count-start_b).min(32);
                    let q_c=if start_b==0 && cnt==b_count {c.range(&q_all,0,q_total*heads*hd)?} else {c.range(&q_all,start_b*q_rows*heads*hd,cnt*q_rows*heads*hd)?};
                    let k_c=if start_b==0 && cnt==b_count {c.range(&k_all,0,total_in*k_new_row)?} else {c.range(&k_all,start_b*rows*k_new_row,cnt*rows*k_new_row)?};
                    let v_c=if start_b==0 && cnt==b_count {c.range(&v_all,0,total_in*kv)?} else {c.range(&v_all,start_b*rows*kv,cnt*rows*kv)?};
                    let mut attn_chats=Vec::with_capacity(cnt);
                    let mut splits_vec=Vec::with_capacity(cnt);
                    let mut max_splits=1usize;
                    for (chat,_,_,_) in &items[start_b..start_b+cnt] {
                        let start=if cfg.is_local(i) {chat.len.saturating_sub(cfg.sliding_window_size.saturating_sub(1))} else {0};
                        let base=chat.local_base[i];
                        need(base<=start,"draft local window base passed read start")?;
                        let rel_start=start-base;
                        let hist_keys=chat.len-start;
                        let keys=hist_keys+rows;
                        let splits_b=if keys>=256 {16} else if keys>=64 {8} else {1};
                        max_splits=max_splits.max(splits_b);
                        let (ck,cv)=chat.kv[i].as_ref().ok_or("draft keys missing")?;
                        attn_chats.push((ck,cv,rel_start,hist_keys,splits_b));
                        splits_vec.push(splits_b);
                    }
                    let part=c.flash_attn_batch(&q_c,&k_c,&v_c,&attn_chats,heads,kvheads,hd,q_rows,rows,batch_kv_bf16,true,0,0)?;
                    let a_c=if max_splits>1 {c.flash_attn_merge_batch(&part,&splits_vec,heads,q_rows,max_splits)?} else {part};
                    chunks_out.push(a_c);
                }
                if chunks_out.len()==1 {chunks_out.pop().unwrap()} else {c.join(&chunks_out.iter().collect::<Vec<_>>())?}
            } else {
                let mut a_parts=Vec::with_capacity(b_count);
                let mut q_off=0usize;
                for (b,(chat,_,_,_)) in items.iter_mut().enumerate() {
                    let q_rows=q_counts[b];
                    let q_b=c.range(&q_all,q_off*heads*hd,q_rows*heads*hd)?;
                    let k_new=c.range(&k_all,b*rows*kv,rows*kv)?;
                    let v_new=c.range(&v_all,b*rows*kv,rows*kv)?;
                    let start=if cfg.is_local(i) {chat.len.saturating_sub(cfg.sliding_window_size.saturating_sub(1))} else {0};
                    let base=chat.local_base[i];
                    need(base<=start,"draft local window base passed read start")?;
                    let (rel_start,rel_pos)=(start-base,chat.len-base);
                    let keys=chat.len-start+rows;
                    let (ck,cv)=chat.kv[i].as_mut().ok_or("draft keys missing")?;
                    let (k_ctx,v_ctx)=if c.words(ck)/kv>=rel_pos+rows {
                        c.state_fill(ck,rel_pos*kv,&k_new)?;c.state_fill(cv,rel_pos*kv,&v_new)?;
                        (c.range(ck,rel_start*kv,keys*kv)?,c.range(cv,rel_start*kv,keys*kv)?)
                    } else {
                        let append=|ctx:&A,new:&A|c.join(&[&c.range(ctx,rel_start*kv,(chat.len-start)*kv)?,new]);
                        (append(ck,&k_new)?,append(cv,&v_new)?)
                    };
                    let a_b=if flash_attn {
                        let splits=if keys>=256 {16} else if keys>=64 {8} else {1};
                        let part=c.lanes3(DraftOp::FlashAttn{rows:q_rows,heads,kv:kvheads,width:hd,keys,splits,causal_base:0,row_off:0},&q_b,&k_ctx,&v_ctx)?;
                        if splits>1 {c.lanes(DraftOp::FlashAttnMerge{rows:q_rows,heads,width:hd,splits},&part,None)?} else {part}
                    } else {
                        let scores=c.lanes(DraftOp::Scores{rows:q_rows,heads,kv:kvheads,width:hd,keys},&q_b,Some(&k_ctx))?;
                        let probs=c.lanes(DraftOp::Normalizer{queries:q_rows*heads,keys},&scores,None)?;
                        c.lanes(DraftOp::Values{rows:q_rows,heads,kv:kvheads,width:hd,keys},&probs,Some(&v_ctx))?
                    };
                    a_parts.push(a_b);
                    q_off+=q_rows;
                }
                c.join(&a_parts.iter().collect::<Vec<_>>())?
            };
            res=c.draft_sum(&res_in,&l.o.rounded(c,&a_all,q_total)?)?;
            let y=norm(&res,q_total,h,&weights.post)?;
            let [gate,up]=l.gate_up(c,&y,q_total)?;
            let act=c.lanes(DraftOp::Gate(c.words(&gate)),&gate,Some(&up))?;
            res=c.draft_sum(&res,&l.down.rounded(c,&act,q_total)?)?;
        }
        let q_total:usize=q_final.iter().sum();
        let drafted_in=if self.layers.is_empty() {
            let parts=q_final.iter().enumerate().map(|(b,&d)|c.range(&res,(b*rows+1)*h,d*h)).collect::<Result<Vec<_>>>()?;
            c.join(&parts.iter().collect::<Vec<_>>())?
        } else {res};
        let drafted_all=norm(&drafted_in,q_total,h,&self.final_norm)?;
        if let Some(anchors)=pre_anchors {
            return self.markov_walk_batch(c,&drafted_all,items,max_depth,&anchors,pre_swap.as_ref(),before_read);
        }
        before_read()?;
        let mut out=Vec::with_capacity(b_count);
        let mut off=0usize;
        for (b,(chat,anchor,_,depth)) in items.iter().enumerate() {
            let drafted_b=c.range(&drafted_all,off*h,q_final[b]*h)?;
            let feature=chat.feature.as_ref().ok_or("draft context empty")?;
            out.push(self.markov_walk(c,w,&drafted_b,feature,*anchor,*depth,true,0.0)?);
            off+=q_final[b];
        }
        Ok(out)
    }

    /// Batched Markov walk across `B <= DRAFT_BATCH_MAX` chats sharing one `head`, `draft_part`, `feat_part`,
    /// `dyn_part`, and `fold` panel read per step. Reads back all `max_depth * B` picks and softmax denominators
    /// in ONE host sync with zero mid-draft H2D uploads; `before_read` runs just before that sync.
    #[allow(clippy::too_many_arguments)]
    fn markov_walk_batch<C:ResidentProduct<V,Value=A>+Lanes>(
        &self,c:&C,drafted_all:&A,items:&[(&mut DraftChat<A>,u32,&A,usize)],max_depth:usize,anchors:&A,swap_tbl:Option<&A>,
        before_read:&mut dyn FnMut()->Result<()>,
    )->Result<Vec<(Vec<u32>,Vec<f32>)>> {
        let cfg=self.config();
        let (b_count,h,rank,vocab)=(items.len(),cfg.hidden_size,cfg.markov_rank,cfg.unpadded_vocab_size);
        let Some(packed::Packed::Panel{weight:head_weight,..})=&self.head else {return Err("loop draft: batched markov requires packed head".into());};
        let split=self.rnn_split.as_ref().ok_or("loop draft: batched markov requires split rnn")?;
        let (bound,epsilon)=(cfg.final_logit_softcapping as f32,engine_model::norm_cap::reciprocal(cfg.final_logit_softcapping));
        // Reorder `[B, max_depth, h]` -> `[max_depth, B, h]` so each Markov step `s` sees all `B` chats contiguously.
        let drafted_step=if max_depth==1 {
            c.range(drafted_all,0,b_count*h)?
        } else if let Some(tbl)=swap_tbl {
            rows::gather_table(c,drafted_all,tbl,b_count*max_depth,h)?
        } else {
            rows::swap(c,drafted_all,b_count,max_depth,h)?
        };
        let mut df_parts=Vec::with_capacity(1+b_count);
        df_parts.push(&drafted_step);
        for (chat,_,_,_) in items {df_parts.push(chat.feature.as_ref().ok_or("draft context empty")?);}
        let df=c.join(&df_parts)?;
        let packed_df=c.lanes(DraftOp::PackActivations(c.words(&df)),&df,None)?;
        let h_packed=(h/64)*18;
        let total_draft_rows=max_depth*b_count;
        let packed_drafted=c.range(&packed_df,0,total_draft_rows*h_packed)?;
        let packed_feats=c.range(&packed_df,total_draft_rows*h_packed,b_count*h_packed)?;
        // Project all `max_depth * B` rows in one launch when `<= PANEL_INPUTS_MAX`, else in `PANEL_INPUTS_MAX` slabs.
        let panel_chunked=|rows:usize,weight:&V,packed_x:&A,total:usize,bf16_out:bool|->Result<A> {
            let make_op=|inputs:usize| if bf16_out {
                DraftOp::PanelBf16{rows,columns:h,inputs}
            } else {
                DraftOp::Panel{rows,columns:h,inputs}
            };
            if total<=PANEL_INPUTS_MAX {
                return c.draft_panel(make_op(total),weight,packed_x,None);
            }
            let mut parts=Vec::with_capacity(total.div_ceil(PANEL_INPUTS_MAX));
            for start in (0..total).step_by(PANEL_INPUTS_MAX) {
                let count=(total-start).min(PANEL_INPUTS_MAX);
                let slice=c.range(packed_x,start*h_packed,count*h_packed)?;
                parts.push(c.draft_panel(make_op(count),weight,&slice,None)?);
            }
            c.join(&parts.iter().collect::<Vec<_>>())
        };
        let head_all=panel_chunked(vocab,head_weight,&packed_drafted,total_draft_rows,true)?;
        let draft_out=panel_chunked(3*rank,&split.draft_part,&packed_drafted,total_draft_rows,false)?;
        let feat_out=c.draft_panel(DraftOp::Panel{rows:3*rank,columns:h,inputs:b_count},&split.feat_part,&packed_feats,None)?;
        let static_base=c.lanes3(DraftOp::MarkovStaticBaseBatch{batch:b_count,depth:max_depth,width:3*rank},&draft_out,&feat_out,&self.rnn_bias)?;
        let tile_count=vocab.div_ceil(16);
        let tile_words=(rank/64)*18;
        let base_row_words=vocab.div_ceil(2);
        let mut state:Option<A>=None;
        let mut prev_result:Option<A>=None;
        let mut pick_stride=0usize;
        let mut step_picks:Vec<A>=Vec::with_capacity(max_depth*b_count);
        for s in 0..max_depth {
            let packed_dyn=match prev_result.as_ref() {
                None=>c.lanes(DraftOp::MarkovPrepBatch{batch:b_count,latent:rank,vocab:cfg.vocab_size,from_pick:false,pick_stride:0},&self.w1,Some(anchors))?,
                Some(prev)=>c.lanes3(DraftOp::MarkovPrepBatch{batch:b_count,latent:rank,vocab:cfg.vocab_size,from_pick:true,pick_stride},&self.w1,prev,state.as_ref().ok_or("missing markov state")?)?,
            };
            let dyn_out=c.draft_panel(DraftOp::Panel{rows:3*rank,columns:2*rank,inputs:b_count},&split.dyn_part,&packed_dyn,None)?;
            let base_s=c.range(&static_base,s*b_count*3*rank,b_count*3*rank)?;
            let stepped=match state.as_ref() {
                None=>c.lanes(DraftOp::MarkovStepBatch{batch:b_count,latent:rank,has_state:false},&dyn_out,Some(&base_s))?,
                Some(st)=>c.lanes3(DraftOp::MarkovStepBatch{batch:b_count,latent:rank,has_state:true},&dyn_out,&base_s,st)?,
            };
            state=Some(c.range(&stepped,0,b_count*rank)?);
            let packed_latent=c.range(&stepped,b_count*rank,b_count*tile_words)?;
            let head_s=c.range(&head_all,s*b_count*base_row_words,b_count*base_row_words)?;
            let tiles=c.draft_panel(DraftOp::FoldPickBatch{rows:vocab,columns:rank,batch:b_count,epsilon,bound,confidence:true,bf16_base:true},&self.fold,&packed_latent,Some(&head_s))?;
            let (result,stride)=if tile_count>128 {
                let blocks=tile_count.div_ceil(256).clamp(1,48).min(tile_count);
                let reduced=c.lanes(DraftOp::FoldReduceBatch{batch:b_count,tiles:tile_count,blocks,confidence:true},&tiles,None)?;
                let finished=c.lanes(DraftOp::FoldFinishBatch{batch:b_count,tiles:blocks,confidence:true},&reduced,None)?;
                (finished,20+blocks)
            } else {
                let finished=c.lanes(DraftOp::FoldFinishBatch{batch:b_count,tiles:tile_count,confidence:true},&tiles,None)?;
                (finished,20+tile_count)
            };
            for b in 0..b_count {step_picks.push(c.range(&result,b*stride,4)?);}
            pick_stride=stride;
            prev_result=Some(result);
        }
        let picks=c.join(&step_picks.iter().collect::<Vec<_>>())?;
        before_read()?;
        let words=c.read(&picks)?;
        need(words.len()==max_depth*b_count*4,"missing batched device draft pick")?;
        let mut out=Vec::with_capacity(b_count);
        for (b,&(_,_,_,depth)) in items.iter().enumerate() {
            let (mut ids,mut probs)=(Vec::with_capacity(depth),Vec::with_capacity(depth));
            for s in 0..depth {
                let row=&words[(s*b_count+b)*4..(s*b_count+b)*4+4];
                need(row[2]==1.0 && row[0].is_finite() && row[1].is_finite(),"nonfinite draft logits")?;
                need(row[1]>=0.0 && row[1].fract()==0.0 && (row[1] as usize)<cfg.vocab_size,"draft ID outside embedding")?;
                ids.push(row[1] as u32);
                let sum=f64::from(row[3]);
                probs.push(if sum>0.0 && sum.is_finite() {(1.0/sum).clamp(0.0,1.0) as f32} else {0.0});
            }
            out.push((ids,probs));
        }
        Ok(out)
    }
}
