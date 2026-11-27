//! One cold prompt per chat. Mixers keep their own plans and state.
use super::*;

pub(crate) struct PromptChat<'a, 'f, C: Compute> {
    pub decoder: &'a Decoder<'f, C>,
    pub state: &'a State<C::Value>,
    pub tokens: &'a [u32],
}

pub(crate) struct PromptResult<V> {
    pub logits: Vec<f32>,
    pub state: State<V>,
    pub tail: Option<OpenRows<V>>,
}

impl<C: Compute> Decoder<'_, C> {
    pub(crate) fn prefill_cohort(chats: &[PromptChat<'_, '_, C>]) -> Result<Vec<PromptResult<C::Value>>> {
        let first = chats.first().ok_or("prompt cohort has no chats")?.decoder;
        let paused = first.compute.rows_unrecorded(true)?;
        let result = first.prefill_cohort_body(chats);
        let restored = first.compute.rows_unrecorded(paused);
        let output = result?;
        restored?;
        Ok(output)
    }

    fn prefill_cohort_body(&self, chats: &[PromptChat<'_, '_, C>]) -> Result<Vec<PromptResult<C::Value>>> {
        let d = self.checkpoint.description();
        let total = chats.iter().map(|chat| chat.tokens.len()).sum::<usize>();
        need((2..=super::super::BATCH_ROWS).contains(&chats.len()) && total <= PROMPT_GROUP_MAX, "prompt cohort exceeds its bound")?;
        for chat in chats {
            need(chat.state.position == 0 && chat.state.layers.len() == d.layers.len() && !chat.tokens.is_empty()
                && chat.tokens.len() as u64 <= chat.decoder.capacity, "prompt cohort chat differs")?;
            need(chat.tokens.iter().all(|&id| u64::from(id) < d.vocab), "prompt token exceeds vocabulary")?;
        }
        if trims_spares_first(total, 0) { self.compute.rows_keep()?; }
        self.compute.rows_cohort_spares_begin()?;
        let result = self.prefill_cohort_layers(chats);
        let cleanup = self.compute.rows_cohort_spares_end();
        let output = result?;
        cleanup?;
        Ok(output)
    }

    fn prefill_cohort_layers(&self, chats: &[PromptChat<'_, '_, C>]) -> Result<Vec<PromptResult<C::Value>>> {
        let d = self.checkpoint.description();
        let streams = d.residual.streams as usize;
        let hidden = d.hidden as usize;
        let mut runs = std::collections::VecDeque::with_capacity(chats.len());
        let mut states = Vec::with_capacity(chats.len());
        for (slot, chat) in chats.iter().enumerate() {
            let decoder = chat.decoder;
            self.compute.rows_cohort_chat(Some(slot));
            let rows = match decoder.compute.rows_embedding(chat.tokens, streams)? {
                Some(value) => value,
                None => {
                    let rows = chat.tokens.iter().map(|&id| decoder.embedding(id)).collect::<Result<Vec<_>>>()?;
                    decoder.compute.join(&rows.iter().collect::<Vec<_>>())?
                }
            };
            runs.push_back((chat.tokens.len(), rows));
            states.push(State { position: chat.tokens.len() as u64, layers: Vec::with_capacity(d.layers.len()), expanded: BTreeMap::new() });
        }
        for layer in 0..d.layers.len() {
            let plan = self.prompt_plan(layer, chats[0].tokens.len())?;
            let mut cut = wide_cut_of(&plan);
            for chat in &chats[1..] {
                let Some((index, carry)) = &cut else { break; };
                let own = chat.decoder.prompt_plan(layer, chat.tokens.len())?;
                let common = wide_cut_of(&own).is_some_and(|(at, name)| at == *index && name == *carry);
                if !common || own.steps[*index..] != plan.steps[*index..] { cut = None; }
            }
            for (slot, (chat, state)) in chats.iter().zip(&mut states).enumerate() {
                let decoder = chat.decoder;
                self.compute.rows_cohort_chat(Some(slot));
                let (n, residual) = runs.pop_front().ok_or("prompt cohort has no residual")?;
                let held = chat.state.expanded.get(&layer).filter(|held| held.1 == 0);
                let (output, next, cache, _) = decoder.layer_rows(
                    layer, 0, n, residual, &chat.state.layers[layer], held, false, false,
                    cut.as_ref().map(|(index, carry)| (*index, carry.as_str())), None,
                )?;
                state.layers.push(next);
                if let Some(cache) = cache {
                    state.expanded.insert(layer, Expanded(std::rc::Rc::new(cache), n, decoder.sparse_layout(layer)?));
                } else if let Some(held) = chat.state.expanded.get(&layer) {
                    state.expanded.insert(layer, held.clone());
                }
                runs.push_back((n, output));
                self.compute.rows_phase_end()?;
            }
            if let Some((index, carry)) = cut {
                spark_gpu_probe::prompt_launches::record("wide_group_layer");
                runs = self.layer_routed_runs(layer, runs.into(), &plan, index, &carry, true)?.into();
                self.compute.rows_phase_end()?;
            }
        }
        let mut s = shape(Op::StreamMean);
        s.heads = number(d.residual.streams)?;
        s.dim = number(d.hidden)?;
        let name = "model.language_model.norm.weight";
        let mut output = Vec::with_capacity(chats.len());
        for (slot, ((chat, state), (n, residual))) in chats.iter().zip(states).zip(runs).enumerate() {
            let decoder = chat.decoder;
            self.compute.rows_cohort_chat(Some(slot));
            let mean = decoder.compute.prompt(RowShape::wrapped(s, n).ok_or("prompt row count exceeds ABI")?,
                [Some(&residual), None, None, None, None], &[])?;
            let normalized = decoder.compute.rows_norm(name, norm_shape(self.checkpoint, name, hidden)?, &mean, n, &decoder.parameters[name])?;
            let logits = decoder.head_rows(&normalized, n, HeadRows::Last, None)?.pop().ok_or("prompt cohort has no final logits")?;
            let tail = if decoder.predictor_enabled() {
                let open = OpenRows { first: 0, rows: n, layers: (0..d.layers.len()).map(|_| None).collect(),
                    hidden: [decoder.compute.range(&mean, 0, n * hidden)?, decoder.compute.range(&normalized, 0, n * hidden)?] };
                Some(decoder.open_tail(open, super::super::PROMPT_TRAIL)?)
            } else { None };
            output.push(PromptResult { logits, state, tail });
        }
        Ok(output)
    }

    fn prompt_plan(&self, layer: usize, rows: usize) -> Result<LayerPlan> {
        let mut schedules = self.schedules.borrow_mut();
        schedules.get_mut(layer).ok_or("prompt layer is outside the recipe")?.at(rows as u64)
            .cloned().map_err(|error| error.to_string())
    }
}
