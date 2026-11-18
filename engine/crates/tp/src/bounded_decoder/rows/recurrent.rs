//! Fuse the recurrent span of one chat's short row pass.
use super::*;
use engine_model::plan::layer::Step;
use spark_gpu_probe::recurrent_rows::{DIM, Shape as RecurrentShape, TAPS};

impl<'f, C: Compute> Decoder<'f, C> {
    /// The interleaved gate projections depend only on input rows. Let the
    /// same projection stage produce them before the composite launch.
    pub(super) fn rows_recurrent_span(&self, layer: usize, index: usize, plan: &LayerPlan,
        chat: &mut RowsChat<'_, C::Value>, values: &mut BTreeMap<String, C::Value>) -> Result<Option<usize>> {
        let m = chat.m;
        if !(2..=8).contains(&m) { return Ok(None); }
        let d = self.checkpoint.description();
        let TokenMixer::Kda(kda) = d.layers[layer].mixer else { return Ok(None); };
        let layout = self.kda_layout(layer)?;
        let shape = RecurrentShape {tokens: number(m as u64)?, heads: number(layout.local_heads() as u64)?,
            bound: kda.gate_lower_bound as f32, epsilon: d.epsilon as f32};
        if kda.head_dim != DIM as u64 || kda.conv != TAPS as u64 || !shape.admitted() { return Ok(None); }

        let Some([conv, forget_down, forget, beta, prepare, step, gate_down, gate, norm]) =
            plan.steps.get(index..index + 9) else { return Ok(None); };
        let projects = [(1, forget_down), (2, forget), (3, beta), (6, gate_down), (7, gate)];
        if prepare.kind != Kind::KdaPrepare || step.kind != Kind::KdaStep || norm.kind != Kind::GatedNorm
            || projects.iter().any(|(_, p)| p.kind != Kind::Project || p.outputs.len() != 1) { return Ok(None); }
        let [q, k, v, history] = conv.inputs.as_slice() else { return Ok(None); };
        let [convolved, next_history] = conv.outputs.as_slice() else { return Ok(None); };
        let [conv_in, forget_in, beta_in] = prepare.inputs.as_slice() else { return Ok(None); };
        let [prepared] = prepare.outputs.as_slice() else { return Ok(None); };
        let [prepared_in, state] = step.inputs.as_slice() else { return Ok(None); };
        let [recurrent, next_state] = step.outputs.as_slice() else { return Ok(None); };
        let [recurrent_in, gate_in] = norm.inputs.as_slice() else { return Ok(None); };
        let [gated] = norm.outputs.as_slice() else { return Ok(None); };
        if conv_in != &convolved.name || prepared_in != &prepared.name || recurrent_in != &recurrent.name
            || forget_in != &forget.outputs[0].name || beta_in != &beta.outputs[0].name || gate_in != &gate.outputs[0].name {
            return Ok(None);
        }
        let internal = [&convolved.name, &prepared.name, &recurrent.name];
        if plan.returns.iter().chain(plan.steps[index + 9..].iter().flat_map(|s| &s.inputs)).any(|name| internal.contains(&name)) {
            return Ok(None);
        }

        // Shared passes supply each chat's Q, K and V to Conv, while
        // the independent gate projections still belong to the shared map.
        for (i, (_, project)) in projects.iter().enumerate() {
            let [source] = project.inputs.as_slice() else { return Ok(None); };
            if !values.contains_key(&project.outputs[0].name) && !values.contains_key(source)
                && !projects[..i].iter().any(|(_, earlier)| &earlier.outputs[0].name == source) {
                return Ok(None);
            }
        }
        for (offset, project) in projects {
            if !values.contains_key(&project.outputs[0].name) {
                self.rows_project_stage(index + offset, plan, m, values)?;
            }
        }
        let Some(old_history) = chat.state.get(history) else { return Ok(None); };
        let Some(old_state) = chat.state.get(state) else { return Ok(None); };
        let weights = |s: &Step| s.weights.iter().flat_map(|w| self.parameters[&w.name].iter().copied()).collect::<Vec<_>>();
        let [cw, pw, nw] = [conv, prepare, norm].map(weights);
        let b = &self.compute;
        let Some(made) = b.rows_recurrent(shape, [&values[q], &values[k], &values[v], old_history,
            &values[forget_in], &values[beta_in], old_state], &values[gate_in], [&cw, &pw, &nw], chat.keep)?
            else { return Ok(None); };

        if chat.keep {
            let row_words = m * layout.local_heads() * DIM;
            chat.kept_conv = Some(([b.range(&values[q], 0, row_words)?, b.range(&values[k], 0, row_words)?,
                b.range(&values[v], 0, row_words)?], history.clone()));
            let (prepared, decay) = made.saved.ok_or("recurrent layer did not save its prefix operands")?;
            chat.kept_step = Some((prepared, decay, state.clone()));
        }
        let gathered = b.gather_attended(&made.gated, m, layout)?;
        values.insert(next_history.name.clone(), made.history);
        values.insert(next_state.name.clone(), made.state);
        values.insert(gated.name.clone(), gathered);
        Ok(Some(8))
    }
}
