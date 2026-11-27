# Whole-model host form of a looped decoder

**`loop_host` computes a looped block-attention text decoder end to end on
the CPU: token IDs in, logits and the greedy next token out.** It is slow on
purpose. It is the host form that a device run of the same model must equal
word for word. It enables no model plan and changes no serving path.

## What it is made of

The module joins the host form of each operation. It holds no arithmetic of
its own beyond the weighted norm call and its one BF16 round.

| Step of a visit | Host form |
| --- | --- |
| Visit list, messages, message gates, residual, visit output | [`visit_loop.rs`](../visit_loop.rs) |
| Block mix and end mix | [`block_mix.rs`](../block_mix.rs) |
| Weighted norm, plain norm, length clip, soft cap | [`norm_cap.rs`](../norm_cap.rs) |
| Attention with window, relative term, sink and output gate | [`sink_attention.rs`](../sink_attention.rs) |
| The four short convolutions of a visit | [`short_conv.rs`](../short_conv.rs) |
| Router, experts, fold | [`capped_moe.rs`](../capped_moe.rs) |
| Expert decode (the one seam for the expert numbers) | [`expert_widen.rs`](../expert_widen.rs) |
| Embedding norm, last norm, head, logit cap, top token | [`capped_head.rs`](../capped_head.rs) |

So the number types, the rounding points and the sum orders are those of the
files above. Each of them names the reference lines that it follows.

## The entry

```text
loop_host::Model::new(description, index, source, limits)
loop_host::Model::open(folder, limits)
model.runner()            -> Runner   bind the host forms to the weights, one time
runner.chat()             -> Chat     a new chat state
runner.read(chat, tokens, want, threads)   -> one TokenOut for each token
runner.greedy(chat, prompt, new_tokens, stop, threads) -> the new tokens
runner.visit(index, state, inputs, threads, record)    -> one visit, for checks
```

- `description` is the `LoopTextDescription` of the config adapter.
- `index` is the `IndexInput` of the tensor index. It gives each stored name
  its role. This module reads roles, never names.
- `source` gives stored bytes by stored name. `Folder` reads the weight files
  of a model folder. `Memory` holds tensors in memory, for a made-up model.
- `limits.expert_cache_bytes` is a host resource limit. It changes no word.

All sizes come from the description and from the tensor shapes. No model name
selects anything. A tensor with another type or shape is refused by name.

## The order of one visit

The order follows SPEC section 3.4 of the model, which the files above name
by reference line.

1. Messages from the block outputs of the pass before (later passes only).
2. Block mix for attention. Then the message gates (later passes only).
3. Weighted norm of each attention input, rounded to BF16. A window layer
   has one norm weight for the three inputs. A full layer has three.
4. Attention. Then the convolution after attention.
5. The residual inside the block, and the block mix for the experts. Then
   the expert message gate (later passes only).
6. Weighted norm, rounded to BF16. Router. The picked experts. The fold.
7. The convolution after the experts. Then the visit output, rounded to BF16.

After the last visit: the end mix, the last norm, the head on its used rows,
the logit cap, and the top token with the lower token ID on a tie.

## State

A `Chat` holds one `VisitState` for each visit of the visit list, in run
order: the keys and values of that visit's attention and the tails of its
four convolutions. Nothing else goes from one token to the next. A clone of a
chat is a fork.

## Threads change no word

Visit `v` of token `t` reads the earlier visits of token `t` and the state
that token `t - 1` left in visit `v`. So all visits with the same `v + t` are
independent. `read` runs them side by side. Each visit is a fixed function of
its inputs, and each sum keeps the order of its host form. The words do not
depend on:

- the thread count,
- how a text is cut into reads (one call, pieces, token by token),
- the limit of the expert cache.

The unit tests check each of the three on a made-up model.

## Experts

One widened expert is `3 * hidden * width` FP32 values. A model can hold more
experts than the host has memory for. So an expert is widened when a token
picks it, through `capped_moe::Expert::from_stored`, and kept in a cache with
a byte limit. The entry that was used longest ago leaves first.

## Layout facts

Each layout fact has one name in the storage plan and one in the host form of
its operation. `Model::new` refuses a build in which the two names of a fact
differ.

| Fact | In the plan | In the host form |
| --- | --- | --- |
| Fixed taps: the last stored tap weights the current token | `plan::inkling::FIXED_TAP_ORDER` | `short_conv::FIXED_TAP_ORDER` |
| Token-made taps: stored row 0 weights the current token | `plan::inkling::TOKEN_TAP_ORDER` | `short_conv::TOKEN_TAP_ORDER` |
| Token-made taps: index `head * taps + tap` | `plan::inkling::TOKEN_TAP_INDEX` | `short_conv::projected_tap` |
| Gate and up rows are interleaved | `plan::storage_groups::GATE_UP_ROWS` | `capped_moe::STORED_GATE_UP_ROWS` |
| The low 4 bits of a byte are the even column | `plan::storage_groups::NVFP4_NIBBLES` | `expert_widen::LOW_NIBBLE_IS_EVEN_COLUMN` |

## Tests

- [`tests.rs`](tests.rs): a made-up model from the fixture of the tensor
  index (16 visits, width 96, window 36, both layer kinds, both expert
  formats). Same words at every thread count and cut. A read equals
  `visit_loop::Driver::token`. The head never reads a padding row.
- [`../../tests/loop_host_real.rs`](../../tests/loop_host_real.rs): ignored.
  The real weights against the reference's module dumps, layer trace and
  answers. See its header for the folder that it reads.

Host tests do not establish GPU exactness or speed.
