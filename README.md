# wmk — watermark capacity measurement

A key-agnostic reference implementation of the SynthID-Text family of keyed LLM text
watermarks (generator, detector, calibration), and the measurement code for the
Watermark Capacity Atlas.

The research question is not "how well does a detector perform on this corpus" but
**"what is the ceiling on any detector's performance, given the model's own next-token
distributions"**. The two are different quantities, and the second has not been measured.

## What this is not

Stated positively, so the edges are unambiguous:

- Not a production watermark or a production detector.
- Not a detector for any specific provider's watermark. No provider's key or config is
  known, sought, or represented anywhere in this repository.
- Not a benchmark against any provider's detector.
- Not a measurement of watermark *removal*. Paraphrase attacks are cited from the
  literature only.

It **does** measure a property of a model's next-token distribution, and derives from
that the upper bound on what any key-based watermark in this family could achieve.

## Dual-use posture

Capacity bounds and scheme-family identification inform policy without helping anyone
strip a watermark. This is a deliberately better position than publishing a detector,
which is a removal oracle for anyone holding a key.

Two non-goals are enforced in code rather than left as assumptions:

- **Per-position g-values never cross the library boundary.** That vector identifies
  exactly which positions carry signal. `wmk.score` returns aggregates only, and window
  sums are floored at 32 positions so a window-of-one cannot reconstruct it.
- **No key recovery.** Nothing in the library infers a key from text; the detector
  requires a key as input.

## The measured quantity

Tournament sampling with depth `d` draws `2**d` candidates from the post-truncation
distribution and picks among them using key-derived g-values. If all `2**d` draws land on
the same token, there is no choice to make and the position carries no watermark signal.

For a post-truncation, renormalized probability vector `p`:

```
p_active(p, d) = 1 - sum_i p_i ** (2 ** d)
```

Effective positions sum this over **unmasked** positions only:

```
N_eff = sum over unmasked t of p_active(p_t, d)
```

Two properties of this quantity drive most of the implementation:

- It must be computed in **float64**. Not for underflow — the reason usually given, which
  does not hold — but because `sum_i p_i**(2**d)` approaches 1 at near-deterministic
  positions and float32 leaves roughly `eps / p_active` relative error in the
  difference. See `.claude/phase_0_report.md`.
- It is **an upper bound**, directionally. Post-training reduces next-token entropy, so a
  bound measured on an open instruct model bounds a more heavily post-trained model from
  above. A low bound is a strong claim; a high one is uninformative.

Every capacity number is conditional on its full
`(temperature, top_p, top_k, tournament_depth, masking_window)` tuple. Serving config
moves capacity more than post-training does, and the two axes are never pooled.

## Install

```bash
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install -e ".[dev]" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match
.venv/bin/python -m pytest
```

Dependencies are pinned exactly. `transformers` changes tokenizer behaviour,
logits-processor interfaces and sampling code across minor versions, which breaks
byte-identical reruns; `tests/test_versions.py` fails if the installed set drifts or if
any dependency is left unpinned.

Drop the `--extra-index-url` on a CUDA host to get a GPU build.

## Use

```bash
python scripts/00_setup.py --download --smoke        # stage + hash weights
python scripts/00c_build_corpora.py --seed 0         # build corpora
python scripts/01_prf_vectors.py --check             # verify PRF determinism
python scripts/00b_smoke_test.py --seed 0            # end-to-end gate
python scripts/02_cache_probs.py --all --threads 16  # capacity scan (one pass, ever)
```

Every script takes `--seed` and reruns byte-identically. Every script is idempotent and
resumable, checkpointing every 50 documents.

```python
from wmk import WatermarkConfig, null_key, tokenizer_hash
from wmk.generate import generate
from wmk.score import score_token_ids

cfg = WatermarkConfig(tokenizer_id="Qwen/Qwen2.5-0.5B-Instruct",
                      tokenizer_hash=tokenizer_hash(tok))
out = generate(model, tok, "prompt", cfg, null_key(0), max_new_tokens=200, seed=0)
res = score_token_ids(out["token_ids"], cfg, null_key(0))
print(res.mean_g, res.z_null, res.n_scored)
```

## Plugging in a provider's key and config

The library is key-agnostic: it becomes a real detector for any provider in this scheme
family once a key and a config are supplied. Nothing else needs to change. The four
plug-in points:

1. **Key** — any 32-byte value. `wmk.keys.null_key(i)` supplies the HKDF-derived null
   ensemble used for calibration; a real key is passed as raw `bytes` to
   `wmk.score.score_text`. Keys are never persisted by the library.
2. **Config** — `WatermarkConfig` in `src/wmk/config.py`. The scheme axis is
   `prf`, `context_depth` (H), `tournament_depth` (d), `masking`, `masking_window`,
   `sampler_position`; the serving axis is `temperature`, `top_p`, `top_k`. Load with
   `WatermarkConfig.load(path)` from canonical JSON. Every result record carries the
   config's SHA-256.
3. **Tokenizer** — `tokenizer_id` plus `tokenizer_hash`, the SHA-256 of a sorted vocab
   dump. `score_text` raises `TokenizerMismatch` on a mismatch rather than returning a
   low score: a differing token-id assignment makes the g-value sequence meaningless
   rather than merely noisy, and a quiet wrong answer is the failure mode this project
   cannot afford.
4. **PRF** — `src/wmk/prf.py`, selected by `WatermarkConfig.prf`. Currently
   `hmac-sha256`, with the canonical encoding
   `round_idx:u16be || candidate:u32be || context tokens:u32be oldest first` and g the
   least significant bit of byte 0. 1000 frozen vectors in `tests/vectors/prf.json` pin
   it across platforms. Adding a variant means adding a branch here and a new frozen
   vector set.

The DeepMind reference accumulator hash is deliberately not reused: its own
documentation states it provides no cryptographic guarantees, and the empirical null
here rests on PRF indistinguishability under a key the scorer does not hold.

## Repository

```
src/wmk/        prf, keys, config, masking, capacity, truncate, scan, generate, score
scripts/        00_setup, 00b_smoke_test, 00c_build_corpora, 00d_dtype_sensitivity,
                01_prf_vectors, 02_cache_probs
tests/          frozen PRF vectors, capacity identities, masking properties,
                truncation-path agreement, version pins, config-literal guard
data/           models.json, grid.json, MANIFEST.json (hashes), corpora/, probs/
results/        capacity/, calibration/, figures/
.claude/        PLAN.md, CLAUDE.md, CLAIMS.md, RISKS.md, phase reports
```

`.claude/CLAIMS.md` is the ledger separating what providers have published, what this
project assumes, and what it has measured. Read it before quoting any number.

## Status

Phase 0 and Phase 0.5 complete, gates passed. Phase 2's capacity scan is next.
`scripts/08_make_figures.py` (C5: every figure regenerates from one command) and
`scripts/c2pa_check.py` are not yet written.

## Licence

Apache-2.0.
