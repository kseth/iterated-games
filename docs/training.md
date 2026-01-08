# Architecture Design: Self-Improving Poetry

## 1. Overview & Motivation

This document describes a self-improving training system for poetry. The core idea: teach a model to write poems by having it discover—and learn from—its own useful intermediate representations.

### The Problem

We have a dataset of `(title, poem)` pairs, but no *descriptions* explaining what each poem is about. Standard instruction tuning would require manually annotating each poem with a natural-language description (e.g., "Write a melancholic sonnet about autumn leaves"). This doesn't scale.

### Our Solution

Rather than annotate descriptions by hand, we have the model generate them, then filter for quality:

1. Generate candidate descriptions for each title.
2. Evaluate descriptions by how much they help the model predict the actual poem (lower training loss = more useful description).
3. Train on the winning `(description, poem)` pairs, so the model learns both to follow descriptions and to infer appropriate titles.

This creates a self-improvement loop: better descriptions lead to better training signal, which produces a model that generates even better descriptions.

### Design Goals

- Flexible inference: Support generation with *or* without a user-provided title. When given a title, echo it; when given only a description, infer an appropriate title.
- Data efficiency: Extract maximum signal from each example via Top-K selection (multiple winners per poem) and format augmentation (multiple input formats per winner).
- No external dependencies: The model is both teacher and student—no separate reward model or human labeling required.

### Model Compatibility

This approach works for both base (pre-trained) and instruction-tuned / thinking models:

- Base models: Use raw completion format with explicit delimiters (e.g., `Output:`).
- Instruction-tuned models: Use native chat templates (system/user/assistant turns). The natural turn boundary replaces explicit delimiters.
- Thinking models (e.g., Qwen3 with `<think>` blocks): Reasoning can help the model deliberate during description generation. Thinking can be stripped or retained in training data depending on goals.

The algorithm is format-agnostic—adapt the prompt structure to your model's native format.

---

## 2. Theoretical Foundation

Our approach synthesizes three complementary techniques from recent self-improvement literature: *instruction backtranslation*, *self-taught reasoning*, and *rejection sampling fine-tuning*.

The core challenge we face—possessing `(title, poem)` pairs without intermediate instructions—mirrors the setup addressed by [*Li et al. (2023)*](https://arxiv.org/abs/2308.06259) in their work on *instruction backtranslation*. Inspired by backtranslation in machine translation, they demonstrate that a model can generate instruction prompts for unlabeled outputs (web documents), then self-curate the highest quality `(instruction, output)` pairs for fine-tuning. Critically, they show that iterating this process—where an improved model produces better curation, which yields a better model—outperforms training on human-annotated data alone.

We adapt this insight to poetry by treating the description as a “missing” piece of supervision that we can recover. For each `(title, poem)` pair, we ask the model to write a description *conditioned on the poem itself* (so it can be specific and accurate), but we make the prompt explicitly protective: the description should not copy lines or distinctive phrasing from the poem. This is where [*STaR (Zelikman et al., 2022)*](https://arxiv.org/abs/2203.14465) becomes relevant. STaR establishes that a model can bootstrap its capabilities by generating many candidates, selecting the ones that succeed under an objective signal, and training on the successful paths. The key insight is that even "accidental" successes contain learnable signal—the model improves by studying *what worked*, not just *what was intended*.

Finally, [*Singh et al. (2023)*](https://arxiv.org/abs/2312.06585) describe a simple pattern for scaling self-training: generate multiple candidates, score them with an automated signal, then fine-tune on the best ones—repeat. Their results show that this kind of “generate → score → train” loop can keep paying off as you scale model size and training compute. In our setting, the automated signal is straightforward: how well a candidate description helps the model predict the target poem.

Together, these techniques form our training loop: backtranslate a description from each poem, score candidates by predictive utility (STaR-style) while discouraging copying via an overlap penalty, and fine-tune on the winners (rejection-sampling style). Each iteration improves both the model's ability to *generate* useful descriptions and to *follow* them when writing poetry.
 

---

## 3. The Pipeline Architecture

Each training iteration cycles through four phases: Generate → Score → Select → Train.

### Phase 1: Candidate Generation

Given a `(title, poem)` pair from the dataset, we ask the model to propose descriptions that could have motivated this poem.

Prompt design (protective):
- Provide both the title and the poem text.
- Instruct the model to describe the poem at a high level without copying phrasing.

Core instructions (adapt to your model's format):
> Write a short description that would help someone write this poem.
> Rules:
> - Do not quote the poem.
> - Do not reuse distinctive phrases or sequences of words from the poem.
> - Prefer themes, imagery, tone, structure, and constraints (e.g., rhyme, meter) over exact wording.

Sampling: Generate `N` candidates (e.g., `N = 4`) using high temperature to encourage diversity.

Output: A set of candidate descriptions (N total).

The model is essentially "reverse-engineering" instructions: given the output (a poem with this title), what input (description) would have produced it?

> Note for thinking models: The model's reasoning (`<think>` blocks) may help it deliberate about what makes a good description. You can strip reasoning from the final description candidates or retain it for analysis.

### Phase 2: Scoring & Selection

Not all generated descriptions are useful. We score and rank candidates by their *predictive utility*—how much they help the model anticipate the actual poem.

We use a combined heuristic rather than hard filtering. The goal is to prefer descriptions that are (a) useful for predicting the poem, while (b) not simply re-stating the poem text.

1. Utility signal (primary): For each candidate description, measure how well the model can predict the poem when given `Title + Description` as context. Use the model’s standard token-level prediction loss (averaged per token). Lower loss means the description provides more useful context.

2. Overlap penalty (soft): Compute an overlap score between the description and the poem text. Higher overlap indicates the description is closer to copying (or heavily echoing) the poem. This complements the protective prompt: the prompt asks nicely; the overlap penalty enforces.

3. Keep scales stable (small-N friendly): With only a few candidates per poem (e.g., ~5), don’t rely on fancy per-poem normalization. Instead:
   - Keep the overlap score naturally bounded (for example, between 0 and 1).
   - Use per-token (average) prediction loss for utility.
   - Tune a single “overlap weight” globally so overlap matters, but doesn’t dominate.

4. Combined score: Rank candidates by a single combined score that rewards low prediction loss and penalizes high overlap. Lower is better. The overlap weight controls how strongly overlap is penalized.

5. Select the best few: Keep the top `K` candidates under that combined score (e.g., `K = 2`). Selecting multiple winners captures synonymous descriptions and increases data diversity.

### Phase 3: Training Example Construction

Each winning description becomes the basis for training examples. To support flexible inference, we create two variants per winner:

| Variant | Use Case | Input | Target |
|---------|----------|-------|--------|
| Echo | User provides a title | Title + Description | Title + Poem |
| Creative | User provides only a description | Description | Title + Poem |

The Echo variant trains the model to follow descriptions when a title is given. The Creative variant trains the model to *infer* an appropriate title from the description alone—useful when users want the model to choose its own title.

### Phase 4: Optimization

We fine-tune the model on the constructed examples. The input context is masked; the target portion (Title + Poem) is unmasked and contributes to the loss.

By treating the winning descriptions as ground truth, we reinforce whatever made them useful. Over iterations, this shapes the model to generate increasingly coherent descriptions in Phase 1.

---

## 4. Implementation Details

### Sequence Format

Training sequences need a clear boundary between input context (what the model is given) and generation target (what it learns to produce). The exact format depends on your model type.

#### Conceptual Structure

| Mode | Input Context | Generation Target |
|------|---------------|-------------------|
| Echo (title provided) | Title + Description | Title + Poem |
| Creative (title omitted) | Description only | Title + Poem |

#### Format Examples

Base model (raw completion):

```
Title: {T}
Description: {D}

Output:
Title: {T}
Poem: {P}
```

The `Output:` marker serves as the generation trigger.

Instruction-tuned model (chat format):

```
[System] You are a poet. Given a title and description, write the poem.
[User] Title: {T}
Description: {D}
[Assistant] Title: {T}
Poem: {P}
```

The assistant turn boundary serves as the generation trigger.

Thinking model (with reasoning):

Same as instruction-tuned, but the model may produce `<think>...</think>` blocks before the output. You can choose to:
- Strip thinking from training data (train on final output only)
- Retain thinking if you want the model to reason during generation

Adapt these templates to your model's native chat format (e.g., Qwen3's `<|im_start|>` / `<|im_end|>` markers).

### Loss Masking Strategy

We use standard causal language modeling loss, but only compute loss on the *target* portion of each sequence:

| Region | Contributes to Loss? | Rationale |
|--------|----------------------|-----------|
| Input context | No | This is provided at inference; we don't need to train generation here. |
| Title (in target) | Yes | In Creative mode, the model must learn to infer appropriate titles. |
| Poem (in target) | Yes | Core generation target. |

The boundary between context and target depends on format:
- Base models: The `Output:` delimiter (or similar) marks the boundary.
- Chat models: The assistant turn start marks the boundary (context = system + user turns).
- Thinking models: Decide whether reasoning tokens contribute to loss. A common choice is to mask thinking and only train on final output.

### Overlap Signal (for Sampling)

The overlap penalty can be computed in a few ways. A good default is an IDF-weighted overlap, which down-weights common words and up-weights rare ones.

- Build an IDF table (offline, once): Treat each poem as a document. For each word, count how many poems contain it. Assign higher weight to words that appear in fewer poems (rare words), and lower weight to words that appear everywhere (common words).
- Compute overlap (per candidate): Use a length-insensitive measure such as cosine similarity between TF-IDF vectors, or a simpler “weighted overlap fraction,” like: “what fraction of the description’s weighted words also appear in the poem?”

### Scaling the Combined Score

To make the combined score behave consistently, ensure the two parts are comparable:

- Use bounded overlap: Define the overlap signal so it lives in a stable range (often 0 to 1).
- Use per-token loss: Use average token-level prediction loss (not total loss) so poem length doesn’t dominate.
- Calibrate the overlap weight globally (recommended for small N): Pick a single overlap weight using a small held-out slice (or recent batches) so that overlap can break ties, but doesn’t overwhelm the utility signal.
- Tune with diagnostics: Track (a) average overlap of selected descriptions and (b) average prediction loss of selected descriptions. Increase the overlap weight if overlap stays high; decrease it if selection starts preferring low-overlap but unhelpful descriptions.

---

## 5. The Self-Improvement Loop

The system improves through iteration. Here's how each cycle builds on the last:

### Iteration 0: Bootstrap

The model starts from its initial weights—whether base pre-trained, instruction-tuned, or a previous checkpoint. Its initial description guesses may be poor (vague, generic, or unrelated to the actual poem), but some fraction will be accidentally useful: they happen to provide context that reduces prediction loss.

> Note: Instruction-tuned models often produce better initial descriptions because they're already trained to follow instructions. This can accelerate the bootstrap phase.

### Iteration 1+: Refinement

1. Generate: The model proposes descriptions for each `(title, poem)` pair.
2. Select: We keep descriptions that score well under the combined heuristic (low prediction loss and low overlap penalty).
3. Train: The model fine-tunes on winning `(description, poem)` pairs.
4. Repeat: The improved model generates better descriptions in the next iteration.

### Why This Works

The key insight is that the Top-K threshold is *relative*, not absolute. Even if early descriptions are mediocre, we select the *best* among them. Training on these raises the model's baseline, so the next iteration's "best" is better than the previous iteration's "best."

Over time, two capabilities co-evolve:
- Description generation: The model learns what kinds of descriptions are predictively useful.
- Description following: The model learns to leverage descriptions when writing poems.

This is the STaR dynamic in action: the model bootstraps its own supervision by identifying and reinforcing its accidental successes.

---

## 6. Pseudocode Reference

The following pseudocode illustrates the training loop at a conceptual level, independent of any specific framework or model format.

```
FUNCTION train_iteration(dataset, model, renderer, config):
    training_examples = []
    
    FOR EACH (title, poem) IN dataset:
        
        # ─── PHASE 1: CANDIDATE GENERATION ───
        # Ask the model to describe the poem without copying it.
        # The renderer formats this as raw completion or chat messages
        # depending on model type.
        prompt = renderer.format_description_request(title, poem)
        candidates = model.generate(
            prompt,
            num_samples = config.N,
            temperature = config.high_temp
        )
        
        # ─── PHASE 2: SCORING & SELECTION ───
        scored = []
        FOR EACH description IN candidates:
            # Primary term: how well the description helps predict the poem
            context = renderer.format_poem_context(title, description)
            target = renderer.format_poem_target(title, poem)
            loss = model.evaluate_loss(context, target)

            # Soft leakage penalty: overlap between description and poem
            overlap = compute_overlap(description, poem, idf=config.idf)

            scored.append((loss, overlap, description))
        
        # Combine the two signals:
        # - loss should be average token-level prediction loss
        # - overlap should be a bounded score (often 0..1)
        winners = top_k(
            scored,
            k=config.K,
            key = (loss, overlap, _) => combine(loss, overlap, overlap_weight=config.overlap_weight)
        )
        
        # ─── PHASE 3: TRAINING EXAMPLE CONSTRUCTION ───
        FOR EACH description IN winners:
            # Echo variant: user provides title
            training_examples.append(
                renderer.format_training_example(
                    title=title, description=description, poem=poem,
                    include_title_in_context=True
                )
            )
            
            # Creative variant: user omits title
            training_examples.append(
                renderer.format_training_example(
                    title=title, description=description, poem=poem,
                    include_title_in_context=False
                )
            )
    
    # ─── PHASE 4: OPTIMIZATION ───
    model.finetune(
        examples = training_examples,
        loss_on = "target_only"  # Mask context, compute loss on target
    )
    
    RETURN model
```

### Key Operations

| Operation | Description |
|-----------|-------------|
| `renderer.format_*(...)` | Adapts prompts/examples to model's native format (raw completion, chat template, etc.). |
| `model.generate(prompt, num_samples, temperature)` | Sample multiple completions from the model with diversity. |
| `model.evaluate_loss(context, target)` | Compute how well the model predicts `target` given `context` (lower = better). |
| `compute_overlap(description, poem, idf)` | Overlap score between description and poem (higher = more overlap). Common: IDF-weighted cosine similarity. |
| `combine(loss, overlap, overlap_weight)` | Combine utility (loss) and overlap into a single score for ranking candidates. |
| `top_k(items, k, key)` | Return the `k` items with lowest value of `key`. |
| `model.finetune(examples, loss_on)` | Update model weights; only compute loss on the specified region. |

### Renderer Examples

The `renderer` abstracts away format differences. Example implementations:

Base model renderer:
- `format_poem_context(T, D)` → `"Title: {T}\nDescription: {D}\n\nOutput:\n"`
- `format_poem_target(T, P)` → `"Title: {T}\nPoem: {P}"`

Chat model renderer (e.g., Qwen3):
- `format_poem_context(T, D)` → `[System: "You are a poet.", User: "Title: {T}\nDescription: {D}"]`
- `format_poem_target(T, P)` → `[Assistant: "Title: {T}\nPoem: {P}"]`

The renderer handles chat templates, special tokens, and thinking blocks as needed.

### Outer Loop

The above function represents a single iteration. The full training process repeats this:

```
model = load_model()  # Base or instruction-tuned
renderer = create_renderer(model)

FOR iteration IN 1..MAX_ITERATIONS:
    model = train_iteration(dataset, model, renderer, config)
    
    # Optional: checkpoint, evaluate, adjust config
```

Each iteration uses the improved model to generate better descriptions, creating a positive feedback loop.