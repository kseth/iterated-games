# Architecture Design: Self-Improving Poetry

## 1. Overview & Motivation

This document describes a self-improving training system for poetry. The core idea: teach a model to write poems by having it discover—and learn from—its own useful intermediate representations.

### The Problem

We have a dataset of `(title, poem)` pairs, but no *descriptions* explaining what each poem is about. Standard instruction tuning would require manually annotating each poem with a natural-language description (e.g., "Write a melancholic sonnet about autumn leaves"). This doesn't scale.

### Our Solution

Rather than annotate descriptions by hand, we have the model generate them, then filter for quality:

1. Generate candidate descriptions for each `(title, poem)` pair.
2. Evaluate descriptions by how much they help the model predict the target (title and poem) (lower training loss = more useful description).
3. Train on the winning `(description, (title, poem))` pairs, so the model learns both to follow descriptions and to infer appropriate titles.

This creates a self-improvement loop: better descriptions lead to better training signal, which produces a model that generates even better descriptions.

### Design Goals

- Flexible inference: Support generation from description alone, where the model infers an appropriate title and the poem's content based on the description.
- Data efficiency: Extract maximum signal from each example via Top-K selection (multiple winners per poem).
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

Finally, [*Singh et al. (2023)*](https://arxiv.org/abs/2312.06585) describe a simple pattern for scaling self-training: generate multiple candidates, score them with an automated signal, then fine-tune on the best ones—repeat. Their results show that this kind of "generate → score → train" loop can keep paying off as you scale model size and training compute. In our setting, the automated signal is straightforward: how well a candidate description helps the model predict the target (title and poem).

Together, these techniques form our training loop: backtranslate a description from each poem, score candidates by predictive utility (STaR-style), and fine-tune on the winners (rejection-sampling style). Each iteration improves both the model's ability to *generate* useful descriptions and to *follow* them when writing poetry.
 

---

## 3. The Pipeline Architecture

Each training iteration cycles through four phases: Generate → Score → Select → Train.

### Phase 1: Candidate Generation

Given a `(title, poem)` pair from the dataset, we ask the model to propose descriptions that could have motivated this poem.

Prompt design (protective):
- Provide both the title and the poem text.
- Instruct the model to describe the poem at a high level without copying phrasing.

Core instructions (adapt to your model's format):
> In about 100 words, describe what this poem is really about.
>
> Write the description like you would write notes or a brief sketch. It should capture:
> - Core emotions or insights
> - Any imagery or sensory details
> - The mood, tone, or voice
> - Structure or progression in the poem
>
> Important: Describe the poem's essence in a way that would allow a poet to create it without seeing it. However, do not quote or copy any lines.

Sampling: Generate `N` candidates (e.g., `N = 4`) using high temperature to encourage diversity.

Output: A set of candidate descriptions (N total).

The model is essentially "reverse-engineering" instructions: given the output (a poem with this title), what input (description) would have produced it?

> Note for thinking models: The model's reasoning (`<think>` blocks) may help it deliberate about what makes a good description. You can strip reasoning from the final description candidates or retain it for analysis.

### Phase 2: Scoring & Selection

Not all generated descriptions are useful. We score and rank candidates by their *predictive utility*—how much they help the model anticipate the target (title and poem).

Scoring process:

1. **Utility signal**: For each candidate description, measure how well the model can predict the target (title and poem) when given the description as context. Use the model's standard token-level prediction loss (averaged per token). Lower loss means the description provides more useful context.

2. **Selection**: Rank candidates by their prediction loss (lower is better) and keep the top `K` candidates (e.g., `K = 2`). Selecting multiple winners captures synonymous descriptions and increases data diversity.

The protective prompt (from Phase 1) encourages the model to avoid copying, and the loss-based selection naturally favors descriptions that provide useful predictive context rather than simply restating the poem text.

### Phase 3: Training Example Construction

Each winning description becomes the basis for a training example. The model learns to infer an appropriate title and the poem's content from the description alone:

| Input | Target |
|-------|--------|
| Description | Title + Poem |

The model is trained to generate both the title and poem from a description, which allows it to learn title and content relationships based on descriptions and ground truth.

### Phase 4: Optimization

We fine-tune the model on the constructed examples. The input context is masked; the target portion (Title + Poem) is unmasked and contributes to the loss.

By treating the winning descriptions as ground truth, we reinforce whatever made them useful. Over iterations, this shapes the model to generate increasingly coherent descriptions in Phase 1.

---

## 4. Implementation Details

### Sequence Format

Training sequences need a clear boundary between input context (what the model is given) and generation target (what it learns to produce). The exact format depends on your model type.

#### Conceptual Structure

| Input Context | Generation Target |
|---------------|-------------------|
| Description only | Title + Poem |

#### Format Examples

Base model (raw completion):

```
Description: {D}

Output:
Title: {T}
Poem: {P}
```

The `Output:` marker serves as the generation trigger.

Instruction-tuned model (chat format):

```
[System] You are a poet.
[User] Description: {D}
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
| Title | Yes | The model learns to infer appropriate titles from the description. |
| Poem | Yes | Core generation target. |

The boundary between context and target depends on format:
- Base models: The `Output:` delimiter (or similar) marks the boundary.
- Chat models: The assistant turn start marks the boundary (context = system + user turns).
- Thinking models: Decide whether reasoning tokens contribute to loss. A common choice is to mask thinking and only train on final output.

### Scoring Details

Candidates are scored using average token-level prediction loss (total loss divided by number of tokens, not total loss) to ensure fair comparison across descriptions of different lengths. The top `K` candidates with lowest loss are selected as winners.

---

## 5. The Self-Improvement Loop

The system improves through iteration. Here's how each cycle builds on the last:

### Iteration 0: Bootstrap

The model starts from its initial weights—whether base pre-trained, instruction-tuned, or a previous checkpoint. Its initial description guesses may be poor (vague, generic, or unrelated to the actual poem), but some fraction will be accidentally useful: they happen to provide context that reduces prediction loss.

> Note: Instruction-tuned models often produce better initial descriptions because they're already trained to follow instructions. This can accelerate the bootstrap phase.

### Iteration 1+: Refinement

1. Generate: The model proposes descriptions for each `(title, poem)` pair.
2. Select: We keep descriptions that score well by prediction loss (lower loss = more useful description).
3. Train: The model fine-tunes on winning `(description, (title, poem))` pairs.
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
            # Score by how well the description helps predict the target (title and poem)
            context = renderer.format_poem_context(description)
            target = renderer.format_poem_target(title, poem)
            loss = model.evaluate_loss(context, target)  # Average token-level loss
            
            scored.append((loss, description))
        
        # Select top-K candidates by lowest loss
        winners = top_k(
            scored,
            k=config.K,
            key = (loss, _) => loss  # Lower loss is better
        )
        
        # ─── PHASE 3: TRAINING EXAMPLE CONSTRUCTION ───
        FOR EACH description IN winners:
            # Model infers title from description
            training_examples.append(
                renderer.format_training_example(
                    description=description, title=title, poem=poem
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
| `model.evaluate_loss(context, target)` | Compute how well the model predicts `target` given `context` (lower = better). Returns average token-level loss. |
| `top_k(items, k, key)` | Return the `k` items with lowest value of `key`. |
| `model.finetune(examples, loss_on)` | Update model weights; only compute loss on the specified region. |

### Renderer Examples

The `renderer` abstracts away format differences. Example implementations:

Base model renderer:
- `format_poem_context(D)` → `"Description: {D}\n\nOutput:\n"`
- `format_poem_target(T, P)` → `"Title: {T}\nPoem: {P}"`

Chat model renderer (e.g., Qwen3):
- `format_poem_context(D)` → `[System: "You are a poet.", User: "Description: {D}"]`
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

---

## 7. Implementation Notes

The `src/train.py` implementation makes several practical choices beyond the core algorithm:

### Fine-Tuning

We use **LoRA** (rank 32) rather than full fine-tuning. This reduces memory and makes checkpoints lightweight while still providing sufficient capacity for the poetry domain.

### Learning Rate Schedule

Linear decay from `1e-4` to `5e-5` over all training batches. The schedule is computed upfront based on dataset size and iteration count.

### Intra-Iteration Resampling

The sampling client can be refreshed *during* an iteration (controlled by `evals_per_iteration`). This means later poems in the same iteration benefit from training updates on earlier poems—a tighter feedback loop than waiting for the full iteration to complete.

### Thinking Blocks

For Qwen3 thinking models, `<think>` reasoning is allowed during description generation, but we extract only the final content (stripping reasoning) before using the description for scoring and training.

### Checkpointing

- Run names are auto-generated with a poet theme (e.g., `whimsical-whitman-0042`)
- Checkpoints include optimizer state for exact resume via `--resume_from`
- Metadata is saved locally alongside Tinker state for debugging

### Poem Filtering

Long poems are filtered out before training (default: 6144 tokens max) to ensure sequences fit within context limits and training remains efficient.