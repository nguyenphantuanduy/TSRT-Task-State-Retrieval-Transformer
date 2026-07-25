# TSRT — Task-State Retrieval Transformer

End-to-end documentation for the **TSRT (Task-State Retrieval Transformer)** project: a custom encoder–decoder language model that adds an explicit, learnable **retrieval state** on top of a Qwen3-1.7B backbone, supervised by a teacher-generated reasoning dataset built from HotpotQA.

This README explains the **why**, the **what**, and the **how** — covering architecture, data pipeline, training, inference, and project layout.

---

## 1. Project Overview

### 1.1 Problem statement

Standard retrieval-augmented generation (RAG) pipelines treat retrieval as an external module:

```
Question → Retrieve Documents → Answer
```

This means the model never explicitly learns:
- **When** retrieval is needed,
- **Which** document is useful at which reasoning step,
- **Why** a document helps the current reasoning step.

TSRT turns retrieval into an **internal, supervised state** inside the model itself. Every decoder layer is augmented with:
- A **retrieval decision head** — should we retrieve here?
- A **usefulness scorer** — how relevant is each document to the current state?
- A **cross-attention with retrieval memory** — attention biased by usefulness scores.
- A **retrieval memory head** — carries retrieval state forward across positions.

### 1.2 Goals

- Build a single Transformer that **decides and acts** on retrieval during reasoning.
- Train it on **HotpotQA (distractor)** with **teacher-generated reasoning chains**.
- Initialize the architecture from a pretrained LLM (Qwen3-1.7B) to inherit language understanding.
- Open-source the dataset, weights, and code.

### 1.3 Repository artifacts

| Artifact | Where |
| --- | --- |
| Teacher-forcing dataset | `nguyenphantuanduy/TSRT-HotpotQA-Teacher` (Hugging Face) |
| Pretrained TSRT model | `nguyenphantuanduy/TSRT-Qwen3-1.7B` (Hugging Face) |
| Source dataset | `hotpotqa/hotpot_qa`, `distractor` config |
| Backbone | `Qwen/Qwen3-1.7B` |

---

## 2. Architecture

TSRT is a **42-layer encoder–decoder–TSRT** stack built on top of Qwen3.

### 2.1 Layer partitioning

The 42 transformer layers are split into three functional blocks:

| Block | Layers | Purpose |
| --- | --- | --- |
| Decoder | `0–13` | Pure self-attention, identical to Qwen3 layers. |
| Encoder | `14–27` | Document encoder, identical to Qwen3 self-attention + MLP. |
| TSRT | `28–41` | Each layer runs **self-attention → cross-attention → MLP**, where cross-attention is biased by retrieval memory. |

```
            Qwen3 embedding
                   │
        ┌──────────▼──────────┐
        │  Decoder (×14)      │   ← frozen at training
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Encoder (×14)      │   ← trained (self-attn only, FFN frozen)
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  TSRT (×14)         │   ← fully trained
        └──────────┬──────────┘
                   │
            Final RMSNorm
                   │
               LM Head
```

### 2.2 New components vs vanilla Qwen3

`backend/models/tsrt/modeling_tsrt.py` introduces:

1. **`TSRTCrossAttention`** — a multi-head cross-attention over `(B, D, L', H)`. Key/value are computed from the encoded documents; query comes from the decoder. Crucially, the attention logits are **biased** by `retrieval_memory`:
   ```
   attn_logits += 1 + log((retrieval_memory + 1) / 2)
   ```
   This converts per-document usefulness scores into additive attention biases.

2. **`TSRTRetrievalDecisionHead`** — an MLP producing a per-token probability `p ∈ [0,1]` of needing retrieval at this position.

3. **`TSRTRetrievalProjection`** — projects hidden states to a shared embedding space, then:
   - Uses an attention-pool over the decoder to compute a per-position `decoder_emb`.
   - Uses an attention-pool over each document to compute a per-document `doc_emb`.
   - Cosine similarity between them yields `usefulness_score ∈ [-1, 1]`.

4. **`TSRTRetrievalMemoryHead`** — combines `usefulness_score`, `retrieval_decision`, and document padding to build the `(B, L, D)` retrieval-memory tensor fed into cross-attention.

5. **`TSRTLayer`** — a single TSRT layer that runs `self-attn → cross-attn → MLP`, with three RMSNorms (input / post-self-attn / post-cross-attn).

### 2.3 Caches

`backend/models/tsrt/cache_utils.py` defines four specialized caches:

| Cache | Stores | Purpose |
| --- | --- | --- |
| `TSRTDecoderCache` | KV for `decoder + tsrt` self-attention | Standard decoder KV cache with `layer_types` sliced past the encoder. |
| `TSRTDocumentCache` | Encoder hidden state + per-layer K/V of TSRT cross-attention | Reuse encoded documents and reuse key/value across generation steps. |
| `TSRTEmbeddingCache` | Per-position retrieval-projection weights + pooled embeddings | Speeds up incremental retrieval scoring during generation. |
| `TSRTChosenDocumentCache` | The last `(chosen_documents, padding_mask, retrieval_memory)` snapshot | Lets the model keep or re-rank the previous retrieval result. |

`TSRTCache` wraps all four and conforms to Hugging Face's `Cache` interface so `model.generate()` works out of the box.

### 2.4 Configuration

`backend/models/tsrt/configuration_tsrt.py` extends `Qwen3Config`:

```python
num_decoder_layers: int = 14
num_encoder_layers: int = 14
num_tsrt_layers: int    = 14
retrieval_embedding_size: int = 1024
```

All other Qwen3 fields (hidden size 2048, 16 heads, 8 KV heads, head_dim 128, RoPE θ=1e6, sliding window disabled) are inherited.

### 2.5 Auxiliary heads and losses

`backend/models/tsrt/utils.py` defines the auxiliary losses:

| Loss | Function | Purpose |
| --- | --- | --- |
| `compute_retrieval_decision_loss` | BCE on per-token decision labels | Teach *when* to retrieve. |
| `compute_retrieval_ranking_loss` | Multi-positive InfoNCE (`temperature=0.07`) | Teach *which* document to favor. |
| `compute_retrieval_scoring_loss` | Cosine pull (positive) / margin push (negative) | Optional auxiliary — currently disabled in the trainer. |
| `compute_positive_negative_scores` | Logging only | Reports the mean cosine score of positives vs negatives per batch. |

In `modeling_tsrt.TSRTForCausalLM.forward`, the total loss is:

```python
loss = lm_loss
loss += 0.3 * retrieval_decision_loss
loss += 0.6 * (retrieval_ranking_loss / 32)
```

These weights are constants in code and can be tuned by editing `modeling_tsrt.py`.

---

## 3. Data Pipeline

The dataset is built in two phases.

### 3.1 Phase 1 — Teacher labeling (`backend/HotpotQA_Distractor/data/teacher_forcing.py`)

A stronger model, **Qwen3-8B** (4-bit NF4 quantized via BitsAndBytes), is prompted to produce a concise, evidence-based reasoning chain for each HotpotQA sample.

**Prompt template:**

```
Question:
<question>

Ground Truth Answer:
<answer>

Supporting Facts:
- <title> | sentence <id>
...

Evidence Documents:
[DOCUMENT]
Title: <title>
Content: <sentences>

Task:
Generate a concise evidence-based reasoning.

Requirements:
- When referring to information from a document, mention the document title instead of its order.

Reasoning:
```

**Generation control:**
- `max_input_length = 2048`, `max_new_tokens = 256`, `batch_size = 32`
- A custom `StoppingCriteria` halts generation when the model begins emitting `Answer`, preventing duplicate answers.
- Post-processing trims any trailing "Answer:" block and appends the ground-truth answer manually. Final format:

  ```
  <reasoning>

  Answer:
  <answer>
  ```

### 3.2 Phase 2 — Chunked dataset export (`backend/HotpotQA_Distractor/pipeline/build_teacher_dataset.py`)

- Iterates `train` and `validation` splits, calling the teacher on every sample.
- Writes results to JSONL chunks of 1,000 samples each under `teacher_chunks/`.
- This guarantees **crash recovery** and **incremental progress**.
- After all chunks are written, they are loaded into `DatasetDict({train, validation})` and pushed to the Hugging Face Hub.

**Dataset repository:** `nguyenphantuanduy/TSRT-HotpotQA-Teacher`

**Final schema** (added one field to HotpotQA):
- `id`, `question`, `answer`, `type`, `level`, `supporting_facts`, `context`, **`teacher_answer`**

### 3.3 Source statistics (HotpotQA distractor)

| Split | Samples |
| --- | ---: |
| Train | 90,447 |
| Validation | 7,405 |

### 3.4 Why teacher forcing?

The teacher supplies intermediate reasoning supervision. TSRT learns:
- When retrieval is needed (last token of each question / answer sentence).
- Which documents are useful (cosine similarity labels from supporting facts).
- How to compose evidence (reasoning trace).

This avoids the brittleness of pure answer-only supervision on multi-hop QA.

---

## 4. Training Data Collator

`backend/HotpotQA_Distractor/collator.py` converts raw dataset rows into model-ready batches.

### 4.1 `build_tsrt_document_batch`

For each sample:
- Real documents: `(title, " ".join(sentences))` for every entry in `context`.
- Extra negatives: each sample borrows one negative document from another sample in the same batch (random pick).
- Labels: `1` if title ∈ `supporting_facts`, else `0`.
- Documents are tokenized into `(B, D, L_doc)` with `document_max_length = 1024` by default.

The output includes `usefulness_score_matrix` of shape `(B, L, D)` initialized to `-1` (ignore) and filled with `0/1` for non-padding docs.

### 4.2 `build_tsrt_question_answer_batch`

Builds the question + teacher-answer sequence:

- `input_ids = tokenize(question + "\n") + tokenize(teacher_answer)`
- `labels = [-100] * len(question_ids) + answer_ids`  (mask question tokens in loss)
- `question_mask = 0` for question tokens (except the last question token), `1` for answer tokens — used to suppress retrieval decisions on the question itself.
- `retrieval_decision_labels`:
  - `-1` for padding,
  - `1` at the last token of the question and the last token of every sentence in the answer,
  - `0` elsewhere.

Sentence splitting uses a hybrid pipeline: spaCy (`en_core_web_sm`) is the primary splitter, with the `wtpsplit.SaT` model (`sat-3l-sm`) as a confidence-based fallback. Both models are loaded lazily, and spaCy is auto-downloaded on first use.

### 4.3 `fix_usefulness_score_matrix`

After the previous step, `usefulness_score_matrix` is set to `-1` everywhere except at retrieval positions. This ensures the ranking loss is only computed at the positions the model is actually expected to retrieve.

### 4.4 `TSRTDataCollator`

The collator orchestrates all three steps and returns a batch with tensors:

| Key | Shape | Meaning |
| --- | --- | --- |
| `input_ids` | `(B, L)` | Question + teacher-answer token ids. |
| `attention_mask` | `(B, L)` | Standard padding mask. |
| `question_mask` | `(B, L)` | 0 = question, 1 = answer. |
| `labels` | `(B, L)` | `-100` over question/padding, token id over answer. |
| `retrieval_decision_labels` | `(B, L)` | `-1/0/1`. |
| `document_ids` | `(B, D, L_doc)` | Tokenized documents. |
| `document_padding_mask` | `(B, D, L_doc)` | Padding mask per document. |
| `usefulness_score_matrix` | `(B, L, D)` | `-1/0/1` relevance matrix. |

---

## 5. Model Conversion: from Qwen3 to TSRT

Because TSRT shares most of Qwen3's weight structure, the first 14 layers are reused as the **decoder**, layers 14–27 are reused as the **encoder**, and layers 28–41 (which didn't exist in the 28-layer Qwen3-1.7B) are **not copied** — they are randomly initialized for the TSRT block.

Wait — Qwen3-1.7B only has 28 layers. The transfer script uses a 42-layer Qwen3 base. If you start from Qwen3-1.7B (28 layers), the script pads up to 42 by initializing the extra layers. See `backend/models/pipeline/transfer_param_from_qwen_2_tsrt.py` for the exact mapping used.

### 5.1 Pipeline steps

```
generate_tsrt_config.py         → write models/tsrt/config.json
transfer_param_from_qwen_2_tsrt.py → build TSRTForCausalLM and copy Qwen weights
check_transfer_tsrt.py          → numerically compare each module
upload_tsrt_hf.py               → push to nguyenphantuanduy/TSRT-Qwen3-1.7B
```

### 5.2 What gets copied

| Source (Qwen) | Destination (TSRT) |
| --- | --- |
| `embed_tokens` | `model.embed_tokens` |
| `norm` | `model.norm` |
| `rotary_emb` | `model.rotary_emb` |
| `lm_head` | `lm_head` |
| `layers[0..13].self_attn` | `decoder_layers[0..13]` |
| `layers[0..13].self_attn` | `encoder_layers[0..13]` |
| `layers[14..27].input_layernorm` | `tsrt_layers[*].input_layernorm` |
| `layers[14..27].post_attention_layernorm` | `tsrt_layers[*].post_self_attention_layernorm` |
| `layers[14..27].post_attention_layernorm` | `tsrt_layers[*].post_cross_attention_layernorm` |
| `layers[14..27].self_attn` | `tsrt_layers[*].self_attn` |
| `layers[14..27].self_attn` (q/k/v/o + q_norm + k_norm) | `tsrt_layers[*].cross_attn` |
| `layers[14..27].mlp` | `tsrt_layers[*].mlp` |

The cross-attention is brand new — only the linear projection weights are copied from Qwen's self-attention. The retrieval bias is added on top.

### 5.3 Verification

`check_transfer_tsrt.py` does a `torch.allclose` comparison module-by-module between the source Qwen and the produced TSRT, plus a parameter count breakdown.

---

## 6. Training Recipe

`backend/HotpotQA_Distractor/pipeline/training.py` is the canonical training script.

### 6.1 Configuration snapshot

| Setting | Value |
| --- | --- |
| Base model | `nguyenphantuanduy/TSRT-Qwen3-1.7B` (bf16) |
| Dataset | `nguyenphantuanduy/TSRT-HotpotQA-Teacher` |
| Per-device batch size | 1 |
| Gradient accumulation | 32 |
| Effective batch size | 32 |
| Optimizer | AdamW (HF Trainer default) |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Epochs | `0.4` (~36k samples seen) |
| Mixed precision | bf16 |
| Eval split | first 2,000 validation samples |
| Eval / save cadence | every 500 steps |
| Early stopping | patience = 3 evals on `eval_loss` |
| Dataloader workers | 4 |
| Document max length | 384 |

### 6.2 Parameter freezing strategy

`backend/utils/utils.py: freeze_for_tsrt_training` implements the freeze policy:

```
Trainable:
  • Encoder self-attention (q/k/v/o + norms)
  • Entire TSRT block (self-attn + cross-attn + MLP + retrieval heads)
  • Final RMSNorm

Frozen:
  • Token embedding
  • LM head
  • Decoder (all 14 layers)
  • Encoder FFN (MLP)
```

This gives ~16–18 GB VRAM at training time on a single GPU (per `research_log/2026-07-12.md`).

### 6.3 Logging

`backend/models/tsrt/trainer.py: TSRTTrainer` extends HF `Trainer` and automatically forwards `model.logged_losses` to the log dict. Each step logs:

| Key | Source |
| --- | --- |
| `lm_loss` | Standard cross-entropy on answer tokens. |
| `retrieval_decision_loss` | BCE on decision labels. |
| `retrieval_ranking_loss` | InfoNCE on usefulness scores. |
| `decision_predict` | Mean sigmoid score where label=1. |
| `non_decision_predict` | Mean sigmoid score where label=0. |
| `positive_score` / `negative_score` | Mean cosine score for pos / neg documents. |

---

## 7. Inference

A reference inference script is in `backend/main.py` (active section at the bottom of the file). The call shape is:

```python
outputs = model.generate(
    input_ids=question_ids,              # (B, L)
    attention_mask=question_attn,        # (B, L)
    document_ids=document_ids,           # (B, D, L_doc)
    document_padding_mask=doc_padding,   # (B, D, L_doc)
    max_new_tokens=128,
    do_sample=False,
    use_cache=True,
    retrieve_top_k=5,
    usefulness_threshold=0.7,
)
```

### 7.1 Retrieval-time gating

Inside `TSRTRetrievalMemoryHead.forward`, when `retrieve_top_k` or `usefulness_threshold` is set, the model:
1. Drops padded documents.
2. Optionally filters by `usefulness_threshold`.
3. Optionally keeps top-k by usefulness score.
4. Stores the chosen `(D, L', H)` documents in `TSRTChosenDocumentCache` so the next step can either re-use them or re-retrieve.

If `retrieval_decision < RETRIEVAL_DECISION_THRESHOLD` (= `0.7`), the previous retrieval is kept unchanged.

### 7.2 Auxiliary masks used at inference

- `prepare_document_attention_mask` → `(B*D, 1, 1, L)` — encoder self-attention mask.
- `prepare_cross_attention_mask` → `(B, 1, 1, D*L)` — decoder cross-attention mask.
- `prepare_projection_mask` → `(B, D, L')` — masks pad tokens before attention-pool.

---

## 8. Project Layout

```
TSRT-Task-State-Retrieval-Transformer/
├── README.md                              ← (this file)
├── SETUP.md                               ← Ubuntu VM setup (apt, venv, pip)
├── .gitignore
├── research_log/
│   ├── 2026-07-12.md                      ← Memory feasibility study
│   └── 2026-07-16.md                      ← Teacher labeling pipeline
└── backend/
    ├── requirements.txt
    ├── main.py                            ← Reference inference + ad-hoc tests
    ├── __init__.py
    │
    ├── models/tsrt/                       ← TSRT model package
    │   ├── __init__.py
    │   ├── configuration_tsrt.py          ← TSRTConfig (extends Qwen3Config)
    │   ├── modeling_tsrt.py               ← TSRTModel / TSRTForCausalLM / layers
    │   ├── modeling_outputs.py            ← TSRTModelOutputWithPast
    │   ├── cache_utils.py                 ← TSRTDecoderCache, TSRTDocumentCache,
    │   │                                    TSRTEmbeddingCache,
    │   │                                    TSRTChosenDocumentCache, TSRTCache
    │   ├── trainer.py                     ← TSRTTrainer (auto-logging)
    │   ├── utils.py                       ← Mask helpers + auxiliary losses
    │   └── config.json                    ← Saved config for HF Hub
    │
    ├── models/pipeline/                   ← Conversion utilities
    │   ├── generate_tsrt_config.py        ← Build config.json from Qwen3
    │   ├── transfer_param_from_qwen_2_tsrt.py ← Build TSRT from Qwen3
    │   ├── check_transfer_tsrt.py         ← Verify transfer + param count
    │   └── upload_tsrt_hf.py              ← Push to Hugging Face Hub
    │
    ├── utils/
    │   ├── __init__.py
    │   └── utils.py                       ← batch_tokenize_documents, freeze
    │
    ├── HotpotQA_Distractor/
    │   ├── __init__.py
    │   ├── collator.py                    ← TSRTDataCollator
    │   ├── collator_not_use.py            ← Older collator variant (kept for ref)
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── load_data.py               ← HotpotQA loader (streaming)
    │   │   └── teacher_forcing.py         ← Qwen3-8B teacher labeling
    │   ├── pipeline/
    │   │   ├── build_teacher_dataset.py   ← Chunk + push teacher dataset
    │   │   └── training.py                ← End-to-end training script
    │   └── test/                          ← Unit / smoke tests for HotpotQA pipeline
    │
    ├── test/                              ← Unit / smoke tests for the model & collator
    └── HotpotQA_Distractor/test/          ← (separate subdir for HotpotQA tests)
```

---

## 9. End-to-End Workflow

The full pipeline, in order:

```
┌────────────────────────────────────────────────────────────────────┐
│ 0.  Environment                                                     │
│     • SETUP.md on Ubuntu VM                                         │
│     • pip install -r backend/requirements.txt                       │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 1.  Generate TSRT config                                            │
│     cd backend && python -m models.pipeline.generate_tsrt_config    │
│     → writes models/tsrt/config.json                                │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 2.  Transfer weights from Qwen3                                     │
│     python -m models.pipeline.transfer_param_from_qwen_2_tsrt       │
│     → writes model.safetensors                                       │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 3.  Verify transfer                                                 │
│     python -m models.pipeline.check_transfer_tsrt                   │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 4.  Upload base TSRT model to HF Hub                                │
│     python -m models.pipeline.upload_tsrt_hf                        │
│     → nguyenphantuanduy/TSRT-Qwen3-1.7B                             │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 5.  Build the teacher-forcing dataset                                │
│     python -m HotpotQA_Distractor.pipeline.build_teacher_dataset    │
│     → nguyenphantuanduy/TSRT-HotpotQA-Teacher                       │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 6.  Train                                                            │
│     python -m HotpotQA_Distractor.pipeline.training                 │
│     → writes ./best_model/                                           │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 7.  Inference                                                        │
│     python -m main                                                   │
│     (see Section 7)                                                  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 10. Running It Locally

### 10.1 Setup

Follow `SETUP.md`:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3-dev python3-venv git
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/nguyenphantuanduy/TSRT-Task-State-Retrieval-Transformer.git
cd TSRT-Task-State-Retrieval-Transformer
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio        # pick a CUDA build if applicable
pip install -r requirements.txt
```

`requirements.txt`:
```
torch
transformers
accelerate
safetensors
sentencepiece
datasets
bitsandbytes
spacy
wtpsplit
```

### 10.2 Quick check — instantiate the model

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "nguyenphantuanduy/TSRT-Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()
```

### 10.3 Inference with documents

```python
from utils.utils import batch_tokenize_documents

questions = ["Who was the first president of the United States?"]
documents  = [["George Washington was the first president...",
               "Abraham Lincoln was the 16th president..."]]

tokenizer.padding_side = "left"
question_inputs = tokenizer(questions, return_tensors="pt", padding=True).to("cuda")
doc_inputs      = batch_tokenize_documents(documents, tokenizer, max_length=512)
document_ids    = doc_inputs["input_ids"].to("cuda")
doc_padding     = doc_inputs["attention_mask"].to("cuda")

outputs = model.generate(
    input_ids=question_inputs.input_ids,
    attention_mask=question_inputs.attention_mask,
    document_ids=document_ids,
    document_padding_mask=doc_padding,
    max_new_tokens=128,
    do_sample=False,
    use_cache=True,
    retrieve_top_k=5,
    usefulness_threshold=0.7,
)

for out in outputs:
    print(tokenizer.decode(out, skip_special_tokens=True))
```

---

## 11. Tests

The repo contains a `backend/test/` and `backend/HotpotQA_Distractor/test/` tree with smoke tests for:

- Cache behavior (`test_retrieval_mem.py`, etc.).
- Collator shape / masking (`test_pad.py`, `test_build_question_answer.py`, `test_build_doc.py`).
- Tokenization throughput (`test_num_token.py`, `test_doc_len.py`).
- Training-step correctness (`test_tsrt_batch_pipeline.py`, `test_train_hotpotqa.py`).
- Teacher mistake ratio (`test_teacher_mistake_ratio.py`).
- VRAM usage (`test_enable_VRAM.py`, `test_enable_VRAM_v2.py`).

These are runnable individually; the configuration values in `main.py` were used during development to debug the caches, cross-attention shapes, and loss functions.

---

## 12. Research Log

Two weekly notes are included:

- `research_log/2026-07-12.md` — VRAM feasibility for Qwen3-1.7B-based TSRT (16–18 GB observed).
- `research_log/2026-07-16.md` — Detailed write-up of the teacher-forcing dataset pipeline, prompt template, generation stopping, chunk-based export, and motivation for intermediate supervision.

These are recommended reading before modifying the pipeline or training configuration.

---

## 13. Key Hyperparameters & Knobs

| Where | Default | Effect |
| --- | --- | --- |
| `RETRIEVAL_DECISION_THRESHOLD` in `modeling_tsrt.py` | `0.7` | Sigmoid threshold to trigger retrieval at inference. |
| `0.3 * decision_loss + 0.6 * ranking_loss/32` weights | hard-coded | Tune auxiliary loss balance. |
| `document_max_length` in `collator.py` | `1024` | Max tokens per document; `384` in actual training. |
| `retrieval_embedding_size` | `1024` | Width of the projection used for cosine similarity. |
| `num_decoder_layers`, `num_encoder_layers`, `num_tsrt_layers` | `14, 14, 14` | Block partitioning of the 42-layer stack. |
| Freeze policy in `utils.utils.freeze_for_tsrt_training` | as listed | Controls which layers train. |

---

## 14. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | First-run spaCy | The collator auto-downloads it; verify network. |
| `KeyError: 'teacher_answer'` | Wrong dataset | Use `nguyenphantuanduy/TSRT-HotpotQA-Teacher`. |
| OOM during training | batch too high or doc length too long | Drop `document_max_length` or `per_device_train_batch_size`. |
| NaN in retrieval scoring | large cosine values | Lower learning rate or inspect `compute_retrieval_ranking_loss`. |
| `trust_remote_code=True` error | AutoConfig not loading | Confirm `models/tsrt/auto_map` is present (added automatically by `upload_tsrt_hf.py`). |

---

## 15. Citation

If you use this codebase, the dataset, or the model, please cite:

```
@software{tsrt2026,
  title  = {TSRT — Task-State Retrieval Transformer},
  author = {Nguyen Phan Tuan Duy},
  year   = {2026},
  url    = {https://github.com/nguyenphantuanduy/TSRT-Task-State-Retrieval-Transformer}
}
```

Datasets and models:

- HotpotQA: `hotpotqa/hotpot_qa` (distractor).
- Teacher: `Qwen/Qwen3-8B` (4-bit).
- Backbone: `Qwen/Qwen3-1.7B`.

---

## 16. License & Contributions

This repository is provided as a research artifact accompanying the TSRT paper. Please open an issue or PR on GitHub for questions, bug reports, or contributions.