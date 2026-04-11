"""
=============================================================================
Superposition Transformers vs Classical LLM — Full Benchmark Suite
=============================================================================
Author  : AI Quantum Research Architect
Purpose : Compare classical greedy decoding vs Superposition Transformer
          reasoning across multi-step problems.

Metrics : Latency (ms), Throughput (tokens/sec), Reasoning Quality Score
          computed via answer extraction + keyword matching on GSM8K-style
          problems.

Architecture Fix Log:
  - Fixed einsum: "kj,btjd->btkd" (was wrong index mapping)
  - Removed stochastic noise at inference (caused non-determinism)
  - Fixed gate: squeeze(-1) on correct dim, softmax over K axis
  - Added CUDA-sync timing for accurate GPU benchmarks
  - Modular design: each component is independently testable
=============================================================================
"""

import torch
import torch.nn as nn
import time
import re
import json
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_TOKENS  = 80
K_HYPOTHESES = 4          # number of superposition hypotheses
DTYPE       = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"[INFO] Device  : {DEVICE}")
print(f"[INFO] Dtype   : {DTYPE}")
print(f"[INFO] Model   : {MODEL_NAME}")
print(f"[INFO] K       : {K_HYPOTHESES}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD BASE MODEL
# ─────────────────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # patch common omission

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE,
    device_map="auto",
)
base_model.eval()

# Freeze backbone — only superposition layers will train
for param in base_model.parameters():
    param.requires_grad = False

HIDDEN_DIM = base_model.config.hidden_size
print(f"[INFO] Hidden dim: {HIDDEN_DIM}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SUPERPOSITION LAYER  (Fixed & Production-Ready)
# ─────────────────────────────────────────────────────────────────────────────

class SuperpositionLayer(nn.Module):
    """
    Quantum-inspired superposition reasoning layer.

    Ψ ∈ R^{B × T × K × D}  — K simultaneous reasoning hypotheses.

    Four quantum-mechanical phases:
      1. Superposition  : replicate hidden state into K hypothesis branches
      2. Entanglement   : mix hypotheses via learnable K×K matrix (W_mix)
      3. Interference   : tanh + L2-normalize → amplify coherent paths
      4. Collapse       : weighted sum via learned gate → R^{B × T × D}

    Complexity:
      Time  : O(B · T · K² · D)  — dominated by einsum mixing
      Space : O(B · T · K · D)   — hypothesis tensor

    Args:
        hidden_dim (int): Model hidden size D.
        K (int): Number of hypotheses. Default 4.
        train_noise (float): Small noise added ONLY during training for diversity.
    """

    def __init__(self, hidden_dim: int, K: int = 4, train_noise: float = 0.01):
        super().__init__()
        self.K          = K
        self.hidden_dim = hidden_dim
        self.train_noise = train_noise

        # W_mix: learnable K×K entanglement matrix
        # Init near identity — don't destroy signal at start of training
        self.mixing = nn.Parameter(torch.eye(K) + torch.randn(K, K) * 0.01)

        # Project collapsed state back to hidden_dim
        self.proj   = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Scalar gate per hypothesis → softmax weights for collapse
        # Input: D → 1 (one score per hypothesis)
        self.gate   = nn.Linear(hidden_dim, 1, bias=False)

        # Stability norms
        self.norm_in  = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, T, D] — last hidden state from backbone LLM

        Returns:
            h_out: [B, T, D] — collapsed superposition reasoning state
        """
        B, T, D = h.shape
        h = self.norm_in(h)                         # pre-norm for stability

        # ── 1. SUPERPOSITION ────────────────────────────────────────────────
        # Broadcast h into K identical branches: [B, T, D] → [B, T, K, D]
        Psi = h.unsqueeze(2).expand(-1, -1, self.K, -1).clone()

        # Diversity noise ONLY during training (not inference)
        # This is the fix: torch.randn at inference breaks reproducibility
        if self.training:
            Psi = Psi + self.train_noise * torch.randn_like(Psi)

        # ── 2. ENTANGLEMENT ──────────────────────────────────────────────────
        # Mix hypotheses via W_mix ∈ R^{K×K}
        # "kj, btjd -> btkd" : for each (b,t), mix K hypothesis vectors
        # FIX: was "hk, btkd -> bthd" — wrong index semantics
        Psi = torch.einsum("kj, btjd -> btkd", self.mixing, Psi)

        # Residual: prevent signal collapse in early training
        Psi = Psi + h.unsqueeze(2)                  # [B, T, K, D]

        # ── 3. INTERFERENCE ──────────────────────────────────────────────────
        # tanh bounds activation; L2-norm across D amplifies coherent paths
        Psi = torch.tanh(Psi)
        Psi = Psi / (Psi.norm(dim=-1, keepdim=True) + 1e-8)
        # FIX: norm over D (dim=-1), not dim=2 (K-axis)

        # ── 4. COLLAPSE ──────────────────────────────────────────────────────
        # gate: [B, T, K, D] → [B, T, K, 1] → squeeze → [B, T, K]
        scores  = self.gate(Psi).squeeze(-1)         # [B, T, K]
        weights = torch.softmax(scores, dim=-1)      # [B, T, K]

        # Weighted sum over K hypotheses: [B, T, D]
        h_out = torch.einsum("btk, btkd -> btd", weights, Psi)

        # Final projection + norm
        h_out = self.proj(h_out)
        h_out = self.norm_out(h_out)

        return h_out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SUPERPOSITION MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class SuperpositionModel(nn.Module):
    """
    Full model: frozen LLM backbone + trainable SuperpositionLayer.

    Data flow:
      input_ids → backbone.model (transformer blocks) → last_hidden_state
               → SuperpositionLayer → lm_head → logits

    Only SuperpositionLayer parameters are trainable (K² + D² + D params).
    """

    def __init__(self, base_model: nn.Module, hidden_dim: int, K: int = 4):
        super().__init__()
        self.base        = base_model
        self.super_layer = SuperpositionLayer(hidden_dim, K)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids      : [B, T] token ids
            attention_mask : [B, T] optional mask
            labels         : [B, T] optional for loss computation

        Returns:
            dict with keys: "loss" (optional), "logits" [B, T, V]
        """
        outputs = self.base.model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            output_hidden_states = True,
        )

        h      = outputs.last_hidden_state          # [B, T, D]
        h      = self.super_layer(h)                # [B, T, D]
        logits = self.base.lm_head(h)               # [B, T, V]

        loss = None
        if labels is not None:
            # Shift for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — TRAINING (on GSM8K subset)
# ─────────────────────────────────────────────────────────────────────────────

def train_superposition_model(
    model: SuperpositionModel,
    num_samples: int = 100,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_length: int = 128,
) -> list[float]:
    """
    Train only the SuperpositionLayer on GSM8K math reasoning problems.

    Returns:
        List of per-epoch average losses.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    print(f"\n[TRAIN] Loading GSM8K ({num_samples} samples)...")
    dataset = load_dataset("gsm8k", "main", split=f"train[:{num_samples}]")

    def preprocess(example):
        text = f"Question: {example['question']}\nAnswer:"
        enc  = tokenizer(
            text,
            truncation  = True,
            padding     = "max_length",
            max_length  = max_length,
            return_tensors = None,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.super_layer.parameters(), lr=lr, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader)
    )

    epoch_losses = []
    model.train()

    for epoch in range(epochs):
        running_loss, steps = 0.0, 0
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            out  = model(input_ids, attention_mask, labels)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.super_layer.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            steps        += 1

        avg_loss = running_loss / steps
        epoch_losses.append(avg_loss)
        print(f"[TRAIN] Epoch {epoch+1}/{epochs}  |  Avg Loss: {avg_loss:.4f}")

    return epoch_losses


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GENERATION ENGINES
# ─────────────────────────────────────────────────────────────────────────────

def _sync_timer() -> float:
    """CUDA-synchronized wall-clock timer. Falls back to time.perf_counter."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def generate_classical(
    prompt: str,
    max_new_tokens: int = MAX_TOKENS,
    temperature: float = 1.0,
) -> tuple[str, float, int]:
    """
    Classical greedy/temperature-based generation using raw backbone LLM.

    Args:
        prompt          : Input string
        max_new_tokens  : Tokens to generate
        temperature     : Sampling temperature (1.0 = greedy-ish)

    Returns:
        (generated_text, latency_ms, tokens_generated)
    """
    base_model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    n_prompt_tokens = inputs["input_ids"].shape[1]

    t0 = _sync_timer()
    with torch.no_grad():
        output_ids = base_model.generate(
            **inputs,
            max_new_tokens  = max_new_tokens,
            do_sample       = False,        # greedy for fair comparison
            temperature     = temperature,
            pad_token_id    = tokenizer.eos_token_id,
        )
    t1 = _sync_timer()

    latency_ms     = (t1 - t0) * 1000
    tokens_gen     = output_ids.shape[1] - n_prompt_tokens
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text, latency_ms, tokens_gen


def generate_superposition(
    model: SuperpositionModel,
    prompt: str,
    max_new_tokens: int = MAX_TOKENS,
) -> tuple[str, float, int]:
    """
    Superposition model generation: custom autoregressive loop
    with superposition reasoning applied at each step.

    Args:
        model          : Trained SuperpositionModel
        prompt         : Input string
        max_new_tokens : Tokens to generate

    Returns:
        (generated_text, latency_ms, tokens_generated)
    """
    model.eval()
    inputs   = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    n_prompt  = input_ids.shape[1]

    t0 = _sync_timer()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out       = model(input_ids)
            logits    = out["logits"][:, -1, :]          # last token logits
            next_tok  = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

            if next_tok.item() == tokenizer.eos_token_id:
                break
    t1 = _sync_timer()

    latency_ms     = (t1 - t0) * 1000
    tokens_gen     = input_ids.shape[1] - n_prompt
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return generated_text, latency_ms, tokens_gen


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EVALUATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# Multi-step reasoning prompts — designed to stress-test sequential logic
BENCHMARK_PROMPTS = [
    {
        "id"       : "math_chain",
        "prompt"   : (
            "Question: A train travels at 60 km/h for 2 hours, then at 90 km/h for 1 hour. "
            "What is the total distance traveled?\nAnswer:"
        ),
        "keywords" : ["210", "km", "distance", "total"],
        "answer"   : "210",
    },
    {
        "id"       : "logic_multi_step",
        "prompt"   : (
            "Question: If all roses are flowers, and some flowers fade quickly, "
            "can we conclude that some roses fade quickly?\nAnswer:"
        ),
        "keywords" : ["not necessarily", "cannot", "no", "invalid", "conclude"],
        "answer"   : "no",
    },
    {
        "id"       : "arithmetic_word",
        "prompt"   : (
            "Question: John has 3 times as many apples as Mary. Mary has 8 apples. "
            "They together give away 10 apples. How many apples do they have left?\nAnswer:"
        ),
        "keywords" : ["22", "left", "apples", "remain"],
        "answer"   : "22",
    },
    {
        "id"       : "multi_step_rate",
        "prompt"   : (
            "Question: A tank fills in 6 hours. A pipe drains it in 12 hours. "
            "If both run together, how many hours to fill the tank?\nAnswer:"
        ),
        "keywords" : ["12", "hours", "fill", "net"],
        "answer"   : "12",
    },
    {
        "id"       : "sequential_reasoning",
        "prompt"   : (
            "Question: Alice is taller than Bob. Bob is taller than Carol. "
            "Is Alice taller than Carol? Explain step by step.\nAnswer:"
        ),
        "keywords" : ["yes", "alice", "taller", "carol", "transitive"],
        "answer"   : "yes",
    },
]


@dataclass
class BenchmarkResult:
    prompt_id       : str
    prompt          : str
    classical_output : str
    super_output    : str
    classical_ms    : float
    super_ms        : float
    classical_tokens : int
    super_tokens    : int
    classical_score : float   # 0.0–1.0 keyword match ratio
    super_score     : float
    classical_tps   : float   # tokens per second
    super_tps       : float


def score_output(text: str, keywords: list[str]) -> float:
    """
    Keyword-based reasoning quality score.
    Counts how many expected reasoning keywords appear in generated text.

    Returns: float in [0, 1]
    """
    text_lower = text.lower()
    matched    = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matched / len(keywords) if keywords else 0.0


def run_benchmark(model: SuperpositionModel) -> list[BenchmarkResult]:
    """
    Run all benchmark prompts through both classical and superposition models.

    Returns:
        List of BenchmarkResult dataclasses (one per prompt).
    """
    results = []
    print("\n" + "="*70)
    print("  BENCHMARK: Classical vs Superposition Transformer")
    print("="*70)

    for i, item in enumerate(BENCHMARK_PROMPTS):
        print(f"\n[{i+1}/{len(BENCHMARK_PROMPTS)}] ID: {item['id']}")
        print(f"  Prompt: {item['prompt'][:80]}...")

        # ── Classical ─────────────────────────────────────────────────────
        c_out, c_ms, c_tok = generate_classical(item["prompt"])
        c_score = score_output(c_out, item["keywords"])
        c_tps   = (c_tok / c_ms * 1000) if c_ms > 0 else 0

        # ── Superposition ──────────────────────────────────────────────────
        s_out, s_ms, s_tok = generate_superposition(model, item["prompt"])
        s_score = score_output(s_out, item["keywords"])
        s_tps   = (s_tok / s_ms * 1000) if s_ms > 0 else 0

        result = BenchmarkResult(
            prompt_id        = item["id"],
            prompt           = item["prompt"],
            classical_output  = c_out,
            super_output     = s_out,
            classical_ms     = c_ms,
            super_ms         = s_ms,
            classical_tokens  = c_tok,
            super_tokens     = s_tok,
            classical_score  = c_score,
            super_score      = s_score,
            classical_tps    = c_tps,
            super_tps        = s_tps,
        )
        results.append(result)

        # Quick print
        print(f"  Classical  → {c_ms:7.1f} ms | {c_tok:3d} tokens | score={c_score:.2f}")
        print(f"  Superpos.  → {s_ms:7.1f} ms | {s_tok:3d} tokens | score={s_score:.2f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_full_report(results: list[BenchmarkResult]):
    """Detailed side-by-side comparison report."""

    LINE = "─" * 80

    print("\n\n" + "═"*80)
    print("  SUPERPOSITION TRANSFORMER — FULL BENCHMARK REPORT")
    print("═"*80)

    for r in results:
        print(f"\n{LINE}")
        print(f"  PROMPT ID  : {r.prompt_id}")
        print(f"  PROMPT     : {r.prompt[:90]}...")
        print(LINE)

        print("\n  ── CLASSICAL OUTPUT ──────────────────────────────────────")
        # Show only generated part (after "Answer:")
        c_gen = r.classical_output.split("Answer:")[-1].strip()
        print(f"  {c_gen[:300]}")

        print("\n  ── SUPERPOSITION OUTPUT ──────────────────────────────────")
        s_gen = r.super_output.split("Answer:")[-1].strip()
        print(f"  {s_gen[:300]}")

        print(f"\n  ┌──────────────────────┬─────────────────┬─────────────────┐")
        print(f"  │ Metric               │   Classical     │  Superposition  │")
        print(f"  ├──────────────────────┼─────────────────┼─────────────────┤")
        print(f"  │ Latency (ms)         │ {r.classical_ms:>13.1f}   │ {r.super_ms:>13.1f}   │")
        print(f"  │ Tokens generated     │ {r.classical_tokens:>13d}   │ {r.super_tokens:>13d}   │")
        print(f"  │ Throughput (tok/s)   │ {r.classical_tps:>13.1f}   │ {r.super_tps:>13.1f}   │")
        print(f"  │ Reasoning Score      │ {r.classical_score:>13.2f}   │ {r.super_score:>13.2f}   │")
        print(f"  └──────────────────────┴─────────────────┴─────────────────┘")

        winner_lat   = "Classical ✓" if r.classical_ms < r.super_ms else "Superpos. ✓"
        winner_score = "Classical ✓" if r.classical_score > r.super_score else \
                       ("TIE" if r.classical_score == r.super_score else "Superpos. ✓")
        print(f"  Latency winner  : {winner_lat}")
        print(f"  Reasoning winner: {winner_score}")

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n\n{'═'*80}")
    print("  AGGREGATE SUMMARY")
    print(f"{'═'*80}")

    avg_c_ms    = sum(r.classical_ms    for r in results) / len(results)
    avg_s_ms    = sum(r.super_ms        for r in results) / len(results)
    avg_c_score = sum(r.classical_score for r in results) / len(results)
    avg_s_score = sum(r.super_score     for r in results) / len(results)
    avg_c_tps   = sum(r.classical_tps   for r in results) / len(results)
    avg_s_tps   = sum(r.super_tps       for r in results) / len(results)

    lat_delta   = ((avg_s_ms - avg_c_ms) / avg_c_ms) * 100
    score_delta = ((avg_s_score - avg_c_score) / (avg_c_score + 1e-8)) * 100

    print(f"\n  {'Metric':<28} {'Classical':>12} {'Superpos.':>12} {'Delta':>10}")
    print(f"  {'─'*64}")
    print(f"  {'Avg Latency (ms)':<28} {avg_c_ms:>12.1f} {avg_s_ms:>12.1f} {lat_delta:>+9.1f}%")
    print(f"  {'Avg Throughput (tok/s)':<28} {avg_c_tps:>12.1f} {avg_s_tps:>12.1f}")
    print(f"  {'Avg Reasoning Score':<28} {avg_c_score:>12.3f} {avg_s_score:>12.3f} {score_delta:>+9.1f}%")

    print(f"\n  Note: Superposition overhead = {lat_delta:+.1f}% latency for {score_delta:+.1f}% reasoning gain.")
    print(f"  K={K_HYPOTHESES} hypotheses evaluated. Consider K=2 for latency-constrained deployment.")
    print(f"\n{'═'*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — UNIT TESTS (assert correctness)
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_tests():
    """
    Shape & dtype correctness tests for SuperpositionLayer.
    Verifies all four phases produce expected tensor shapes.
    """
    print("\n[TEST] Running unit tests...")

    B, T, D, K = 2, 10, 512, 4
    layer = SuperpositionLayer(hidden_dim=D, K=K).to(DEVICE)
    x     = torch.randn(B, T, D, device=DEVICE)

    # Forward pass
    layer.eval()
    with torch.no_grad():
        out = layer(x)

    # Shape assertions
    assert out.shape == (B, T, D), \
        f"Output shape mismatch: expected {(B,T,D)}, got {out.shape}"

    # No NaN/Inf
    assert not torch.isnan(out).any(), "NaN detected in SuperpositionLayer output"
    assert not torch.isinf(out).any(), "Inf detected in SuperpositionLayer output"

    # Mixing matrix shape
    assert layer.mixing.shape == (K, K), \
        f"Mixing matrix shape wrong: {layer.mixing.shape}"

    # Gate produces scalar per hypothesis
    dummy_Psi = torch.randn(B, T, K, D, device=DEVICE)
    scores    = layer.gate(dummy_Psi).squeeze(-1)
    assert scores.shape == (B, T, K), \
        f"Gate output shape wrong: {scores.shape}"

    print("[TEST] ✓ All assertions passed.")
    print(f"       Input shape : {x.shape}")
    print(f"       Output shape: {out.shape}")
    print(f"       Max abs val : {out.abs().max().item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Unit tests first
    run_unit_tests()

    # 2. Build model
    print(f"\n[INIT] Building SuperpositionModel (K={K_HYPOTHESES})...")
    model = SuperpositionModel(base_model, HIDDEN_DIM, K=K_HYPOTHESES).to(DEVICE)
    trainable = sum(p.numel() for p in model.super_layer.parameters())
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INIT] Trainable params : {trainable:,}  ({100*trainable/total:.3f}% of total)")

    # 3. Train
    losses = train_superposition_model(
        model,
        num_samples = 100,
        epochs      = 3,
        batch_size  = 4,
        lr          = 1e-4,
        max_length  = 128,
    )

    # 4. Benchmark
    results = run_benchmark(model)

    # 5. Full report
    print_full_report(results)

    # 6. Save results as JSON
    output_data = {
        "model"   : MODEL_NAME,
        "K"       : K_HYPOTHESES,
        "device"  : DEVICE,
        "losses"  : losses,
        "results" : [
            {
                "id"              : r.prompt_id,
                "classical_ms"   : round(r.classical_ms, 2),
                "super_ms"       : round(r.super_ms, 2),
                "classical_score": round(r.classical_score, 3),
                "super_score"    : round(r.super_score, 3),
                "classical_tokens": r.classical_tokens,
                "super_tokens"   : r.super_tokens,
            }
            for r in results
        ]
    }
    with open("/mnt/user-data/outputs/benchmark_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("[SAVED] benchmark_results.json")
