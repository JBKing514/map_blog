import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
import gc

# ================= CONFIG =================

models_to_run = [
    {
        "name": "Llama-3-8B",
        "path": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "color_tight": "tab:blue",
        "color_sparse": "tab:cyan",
        "color_delta": "tab:gray"
    },
    {
        "name": "Qwen-2.5-7B",
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "color_tight": "tab:red",
        "color_sparse": "tab:orange",
        "color_delta": "tab:gray"
    }
]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Semantic tight clusters
paraphrase_prompts = [
    "Define the concept of justice.",
    "What does it mean to be fair?",
    "Explain the essence of legal equity.",
    "Describe the philosophical basis of justice.",
    "In simple terms, what is justice?",
    "Elaborate on the definition of fairness in society.",
    "What is the core principle of a just legal system?",
    "How would you define true equity?",
    "Summarize the idea of justice.",
    "Give a definition for the word justice."
]

# Semantic sparse clusters
random_prompts = [
    "What will the weather be like in Tokyo tomorrow?",
    "Give me a simple recipe for chocolate cake.",
    "Explain the Pythagorean theorem in geometry.",
    "Who won the last World Cup in football?",
    "How does photosynthesis work in plants?",
    "Write a short story about a robot who learns to paint.",
    "What are the health benefits of regular exercise?",
    "Explain the concept of inflation in economics.",
    "Describe the main causes of climate change.",
    "How can I improve my time management skills?"
]

# ================= CORE FUNCTIONS =================

def extract_layer_states(model, tokenizer, prompts):
    """
    Using the already loaded model/tokenizer,
    extract the hidden state of the last token at each level of prompts.
    """
    trajectories = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # hidden_states: tuple(num_layers+1) of (batch, seq, dim)
        layer_vecs = []
        for layer_output in outputs.hidden_states:
            vec = layer_output[0, -1, :].detach().cpu().numpy().astype(np.float32)
            layer_vecs.append(vec)

        trajectories.append(np.stack(layer_vecs, axis=0))

    return trajectories


def compute_alignment_profile(trajectories):
    """
    Calculate the layer-by-layer alignment A_ell based on a set of trajectories.
    """
    traj_arr = np.stack(trajectories, axis=0)   # (num_prompts, num_layers, dim)
    num_prompts, num_layers, dim = traj_arr.shape

    A_profile = []

    for ell in range(num_layers):
        X = traj_arr[:, ell, :]                 # (num_prompts, dim)

        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X_norm = X / norms
        sim = X_norm @ X_norm.T                 # (num_prompts, num_prompts)

        iu = np.triu_indices(num_prompts, k=1)
        sims = sim[iu]
        if len(sims) == 0:
            A_profile.append(0.0)
            continue

        A_layer = np.mean((sims + 1.0) / 2.0)
        A_profile.append(float(A_layer))

    return np.array(A_profile, dtype=np.float32)


# ================= EXECUTION =================

def main():
    results = []

    for cfg in models_to_run:
        print(f"=== Processing model: {cfg['name']} ===")

        # ---- Load only once ----
        print(f"--- Loading {cfg['path']} ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                output_hidden_states=True,
                dtype=dtype
            ).to(device)
        except Exception as e:
            print(f"[ERROR] Failed to load {cfg['path']}: {e}")
            continue

        # Tight
        print("Running tight (paraphrase) prompts...")
        tight_traj = extract_layer_states(model, tokenizer, paraphrase_prompts)
        A_tight = compute_alignment_profile(tight_traj)

        # Sparse
        print("Running sparse (random) prompts...")
        sparse_traj = extract_layer_states(model, tokenizer, random_prompts)
        A_sparse = compute_alignment_profile(sparse_traj)

        # ---- Unload ----
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        print(f"--- Unloaded {cfg['path']} ---\n")

        # Alignment layers
        num_layers = min(len(A_tight), len(A_sparse))
        A_tight = A_tight[:num_layers]
        A_sparse = A_sparse[:num_layers]
        DeltaA = A_tight - A_sparse

        results.append({
            "config": cfg,
            "A_tight": A_tight,
            "A_sparse": A_sparse,
            "DeltaA": DeltaA,
            "num_layers": num_layers
        })

        # Print ΔA Overview
        L = np.arange(num_layers)
        mid = num_layers // 2

        print(f"Model: {cfg['name']}")
        print(f"  Layers: {num_layers}")
        print(f"  A_tight (first/mid/last)  = "
              f"{A_tight[0]:.4f}, {A_tight[mid]:.4f}, {A_tight[-1]:.4f}")
        print(f"  A_sparse(first/mid/last)  = "
              f"{A_sparse[0]:.4f}, {A_sparse[mid]:.4f}, {A_sparse[-1]:.4f}")
        print(f"  DeltaA (first/mid/last)  = "
              f"{DeltaA[0]:.4f}, {DeltaA[mid]:.4f}, {DeltaA[-1]:.4f}")
        print(f"  mean DeltaA over layers  = {DeltaA.mean():.4f}")
        print("")

    if not results:
        print("No models successfully processed, abort plotting.")
        return

    # ===== Plot =====
    print("Plotting alignment + ΔA profiles...")
    n_models = len(results)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(6 * n_models, 5),
        dpi=300,
        sharey=True
    )
    if n_models == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cfg = res["config"]
        A_tight = res["A_tight"]
        A_sparse = res["A_sparse"]
        DeltaA = res["DeltaA"]
        L = np.arange(len(A_tight))

        ax.plot(
            L, A_tight,
            marker='o', linewidth=1.5, markersize=3,
            label="Tight semantics A",
            color=cfg["color_tight"]
        )
        ax.plot(
            L, A_sparse,
            marker='s', linewidth=1.5, markersize=3,
            linestyle='--',
            label="Sparse semantics A",
            color=cfg["color_sparse"]
        )
        ax.plot(
            L, DeltaA,
            linewidth=1.2, linestyle='-.',
            label="ΔA = A_tight - A_sparse",
            color=cfg["color_delta"]
        )

        ax.set_xlabel("Layer Index")
        ax.set_title(cfg["name"])
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Alignment A / ΔA")
    fig.suptitle(
        "Layer-wise Alignment A for Tight vs Sparse Semantics\n"
        "(and ΔA = A_tight - A_sparse)",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = "alignment_delta_profile.png"
    plt.savefig(save_path)
    print(f"Done! Alignment + ΔA plot saved to {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()
