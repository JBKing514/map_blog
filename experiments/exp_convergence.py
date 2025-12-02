import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import numpy as np
import os
import gc 

# ================= CONFIGURATION =================
# We compare two heterogeneous architectures to test the 
# "Substrate Independence" hypothesis of MAP.
models_to_run = [
    {
        "name": "Llama-3-8B", 
        # Using non-gated version for easier reproducibility
        "path": "NousResearch/Meta-Llama-3.1-8B-Instruct", 
        "color_map": "Blues" 
    },
    {
        "name": "Qwen-2.5-7B", 
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "color_map": "Reds" 
    }
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# MAP Prompt Annealing:
# We use semantically equivalent but structurally distinct prompts
# to initialize trajectories at different points on the manifold.
prompts = [
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

# ================= CORE FUNCTIONS =================

def extract_trajectories(model_path):
    """
    Extracts the hidden states (L1 Substrate) from the model for each prompt.
    This captures the 'cognitive trajectory' as it evolves through layers.
    """
    print(f"--- Loading {model_path} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load in FP16 to fit consumer GPUs (e.g., RTX 3090/4090/5090)
        model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True, torch_dtype=torch.float16).to(device)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

    trajectories = []
    print(f"Running inference on {len(prompts)} prompts...")
    
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # MAP Strategy: We analyze the vector of the last token at each layer.
        # This represents the aggregated semantic state at that depth.
        traj = []
        for layer_output in outputs.hidden_states:
            # layer_output shape: (batch, seq, dim)
            vec = layer_output[0, -1, :].cpu().numpy().astype(np.float32) 
            traj.append(vec)
        trajectories.append(np.array(traj))
    
    # Cleanup to prevent OOM on single-GPU setups
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"--- Unloaded {model_path} ---\n")
    
    return trajectories

def compute_pca(trajectories):
    """
    Projects high-dimensional manifold dynamics into 2D using PCA.
    Per MAP Section 4.2, we prefer Linear Projection to preserve global structure
    and verify the 'Funnel Effect'.
    """
    # Stack all points to learn a common projection basis
    all_points = np.vstack(trajectories)
    pca = PCA(n_components=2)
    all_points_2d = pca.fit_transform(all_points)
    
    # Reshape back into individual trajectories
    traj_2d = []
    start = 0
    for t in trajectories:
        end = start + len(t)
        traj_2d.append(all_points_2d[start:end])
        start = end
    return traj_2d

# ================= EXECUTION =================
results = []

# 1. Run Models Sequentially
for m in models_to_run:
    raw_traj = extract_trajectories(m["path"])
    if raw_traj is not None:
        pca_traj = compute_pca(raw_traj)
        results.append({"config": m, "data": pca_traj})

# 2. Visualization (The "Cognitive UI")
print("Plotting combined figure...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

for idx, res in enumerate(results):
    ax = axes[idx]
    config = res["config"]
    data = res["data"]
    cmap = plt.get_cmap(config["color_map"])
    colors = cmap(np.linspace(0.4, 1, len(data))) 
    
    for i, path in enumerate(data):
        ax.plot(path[:, 0], path[:, 1], marker='.', markersize=2, 
                 color=colors[i], alpha=0.7, linewidth=1)
        # Start Point (Input / Surface Layer)
        ax.scatter(path[0, 0], path[0, 1], color=colors[i], s=40, marker='x', label='Input' if i==0 else "")
        # End Point (Attractor Well / Deep Layer)
        ax.scatter(path[-1, 0], path[-1, 1], color='black', s=20, marker='o', zorder=10, label='Attractor' if i==0 else "")

    ax.set_title(f"Model: {config['name']}", fontsize=14, fontweight='bold')
    ax.set_xlabel("PC 1 (Primary Drift)", fontsize=10)
    ax.set_ylabel("PC 2 (Secondary Variance)", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.4)
    
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)

    # Annotate the funnel effect
    start_mean = np.mean([p[0] for p in data], axis=0)
    end_mean = np.mean([p[-1] for p in data], axis=0)
    ax.annotate("", xy=end_mean, xytext=start_mean, 
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3, lw=2))

plt.suptitle("Geometric Convergence of Reasoning Trajectories (MAP Validation)", fontsize=16, y=0.98)
plt.tight_layout()

save_path = "convergence_dual_model.png"
plt.savefig(save_path)
print(f"Done! Combined plot saved to {os.path.abspath(save_path)}")