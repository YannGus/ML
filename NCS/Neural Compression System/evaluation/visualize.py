import matplotlib.pyplot as plt
import json

def plot_results(results):
    token_sizes = [r[3] for r in results]
    mse_vals = [r[1] for r in results]
    ssim_vals = [r[2] for r in results]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(token_sizes, mse_vals, marker='o')
    plt.title("MSE vs. Token Size")
    plt.xlabel("Token Size (KB)")
    plt.ylabel("MSE")

    plt.subplot(1, 2, 2)
    plt.plot(token_sizes, ssim_vals, marker='o', color='green')
    plt.title("SSIM vs. Token Size")
    plt.xlabel("Token Size (KB)")
    plt.ylabel("SSIM")

    plt.tight_layout()
    plt.savefig("outputs/compression_vs_quality.png")
    plt.show()

def visualize_grid(indices):
    import os
    import matplotlib.image as mpimg
    config = json.load(open("config.json"))
    codebook_sizes = config["codebook_sizes"]
    base_dir = config["recon_dir"]

    fig, axes = plt.subplots(len(indices), len(codebook_sizes) + 1, figsize=(10, 8))

    for row, idx in enumerate(indices):
        original_path = os.path.join(base_dir, str(codebook_sizes[-1]), f"original_{idx}.png")
        ax = axes[row, 0]
        if os.path.exists(original_path):
            img = mpimg.imread(original_path)
            ax.imshow(img, cmap='gray')
            ax.set_title("Original", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Missing", ha='center', va='center')
        ax.axis('off')

        for col, emb in enumerate(codebook_sizes):
            recon_path = os.path.join(base_dir, str(emb), f"img_{idx}.png")
            ax = axes[row, col + 1]
            if os.path.exists(recon_path):
                img = mpimg.imread(recon_path)
                ax.imshow(img, cmap='gray')
                ax.set_title(f"Recon {emb}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "Missing", ha='center', va='center')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("outputs/comparaison_tableau_5x5.png", dpi=150)
    plt.show()
