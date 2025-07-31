import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict
import os

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ExperimentVisualizer:
    """
    Utility class for visualizing training results and performance comparisons.
    """

    def __init__(self, save_dir: str = "results/plots") -> None:
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_training_curves(self, traditional_metrics, predictor_metrics, save_name: str = "training_curves.png") -> None:
        """
        Plots training and validation loss, accuracy, learning rate, and average epoch time.
        :param traditional_metrics: Metrics from traditional training.
        :param predictor_metrics: Metrics from predictor-based training.
        :param save_name: Output filename for the saved figure.
        :return: Nothing...
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Loss
        axes[0, 0].plot(traditional_metrics.train_losses, label="Traditional - Train", linewidth=2)
        axes[0, 0].plot(traditional_metrics.val_losses, label="Traditional - Val", linewidth=2)
        axes[0, 0].plot(predictor_metrics.train_losses, label="Predictor - Train", linestyle="--", linewidth=2)
        axes[0, 0].plot(predictor_metrics.val_losses, label="Predictor - Val", linestyle="--", linewidth=2)
        axes[0, 0].set_title("Loss Evolution", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(traditional_metrics.train_accuracies, label="Traditional - Train", linewidth=2)
        axes[0, 1].plot(traditional_metrics.val_accuracies, label="Traditional - Val", linewidth=2)
        axes[0, 1].plot(predictor_metrics.train_accuracies, label="Predictor - Train", linestyle="--", linewidth=2)
        axes[0, 1].plot(predictor_metrics.val_accuracies, label="Predictor - Val", linestyle="--", linewidth=2)
        axes[0, 1].set_title("Accuracy Evolution", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 0].plot(traditional_metrics.learning_rates, label="Traditional", linewidth=2)
        axes[1, 0].plot(predictor_metrics.learning_rates, label="Predictor", linestyle="--", linewidth=2)
        axes[1, 0].set_title("Learning Rate Evolution", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_yscale("log")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Average epoch time
        axes[1, 1].bar(
            ["Traditional", "Predictor"],
            [np.mean(traditional_metrics.epoch_times), np.mean(predictor_metrics.epoch_times)],
            color=["skyblue", "lightcoral"]
        )
        axes[1, 1].set_title("Average Epoch Time", fontsize=14, fontweight="bold")
        axes[1, 1].set_ylabel("Time (seconds)")
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches="tight")
        plt.show()

    def plot_comparison_summary(self, results_summary: Dict[str, Dict], save_name: str = "comparison_summary.png") -> None:
        """
        Plots a summary comparison of training metrics across methods.
        :param results_summary: Dictionary containing summary metrics for each method.
        :param save_name: Output filename for the saved figure.
        :return: Nothing...
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        methods = list(results_summary.keys())

        # Total time
        total_time = [results_summary[m]["total_time"] for m in methods]
        axes[0, 0].bar(methods, total_time, color=["skyblue", "lightcoral"])
        axes[0, 0].set_title("Total Training Time", fontsize=14, fontweight="bold")
        axes[0, 0].set_ylabel("Time (seconds)")
        axes[0, 0].grid(True, alpha=0.3, axis="y")

        # Best accuracy
        best_acc = [results_summary[m]["best_accuracy"] for m in methods]
        axes[0, 1].bar(methods, best_acc, color=["skyblue", "lightcoral"])
        axes[0, 1].set_title("Best Validation Accuracy", fontsize=14, fontweight="bold")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].grid(True, alpha=0.3, axis="y")

        # Convergence epoch
        conv_epoch = [results_summary[m]["convergence_epoch"] for m in methods]
        axes[0, 2].bar(methods, conv_epoch, color=["skyblue", "lightcoral"])
        axes[0, 2].set_title("Convergence Epoch", fontsize=14, fontweight="bold")
        axes[0, 2].set_ylabel("Epoch")
        axes[0, 2].grid(True, alpha=0.3, axis="y")

        # Final loss
        final_loss = [results_summary[m]["final_val_loss"] for m in methods]
        axes[1, 0].bar(methods, final_loss, color=["skyblue", "lightcoral"])
        axes[1, 0].set_title("Final Validation Loss", fontsize=14, fontweight="bold")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True, alpha=0.3, axis="y")

        # Total epochs
        total_epochs = [results_summary[m]["total_epochs"] for m in methods]
        axes[1, 1].bar(methods, total_epochs, color=["skyblue", "lightcoral"])
        axes[1, 1].set_title("Total Epochs Used", fontsize=14, fontweight="bold")
        axes[1, 1].set_ylabel("Epochs")
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        # Efficiency
        efficiency = [best_acc[i] / total_time[i] for i in range(len(methods))]
        axes[1, 2].bar(methods, efficiency, color=["skyblue", "lightcoral"])
        axes[1, 2].set_title("Training Efficiency (Accuracy / Time)", fontsize=14, fontweight="bold")
        axes[1, 2].set_ylabel("Accuracy per Second")
        axes[1, 2].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches="tight")
        plt.show()

    def create_results_table(self, results_summary: Dict[str, Dict]) -> pd.DataFrame:
        """
        Creates a formatted summary table from training results.
        :param results_summary: Dictionary containing summary metrics for each method.
        :return: Pandas DataFrame with formatted results.
        """
        df = pd.DataFrame(results_summary).T

        format_map = {
            "total_time": "{:.2f}s",
            "final_accuracy": "{:.2f}%",
            "best_accuracy": "{:.2f}%",
            "avg_epoch_time": "{:.2f}s",
            "convergence_epoch": lambda x: str(int(x)),
        }

        for col, fmt in format_map.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: fmt.format(x) if isinstance(fmt, str) else fmt(x))

        return df
