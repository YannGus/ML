from experiments.traditional_training import run_traditional_experiment
from experiments.predictor_training import run_predictor_experiment
from utils.visualization import ExperimentVisualizer
from utils.metrics import ExperimentMetrics
from typing import Dict, List, Tuple, Any
from datetime import datetime
from config import Config
import numpy as np
import torch
import json
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def setup_experiment() -> None:
    """
    Setup the experiment environment, including directories and random seeds
    """
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)

    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.RANDOM_SEED)


def run_single_experiment() -> Tuple[Dict[str, Any], ExperimentMetrics, ExperimentMetrics]:
    """
    Run a single experiment with traditional and predictor training methods
    :return: Tuple containing results summary, traditional metrics, and predictor metrics
    """
    setup_experiment()

    traditional_metrics: ExperimentMetrics = run_traditional_experiment()
    traditional_summary: Dict[str, Any] = traditional_metrics.get_summary()

    predictor_metrics: ExperimentMetrics = run_predictor_experiment()
    predictor_summary: Dict[str, Any] = predictor_metrics.get_summary()

    results_summary: Dict[str, Dict[str, Any]] = {
        'Traditional': traditional_summary,
        'Predictor': predictor_summary
    }

    print_results_summary(results_summary)

    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file: str = os.path.join(Config.RESULTS_DIR, f'experiment_results_{timestamp}.json')

    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=4)

    visualizer = ExperimentVisualizer(Config.PLOTS_DIR)
    visualizer.plot_training_curves(traditional_metrics, predictor_metrics, f'training_curves_{timestamp}.png')
    visualizer.plot_comparison_summary(results_summary, f'comparison_summary_{timestamp}.png')

    results_table = visualizer.create_results_table(results_summary)
    print(results_table.to_string())

    return results_summary, traditional_metrics, predictor_metrics


def run_multiple_experiments(num_runs: int = 3) -> Dict[str, Any]:
    """
    Run multiple experiments for statistical analysis
    :param num_runs: Number of experimental runs
    :return: Dictionary containing statistical results and individual run metrics"""
    setup_experiment()

    all_traditional: List[Dict[str, Any]] = []
    all_predictor: List[Dict[str, Any]] = []

    for run in range(num_runs):
        torch.manual_seed(Config.RANDOM_SEED + run)
        np.random.seed(Config.RANDOM_SEED + run)

        traditional_metrics = run_traditional_experiment()
        all_traditional.append(traditional_metrics.get_summary())

        predictor_metrics = run_predictor_experiment()
        all_predictor.append(predictor_metrics.get_summary())

    traditional_stats: Dict[str, Dict[str, float]] = calculate_statistics(all_traditional)
    predictor_stats: Dict[str, Dict[str, float]] = calculate_statistics(all_predictor)

    print_statistical_results(traditional_stats, predictor_stats)

    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file: str = os.path.join(Config.RESULTS_DIR, f'statistical_results_{timestamp}.json')

    results: Dict[str, Any] = {
        'traditional_stats': traditional_stats,
        'predictor_stats': predictor_stats,
        'num_runs': num_runs,
        'individual_runs': {
            'traditional': all_traditional,
            'predictor': all_predictor
        }
    }

    with open(stats_file, 'w') as f:
        json.dump(results, f, indent=4)

    return results


def calculate_statistics(results_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics (mean, std, min, max) for a list of results
    :param results_list: List of dictionaries containing results from multiple runs
    :return: Dictionary containing statistics for each metric
    """
    stats: Dict[str, Dict[str, float]] = {}
    metrics = results_list[0].keys()

    for metric in metrics:
        values = [result[metric] for result in results_list]
        valid_values = [v for v in values if v is not None]

        if valid_values:
            stats[metric] = {
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values))
            }
        else:
            stats[metric] = {'mean': None, 'std': None, 'min': None, 'max': None}

    return stats


def print_results_summary(results_summary: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a summary of the results from the experiment
    :param results_summary: Dictionary containing results for traditional and predictor methods
    :return: Nothing...
    """
    print(f"{'Metric':<25} {'Traditional':<15} {'Predictor':<15} {'Improvement':<15}")
    print("-" * 70)

    traditional = results_summary['Traditional']
    predictor = results_summary['Predictor']

    time_improvement = ((traditional['total_time'] - predictor['total_time']) / traditional['total_time']) * 100
    print(f"{'Total Time (s)':<25} {traditional['total_time']:<15.2f} {predictor['total_time']:<15.2f} {time_improvement:<15.1f}%")

    acc_diff = predictor['best_accuracy'] - traditional['best_accuracy']
    print(f"{'Best Accuracy (%)':<25} {traditional['best_accuracy']:<15.2f} {predictor['best_accuracy']:<15.2f} {acc_diff:<15.2f}")

    epoch_improvement = ((traditional['convergence_epoch'] - predictor['convergence_epoch']) / traditional['convergence_epoch']) * 100
    print(f"{'Convergence Epoch':<25} {traditional['convergence_epoch']:<15} {predictor['convergence_epoch']:<15} {epoch_improvement:<15.1f}%")

    total_epoch_reduction = ((traditional['total_epochs'] - predictor['total_epochs']) / traditional['total_epochs']) * 100
    print(f"{'Total Epochs':<25} {traditional['total_epochs']:<15} {predictor['total_epochs']:<15} {total_epoch_reduction:<15.1f}%")

    loss_improvement = ((traditional['final_val_loss'] - predictor['final_val_loss']) / traditional['final_val_loss']) * 100
    print(f"{'Final Val Loss':<25} {traditional['final_val_loss']:<15.4f} {predictor['final_val_loss']:<15.4f} {loss_improvement:<15.1f}%")


def print_statistical_results(traditional_stats: Dict[str, Dict[str, float]], predictor_stats: Dict[str, Dict[str, float]]) -> None:
    """
    Print statistical results comparing traditional and predictor methods
    :param traditional_stats: Dictionary containing statistics for traditional method
    :param predictor_stats: Dictionary containing statistics for predictor method
    :return: Nothing..."""
    print(f"{'Metric':<25} {'Traditional (μ±σ)':<20} {'Predictor (μ±σ)':<20} {'p-value':<10}")
    print("-" * 75)

    key_metrics = ['total_time', 'best_accuracy', 'convergence_epoch', 'total_epochs']

    for metric in key_metrics:
        if metric in traditional_stats and metric in predictor_stats:
            trad_mean = traditional_stats[metric]['mean']
            trad_std = traditional_stats[metric]['std']
            pred_mean = predictor_stats[metric]['mean']
            pred_std = predictor_stats[metric]['std']

            if trad_mean is not None and pred_mean is not None:
                print(f"{metric:<25} {trad_mean:.2f}±{trad_std:.2f:<12} {pred_mean:.2f}±{pred_std:.2f:<12} {'TBD':<10}")


def generate_research_paper_data(results: Dict[str, Any]) -> None:
    """
    Generate a text file with research paper data based on the experiment results
    :param results: Dictionary containing statistical results and individual run metrics
    :return: Nothing...
    """
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_data_file: str = os.path.join(Config.RESULTS_DIR, f'research_paper_data_{timestamp}.txt')

    with open(paper_data_file, 'w') as f:
        f.write("LEARNING RATE PREDICTOR EXPERIMENT - RESEARCH DATA\n")
        f.write("=" * 60 + "\n\n")

        f.write("ABSTRACT DATA:\n")
        f.write("-" * 20 + "\n")
        if 'traditional_stats' in results:
            trad_time = results['traditional_stats']['total_time']['mean']
            pred_time = results['predictor_stats']['total_time']['mean']
            time_reduction = ((trad_time - pred_time) / trad_time) * 100

            f.write(f"Average training time reduction: {time_reduction:.1f}%\n")
            f.write(f"Traditional method: {trad_time:.2f}s ± {results['traditional_stats']['total_time']['std']:.2f}s\n")
            f.write(f"Predictor method: {pred_time:.2f}s ± {results['predictor_stats']['total_time']['std']:.2f}s\n\n")

        f.write("METHODOLOGY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Dataset: {Config.DATASET}\n")
        f.write(f"Model architecture: Simple CNN\n")
        f.write(f"Traditional epochs: {Config.TRADITIONAL_EPOCHS}\n")
        f.write(f"Pre-training epochs: {Config.PRETRAINING_EPOCHS}\n")
        f.write(f"Fine-tuning epochs: {Config.FINE_TUNING_EPOCHS}\n")
        f.write(f"Number of experimental runs: {results.get('num_runs', 1)}\n\n")

        f.write("RESULTS FOR PAPER:\n")
        f.write("-" * 20 + "\n")


if __name__ == "__main__":
    print("Learning Rate Predictor Experiment")
    print("Choose experiment type:")
    print("1. Single experiment (quick test)")
    print("2. Multiple experiments (statistical analysis)")

    choice: str = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        results_summary, trad_metrics, pred_metrics = run_single_experiment()
    elif choice == "2":
        num_runs: int = int(input("Enter number of runs (default 3): ") or "3")
        results = run_multiple_experiments(num_runs)
        generate_research_paper_data(results)
    else:
        print("Invalid choice. Please run again and select 1 or 2.")
