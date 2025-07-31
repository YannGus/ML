from evaluation.evaluator import evaluate_codebook_sizes
from evaluation.visualize import plot_results

if __name__ == "__main__":
    """
    Main script to run all the project
    """
    results = evaluate_codebook_sizes()
    plot_results(results)
