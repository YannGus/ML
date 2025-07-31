from evaluation.visualize import visualize_grid
import random
import json

if __name__ == "__main__":
    """
    Run this if you want to visualize the grid of images
    """
    config = json.load(open("config.json"))
    total = config["num_eval_images"]
    indices = random.sample(range(total), 5)
    visualize_grid(indices)
