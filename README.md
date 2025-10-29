# Research Projects in Neural Compression, Multimodal AI, and Optimization

This repository contains my research work in **neural network compression**, **multimodal AI architectures**, and *
*training optimization**, along with practical implementations in Python and PyTorch. Each project is documented with a
research paper and accompanying code.

---

## Overview

I am **Yann Guszkiewicz**, a high school student specializing in **mathematics, physics, and engineering sciences**,
with a focus on **machine learning, neural compression, and embedded systems**. My research explores efficient AI
architectures, cross-modal representation learning, and algorithmic optimization for real-world applications.

---

## Disclaimer

I am only 16 years old and do not claim to be an expert in machine learning or AI. I share these projects purely out of
personal curiosity and passion for research. As noted in each paper, some of this work has been done with the assistance
of certain LLMs.

---

## Research Projects

### [SENSE: Shared Embeddings for Naturalistic Sensing and Episodic Memory](SENSE/SENSE%20Shared%20Embeddings%20for%20Naturalistic%20Sensing%20and%20Episodic%20Memory.pdf)

**Objective:**
SENSE proposes a novel architecture for multimodal AI systems, where vision, audio, and text modalities share a *
*unified tokenizer** and embedding space. This design facilitates direct semantic alignment across modalities and
enables scalable, real-time reasoning through distributed episodic memory.

**Key Contributions:**

- Introduced a **shared tokenizer** that maps inputs from different modalities into a single embedding space,
  eliminating the need for intermediate translations.
- Developed **mini-RAG modules**, lightweight retrieval-augmented memories that store tokenized episodes, extending the
  effective context window of large language models (LLMs) without increasing their input size.
- Enabled cross-modal attention mechanisms to dynamically fuse retrieved tokens for context-aware reasoning.

**Potential Impact:**

- Enhances the ability of AI systems to integrate and reason over multimodal inputs in real time.
- Supports privacy-preserving and edge-compatible deployments by distributing memory storage.

---

### [Flexible Neural Network with Structural Plasticity (FNNSP)](FNNSP/)

[Paper](FNNSP/Flexible%20Neural%20Network.pdf) | [Code](FNNSP/flexible_network.py)

**Objective:**
This project introduces a neural network capable of dynamically adapting its internal topology for multiple boolean classification tasks,
inspired by biological structural plasticity. Instead of retraining all weights for new tasks, the network selectively activates,
reconfigures, and specializes modular units for efficient transfer learning.

**Key Contributions:**

- Designed a **modular neural network** with worker modules controlled by a task-specific controller mask.
- Implemented **structural plasticity**, allowing selective module activation, reconnection, and specialization based on task requirements.
- Demonstrated **rapid adaptation** on MNIST and FashionMNIST binary tasks using a hybrid fine-tuning protocol, minimizing catastrophic forgetting.
- Visualized **module specialization and correlation**, confirming efficient reuse and dynamic topology adjustment across tasks.

**Results:**

- Achieved competitive performance on all tasks (average accuracy ≈ 86.6%), with near-perfect accuracy on MNIST 2vs3.
- Showed substantial topology reconfiguration between tasks, indicating effective structural plasticity and module reuse.

**Implications:**
- Supports fast transfer learning without full retraining.
- Opens avenues for adaptive AI architectures capable of handling multiple tasks efficiently with minimal weight updates.

---

### [Neural Compression System (NCS)](NCS/Neural%20Compression%20System/)

[Paper](NCS/Neural%20Compression%20System/Neural%20Compression%20System.pdf) | [Code](NCS/Neural%20Compression%20System/)

**Objective:**
Neural Compression System (NCS) investigates the use of **Vector Quantized Variational Autoencoders (VQ-VAE)** to
transform images into discrete latent codes, enabling both efficient storage and downstream task performance.
Traditional compression methods focus on pixel-level redundancy but fail to preserve semantic information. NCS addresses
this by learning compact, semantically meaningful representations.

**Key Contributions:**

- Achieved an 18:1 compression ratio on MNIST while maintaining high reconstruction quality.
- Identified that a codebook of 128 vectors provides the best balance between fidelity and efficiency for this dataset.
- Demonstrated that the architecture effectively leverages recurring patterns in structured datasets for compression.

**Applications:**

- Edge computing and embedded systems with constrained storage and computational resources.
- Unified interfaces for both data compression and inference tasks.

---

### [Neural Compression System is All You Need](NCS/Neural%20Compression%20System%20is%20all%20you%20need/)

[Paper](NCS/Neural%20Compression%20System%20is%20all%20you%20need/Neural%20Compression%20System%20is%20all%20you%20need.pdf) | [Code](NCS/Neural%20Compression%20System%20is%20all%20you%20need/)

**Objective:**
This follow-up study explores the sufficiency of discrete latent codes, generated by a pretrained VQ-VAE, for downstream
tasks such as classification. The research evaluates whether these compressed representations can serve as a minimal,
unified interface for both storage and inference.

**Key Contributions:**

- Showed that a simple classifier trained on token sequences achieves 94.76% accuracy on MNIST.
- Reduced model size from 1.20M parameters (baseline CNN) to 206k parameters (~5.8× smaller).
- Reduced training time from 128.5 seconds to 8.6 seconds (~15× faster) while retaining most of the baseline accuracy.

**Implications:**

- Supports the development of task-agnostic pipelines where a single compressed representation can be used for multiple
  purposes.
- Enables efficient transmission and processing of visual data in resource-constrained environments.

---

### [Learning Rate Predictor Optimizer (LRPO)](LRPO/Learning%20Rate%20Predictor%20Optimizer.pdf)

**Objective:**
LRPO introduces a novel approach to accelerate neural network training by predicting gradient descent trajectories and
dynamically adjusting learning rates. Traditional learning rate schedules often result in redundant training epochs,
increasing computational costs.

**Key Contributions:**

- Proposed a **predictive framework** that reduces training time by up to **84%** (from 477 seconds to 76 seconds) on
  the Fashion-MNIST dataset.
- Achieved convergence in **16 epochs** compared to 100 epochs for traditional methods, with minimal loss in accuracy (
  86.45% vs. 88.45%).
- Demonstrated that predictive learning rate scheduling can significantly improve training efficiency while maintaining
  competitive performance.

**Implications:**

- Offers a paradigm shift from reactive to proactive learning rate adjustment, enabling faster prototyping and reduced
  computational costs.
- Particularly valuable in resource-constrained environments where training time is a critical factor.

---

### [Cognitive Fingerprints: A Framework for Behavioral Prediction from Mathematical Reasoning Traces](CF/Cognitive%20Fingerprints%20A%20Framework%20for%20Behavioral%20Prediction%20from%20Mathematical%20Reasoning%20Traces.pdf)

**Objective:**
This project presents a deep learning framework for predicting an individual's unique problem-solving style, which is
referred to as their "cognitive fingerprint". The approach analyzes step-by-step reasoning, including action sequences,
temporal dynamics, and error patterns, to model a unique behavioral signature. It uses a model pre-trained on a large
population to then be rapidly personalized for a specific individual with limited data
**Key Contributions:**

- A methodology to formally define and capture fine-grained problem-solving traces as a structured dataset.

- A multi-task learning architecture that simultaneously predicts the next action, the time to complete it, and the
  likelihood of making a specific error.

- A two-stage training approach that pre-trains a population-level model and then uses efficient fine-tuning techniques,
  like LoRA, for rapid personalization with limited data.

**Implications:**

- In education, the personalized model could anticipate a student's mistakes and provide tailored feedback

- The work also includes a detailed discussion on ethical considerations, such as user privacy and avoiding sensitive
  inferences.

---

### [Neural Network Capacity from Task Difficulty](NNCTD/Neural%20Network%20Capacity%20from%20Task%20Difficulty.pdf)

**Objective:**
This work refines the Lambert W-based formula for estimating neural network capacity by incorporating **data quality**,
**architectural efficiency**, and **task complexity** factors. Traditional sizing formulas assume uniform data quality
and task difficulty, leading to suboptimal parameter estimates.

**Key Contributions:**

- Introduced an **enhanced formula** that accounts for noise, redundancy, and the intrinsic difficulty of the learning
  problem.
- Defined **correction factors** for effective dataset size, architecture-dependent parameter efficiency, and
  task-specific complexity.
- Validated the formula on multiple datasets, showing a **45% improvement** in alignment with empirically successful
  architectures.

**Implications:**

- Provides a more practical tool for initial architecture sizing, reducing parameter waste and improving generalization.
- Offers a principled approach to estimating model capacity based on dataset and task characteristics.

---

## Contact

For discussions, collaborations, or questions about this research, please reach out via:

- **Email:** [yannougus@gmail.com](mailto:yannougus@gmail.com)
- **LinkedIn:** [Yann Guszkiewicz](https://www.linkedin.com/in/yann-guszkiewicz-32332b374/)
