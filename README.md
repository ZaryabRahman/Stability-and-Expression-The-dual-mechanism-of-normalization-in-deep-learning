# "Stability and Expression: The Dual-Mechanism of Normalization in Deep Learning"

![Journal Status](https://img.shields.io/badge/Journal%20Status-Under%20Review%20(SN%20Computer%20Science)-blue)

This repository contains the official PyTorch implementation for the experiments in the paper "Stability and Expression: The Dual-Mechanism of Normalization in Deep Learning" by Zaryab Rahman.

## Abstract

The remarkable ability of over-parameterized Transformer models to generalize is largely attributed to an implicit regularization that guides optimization towards "simple" solutions, yet the precise nature of this regularization remains poorly understood. While normalization layers like LayerNorm are critical for stable training, their role is often viewed as a simple optimization aid. In this work, we challenge this view and demonstrate that normalization is a powerful source of implicit geometric regularization. We propose that normalization’s success stems from a crucial two-part mechanism: a non-parametric geometric constraint that prevents representational collapse by forcing activations onto a hypersphere, and a learnable affine transformation that acts as an expressive engine to find high-performance solutions.
We systematically dissect this mechanism through a multi-stage empirical investigation. First, using a series of controlled experiments on a novel synthetic task, we isolate and verify the distinct roles of the two components. Second, we validate our hypothesis in a deep Transformer trained on a real-world task. This decisive experiment provides direct, layer-wise evidence that the geometric constraint prevents the pathological phenomenon of cascading representational collapse, thereby preserving the model's expressive capacity. Our findings re-frame normalization not as an engineering trick, but as a fundamental architectural principle that actively sculpts the geometry of the solution space, providing a new, causal link between a specific architectural component, the quality of learned representations, and the generalization capabilities of deep learning models.

## The Dual-Mechanism Hypothesis

Our central finding is that the success of normalization layers hinges on a sophisticated, two-part mechanism:

1.  **The Geometric Stabilizer:** The core act of normalizing activation vectors (e.g., via L2 norm). This non-parametric operation forces activations onto a hypersphere, preventing representational collapse and guaranteeing convergence to a stable, flat region of the loss landscape.

2.  **The Expressive Engine:** The learnable affine transformation (`gain` and `bias` parameters). After geometric stability is established by the stabilizer, these parameters provide the model the expressive freedom to rescale and shift its representations to find a high-performance solution within that stable region.

## Repository Structure

This repository is organized into a series of Jupyter notebooks, each corresponding to a key experiment in the paper.

-   `Experiment_1_Foundational_Effect_of_LayerNorm.ipynb`: Reproduces **Figure 1**. Compares a minimalist Transformer with and without LayerNorm to establish the foundational effect on the loss landscape and representation quality.
-   `Experiment_2_Generalizing_the_Principle_Beyond_LayerNorm.ipynb`: Reproduces **Figure 2**. Compares LayerNorm and BatchNorm to show that the regularization effect is a general principle, not specific to LayerNorm.
-   `Experiment_3_Decomposing_the_Mechanism.ipynb`: Reproduces **Figure 3**. The decisive test that introduces a "Manual L2 Norm" to isolate the Geometric Stabilizer from the Expressive Engine.
-   `Experiment_4_Real-World_Validation_on_AG_News.ipynb`: Reproduces **Figure 4**. Validates the dual-mechanism hypothesis in a deep, 6-layer Transformer on a real-world text classification task, demonstrating the effect of cascading representational collapse.
-   `requirements.txt`: A list of required Python packages to run the experiments.
-   `README.md`: This file.

## Setup and Installation

To run these experiments, you will need a Python environment with PyTorch and the Hugging Face libraries. A GPU is highly recommended, especially for Experiment 4.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ZaryabRahman/Stability-and-Expression-The-dual-mechanism-of-normalization-in-deep-learning/tree/main
    cd Stability-and-Expression-The-dual-mechanism-of-normalization-in-deep-learning
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r req.txt
    ```

## Running the Experiments

The experiments are designed to be run sequentially within their respective Jupyter notebooks (`.ipynb` files). You can open them using Jupyter Lab or Google Colab.

-   **Synthetic Task (Experiments 1-3):**
    -   Open and run the cells in `Experiment_1_...`, `Experiment_2_...`, and `Experiment_3_...` in order.
    -   Each notebook is self-contained and will train the necessary models and generate the corresponding plots from the paper.
    -   These experiments run quickly on a standard GPU (approx. 1-2 minutes each).

-   **Real-World Validation (Experiment 4):**
    -   Open and run the cells in `Experiment_4_...`.
    -   This notebook will automatically download the AG News dataset and the `distilbert-base-uncased` tokenizer from Hugging Face.
    -   Training takes longer (approx. 20-30 minutes on a T4 GPU in Colab). It will train all three deep models and then generate the layer-wise SVD plot (Figure 4).

### Expected Results

When you run the notebooks, the generated plots should closely match the figures in the paper:

-   **Experiment 1 (Figure 1):** You will see that the LayerNorm model converges to a flat minimum and has a slowly decaying (isotropic) activation spectrum, while the No Norm model converges to a sharp minimum and has a rapidly decaying (collapsed) spectrum.

-   **Experiment 3 (Figure 3):** You will observe that the **Manual L2 Norm** model (purple) has a flat loss landscape and isotropic spectrum, just like LayerNorm and BatchNorm. However, its training loss will stagnate at a high value, demonstrating that the geometric stabilizer alone is insufficient for high performance.

-   **Experiment 4 (Figure 4):** The plots will show the spectra for layers 1, 3, and 6. The **No Norm** model (red) will show a progressively sharper decay from layer to layer (cascading collapse). In contrast, the **LayerNorm** (blue) and **Manual L2 Norm** (purple) models will maintain their isotropic representations across all depths. The final training accuracies printed below the plot will confirm the superior performance of the complete LayerNorm mechanism.

## Citation

If you find this work useful in your research, please consider citing our paper (details will be updated upon publication).

```bibtex
@misc{rahman2025stability,
      title={Stability and Expression: The Dual-Mechanism of Normalization in Deep Learning}, 
      author={Zaryab Rahman},
      year={2025},
      eprint={},
      archivePrefix={},
      primaryClass={},
      note={Under review at SN Computer Science journal}
}
