# Neural quantum state and Machine Learning for Many-body Physics Lecture Note University of Dschang
This repository contains the lecture notes and code for the lecture series **Machine Learning and AI for Many-body Physics** I delivered at University of Dschang from February 2026. The implemenation is an adaptation of https://link.aps.org/doi/10.1103/q8p7-k7ms

# Summary

This lecture series is intended for any machine learning (ML) or artificial intelligence (AI) enthusiast, regardless of background or prior knowledge, provided they have minimal coding experience. The goal is to familiarize students with ML tools that can be applied to physical problems. In particular, we will explore how neural networks can be used to represent quantum many-body states and how such representations allow us to extract physical properties such as ground-state energies and quantum phase transitions. We will also employ methods such as exact diagonalization, which will serve as a benchmark for our neural-network-based quantum states. While no prior knowledge of ML is required, students should be comfortable with quantum mechanics, including—but not limited to—Dirac notation, the Schrödinger equation, and the variational principle.
Part of this lecture will be adapted from Florian Marquardt's course `Machine Learning for Physicists: Neural Networks and Their Application`, and from <a href="https://link.aps.org/doi/10.1103/q8p7-k7ms" target="_blank">our paper</a>. The lecture will be delivered in English and French, and each student is expected to have a laptop with stable internet.

## Content
This repository contains the following folders:
* **ED**: an implementation of exact diagonalization (ED) for the the 1D Hermitian Transverse-field Ising Model (TFIM). To run the code, run the file `ED_NH_core.py`.

* **RNN**: an implementation of the RNN Wave Function for the finding of the ground state of the 1D Hermitian Transverse-field Ising Model.

* **RBM**: an implementation of the RBM Wave Function for the finding of the ground state of the 1D non-Hermitian PT-symmetric Transverse-field Ising Model.

* **MLP**:an implementation of the MLP Wave Function for the finding of the ground state of the 1D non-Hermitian PT-symmetric Transverse-field Ising Model.


To learn more about this approach, you can check out our paper on Physical Review Research: https://link.aps.org/doi/10.1103/q8p7-k7ms; ArXiv https://arxiv.org/abs/2506.11222

For further questions or inquiries, please feel free to send an email to lavoisier.wahkenounouh@mpl.mpg.de. We are looking forward to future contributions.

## Dependencies
Our implementation works with the packages in the `requirements.txt` file (for MacOs users), or `requirements2.txt` file (for Windows users). They can be installed by running:
```
pip install -r requirements.txt
```
## Clone repository

```
git clone https://github.com/Kenounouh/Machine-Learning-and-AI-for-Many-body-Physics-Lecture-Note-University-of-Dschang.git
```

## Run
For simplicity all the NQSs where stored as jupyter notebooks.

## Citing
```bibtex
@article{q8p7-k7ms,
  title = {Many-body neural network wavefunction for a non-Hermitian Ising chain},
  author = {Wah, Lavoisier and Zen, Remmy and Kunst, Flore K.},
  journal = {Phys. Rev. Res.},
  volume = {7},
  issue = {4},
  pages = {043291},
  numpages = {11},
  year = {2025},
  month = {Dec},
  publisher = {American Physical Society},
  doi = {10.1103/q8p7-k7ms},
  url = {https://link.aps.org/doi/10.1103/q8p7-k7ms}
}
```
## License
The [license](https://github.com/Kenounouh/Machine-Learning-and-AI-for-Many-body-Physics-Lecture-Note-University-of-Dschang/edit/main/LICENSE) of this work is derived from the BSD-3-Clause license. Ethical clauses are added to promote good uses of this code.

# Neural Quantum States and Machine Learning for Physics

## Lecture Overview

The lecture consists of **7** lecture sessions and **6** tutorials, for a total of **28 hours**, structured as follows:

- **Session 0:** Introduction (exceptionally in French)
- **Lecture 1:** Fundamentals of machine learning
- **Tutorial 1:** Constructing a neural network from scratch
- **Lecture 2:** Training and analyzing a neural network
- **Tutorial 2:** Training a neural network
- **Lecture 3:** Exact diagonalization: sparse methods and `NetKet`
- **Tutorial 3:** Performing exact diagonalization on an Ising chain and extracting physical observables
- **Lecture 4:** Multilayer perceptrons
- **Tutorial 4:** Building a neural quantum state — Part 1
- **Lecture 5:** Boltzmann machines
- **Tutorial 5:** Building a neural quantum state — Part 2
- **Lecture 6:** Recurrent neural networks
- **Lecture 7:** Transfer learning
- **Tutorial 6:** Building a neural quantum state — Part 3

## Lectures & Tutorials

### Session 0: Introduction

This session serves as both the setup phase and an opportunity to get to know one another. During this session, we will install all required packages, the appropriate Python version, and the Python interpreter. Students will also need to create a **GitHub** account at [https://github.com](https://github.com) to access all notebooks and course material.

The following should be completed prior to this session:
- A stable internet connection
- Download the Python-friendly editor **VS Code** ([https://code.visualstudio.com/download](https://code.visualstudio.com/download)) **do not install it yet!**
- Create a **GitHub** account at [https://github.com](https://github.com) (optional)
- Download `Python 3.11.5` but **do not install it yet!**
- **Students with a stable internet connection (during the whole lecture series) but less powerful laptops may run all simulations remotely on [https://colab.google](https://colab.google) and therefore do not need to install anything locally.**

**Note:** **A stable internet connection is required during this session in order to install all additional dependencies!**

This lecture is also devoted to a concise introduction to basic Python packages, including `NumPy`, `SciPy`, `Tensorflow`, `Numba`, and `Matplotlib`.

---

### Lecture 1: Fundamentals of Machine Learning
Introduction to key concepts of machine learning, including supervised and unsupervised learning. Overview of common architectures such as feedforward neural networks. Basic Python packages like `NumPy`, `SciPy`, and `Matplotlib` will be demonstrated.

**Tutorial 1: Constructing a Neural Network from Scratch**

Hands-on session to build a simple neural network without external libraries. Students will implement forward and backward propagation in Python. The focus is on understanding the internal structure and operations of a neural network.

---

### Lecture 2: Training and Analyzing a Neural Network
Covers optimization techniques, loss functions, and evaluation metrics. Students learn how to train networks effectively and interpret their performance. Practical examples with Python will illustrate model analysis and debugging.

**Tutorial 2: Training a Neural Network**

Students train a neural network on a sample dataset using Python. Techniques such as gradient descent and mini-batching are applied. Emphasis on visualizing training dynamics and understanding convergence behavior.

---

### Lecture 3: Exact Diagonalization: Sparse Methods and `NetKet`
Introduction to exact diagonalization for quantum systems. Sparse matrix techniques and the `NetKet` library will be presented. Applications to small spin chains and benchmarking methods are discussed.

**Tutorial 3: Performing Exact Diagonalization on an Ising Chain and Extracting Physical Observables**

Hands-on implementation of exact diagonalization on an Ising chain. Students compute eigenvalues, eigenvectors, and observables. Results are compared to theoretical predictions and analyzed.

---

### Lecture 4: Multilayer Perceptrons
Covers feedforward neural networks with multiple hidden layers. Discussion on activation functions, network depth, and representational power. Applications to approximating quantum many-body states are introduced.

**Tutorial 4: Building a Neural Quantum State — Part 1**

Students implement a neural network to represent a quantum wavefunction. Focus on constructing the network architecture and encoding spin configurations. Initial training setup is demonstrated.

---

### Lecture 5: Boltzmann Machines
Introduction to energy-based models, including Restricted Boltzmann Machines (RBMs). Theory behind probabilistic neural networks and their applications to physics. Connections to neural quantum states are discussed.

**Tutorial 5: Building a Neural Quantum State — Part 2**

Hands-on implementation of RBM-based quantum states. Students learn how to train the network to approximate the ground state. Methods to evaluate accuracy and physical properties are demonstrated.

---

### Lecture 6: Recurrent Neural Networks
Covers architectures suited for sequential data, such as RNNs and LSTMs. Emphasis on autoregressive models for representing quantum states. Examples and potential advantages over other architectures are highlighted.

---

### Lecture 7: Transfer Learning
Introduction to transfer learning concepts in machine learning. Applications to neural quantum states, including reusing pre-trained networks for new problems. Benefits in efficiency and accuracy are discussed.

**Tutorial 6: Building a Neural Quantum State — Part 3**

Final hands-on session integrating previous lessons. Students refine and train neural quantum states using RNNs and transfer learning. Comparison with exact diagonalization results and performance evaluation are performed.

**Mini-project:** Recover the results of Ref. [wah2025many] (optional).

---

## Additional Information

The lecture will be held online at the University of Dschang once per week, starting in **January 2026** (exact dates to be determined), and will be supervised on-site by *Prof. Christian Sadem*.

**Location:** Condensed Matter Laboratory, **B1.5**, Physics Department, University of Dschang, Cameroon.




