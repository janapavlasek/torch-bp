# PyTorch Belief Propagation

PyTorch implementation of Belief Propagation (BP) algorithms. This code accompanies the paper *Stein variational belief propagation for multi-robot coordination*. Please [cite us](#citation) if you use this code for your research.

[*Project Webpage*](https://progress.eecs.umich.edu/projects/stein-bp/)

The following algorithms are implemented (see the full citations [below](#references)):
* **Discrete Belief Propagation:** This is the standard belief propagation algorithm (Wainwright & Jordan, 2008).
* **Stein Variational Belief Propagation (SVBP):**
* **Particle Belief Propagation (PBP):** Importance sampling-based belief propagation (Ihler & McAllester, 2009).
* **Gaussian Belief Propagation (GaBP):** Based on the implementation described by Davison & Ortiz (2019). See also the excellent [interactive article](https://gaussianbp.github.io/) by the same authors.

The first three algorithms (discrete BP, SVBP, PBP) share a common base implementation, since the belief of each node can be represented by a batch of PyTorch tensors, and the standard belief propagation equations can be implemented using PyTorch operations. Gaussian BP is also implemented in PyTorch, but uses operations on Gaussian distributions which are separately implemented.

## Installation

It is recommended to install in a Python virtual environment.

Install requirements:
* Python >=3.8
* PyTorch >=2.0 (tested with up to v2.2 with CUDA 12.1)

Then install this package:
```bash
pip install -e .
```

### Development Dependencies

To run the example notebooks, you will also need to install matplotlib (>=3.7) and Jupyter.

## Usage

Example usage can be found in the provided [Jupyter notebooks](notebooks/).

TODO: Document how to define factors.

## References

* M. J. Wainwright, M. I. Jordan et al., “Graphical models, exponential families, and variational inference,” Foundations and Trends in Machine Learning, vol. 1, no. 1–2, pp. 1–305, 2008. ([link](https://people.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf))
* J. Pavlasek, J. Mah, R. Xu, O. C. Jenkins, and F. Ramos, "Stein variational belief propagation for multi-robot coordination," in Robotics and Automation Letters (RA-L), 2024. ([link](https://arxiv.org/abs/2311.16916))
* A. Ihler and D. McAllester, “Particle belief propagation,” in Artificial Intelligence and Statistics, 2009, pp. 256–263. ([link](https://proceedings.mlr.press/v5/ihler09a.html))
* A. J. Davison and J. Ortiz, “FutureMapping 2: Gaussian belief propagation for spatial AI,” arXiv preprint arXiv:1910.14139, 2019. ([link](https://arxiv.org/abs/1910.14139))

## Citation

This code accompanies the paper *Stein variational belief propagation for multi-robot coordination* (Robotics and Automation Letters, 2024). If you use it in your research, please cite:
```bibtex
@inproceedings{pavlasek2023stein,
  title={Stein Variational Belief Propagation for Multi-Robot Coordination},
  author={Pavlasek, Jana and Mah, Joshua and Xu, Ruihan and Jenkins, Odest Chadwicke and Ramos, Fabio},
  booktitle={Robotics and Automation Letters (RA-L)},
  year={2024}
}
```
