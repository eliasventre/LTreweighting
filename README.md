# LTreweighting
This repository allows to reproduce the figures of the paper Ventre et al. [1], which describes a method of trajectory inference for scRNA-seq data with lineage tracing. 
Note that some of the functions in these files, in particular the ones to simulate branching SDEs and to solve a convex minimization problem using Mean-field Langevin dynamics, are directly taken or adapted from two previously published papers, [2] and [3] respectively. This is of course always specified at the head of the files.

[1] Ventre, E., Forrow, A., Gadhiwala, N., Chakraborty, P., Angel, O., & Schiebinger, G. (2023). Trajectory inference for a branching SDE model of cell differentiation. arXiv preprint arXiv:2205.19145.

[2] Forrow, A., & Schiebinger, G. (2021). LineageOT is a unified framework for lineage tracing and trajectory inference. Nature communications, 12(1), 4940.

[3] Chizat, L., Zhang, S., Heitz, M., & Schiebinger, G. (2022). Trajectory inference via mean-field Langevin in path space. arXiv preprint arXiv:2205.07146.

# Requirements
First, clone this repository with

```bash
git clone https://github.com/eliasventre/LTreweighting
```

Dependencies are listed in `pip.requirements`. They can be installed with

```bash
pip install -r pip_requirements.txt
```
# Approximate execution time

The times expected for running the python files on a personal laptop are the following:

Figure2.py : about 15 minutes,

Figure3.py: about an hour,

Figure4.py: about a minute,

Figure5.py: about 20 minutes,

Figure6.py: about two hours.


You should then be able to run the script for the figure you want to reproduce.
