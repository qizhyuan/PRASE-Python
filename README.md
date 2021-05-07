# PRASE-Python
The code of paper "Unsupervised Knowledge Graph Alignment by Probabilistic Reasoning and Semantic Embedding" (in IJCAI)

## Package
The packages of PRASE-Python is presented below.

      ├─PRASE-Python
      │  ├─data: necessary data for running the test script
      │  ├─model: the implementation of PARIS
      │  ├─objects: Entity/Relation/KG/KGs objects of PRASEMap
      │  ├─test.py: a test script example

## Installation
### Dependencies
- Python 3.x
- Numpy

### Start
Use the following command to get a quick start:

    python test.py

This test script performs PRASE on D-W-15K-V2 with both the embedding and the mapping feedback from SE module (i.e., BootEA).
You can revise the code to customize the PRASE model.

### Usage
The core of this package is actually the Python-version implementation of [PARIS](http://webdam.inria.fr/paris/).
Since this package does not contain the implementations of embedding-based approaches, you should adopt the embedding-based implementations from external libraries to run PRASE, such as [OpenEA](https://github.com/nju-websoft/OpenEA "OpenEA").

