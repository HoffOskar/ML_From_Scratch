# ML From Scratch

Welcome to the ML From Scratch repository! This project aims to implement various machine learning algorithms from scratch using Python. The goal is to understand the inner workings of these algorithms by building them without relying on any machine learning libraries.

Some of the work in this repository was done as part of the practical course **"Mathematics of Machine Learning"** at the **University of Graz**.

## Algorithms Implemented
Each algorithm is described in a **jupyter notebook**, implemented in a **utility class**, and demonstrated in a simple **example file**:

**Regression**
- [Ridge Regression](Regression/Ridge.ipynb) [`exp.py`](example_Ridge.py) | [utility class](utils/linear.py)
- [Decision Tree](Regression/RegTree.ipynb) [`exp.py`](example_RegTree.py) | [utility class](utils/tree.py)
- [Shallow Neural Network](Regression/MLP.ipynb) [`exp.py`](example_MLP.py) | [utility class](utils/networks.py)

**Classificaton**
- [k-Nearest Neighbor](Classification/KNN.ipynb) [`exp.py`](example_KNN_clf.py) | [utility class](utils/knn.py)
- [Binary Logistic Regression](Classification/LogReg.ipynb) [`exp.py`](example_LogReg.py) | [utility class](utils/linear.py)

**Clustering**
- [KMeans Clustering](Clustering/KMeans.ipynb) [`exp.py`](example_KMeans.py) | [utility class](utils/knn.py)
- [Butina Clustering](Clustering/Butina.ipynb) [`exp.py`](example_Butina.py) | [utility class](utils/chem.py)
- [Kennard-Stone Splitting](Clustering/KennardStone.ipynb) [`exp.py`](example_KennardStone.py) | [utility class](utils/chem.py)

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/HoffOskar/ML_From_Scratch.git
cd ML_From_Scratch
pip install -r requirements.txt
```

## Usage

Each algorithm is implemented as a class object. The code can be found in the respective Python file in [utils](utils/). For each algorithm exists a simple example as Python file and a more detailed jupyter notebook. E.g.

### Decision Tree for Regression

A simple example how to use the module is demonstrated in this Python [file](example_RegTree.py):
```bash
python RegressionTree_example.py
```
More detailed documentation and performance tests against various data sets can be found in this [notebook](Regression/RegTree.ipynb). The same structure exists for every algorithm. 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.