# ML From Scratch

Welcome to the ML From Scratch repository! This project aims to implement various machine learning algorithms from scratch using Python. The goal is to understand the inner workings of these algorithms by building them without relying on any machine learning libraries.

Some of the work in this repository was done as part of the practical course **"Mathematics of Machine Learning"** at the **University of Graz**.

## Algorithms Implemented

- Decision Trees for Regression
- k-Nearest Neighbor Classifier

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/HoffOskar/ML_From_Scratch.git
cd ML_From_Scratch
pip install -r requirements.txt
```

## Usage

Each algorithm is implemented in its own Python file in [utils](utils/). 

### Decision Tree for Regression

A simple example how to use the module is demonstrated [here](RegTree_example.py):
```bash
python RegressionTree_example.py
```
More detailed documentation and performance tests against various data sets can be found [here](RegTree.ipynb). 

## k-Nearest Neighbor Classifier

A simple example how to use the module is demonstrated [here](KNN_clf_example.py):
```bash
python KNN_clf_example.py
```
More detailed documentation and decision boundary visualization with various data sets can be found [here](KNN.ipynb). 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.