The updated problem statement for this assignment can be found at https://github.com/LS-Computer-Vision/Python-Basics

# Python-Basics

This assignment is to test the basics of Python and Numpy.

[Here](https://docs.python.org/3/tutorial/index.html) is a Python tutorial to get you started

Make sure you have at least Python 3.7

## Part 0: Setup
Open up your terminal and execute the following commands:

	pip install venv
	python -m venv venv

Next command depends on your OS
### Windows
	venv/Scripts/activate
### Linux / OSX
	source venv/bin/activate

Next step is common

	pip install -r requirements.txt

## Part 1: ```SlowMatrix```
We will attempt to build a matrix class, albeit slow. Let us call it ```SlowMatrix```. Boilerplate for this class is already provided in the file ```SlowMatrix.py```. Implement all the methods provided (the comments explain the functionality you need to implement)

PS: You are not allowed to use anything except standard Python (no matrix libraries like ```numpy```)

## Part 2: ```Numpy```

We will implement [PCA](https://medium.com/analytics-vidhya/understanding-principle-component-analysis-pca-step-by-step-e7a4bb4031d9) (Principal Component Analysis) using ```numpy```

1. Read a D dimensional dataset of N datapoints from a .txt file and project it onto a 2-dimensional space (N, D can be variable). The file will have N rows, with each row containing D comma-
separated values. There will be no whitespaces in this file.

2. Plot the projected data (using a scatter plot) and save the plot to file in the data directory
as ’out.png’ [This saving to file should be automatically done by the code]. While plotting,
ensure both the x and y axis have the same aspect, and show values from [-15,15].

Run this program using

	python pca.py data.txt

