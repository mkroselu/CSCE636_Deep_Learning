﻿# Instructions for programming assignments of CSCE636: Deep Learning


You will use the Python programming language and PyTorch for this assignment.


Installation of Python
-----------------------
For all assignments in this course, we will use a few popular libraries (numpy, matplotlib, math) for scientific computing. 
We expect that many of you already have some basic experience with Python and NumPy. We also provide basic guide to learn Python and NumPy.

Download and install Anaconda with Python3.6 version:
- Download at the website: https://www.anaconda.com/download/
- Install Python3.6 version(not Python 2.7)
Anaconda will include all the Python libraries we need. 


Python & NumPy Tutorial
-----------------------
- Official Python tutorial: https://docs.python.org/3/tutorial/
- Official NumPy tutorial: https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
- Good tutorial sources: http://cs231n.github.io/python-numpy-tutorial/ 


Installation of tqdm
---------------------
Please install python packages tqdm using:
"pip install tqdm" or
"conda install tqdm"


Installation of PyTorch
--------------------------
The recommended PyTorch version is 1.6.0, and you can choose to install either the cpu only or gpu version.

Check https://pytorch.org/get-started/previous-versions/ for other ways to install PyTorch or more details.

Dataset Descriptions
--------------------
We will use USPS dataset for this assignment. The USPS dataset is in the “data” folder: USPS.mat. 
The whole data has already been loaded into the matrix A. The matrix A contains all the images of size 16 × 16. 
Each of the 3000 rows in A corresponds to the image of one handwritten digit (between 0 and 9). 


Assignment Descriptions
-----------------------
There are total four Python files including 'main.py', 'solution.py', 'helper.py' and 'test_tf.install.py'. 
In this assignment, you need to implement your solution in 'solution.py' and 'main.py' files following the given instruction. 
However, you might need to read all the files to fully understand the requirement. 

The 'helper.py' includes all the helper functions for the assignments, like load data, show images, etc.

Only implement your code in 'solution.py' and 'main.py' files.
Only write your code between the following lines. Do not modify other parts. Only the code
between the following lines will be graded. Don NOT include experimental results in your
code submission. Do NOT change file names.

\### YOUR CODE HERE

\### END YOUR CODE


Useful Numpy functions
----------------------
In this assignment, you may use following numpy functions:
- np.linalg.svd(): compute the singular value decomposition on a given matrix. 
- np.dot(): matrices multiplication operation.
- np.mean(): compute the mean value on a given matrix.
- np.zeros(): generate a all '0' matrix with a certain shape.
- np.expand_dims: expand the dimension of an array at the referred axis.
- np.squeeze: Remove single-dimensional entries from the shape of an array. 
- np.transpose(): matrix transpose operation.
- np.linalg.norm(): compute the norm value of a matrix. You may use it for the reconstruct_error function.


PyTorch functions and APIs you may need
------------------------------------------
torch.tensor
torch.empty
torch.mm
torch.transpose
nn.Parameter
nn.init.kaiming_normal_
nn.Tanh
nn.ReLU
nn.Sigmoid

Refer to https://pytorch.org/docs/1.6.0/ for documents of PyTorch 1.6.0.






