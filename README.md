# machine-learning-1-week-8-support-vector-machines-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning 1 Week 8-Support Vector Machines Solved](https://www.ankitcodinghub.com/product/machine-learning-1-week-8-support-vector-machines-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;98760&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning 1 Week 8-Support Vector Machines Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
Exercise 1: Dual formulation of the Soft-Margin SVM

The primal program for the linear soft-margin SVM is

</div>
</div>
<div class="layoutArea">
<div class="column">
subject to

</div>
<div class="column">
1 2 􏰉N min ∥w∥+C ξi w,b,ξ 2 i=1

∀Ni=1 : yi ·(w⊤φ(xi)+b)≥1−ξi and ξi ≥0

</div>
</div>
<div class="layoutArea">
<div class="column">
where ∥.∥ denotes the Euclidean norm, φ is a feature map, w ∈ Rd,b ∈ R are the parameter to optimize, and xi ∈ Rd,yi ∈ {−1,1} are the labeled data points regarded as fixed constants. Once the hard-margin SVM has been learned, prediction for any data point x ∈ Rd is given by the function

f(x) = sign(w⊤φ(x) + b).

<ol>
<li>(a) &nbsp;State the conditions on the data under which a solution to this program can be found from the Lagrange
dual formulation (Hint: verify the Slater’s conditions).
</li>
<li>(b) &nbsp;Derive the Lagrange dual and show that it reduces to a constrained quadratic optimization problem. State
both the objective function and the constraints of this optimization problem.
</li>
<li>(c) &nbsp;Describe how the solution (w, b) of the primal program can be obtained from a solution of the dual program.</li>
<li>(d) &nbsp;Write a kernelized version of the dual program and of the learned decision function. Exercise 2: SVMs and Quadratic Programming (10 P)
We consider the CVXOPT Python software for convex optimization. The method cvxopt.solvers.qp solves quadratic optimization problems given in the matrix form:

min 1x⊤Px+q⊤x x2

subject to Gx ≼ h and Ax = b.

Here, ≼ denotes the element-wise inequality: (h ≼ h′) ⇔ (∀i : hi ≤ h′i). Note that the meaning of the variables x and b is different from that of the same variables in the previous exercise.
</li>
</ol>
(a) Express the matrices and vectors P,q,G,h,A,b in terms of the variables of Exercise 1, such that this quadratic minimization problem corresponds to the kernel dual SVM derived above.

Exercise 3: Programming (50 P)

Download the programming files on ISIS and follow the instructions.

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="section">
<div class="layoutArea">
<div class="column">
Exercise sheet 8 (programming) [WiSe 2021/22] Machine Learning 1

</div>
</div>
<div class="layoutArea">
<div class="column">
Kernel Support Vector Machines

In this exercise sheet, we will implement a kernel SVM. Our implementation will be based on a generic quadratic programming optimizer provided in CVXOPT(python-cvxopt package,ordirectlyfromthewebsite www.cvxopt.org).TheSVMwillthenbetestedontheUCIbreastcancerdataset, asimplebinaryclassificationdatasetaccessibleviathe scikit-learn library.

1. Building the Gaussian Kernel (5 P)

As a starting point, we would like to implement the Gaussian kernel, which we will make use of in our kernel SVM implementation. It is defined as:

k(x,x′) = exp(− ‖x−x′‖2) 2σ2

Implement a function getGaussianKernel that returns for a Gaussian kernel of scale σ, the Gram matrix of the two data sets given as argument.

In [1]:

import numpy,scipy,scipy.spatial

def getGaussianKernel(X1,X2,scale):

### TODO: REPLACE BY YOUR OWN CODE

import solutions

K = solutions.getGaussianKernel(X1,X2,scale) return K

###

2. Building the Matrices for the CVXOPT Quadratic Solver (20 P)

We would like to learn a nonlinear SVM by optimizing its dual. An advantage of the dual SVM compared to the primal SVM is that it allows to use nonlinear kernels such as the Gaussian kernel. The dual SVM consists of solving the following quadratic program:

max

WewouldliketorelyonaCVXOPTsolvertoobtainasolutiontoourSVMdual.Thefunction cvxopt.solvers.qp solvesanoptimizationproblemof the type:

\begin{align*} \min_{\boldsymbol{x}} \quad &amp;\frac12 \boldsymbol{x}^\top P \boldsymbol{x} + \boldsymbol{q}^\top \boldsymbol{x}\\ \text{subject to} \quad &amp; G \boldsymbol{x} \preceq \boldsymbol{h}\\ \text{and} \quad &amp; A \boldsymbol{x} = \boldsymbol{b}. \end{align*}

which is of similar form to our dual SVM (note that \boldsymbol{x} will correspond to the parameters (\alpha_i)_i of the SVM). We need to build the data structures (vectors and matrices) that makes solving this quadratic problem equivalent to solving our dual SVM.

Implement a function getQPMatrices that builds the matrices P , q , G , h , A , b (of type cvxopt.matrix ) that need to be passed as argument to the optimizer cvxopt.solvers.qp .

In [2]:

import cvxopt,cvxopt.solvers

cvxopt.solvers.options[‘show_progress’] = False

def getQPMatrices(K,T,C):

### TODO: REPLACE BY YOUR CODE

import solutions

P,q,G,h,A,b = solutions.getQPMatrices(K,T,C) return P,q,G,h,A,b

###

</div>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="section">
<div class="layoutArea">
<div class="column">
3. Computing the Bias Parameters (10 P)

Given the parameters (\alpha_i)_i the optimization procedure has found, the prediction of the SVM is given by: f(x) = \text{sign}\Big(\sum_{i=1}^N \alpha_i y_i k(x,x_i) + \theta\Big)

Note that the parameter \theta has not been computed yet. It can be obtained from any support vector that lies exactly on the margin, or equivalently, whose associated parameter \alpha is not equal to 0 or C. Calling one such vector “x_M”, the parameter \theta can be computed as:

\theta = y_M – \sum_{j=1}^N \alpha_j y_j k(x_M,x_j)

Implement a function getTheta that takes as input the Gram Matrix used for training, the label vector, the solution of our quadratic program, and the hyperparameter C. The function should return the parameter \theta.

In [3]:

def getTheta(K,T,alpha,C):

### TODO: REPLACE BY YOUR CODE

import solutions

theta = solutions.getTheta(K,T,alpha,C) return theta

###

4. Implementing a class GaussianSVM (15 P)

All functions that are needed to learn the SVM have now been built. We would like to implement a SVM class that connects them and make the SVM

easily usable. The class structure is given below and contains two functions, one for training the model, and one for applying it to test data.

Implement the function fit that makes use of the functions getGaussianKernel , getQPMatrices , getTheta you have already implemented. The function should learn the SVM model and store the support vectors, their label, (\alpha_i)_i and \theta into the object ( self ).

Implement the function predict that makes use of the stored information to compute the SVM output for any new collection of data points

In [4]:

class GaussianSVM:

def __init__(self,C=1.0,scale=1.0): self.C, self.scale = C, scale

def fit(self,X,T):

<pre>        ### TODO: REPLACE BY YOUR CODE
</pre>
<pre>        import solutions
</pre>
<pre>        solutions.fit(self,X,T)
</pre>
###

def predict(self,X):

<pre>        ### TODO: REPLACE BY YOUR CODE
</pre>
<pre>        import solutions
</pre>
Y = solutions.predict(self,X) return Y

###

5. Analysis

The following code tests the SVM on some breast cancer binary classification dataset for a range of scale and soft-margin parameters. For each combination of parameters, we output the number of support vectors as well as the train and test accuracy averaged over a number of random train/test splits. Running the code below should take approximately 1-2 minutes.

</div>
</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="section">
<div class="layoutArea">
<div class="column">
In [5]:

import numpy,sklearn,sklearn.datasets,numpy

<pre>D = sklearn.datasets.load_breast_cancer()
X = D['data']
T = D['target']
T = (D['target']==1)*2.0-1.0
</pre>
for scale in [30,100,300,1000,3000]: for C in [10,100,1000,10000]:

acctrain,acctest,nbsvs = [],[],[] svm = GaussianSVM(C=C,scale=scale) for i in range(10):

<pre>            # Split the data
</pre>
<pre>            R = numpy.random.mtrand.RandomState(i).permutation(len(X))
            Xtrain,Xtest = X[R[:len(R)//2]]*1,X[R[len(R)//2:]]*1
            Ttrain,Ttest = T[R[:len(R)//2]]*1,T[R[len(R)//2:]]*1
</pre>
<pre>            # Train and test the SVM
</pre>
<pre>            svm.fit(Xtrain,Ttrain)
            acctrain += [(svm.predict(Xtrain)==Ttrain).mean()]
            acctest  += [(svm.predict(Xtest)==Ttest).mean()]
            nbsvs += [len(svm.X)*1.0]
</pre>
print(‘scale=%9.1f C=%9.1f nSV: %4d train: %.3f test: %.3f’%( scale,C,numpy.mean(nbsvs),numpy.mean(acctrain),numpy.mean(acctest)))

print(”)

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>scale=
scale=
scale=
scale=
</pre>
</div>
<div class="column">
<pre>30.0  C=     10.0  nSV:  183  train: 0.997  test: 0.921
30.0  C=    100.0  nSV:  178  train: 1.000  test: 0.918
30.0  C=   1000.0  nSV:  184  train: 1.000  test: 0.918
30.0  C=  10000.0  nSV:  182  train: 1.000  test: 0.918
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>100.0  C=     10.0  nSV:  117  train: 0.965  test: 0.935
100.0  C=    100.0  nSV:   97  train: 0.987  test: 0.940
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>scale=
scale=
scale=    100.0  C=   1000.0  nSV:   85  train: 0.998  test: 0.932
scale=    100.0  C=  10000.0  nSV:   71  train: 1.000  test: 0.926
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>scale=    300.0  C=     10.0  nSV:   88  train: 0.939  test: 0.924
scale=    300.0  C=    100.0  nSV:   48  train: 0.963  test: 0.943
scale=    300.0  C=   1000.0  nSV:   36  train: 0.978  test: 0.946
scale=    300.0  C=  10000.0  nSV:   32  train: 0.991  test: 0.941
</pre>
<pre>scale=   1000.0  C=     10.0  nSV:   66  train: 0.926  test: 0.916
scale=   1000.0  C=    100.0  nSV:   55  train: 0.935  test: 0.929
scale=   1000.0  C=   1000.0  nSV:   49  train: 0.956  test: 0.946
scale=   1000.0  C=  10000.0  nSV:   38  train: 0.971  test: 0.951
</pre>
<pre>scale=   3000.0  C=     10.0  nSV:   87  train: 0.912  test: 0.903
scale=   3000.0  C=    100.0  nSV:   68  train: 0.926  test: 0.919
scale=   3000.0  C=   1000.0  nSV:   58  train: 0.934  test: 0.929
scale=   3000.0  C=  10000.0  nSV:   49  train: 0.953  test: 0.943
</pre>
We observe that the highest accuracy is obtained with a scale parameter that is neither too small nor too large. Best parameters are also often associated to a low number of support vectors.

</div>
</div>
</div>
</div>
<div class="page" title="Page 5"></div>
<div class="page" title="Page 6"></div>
<div class="page" title="Page 7"></div>
<div class="page" title="Page 8"></div>
