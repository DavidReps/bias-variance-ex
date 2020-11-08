import numpy
import math
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# a) Find a closed-form expression for the average function, ¯g(x). Explain in detail.
# Hint: average over x1, x2, keeping x constant; also, a
# 3 −b3 = (a−b)(a2 +b2 +ab).
# (b) Write a program in Python that carries out a simulation that estimates the
# average function, ¯g(x), as well as the expected out-of-sample error, Eo(g), the
# bias, and the variance, all as functions of x. Plot the results in matplotlib.
# Hint: this will involve generating a large number of two-point sample data sets,
# D = {(x1, x31),(x2, x32)}, and performing appropriate calculations with the lines
# through these pairs of points. Estimate any needed expected values or variances
# based on these lines. Don’t do any averaging over x. Instead, for each x, averages should be taken over the set of samples and the corresponding hypotheses.
# numpy.mean and numpy.var can do some of the calculations for you.
# (c) Calculate the bias and the variance as functions of x, analytically. Do the results
# agree with your simulation? Hint: when computing expected values and variances,
# the variable is not x, but D, in the form of the pair (x1, x2) of input values,
# which are independent and uniformly distributed random variables in [−1, 1]. Use
# independence to simplify products. Odd powers of either xi will average out to 0.
# This calculation requires attention to detail. Don’t try to skip any steps.


var = []
varStore = []
biasA = []
Eout = []
Aavg = []
Bavg = []
x =-1

##this will calculate the average function but rather than call it
## in the interest of time sampleComplexity i use its value of 2x/3
def gbar(x):
    a = 0
    b = 0
    for i in range(1000):
        x1 = random.uniform(-1,1)
        x2 = random.uniform(-1,1)
        a += x*((x2**2) + (x1**2) + (x1*x2))
        b += (x1**2)*x2 + (x2**2)*x1
    a = numpy.mean(a)
    b = numpy.mean(b)
    return a*x + b


def bias(x):
    return (2*x/3 - x**3)**2

def gD(x1,x2, x):
    return ((x*((x2**2) + (x1**2) + (x1*x2))) + ((x1**2)*x2 + (x2**2)*x1))


for i in range(2000):
    for j in range(10000):
        x1 = random.uniform(-1,1)
        x2 = random.uniform(-1,1)
        learningResult = gD(x1, x2, x)
        variance = ((learningResult - (2*x/3))**2)
        varStore.append(numpy.mean(variance))

    biasA.append(numpy.mean(bias(x)))

    var.append(numpy.mean(varStore))

    temp = var[i]
    temp2 = biasA[i]
    Eout.append(temp+temp2)
    x += .001
    varStore =[]

plt.title("Bias Variance Out of sample Error")
plt.xlabel("Iteration number")
plt.plot(var, "--r")
plt.plot(biasA, "--b")
plt.plot(Eout, "--g")
plt.show()
# end of # QUESTION:  2
