# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# TODO: Write this.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

'''
Created on Nov 22, 2015
First Name: Sampath Sree Kumar
Last Name: Kolluru
StudentID: 800887568
'''


import sys
import numpy as np

from pyspark import SparkContext

#To print output with precision
np.set_printoptions(precision=13)

def keyA(l):
    "This function is to find X * (X_transpose)"
    l[0]=1.0
    temp_x = np.array(l).astype('float')
    X = np.asmatrix(temp_x).T
    #print X
    X_Xt = np.dot(X,X.T)
    return X_Xt

def keyB(l):
    "This function is to find X * y"
    y = float(l[0])
    l[0] = 1.0
    temp_x = np.array(l).astype('float')
    X = np.asmatrix(temp_x).T
    #print "Y: ",y
    #print "X: ",X
    X_y = np.multiply(X,y)
    return X_y

if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])
  yxlines = yxinputFile.map(lambda line: line.split(','))
  
  #Now we need to calculate A. First Calculate (X * (X_Transpose)) and add them using reduceBYKey function  
  A = np.asmatrix(yxlines.map(lambda l: ("KeyA",keyA(l))).reduceByKey(lambda x1,x2: np.add(x1,x2)).map(lambda l: l[1]).collect()[0])
  #print A
  
  #Now we need to calculate B. First Calculate (X * y) and add them using reduceBYKey function
  B = np.asmatrix(yxlines.map(lambda l: ("KeyB",keyB(l))).reduceByKey(lambda x1,x2: np.add(x1,x2)).map(lambda l: l[1]).collect()[0])
  #print B
  
  #Shape give the dimension of matrix
  #print A.shape
  #print B.shape
  
  #Now multiply A_inverese with B to get the coefficients
  beta1 = np.dot(np.linalg.inv(A),B)
  #Convert the matrix to list for displaying
  beta = np.array(beta1).tolist()
  print beta
  

  # print the linear regression coefficients in desired output format
  print "beta: "
  for coeff in beta:
      print coeff[0]

  sc.stop()
