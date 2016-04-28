
# coding: utf-8

# In[7]:

import os
import math
import numpy as np
from operator import add
from fileinput import input
from glob import glob
import matplotlib as mpl
import matplotlib.pyplot as plt

# Load standardized data from csv
inputFile = os.path.join("Z_data.csv")
data = sc.textFile(inputFile)

#print "Data from CSV: ", data.count()

# Order of columns
# S&P, DOW, NASDAQ, BHP, CVX, F, GE, JPM, KGC, LMT, MRK, PHM
# 12 columns 0-11
# 3993 rows of data

# First Iteration **************************************************************** 
# convert to string, then split by comma
M = (data
     .map(lambda x: str(x))
     .map(lambda x: x.split(","))
    )
print "Data as csv imported"
# rdd is a 3993 rows by 12 columns of strings
# [['2.2','2.08','2.2','-0.3','0.9','0.68','1.47',...,'0.14'], ... [...]] 

# dot product of Original Data (M) with initial vector of 1s
Mx0 = M.map(lambda x: (1,float(x[0])+float(x[1])+float(x[2])+float(x[3])+float(x[4])+
                       float(x[5])+float(x[6])+float(x[7])+float(x[8])+float(x[9])+
                       float(x[10])+float(x[11]))
           )

# Frobenius Operation    
x1a = (Mx0
       .map(lambda (x,y):(x,y*y))
       .reduceByKey(add)
       .map(lambda (x,y):(x,math.sqrt(y)))
      )

print "Frobenius Calc: ", x1a.collect()     

# divide elements by frobenius value
x1 = (Mx0
      .rightOuterJoin(x1a)
      .map(lambda (x,(y,z)):(y/z))
      )
print "Frobenius row check: ", x1.count()

# Vector for next iteration - take ordered top 12
v1 = (x1
      .map(lambda x:(x,1))
      .sortByKey(False)
      .map(lambda (x,y):x)
      .take(12)
     )
print "iteration:  1"
# saves first top eg for later comparison
started = v1[0]
np.asarray(v1)
# End First Iteration **************************************************************************
# BEGIN LOOP ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# M x Top 12 EigenV (Dot Product)
# to set the number of iterations change value of iters
iters = 99

for i in range(0, iters):
    print "iteration: ", i+2
    M2 = (M
          .map(lambda x: (1,float(x[0])*v1[0]+float(x[1])*v1[1]+float(x[2])*v1[2]+float(x[3])*v1[3]+
              float(x[4])*v1[4]+float(x[5])*v1[5]+float(x[6])*v1[6]+float(x[7])*v1[7]+float(x[8])*v1[8]+
              float(x[9])*v1[9]+float(x[10])*v1[10]+float(x[11])*v1[11]))
     )  
    # Results in a 1 x 3993 matrix
  
    # Frobenius Operation    
    x1a_loop = (M2
                .map(lambda (x,y):(x,y*y))
                .reduceByKey(add)
                .map(lambda (x,y):(x,math.sqrt(y)))
               )

    # divide elements by frobenius value
    xn_loop = (M2
               .rightOuterJoin(x1a_loop)
               .map(lambda (x,(y,z)):(y/z))
               )

    # Find top eigens
    v1 = (xn_loop
               .map(lambda x:(x,1))
               .sortByKey(False)
               .map(lambda (x,y):x)
               .take(12)
          )
    np.asarray(v1)
    #print i, " iter: ", v1
ended = v1[0]
print "Conducted ", iters, " iterations."
print "Difference betweeen first and last iteration: ", started - ended
print "Sample of output:"
print xn_loop.take(50)
#xn_loop.saveAsTextFile('PCA_iter_out.txt')


# In[ ]:



