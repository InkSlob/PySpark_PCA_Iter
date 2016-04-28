
# coding: utf-8

# In[1]:

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
M = (data.map(lambda x: str(x)))
M = M.map(lambda x: x.split(","))
# rdd is a 3993 rows by 11 columns of strings
# [['2.2','2.08','2.2','-0.3','0.9','0.68','1.47',...,'0.14'], ... [...]] 

# dot product of Original Data (M)
Mx0 = M.map(lambda x: (1,float(x[0])+float(x[1])+float(x[2])+float(x[3])+float(x[4])+float(x[5])+float(x[6])+float(x[7])+float(x[8])+float(x[9])+float(x[10])+float(x[11])))


# Frobenius Operation    
x1a = (Mx0
       .map(lambda (x,y):(x,y*y))
       .reduceByKey(add)
       .map(lambda (x,y):(x,math.sqrt(y))))

print x1a.collect()     

# divide elements by frobenius value
x1 = (Mx0
      .rightOuterJoin(x1a)
      .map(lambda (x,(y,z)):(y/z)))

print x1.count()

#x1.saveAsTextFile('PCAout.txt')
#print x1.collect()
# First Iteration **************************************************************** END


# Plotting Results *************************************************************** START
# S&P, DOW, NASDAQ, BHP, CVX, F, GE, JPM, KGC, LMT, MRK, PHM
sp = (M.map(lambda x: float(x[0])).collect())
dow = (M.map(lambda x: float(x[1])).collect())
nasdaq = (M.map(lambda x: float(x[2])).collect())
bhp = (M.map(lambda x: float(x[3])).collect())
cvx = (M.map(lambda x: float(x[4])).collect())
f = (M.map(lambda x: float(x[5])).collect())
ge = (M.map(lambda x: float(x[6])).collect())
jpm = (M.map(lambda x: float(x[7])).collect())
kgc = (M.map(lambda x: float(x[8])).collect())
lmt = (M.map(lambda x: float(x[9])).collect())
mrk = (M.map(lambda x: float(x[10])).collect())
phm = (M.map(lambda x: float(x[11])).collect())
Data = x1.collect()

# S&P, DOW, NASDAQ
# Three subplots sharing both x/y axes12
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
#, ax6, ax7, ax8, ax9, ax10, ax11, ax12
ax1.plot(Data, sp)
ax1.set_title('S&P')
ax2.scatter(Data, dow)
ax2.set_title('DOW')
ax3.scatter(Data, nasdaq, color='g')
ax3.set_title('NASDAQ')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.show()

#BHP, CVX, F, GE, JPM, KGC, LMT, MRK, PHM
fig1 = plt.figure()
a1 = fig1.add_subplot(221)
a1.plot(Data, bhp, '.')
a1.set_title('BHP')
a2 = fig1.add_subplot(222)
a2.plot(Data, cvx, 'c.')
a2.set_title("CVX")
a4 = fig1.add_subplot(223)
a4.plot(Data, ge, '.')
a4.set_title('GE')
a5 = fig1.add_subplot(224)
a5.plot(Data, jpm, 'g.')
a5.set_title('JPM')


fig5 = plt.figure()
a6 = fig5.add_subplot(221)
a6.plot(Data, kgc, '.')
a6.set_title('KGC')
a7 = fig5.add_subplot(222)
a7.plot(Data, lmt, 'm.')
a7.set_title('LMT')
a8 = fig5.add_subplot(223)
a8.plot(Data, mrk, 'y.')
a8.set_title('MRK')
a9 = fig5.add_subplot(224)
a9.plot(Data, phm, 'g.')
a9.set_title('PHM')


# In[ ]:




# In[ ]:



