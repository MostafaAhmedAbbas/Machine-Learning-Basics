#!/usr/bin/env python
# coding: utf-8

# In[112]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[113]:


def Train():
    xDash = []
    jpg =".jpg"
    for i in range(1,2401):
        name= str(i)
        name = name + jpg
        img = mpimg.imread(name)
        np_img = np.array(img).flatten()     #Flattened each Image from 2D Array to 1D array
        np_img = np.append(np_img,1)
        xDash.append(np_img)
        
    xDash=np.asarray(xDash)
    xDashT= xDash.transpose()
    xDot = np.dot(xDashT,xDash)
    xDotInv=np.linalg.pinv(xDot)
    matrixA = np.dot(xDotInv,xDashT)         #Matrix A which will be multiplyed by Different T vectors
    
    
    
    
    
    
    
    T = [-1] * 2400
    for i in range(0,241):
        T[i]=1
        
    WOne= np.dot(matrixA,T)
    
        
   
    T = [-1] * 2400
    for i in range(241,481):
        T[i]=1
    WTwo= np.dot(matrixA,T)
    
    
    T = [-1] * 2400
    for i in range(481,720):
        T[i]=1
    WThree= np.dot(matrixA,T)
    
    
    T = [-1] * 2400
    for i in range(721,960):
        T[i]=1
    WFour= np.dot(matrixA,T)
    
    
    T = [-1] * 2400
    for i in range(961,1200):
        T[i]=1
    WFive= np.dot(matrixA,T)
    
    
    T = [-1] * 2400
    for i in range(1201,1440):
        T[i]=1
    WSix= np.dot(matrixA,T)
    
   
    T = [-1] * 2400
    for i in range(1441,1680):
        T[i]=1
    WSeven= np.dot(matrixA,T)
    
   
    T = [-1] * 2400
    for i in range(1681,1920):
        T[i]=1
    WEight= np.dot(matrixA,T)
    
    
    T = [-1] * 2400
    for i in range(1921,2160):
        T[i]=1
    WNine= np.dot(matrixA,T)
    
    
    T = [-1] * 2400
    for i in range(2161,2400):
        T[i]=1
    WTen= np.dot(matrixA,T)
    
    ret = [WOne,WTwo,WThree,WFour,WFive,WSix,WSeven,WEight,WNine,WTen]
    return ret
    
        
        
    
    
    
    
 #   print(WOne[0])
#     print(TestImages[0])
#     print (TestImages.shape)
    
    
    
        
        
        
        


# In[69]:


Train()


# In[114]:


def Test():
    WArray = Train()
    TestImages=[]
    result=[]
    jpg =".jpg"
    for i in range(1,201):
        name= str(i)
        name = name + jpg
        img = mpimg.imread(name)
        np_img = np.array(img).flatten()     #Flattened each Image from 2D Array to 1D array
        np_img = np.append(np_img,1)
        TestImages.append(np_img)
        
        
    for i in range (0,200):
        res=[0] *10
        for j in range(0,10):
            res[j]=np.dot(TestImages[i],WArray[j])
        Klass = res.index(max(res))
    #    print(Klass)
        result.append(Klass)
        
        
   
    return result      
    


# In[102]:


Test()


# In[115]:


def Confused():
    result = Test()
    matrix =np.zeros((10,10))
    factor =0
    
    for i in range (0,10):
        for j in range (0 +factor,20 +factor):
            matrix[i][result[j]] =   matrix[i][result[j]] + 1
            
        factor +=20
        
        
    print(matrix)
    imgplot = plt.imshow(matrix)
    plt.savefig('Confusion.jpg')
            
    
    
    


# In[116]:


Confused()


# In[ ]:




