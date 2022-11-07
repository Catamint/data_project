import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### From mp.array to Series 
# a=np.array([1,2])
# print(a)
# s=pd.Series(a,index=["i","j"])
# print(s)

# ### from dict to Series
# d=dict({"a":1, "b":2, "c":3})
# # print(d)
# s2=pd.Series(d)
# print(s2)

# ###from dict to DataFrame
#print(pd.DataFrame({"a":[1,2,3], "b":[4,5,6]}))

# ###from Series*n to DataFrame 
# s3=pd.Series(dict({"a":6, "b":5, "c":4}))
# f=pd.DataFrame({"01":s2, "02":s3})
# print(f)
# print(f.index)
# print(f.columns)

b=[]
for i in range(0,100):    
    b.append(np.random.randn())
b=pd.Series(b,index=range(0,100,1))
print(b)
b.plot.box()
print(b.dtype)

# ### return a Series whose value is bool.
# c=b.isnull()
# d=b.notnull()

# b.plot.scatter()
b=pd.DataFrame({"random":b,"range":pd.Series(range(0,100))})
b.insert(2,column="c",value=range(0,100))
# print(b.values)
print(b)
b.plot.bar(stacked="True")
print(b["random"])
b.plot.box()
# b.plot.bar("range","random")
# b.plot.scatter("range","random")
plt.show()