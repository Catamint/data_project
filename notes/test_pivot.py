
import pandas as pd
import numpy as np
a=pd.DataFrame({"c1": [1,1,1,4,3],"c2": [2,2,3,5,6]})
c=pd.DataFrame(a.copy())
c.insert(1,'c6',a['c1']-a['c2'])
c['cnt']=1
# d=c[c['c6']==-1]
# # b=pd.pivot_table(c,index=['c1','c6'],values='cnt',aggfunc=len).reset_index()
# # e=pd.pivot_table(b,index='c1',values='c6',aggfunc=len)
# b=c.groupby('c1')
# print(list(b))
# print(a)
d=c.copy()
d['cnt']=0
d['c6']=[1,2,3,4,5]
print(pd.concat([c,d],axis=0))


# print(e)