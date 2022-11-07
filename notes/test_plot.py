from re import T
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# b=pd.Series(np.random.randn(10))
# print(b)
# b.plot.box()
# b.plot.scatter()
# print(b.dtype)

# ### return a Series whose value is bool.
# c=b.isnull()
# d=b.notnull()

b = pd.DataFrame(np.random.randn(10, 5),
                 columns=["data0", "data1", "data2", "data3", "data4"])
# b.insert(2,column="c",value=range(0,10))
# print(b.values)
b.insert(loc=1, value=b["data0"].map(lambda x: abs(x)),
         column="abs_data0")
print(b)

# b.plot()
b["abs_data0"].plot.pie()
b.plot.bar(stacked="True")
b.plot.box()
b.plot.scatter("data0", "data2")
b.plot.hist(bins=5)
# print(b["data1"])
# b.plot.bar("id","data1")

plt.show()
