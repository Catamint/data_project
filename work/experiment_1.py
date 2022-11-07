import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

b=pd.DataFrame(np.random.randn(10,5),
              columns=['sepal_lenth',
                     'sepal_width',
                     'petal_lenth',
                     'petal_width',
                     'kind'])
print(b)

# 散点图
b.plot.scatter( "sepal_width",'petal_width')

# 柱状图
b.plot.bar(y="sepal_width")
b.plot.bar()
b.plot.bar(stacked="True")

b.plot.hist(bins=10)

b=pd.read_csv("work\iris.data",
         names=['sepal_lenth',
                'sepal_width',
                'petal_lenth',
                'petal_width',
                'kind'])
print(b)


# 直方图
d=b.drop('kind',axis=1)
print(d)
d.plot.hist(bins=10)

plt.show()
