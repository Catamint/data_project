'''
draw the feature importance
'''

import pandas
from matplotlib import pyplot

f=pandas.read_csv('result/feat_importance.csv',names=['feature','score'])
f.set_index('feature')
print(f)
f.sort_index(inplace=True)
f=f[:15]
ax=pyplot.axes((0.4,0.05,0.4,0.9))
f.plot.barh(x='feature',ax=ax)
pyplot.show()