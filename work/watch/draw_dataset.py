
import pandas
from matplotlib import pyplot

f=pandas.read_csv('result/result.csv',names=['User_id','Coupon_id','Date_received','Probability'])
print(f)
# ax=pyplot.axes((0.1,0.05,0.8,0.9))
f['Probability'].plot.hist()
pyplot.show()
