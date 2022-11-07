from this import d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
off_train = pd.read_csv("#test\data\ccf_offline_stage1_train.csv")

print(off_train["Discount_rate"])

# 数据预处理

## 时间格式转换
###


## 满减格式规范
### 满减转换成折扣率 <注意不能传空值>
def rate_to_count(rate):
    rate = str(rate)
    a, b = rate.split(sep=":")
    return (int(a)-int(b))/int(a)

### have_discount = 是否有优惠券
off_train.insert(loc=4, column="have_discount",
              value=off_train["Discount_rate"].notnull())
print(off_train["have_discount"])

### discount = 折扣率
off_train.insert(loc=5, column="discount",
              value=off_train["Discount_rate"].map(
                  lambda x: float(rate_to_count(x)) \
                    if ":" in str(x) else float(x))
              )
print(off_train["discount"])

### 满减门槛价格
off_train.insert(loc=6,column="price_before",
              value=off_train["Discount_rate"].map(
                lambda x: float(str(x).split(sep=":")[0]) \
                    if ":" in str(x) else float(x))
             )
print(off_train["price_before"])

print(off_train.count(axis=0))
print(off_train["Discount_rate"].value_counts())
print(off_train["discount"].min())


## 异常值剔除
###


## 缺失值处理
###


## 统一数据格式
###


# 数据划分


# 数据打标