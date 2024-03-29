# Work

任务1到任务5的所有代码.

## catalogue 

- experiment_1 ················ Pandas入门
- experiment_2 ················ K-Means
- experiment_3 ················ 数据观察
- experiment_4 ················ 数据预处理
- experiment_5 ················ (任务2) 比赛代码

## 提取的特征

### 标签区间的用户特征

- 用户当月领券数
- 用户当月领取相同券数
- 用户当天领券数
- 平均每天领券数
- 用户是否当天领取同一券
- 当天领取相同券数
- 是否第一次/最后一次领券

排序特征:

- 折扣率排序
- 距离排序
- 领券日期排序

### 标签区间的商家特征

- 发放优惠券数
- 发放优惠券种类数
- 当天发券数
- 当天发券的优惠券种数

排序特征:

- 折扣率排序
- 距离排序
- 领券日期排序

### 标签区间的优惠券特征

排序特征:

- 折扣率排序
- 距离排序
- 领券日期排序

### 历史区间(需要标签数据)的用户特征

- 用户领券数
- 用户领券"并消费"数
- 用户领券"未消费"数
- 用户领券"并消费"数 / 领券数
- 用户领取"并消费"优惠券的平均折扣率
- 用户领取"并消费"优惠券的平均距离
- 用户在多少不同商家领取"并消费"优惠券
- 用户在多少不同商家领取优惠券
- 用户在多少不同商家领取"并消费"优惠券 / 用户在多少不同商家领取优惠券
- 用户用券购买平均距离
- 用户用券购买最小距离 最大距离
- 用户领券到消费平均时间间隔

### 历史区间(需要标签数据)的优惠券特征

- 当前种类优惠券被领取数
- 当前种类优惠券被领取"并消费"数
- 当前种类优惠券被领取"未消费"数

### 历史区间的商家特征

- 发放优惠券数
- 15天内被领券购买量
- 店家总销量

### 历史区间商家和用户特征

- 用户在店家领券并消费数
- 用户在店家消费数
- 用户在店家领券并消费率
- 用户在此商家领券购买数/领券购买总数
- 用户在此商家购买数/购买总数
- 用户与店家平均距离
- 一个客户在一个商家一共收到的优惠券
