# 训练模型步骤总览

1. 数据总览
2. 分析数据关系
3. 变量转换
	- 定性转换
		- DummyVariables
		- Factorizing
	- 定量转换：
		- Scaling
		- Binning：这之后接Dummy或者Factorizing
4. 特征工程
	- 处理缺失值：
		- 缺失较少的特征：可以利用众数/平均数等填充
		- 缺失较少的特征：可以利用机器学习进行预测填充
		- 大量缺失的特征：直接丢弃
	- 丢弃没用的特征
	- 利用现有特征创建/提取新特征：如名字长度（虽然不一定真的和存活结果有关系），两个特征求和如亲友数量Parch和Sibsp求和作为family_size新特征
	- 数据的正则化
	- 训练数据和测试数据分开
5. 模型融合及测试
	- 利用不同模型进行特征筛选，选出较为重要的特征
	- 依据筛选的特征构建训练集和测试集
	- 模型融合（Model Ensemble）：Bagging、Boosting、Stacking、Blending
		- Bagging：Bagging 将多个模型，也就是多个基学习器的预测结果进行简单的加权平均或者投票。它的好处是可以并行地训练基学习器。Random Forest就用到了Bagging的思想。
		- Boosting：Boosting 的思想有点像知错能改，每个基学习器是在上一个基学习器学习的基础上，对上一个基学习器的错误进行弥补。我们将会用到的 AdaBoost，Gradient Boost 就用到了这种思想。
		- Stacking：Stacking是用新的次学习器去学习如何组合上一层的基学习器。如果把 Bagging 看作是多个基分类器的线性组合，那么Stacking就是多个基分类器的非线性组合。Stacking可以将学习器一层一层地堆砌起来，形成一个网状的结构。相比来说Stacking的融合框架相对前面的二者来说在精度上确实有一定的提升，所以在下面的模型融合上，我们也使用Stacking方法。
		- Blending：Blending 和 Stacking 很相似，但同时它可以防止信息泄露的问题


# 数据部分

```python
import pandas as pd
```

### 简单操作

- 导入csv数据
```python
# DataFrame类型
train_data = pd.read_csv('data/train.csv')
```
- 查看csv数据内容
```python
# 默认查看前5行
train_data.head()
# 给出列表头
train_data.columns.values
# 数据信息总览
train_data.info()
# 查看csv的某列属性，如总数、均值、标准差、最小值等
train_data['Age'].describe()
```

- 利用众数填补缺失值：
```python
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
```
- 利用给定值填补缺失值：
```python
train_data['Cabin'] = train_data.Cabin.fillna('U0')
```
- 丢弃列特征
```python
# 设置axis=1以代表列
train_data.drop(['Embarked'], axis=1,inplace=True)
```


- 对csv文件中不同列进行分组：Sex和Sruvived分组，并对Survived进行count计数
```python
# ['这里换成Sex得到的结果相同'].count()
train_data.groupby(['Sex','Survived'])['Survived'].count()
```

- 将某列特征进行划分：
```python
bins = [0, 12, 18, 65, 100]
# 这里会自动新创建一个”Age_group“列特征
train_data['Age_group'] = pd.cut(train_data['Age'], bins)
```

- 涉及到**正则表达式**
```python
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_data['Title'], train_data['Sex'])
```

### 定性转换(Qualitative)
1. **Dummy Variables**
	就是**类别变量或者二元变量**，当qualitative variable是一些频繁出现的几个独立变量时，Dummy Variables比较适合使用。
	我们以Embarked为例，Embarked只包含三个值'S','C','Q'，我们可以使用下面的代码将其转换为dummies
```python
# 主要是这一行，get_dummies()，自动转成独热编码
embark_dummies  = pd.get_dummies(train_data['Embarked'])
train_data = train_data.join(embark_dummies)

train_data.drop(['Embarked'], axis=1,inplace=True)
embark_dummies = train_data[['S', 'C', 'Q']]
```
**自定义dummy**：
```python
map_sex = {'female':1, 'male':0}
train_df['Sex'] = train_df['Sex'].map(map_sex).astype(int)
```

2. **Factorizing**
	dummy不好处理Cabin（船舱号）这种标称属性，因为他**出现的变量比较多**。所以Pandas有一个方法叫做factorize()，它可以创建一些数字，来表示类别变量，对每一个类别映射一个ID，这种映射最后只生成一个特征，不像dummy那样生成多个特征

### 定量转换(Quantitative)
1. **Scaling**
	Scaling可以将一个很大范围的数值映射到一个很小的范围(通常是-1 - 1，或则是0 - 1)，很多情况下我们需要将数值做Scaling使其范围大小一样，否则大范围数值特征将会由更高的权重。比如：Age的范围可能只是0-100，而income的范围可能是0-10000000，在某些对数组大小敏感的模型中会影响其结果
```python
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'].values.reshape(-1, 1))
```
2. **Binning**
	Binning通过**观察“邻居”(即周围的值)将连续数据离散化**。存储的值被分布到一些“桶”或“箱“”中，就像直方图的bin将数据划分成几块一样。下面的代码对Fare进行Binning。
	在将数据Bining化后，要么将数据factorize化，要么dummies化。
```python
# Divide all fares into quartiles
train_data['Fare_bin'] = pd.qcut(train_data['Fare'], 5)
train_data['Fare_bin'].head()
```
# 绘图部分

```python
import seaborn as sns
import pandas as pd
```

## Pandas
- 对csv中某列内容绘图![[Pasted image 20231031213944.png#center|300]]
```python
# 去掉.plot.pie()部分输出的就是单纯的以Survived进行分组的计数列表
train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')
```
- 对csv中两列进行柱状图绘制：Sex和Survived两列，按照Sex类别分组![[Pasted image 20231031214441.png#center|300]]
```python
# 去掉.plot.bar()部分输出的就是单纯的以Sex进行分组的Survived均值列表
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
```
- 上方这种柱状图还可以绘制双变量如Sex和Pclass与Survived的柱状图关系：![[Pasted image 20231031215146.png#center|300]]
```python
train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()
```
## Seaborn
- **小提琴图**![[Pasted image 20231031215328.png#center]]
```python
fig, ax = plt.subplots(1, 2, figsize = (18, 8))

sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train_data, split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))
  
sns.violinplot(x="Sex", y="Age", hue="Survived", data=train_data, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))

plt.show()
```

- ![[Pasted image 20231031215500.png#center]]
```python
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
```

- ![[Pasted image 20231031215616.png#center]]
```python
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
train_data["Age_int"] = train_data["Age"].astype(int)
average_age = train_data[["Age_int", "Survived"]].groupby(['Age_int'],as_index=False).mean()
sns.barplot(x='Age_int', y='Survived', data=average_age, palette="husl")
```
# 模型部分