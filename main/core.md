# 泰坦尼克号幸存者预测

## 概述—问题处理的流程

在竞赛过程中，处理问题一般分为7个阶段：

```
1.确认竞赛题目问题的定义，即读题。

2.获取问题的训练和测试数据。

3.整理、预处理数据。

4.可视化分析数据，探索数据的意义。

5.建立模型，初步预测解决问题。

6.优化解决方案，记录问题解决步骤，生成最终解决方案。

7.提交结果。
```

工作流表示了阶段的顺序以及阶段需要解决的问题，在实际过程中可能不需要分这么细：

```
1.我们可以将多个阶段合并为一个阶段，比如我们可以通过可视化数据来分析数据。

2.我们可以将某个阶段提前或改变，比如我们可以在整理完数据之前和之后分析数据。

3.我们可以将某个步骤使用多次，比如我们可能多次使用可视化步骤。

4.我们也可以不执行某个步骤。
```

## 1.读题—确定问题的定义

像Kaggle这样的竞赛网站定义要解决的问题或要问的问题，同时提供训练数据集来训练模型，并根据测试数据集来测试模型结果。

本题是让我们使用机器学习方法来创建一个模型，用以预测哪些乘客在泰坦尼克号沉船事故中幸存下来。题目定义如下：

```
The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.
```

同时，在介绍题目背景时，我们可以从中找到一些对于处理题目或许有用的信息。背景介绍如下：

```
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
```

从中我们可以了解到：

```
1.在泰坦尼克号沉没之后，2224名乘客和船员中有1502名死亡，可计算出确凿的幸存率为（2224 - 1502）/ 2224 = 32.4%。

2.幸存率比较低的原因是船上的救生艇是有限的。

3.尽管幸存下来会有运气的成分，但明显一部分人会比其他人幸存的机会大，比如妇女儿童或者上层社会阶级的人，这是很直观的。
```

## 2.处理数据—数据分析

我们在数据分析阶段主要解决7个主要问题：

```
1.分类。我们可能需要对样本数据进行分类。我们可能还想了解不同类与解决方案目标之间的关系。

2.相关性。我们可以根据训练数据集中的可用特征来解决这个问题。数据集中的哪些特性对我们的解决方案目标有重大贡献？从统计学上讲，特征和解决方案目标之间是否存在相关性？随着特征值的变化，解决方案的状态是否也发生变化，反之亦然？这可以测试给定数据集中的数值特征和分类特征。我们可能还想确定特征之间的相关性，将某些特征关联起来可能有助于接下来更好地处理数据。

3.特征转换。在建模阶段，需要准备数据， 根据模型算法的选择，可能需要将所有特征都转换为数值等效值。 例如将分类字符串形式值转换为数字值。

4.处理缺失数据。数据准备也可能要求我们估计一个特征中的缺失值。因为当不存在缺失值时，模型算法可能效果最好。

5.修正数据。我们还可以分析给定的训练数据集中的错误或可能不正确的数据，并尝试更正这些值或排除包含错误的样本。一种方法是检测样本或特征中的异常值，如果某个特性对分析没有帮助，或者可能会显著地扭曲结果，我们也可能会完全丢弃它。

6.创建新特征。我们是否可以基于现有特性或一组特性来创建新特性，以便新特性遵循相关性、转换和完整性目标（2，3，4）。

7.制图。如何根据数据的性质以及为了目标解决方案选择正确的可视化绘图和图表。
```

在具体解决问题时，建议：

```
1.在项目前期进行特征相关性分析。
2.使用多个图代替覆盖图以提高可读性。
```

下面开始正式代码实践了，首先加载可能会用到的头文件：

```
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
```

Python中的Pandas包能帮助我们处理数据，我们可以选择将训练和测试数据读入Pandas中的DataFrames来查看数据，我们也可以将数据合并起来一起查看：

```
# acquire data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
combine = [train_df, test_df]
```

接下来，查看数据中的特征：

```
print(train_df.columns.values)
```

![image-20201211200352154](../image/image_1.png)

查看前几行和后几行数据，对数据有个初步认识：

```
# preview the data
train_df.head()
```

![image-20201213183024785](../image/image_2.png)

```
train_df.tail()
```

![image-20201213184111320](../image/image_3.png)

探索特征值的性质,**有哪些特征的值是相对固定的？**也就是特征值是可分类的，拥有几个固定的取值？这些值将样本分为相似样本集。 在分类特征内，值是基于标称值、序数、比率还是区间？ 除其他外，这有助于我们选择合适的图表进行可视化。

```
在本题中，分类特征为Categorical: Survived, Sex, and Embarked. 其中取值是有序数的为Ordinal: Pclass.
```

同样，**有哪些特征的值是数值化的？**这些值随着样本不同而变化。在数值特征中，数值是离散的、连续的还是基于时间序列的？

```
在本题中，连续的数值特征为Continous: Age, Fare. 离散的数值特征为Discrete: SibSp, Parch.
```

**有哪些特征值是混合数据类型？**例特征值既有数字又有字母，或有些为数字有些为字母，这样的特征值可能不是正确的数据，也许需要修正。

```
在本题中，Ticket特征值是混合字母和字母数字类型的，Cabin特征值是字母数字类型的。
```

**有哪些特征值的数据可能是错误的？**如果面对较多数据集，判断出这一点难度很大，但是可以从其中某段较小数据集上的几个示例推测出有哪些特性可能需要更正。

```
在本题中，Name特征值可能存在错误，因为有多种方法用于描述这个特征，包括圆括号和用于替代或短名称的引号。
```

查看DataFrame的大致信息：

```
# pandas.DataFrame.info 打印DataFrame的简要摘要。
# 此方法显示有关DataFrame的信息，包括索引dtype和列，非空值和内存使用情况。
train_df.info()
print('_'*40)
test_df.info()
```

![image-20201213184833327](../image/image_4.png)

**有哪些特征值的数据是缺失的，有空值？**这些数据将需要被修正。

```
在本题中，在训练数据集中Cabin，Age，Embarked这三个特征的数据存在缺失，而且缺失程度为Cabin > Age > Embarked；在测试数据中Cabin，Age，Fare这三个特征的数据存在缺失，但Fare只有一个空值，可以先不考虑，那么缺失程度为Cabin > Age 
```

**特征值的数据类型情况是怎样的？**弄明白这一点对于后面特征值转化有帮助。

```
在本题中，在训练数据集中有七个特征是integer或floats类型的；在测试数据集中有六个特征是integer或floats类型的，有五个特征是strings类型的。
```

查看样本中数字特征值的分布：

```
"""
DataFrame.describe(percentiles=None, include=None, exclude=None, datetime_is_numeric=False) 生成描述性统计信息。
描述性统计包括总结数据集分布的集中趋势、离散度和形状的统计，不包括NaN(空)值。
分析数值和对象类型，以及混合数据类型的DataFrame列集，输出将根据参数而有所不同。
参数列表：
percentiles：
    描述要包含在输出中的百分比。 全部应介于0和1之间，默认值为[.25，.5，.75]，它返回第25、50和75个百分位数。
include:
    参数为结果中要包含的数据类型白名单。默认输出数值类型的统计情况；‘all’输出所有情况；['O']输出所有字符串类型情况，注意这是大写的o。还有其他参数值，要用的时候查看文档。
exclude：
    参数为要从结果中忽略的数据类型黑名单。默认不忽略任何特征值情况。
datetime_is_numeric：
    是否将日期时间数据类型视为数字。默认情况下为False，若为日期参数值为True。

"""
train_df.describe()
```

![image-20201213192711009](../image/image_5.png)

分析数据的分布有助于我们了解训练数据集中一些重要的东西：

```
在本题中，通过设置不同的参数，我们可以得到如下结论：
1.样本总数为891，即泰坦尼克号（2224）上实际乘客人数的40%。
2.Survived是否幸存是用0或1表示的分类特征。
3.训练数据中约有38％的样本存活下来，代表了实际存活率的32％。
4.大多数乘客（> 75％）没有和父母或孩子一起旅行。
5.将近30％的乘客有兄弟姐妹和/或配偶。
6.票价差异很大，只有极少的乘客（<1％）支付的费用高达512美元。
7.65-80岁年龄段的老年乘客很少（<1％）。
```

查看样本中对象object特征值的分布

```
train_df.describe(include=['O'])
```

![image-20201213214707073](../image/image_6.png)

我们可以得到分类特征的分布信息：

```
1.Name在数据集中是唯一的（count = unique = 891）
2.Sex变量有两个可能的值，其中男性占65％（顶部=男性，freq = 577 / count = 891）。
3.Cabin在样本中具有多个重复项，可能几个乘客共用一个客舱。
4.Embarked拥有三个可能的值，大多数乘客使用的S港口（top= S）。
5.Ticket具有很高的重复值比率约22％，（唯一值= 681）。
```

