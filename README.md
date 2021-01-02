### 1.题目选择：

阿里云天池长期赛--测测你的一见钟情指数

原链接https://tianchi.aliyun.com/competition/entrance/531825/introduction

### 2.开发环境：

- python==3.7.1
- numpy==1.19.1
- pandas==1.1.0
- matplotlib==3.3.0
- seaborn==0.11.0
- scikit_learn==0.23.2

### 3.运行方法：

#### 3.1 逻辑回归：

代码在src/LogisticRegression文件夹下，在这里可以运行代码。

- 安装依赖:

  ```python
  pip install -r requirements.txt
  ```

- 运行特征选择（热力图）:

  ```python
  python data_process.py
  ```

- 运行grid_search：

  ```python
  python test.py
  ```

#### 3.2 决策树：

在命令行中转到 `src/DecisionTree`目录下，执行 python main.py。会输出C4.5决策树的交叉验证准确率和测试集准确率。需要安装依赖，同时在电脑中配置好 graphviz 并设置环境变量。

将 `main.py` 第49到51行的注释去掉即会开始暴力搜索最优的超参数。该过程耗时较长。

```python
run_object = search_best_params(params)
# run_object.exec()
#
# print(run_object.max_vc, run_object.best_params)
```

#### 3.3 随机森林：

在 `src/RandomForest`目录下运行代码

首先安装依赖：

```shell
pip install -r requirements.txt
```

执行模型训练：

```shell
python RF_classifier.py
```

#### 3.4 Adaboost：

代码在src/AdaBoost文件夹下，需要到这里运行。

首先安装依赖：

```shell
pip install -r requirements.txt
```

运行主函数（特征为3.1提到的方法得到的特征组合1）：

```shell
python main.py 
```

运行grid search

```shell
python grid_search.py
```

复现最优结果

```shell
python grid_search.py --best=1
```