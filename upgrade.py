# 步骤 1：导入必要的库
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import statsmodels.api as sm  # 导入statsmodels
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression



# 步骤 2：定义函数进行数据处理
def preprocess_data(file_path):
    # 定义表头
    header = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
              "relationship",
              "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"]
    #从CSV文件中读取数据
    data = pd.read_csv(file_path, names=header, delimiter=',', skipinitialspace=True)

    #用NA代替"?"的缺失值
    data.replace("?", pd.NA, inplace=True)
    #计算缺失值的总数
    missing_values_count = data[data.isnull().any(axis=1)].shape[0]

    #对缺失值所在列进行处理
    if missing_values_count / data.size <= 0.1:
        print("不超过10%，因此删除带有缺失值的行")
        data.dropna(inplace=True)
    else:
        data.fillna(data.mean(), inplace=True)
        print("超过10%，因此用均值填充")
    #提取输入特征（从data中除去标签的所在列）
    X = data.drop("label", axis=1)
    #提取模型训练和测试时，预测结果的真实值
    Y = data["label"]
    #将工资按>50k和<=50k分成两类
    Y = (Y == ">50K")

    #提取类别型变量
    categorical_cols = X.select_dtypes(include='object').columns
    #将类别型变量转化成独热码形式
    X = pd.get_dummies(X, columns=categorical_cols)
    #将所有的输入特征转化为float类型
    X = X.astype(float)
    print(X)
    return X, Y


# 步骤 3：定义函数进行模型训练
def train_model(X, Y, regularization='l1', alpha=1.0):
    X = sm.add_constant(X)
    #使用l1范式对训练进行优化
    if regularization.lower() == 'l1':
        print("use-l1")
        # 初始化原始逻辑模型
        model = sm.Logit(Y, X)
        # 训练模型
        result_model = model.fit_regularized(method='l1', alpha=alpha)
    # 使用l2范式对训练进行优化
    elif regularization.lower() == 'l2':
        print("use-l2")
        # 初始化原始逻辑模型
        model = sm.Logit(Y, X)
        # 训练模型
        result_model = model.fit_regularized(method='l2', alpha=alpha)
    else:
        raise ValueError("Invalid regularization type. Use 'l1' or 'l2'.")
    return result_model


# 步骤 4：定义函数提取模型系数
def extract_coefficients(features, coefficients):
    for feature, coefficient in zip(features, coefficients):
        print(f"{feature}: {coefficient}")


# 步骤 5：定义函数获取模型摘要
def get_model_summary(result_model):
    model_summary = result_model.summary()
    print(model_summary)

# 步骤 7：定义函数进行交叉验证
def perform_cross_validation(X, Y, cv=10):
    # 使用StratifiedKFold来保持类别分布
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # 初始化逻辑回归模型
    logistic_model = LogisticRegression(max_iter=2000)

    # 交叉验证
    scores = cross_val_score(logistic_model, X, Y, cv=kfold, scoring='accuracy')

    # 输出交叉验证准确率
    print(f"Cross-Validation Accuracy: {scores.mean()}")

# 步骤 6：定义函数进行模型预测和性能指标计算
def evaluate_model(result_model, test_X, test_Y, threshold=0.5):
    # 对测试数据进行与训练数据相同的处理
    test_X.replace("?", pd.NA, inplace=True)

    missing_values_count = test_X[test_X.isnull().any(axis=1)].shape[0]

    if missing_values_count / test_X.size <= 0.1:
        print("Not exceeding 10%, so delete rows with missing values")
        test_X.dropna(inplace=True)
    else:
        test_X.fillna(test_X.mean(), inplace=True)
        print("Exceeding 10%, so fill with mean")

    # 独热编码
    categorical_cols_test = test_X.select_dtypes(include='object').columns
    test_X = pd.get_dummies(test_X, columns=categorical_cols_test)

    # 确保测试数据与训练数据具有相同的列，缺失的列补0
    missing_cols = list(set(train_X.columns) - set(test_X.columns))
    test_X = pd.concat([test_X, pd.DataFrame(0, index=test_X.index, columns=missing_cols)], axis=1)

    # 确保列的顺序一致
    test_X = test_X[train_X.columns]

    # 转换数据类型为数值型
    test_X = test_X.astype(float)

    # 使用已训练的模型进行预测
    test_X = sm.add_constant(test_X)
    predictions = result_model.predict(test_X)
    print(predictions)


    # 计算性能指标
    predicted_labels = predictions > threshold

    accuracy = accuracy_score(test_Y, predicted_labels)
    recall = recall_score(test_Y, predicted_labels, zero_division=1)
    f1 = f1_score(test_Y, predicted_labels, zero_division=1)

    # ROC AUC Score 处理异常情况
    try:
        roc_auc = roc_auc_score(test_Y, predictions)
        print(f"ROC AUC Score: {roc_auc}")
    except ValueError:
        print("ROC AUC Score is not defined in this case.")

    # 展示性能指标
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


# 步骤 7：应用上述函数进行训练和测试
train_file_path = 'Income.data'
test_file_path = 'answer.csv'

# 训练数据处理
train_X, train_Y = preprocess_data(train_file_path)

# 模型训练
result_model = train_model(train_X, train_Y)

# 提取系数
extract_coefficients(train_X.columns, result_model.params)

# 获取模型摘要
get_model_summary(result_model)

# 测试数据处理
test_X, test_Y = preprocess_data(test_file_path)


# 模型预测和性能指标计算
evaluate_model(result_model, test_X, test_Y)


#进行交叉验证
perform_cross_validation(train_X, train_Y)