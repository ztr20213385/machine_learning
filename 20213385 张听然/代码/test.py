# 步骤 1：导入必要的库
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import statsmodels.api as sm  # 导入statsmodels

# 步骤 2：指定文件路径
train_file_path = 'Income.data'

# 步骤 3：定义表头
header = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
          "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"]

# 步骤 4：读取数据并预处理
train_data = pd.read_csv(train_file_path, names=header, delimiter=',', skipinitialspace=True)

train_data.replace("?", pd.NA, inplace=True)

missing_values_count = train_data[train_data.isnull().any(axis=1)].shape[0]

if missing_values_count / train_data.size <= 0.1:
    print("不超过10%，因此删除带有缺失值的行")
    train_data.dropna(inplace=True)
else:
    # 用均值填充
    train_data.fillna(train_data.mean(), inplace=True)
    print("超过10%，因此用均值填充")

train_X = train_data.drop("label", axis=1)
train_Y = train_data["label"]

# 步骤 5：数据处理
#print(train_Y)
train_Y = (train_Y == ">50K")
#print(train_Y.value_counts())

categorical_cols = train_X.select_dtypes(include='object').columns
train_X = pd.get_dummies(train_X, columns=categorical_cols)
#print(train_X)

# 转换数据类型为数值型
train_X = train_X.astype(float)

# 步骤 6：使用statsmodels训练模型
train_X = sm.add_constant(train_X)  # 为截距项添加常数
model = sm.Logit(train_Y, train_X)
result_model = model.fit()

# 步骤 7：提取系数
coefficients = result_model.params

for feature, coefficient in zip(train_X.columns, coefficients):
    print(f"{feature}: {coefficient}")

# 步骤 8：获取模型摘要
model_summary = result_model.summary()
print(model_summary)

print("\n\n\n\n\n")

#################以下为在 salary.test 中展示模型效果####################
print("以下为在 salary.test 中展示模型效果")
# 步骤 1：加载测试数据
test_file_path = 'salary.test.csv'
test_data = pd.read_csv(test_file_path, names=header, delimiter=',', skipinitialspace=True)

# 与训练数据相同的数据预处理步骤
test_data.replace("?", pd.NA, inplace=True)

missing_values_count = test_data[test_data.isnull().any(axis=1)].shape[0]

if missing_values_count / test_data.size <= 0.1:
    print("Not exceeding 10%, so delete rows with missing values")
    test_data.dropna(inplace=True)
else:
    # Fill with mean
    test_data.fillna(test_data.mean(), inplace=True)
    print("Exceeding 10%, so fill with mean")

test_X = test_data.drop("label", axis=1)
test_Y = test_data["label"]


# 数据处理
test_Y = (test_Y == ">50K")
categorical_cols_test = test_X.select_dtypes(include='object').columns
test_X = pd.get_dummies(test_X, columns=categorical_cols_test)

# 转换数据类型为数值型
test_X = test_X.astype(float)

# 步骤 2：使用已训练的模型进行预测
test_X = sm.add_constant(test_X)
predictions = result_model.predict(test_X)

# 步骤 3：计算性能指标
threshold = 0.5  # 二分类的阈值
predicted_labels = predictions > threshold

accuracy = accuracy_score(test_Y, predicted_labels)
recall = recall_score(test_Y, predicted_labels)
f1 = f1_score(test_Y, predicted_labels)
roc_auc = roc_auc_score(test_Y, predictions)

# 步骤 4：展示性能指标
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")
