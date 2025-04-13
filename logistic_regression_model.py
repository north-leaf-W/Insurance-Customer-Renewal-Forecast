import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取Excel文件
file_path = 'policy_data.xlsx'
try:
    # 读取Excel文件
    print("正在读取数据...")
    df = pd.read_excel(file_path)
    
    # 数据基本信息
    print(f"数据维度: {df.shape}")
    
    # 对目标变量进行编码（如果是文本形式）
    le = LabelEncoder()
    if df['renewal'].dtype == 'object':
        df['renewal_encoded'] = le.fit_transform(df['renewal'])
        print(f"目标变量编码映射: {dict(zip(le.classes_, range(len(le.classes_))))}")
    else:
        df['renewal_encoded'] = df['renewal']
    
    # 特征处理
    # 1. 删除不需要的列
    X = df.drop(['renewal', 'renewal_encoded', 'policy_id'], axis=1)  # policy_id 不是有用特征
    y = df['renewal_encoded']
    
    # 2. 处理日期特征 - 计算保单年限
    X['policy_duration'] = (X['policy_end_date'] - X['policy_start_date']).dt.days / 365.25
    X = X.drop(['policy_start_date', 'policy_end_date'], axis=1)
    
    # 分类变量列表
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"分类特征: {categorical_cols}")
    print(f"数值特征: {numerical_cols}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建特征处理管道
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ])
    
    # 创建模型管道
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # 训练模型
    print("正在训练模型...")
    model.fit(X_train, y_train)
    
    # 获取预处理后的特征名称
    # 获取分类变量的one-hot编码后的特征名称
    cat_features = []
    for i, col in enumerate(categorical_cols):
        # 获取分类变量的唯一值（除了第一个因为使用了drop='first'）
        unique_values = df[col].unique()[1:]
        for val in unique_values:
            cat_features.append(f"{col}_{val}")
    
    # 数值特征名称保持不变
    feature_names = cat_features + numerical_cols
    
    # 获取模型系数
    coefficients = model.named_steps['classifier'].coef_[0]
    
    # 创建一个DataFrame来存储特征名称和对应的系数
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    
    # 按系数绝对值排序
    coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
    
    # 评估模型
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n==== 模型评估 ====")
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 输出混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    class_names = le.classes_
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵已保存为: confusion_matrix.png")
    
    # 可视化 ROC 曲线
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('ROC 曲线')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    print("ROC 曲线已保存为: roc_curve.png")
    
    # 可视化逻辑回归系数
    # 按照系数绝对值排序，选择最重要的特征
    top_n = 20  # 显示前20个特征
    top_coef = coef_df.head(top_n)
    
    plt.figure(figsize=(12, 10))
    
    # 创建条形图，根据系数正负设置颜色
    colors = ['red' if c < 0 else 'green' for c in top_coef['Coefficient']]
    
    ax = sns.barplot(
        x='Coefficient', 
        y='Feature', 
        data=top_coef,
        palette=colors
    )
    
    # 添加垂直线表示零点
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 添加标题和标签
    plt.title('逻辑回归模型的前20个重要特征系数')
    plt.xlabel('系数值')
    plt.ylabel('特征')
    
    # 添加注释
    plt.text(0.95, 0.05, '绿色 = 正相关 (增加续保概率)\n红色 = 负相关 (降低续保概率)', 
             transform=plt.gca().transAxes, ha='right', va='bottom', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('logistic_regression_coefficients.png')
    print("逻辑回归系数可视化已保存为: logistic_regression_coefficients.png")
    
    # 打印出重要特征系数
    print("\n==== 重要特征系数 ====")
    print(top_coef)
    
    print("\n模型训练和评估完成！")

except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'")
except Exception as e:
    print(f"发生错误：{e}") 