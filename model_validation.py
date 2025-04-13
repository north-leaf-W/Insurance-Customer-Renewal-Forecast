import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取训练数据和测试数据
train_file = 'policy_data.xlsx'
test_file = 'policy_test.xlsx'

try:
    print("正在读取数据...")
    train_df = pd.read_excel(train_file)
    test_df = pd.read_excel(test_file)
    
    print(f"训练数据集大小: {train_df.shape}")
    print(f"测试数据集大小: {test_df.shape}")
    
    # 检查测试集是否有目标变量
    has_target = 'renewal' in test_df.columns
    if has_target:
        print("测试数据集包含目标变量 'renewal'，将进行完整评估")
    else:
        print("测试数据集不包含目标变量，只进行预测")
    
    # 对目标变量进行编码
    le = LabelEncoder()
    le.fit(train_df['renewal'])  # 使用训练集拟合编码器
    
    # 编码训练集中的目标变量
    train_df['renewal_encoded'] = le.transform(train_df['renewal'])
    
    if has_target:
        # 编码测试集中的目标变量
        test_df['renewal_encoded'] = le.transform(test_df['renewal'])
    
    print(f"目标变量编码映射: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    #---------------------------
    # 特征处理函数
    #---------------------------
    def preprocess_features(df, is_train=True):
        # 1. 提取特征集
        if is_train:
            X = df.drop(['renewal', 'renewal_encoded', 'policy_id'], axis=1)
        else:
            if has_target:
                X = df.drop(['renewal', 'renewal_encoded', 'policy_id'], axis=1)
            else:
                X = df.drop(['policy_id'], axis=1)
        
        # 2. 处理日期特征 - 计算保单年限
        X['policy_duration'] = (X['policy_end_date'] - X['policy_start_date']).dt.days / 365.25
        X = X.drop(['policy_start_date', 'policy_end_date'], axis=1)
        
        return X
    
    # 预处理训练集和测试集
    X_train = preprocess_features(train_df, is_train=True)
    X_test = preprocess_features(test_df, is_train=False)
    
    # 目标变量
    y_train = train_df['renewal_encoded']
    if has_target:
        y_test = test_df['renewal_encoded']
    
    # 分类变量和数值变量列表
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"分类特征: {categorical_cols}")
    print(f"数值特征: {numerical_cols}")
    
    #---------------------------
    # 逻辑回归模型
    #---------------------------
    print("\n=== 逻辑回归模型验证 ===")
    
    # 创建特征处理管道
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ])
    
    # 创建模型管道
    log_reg_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # 训练模型
    print("正在训练逻辑回归模型...")
    log_reg_model.fit(X_train, y_train)
    
    # 在测试集上预测
    log_reg_pred = log_reg_model.predict(X_test)
    log_reg_pred_proba = log_reg_model.predict_proba(X_test)[:, 1]
    
    # 如果有目标变量，计算评估指标
    if has_target:
        log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
        log_reg_auc = roc_auc_score(y_test, log_reg_pred_proba)
        
        print(f"逻辑回归准确率: {log_reg_accuracy:.4f}")
        print(f"逻辑回归AUC: {log_reg_auc:.4f}")
        print("\n逻辑回归分类报告:")
        print(classification_report(y_test, log_reg_pred, target_names=le.classes_))
        
        # 混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, log_reg_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('逻辑回归模型混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig('validation_log_reg_confusion.png')
        print("逻辑回归混淆矩阵已保存为: validation_log_reg_confusion.png")
        
        # ROC曲线
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, log_reg_pred_proba)
        plt.plot(fpr, tpr, label=f'AUC = {log_reg_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假正例率 (FPR)')
        plt.ylabel('真正例率 (TPR)')
        plt.title('逻辑回归 ROC 曲线')
        plt.legend(loc='lower right')
        plt.savefig('validation_log_reg_roc.png')
        print("逻辑回归ROC曲线已保存为: validation_log_reg_roc.png")
    
    #---------------------------
    # 决策树模型
    #---------------------------
    print("\n=== 决策树模型验证 ===")
    
    # 选择最重要的特征：这里我们复用之前模型选择的特征
    # 基于之前的分析，我们知道年龄、家庭成员数量和某些分类特征是最重要的
    top_categorical = ['gender', 'birth_region']
    top_numerical = ['age', 'premium_amount', 'family_members']
    
    selected_features = top_categorical + top_numerical
    
    # 使用选择的特征创建训练集和测试集
    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()
    
    # 对分类特征进行编码
    X_train_encoded = pd.get_dummies(X_train_selected, columns=top_categorical, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_selected, columns=top_categorical, drop_first=True)
    
    # 确保测试集和训练集有相同的特征列
    missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
    for col in missing_cols:
        X_test_encoded[col] = 0
    
    # 确保列的顺序一致
    X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    # 创建决策树模型（深度为3）
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    
    # 训练模型
    print("正在训练决策树模型...")
    dt_model.fit(X_train_encoded, y_train)
    
    # 在测试集上预测
    dt_pred = dt_model.predict(X_test_encoded)
    dt_pred_proba = dt_model.predict_proba(X_test_encoded)[:, 1]
    
    # 如果有目标变量，计算评估指标
    if has_target:
        dt_accuracy = accuracy_score(y_test, dt_pred)
        dt_auc = roc_auc_score(y_test, dt_pred_proba)
        
        print(f"决策树准确率: {dt_accuracy:.4f}")
        print(f"决策树AUC: {dt_auc:.4f}")
        print("\n决策树分类报告:")
        print(classification_report(y_test, dt_pred, target_names=le.classes_))
        
        # 混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, dt_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('决策树模型混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig('validation_dt_confusion.png')
        print("决策树混淆矩阵已保存为: validation_dt_confusion.png")
        
        # ROC曲线
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, dt_pred_proba)
        plt.plot(fpr, tpr, label=f'AUC = {dt_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假正例率 (FPR)')
        plt.ylabel('真正例率 (TPR)')
        plt.title('决策树 ROC 曲线')
        plt.legend(loc='lower right')
        plt.savefig('validation_dt_roc.png')
        print("决策树ROC曲线已保存为: validation_dt_roc.png")
    
    #---------------------------
    # 模型比较
    #---------------------------
    if has_target:
        print("\n=== 模型比较 ===")
        
        # 创建结果表格
        results = pd.DataFrame({
            '指标': ['准确率', 'AUC'],
            '逻辑回归': [log_reg_accuracy, log_reg_auc],
            '决策树': [dt_accuracy, dt_auc]
        })
        
        print(results)
        
        # 可视化比较
        plt.figure(figsize=(10, 6))
        
        # 准确率比较
        plt.subplot(1, 2, 1)
        model_names = ['逻辑回归', '决策树']
        accuracies = [log_reg_accuracy, dt_accuracy]
        plt.bar(model_names, accuracies, color=['blue', 'green'])
        plt.ylim(0.5, 1)
        plt.title('准确率比较')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # AUC比较
        plt.subplot(1, 2, 2)
        aucs = [log_reg_auc, dt_auc]
        plt.bar(model_names, aucs, color=['blue', 'green'])
        plt.ylim(0.5, 1)
        plt.title('AUC比较')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('validation_model_comparison.png')
        print("模型比较图已保存为: validation_model_comparison.png")
    
    # 生成预测结果文件
    predictions_df = test_df[['policy_id']].copy()
    predictions_df['逻辑回归_预测'] = le.inverse_transform(log_reg_pred)
    predictions_df['逻辑回归_概率'] = log_reg_pred_proba
    predictions_df['决策树_预测'] = le.inverse_transform(dt_pred)
    predictions_df['决策树_概率'] = dt_pred_proba
    
    if has_target:
        predictions_df['真实标签'] = test_df['renewal']
        predictions_df['逻辑回归_正确'] = (predictions_df['逻辑回归_预测'] == predictions_df['真实标签'])
        predictions_df['决策树_正确'] = (predictions_df['决策树_预测'] == predictions_df['真实标签'])
    
    # 保存预测结果
    predictions_df.to_excel('model_predictions.xlsx', index=False)
    print("预测结果已保存为: model_predictions.xlsx")
    
    print("\n模型验证完成！")

except FileNotFoundError as e:
    print(f"错误：找不到文件，{e}")
except Exception as e:
    print(f"发生错误：{e}")
    import traceback
    traceback.print_exc()