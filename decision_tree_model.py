import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
    
    # 特征工程 - 为了让决策树更易于解释，我们将使用较少的特征
    # 我们先获取数值特征的重要性
    # 我们先拆分数据以便训练一个初步的决策树来确定重要特征
    X_train_init, X_test_init, y_train_init, y_test_init = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 使用处理过的数据构建简单的训练集
    X_train_simple = X_train_init.select_dtypes(include=['int64', 'float64']).copy()
    X_test_simple = X_test_init.select_dtypes(include=['int64', 'float64']).copy()
    
    # 训练一个初步的决策树模型来识别重要特征
    initial_tree = DecisionTreeClassifier(random_state=42)
    initial_tree.fit(X_train_simple, y_train_init)
    
    # 根据特征重要性选择前几个最重要的数值特征
    feature_importances = pd.DataFrame({
        'feature': X_train_simple.columns,
        'importance': initial_tree.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("数值特征重要性:")
    print(feature_importances)
    
    # 接下来，为分类变量创建一个更简单的表示
    # 我们将为每个分类变量创建一个小型的决策树，看看哪些分类变量最重要
    categorical_importance = {}
    for col in categorical_cols:
        # 对分类变量进行编码
        df_temp = pd.get_dummies(df[col], prefix=col, drop_first=True)
        X_cat = pd.concat([df_temp, X.select_dtypes(include=['int64', 'float64'])], axis=1)
        
        # 拆分数据
        X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(X_cat, y, test_size=0.2, random_state=42)
        
        # 训练决策树
        tree_temp = DecisionTreeClassifier(max_depth=1, random_state=42)
        tree_temp.fit(X_cat_train, y_cat_train)
        
        # 获取准确率来评估重要性
        accuracy = accuracy_score(y_cat_test, tree_temp.predict(X_cat_test))
        categorical_importance[col] = accuracy
    
    # 按重要性排序分类特征
    categorical_importance = {k: v for k, v in sorted(categorical_importance.items(), key=lambda item: item[1], reverse=True)}
    print("\n分类特征重要性 (基于单独的决策树性能):")
    for col, acc in categorical_importance.items():
        print(f"{col}: {acc:.4f}")
    
    # 选择前2个最重要的分类特征和前3个最重要的数值特征
    top_categorical = list(categorical_importance.keys())[:2]
    top_numerical = feature_importances['feature'].tolist()[:3]
    
    selected_features = top_categorical + top_numerical
    print(f"\n选择的特征: {selected_features}")
    
    # 使用选择的特征创建训练集
    X_selected = X[selected_features].copy()
    
    # 对分类特征进行编码
    X_encoded = pd.get_dummies(X_selected, columns=top_categorical, drop_first=True)
    
    # 拆分数据
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # 创建一个深度为3的决策树模型
    decision_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    
    # 训练模型
    print("\n正在训练深度为3的决策树模型...")
    decision_tree.fit(X_train, y_train)
    
    # 评估模型
    y_pred = decision_tree.predict(X_test)
    
    print("\n==== 决策树模型评估 ====")
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 可视化混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('decision_tree_confusion_matrix.png')
    print("混淆矩阵已保存为: decision_tree_confusion_matrix.png")
    
    # 可视化决策树 - 使用matplotlib
    plt.figure(figsize=(20, 15))
    class_names = le.classes_.tolist()
    
    # 添加更多的树节点参数以改善可读性
    plot_tree(decision_tree, 
              filled=True, 
              feature_names=X_train.columns, 
              class_names=class_names, 
              rounded=True,
              precision=2,
              fontsize=12,
              proportion=True)
    
    plt.title("决策树模型 (max_depth=3)", fontsize=18)
    plt.tight_layout()
    plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
    print("决策树图已保存为: decision_tree.png")
    
    # 提取决策树规则（简化版本，文本输出而非图形）
    def get_decision_rules(tree, feature_names, class_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != -2 else "undefined!"
            for i in tree_.feature
        ]
        
        print("\n==== 决策树规则 ====")
        
        def recurse(node, depth, rules):
            indent = "  " * depth
            if tree_.feature[node] != -2:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                
                print(f"{indent}如果 {name} <= {threshold:.2f}:")
                recurse(tree_.children_left[node], depth + 1, rules + [f"{name} <= {threshold:.2f}"])
                
                print(f"{indent}如果 {name} > {threshold:.2f}:")
                recurse(tree_.children_right[node], depth + 1, rules + [f"{name} > {threshold:.2f}"])
            else:
                class_probabilities = tree_.value[node][0] / sum(tree_.value[node][0])
                predicted_class = np.argmax(class_probabilities)
                samples = np.sum(tree_.value[node])
                percentage = samples / tree_.n_node_samples[0] * 100
                
                rule_str = " AND ".join(rules)
                if len(rules) == 0:
                    rule_str = "所有情况"
                
                print(f"{indent}预测: {class_names[predicted_class]} (概率: {class_probabilities[predicted_class]:.2f}, 样本数: {int(samples)}, 占比: {percentage:.1f}%)")
                print(f"{indent}规则: {rule_str}\n")
                
        recurse(0, 0, [])
    
    # 输出决策规则
    get_decision_rules(decision_tree, X_train.columns, class_names)
    
    # 保存规则到文件
    with open('decision_tree_rules.txt', 'w', encoding='utf-8') as f:
        # 重定向标准输出到文件
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        get_decision_rules(decision_tree, X_train.columns, class_names)
        sys.stdout = original_stdout
    
    print("决策树规则已保存到: decision_tree_rules.txt")
    
    # 获取决策树的特征重要性
    feature_importance = pd.DataFrame({
        '特征': X_train.columns,
        '重要性': decision_tree.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    print("\n特征重要性:")
    print(feature_importance)
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 6))
    sns.barplot(x='重要性', y='特征', data=feature_importance)
    plt.title('决策树特征重要性', fontsize=16)
    plt.tight_layout()
    plt.savefig('decision_tree_feature_importance.png')
    print("特征重要性图已保存为: decision_tree_feature_importance.png")
    
    print("\n决策树模型训练和评估完成！")

except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'")
except Exception as e:
    print(f"发生错误：{e}") 