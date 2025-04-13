import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取Excel文件
file_path = 'policy_data.xlsx'
try:
    # 读取Excel文件
    print("正在读取数据...")
    df = pd.read_excel(file_path)
    
    # 1. 数据基本信息
    print("\n==== 数据基本信息 ====")
    print(f"数据维度: {df.shape}")
    print(f"总行数: {df.shape[0]}, 总列数: {df.shape[1]}")
    
    # 2. 查看前几行数据
    print("\n==== 数据前5行 ====")
    print(df.head())
    
    # 3. 数据类型
    print("\n==== 数据类型信息 ====")
    print(df.dtypes)
    
    # 4. 统计信息概览
    print("\n==== 数值型数据统计描述 ====")
    print(df.describe())
    
    # 5. 非数值列的统计
    print("\n==== 分类数据统计 ====")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\n{col} 值分布:")
        value_counts = df[col].value_counts()
        print(value_counts)
        print(f"唯一值数量: {df[col].nunique()}")
    
    # 6. 缺失值分析
    print("\n==== 缺失值分析 ====")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_info = pd.DataFrame({
        '缺失值数量': missing_data,
        '缺失比例(%)': missing_percent
    })
    print(missing_info[missing_info['缺失值数量'] > 0])
    
    # 如果没有缺失值
    if missing_data.sum() == 0:
        print("数据中没有缺失值")
    
    # 7. 创建可视化
    print("\n==== 生成数据可视化 ====")
    
    # 设置图形大小
    plt.figure(figsize=(15, 10))
    
    # 7.1 年龄分布
    plt.subplot(2, 3, 1)
    sns.histplot(df['age'], kde=True)
    plt.title('年龄分布')
    
    # 7.2 性别分布
    plt.subplot(2, 3, 2)
    gender_counts = df['gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
    plt.title('性别分布')
    
    # 7.3 收入水平分布
    plt.subplot(2, 3, 3)
    income_counts = df['income_level'].value_counts()
    sns.barplot(x=income_counts.index, y=income_counts.values)
    plt.title('收入水平分布')
    plt.xticks(rotation=45)
    
    # 7.4 教育水平分布
    plt.subplot(2, 3, 4)
    edu_counts = df['education_level'].value_counts()
    sns.barplot(x=edu_counts.index, y=edu_counts.values)
    plt.title('教育水平分布')
    plt.xticks(rotation=45)
    
    # 7.5 保费金额分布
    plt.subplot(2, 3, 5)
    sns.histplot(df['premium_amount'], kde=True)
    plt.title('保费金额分布')
    
    # 7.6 续保情况
    plt.subplot(2, 3, 6)
    renewal_counts = df['renewal'].value_counts()
    plt.pie(renewal_counts, labels=renewal_counts.index, autopct='%1.1f%%')
    plt.title('续保情况分布')
    
    plt.tight_layout()
    plt.savefig('eda_基本分布图.png')
    print("基本分布图已保存为: eda_基本分布图.png")
    
    # 8. 相关性分析
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('数值特征相关性分析')
    plt.tight_layout()
    plt.savefig('eda_相关性分析.png')
    print("相关性分析图已保存为: eda_相关性分析.png")
    
    # 9. 按续保情况分组分析
    print("\n==== 按续保情况分组分析 ====")
    renewal_groups = df.groupby('renewal')
    for name, group in renewal_groups:
        print(f"\n续保情况 '{name}' 的描述统计:")
        print(group.describe())
    
    # 10. 年龄与续保关系
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='renewal', y='age', data=df)
    plt.title('续保情况与年龄的关系')
    plt.savefig('eda_续保与年龄.png')
    print("续保与年龄关系图已保存为: eda_续保与年龄.png")
    
    # 11. 收入水平与续保关系
    plt.figure(figsize=(10, 6))
    income_renewal = pd.crosstab(df['income_level'], df['renewal'])
    income_renewal_percent = income_renewal.div(income_renewal.sum(axis=1), axis=0) * 100
    income_renewal_percent.plot(kind='bar', stacked=True)
    plt.title('收入水平与续保率的关系')
    plt.ylabel('百分比 (%)')
    plt.savefig('eda_收入与续保.png')
    print("收入与续保关系图已保存为: eda_收入与续保.png")
    
    # 12. 保单类型与续保关系
    plt.figure(figsize=(14, 8))
    policy_renewal = pd.crosstab(df['policy_type'], df['renewal'])
    policy_renewal_percent = policy_renewal.div(policy_renewal.sum(axis=1), axis=0) * 100
    policy_renewal_percent.plot(kind='barh', stacked=True)
    plt.title('保单类型与续保率的关系')
    plt.ylabel('保单类型')
    plt.xlabel('百分比 (%)')
    plt.tight_layout()
    plt.savefig('eda_保单类型与续保.png')
    print("保单类型与续保关系图已保存为: eda_保单类型与续保.png")
    
    # 13. 理赔历史与续保关系
    plt.figure(figsize=(10, 6))
    claim_renewal = pd.crosstab(df['claim_history'], df['renewal'])
    claim_renewal_percent = claim_renewal.div(claim_renewal.sum(axis=1), axis=0) * 100
    claim_renewal_percent.plot(kind='bar', stacked=True)
    plt.title('理赔历史与续保率的关系')
    plt.ylabel('百分比 (%)')
    plt.savefig('eda_理赔与续保.png')
    print("理赔与续保关系图已保存为: eda_理赔与续保.png")
    
    # 14. 婚姻状况与续保关系
    plt.figure(figsize=(10, 6))
    marital_renewal = pd.crosstab(df['marital_status'], df['renewal'])
    marital_renewal_percent = marital_renewal.div(marital_renewal.sum(axis=1), axis=0) * 100
    marital_renewal_percent.plot(kind='bar', stacked=True)
    plt.title('婚姻状况与续保率的关系')
    plt.ylabel('百分比 (%)')
    plt.savefig('eda_婚姻与续保.png')
    print("婚姻与续保关系图已保存为: eda_婚姻与续保.png")
    
    print("\nEDA分析完成!")
    
except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'")
except Exception as e:
    print(f"发生错误：{e}") 