import pandas as pd

# 读取预测结果
predictions = pd.read_excel('model_predictions.xlsx')

# 显示前5行
print("预测结果前5行:")
print(predictions.head())

# 显示预测统计
print("\n逻辑回归预测统计:")
print(predictions['逻辑回归_预测'].value_counts())
print("\n决策树预测统计:")
print(predictions['决策树_预测'].value_counts())

# 统计两个模型预测一致和不一致的数量
predictions['预测一致'] = predictions['逻辑回归_预测'] == predictions['决策树_预测']
agreement_count = predictions['预测一致'].sum()
disagreement_count = len(predictions) - agreement_count

print(f"\n两个模型预测一致的样本数: {agreement_count} ({agreement_count/len(predictions)*100:.1f}%)")
print(f"两个模型预测不一致的样本数: {disagreement_count} ({disagreement_count/len(predictions)*100:.1f}%)")

# 计算概率分布
print("\n逻辑回归预测概率分布:")
print(predictions['逻辑回归_概率'].describe())

print("\n决策树预测概率分布:")
print(predictions['决策树_概率'].describe())