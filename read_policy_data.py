import pandas as pd

# 读取Excel文件
file_path = 'policy_data.xlsx'
try:
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 获取前五行数据
    first_five_rows = df.head(5)
    
    # 设置显示选项，确保所有列都能显示
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', 1000)  # 加宽显示窗口
    pd.set_option('display.unicode.east_asian_width', True)  # 正确处理中文宽度
    
    # 打印前五行数据
    print("policy_data.xlsx 的前五行数据：")
    print(first_five_rows)
    
    # 打印列名和数据类型信息
    print("\n数据列信息：")
    print(df.dtypes)
    
except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'")
except Exception as e:
    print(f"发生错误：{e}")