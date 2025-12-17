# LSTM Multi-Slice Time Series Forecast

## 项目概述

这是一个基于 LSTM（长短期记忆网络）的时间序列多切片递归预测系统。该项目实现了在不同时间点对时间序列数据进行训练和预测，用于评估模型在不同数据规模下的性能。

## 主要特性

- **多切片回测**：在不同的时间点进行数据切分，分别训练和预测
- **递归预测**：使用已有预测结果来预测未来值，实现长期预测
- **自动设备选择**：自动选择 GPU (MPS/CUDA) 或 CPU 进行计算
- **归一化处理**：使用 MinMaxScaler 进行数据归一化
- **隐藏状态连续传递**：在批次间传递 LSTM 隐藏状态，保持序列连贯性
- **性能评估**：计算 MAE、RMSE、R² 等指标

## 项目结构

```
LSTM_multi_slice_forecast.py  # 主程序
README.md                       # 项目说明
```

## 环境要求

```
Python >= 3.7
torch >= 1.9.0
pandas >= 1.2.0
numpy >= 1.19.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
openpyxl >= 3.0.0  # 用于读取 Excel 文件
```

### 安装依赖

```bash
pip install torch pandas numpy scikit-learn matplotlib openpyxl
```

## 使用说明

### 1. 准备数据

准备一个 Excel 文件，格式要求：
- 包含日期列和数值列
- 日期格式：YYYYMMDD（例如：20230101）
- 数值列包含要预测的目标变量

### 2. 修改配置参数

打开 `LSTM_multi_slice_forecast.py`，修改以下配置：

```python
# 文件路径配置
DATA_XLSX = "path/to/your/data.xlsx"     # 替换为你的数据文件路径
SHEET_NAME = "Sheet1"                     # 替换为工作表名称
DATE_COL = "date_column_name"             # 替换为日期列名
VALUE_COL = "value_column_name"           # 替换为数值列名

# 预测与回测配置
HORIZON = 12                              # 预测步数
TEST_CUT_POINTS = [100, 110, 120, 130, 140]  # 切片点

# LSTM 超参数
LSTM_CONFIG = {
    'n_seq': 48,          # 滑动窗口大小
    'n_hidden': 128,      # 隐层神经元数
    'n_layer': 2,         # LSTM 层数
    'batch_size': 64,     # 批次大小
    'n_epoch': 200,       # 训练轮数
    'learn_rate': 0.001   # 学习率
}
```

### 3. 运行程序

```bash
python LSTM_multi_slice_forecast.py
```

## 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `HORIZON` | 预测步数 | 12-24 |
| `n_seq` | 滑动窗口大小 | 24-48 |
| `n_hidden` | 隐层神经元数 | 64-256 |
| `n_layer` | LSTM 层数 | 1-3 |
| `batch_size` | 批次大小 | 32-128 |
| `n_epoch` | 训练轮数 | 100-500 |
| `learn_rate` | 学习率 | 0.0001-0.01 |

## 输出结果

程序会生成：

1. **控制台输出**：
   - 每个切片的训练进度
   - 预测效果指标（MAE、RMSE、R²）
   - 汇总表格

2. **CSV 文件**：`lstm_forecast_results.csv`
   - 保存所有切片的预测结果和评估指标

3. **可视化图表**：
   - 显示历史数据、预测数据和真实验证数据
   - 不同切片点用不同颜色标识

## 核心算法

### LSTM 模型结构

```
输入层 -> LSTM 层 -> 全连接层 -> 输出层
```

### 递归预测流程

1. 使用训练数据末尾的窗口作为初始输入
2. 预测下一步值
3. 将预测值加入窗口，移除最早的值
4. 重复步骤 2-3 直到达到预测步数

### 隐藏状态管理

- 在批次间传递 LSTM 隐藏状态（h, c）
- 使用 `detach()` 防止梯度爆炸
- 保持时间序列的连贯性

## 性能评估指标

- **MAE**（Mean Absolute Error）：平均绝对误差，单位同目标变量
- **RMSE**（Root Mean Squared Error）：均方根误差，对大误差更敏感
- **R²**（R-squared）：决定系数，范围 [-∞, 1]，值越接近 1 越好

## 常见问题

### Q: 训练速度太慢怎么办？
A: 可以尝试：
- 减少 `n_epoch`
- 减少 `n_hidden` 或 `n_layer`
- 增加 `batch_size`
- 使用 GPU（确保已安装 PyTorch GPU 版本）

### Q: 预测结果不好怎么办？
A: 可以尝试：
- 增加 `n_seq`（使用更多历史信息）
- 调整学习率
- 增加 `n_epoch`
- 检查数据质量

### Q: 如何处理非 Excel 数据源？
A: 修改 `read_full_data()` 函数，使用 pandas 支持的其他数据格式（CSV、JSON、SQL 等）

## 输出文件

- `lstm_forecast_results.csv`：包含以下列
  - `Cut_Point`：切片点索引
  - `Test_Date_Start`：测试开始日期
  - `MAE`：平均绝对误差
  - `RMSE`：均方根误差
  - `R2`：R² 系数

## 许可证

MIT License

## 作者

Created for time series forecasting research

## 更新日志

### v1.0 (2025-12-17)
- 初始版本发布
- 支持多切片回测
- 完整的性能评估
- 可视化结果展示
