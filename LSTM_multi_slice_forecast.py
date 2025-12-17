#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt

# =============== 1. 基础设置与配置 ===============

# 尝试自动选择 GUI 后端
def _select_gui_backend():
    if "ipykernel" in sys.modules or os.environ.get("MPLBACKEND"):
        return
    for b in ["MacOSX", "TkAgg", "QtAgg"]:
        try:
            matplotlib.use(b, force=True)
            return
        except Exception:
            continue
_select_gui_backend()
print("Matplotlib backend:", matplotlib.get_backend())

# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("device:", device)

# ===== 文件路径配置（请根据实际数据修改）=====
DATA_XLSX = "path/to/your/data.xlsx"  # 替换为你的数据文件路径
SHEET_NAME = "Sheet1"  # 替换为你的工作表名称
DATE_COL = "date_column_name"  # 替换为你的日期列名
VALUE_COL = "value_column_name"  # 替换为你的数值列名

# ===== 预测与回测配置（可根据需要修改）=====
HORIZON = 12             # 往后预测多少步
TEST_CUT_POINTS = [100, 110, 120, 130, 140]  # 在这些时间点分别切断，训练并预测

# ===== LSTM 超参数（可根据需要调整）=====
LSTM_CONFIG = {
    'n_seq': 48,         # 滑动窗口大小
    'n_input': 1,
    'n_hidden': 128,     # 隐层神经元数
    'n_layer': 2,        # LSTM层数
    'n_output': 1,
    'batch_size': 64,    # 批次大小
    'n_epoch': 200,      # 每个切片重新训练的轮数
    'learn_rate': 0.001  # 学习率
}

# =============== 2. 模型定义 ===============

class LSTM(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, n_output, n_seq):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.n_seq = n_seq
        # batch_first=True -> (batch, seq, feature)
        self.lstm = nn.LSTM(n_input, n_hidden, n_layer, batch_first=True)
        self.linear = nn.Linear(n_hidden * n_seq, n_output)

    def forward(self, x, h0=None, c0=None):
        # 如果没有提供 h0, c0，则默认为 0
        if h0 is None or c0 is None:
            batch_size = x.size(0)
            h0 = torch.zeros(self.n_layer, batch_size, self.n_hidden).to(x.device)
            c0 = torch.zeros(self.n_layer, batch_size, self.n_hidden).to(x.device)
            
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        # Flatten: (batch, seq, hidden) -> (batch, seq * hidden)
        out = lstm_out.reshape(x.size(0), -1)
        out = self.linear(out)
        return out, hn, cn

# =============== 3. 工具函数 ===============

def read_full_data(xlsx_path: str) -> pd.Series:
    """读取并清洗整个数据集"""
    print(f"正在读取文件: {xlsx_path} ...")
    try:
        df = pd.read_excel(xlsx_path, sheet_name=SHEET_NAME)
        # 处理日期
        df[DATE_COL] = pd.to_datetime(df[DATE_COL].astype(str), format='%Y%m%d')
        # 设置索引
        df = df.set_index(DATE_COL).sort_index()
        s = df[VALUE_COL].dropna()
        print(f"数据加载成功，总长度: {len(s)}")
        return s
    except Exception as e:
        print(f"读取数据出错: {e}")
        raise e

def make_sequences(data: np.ndarray, seq_len: int):
    """构造滑动窗口样本 (X, Y)"""
    X, Y = [], []
    arr = data.reshape(-1, 1)
    if len(arr) <= seq_len:
        return np.array([]), np.array([])
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i + seq_len])
        Y.append(arr[i + seq_len])
    return np.array(X), np.array(Y)

def train_lstm_for_slice(train_data_raw, config):
    """
    针对某个切片的数据进行 LSTM 训练（使用归一化）
    train_data_raw: 原始尺度的一维 numpy 数组
    """
    # 1. 数据归一化
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data_raw.reshape(-1, 1))
    
    # 2. 构造训练集
    X_arr, Y_arr = make_sequences(train_scaled, config['n_seq'])
    if len(X_arr) == 0:
        raise ValueError("训练数据太少，不足以构成一个窗口")
        
    X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_arr, dtype=torch.float32, device=device)
    
    # 3. 初始化模型
    model = LSTM(
        config['n_input'], config['n_hidden'], config['n_layer'], 
        config['n_output'], config['n_seq']
    ).to(device)
    
    loss_fun = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learn_rate'])
    
    # 4. 训练循环
    model.train()
    n_samples = X_t.size(0)
    
    # 使用顺序批次训练，隐藏状态连续传递
    for epoch in range(config['n_epoch']):
        epoch_loss = 0.0
        h = None  # 初始隐藏状态设为None
        c = None  # 初始细胞状态设为None
        
        for j in range(0, n_samples, config['batch_size']):
            # 顺序取批次（不随机打乱，保持时序连贯性）
            x_batch = X_t[j : j + config['batch_size']]
            y_batch = Y_t[j : j + config['batch_size']]
            
            out, hn, cn = model(x_batch, h, c)
            loss = loss_fun(out, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
            
            # 更新隐藏状态和细胞状态，用于下一批次
            h = hn.detach()  # detach() 防止梯度爆炸
            c = cn.detach()
        
        # 可选：打印训练进度
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{config['n_epoch']} Loss: {epoch_loss/n_samples:.6f}")
            
    return model, scaler

def recursive_forecast(model, initial_window_scaled, n_steps, scaler):
    """
    递归预测（使用归一化）：预测 -> 移位 -> 预测
    initial_window_scaled: 形状 (n_seq, 1) 的归一化后数据
    scaler: 用于反归一化的 MinMaxScaler 对象
    """
    model.eval()
    curr_window = torch.tensor(initial_window_scaled, dtype=torch.float32, device=device).unsqueeze(0) # (1, seq, 1)
    
    preds_raw = []
    h = None  # 初始隐藏状态
    c = None  # 初始细胞状态
    
    with torch.no_grad():
        for _ in range(n_steps):
            # 1. 预测下一步 (归一化值)，传递隐藏状态
            out, hn, cn = model(curr_window, h, c)
            pred_val_scaled = out.item()
            
            # 2. 反归一化得到原始尺度值
            pred_val_raw = scaler.inverse_transform(np.array([[pred_val_scaled]]))[0][0]
            
            # 3. 保存原始尺度的预测值
            preds_raw.append(pred_val_raw)
            
            # 4. 更新窗口：丢弃第一个，追加归一化的预测值
            # curr_window: (1, seq, 1)
            pred_tensor = torch.tensor([[[pred_val_scaled]]], device=device) # (1, 1, 1)
            curr_window = torch.cat((curr_window[:, 1:, :], pred_tensor), dim=1)
            
            # 5. 更新隐藏状态和细胞状态
            h = hn
            c = cn
            
    return np.array(preds_raw)

# =============== 4. 主程序 ===============

def main():
    # A. 读取全量数据
    full_series = read_full_data(DATA_XLSX)
    full_values = full_series.to_numpy(np.float32)
    
    results_summary = []
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    plt.figure(figsize=(14, 7))
    # 背景：全量真实数据
    plt.plot(full_series.index, full_series.values, color='gray', alpha=0.5, label='History (All Data)')

    print(f"\n开始 LSTM 多切片回测 (Horizon={HORIZON})...")
    
    for i, cut_idx in enumerate(TEST_CUT_POINTS):
        print(f"\n>>> 处理切片点 iloc[{cut_idx}] ...")
        
        # 1. 数据切分
        if cut_idx > len(full_values):
            print("切片点超出数据长度，跳过")
            continue
            
        train_raw = full_values[:cut_idx]
        
        # 验证集 (用于计算误差)
        if cut_idx + HORIZON > len(full_values):
            print(f"警告：剩余数据不足 {HORIZON} 步，跳过")
            continue
            
        truth_slice = full_values[cut_idx : cut_idx + HORIZON]
        truth_dates = full_series.index[cut_idx : cut_idx + HORIZON]
        
        # 3. 训练模型
        # 注意：这里每次都是重新训练一个新模型，保证无未来信息泄露
        print(f"  正在训练 LSTM ({LSTM_CONFIG['n_epoch']} epochs)...")
        try:
            model, scaler = train_lstm_for_slice(train_raw, LSTM_CONFIG)
        except Exception as e:
            print(f"  训练失败: {e}")
            continue
        
        # 4. 递归预测
        # 初始窗口：训练集最后 n_seq 个点
        n_seq = LSTM_CONFIG['n_seq']
        if len(train_raw) < n_seq:
            print("训练数据少于窗口长度，跳过")
            continue
            
        # 对初始窗口进行归一化
        initial_window_raw = train_raw[-n_seq:].reshape(-1, 1)
        initial_window_scaled = scaler.transform(initial_window_raw)
        
        print("  正在递归预测...")
        pred_values = recursive_forecast(model, initial_window_scaled, HORIZON, scaler)
        
        # 5. 计算指标
        mae = mean_absolute_error(truth_slice, pred_values)
        rmse = np.sqrt(mean_squared_error(truth_slice, pred_values))
        r2 = r2_score(truth_slice, pred_values) if len(truth_slice) > 1 else 0
        
        print(f"  结果: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        results_summary.append({
            "Cut_Point": cut_idx,
            "Test_Date_Start": str(truth_dates[0].date()),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
        
        # 6. 绘图
        color = colors[i % len(colors)]
        plt.plot(truth_dates, pred_values, marker='o', markersize=4, linestyle='--', color=color, 
                 label=f'LSTM Pred (Cut={cut_idx})')
        # 加深显示对应的真实值段
        plt.plot(truth_dates, truth_slice, color=color, alpha=0.3, linewidth=3)

    # B. 总结输出
    print("\n===== LSTM 多切片回测总结 =====")
    df_res = pd.DataFrame(results_summary)
    if not df_res.empty:
        print(df_res.to_string(index=False))
        # 可选：保存结果到CSV
        df_res.to_csv("lstm_forecast_results.csv", index=False)
        print("\n结果已保存到 lstm_forecast_results.csv")
    else:
        print("没有产生有效预测结果。")

    plt.title(f"LSTM Multi-Slice Recursive Forecast (Horizon={HORIZON}, Seq={LSTM_CONFIG['n_seq']})")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
