import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#测试准确率: 72.92%

# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 选择第一个GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# 数据集路径
dataset_path = r"C:\Users\chr\Desktop\语音情感识别\CASIA\6"  # Windows路径

# 提取音频特征
def extract_features(file_path):
    # 加载音频文件
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    
    # 提取MFCC特征
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
    # 转置以适应LSTM输入要求
    mfccs_processed = mfccs.T
    
    return mfccs_processed

# 准备数据集
def prepare_dataset():
    features = []
    labels = []
    
    # 遍历数据集中的每个文件夹
    for emotion in os.listdir(dataset_path):
        emotion_path = os.path.join(dataset_path, emotion)
        if os.path.isdir(emotion_path):
            # 遍历文件夹中的每个音频文件
            for file_name in os.listdir(emotion_path):
                if file_name.endswith(".wav"):
                    # 提取特征
                    file_path = os.path.join(emotion_path, file_name)
                    feature = extract_features(file_path)
                    
                    # 添加到特征和标签列表
                    features.append(feature)
                    labels.append(emotion)
    
    return features, labels

# 构建LSTM模型
class EmotionClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EmotionClassifierLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 准备数据集
    features, labels = prepare_dataset()
    
    # 将特征和标签转换为NumPy数组
    features = [torch.tensor(feature, dtype=torch.float32) for feature in features]
    labels = np.array(labels)
    
    # 标签编码
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # 转换为PyTorch张量
    X_train = torch.nn.utils.rnn.pad_sequence(X_train, batch_first=True)
    X_test = torch.nn.utils.rnn.pad_sequence(X_test, batch_first=True)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # 创建数据加载器
    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 定义模型、损失函数和优化器
    input_size = X_train.shape[2]
    hidden_size = 128
    num_layers = 2
    num_classes = len(label_encoder.classes_)
    model = EmotionClassifierLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # 每个epoch打印一次损失
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        print(f"测试准确率: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()