import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

#测试准确率: 81.25%

# 数据集路径
dataset_path = r"C:\Users\chr\Desktop\语音情感识别\CASIA\6"  # Windows路径

# 提取音频特征
def extract_features(file_path):
    # 加载音频文件
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    
    # 提取MFCC特征
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
    # 将MFCC特征调整为固定大小（例如，200个时间步长）
    mfccs = librosa.util.fix_length(mfccs, size=200, axis=1)
    
    return mfccs

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

# 构建改进的CNN模型
class ImprovedEmotionClassifierCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ImprovedEmotionClassifierCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 5 * 25, 256)  # 输入维度根据卷积层输出计算
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# 主函数
def main():
    # 准备数据集
    features, labels = prepare_dataset()
    
    # 将特征转换为NumPy数组
    features = np.array(features)
    labels = np.array(labels)
    
    # 标签编码
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # 转换为PyTorch张量
    # CNN输入格式: (batch_size, channels, height, width)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # 创建数据加载器
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 定义模型、损失函数和优化器
    input_channels = 1  # 单通道
    num_classes = len(label_encoder.classes_)
    model = ImprovedEmotionClassifierCNN(input_channels, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 学习率调度器
    
    # 打印模型结构
    print(model)
    
    # 训练模型
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # 更新学习率
        scheduler.step()
        
        # 每个epoch打印一次损失
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        print(f"测试准确率: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()