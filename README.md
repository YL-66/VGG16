# VGG16 Fashion-MNIST

使用 PyTorch 实现的 VGG16 风格卷积神经网络，用于 Fashion-MNIST 图像分类。输入灰度图会缩放到 `224x224`，模型输出 10 个类别的预测结果。

## 项目文件

- `model.py`：VGG16 网络结构与参数初始化
- `model_train.py`：训练、验证及损失/准确率曲线绘制
- `model_test.py`：加载最佳模型并在测试集上评估
- `data/`：Fashion-MNIST 数据集（首次运行时也会自动下载）

## 环境安装

建议使用 Python 3.9 或更高版本，并根据本机 CUDA 环境安装对应版本的 PyTorch。

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 查看模型结构

```bash
python model.py
```

## 训练模型

```bash
python model_train.py
```

训练脚本默认训练 20 个 epoch，使用 Adam（学习率 `0.001`、批大小 `32`）。训练完成后会生成：

- `best_model.pth`：验证集表现最佳的模型参数
- `acc_loss.png`：训练/验证损失与准确率曲线

程序会自动选择 CUDA；没有可用 GPU 时使用 CPU。Fashion-MNIST 下载可能需要网络连接。

## 测试模型

训练完成后运行：

```bash
python model_test.py
```

脚本会加载当前目录下的 `best_model.pth`，打印测试集准确率，并输出样本预测类别与真实类别。

## Fashion-MNIST 类别

`T-shirt/top`、`Trouser`、`Pullover`、`Dress`、`Coat`、`Sandal`、`Shirt`、`Sneaker`、`Bag`、`Ankle boot`。

## 许可证

本项目未指定额外许可证；使用前请根据需要补充许可证文件。
