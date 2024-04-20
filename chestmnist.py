import torch
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

data_flag = 'chestmnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
n_channels = info['n_channels']
n_classes = len(info['label'])
BATCH_SIZE = 64

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data

#################################### 可选配置 ##################################
val_data = None  # 验证集
test_data = DataClass(split='test', transform=data_transform, download=True)  # 测试集
DataLoader = None  # 默认为torch.utils.data.DataLoader
collate_fn = None  # 整理batch的函数，默认为None
criterion = torch.nn.CrossEntropyLoss()

#################################### 必选配置 ###############################
train_data = DataClass(split='train', transform=data_transform, download=True)  # 训练集，必选


# 把batch数据放到gpu device上
def data_to_device(batch_data, device):
    raise NotImplementedError


# 在data_loader上评估模型，结果以字典返回
def eval(model, data_loader, device) -> dict:
    raise NotImplementedError


# 在batch_data上计算当前模型的损失，结果以字典返回（e.g., {'loss': loss}）
def compute_loss(batch_data, model, device) -> dict:
    raise NotImplementedError


# 获取需要训练的模型
def get_model(*args, **kwargs) -> torch.nn.Module:
    raise NotImplementedError
