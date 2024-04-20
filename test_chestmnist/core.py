import importlib
import json
import os
from flgo.benchmark.toolkits.cv.classification import GeneralCalculator, FromDatasetPipe, FromDatasetGenerator
from .config import train_data
import torch.utils.data
import flgo.benchmark.base as fbb

try:
    from .config import test_data
except:
    test_data = None
try:
    from .config import val_data
except:
    val_data = None


class TaskPipe(fbb.FromDatasetPipe):
    TaskDataset = torch.utils.data.Subset

    def __init__(self, task_path, train_data, val_data=None, test_data=None):
        super(FromDatasetPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))  # 生成用户的名字
        feddata = {'client_names': client_names}  # 记录用户的名字属性为client_names
        for cid in range(len(client_names)): feddata[client_names[cid]] = {
            'data': generator.local_datas[cid], }  # 记录每个用户的本地数据划分信息，以其名字为关键字索引
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:  # 保存为data.json文件到任务目录中
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option: dict) -> dict:
        """
        该函数返回一个具有以下格式的字典{'server': {'test':data1, 'val':data2}, 'client_name1':{'train':..., 'val':..., 'test':...}, ...}
        其中'server'和'client_name1'等为联邦里个体的名字，服务器默认名字叫server，用户的名字与client_names中生成的名字一致，存储在self.feddata['client_names']中
        每个个体包含的数据集用一个字典表示，该字典的键值对会被作为相应个体的属性，例如上述字典中，服务器个体最后将有属性
        Server.test_data = data1, Server.val_data=data2。对于其他个体的情况以此类推。
        """
        train_data = self.train_data
        test_data = self.test_data
        val_data = self.val_data
        # 默认服务器持有测试集和验证集，训练时动态从测试集中划分一部分出来作为验证集，划分比例由外部参数test_holdout决定
        if val_data is None:
            server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        else:
            server_data_test = test_data
            server_data_val = val_data
        # 构造服务器的数据
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        # rearrange data for clients
        # 挨个构造各用户的数据，
        for cid, cname in enumerate(self.feddata['client_names']):
            # 根据数据索引从原数据集中构造Subset子集
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'])
            # 本地数据划分成训练集和测试集，划分比例由外部参数字典键train_holdout决定，默认为0.2
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            # 若执行本地测试，则从验证集里划一半出来当本地测试集
            if running_time_option['train_holdout'] > 0 and running_time_option['local_test']:
                cdata_val, cdata_test = self.split_dataset(cdata_val, 0.5)
            else:
                cdata_test = None
            # 构造当前用户的本地数据字典
            task_data[cname] = {'train': cdata_train, 'val': cdata_val, 'test': cdata_test}
        return task_data


class BasicTaskCalculator(fbb.AbstractTaskCalculator):
    r"""
    Support task-specific computation when optimizing models, such
    as putting data into device, computing loss, evaluating models,
    and creating the data loader
    """

    def __init__(self, device, optimizer_name='sgd'):
        r"""
        Args:
            device (torch.device): device
            optimizer_name (str): the name of the optimizer
        """
        self.device = device
        self.optimizer_name = optimizer_name
        self.criterion = None
        self.DataLoader = torch.utils.data.DataLoader
        self.collect_fn = None

    def to_device(self, data, *args, **kwargs):
        return NotImplementedError

    def get_dataloader(self, dataset, batch_size=64, *args, **kwargs):
        return NotImplementedError

    def test(self, model, data, *args, **kwargs):
        return NotImplementedError

    def compute_loss(self, model, data, *args, **kwargs):
        return NotImplementedError

    def get_optimizer(self, model=None, lr=0.1, weight_decay=0, momentum=0):
        r"""
        Create optimizer of the model parameters

        Args:
            model (torch.nn.Module): model
            lr (float): learning rate
            weight_decay (float): the weight_decay coefficient
            momentum (float): the momentum coefficient

        Returns:
            the optimizer
        """
        OPTIM = getattr(importlib.import_module('torch.optim'), self.optimizer_name)
        filter_fn = filter(lambda p: p.requires_grad, model.parameters())
        if self.optimizer_name.lower() == 'sgd':
            return OPTIM(filter_fn, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif self.optimizer_name.lower() in ['adam', 'rmsprop', 'adagrad']:
            return OPTIM(filter_fn, lr=lr, weight_decay=weight_decay)
        else:
            raise RuntimeError("Invalid Optimizer.")
