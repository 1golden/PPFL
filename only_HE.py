import flgo.algorithm.fedbase as fedbase
from phe import paillier
import collections
import copy
import torch
import numpy as np
from flgo.utils import fmodule


def _model_from_tensor_with_grad(mt, model_class):
    r"""
    Create model from torch.Tensor with gradients enabled

    Args:
        mt (torch.Tensor): the tensor
        model_class (FModule): the class defines the model architecture

    Returns:
        The new model created from tensors with gradients enabled
    """
    res = model_class
    cnt = 0
    end = 0
    for i, p in enumerate(res.parameters()):
        beg = 0 if cnt == 0 else end
        end = end + p.view(-1).size()[0]
        p.data = mt[beg:end].contiguous().view(p.data.size())
        cnt += 1
    return res


# 注意：已移除 torch.no_grad() 上下文管理器以启用梯度跟踪。

def _model_to_tensor(m):
    r"""
    Convert the model parameters to torch.Tensor

    Args:
        m (FModule): the model

    Returns:
        The torch.Tensor of model parameters
    """
    return torch.cat([mi.data.view(-1) for mi in m.parameters()])


"""
def _model_from_tensor(mt, model_class):

    res = model_class
    cnt = 0
    end = 0
    with torch.no_grad():
        for i, p in enumerate(res.parameters()):
            beg = 0 if cnt == 0 else end
            end = end + p.view(-1).size()[0]
            p.data = mt[beg:end].contiguous().view(p.data.size())
            cnt += 1
    return res
"""


def add_noise(parameters, sigma, dp, device):
    noise = None
    # 不加噪声
    if dp == 0:
        return parameters
    # 拉普拉斯噪声
    elif dp == 1:
        noise = torch.tensor(np.random.laplace(0, sigma, parameters.shape), device=device)
    # 高斯噪声
    else:
        noise = torch.cuda.FloatTensor(parameters.shape).normal_(0, sigma)

    return parameters.add_(noise)


# torch转list加密
def encrypt_vector(public_key, parameters):
    parameters = parameters.tolist()
    parameters = [public_key.encrypt(parameter, precision=1e-3) for parameter in parameters]
    return parameters


'''

def encrypt_model(model, public_key) -> list:
    encrypted_model = []
    count = 0
    for param in model.parameters():
        param_data_list = param.data.to('cuda').view(-1).tolist()
        count += 1
        print('done', count)
        encrypted_model.append([public_key.encrypt(x, precision=1e-4) for x in param_data_list])
    return encrypted_model


def decrypt_model(encrypted_model, private_key, original_model_shapes):
    decrypted_model = []
    for encrypted_param, shape in zip(encrypted_model, original_model_shapes):
        decrypted_param = [private_key.decrypt(x) for x in encrypted_param]
        param_tensor = torch.tensor(decrypted_param)
        param_tensor = torch.reshape(param_tensor, shape)
        decrypted_model.append(param_tensor)
    return decrypted_model


'''


# list解密为torch
def decrypt_model(encrypted_model: list, private_key):
    decrypted_param = torch.tensor([private_key.decrypt(x) for x in encrypted_model])
    return decrypted_param


class Server(fedbase.BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'sigma': 0.1})
        self.gv.public_key, self.gv.private_key = paillier.generate_paillier_keypair(n_length=128)
        model_shapes = [param.shape for param in self.model.parameters()]
        print(model_shapes)
        self.tensors = _model_to_tensor(self.model)
        # print(self.tensors)
        tmp = _model_from_tensor_with_grad(self.tensors.to('cuda'), self.model)
        print(self.model, '\n', tmp, (self.model == tmp))
        self.encrypted_tensors = encrypt_vector(self.gv.public_key, self.tensors)

    def aggregate(self, encrypted_models: list, *args, **kwargs):
        if len(encrypted_models) == 0: return self.encrypted_tensors
        # Assuming all encrypted models have the same structure
        aggregated_encrypted_model = [0] * len(encrypted_models[0])
        local_data_vols = [c.datavol for c in self.clients]  # 一个列表 local_data_vols，包含每个客户端的数据量
        total_data_vol = sum(local_data_vols)  # 所有客户端数据量的总和
        p = [1.0 * local_data_vols[cid] / total_data_vol for cid in
             self.received_clients]
        sump = sum(p)
        p = [pk / sump for pk in p]  # 一个列表，当前轮次选择的每个参与聚合的客户端的数据量占总数据量的比例
        K = len(encrypted_models)  # 当前轮次参与训练的客户端数量
        N = self.num_clients  # 和所有客户端的总数

        for i in range(len(aggregated_encrypted_model)):
            for pk, encrypted_model in zip(p, encrypted_models):
                encrypted_gradient = encrypted_model[i]
                if self.aggregation_option == 'weighted_scale':
                    aggregated_encrypted_model[i] += encrypted_gradient * pk * K / N
                elif self.aggregation_option == 'uniform':
                    aggregated_encrypted_model[i] += encrypted_gradient / K
                elif self.aggregation_option == 'weighted_com':
                    aggregated_encrypted_model[i] += (1.0 - sum(p)) * self.encrypted_tensors[
                        i] + encrypted_gradient * pk
                else:
                    aggregated_encrypted_model[i] += encrypted_gradient * pk

        self.encrypted_tensors = aggregated_encrypted_model
        print('---------------------------------------------------------------------------')

    def iterate(self):
        self.selected_clients = self.sample()
        en_grads = self.communicate(self.selected_clients)['encrypted_model']
        self.aggregate(en_grads)

    def pack(self, *args, **kwargs):
        return {
            'model': copy.deepcopy(self.model),
            'encrypted_tensors': self.encrypted_tensors
        }

    def global_test(self, model=None, flag: str = 'val'):
        tmp = decrypt_model(self.encrypted_tensors, self.gv.private_key)
        print(tmp)
        model = _model_from_tensor_with_grad(tmp.to('cuda'), self.model)
        print(model)
        print('标记一下！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！')
        all_metrics = collections.defaultdict(list)
        for c in self.clients:
            client_metrics = c.test(model, flag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        print(all_metrics)
        print("结束服务器端global_test训练时间----------------------------------------------------------")
        return all_metrics

    def test(self, model=None, flag: str = 'test'):
        data = self.test_data if flag == 'test' else self.val_data
        print(data)
        if data is None:
            return {}
        tmp = decrypt_model(self.encrypted_tensors, self.gv.private_key)
        model = _model_from_tensor_with_grad(tmp.to('cuda'), self.model)
        print(self.calculator.test(model=model, dataset=data,
                                   batch_size=min(self.option['test_batch_size'], len(data)),
                                   num_workers=self.option['num_workers'], pin_memory=self.option['pin_memory']))
        print("结束服务器端test训练时间----------------------------------------------------------")

        return self.calculator.test(model=model, dataset=data,
                                    batch_size=min(self.option['test_batch_size'], len(data)),
                                    num_workers=self.option['num_workers'], pin_memory=self.option['pin_memory'])


class Client(fedbase.BasicClient):

    def pack(self, disturbed_model, *args, **kwargs):
        tensors = _model_to_tensor(disturbed_model)
        print('客户端打包前的模型转张量', tensors)
        encrypted_gradient = encrypt_vector(self.gv.public_key, tensors)
        print('客户端pack完毕！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！')
        print('---------------------------------------------------------------------------')
        return {'encrypted_model': encrypted_gradient}

    def unpack(self, received_pkg):
        encrypted_tensors = received_pkg['encrypted_tensors']
        model = received_pkg['model']
        decrypt_tensors = decrypt_model(encrypted_tensors, self.gv.private_key)
        dec_model = _model_from_tensor_with_grad(decrypt_tensors.to('cuda'), model)
        print('客户端unpack完毕！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！')
        print('客户端解密得到的全局模型张量', decrypt_tensors)
        print(dec_model.parameters())
        print('---------------------------------------------------------------------------')
        return dec_model

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        disturbed = self.train(model)
        cpkg = self.pack(disturbed)
        return cpkg

    @fmodule.with_multi_gpus
    def train(self, global_model):
        # 记录全局模型参数
        original_model = copy.deepcopy(global_model)
        # 冻结全局模型梯度
        global_model.zero_grad()
        optimizer = self.calculator.get_optimizer(global_model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for epoch in range(self.num_steps):
            global_model.zero_grad()  # 初始化梯度为0
            batch_data = self.get_batch_data()
            loss = self.calculator.compute_loss(global_model, batch_data)['loss']
            loss.backward()
            # 裁剪梯度
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(parameters=global_model.parameters(), max_norm=self.clip_grad)
            optimizer.step()

        print('客户端在本地训练梯度下降完的的模型参数转张量', _model_to_tensor(global_model))
        # Add Laplace noise for differential privacy

        model = global_model.to(torch.device('cuda'))
        return model

    def test(self, global_model, flag='val'):
        data = self.train_data if flag == 'train' else self.val_data
        if data is None: return {}
        print("现在是客户端test时间----------------------------------------------------------")
        print(
            self.calculator.test(global_model, data, min(self.test_batch_size, len(data)), self.option['num_workers']))
        return self.calculator.test(global_model, data, min(self.test_batch_size, len(data)),
                                    self.option['num_workers'])


class PPFL_HE:
    Server = Server
    Client = Client
