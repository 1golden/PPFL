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


def _model_to_tensor(m):
    r"""
    Convert the model parameters to torch.Tensor

    Args:
        m (FModule): the model

    Returns:
        The torch.Tensor of model parameters
    """
    return torch.cat([mi.data.view(-1) for mi in m.parameters()])


def calculate_sensitivity(lr, clip, dataset_size):
    return 2 * lr * clip / dataset_size


def custom_clip_grad_norm_(parameters, max_norm, norm_type=2):
    """
    Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    :param parameters: (Iterable[Tensor] or Tensor): 一个包含参数的迭代器
    :param max_norm: (float or int): 最大范数值
    :param norm_type: (float or int): 范数类型，默认为2范数
    """
    total_norm = 0.0
    for p in parameters:
        param_norm = p.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    clip_coef = max(1, total_norm / max_norm)
    for p in parameters:
        p.data.mul_(1 / clip_coef)


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
    parameters = [public_key.encrypt(parameter, precision=1e-4) for parameter in parameters]
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
        self.init_algo_para({'epsilon': 10})
        # self.init_algo_para({'epsilon': 10,''})
        self.gv.public_key, self.gv.private_key = paillier.generate_paillier_keypair(n_length=128)
        self.tensors = _model_to_tensor(self.model)
        # print(self.tensors)
        # tmp = _model_from_tensor_with_grad(self.tensors.to('cuda'), self.model)
        # print(self.model, '\n', tmp, (self.model == tmp))
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
        self.sensitivity = cal_sensitivity_part(self.learning_rate, len(self.selected_clients))
        print("\n before communicate sensitivity `````````````````:", self.sensitivity)
        en_grads = self.communicate(self.selected_clients)['encrypted_model']
        self.aggregate(en_grads)

    def pack(self, *args, **kwargs):
        return {
            'model': copy.deepcopy(self.model),
            'encrypted_tensors': self.encrypted_tensors,
            'sensitivity': self.sensitivity
        }

    def sample(self):
        """
        Sample the clients with an additional sequential sampling feature.
        The proportion per round is determined by self.proportion.
        """
        all_clients = self.available_clients if 'available' in self.sample_option else [cid for cid in
                                                                                        range(self.num_clients)]
        clients_per_round = max(min(int(self.num_clients * self.proportion), len(all_clients)), 1)

        # full sampling with unlimited communication resources of the server
        if 'full' in self.sample_option:
            return all_clients

        # sequential sampling
        elif 'sequential' in self.sample_option:
            start_index = (self.current_round * clients_per_round) % len(all_clients)
            end_index = min(start_index + clients_per_round, len(all_clients))
            selected_clients = all_clients[start_index:end_index]
            # If the end index wraps around, extend the list by the start of the list
            if end_index < start_index + clients_per_round:
                selected_clients.extend(all_clients[:start_index + clients_per_round - len(all_clients)])

        # uniform sampling without replacement
        elif 'uniform' in self.sample_option:
            selected_clients = list(np.random.choice(all_clients, clients_per_round, replace=False)) if len(
                all_clients) > 0 else []

        # MDSample with replacement
        elif 'md' in self.sample_option:
            local_data_vols = [self.clients[cid].datavol for cid in all_clients]
            total_data_vol = sum(local_data_vols)
            p = np.array(local_data_vols) / total_data_vol
            selected_clients = list(np.random.choice(all_clients, clients_per_round, replace=True, p=p)) if len(
                all_clients) > 0 else []

        return selected_clients

    def global_test(self, model=None, flag: str = 'val'):
        tmp = decrypt_model(self.encrypted_tensors, self.gv.private_key)
        model = _model_from_tensor_with_grad(tmp.to('cuda'), self.model)
        all_metrics = collections.defaultdict(list)
        for c in self.clients:
            client_metrics = c.test(model, flag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        print("结束服务器端global_test训练时间----------------------------------------------------------")
        return all_metrics

    def test(self, model=None, flag: str = 'test'):
        data = self.test_data if flag == 'test' else self.val_data
        if data is None:
            return {}
        tmp = decrypt_model(self.encrypted_tensors, self.gv.private_key)
        model = _model_from_tensor_with_grad(tmp.to('cuda'), self.model)
        print("结束服务器端test训练时间----------------------------------------------------------")

        return self.calculator.test(model=model, dataset=data,
                                    batch_size=min(self.option['test_batch_size'], len(data)),
                                    num_workers=self.option['num_workers'], pin_memory=self.option['pin_memory'])


class Client(fedbase.BasicClient):

    def pack(self, disturbed_model, *args, **kwargs):
        tensors = _model_to_tensor(disturbed_model)
        encrypted_gradient = encrypt_vector(self.gv.public_key, tensors)
        print('客户端pack完毕！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！')
        print('---------------------------------------------------------------------------')
        return {'encrypted_model': encrypted_gradient}

    def unpack(self, received_pkg):
        encrypted_tensors = received_pkg['encrypted_tensors']
        model = received_pkg['model']
        decrypt_tensors = decrypt_model(encrypted_tensors, self.gv.private_key)
        dec_model = _model_from_tensor_with_grad(decrypt_tensors.to('cuda'), model)
        sensitivity = received_pkg['sensitivity']
        print('客户端所接收到的全局敏感度', sensitivity)
        print('裁剪所需', self.clip_grad)
        local_sensitivity = cal_sensitivity(sensitivity, self.clip_grad)
        print('接收到全局敏感度后计算的局部敏感度', local_sensitivity)
        print('客户端unpack完毕！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！')
        print('客户端解密得到的全局模型张量', decrypt_tensors)
        print('---------------------------------------------------------------------------')
        return dec_model, sensitivity

    def reply(self, svr_pkg):
        model, sensitivity = self.unpack(svr_pkg)
        disturbed = self.train(model, sensitivity)
        cpkg = self.pack(disturbed)
        return cpkg

    @fmodule.with_multi_gpus
    def train(self, global_model, sensitivity):
        global_model.train()
        optimizer = self.calculator.get_optimizer(global_model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for epoch in range(self.num_steps):
            global_model.zero_grad()  # 初始化梯度为0
            batch_data = self.get_batch_data()
            loss = self.calculator.compute_loss(global_model, batch_data)['loss']
            loss.backward()
            # 裁剪梯度
            if self.clip_grad > 0:
                # torch.nn.utils.clip_grad_norm_(parameters=global_model.parameters(), max_norm=self.clip_grad)
                custom_clip_grad_norm_(global_model.parameters(), self.clip_grad)
            optimizer.step()

        print('客户端在本地训练进行加噪前、裁剪后的模型参数转张量', _model_to_tensor(global_model))
        print('敏感度······························································', sensitivity)
        print('Laplace 参数························································', sensitivity / self.epsilon)
        # Add Laplace noise for differential privacy
        for param in global_model.parameters():
            new_param_data = add_noise(param.data, sensitivity / self.epsilon, 1, device='cuda')
            param.data = new_param_data

        disturbed = global_model.to(torch.device('cuda'))
        print('客户端在本地训练进行加噪后的模型', _model_to_tensor(disturbed))
        return disturbed

    def test(self, global_model, flag='val'):
        data = self.train_data if flag == 'train' else self.val_data
        if data is None: return {}
        print("现在是客户端test时间----------------------------------------------------------")
        return self.calculator.test(global_model, data, min(self.test_batch_size, len(data)),
                                    num_workers=self.option['num_workers'], pin_memory=self.option['pin_memory'])


class PPFL_HEDP:
    Server = Server
    Client = Client
