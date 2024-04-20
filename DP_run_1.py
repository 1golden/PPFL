import flgo.algorithm.fedavg as fedavg
import flgo.experiment.analyzer
import HEDP
import DP_Gaussian
import only_HE

# 创建svhn_classification数据集的横向联邦benchmark
bmk = flgo.gen_benchmark_from_file(
    benchmark='svhn_classification',
    config_file='./config.py',
    target_path='.',
    data_type='cv',
    task_type='classification',
)

# 从benchmark构造IID划分的联邦任务
task = './svhn_DP_Gaussian'  # 任务名称
task_config = {
    'benchmark': bmk,
    'partitioner': {
        'name': 'IIDPartitioner'
    }
}  # 任务配置
flgo.gen_task(task_config, task)  # 生成任务
option = {'gpu': 0, 'num_rounds': 20, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
          'save_checkpoint': 0, 'load_checkpoint': 0, 'log_file': True, 'log_level': 'DEBUG'}
option_1 = {'gpu': 0, 'num_rounds': 20, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 1, 'load_checkpoint': 1, 'algo_para': [1, 2], 'clip_grad': 5, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG'}
option_2 = {'gpu': 0, 'num_rounds': 20, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 2, 'load_checkpoint': 2, 'algo_para': [2, 2], 'clip_grad': 5, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG'}
option_3 = {'gpu': 0, 'num_rounds': 20, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 3, 'load_checkpoint': 3, 'algo_para': [3, 2], 'clip_grad': 5, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG'}
option_4 = {'gpu': 0, 'num_rounds': 20, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 4, 'load_checkpoint': 4, 'algo_para': [4, 2], 'clip_grad': 5, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG'}
option_5 = {'gpu': 0, 'num_rounds': 20, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 5, 'load_checkpoint': 5, 'algo_para': [5, 2], 'clip_grad': 5, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG'}
option_6 = {'gpu': 0, 'num_rounds': 20, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 6, 'load_checkpoint': 6, 'algo_para': [7, 2], 'clip_grad': 5, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG'}
option_7 = {'gpu': 0, 'num_rounds': 20, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 7, 'load_checkpoint': 7, 'algo_para': [10, 2], 'clip_grad': 5, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG'}

# fedavg_runner = flgo.init(task, fedavg, option=option)
MY_runner_DP_1 = flgo.init(task, DP_Gaussian.PPFL_DP, option=option_1)
MY_runner_DP_2 = flgo.init(task, DP_Gaussian.PPFL_DP, option=option_2)
MY_runner_DP_3 = flgo.init(task, DP_Gaussian.PPFL_DP, option=option_3)
MY_runner_DP_4 = flgo.init(task, DP_Gaussian.PPFL_DP, option=option_4)
MY_runner_DP_5 = flgo.init(task, DP_Gaussian.PPFL_DP, option=option_5)
MY_runner_DP_6 = flgo.init(task, DP_Gaussian.PPFL_DP, option=option_6)
MY_runner_DP_7 = flgo.init(task, DP_Gaussian.PPFL_DP, option=option_7)
# MY_runner_DP_8 = flgo.init(task, DP.PPFL_DP, option=option_8)

# fedavg_runner.run()
MY_runner_DP_1.run()
MY_runner_DP_2.run()
MY_runner_DP_3.run()
MY_runner_DP_4.run()
MY_runner_DP_5.run()
MY_runner_DP_6.run()
MY_runner_DP_7.run()
# MY_runner_DP_8.run()

analysis_plan = {
    'Selector': {
        'task': task,
        'header': ['PPFL_DP'],
    },
    'Painter': {
        'Curve': [
            {'args': {'x': 'communication_round', 'y': 'test_loss'},
             'fig_option': {'title': 'test loss on Synthetic'}},
            {'args': {'x': 'communication_round', 'y': 'test_accuracy'},
             'fig_option': {'title': 'test accuracy on Synthetic'}},
            {'args': {'x': 'communication_round', 'y': 'val_accuracy'},
             'fig_option': {'title': 'valid accuracy on Synthetic'}},
            {'args': {'x': 'communication_round', 'y': 'val_loss'},
             'fig_option': {'title': 'valid loss  on Synthetic'}},
        ]
    }
}
flgo.experiment.analyzer.show(analysis_plan)
