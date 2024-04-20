import flgo.algorithm.fedavg as fedavg
import flgo.experiment.analyzer
import HEDP

# 创建svhn_classification数据集的横向联邦benchmark
bmk = flgo.gen_benchmark_from_file(
    benchmark='svhn_classification',
    config_file='./config.py',
    target_path='.',
    data_type='cv',
    task_type='classification',
)

# 从benchmark构造IID划分的联邦任务
task = './svhn_HEDP'  # 任务名称
task_config = {
    'benchmark': bmk,
    'partitioner': {
        'name': 'IIDPartitioner'
    }
}  # 任务配置
flgo.gen_task(task_config, task)  # 生成任务
option = {'gpu': [0, 1], 'num_rounds': 15, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
          'save_checkpoint': 0, 'load_checkpoint': 0, 'log_file': True, 'log_level': 'DEBUG'}
option_1 = {'gpu': [0, 1], 'num_rounds': 15, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 1, 'load_checkpoint': 1, 'algo_para': [10], 'clip_grad': 50, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG', 'pin_memory': True, 'num_workers': 8}
option_2 = {'gpu': [0, 1], 'num_rounds': 15, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 2, 'load_checkpoint': 2, 'algo_para': [20], 'clip_grad': 50, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG', 'pin_memory': True, 'num_workers': 8}
option_3 = {'gpu': [0, 1], 'num_rounds': 15, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 3, 'load_checkpoint': 3, 'algo_para': [30], 'clip_grad': 50, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG', 'pin_memory': True, 'num_workers': 8}
option_4 = {'gpu': [0, 1], 'num_rounds': 15, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 4, 'load_checkpoint': 4, 'algo_para': [40], 'clip_grad': 50, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG', 'pin_memory': True, 'num_workers': 8}
option_5 = {'gpu': [0, 1], 'num_rounds': 15, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 5, 'load_checkpoint': 5, 'algo_para': [50], 'clip_grad': 50, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG', 'pin_memory': True, 'num_workers': 8}
option_6 = {'gpu': [0, 1], 'num_rounds': 15, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 6, 'load_checkpoint': 6, 'algo_para': [75], 'clip_grad': 50, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG', 'pin_memory': True, 'num_workers': 8}
option_7 = {'gpu': [0, 1], 'num_rounds': 15, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 7, 'load_checkpoint': 7, 'algo_para': [100], 'clip_grad': 50, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG', 'pin_memory': True, 'num_workers': 8}
option_8 = {'gpu': [0, 1], 'num_rounds': 15, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 8, 'load_checkpoint': 8, 'algo_para': [10], 'clip_grad': 30, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG', 'pin_memory': True, 'num_workers': 8}
option_9 = {'gpu': [0, 1], 'num_rounds': 15, 'proportion': 0.1, 'batch_size': 64, 'learning_rate': 0.1,
            'save_checkpoint': 9, 'load_checkpoint': 9, 'algo_para': [10], 'clip_grad': 20, 'sample': 'sequential',
            'log_file': True, 'log_level': 'DEBUG', 'pin_memory': True, 'num_workers': 8}

fedavg_runner = flgo.init(task, fedavg, option=option)
MY_runner_DP_1 = flgo.init(task, HEDP.PPFL_HEDP, option=option_1)
MY_runner_DP_2 = flgo.init(task, HEDP.PPFL_HEDP, option=option_2)
MY_runner_DP_3 = flgo.init(task, HEDP.PPFL_HEDP, option=option_3)
MY_runner_DP_4 = flgo.init(task, HEDP.PPFL_HEDP, option=option_4)
MY_runner_DP_5 = flgo.init(task, HEDP.PPFL_HEDP, option=option_5)
MY_runner_DP_6 = flgo.init(task, HEDP.PPFL_HEDP, option=option_6)
MY_runner_DP_7 = flgo.init(task, HEDP.PPFL_HEDP, option=option_7)
MY_runner_DP_8 = flgo.init(task, HEDP.PPFL_HEDP, option=option_8)
MY_runner_DP_9 = flgo.init(task, HEDP.PPFL_HEDP, option=option_9)

fedavg_runner.run()
MY_runner_DP_1.run()
MY_runner_DP_2.run()
MY_runner_DP_3.run()
MY_runner_DP_4.run()
MY_runner_DP_5.run()
MY_runner_DP_6.run()
MY_runner_DP_7.run()
MY_runner_DP_8.run()
MY_runner_DP_9.run()

analysis_plan = {
    'Selector': {
        'task': task,
        'header': ['PPFL_HEDP']
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
    },
    'Table': {'min_value': [{'x': 'val_loss'}]},
}
flgo.experiment.analyzer.show(analysis_plan)
