import flgo.algorithm.fedavg as fedavg
import flgo.experiment.analyzer

# 创建svhn_classification数据集的横向联邦benchmark
bmk = flgo.gen_benchmark_from_file(
    benchmark='test_chestmnist',
    config_file='test_chestmnist/config.py',
    target_path='test_medical',
    data_type='cv',
    task_type='classification',
)

# 从benchmark构造IID划分的联邦任务
task = './chestmnist'  # 任务名称
task_config = {
    'benchmark': bmk,
    'partitioner': {
        'name': 'IIDPartitioner'
    }
}  # 任务配置
flgo.gen_task(task_config, task)  # 生成任务
runner = flgo.init(task, fedavg)  # 初始化fedavg运行器
runner.run()

analysis_plan = {
    'Selector': {
        'task': task,
        'header': ['fedavg']
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
