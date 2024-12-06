# Toolkit-for-batch-experiments
用于深度学习批量实验的工具箱


# get_models
When conducting comparative and ablation experiments, it is necessary to call a variety of different models. Modifying the train function with "import XX as models" is cumbersome and inconvenient for saving logs.
This function provides a simple way to call models, returning the corresponding model based on its name. You can use the following statement in the training code to call it:
进行对比实验和消融实验时，需要调用大量不同模型。在train函数修改“import  XX as models “比较麻烦，且不方便保存log。本函数提供了一个简单的调用函数，根据模型名称返回对应模型。在训练代码中通过以下语句调用：

def get_instance(module, name, config, *args):   
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])  

def main(config, resume):  # from config to paras and functions
    # MODEL
    models = get_models(config['arch']['type'])  # import models by arch name
    model = get_instance(models, 'arch', config) # feed in setting
