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

# test
root_files应该包含ckpt和cfg.json文件。当包含多个子文件夹时，可以调整176行的glob条件。
在load_args函数初始化数据集和其它相关的信息。主函数可以自动从cfg中读取模型路径和数据集超参，将输出的图片结果保存到output_path中。本函数的示例用于分割任务，可以修改168行的代码调整输出的形式，也可在此处接metric evaluation,保存定量结果。

The root_files directory should include both ckpt and cfg.json files. If it contains multiple subfolders, you can adjust the glob condition on line 176 accordingly.
The load_args function initializes information related to the dataset and so on. The main function can automatically read the model path and dataset hyperparameters from the configuration (cfg) and save the output image results to output_path.
This function example is designed for segmentation tasks. You can modify the code on line 168 to adjust the output format or add metric evaluation at this point to save quantitative results.
