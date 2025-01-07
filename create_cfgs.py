import json
import os
import  sys
from typing import Dict, Any
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
items = current_dir.split('/')
for item in items:
    if 'v2024' in item or 'V2024' in item:
        obj_name = item
# 修改这行，使用项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '../../'))  # 回退到v20240614目录
sys.path.append(project_root)  # 添加项目根目录到Python路径


# 获取模型类型映射
MODEL_TYPES = {
    # Transformer-based models that need img_size
    "TRANSFORMER_MODELS": [
        "TransUNet", "Res34_Swin_MS", "CPCANet", 
        "UNext_S", "FastSCNN"
    ],
    # Models with special parameters
    "SPECIAL_PARAMS": {
        "BioNet": ["gms_channels"],
    }
}

def create_config(
    arch_name: str,
    exp_group: str = "BOE/SOTAs",
    in_channels: int = 1,
    num_classes: int = 9,
    dataset_type: str = "BOE",
    task: str = "layerseg",
    cv_group: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """创建配置文件
    Args:
        arch_name: 模型架构名称
        exp_group: 实验组别名称，用于组织实验结果，如 "BOE/layerseg/SOTAS"
        cv_group: 交叉验证分组编号，用于指定使用哪个子集进行训练和验证
        ...
    """
    
    # 数据集路径映射
    DATASET_PATHS = {
        "BOE": "/disk1/Data/BOE_trans_data/",
        "AROD": "/disk1/Data/imed_Topcon_Normal_AROD/signal_crop_512/",
        "RETOUCH":"/disk1//Data/RETOUCH/",
        "NIDEK":"/disk1/Data/Nidek/",
        "GOALS": "/disk1/Data/GOALS/"       
    }
    
    # 基础配置
    base_size = kwargs.get("base_size", 512)
    
    # 特殊模型的参数配置
    model_args = {
        "in_channels": in_channels,
        "num_classes": num_classes
    }
    
   
        
    # 添加其他特殊参数
    if arch_name in MODEL_TYPES["SPECIAL_PARAMS"]:
        for param in MODEL_TYPES["SPECIAL_PARAMS"][arch_name]:
            model_args[param] = kwargs.get(param, 32)  # 使用默认值32
    
    config = {
        "name": exp_group,
        "n_gpu": kwargs.get("n_gpu", 4),
        "arch": {
            "type": arch_name,
            "args": model_args
        },
        "train_loader": {
            "type": dataset_type,
            "args": {
                "data_dir": DATASET_PATHS[dataset_type],
                "batch_size": kwargs.get("batch_size", 4),
                "base_size": base_size,
                "crop_size": kwargs.get("crop_size", None),
                "augment": kwargs.get("augment", True),
                "shuffle": kwargs.get("shuffle", True),
                "scale": kwargs.get("scale", False),
                "flip": kwargs.get("flip", True),
                "rotate": kwargs.get("rotate", False),
                "blur": kwargs.get("blur", False),
                "split": "train",
                "num_workers": kwargs.get("num_workers", 2),
                "task": task,
                "group": cv_group
            }
        },
        "val_loader": {
            "type": dataset_type,
            "args": {
                "data_dir": DATASET_PATHS[dataset_type],
                "batch_size": 1,
                "base_size": base_size,
                "val": True,
                "split": "val",
                "num_workers": kwargs.get("num_workers", 2),
                "task": task,
                "group": cv_group
            }
        },
        "optimizer": {
            "type": kwargs.get("optimizer_type", "SGD"),
            "differential_lr": False,
            "args": {
                "lr": kwargs.get("lr", 0.01),
                "weight_decay": kwargs.get("weight_decay", 1e-4),
                "momentum": kwargs.get("momentum", 0.9)
            }
        },
        "loss": kwargs.get("loss", "CrossEntropyLoss2d"),
        "ignore_index": 255,
        "lr_scheduler": {
            "type": kwargs.get("scheduler_type", "Poly"),
            "args": {}
        },
        "trainer": {
            "epochs": kwargs.get("epochs", 250), #400
            "save_dir": "saved/checkpoints",
            "save_period": kwargs.get("save_period", 40), #50
            "monitor": "max Mean_IoU",
            "early_stop": kwargs.get("early_stop", 40),
            "tensorboard": True,
            "log_dir": "saved/runs",
            "log_per_iter": kwargs.get("log_per_iter", 5), #5
            "val": True,
            "val_per_epochs": kwargs.get("val_per_epochs", 5) #5
        }
    }
    
     # 为Transformer模型添加img_size参数, batch_size减半
    if arch_name in MODEL_TYPES["TRANSFORMER_MODELS"]:
        model_args["img_size"] = base_size
        config["train_loader"]["args"]["batch_size"] = kwargs.get("batch_size", 2)
        # config["val_loader"]["args"]["batch_size"] = kwargs.get("batch_size", 2)
        
    
    return config

def save_config(config: Dict[str, Any], exp_group: str, arch_name: str, index: int):
    """
    保存配置文件
    
    Args:
        config: 配置字典
        exp_group: 实验组别名称
        arch_name: 模型名称
        index: 配置文件索引
    """
    # 将exp_group中的斜杠替换为下划线
    folder_name = exp_group.replace('/', '_')
    
    # 构建保存路径
    base_path = Path(project_root)/f"scripts/cfgs/{folder_name}"
    # save_dir = base_path / folder_name
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名
    file_name = f"cfg_{index}_{arch_name.lower()}.json"
    save_path = base_path / file_name
    
    # 保存配置文件
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Saved config to: {save_path}")

def generate_all_configs():
    """生成所有模型的配置文件"""
    # 从get_models.py导入模型列表
    architectures = [
        # basic models
        "UNet", "UNetResnet", "AttU_Net", "SegNet",
        
        # semantic segmentation models
        "FastSCNN",
        
        # medical image segmentation models
        "CE_Net_OCT", "UTNet_Encoderonly", "UNeXt", #"MTUNet"
        "MWGNet", "Res34_Swin_MS", "CPCANet",
        "EMCAD_Net", "TransUNet",
        
        # layers segmentation models 
        "MGUNet", "M2SNet", "WATNet", "ISLAM", #"LightReSeg"
        
        # lesion segmentation models
        "ReLayNet", "EdgeAL", "YNet_general",
        
        # wavelet attention models
        "FCA_SegNet", "DWAN_SegNet"
    ]
    
    task = "layerseg"
    # exp_group = f'BOE/{task}/SOTAS_G1'
    exp_group = f'BOE/{task}/SOTAS_G5'
    index = 1
    for arch in architectures:
        config = create_config(
            arch_name=arch,
            exp_group=exp_group,
            dataset_type="BOE",
            task=task,
            cv_group=0
        )
        save_config(config, exp_group, arch, index)
        index += 1

if __name__ == "__main__":
    generate_all_configs()