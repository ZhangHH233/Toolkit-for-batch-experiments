
import os
import sys


object = r"/data2/home/Experiments/Lesion_2024"
current_version ='v20241001'
obj_name = os.path.abspath(f'{object}/{current_version}') # add current object name to sys
sys.path.append(obj_name)


def get_models(arch="CE_Net_OCT"):   
    
    if arch == "UNet":
        import models.backbones.unet as models
        return models      
    
    elif arch in ["U_Net_SA","U_Net_SK","HW_Net"]:
        import models.ablations.DWT_sa_sk_UNet as models
        return models
    
    elif arch in ["U_Net_GA_SA","U_Net_GA_SK"]:
        import models.ablations.DWT_GA as models
        return models
        
    else:
        raise ValueError('Check your arch type!')
    
if __name__=='__main__': 
    
    for arch in ["UNet","U_Net_SA","U_Net_SK","HW_Net"]:
        models = get_models(arch)
        model = getattr(models, arch)
        print(model)