import argparse
import scipy
import os
import sys
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image

object = r"/data2/home/Experiments/Lesion_2024"
current_version ='v20241001'
obj_name = os.path.abspath(f'{object}/{current_version}') # add current object name to sys
sys.path.append(obj_name)



from get_models import get_models
from collections import OrderedDict

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])  


def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

def sliding_predict(model, image, num_classes, flip=True):
    image_size = image.shape
    tile_size = (int(image_size[2]//2.5), int(image_size[3]//2.5))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = image[:, :, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction = model(padded_img)
            if flip:
                fliped_img = padded_img.flip(-1)
                fliped_predictions = model(padded_img.flip(-1))
                padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)
            predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy().squeeze(0)

    total_predictions /= count_predictions
    return total_predictions


def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img).cpu())

        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions


def save_images(image, mask, output_path, image_file, palette=None):
	# Saves the image, the model output and the results after the post processing
    # if args.dataset == "GOALS":
    #     save_name = image_file.split('GOALS2022')[-1]
    #     save_name = output_path + save_name        
    # else:
    #     save_name = image_file.replace(args.image_path, output_path)
    #     save_name = save_name.replace(args.extension,'png')
    save_name = os.path.join(output_path,image_file)    
    (Path(save_name).parent).mkdir(parents=True,exist_ok=True)    
    
    
    mask =  Image.fromarray(mask.astype(np.uint8)).convert('P')    
    mask.save(save_name)
    # mask.save(os.path.join(output_path, image_file+'.png'))
    

def main(args):
    # args = parse_arguments()
    config = json.load(open(args.config))

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(0.5, 0.5)
    
    # Model
    models = get_models(config['arch']['type'])
    
    model = getattr(models, config['arch']['type'])(**config['arch']['args'])
   
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
        # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    val_loader = get_instance(dataloaders, 'val_loader', config)
    image_files = val_loader.dataset.files['image']
    
    # image_files = sorted(glob(os.path.join(args.image_path, f'*.{args.extension}')))
    
    with torch.no_grad():
        tbar = tqdm(image_files, ncols=100)
        for img_file in tbar:
            image = Image.open(img_file).convert('RGB')
            image=image.resize((512,512),Image.BILINEAR)
            input = normalize(to_tensor(image)).unsqueeze(0)
            
            prediction = model(input.to(device))
            prediction = prediction.squeeze(0).cpu().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()           
            
            
            
            mask =prediction*255//(args.num_classes-1)
            
            output_path = r'/data2/home/Experiments/saved/Lesion_seg/images/'
            name = cfg
            save_images(image, mask, output_path,name)
            
def get_cfgs(ckpt_dir="/data2/home/Experiments/saved/HW_Net/ckpts/UNets/NIDEK/"):
 
    ckpt_list = sorted(list(Path(ckpt_dir).glob("*.pth")))
    
    def get_cfg(ckpt_dir):
        cfg_dir =ckpt_dir.parent / 'config.json'
        return cfg_dir
    
    cfg_list = [get_cfg(ckpt) for ckpt in ckpt_list]
    
    return ckpt_list, cfg_list

def load_args(ckpt,cfg):
    
    # model and save_path    
    image_save = str(ckpt).replace('ckpts','images')
    image_save = image_save.replace('.pth','')  
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default=cfg,type=str,
                        help='The config used to train the model')    
    parser.add_argument('-m', '--model', default=str(ckpt), type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')    
    parser.add_argument('-o', '--output', default=str(image_save), type=str,  
                        help='Output Path')
    
    #  load dateset involved
    config = json.load(open(cfg))
    dataset = config['train_loader']['type']
    parser.add_argument('-d', '--dataset', default=dataset, type=str,  
                        help='Output Path')
    data_dirs = {
            'AROD':r"/data2/home/Data/AROD",
            'NIDEK':r"/data2/home/Data/Nidek",
            'GOALS':r"/data2/home/Data/GOALS2022"
        }    
    extensions={
            'AROD':'png',
            'NIDEK':'bmp',
            'GOALS':'png'
        }
    num_classes={
        'AROD':12,
        "GOALS":4,
        "NIDEK":8
    }    
    # parser.add_argument('-i', '--image_path', default=data_dirs[dataset], type=str,
    #                     help='Path to the images to be segmented')
    parser.add_argument('-e', '--extension', default=extensions[dataset], type=str,
                        help='The extension of the images to be segmented')
    parser.add_argument('-n', '--num_classes', default=num_classes[dataset], type=str,
                        help='The numbeer of output classes')
    args = parser.parse_args() 
    print(args)   
    return args


    
    

if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'
    from pathlib import Path
    from datasets.BOE_dataset import BOE as dataloaders

    root_files=r"/data2/home/Experiments/saved/Lesion_seg/ckpts/BOE/Ablations_wo_pts/Group2/11-06_16-36/"

    ckpt_list, cfg_list= get_cfgs(root_files)
    for i in range(len(ckpt_list)):
        ckpt,cfg = ckpt_list[i], cfg_list[i]
        args = load_args(ckpt,cfg)

        main(args=args)
