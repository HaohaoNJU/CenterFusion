from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import pickle as pkl
import cv2
import numpy as np
import time

from progress.bar import Bar
import torch
import copy
import pdb

from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from dataset.dataset_factory import dataset_factory
from detector import Detector


ROOT_DIR = "../../../data"
SAVE_PATH = "../../../data/predata"
os.makedirs(SAVE_PATH+"/images",exist_ok=True)
os.makedirs(SAVE_PATH+"/calibs",exist_ok=True)
os.makedirs(SAVE_PATH+"/pc_deps",exist_ok=True)
os.makedirs(SAVE_PATH+"/pc_3ds",exist_ok=True)

class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.get_default_calib = dataset.get_default_calib
        self.opt = opt
        self.dataset = dataset
    
    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        images, meta = {}, {}
        for scale in opt.test_scales:
            input_meta = {}
            calib = img_info['calib'] if 'calib' in img_info \
                else self.get_default_calib(image.shape[1], image.shape[0])
            input_meta['calib'] = calib
            images[scale], meta[scale] = self.pre_process_func(
                image, scale, input_meta)
                
        meta['img_path'] = img_path
        ret = {'images': images, 'image': image, 'meta': meta}
        if 'frame_id' in img_info and img_info['frame_id'] == 1:
            ret['is_first_frame'] = 1
            ret['video_id'] = img_info['video_id']
        
        # add point cloud
        if opt.pointcloud:
            assert len(opt.test_scales)==1, "Multi-scale testing not supported with pointcloud."
            scale = opt.test_scales[0]
            pc_2d, pc_N, pc_dep, pc_3d = self.dataset._load_pc_data(image, img_info, 
                meta[scale]['trans_input'], meta[scale]['trans_output'])
            ret['pc_2d'] = pc_2d
            ret['pc_N'] = pc_N
            ret['pc_dep'] = pc_dep
            ret['pc_3d'] = pc_3d
        #     ret['img_info'] = img_info
        return img_id, ret

    def __len__(self):
        return len(self.images)


def export(opt):
    opt.data_dir =    ROOT_DIR
    opt.save_results = True
    if not opt.not_set_cuda_env:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    Dataset = dataset_factory[opt.test_dataset] # get nuscenes.py/nuScenes class
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print("opt \n",opt)
    Logger(opt)
    
#     split = 'val' if not opt.trainval else 'test'
#     if split == 'val':
    split = "mini_val"
    dataset = Dataset(opt, split)
#     dataset.img_dir = ROOT_DIR + "/nuScenes"
    print('dataset img dir ', dataset.img_dir)
    model = Detector(opt).model
    import pdb
    pdb.set_trace()
    


def prefetch_test(opt):
    opt.data_dir =    ROOT_DIR
    opt.save_results = True
    if not opt.not_set_cuda_env:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    Dataset = dataset_factory[opt.test_dataset] # get nuscenes.py/nuScenes class
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print("opt \n",opt)
    Logger(opt)
    
#     split = 'val' if not opt.trainval else 'test'
#     if split == 'val':
    split = opt.val_split
    dataset = Dataset(opt, split)
#     dataset.img_dir = ROOT_DIR + "/nuScenes"
    print('==============Dataset img dir ', dataset.img_dir,"============")
    detector = Detector(opt)

#     import pdb;pdb.set_trace()    
    if opt.load_results != '':
        load_results = json.load(open(opt.load_results, 'r'))
        for img_id in load_results:
            for k in range(len(load_results[img_id])):
                if load_results[img_id][k]['class'] - 1 in opt.ignore_loaded_cats:
                    load_results[img_id][k]['score'] = -1
    else:
        load_results = {}

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process), 
        batch_size=1, shuffle=False, num_workers=5, pin_memory=True)
    print("============Done creating dataloader============")
    results = {}
    pre_annos = {}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'track']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    if opt.use_loaded_results:
        for img_id in data_loader.dataset.images:
            results[img_id] = load_results['{}'.format(img_id)]
        num_iters = 0
    data_nums = []
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        # if ind >= 1: #num_iters:
        #     break
        if opt.tracking and ('is_first_frame' in pre_processed_images):
            if '{}'.format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
                pre_processed_images['meta']['pre_dets'] = \
                    load_results['{}'.format(int(img_id.numpy().astype(np.int32)[0]))]
            else:
                print('No pre_dets for', int(img_id.numpy().astype(np.int32)[0]), 
                    '. Use empty initialization.')
                pre_processed_images['meta']['pre_dets'] = []
            detector.reset_tracking()

            print('Start tracking video', int(pre_processed_images['video_id']))
        if opt.public_det:
            if '{}'.format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
                pre_processed_images['meta']['cur_dets'] = \
                    load_results['{}'.format(int(img_id.numpy().astype(np.int32)[0]))]
            else:
                print('No cur_dets for', int(img_id.numpy().astype(np.int32)[0]))
                pre_processed_images['meta']['cur_dets'] = []
        
                """
                imgs = data['images'][1.0].numpy() :    1 x 2 x 3 x H_in x W_in, two input image, uint8 dtype
                img = data['image'].numpy().squeeze()    : torch.Size([1, 900, 1600, 3]), one input image , fp32 dtype
        
                meta = data['meta'] :
                {1.0: {'calib': tensor([[[1.2528e+03, 0.0000e+00, 8.2659e+02, 0.0000e+00],
                             [0.0000e+00, 1.2528e+03, 4.6998e+02, 0.0000e+00],
                             [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00]]]),
            'c': tensor([[800., 450.]]),
            's': tensor([1600.], dtype=torch.float64),
            'height': tensor([900]),
            'width': tensor([1600]),
            'out_height': tensor([112]),
            'out_width': tensor([200]),
            'inp_height': tensor([448]),
            'inp_width': tensor([800]),
            'trans_input': tensor([[[ 0.5000, -0.0000,    0.0000],
                             [ 0.0000,    0.5000, -1.0000]]], dtype=torch.float64),
            'trans_output': tensor([[[ 0.1250, -0.0000,    0.0000],
                             [ 0.0000,    0.1250, -0.2500]]], dtype=torch.float64)}}
        
                pc_2d = data['pc_2d']    : 1 x 3 x 1000, point clout dense tensor
                pc_N = data['pc_N']    : 53, valid point num
                pc_dep = data['pc_dep'] : torch.Size([1, 3, 112, 200]),    radar points    in    sparse image shape
                pc_3d = data['pc_3d'] : 1 x 18 x 1000, point cloud dense tensor , why 18 channels ? 
                """
        pre_anno = {k:v for k,v in pre_processed_images.items() if not k in ['image','images']}
        for k,v in pre_anno.items():
                        if isinstance(v,torch.Tensor):
                                v = v.cpu().numpy()
                        pre_anno[k] = v
                        
# ####################################################################################################
        img,pc_dep, calib = detector.get_data(pre_processed_images)


        pc_3d = pre_processed_images['pc_3d']
        pc_vx = pc_3d[0,8:9]
        pc_vz = pc_3d[0,9:10]
        pc_3d = torch.cat([pc_3d[0,:3], pc_vx, pc_vz],dim=0)
        pc_3d.cpu().numpy().astype(np.float32).tofile(SAVE_PATH+"/pc_3ds/"+"data{}.bin".format(ind))
        
        data_nums.append(pre_processed_images['pc_N'][0].item())
        if ind == len(data_loader)-1:
            data_nums = np.array(data_nums)
            data_nums.astype(np.float32).tofile(SAVE_PATH+"/data_num.bin".format(ind))
            
        img.cpu().numpy().astype(np.float32).tofile(SAVE_PATH+"/images/"+"data{}.bin".format(ind))
        pc_dep.cpu().numpy().astype(np.float32).tofile(SAVE_PATH+"/pc_deps/"+"data{}.bin".format(ind))
        calib.cpu().numpy().astype(np.float32).tofile(SAVE_PATH+"/calibs/"+"data{}.bin".format(ind))


def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
#     import pdb;pdb.set_trace()
    Dataset = dataset_factory[opt.test_dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    
    split = 'val' if not opt.trainval else 'test'
    if split == 'val':
        split = opt.val_split
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    if opt.load_results != '': # load results in json
        load_results = json.load(open(opt.load_results, 'r'))

    results = {}
    num_iters = len(dataset) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind in range(num_iters):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])
        input_meta = {}
        if 'calib' in img_info:
            input_meta['calib'] = img_info['calib']
        if (opt.tracking and ('frame_id' in img_info) and img_info['frame_id'] == 1):
            detector.reset_tracking()
            input_meta['pre_dets'] = load_results[img_id]


        ret = detector.run(img_path, input_meta)        
        results[img_id] = ret['results']

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                                     ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        bar.next()
    bar.finish()
    if opt.save_results:
        print('saving results to', opt.save_dir + '/save_results_{}{}.json'.format(
            opt.test_dataset, opt.dataset_version))
        json.dump(_to_list(copy.deepcopy(results)), 
                            open(opt.save_dir + '/save_results_{}{}.json'.format(
                                opt.test_dataset, opt.dataset_version), 'w'))
    dataset.run_eval(results, opt.save_dir, n_plots=opt.eval_n_plots, 
                                     render_curves=opt.eval_render_curves)
                                     
def eval(opt):
    opt.data_dir =    ROOT_DIR
    opt.save_results = True
    if not opt.not_set_cuda_env:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    Dataset = dataset_factory[opt.test_dataset] # get nuscenes.py/nuScenes class
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print("opt \n",opt)
    Logger(opt)
    
#     split = 'val' if not opt.trainval else 'test'
#     if split == 'val':
    split = opt.val_split
    dataset = Dataset(opt, split)
    results =     json.load(
                            open(opt.save_dir + '/save_results_{}{}.json'.format(
                                opt.test_dataset, opt.dataset_version), 'r'))

    dataset.run_eval(results, opt.save_dir, n_plots=opt.eval_n_plots, 
                                     render_curves=opt.eval_render_curves)



def _to_list(results):
    for img_id in results:
        for t in range(len(results[img_id])):
            for k in results[img_id][t]:
                if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                    results[img_id][t][k] = results[img_id][t][k].tolist()
    return results

if __name__ == '__main__':
    opt = opts().parse()
    if    opt.export_onnx:
        export(opt)
    elif opt.not_prefetch_test:
        test(opt)
    else:
        # run from here ! 
        prefetch_test(opt) 
        # eval(opt)






