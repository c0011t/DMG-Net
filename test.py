import joblib,copy
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch,sys
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
from collections import OrderedDict
from lib.visualize import save_img,group_images,concat_result,save_heatmap,save_colour
import os
import argparse
from lib.logger import Logger, Print_Logger
from lib.extract_patches import *
from os.path import join
from lib.dataset import TestDataset
from lib.metrics import Evaluate
import models
from lib.common import setpu_seed,dict_round
from config import parse_args
from lib.pre_processing import my_PreProc
import time 

setpu_seed(2021)

class Test():
    def __init__(self, args):
        self.args = args
        assert (args.stride_height <= args.test_patch_height and args.stride_width <= args.test_patch_width)
        # save path
        self.path_experiment = join(args.outf, args.save)

        self.patches_imgs_test, self.test_imgs, self.test_masks, self.test_FOVs, self.new_height, self.new_width = get_data_test_overlap(
            test_data_path_list=args.test_data_path_list,
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=3)
        #nn.softmax2d()
    # Inference prediction process

    # def inference(self, net):
    #     begin_time=time.time()
    #     net.eval()
    #     preds = []
    #     with torch.no_grad():
    #         for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
    #             inputs = inputs.cuda()
    #             outputs = net(inputs)
    #             outputs = outputs[:,1].data.cpu().numpy()
    #             preds.append(outputs)
    #     predictions = np.concatenate(preds, axis=0)
    #     self.pred_patches = np.expand_dims(predictions,axis=1)
    #     end_time = time.time()
    #     pred_time = end_time - begin_time
    #     print('The inference time: {time: .4f}'.format(time=pred_time))

    def inference(self, net):
        begin_time=time.time()
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                # inputs -> GPU
                inputs = inputs.cuda()
                outputs = net(inputs)  # logits or 2-channel logits

                # Case A: outputs shape [B, 2, H, W] (softmax style logits for 2 classes)
                # Case B: outputs shape [B, 1, H, W] (single-channel logits for vessel prob)
                if outputs.dim() == 4 and outputs.size(1) == 2:
                    # 两通道：把 softmax -> 取通道1
                    probs = torch.softmax(outputs, dim=1)[:, 1, :, :].unsqueeze(1)
                elif outputs.dim() == 4 and outputs.size(1) == 1:
                    # 单通道 logits：sigmoid -> 概率
                    probs = torch.sigmoid(outputs)
                else:
                    print("Warning: unexpected output shape:", outputs.shape)
                    probs = outputs
                    if probs.dim() == 4 and probs.size(1) != 1:
                        try:
                            probs = probs.squeeze(1)
                            probs = probs.unsqueeze(1)
                        except:
                            raise RuntimeError("Cannot interpret model output shape: {}".format(outputs.shape))

                # 把 probs 转到 cpu numpy 并 append
                preds.append(probs.data.cpu().numpy())

        predictions = np.concatenate(preds, axis=0)  # [N,1,H,W] or similar
        self.pred_patches = np.expand_dims(predictions[:,0,:,:], axis=1) if predictions.ndim==4 else np.expand_dims(predictions,axis=1)
        end_time = time.time()
        pred_time = end_time - begin_time
        print('The inference time: {time: .4f}'.format(time=pred_time))
        
        
    # Evaluate ate and visualize the predicted images
    def evaluate(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## restore to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        log = eval.save_all_result(plot_curve=True,save_name="performance.txt")
        # save labels and probs for plot ROC and PR curve when k-fold Cross-validation
        np.save('{}/result.npy'.format(self.path_experiment), np.asarray([y_true, y_scores]))
        return dict_round(log, 6)

    # save segmentation imgs
    def save_segmentation_result(self):
        img_path_list, _, _ = load_file_path_txt(self.args.test_data_path_list)
        img_name_list = [item.split('/')[-1].split('.')[0] for item in img_path_list]

        kill_border(self.pred_imgs, self.test_FOVs) # only for visualization
        self.save_img_path = join(self.path_experiment,'result_img')
        self.save_color_path =  join(self.path_experiment,'result_img','coluor_seg')
        if not os.path.exists(join(self.save_img_path)):
            os.makedirs(self.save_img_path)
        if not os.path.exists(join(self.save_color_path)):
            os.makedirs(self.save_color_path)
        # self.test_imgs = my_PreProc(self.test_imgs) # Uncomment to save the pre processed image
        for i in range(self.test_imgs.shape[0]):
            total_img = concat_result(self.test_imgs[i],self.pred_imgs[i],self.test_masks[i])
            save_img(total_img,join(self.save_img_path, "Result_"+img_name_list[i]+'.png'))
            save_heatmap(self.pred_imgs[i],join(self.save_img_path, "heatmap_"+img_name_list[i]+'.png'))
            save_colour(self.pred_imgs[i],self.test_masks[i],join(self.save_color_path, "color_"+img_name_list[i]+'.png'))
    # Val on the test set at each epoch
    def val(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## recover to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        confusion,accuracy,specificity,sensitivity,precision = eval.confusion_matrix()
        log = OrderedDict([('val_auc_roc', eval.auc_roc()),
                           ('val_f1', eval.f1_score()),
                           ('val_acc', accuracy),
                           ('SE', sensitivity),
                           ('SP', specificity)])
        return dict_round(log, 6)

if __name__ == '__main__':
    args = parse_args()
    save_path = join(args.outf, args.save)
    sys.stdout = Print_Logger(os.path.join(save_path, 'test_log.txt'))
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    #net = models.UNetFamily.Dense_Unet(1,2).to(device)
    #net = models.LadderNet(inplanes=1, num_classes=2, layers=3, filters=16).to(device)
    #net = models.AmeUnet_v4(in_channel=1,out_ch=2).to(device)
    #net = models.R2U_Net(img_ch=1,output_ch=2).to(device)
    #net = models.AttU_Net(img_ch=1, output_ch=2).to(device)
    #net = models.Baseline(in_channel=1,out_ch=2).to(device)
    #net = models.DSNet.U_Net(img_ch=1, output_ch=2).to(device)    
    #net = models.UNetFamily.U_Net(1,2).to(device)
    #net=models.Uli_Deform_skip4(1,2).to(device)
    #net = models.FAT_Net(n_channels=1, n_classes=2)
    #net=models.Ghost_Net(1,2).to(device)
    #net = models.Ghost_T_TCGCN37(1,2).to(device)
    #net = models.ghost.Ghost_T_TCGCN37(1,2).to(device)
    #net = models.mghost.Ghost_T_TCGCN37_Mamba(1,1).to(device)
    #net = models.ThreeUnet_S_to_B(1,1).to(device)
    cudnn.benchmark = True

    # Load checkpoint
    print('==> Loading checkpoint...')
    checkpoint = torch.load(join(save_path, 'best_model.pth'))
    net.load_state_dict(checkpoint['net'])

    eval = Test(args)
    eval.inference(net)
    print(eval.evaluate())
    eval.save_segmentation_result()
