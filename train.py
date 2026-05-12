import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import sys,time
from os.path import join
import torch
from lib.losses.loss import *
from lib.common import *
from config import parse_args
from lib.logger import Logger, Print_Logger
import models
from test import Test
from torchstat import stat

from function import get_dataloader, train, val, get_dataloaderV2


def main():
    setpu_seed(2021)
    args = parse_args()
    save_path = join(args.outf, args.save)
    save_args(args,save_path)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    
    log = Logger(save_path)
    sys.stdout = Print_Logger(os.path.join(save_path,'train_log.txt'))
    print('The computing device used is: ','GPU' if device.type=='cuda' else 'CPU')
    
    #net = models.UNetFamily.U_Net(1,2).to(device)
    #net = models.LadderNet(inplanes=args.in_channels, num_classes=args.classes, layers=3, filters=16).to(device)
    #net = models.NU_Net(img_ch=1, output_ch=2).to(device)
    #net = models.multiSE.AttU_Net(img_ch=1, output_ch=2).to(device)
    #net = models.DSNet.U_Net(img_ch=1, output_ch=2).to(device)
    #net = models.FAT_Net(n_channels=1, n_classes=2)
    #net=models.Uli_Deform_skip4(1,2).to(device)
    #net=models.Ghost_Net(1,2).to(device)
    #net = models.Ghost_T_TCGCN37(1,2).to(device)
    #net = models.ghostO.Ghost_T_TCGCN37(1,2).to(device)
    #net = models.ghost.Ghost_T_TCGCN37(1,2).to(device)
    #net = models.mghost.Ghost_T_TCGCN37_Mamba(1,1).to(device)
    #net = models.FRD_Net().to(device)
    #net = models.Low_U_Net(1,1).to(device)
    #net = models.PMDC_Net_v2(1,1).to(device)
    #net = models.PMFF_Net(1,1).to(device)


    #stat(net,(1,64,64))
    print("Total number of parameters: " + str(count_parameters(net)))

    log.save_graph(net,torch.randn((1,1,64,64)).to(device).to(device=device))  # Save the model structure to the tensorboard file
    #log.save_graph(net,torch.randn((1,1,64,64)).to(device))
    # torch.nn.init.kaiming_normal(net, mode='fan_out')      # Modify default initialization method
    # net.apply(weight_init)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # The training speed of this task is fast, so pre training is not recommended
    if args.pre_trained is not None:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.outf + '%s/latest_model.pth' % args.pre_trained)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']+1

    # criterion = LossMulti(jaccard_weight=0,class_weights=np.array([0.5,0.5]))
    #criterion = CrossEntropyLoss2d() # Initialize loss function 原
    
    #criterion = BCEDiceLoss(bce_weight=0.7, dice_weight=0.3, pos_weight=2) #CHASEDB1
    criterion = BCEDiceLoss(bce_weight=0.7, dice_weight=0.3, pos_weight=1) #DRIVE
    #criterion = BCEDiceLoss(bce_weight=0.7, dice_weight=0.3, pos_weight=3) #STARE
    
    #criterion = BCEDiceLossWithFPPenalty(bce_weight=0.5,dice_weight=0.5,pos_weight=1,lambda_fp=0.05)
    #criterion = torch.nn.BCELoss()
    #optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # create a list of learning rate with epochs
    # lr_schedule = make_lr_schedule(np.array([50, args.N_epochs]),np.array([0.001, 0.0001]))
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0)
    train_loader, val_loader = get_dataloaderV2(args) # create dataloader
    # train_loader, val_loader = get_dataloader(args)
    
    if args.val_on_test: 
        print('\033[0;32m===============Validation on Testset!!!===============\033[0m')
        val_tool = Test(args) 

    best = {'epoch':0,'f1':0.5} # Initialize the best epoch and performance(AUC of ROC)
    trigger = 0  # Early stop Counter
    for epoch in range(args.start_epoch,args.N_epochs+1):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
            (epoch, args.N_epochs,optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))
        
        # train stage
        train_log = train(train_loader,net,criterion, optimizer,device) 
        # val stage
        if not args.val_on_test:
            val_log = val(val_loader,net,criterion,device)
        else:
            val_tool.inference(net)
            val_log = val_tool.val()

        log.update(epoch,train_log,val_log) # Add log information
        lr_scheduler.step()

        # Save checkpoint of latest and best model.
        state = {'net': net.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['val_f1'] > best['f1']:
            print('\033[0;33mSaving best model!\033[0m')
            torch.save(state, join(save_path, 'best_model.pth'))
            best['epoch'] = epoch
            best['f1'] = val_log['val_f1']
            trigger = 0
        print('Best performance at Epoch: {} | f1: {}'.format(best['epoch'],best['f1']))
        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()
if __name__ == '__main__':
    main()


#     best = {'epoch':0,'SE':0.5} # Initialize the best epoch and performance(AUC of ROC)
#     trigger = 0  # Early stop Counter
#     for epoch in range(args.start_epoch,args.N_epochs+1):
#         print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
#             (epoch, args.N_epochs,optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))
        
#         # train stage
#         train_log = train(train_loader,net,criterion, optimizer,device) 
#         # val stage
#         if not args.val_on_test:
#             val_log = val(val_loader,net,criterion,device)
#         else:
#             val_tool.inference(net)
#             val_log = val_tool.val()

#         log.update(epoch,train_log,val_log) # Add log information
#         lr_scheduler.step()

#         # Save checkpoint of latest and best model.
#         state = {'net': net.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
#         torch.save(state, join(save_path, 'latest_model.pth'))
#         trigger += 1
#         if val_log['SE'] > best['SE']:
#             print('\033[0;33mSaving best model!\033[0m')
#             torch.save(state, join(save_path, 'best_model.pth'))
#             best['epoch'] = epoch
#             best['SE'] = val_log['SE']
#             trigger = 0
#         print('Best performance at Epoch: {} | SE: {}'.format(best['epoch'],best['SE']))
#         # early stopping
#         if not args.early_stop is None:
#             if trigger >= args.early_stop:
#                 print("=> early stopping")
#                 break
#         torch.cuda.empty_cache()
# if __name__ == '__main__':
#     main()
