import os
import argparse
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from sklearn.metrics import precision_recall_curve, confusion_matrix, plot_confusion_matrix
from sklearn.metrics import roc_auc_score

from models.resnet_backbone import modified_resnet18
from utils.util import  time_string, convert_secs2time, AverageMeter, cal_anomaly_maps, cal_loss, cal_confusion_matrix, save_graph
from utils.visualization import plt_fig_cls, plt_subfig_cls
from utils.custom import CustomDataset


class STPM():
    def __init__(self, args, seed_num):
        self.device = args.device
        self.data_path = args.data_path
        self.save_path = args.save_path
        self.obj = args.obj
        self.img_inputsize = args.img_inputsize
        self.img_resize = args.img_resize
        self.img_cropsize = args.img_cropsize
        self.validation_ratio = args.validation_ratio
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.vis = args.vis
        self.model_dir = args.model_dir
        self.img_dir = args.img_dir
        self.matrix = args.matrix
        self.img_nums = 0
        self.seed = seed_num

        self.load_model()
        self.load_dataset()

        # self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model_s.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)


    def load_dataset(self):
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

        #load all train dataset
        train_dataset = CustomDataset(self.data_path, class_name=self.obj, is_train=True, inputsize=self.img_inputsize, resize=self.img_resize, cropsize=self.img_cropsize)
        
        #split train/valid
        self.valid_num = int(len(train_dataset) * self.validation_ratio)
        self.train_num = len(train_dataset) - self.valid_num
        train_data, val_data = torch.utils.data.random_split(train_dataset, [self.train_num, self.valid_num], generator=torch.Generator().manual_seed(self.seed))
        
        #load train/val dataset
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=False, **kwargs)


    def load_model(self):
        self.model_t = modified_resnet18().to(self.device)
        self.model_s = modified_resnet18(pretrained=False).to(self.device)
        for param in self.model_t.parameters():
            param.requires_grad = False
        self.model_t.eval()


    def train(self):
        self.model_s.train()
        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()
        train_loss = []
        valid_loss = []
        x_epoch = []

        for epoch in range(1, self.num_epochs+1):
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * ((self.num_epochs+1) - epoch))
            need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
            print('{:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, self.num_epochs, time_string(), need_time))

            losses = AverageMeter()

            for (data, _) in tqdm(self.train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    features_t = self.model_t(data)
                    features_s = self.model_s(data)
                    loss = cal_loss(features_s, features_t)

                    losses.update(loss.sum().item(), data.size(0))
                    loss.backward()
                    self.optimizer.step()

            print('Train Epoch: {} loss: {:.6f}'.format(epoch, losses.avg))

            val_loss = self.val(epoch)

            #save checkpoint at best score
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()

            x_epoch.append(epoch)
            train_loss.append(losses.avg)
            valid_loss.append(val_loss)

            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        print('Training end.')


        return x_epoch, train_loss, valid_loss
    

    def val(self, epoch):
        self.model_s.eval()
        losses = AverageMeter()

        for (data, _) in tqdm(self.val_loader):
            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                features_t = self.model_t(data)
                features_s = self.model_s(data)
                loss = cal_loss(features_s, features_t)
                losses.update(loss.item(), data.size(0))

        print('Val Epoch: {} loss: {:.6f}'.format(epoch, losses.avg))

        return losses.avg


    def save_checkpoint(self):
        print('Save model !!!')
        state = {'model':self.model_s.state_dict()}
        torch.save(state, os.path.join(self.model_dir, 'model_s_'+str(self.seed)+'.pth'))


    def test(self):
        try:
            checkpoint = torch.load(os.path.join(self.model_dir, 'model_s_'+str(self.seed)+'.pth'))
        except:
            raise Exception('Check saved model path.')

        self.model_s.load_state_dict(checkpoint['model'])
        self.model_s.eval()

        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        
        #load test dataset
        test_dataset = CustomDataset(self.data_path, class_name=self.obj, is_train=False, inputsize=self.img_inputsize, resize=self.img_resize, cropsize=self.img_cropsize)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

        #value initialize
        scores = []
        test_imgs = []
        gt_list = []

        print('Testing')

        for (data, label) in tqdm(test_loader):
            test_imgs.extend(data.cpu().numpy())
            gt_list.extend(label.cpu().numpy())

            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                features_t = self.model_t(data)
                features_s = self.model_s(data)

                score = cal_anomaly_maps(features_s, features_t, self.img_inputsize)
            scores.extend(score)

        #nomalize anomaly score
        scores = np.asarray(scores)
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
        
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)

        #check ROCAUC
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        print('image ROCAUC: %.3f' % (img_roc_auc))

        #extract cls threshold according to precision and recall
        precision, recall, thresholds = precision_recall_curve(gt_list.flatten(), img_scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        cls_threshold = thresholds[np.argmax(f1)]

        
        #visualize result heatmap image
        if self.vis:
            #overall image result
            plt_fig_cls(test_imgs, gt_list.flatten(), scores, img_scores, cls_threshold, 
                self.img_dir, self.obj)

            #sub image  result
            # plt_subfig_cls(test_imgs, gt_list.flatten(), scores, img_scores, cls_threshold, 
            #         self.img_dir, self.obj)

        #calculate confusion matrix
        if self.matrix:
            TP, TN, FP, FN, Accuracy, r_ng, f_ng = cal_confusion_matrix(gt_list.flatten(), img_scores.flatten(), cls_threshold)

            cm_file = open(f'{self.save_path}/cm_result_'+ self.obj +'.txt', 'a')
            cm_file.write(f'============[seed={self.seed}] Class: {self.obj}============\n')
            cm_file.write('train data: %d || valid data: %d || test data: %d \n' % (self.train_num, self.valid_num, len(test_dataset)))
            cm_file.write('TP: ' + str(TP) + ', TN: ' + str(TN) + ', FP: ' + str(FP) + ', FN: ' + str(FN) + '\n')
            cm_file.write('Accuracy: ' + str(Accuracy) + '\n')
            cm_file.write('image ROCAUC: %.3f \n' % (img_roc_auc))
            cm_file.write('cls_threshold: %.3f \n\n' % (cls_threshold))

        return Accuracy, r_ng, f_ng

def get_args():
    parser = argparse.ArgumentParser(description='STPM anomaly detection')
    parser.add_argument('--phase', default='train')
    parser.add_argument("--data_path", type=str, default="./datasets/")
    parser.add_argument('--obj', type=str, default='cavity')
    parser.add_argument('--img_inputsize', type=int, default=(500,500))
    parser.add_argument('--img_resize', type=int, default=(1824,1824))   #default: CW=(600,800) / KH=(1200,1600) / CA=(1824,1824)
    parser.add_argument('--img_cropsize', default=(1824,1000))             #default: CW=(190,700) / KH=(600,800) / CA=(1824,1000)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--vis', type=eval, choices=[True, False], default=True)
    parser.add_argument('--matrix', type=bool, default=True)
    parser.add_argument('--loss_graph', type=bool, default=True)
    parser.add_argument("--save_path", type=str, default="./results/")
    parser.add_argument("--seed", type=list, default=[100])
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())

    args = get_args()

    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #define path
    args.device = device
    args.save_path = args.save_path + args.obj
    args.model_dir = args.save_path + '/models/'
    args.img_dir = args.save_path + '/imgs/'
    args.graph_dir = args.save_path + '/graphs/'

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)
    os.makedirs(args.graph_dir, exist_ok=True)
    
    accuracy_list = []
    r_ng_list = []
    f_ng_list = []
    start_time = time.time()

    #k-fold processing
    for i in tqdm(args.seed):
        seed_num = i
        stpm = STPM(args, seed_num)
        
        #train step
        if args.phase == 'train':
            print(f'train at seed: {seed_num}')
            x_epoch, train_loss, val_loss = stpm.train()

            #save loss graph
            if args.loss_graph:
                save_graph(x_epoch, train_loss, 'train', args.graph_dir, seed_num)
                save_graph(x_epoch, val_loss, 'val', args.graph_dir, seed_num)

        #test step
        elif args.phase == 'test':
            accuracy, r_ng, f_ng = stpm.test()

            accuracy_list.append(accuracy)
            r_ng_list.append(r_ng)
            f_ng_list.append(f_ng)

        #need set phase
        else:
            print('Phase argument must be train or test.')
            
        print(f'=====Complete at seed num: {seed_num}=====')
    

    #calculate cross validation reuslt
    if accuracy_list and r_ng_list and f_ng_list:
        k = len(accuracy_list)

        total_accuracy = sum(accuracy_list)
        total_r_ng = sum(r_ng_list)
        total_f_ng = sum(f_ng_list)

        kmean_accuracy = total_accuracy / k
        kmean_r_ng = total_r_ng / k
        kmean_f_ng = total_f_ng / k

        print(f'[{k}-fold result]\n')
        print(f'Accuracy: {round(kmean_accuracy*100, 2)}, 진성불량: {round(kmean_r_ng*100, 2)}, 가성불량: {round(kmean_f_ng*100, 2)}')
        print(f'=====Complete test=====')

    spend_time = time.time() - start_time
    print(f'total spend time: {spend_time}')