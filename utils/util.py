import time
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def cal_loss(fs_list, ft_list):
    t_loss = 0
    N = len(fs_list)
    for i in range(N):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        f_loss = 0.5 * (ft_norm - fs_norm)**2
        f_loss = f_loss.sum() / (h*w)
        t_loss += f_loss

    return t_loss / N


def cal_anomaly_maps(fs_list, ft_list, out_size):
    anomaly_map = 0
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        _, _, h, w = fs.shape
        a_map = (0.5 * (ft_norm - fs_norm)**2) / (h*w)
        a_map = a_map.sum(1, keepdim=True)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
        anomaly_map += a_map

    anomaly_map = anomaly_map.squeeze(dim=1).cpu().numpy()

    for i in range(anomaly_map.shape[0]):
        anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

    return anomaly_map


def cal_confusion_matrix(gt, pred, cls_threshold):
    pred_label = []
    FN = 0
    FP = 0
    TP = 0
    TN = 0

    for i in range(len(pred)):
        if pred[i] > cls_threshold:
            pred_label.append(1) #NG
        else:
            pred_label.append(0) #OK

    cm = confusion_matrix(gt, pred_label)
    TP, TN, FP, FN = cm[1][1], cm[0][0], cm[0][1], cm[1][0]
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    r_ng = TP / (TP+FN)
    f_ng = FP / (TN+FP)

    print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
    print(f'Accuracy: {accuracy}')
    print(f'진성불량: {r_ng} || 가성불량: {f_ng}')

    return TP, TN, FP, FN, accuracy, r_ng, f_ng


def save_graph(x_value_list, y_value_list, label, dir_save, seed):
    plt.figure()
    plt.plot(x_value_list, y_value_list, '.-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(f'{dir_save}{label}_{seed}.png', dpi=300)
    plt.close()
