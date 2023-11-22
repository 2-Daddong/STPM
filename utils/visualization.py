import os
import numpy as np
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib
import matplotlib.pyplot as plt


#overall
def plt_fig_cls(test_img, gt_list, scores, img_scores, cls_threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    vmax = vmax * 0.5 + vmin * 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt_label = gt_list[i]
        heat_map = scores[i] * 255
        mask = scores[i]
        # mask[mask > threshold] = 1
        # mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        # vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 3, figsize=(9, 3), gridspec_kw={'width_ratios': [4, 4, 4]})

        fig_img.subplots_adjust(wspace=0.05, hspace=0)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Input image')
        ax_img[1].imshow(heat_map, cmap='jet', norm=norm, interpolation='none')
        ax_img[1].imshow(img, cmap='gray', alpha=0.7, interpolation='none')
        ax_img[1].title.set_text('Heat map')
        black_mask = np.zeros((int(mask.shape[0]), int(3 * mask.shape[1] / 4)))
        ax_img[2].imshow(black_mask, cmap='gray')
        ax = plt.gca()
        if img_scores[i] > cls_threshold:
            cls_result = 'NG'

            if gt_label == 1:
                result_str='True: NG / Pred: NG'
            else:
                result_str='True: OK / Pred: NG'
        else:
            cls_result = 'OK'

            if gt_label == 1:
                result_str='True: NG / Pred: OK'
            else:
                result_str='True: OK / Pred: OK'

        ax.text(0.05,
                0.89,
                'Detected anomalies',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.79,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.72,
                'Results',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.67,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.59,
                '\'{}\''.format(result_str),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='r',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.47,
                'Anomaly scores: {:.2f}'.format(img_scores[i]),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.37,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.30,
                'Thresholds',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.25,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.17,
                'Classification: {:.2f}'.format(cls_threshold),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        
        ax_img[2].title.set_text('Classification')

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=300, bbox_inches='tight')
        plt.close()        


#sub
def plt_subfig_cls(test_img, gt_list, scores, img_scores, cls_threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    vmax = vmax * 0.5 + vmin * 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt_label = gt_list[i]
        heat_map = scores[i] * 255
        mask = scores[i]
        # mask[mask > threshold] = 1
        # mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        # vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 3, figsize=(9, 3), gridspec_kw={'width_ratios': [4, 4, 4]})

        fig_img.subplots_adjust(wspace=0.05, hspace=0)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Input image')
        ax_img[1].imshow(heat_map, cmap='jet', norm=norm, interpolation='none')
        ax_img[1].imshow(img, cmap='gray', alpha=0.7, interpolation='none')
        ax_img[1].title.set_text('Heat map')
        black_mask = np.zeros((int(mask.shape[0]), int(3 * mask.shape[1] / 4)))
        ax_img[2].imshow(black_mask, cmap='gray')
        ax = plt.gca()
        if img_scores[i] > cls_threshold:
            cls_result = 'NG'

            if gt_label == 1:
                result_str='True: NG / Pred: NG'
            else:
                result_str='True: OK / Pred: NG'

                make_fig(ax, cls_threshold, img_scores[i], result_str)
                ax_img[2].title.set_text('Classification')
                fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=300, bbox_inches='tight')
                plt.close()
        else:
            cls_result = 'OK'

            if gt_label == 1:
                result_str='True: NG / Pred: OK'

                make_fig(ax, cls_threshold, img_scores[i], result_str)
                ax_img[2].title.set_text('Classification')
                fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                result_str='True: OK / Pred: OK'



def make_fig(ax, cls_threshold, img_score, result_str):
    ax.text(0.05,
            0.89,
            'Detected anomalies',
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes,
            fontdict=dict(
                fontsize=8,
                color='w',
                family='sans-serif',
            ))
    ax.text(0.05,
            0.79,
            '------------------------',
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes,
            fontdict=dict(
                fontsize=8,
                color='w',
                family='sans-serif',
            ))
    ax.text(0.05,
            0.72,
            'Results',
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes,
            fontdict=dict(
                fontsize=8,
                color='w',
                family='sans-serif',
            ))
    ax.text(0.05,
            0.67,
            '------------------------',
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes,
            fontdict=dict(
                fontsize=8,
                color='w',
                family='sans-serif',
            ))
    ax.text(0.05,
            0.59,
            '\'{}\''.format(result_str),
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes,
            fontdict=dict(
                fontsize=8,
                color='r',
                family='sans-serif',
            ))
    ax.text(0.05,
            0.47,
            'Anomaly scores: {:.2f}'.format(img_score),
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes,
            fontdict=dict(
                fontsize=8,
                color='w',
                family='sans-serif',
            ))
    ax.text(0.05,
            0.37,
            '------------------------',
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes,
            fontdict=dict(
                fontsize=8,
                color='w',
                family='sans-serif',
            ))
    ax.text(0.05,
            0.30,
            'Thresholds',
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes,
            fontdict=dict(
                fontsize=8,
                color='w',
                family='sans-serif',
            ))
    ax.text(0.05,
            0.25,
            '------------------------',
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes,
            fontdict=dict(
                fontsize=8,
                color='w',
                family='sans-serif',
            ))
    ax.text(0.05,
            0.17,
            'Classification: {:.2f}'.format(cls_threshold),
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes,
            fontdict=dict(
                fontsize=8,
                color='w',
                family='sans-serif',
            ))
            

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x