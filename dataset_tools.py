import os
import cv2
import numpy as np
from tqdm import tqdm

import splitfolders
from pycocotools.coco import COCO


def cocoform2cropimg(annot_file, data_dir, save_dir):
	### create save directory ###
	ok_save_dir = save_dir + 'OK/'
	ng_save_dir = save_dir + 'NG/'
	os.makedirs(ok_save_dir, exist_ok=True)
	os.makedirs(ng_save_dir, exist_ok=True)

	### read annot json file ###
	img_list = os.listdir(data_dir)
	img_ids = len(img_list)
	coco_annot = COCO(annot_file)

	for i in tqdm(range(img_ids)):
		img_info = coco_annot.loadImgs(i)

		### load image ###
		img_name = os.path.splitext(img_info[0]['file_name'])[0]
		img_ext = os.path.splitext(img_info[0]['file_name'])[1]

		# img_dir = np.fromfile(data_dir+img_name+img_ext, np.uint8)
		# img = cv2.imdecode(img_dir)

		img = cv2.imread(data_dir+img_name+img_ext)

		### load normal/abnormal annotation info ###
		annot_ng0_ids = coco_annot.getAnnIds(imgIds=i, catIds=0) #cavity 0
		annot_ng1_ids = coco_annot.getAnnIds(imgIds=i, catIds=1) #cavity 1
		annot_ok_ids = coco_annot.getAnnIds(imgIds=i, catIds=2) #normal

		annot_ng0_info = coco_annot.loadAnns(annot_ng0_ids)
		annot_ng1_info = coco_annot.loadAnns(annot_ng1_ids)
		annot_ok_info = coco_annot.loadAnns(annot_ok_ids)

		# normal image crop
		for j, annot in enumerate(annot_ok_info):
			x, y, w, h = annot['bbox'][0], annot['bbox'][1], annot['bbox'][2], annot['bbox'][3]
			crop_img = img[int(y):int(y+h), int(x):int(x+w)]

			if w > 50 and h > 50:
				cv2.imwrite(ok_save_dir+img_name+f'_{j}'+img_ext, crop_img) # save crop ok image

		# anormal0 image crop
		for j, annot in enumerate(annot_ng0_info):
			x, y, w, h = annot['bbox'][0], annot['bbox'][1], annot['bbox'][2], annot['bbox'][3]
			crop_img = img[int(y):int(y+h), int(x):int(x+w)]

			if w > 50 and h > 50:
				cv2.imwrite(ng_save_dir+'0/'+img_name+f'_{j+1}'+img_ext, crop_img) # save crop ng image

		# anormal1 image crop
		for j, annot in enumerate(annot_ng1_info):
			x, y, w, h = annot['bbox'][0], annot['bbox'][1], annot['bbox'][2], annot['bbox'][3]
			crop_img = img[int(y):int(y+h), int(x):int(x+w)]

			if w > 50 and h > 50:
				cv2.imwrite(ng_save_dir+'1/'+img_name+f'_{j+1}'+img_ext, crop_img) # save crop ng image
	



def split_dataset(data_dir, save_dir):
	ratio = (.9,.1) # train/val/test

	splitfolders.ratio(data_dir, output=save_dir, seed=64, ratio=ratio)




if __name__ == '__main__':

	''' create crop image mode '''
	annot_dir = 'datasets/cavity_recheck/_annotations_test.json'
	data_dir = 'datasets/cavity_recheck/test/'
	save_dir = 'datasets/cavity_recheck/'

	cocoform2cropimg(annot_dir, data_dir, save_dir)


	''' split dataset '''
	# data_dir = 'datasets/cavity/data/'
	# save_dir = 'datasets/cavity/split_dataset_ok'

	# split_dataset(data_dir, save_dir)




	''' remove train crop image list '''
	# annot ids: 873, 2829
	# {
 #      "id": 873,
 #      "image_id": 64,
 #      "category_id": 1,
 #      "bbox": [
 #          98,
 #          183,
 #          33.72,
 #          0.02
 #      ],
 #      "area": 0.6744,
 #      "segmentation": [],
 #      "iscrowd": 0
	# }
	# {
	#     "id": 2829,
	#     "image_id": 236,
	#     "category_id": 1,
	#     "bbox": [
	#         204,
	#         108,
	#         0.16,
	#         0.42
	#     ],
	#     "area": 0.0672,
	#     "segmentation": [],
	#     "iscrowd": 0
	# }