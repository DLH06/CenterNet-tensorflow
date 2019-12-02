from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tfrecord_voc_utils as voc_utils
import tensorflow as tf
import numpy as np
import CenterNet as net
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform
from utils.voc_classname_encoder import classname_to_ids

lr = 0.0001
batch_size = 15
buffer_size = 128
epochs = 100
reduce_lr_epoch = []
config = {
    'mode': 'test',                                       # 'train', 'test'
    'input_size': 384,
    'data_format': 'channels_last',                        # 'channels_last' 'channels_first'
    'num_classes': 11,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,                                      # not used
    'batch_size': batch_size,

    'score_threshold': 0.1,
    'top_k_results_output': 100,
}

image_augmentor_config = {
    'data_format': 'channels_last',
    'output_shape': [384, 384],
    'pad_truth_to': 60,
}

data = os.listdir('./train/')
data = [os.path.join('./train/', name) for name in data]

train_gen = voc_utils.get_generator(data, batch_size, buffer_size, image_augmentor_config)
trainset_provider = {
    'data_shape': [384, 384, 3],
    'num_train': 3200,
    'num_val': 0,                                         # not used
    'train_generator': train_gen,
    'val_generator': None                                 # not used
}
centernet = net.CenterNet(config, trainset_provider)
centernet.load_weight('./centernet/test-37062')

img = io.imread('./valid/colon_with_number/93.png')
img = transform.resize(img, [384,384, 3])
img = np.expand_dims(img, 0)
result = centernet.test_one_image(img)
id_to_clasname = {k:v for (v,k) in classname_to_ids.items()}
scores = result[0]
bbox = result[1]
class_id = result[2]
print(scores, bbox, class_id)
plt.figure(1)
plt.imshow(np.squeeze(img))
axis = plt.gca()
for i in range(len(scores)):
    rect = patches.Rectangle((bbox[i][1],bbox[i][0]), bbox[i][3]-bbox[i][1],bbox[i][2]-bbox[i][0],linewidth=2,edgecolor='b',facecolor='none')
    axis.add_patch(rect)
    plt.text(bbox[i][1],bbox[i][0], id_to_clasname[class_id[i]]+str(' ')+str(scores[i]), color='red', fontsize=12)
plt.show()
