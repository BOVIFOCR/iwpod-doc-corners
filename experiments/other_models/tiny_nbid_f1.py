
data_root = '.data/nbid_detr/'

train_ann_file = 'stf/bt1/train_lbs/labelTxt'
train_data_prefix = 'stf/bt1/train_ims/images/'

val_ann_file = 'stf/bt1/valid_lbs/labelTxt'
val_data_prefix = 'stf/bt1/valid_ims/images'

test_data_prefix = 'stf/bt1/test_ims/images'
data_prefix = 'images/'

class_name = ('doc')
classes = ('doc')
CLASSES = ('doc')
num_classes = 1

_base_ = './rtmdet-r_l_1.py'
train_batch_size_per_gpu = 8

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

# ========================modified parameters======================
deepen_factor = 0.167
widen_factor = 0.375

# Batch size of a single GPU during training
#train_batch_size_per_gpu = 8

# Submission dir for result submit
submission_dir = './work_dirs/{{fileBasenameNoExtension}}/submission'

# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(checkpoint=checkpoint)),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

train_dataloader = dict(batch_size=train_batch_size_per_gpu)



# Inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
# test_dataloader = dict(
#     dataset=dict(
#         data_root=_base_.data_root,
#         ann_file='', # test set has no annotation
#         data_prefix=dict(img_path=_base_.test_data_prefix),
#         pipeline=_base_.test_pipeline))
# test_evaluator = dict(
#     type='mmrotate.DOTAMetric',
#     format_only=True,
#     merge_patches=True,
#     outfile_prefix=submission_dir)
