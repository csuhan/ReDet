_base_ = [
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ReResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        orientation=8,
        fixparams=False),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
total_epochs = 100
checkpoint_config = dict(interval=5)
evaluation = dict(interval=10, metric='accuracy')