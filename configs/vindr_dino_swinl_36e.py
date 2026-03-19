from __future__ import annotations

custom_imports = dict(
    imports=['vindr_detector.transforms', 'vindr_detector.metrics'],
    allow_failed_imports=False,
)

_base_ = 'mmdet::dino/dino-5scale_swin-l_8xb2-36e_coco.py'

classes = (
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Clavicle fracture',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Enlarged PA',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Lung cavity',
    'Lung cyst',
    'Mediastinal shift',
    'Nodule/Mass',
    'Other lesion',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
    'Rib fracture',
)
metainfo = dict(classes=classes)
num_classes = len(classes)

data_root = '.'
prepared_root = 'artifacts/vindr_cxr/annotations'
train_ann_file = f'{prepared_root}/train.json'
val_ann_file = f'{prepared_root}/val.json'
test_ann_file = f'{prepared_root}/test.json'

load_from = 'checkpoints/checkpoint0027_5scale_swin.pth'
resume = False

model = dict(
    bbox_head=dict(num_classes=num_classes),
    test_cfg=dict(max_per_img=300),
)

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='EnsureThreeChannelGray'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1024, 2048), (1152, 2048), (1280, 2048), (1408, 2048), (1536, 2048)],
        keep_ratio=True,
    ),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='EnsureThreeChannelGray'),
    dict(type='Resize', scale=(1536, 2048), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.005,
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            ann_file=train_ann_file,
            data_prefix=dict(img='vincxr/train/'),
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=False),
            pipeline=train_pipeline,
        ),
    ),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img='vincxr/train/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=test_ann_file,
        data_prefix=dict(img='vincxr/test/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=16,
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1,
    )
]

val_evaluator = [
    dict(type='CocoMetric', ann_file=val_ann_file, metric='bbox'),
    dict(type='VinDRMetric', iou_thr=0.4, froc_max_fp_per_img=8.0),
]

test_evaluator = [
    dict(type='CocoMetric', ann_file=test_ann_file, metric='bbox'),
    dict(type='VinDRMetric', iou_thr=0.4, froc_max_fp_per_img=8.0),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater',
    ))
