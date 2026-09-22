common_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        0.0,
        0.0,
        0.0,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        0,
        0,
    ),
    std=[
        0.0,
        0.0,
        0.0,
    ],
    type='SegDataPreProcessor')
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=13966,
        max_keep_ckpts=1,
        save_best='average_mIoU',
        save_last=True,
        type='CheckpointHook'),
    early_stopping=dict(
        min_delta=0.01,
        monitor='average_mIoU',
        patience=10,
        rule='greater',
        type='EarlyStoppingHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=2,
        loss_decode=dict(
            loss_name='loss_custom',
            loss_weight=0.4,
            type='MyConfidenceLoss',
            use_mask=True),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=16,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        contract_dilation=True,
        depth=101,
        dilations=(
            1,
            1,
            2,
            4,
        ),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        strides=(
            1,
            2,
            1,
            1,
        ),
        style='pytorch',
        type='ResNetV1c'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            89.62896351465201,
            91.91043569689067,
            81.87682341265766,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            320,
            320,
        ),
        std=[
            34.175762420665585,
            29.145362003272353,
            31.408559393268405,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        c1_channels=48,
        c1_in_channels=256,
        channels=512,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        dropout_ratio=0.1,
        in_channels=2048,
        in_index=3,
        loss_decode=dict(
            loss_name='loss_custom',
            loss_weight=1.0,
            type='MyConfidenceLoss',
            use_mask=True),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=16,
        type='DepthwiseSeparableASPPHead'),
    pretrained='open-mmlab://resnet101_v1c',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=160000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = None
train_cfg = dict(
    max_iters=1490560, type='IterBasedTrainLoop', val_interval=13966)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        datasets=[
            dict(
                ann_file='train.txt',
                data_prefix=dict(
                    img_path='patch/image', seg_map_path='patch/label'),
                data_root=
                'dataset/Mmseg_UnifiedSegmentation/unifiedDataset/clds21',
                img_suffix='',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                seg_map_suffix='',
                type='MyUnifiedUnifiedDataset'),
            dict(
                ann_file='train.txt',
                data_prefix=dict(
                    img_path='patch/image', seg_map_path='patch/label'),
                data_root=
                'dataset/Mmseg_UnifiedSegmentation/unifiedDataset/flair',
                img_suffix='',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                seg_map_suffix='',
                type='MyUnifiedUnifiedDataset'),
            dict(
                ann_file='train.txt',
                data_prefix=dict(
                    img_path='patch/image', seg_map_path='patch/label'),
                data_root=
                'dataset/Mmseg_UnifiedSegmentation/unifiedDataset/oem',
                img_suffix='',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                seg_map_suffix='',
                type='MyUnifiedUnifiedDataset'),
        ],
        type='ConcatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                ann_file='val.txt',
                data_prefix=dict(
                    img_path='patch/image', seg_map_path='patch/label'),
                data_root=
                'dataset/Mmseg_UnifiedSegmentation/unifiedDataset/clds21',
                img_suffix='',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                seg_map_suffix='',
                type='MyUnifiedUnifiedDataset'),
            dict(
                ann_file='val.txt',
                data_prefix=dict(
                    img_path='patch/image', seg_map_path='patch/label'),
                data_root=
                'dataset/Mmseg_UnifiedSegmentation/unifiedDataset/flair',
                img_suffix='',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                seg_map_suffix='',
                type='MyUnifiedUnifiedDataset'),
            dict(
                ann_file='val.txt',
                data_prefix=dict(
                    img_path='patch/image', seg_map_path='patch/label'),
                data_root=
                'dataset/Mmseg_UnifiedSegmentation/unifiedDataset/oem',
                img_suffix='',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                seg_map_suffix='',
                type='MyUnifiedUnifiedDataset'),
        ],
        type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    metrics=[
        dict(iou_metrics=[
            'mIoU',
        ], prefix='clds21', type='MyConfidenceIoU'),
        dict(iou_metrics=[
            'mIoU',
        ], prefix='flair', type='MyConfidenceIoU'),
        dict(iou_metrics=[
            'mIoU',
        ], prefix='oem', type='MyConfidenceIoU'),
    ],
    type='MyAveragingEvaluator')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'model/unified/holdout/weights/deeplabv3plus-resnet101'
