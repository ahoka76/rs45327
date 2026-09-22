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
        interval=1123,
        max_keep_ckpts=1,
        save_best='mIoU',
        save_last=True,
        type='CheckpointHook'),
    early_stopping=dict(
        min_delta=0.01,
        monitor='mIoU',
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
    backbone=dict(
        extra=dict(
            stage1=dict(
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_branches=1,
                num_channels=(64, ),
                num_modules=1),
            stage2=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                ),
                num_branches=2,
                num_channels=(
                    18,
                    36,
                ),
                num_modules=1),
            stage3=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                ),
                num_branches=3,
                num_channels=(
                    18,
                    36,
                    72,
                ),
                num_modules=4),
            stage4=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                    4,
                ),
                num_branches=4,
                num_channels=(
                    18,
                    36,
                    72,
                    144,
                ),
                num_modules=3)),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        type='HRNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            89.18643620191202,
            93.37468188584575,
            74.88180733607905,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            320,
            320,
        ),
        std=[
            35.03594445467898,
            29.023596816022877,
            32.257740870961655,
        ],
        type='SegDataPreProcessor'),
    decode_head=[
        dict(
            align_corners=False,
            channels=270,
            concat_input=False,
            dropout_ratio=-1,
            in_channels=[
                18,
                36,
                72,
                144,
            ],
            in_index=(
                0,
                1,
                2,
                3,
            ),
            input_transform='resize_concat',
            kernel_size=1,
            loss_decode=dict(
                loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=9,
            num_convs=1,
            type='FCNHead'),
        dict(
            align_corners=False,
            channels=512,
            dropout_ratio=-1,
            in_channels=[
                18,
                36,
                72,
                144,
            ],
            in_index=(
                0,
                1,
                2,
                3,
            ),
            input_transform='resize_concat',
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=9,
            ocr_channels=256,
            type='OCRHead'),
    ],
    num_stages=2,
    pretrained='open-mmlab://msra/hrnetv2_w18',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='CascadeEncoderDecoder')
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
train_cfg = dict(max_iters=89840, type='IterBasedTrainLoop', val_interval=1123)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='train.txt',
        data_prefix=dict(img_path='patch/image', seg_map_path='patch/label'),
        data_root=
        'dataset/Mmseg_OemSegmentation',
        img_suffix='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='',
        type='MyOpenEarthMapDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val.txt',
        data_prefix=dict(img_path='patch/image', seg_map_path='patch/label'),
        data_root=
        'dataset/Mmseg_OemSegmentation',
        img_suffix='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='',
        type='MyOpenEarthMapDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'model/oem/holdout/weights/hrnetocr-hrnetw18'
