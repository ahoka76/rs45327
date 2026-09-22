common_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
conv_kernel_size = 1
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmsegext',
    ])
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
        interval=7115,
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
    logger=dict(
        _scope_='mmseg',
        interval=50,
        log_metric_by_epoch=False,
        type='LoggerHook'),
    param_scheduler=dict(_scope_='mmseg', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmseg', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmseg', type='IterTimerHook'),
    visualization=dict(_scope_='mmseg', type='SegVisualizationHook'))
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
        in_channels=768,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        cffn_ratio=0.25,
        conv_inplane=64,
        deform_num_heads=12,
        deform_ratio=0.5,
        depth=12,
        dim_ratio=1.0,
        drop_path_rate=0.3,
        embed_dim=768,
        interaction_indexes=[
            [
                0,
                2,
            ],
            [
                3,
                5,
            ],
            [
                6,
                8,
            ],
            [
                9,
                11,
            ],
        ],
        mlp_ratio=4,
        n_points=4,
        num_heads=12,
        patch_size=16,
        pretrained=
        '/home/kari_kjle/.cache/torch/hub/checkpoints/deit_base_patch16_224-b5f2ef4d.pth',
        type='ext-ViTCoMer',
        window_attn=[
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        window_size=[
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            91.03998109421234,
            89.65864588669045,
            86.78493896410399,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            320,
            320,
        ),
        std=[
            31.380135632356495,
            29.231511521759224,
            32.482622376107,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        kernel_generate_head=dict(
            align_corners=False,
            channels=512,
            dropout_ratio=0.1,
            in_channels=[
                768,
                768,
                768,
                768,
            ],
            in_index=[
                0,
                1,
                2,
                3,
            ],
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=2,
            pool_scales=(
                1,
                2,
                3,
                6,
            ),
            type='UPerHead'),
        kernel_update_head=[
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='KernelUpdator'),
                num_classes=2,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='KernelUpdator'),
                num_classes=-1,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='KernelUpdator'),
                num_classes=-1,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
        ],
        num_stages=3,
        type='IterativeDecodeHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_stages = 3
optim_wrapper = dict(
    clip_grad=None,
    constructor='ext-LayerDecayOptimizerConstructorViTAdapter',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(layer_decay_rate=0.95, num_layers=12),
    type='OptimWrapper')
optimizer = dict(
    _scope_='mmseg', lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=160000,
        eta_min=0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = None
train_cfg = dict(
    _scope_='mmseg',
    max_iters=756800,
    type='IterBasedTrainLoop',
    val_interval=7115)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='train.txt',
        data_prefix=dict(img_path='patch/image', seg_map_path='patch/label'),
        data_root=
        'dataset/Mmseg_Clds21SegmentationPartA',
        img_suffix='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='',
        type='MyClds21Dataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
tta_model = dict(_scope_='mmseg', type='SegTTAModel')
val_cfg = dict(_scope_='mmseg', type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val.txt',
        data_prefix=dict(img_path='patch/image', seg_map_path='patch/label'),
        data_root=
        'dataset/Mmseg_Clds21SegmentationPartA',
        img_suffix='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='',
        type='MyClds21Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(_scope_='mmseg', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmseg',
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'model/cld/partA/weights/knet-vitcomerbase'
