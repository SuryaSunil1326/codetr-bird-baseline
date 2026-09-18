# CoDETR bird instance segmentation, single-class, 512x512 images

_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]
checkpoint_config = dict(interval=25, max_keep_ckpts=3)
resume_from = None
load_from = "checkpoints/coco_dino_5scale_vit_large_coco_instance.pth"
pretrained = None
window_block_indexes = (
    list(range(0, 3)) + list(range(4, 7)) + list(range(8, 11)) + list(range(12, 15)) + list(range(16, 19)) +
    list(range(20, 23)) + list(range(24, 27)))
residual_block_indexes = []

num_dec_layer = 6
lambda_2 = 2.0

model = dict(
    type='CoDETR',
    backbone=dict(
        type='ViT',
        img_size=512,
        pretrain_img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0.3,
        window_size=24,
        window_block_indexes=window_block_indexes,
        residual_block_indexes=residual_block_indexes,
        qkv_bias=True,
        use_act_checkpoint=True,
        init_cfg=None),
    neck=dict(
        type='SFP',
        in_channels=[1024],
        out_channels=256,
        num_outs=5,
        use_p2=True,
        use_act_checkpoint=False),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=3,
            scales_per_octave=5,
            ratios=[0.25, 0.5, 1.0, 2.0, 3.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0 * num_dec_layer * lambda_2),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0 * num_dec_layer * lambda_2)),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32, 64],
        finest_scale=56),
    mask_head=dict(
        type='SimpleRefineMaskHead',
        num_convs_instance=1,
        num_convs_semantic=2,
        conv_in_channels_instance=256,
        conv_in_channels_semantic=256,
        conv_kernel_size_instance=3,
        conv_kernel_size_semantic=3,
        conv_out_channels_instance=256,
        conv_out_channels_semantic=256,
        conv_cfg=None,
        norm_cfg=dict(type='LN2d'),
        fusion_type='MultiBranchFusionAvg',
        dilations=[1, 3, 5],
        semantic_out_stride=4,
        stage_num_classes=[1, 1, 1, 1],
        stage_sup_size=[14, 28, 56, 112],
        pre_upsample_last_stage=False,
        upsample_cfg=dict(type='bilinear', scale_factor=2),
        loss_weight=1.33 * num_dec_layer * lambda_2,
        loss_cfg=dict(
            type='BARCrossEntropyLoss',
            stage_instance_loss_weight=[0.6, 0.9, 1.2, 1.5],
            boundary_width=2,
            start_stage=1)),
    mask_iou_head=dict(
        type='MaskIoUHead',
        num_convs=2,
        num_fcs=1,
        roi_feat_size=14,
        in_channels=256,
        conv_out_channels=256,
        fc_out_channels=1024,
        num_classes=1,
        score_use_sigmoid=True,
        norm_cfg=dict(type='LN2d'),
        loss_iou=dict(type='MSELoss', loss_weight=0.5 * num_dec_layer * lambda_2)),
    query_head=dict(
        type='CoDINOHead',
        num_query=1500,
        num_classes=1,
        num_feature_levels=5,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        mixed_selection=True,
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=0.4),
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=600)
        ),
        transformer=dict(
            type='CoDinoTransformer',
            with_pos_coord=True,
            with_coord_feat=False,
            num_co_heads=2,
            num_feature_levels=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                with_cp=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256, num_levels=5, dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=5,
                            dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    roi_head=[dict(
        type='CoStandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64],
            finest_scale=56),
        bbox_head=dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1]),
            reg_class_agnostic=True,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0 * num_dec_layer * lambda_2),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0 * num_dec_layer * lambda_2)))],
    bbox_head=[dict(
        type='CoATSSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=1,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=4,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0 * num_dec_layer * lambda_2),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0 * num_dec_layer * lambda_2),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0 * num_dec_layer * lambda_2)),],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),

        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_thr_binary=0.4,
            mask_size=128,
            pos_weight=-1,
            debug=False),

        dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.4,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.4,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        dict(
            assigner=dict(type='ATSSAssigner', topk=12),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),],
    test_cfg=[
        dict(
            max_per_img=1000,
            nms=dict(type='soft_nms', iou_threshold=0.8),
            mask_thr_binary=0.2),
        dict(
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.5),
            mask_thr_binary=0.2,
            max_per_img=1000),
        dict(
            rpn=dict(
                nms_pre=8000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.9),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                mask_thr_binary=0.2,
                nms=dict(type='soft_nms', iou_threshold=0.5),
                max_per_img=1000)),
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='soft_nms', iou_threshold=0.6),
            max_per_img=300),  # keep higher max_per_img for dense scenes
    ])
    

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),

    # -------------------------------------------------
    # ENHANCED PHOTOMETRIC DISTORTION (stronger contrast)
    # -------------------------------------------------
       # -------------------------------------------------
    # Enhanced photometric distortion (white-on-white robustness)
    # -------------------------------------------------
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=48,
        contrast_range=(0.3, 1.7),
        saturation_range=(0.3, 1.7),
        hue_delta=8
    ),

    # Slight additional color transform (forces reduced color reliance)
    dict(
        type='ColorTransform',
        level=6,
        prob=0.3
    ),

    # -------------------------------------------------
    # YOUR EXISTING AUTOAUGMENT (UNCHANGED)
    # -------------------------------------------------
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[
                        (512, 512),
                        (576, 576),
                        (640, 640),
                        (704, 704),
                        (768, 768),
                        (896, 896)
                    ],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 512), (512, 512), (544, 544)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 384),
                    allow_negative_crop=False),
                dict(
                    type='Resize',
                    img_scale=[
                        (512, 512),
                        (576, 576),
                        (640, 640),
                        (704, 704)
                    ],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True),
            ],
            [
                dict(
                    type='Resize',
                    img_scale=[(256, 256), (320, 320)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(224, 224),
                    allow_negative_crop=False),
                dict(
                    type='Resize',
                    img_scale=[(512, 512)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True),
            ]
        ]),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),     # SINGLE scale only
        flip=False,               # MUST be False
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]



data = dict(
    samples_per_gpu=7,
    workers_per_gpu=0,
    train=dict(filter_empty_gt=False, pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)


evaluation = dict(metric=['bbox', 'segm'])

# learning policy – tuned for ~80 epochs
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[80, 110])

runner = dict(
    type='EpochBasedRunner',
    max_epochs=130)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=0.01,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=24, layer_decay_rate=0.8))

custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        momentum=0.0001,
        priority=49),]

evaluation = dict(metric=['bbox', 'segm'])
