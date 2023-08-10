# SFD-re  
### 0708:  
1. （配置环境）第一步：Prepare for the running environment
- 下载了SFD的git库后，运行setup.py（`python setup.py develop`）
2. （配置环境）准备数据集：
```
SFD
├── data
│   ├── kitti_sfd_seguv_twise
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & depth_dense_twise & depth_pseudo_rgbseguv_twise
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2 & depth_dense_twise & depth_pseudo_rgbseguv_twise
│   │   │── gt_database
│   │   │── gt_database_pseudo_seguv
│   │   │── kitti_dbinfos_train_sfd_seguv.pkl
│   │   │── kitti_infos_test.pkl
│   │   │── kitti_infos_train.pkl
│   │   │── kitti_infos_trainval.pkl
│   │   │── kitti_infos_val.pkl
├── pcdet
├── tools
```
按照以上目录准备完毕。  
### 0709：  
1. 测试学姐的代码，结果把ubuntu图形界面搞的卡死了，还好有这个[帖子](https://blog.51cto.com/u_15060511/4147533)，安全重启，救我狗命。。。
2. setup SFD;
3. train:
- 整个网络架构：
```
  (module): SFD(
    (vfe): MeanVFE()
    (backbone_3d): VoxelBackBone8x(
      (conv_input): SparseSequential(
        (0): SubMConv3d()
        (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (conv1): SparseSequential(
        (0): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (conv2): SparseSequential(
        (0): SparseSequential(
          (0): SparseConv3d()
          (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (2): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (conv3): SparseSequential(
        (0): SparseSequential(
          (0): SparseConv3d()
          (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (2): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (conv4): SparseSequential(
        (0): SparseSequential(
          (0): SparseConv3d()
          (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (2): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (conv_out): SparseSequential(
        (0): SparseConv3d()
        (1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (map_to_bev_module): HeightCompression()
    (pfe): None
    (backbone_2d): BaseBEVBackbone(
      (blocks): ModuleList(
        (0): Sequential(
          (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
          (1): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
          (2): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (3): ReLU()
          (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (5): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (6): ReLU()
          (7): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (8): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (9): ReLU()
          (10): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (11): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (12): ReLU()
          (13): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (14): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (15): ReLU()
        )
        (1): Sequential(
          (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
          (1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)
          (2): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (3): ReLU()
          (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (5): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (6): ReLU()
          (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (8): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (9): ReLU()
          (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (11): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (12): ReLU()
          (13): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (14): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (15): ReLU()
        )
      )
      (deblocks): ModuleList(
        (0): Sequential(
          (0): ConvTranspose2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Sequential(
          (0): ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
    )
    (dense_head): AnchorHeadSingle(
      (cls_loss_func): SigmoidFocalClassificationLoss()
      (reg_loss_func): WeightedSmoothL1Loss()
      (dir_loss_func): WeightedCrossEntropyLoss()
      (conv_cls): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
      (conv_box): Conv2d(256, 14, kernel_size=(1, 1), stride=(1, 1))
      (conv_dir_cls): Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))
    )
    (point_head): None
    (roi_head): SFDHead(
      (proposal_target_layer): ProposalTargetLayer()
      (reg_loss_func): WeightedSmoothL1Loss()
      (roi_grid_pool_layers): ModuleList(
        (0): NeighborVoxelSAModuleMSG(
          (groupers): ModuleList(
            (0): VoxelQueryAndGrouping()
            (1): VoxelQueryAndGrouping()
          )
          (mlps_in): ModuleList(
            (0): Sequential(
              (0): Conv1d(64, 32, kernel_size=(1,), stride=(1,), bias=False)
              (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Conv1d(64, 32, kernel_size=(1,), stride=(1,), bias=False)
              (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (mlps_pos): ModuleList(
            (0): Sequential(
              (0): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (mlps_out): ModuleList(
            (0): Sequential(
              (0): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
              (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
            (1): Sequential(
              (0): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
              (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
          )
          (relu): ReLU()
        )
        (1): NeighborVoxelSAModuleMSG(
          (groupers): ModuleList(
            (0): VoxelQueryAndGrouping()
            (1): VoxelQueryAndGrouping()
          )
          (mlps_in): ModuleList(
            (0): Sequential(
              (0): Conv1d(64, 32, kernel_size=(1,), stride=(1,), bias=False)
              (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Conv1d(64, 32, kernel_size=(1,), stride=(1,), bias=False)
              (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (mlps_pos): ModuleList(
            (0): Sequential(
              (0): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (mlps_out): ModuleList(
            (0): Sequential(
              (0): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
              (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
            (1): Sequential(
              (0): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
              (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
          )
          (relu): ReLU()
        )
      )
      (cpconvs_layer): CPConvs(
        (pointnet1_fea): PointNet(
          (conv1): Conv1d(6, 12, kernel_size=(1,), stride=(1,))
          (conv2): Conv1d(12, 12, kernel_size=(1,), stride=(1,))
          (bn1): BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (pointnet1_wgt): PointNet(
          (conv1): Conv1d(6, 12, kernel_size=(1,), stride=(1,))
          (conv2): Conv1d(12, 12, kernel_size=(1,), stride=(1,))
          (bn1): BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (pointnet1_fus): PointNet(
          (conv1): Conv1d(108, 12, kernel_size=(1,), stride=(1,))
          (conv2): Conv1d(12, 12, kernel_size=(1,), stride=(1,))
          (bn1): BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (pointnet2_fea): PointNet(
          (conv1): Conv1d(12, 24, kernel_size=(1,), stride=(1,))
          (conv2): Conv1d(24, 24, kernel_size=(1,), stride=(1,))
          (bn1): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (pointnet2_wgt): PointNet(
          (conv1): Conv1d(6, 24, kernel_size=(1,), stride=(1,))
          (conv2): Conv1d(24, 24, kernel_size=(1,), stride=(1,))
          (bn1): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (pointnet2_fus): PointNet(
          (conv1): Conv1d(216, 24, kernel_size=(1,), stride=(1,))
          (conv2): Conv1d(24, 24, kernel_size=(1,), stride=(1,))
          (bn1): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (pointnet3_fea): PointNet(
          (conv1): Conv1d(24, 48, kernel_size=(1,), stride=(1,))
          (conv2): Conv1d(48, 48, kernel_size=(1,), stride=(1,))
          (bn1): BatchNorm1d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm1d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (pointnet3_wgt): PointNet(
          (conv1): Conv1d(6, 48, kernel_size=(1,), stride=(1,))
          (conv2): Conv1d(48, 48, kernel_size=(1,), stride=(1,))
          (bn1): BatchNorm1d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm1d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (pointnet3_fus): PointNet(
          (conv1): Conv1d(432, 48, kernel_size=(1,), stride=(1,))
          (conv2): Conv1d(48, 48, kernel_size=(1,), stride=(1,))
          (bn1): BatchNorm1d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm1d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (roiaware_pool3d_layer): RoIAwarePool3d()
      (conv_pseudo): SparseSequential(
        (0): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(90, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): SparseSequential(
          (0): SparseConv3d()
          (1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (2): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (fusion_layer): GAF(
        (attention): Attention(
          (fc1): Linear(in_features=128, out_features=32, bias=True)
          (fc2): Linear(in_features=128, out_features=32, bias=True)
          (fc3): Linear(in_features=64, out_features=2, bias=True)
          (conv1): Sequential(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (conv2): Sequential(
            (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
        (conv1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
        (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (shared_fc_layer): Sequential(
        (0): Linear(in_features=55296, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.3, inplace=False)
        (4): Linear(in_features=512, out_features=512, bias=False)
        (5): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
      )
      (cls_fc_layers): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Dropout(p=0.3, inplace=False)
        (4): Linear(in_features=512, out_features=512, bias=False)
        (5): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU()
      )
      (cls_pred_layer): Linear(in_features=512, out_features=1, bias=True)
      (reg_fc_layers): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=False)
        (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Dropout(p=0.3, inplace=False)
        (4): Linear(in_features=512, out_features=512, bias=False)
        (5): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU()
      )
      (reg_pred_layer): Linear(in_features=512, out_features=7, bias=True)
      (shared_fc_layer_pseudo): Sequential(
        (0): Linear(in_features=27648, out_features=256, bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.3, inplace=False)
        (4): Linear(in_features=256, out_features=256, bias=False)
        (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
      )
      (cls_fc_layers_pseudo): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Dropout(p=0.3, inplace=False)
        (4): Linear(in_features=256, out_features=256, bias=False)
        (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU()
      )
      (cls_pred_layer_pseudo): Linear(in_features=256, out_features=1, bias=True)
      (reg_fc_layers_pseudo): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Dropout(p=0.3, inplace=False)
        (4): Linear(in_features=256, out_features=256, bias=False)
        (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU()
      )
      (reg_pred_layer_pseudo): Linear(in_features=256, out_features=7, bias=True)
      (shared_fc_layer_valid): Sequential(
        (0): Linear(in_features=27648, out_features=256, bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.3, inplace=False)
        (4): Linear(in_features=256, out_features=256, bias=False)
        (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
      )
      (cls_fc_layers_valid): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Dropout(p=0.3, inplace=False)
        (4): Linear(in_features=256, out_features=256, bias=False)
        (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU()
      )
      (cls_pred_layer_valid): Linear(in_features=256, out_features=1, bias=True)
      (reg_fc_layers_valid): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Dropout(p=0.3, inplace=False)
        (4): Linear(in_features=256, out_features=256, bias=False)
        (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU()
      )
      (reg_pred_layer_valid): Linear(in_features=256, out_features=7, bias=True)
      (reg_loss_func_pseudo): WeightedSmoothL1Loss()
      (reg_loss_func_valid): WeightedSmoothL1Loss()
    )
  )
)
```  
4. debug环节：代码简直是疯狂报错，按照这个[教程](https://zhuanlan.zhihu.com/p/524097054)修改了代码，并把LR改为0.001，又弹出了新的错误：
```
Original Traceback (most recent call last):
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/utils/data/_utils/worker.py", line 202, in _worker_loop
    data = fetcher.fetch(index)
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/home/xilm/fuxian/SFD/pcdet/datasets/kitti/kitti_dataset_sfd.py", line 491, in collate_batch
    for key, val in data_dict.items():
AttributeError: 'list' object has no attribute 'items'
```
- 把`kitti_datasets_sfd.py`里第490行的`data_dict = data_dict.pop('valid_noise')`注释掉，可以运行了；
- 但是运行了一张图之后又一次
```
Traceback (most recent call last):
  File "/home/xilm/fuxian/SFD/tools/train.py", line 200, in <module>
    main()
  File "/home/xilm/fuxian/SFD/tools/train.py", line 155, in main
    train_model(
  File "/home/xilm/fuxian/SFD/tools/train_utils/train_utils.py", line 86, in train_model
    accumulated_iter = train_one_epoch(
  File "/home/xilm/fuxian/SFD/tools/train_utils/train_utils.py", line 38, in train_one_epoch
    loss, tb_dict, disp_dict = model_func(model, batch)
  File "/home/xilm/fuxian/SFD/pcdet/models/__init__.py", line 30, in model_func
    ret_dict, tb_dict, disp_dict = model(batch_dict)
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/xilm/fuxian/SFD/pcdet/models/detectors/sfd.py", line 11, in forward
    batch_dict = cur_module(batch_dict)
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/xilm/fuxian/SFD/pcdet/models/roi_heads/sfd_head.py", line 573, in forward
    targets_dict = self.assign_targets(batch_dict)
  File "/home/xilm/fuxian/SFD/pcdet/models/roi_heads/roi_head_template.py", line 117, in assign_targets
    targets_dict = self.proposal_target_layer.forward(batch_dict)
  File "/home/xilm/fuxian/SFD/pcdet/models/roi_heads/target_assigner/proposal_target_layer.py", line 32, in forward
    batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
  File "/home/xilm/fuxian/SFD/pcdet/models/roi_heads/target_assigner/proposal_target_layer.py", line 107, in sample_rois_for_rcnn
    sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)
  File "/home/xilm/fuxian/SFD/pcdet/models/roi_heads/target_assigner/proposal_target_layer.py", line 159, in subsample_rois
    raise NotImplementedError
NotImplementedError
```
- 决定把LR再调小一点
- 绝望了，啥方法都用过了，包括LR，normalize，以及把giou改为diou，都是notimplementederror...
- [其他的3D目标检测论文](https://cloud.tencent.com/developer/article/2233781)
### 0711:  
1. 按照论文原作者的提示，跑了一下voxel_rcnn，发现也会报同样的错。。。
```
VoxelRCNN(
  (vfe): MeanVFE()
  (backbone_3d): VoxelBackBone8x(
    (conv_input): SparseSequential(
      (0): SubMConv3d()
      (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (conv1): SparseSequential(
      (0): SparseSequential(
        (0): SubMConv3d()
        (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (conv2): SparseSequential(
      (0): SparseSequential(
        (0): SparseConv3d()
        (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): SparseSequential(
        (0): SubMConv3d()
        (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): SparseSequential(
        (0): SubMConv3d()
        (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (conv3): SparseSequential(
      (0): SparseSequential(
        (0): SparseConv3d()
        (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): SparseSequential(
        (0): SubMConv3d()
        (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): SparseSequential(
        (0): SubMConv3d()
        (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (conv4): SparseSequential(
      (0): SparseSequential(
        (0): SparseConv3d()
        (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): SparseSequential(
        (0): SubMConv3d()
        (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): SparseSequential(
        (0): SubMConv3d()
        (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (conv_out): SparseSequential(
      (0): SparseConv3d()
      (1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (map_to_bev_module): HeightCompression()
  (pfe): None
  (backbone_2d): BaseBEVBackbone(
    (blocks): ModuleList(
      (0): Sequential(
        (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
        (1): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        (2): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (3): ReLU()
        (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (6): ReLU()
        (7): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (8): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (9): ReLU()
        (10): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (11): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (12): ReLU()
        (13): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (14): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (15): ReLU()
        (16): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (17): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (18): ReLU()
      )
      (1): Sequential(
        (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
        (1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)
        (2): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (3): ReLU()
        (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (6): ReLU()
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (8): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (9): ReLU()
        (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (11): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (12): ReLU()
        (13): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (14): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (15): ReLU()
        (16): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (17): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (18): ReLU()
      )
    )
    (deblocks): ModuleList(
      (0): Sequential(
        (0): ConvTranspose2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): Sequential(
        (0): ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
  )
  (dense_head): AnchorHeadSingle(
    (cls_loss_func): SigmoidFocalClassificationLoss()
    (reg_loss_func): WeightedSmoothL1Loss()
    (dir_loss_func): WeightedCrossEntropyLoss()
    (conv_cls): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    (conv_box): Conv2d(256, 14, kernel_size=(1, 1), stride=(1, 1))
    (conv_dir_cls): Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))
  )
  (point_head): None
  (roi_head): VoxelRCNNHead(
    (proposal_target_layer): ProposalTargetLayer()
    (reg_loss_func): WeightedSmoothL1Loss()
    (roi_grid_pool_layers): ModuleList(
      (0): NeighborVoxelSAModuleMSG(
        (groupers): ModuleList(
          (0): VoxelQueryAndGrouping()
        )
        (mlps_in): ModuleList(
          (0): Sequential(
            (0): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
            (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (mlps_pos): ModuleList(
          (0): Sequential(
            (0): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (mlps_out): ModuleList(
          (0): Sequential(
            (0): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
            (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
        (relu): ReLU()
      )
      (1): NeighborVoxelSAModuleMSG(
        (groupers): ModuleList(
          (0): VoxelQueryAndGrouping()
        )
        (mlps_in): ModuleList(
          (0): Sequential(
            (0): Conv1d(64, 32, kernel_size=(1,), stride=(1,), bias=False)
            (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (mlps_pos): ModuleList(
          (0): Sequential(
            (0): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (mlps_out): ModuleList(
          (0): Sequential(
            (0): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
            (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
        (relu): ReLU()
      )
      (2): NeighborVoxelSAModuleMSG(
        (groupers): ModuleList(
          (0): VoxelQueryAndGrouping()
        )
        (mlps_in): ModuleList(
          (0): Sequential(
            (0): Conv1d(64, 32, kernel_size=(1,), stride=(1,), bias=False)
            (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (mlps_pos): ModuleList(
          (0): Sequential(
            (0): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (mlps_out): ModuleList(
          (0): Sequential(
            (0): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
            (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
        (relu): ReLU()
      )
    )
    (shared_fc_layer): Sequential(
      (0): Linear(in_features=20736, out_features=256, bias=False)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Dropout(p=0.3, inplace=False)
      (4): Linear(in_features=256, out_features=256, bias=False)
      (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): ReLU(inplace=True)
    )
    (cls_fc_layers): Sequential(
      (0): Linear(in_features=256, out_features=256, bias=False)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.3, inplace=False)
      (4): Linear(in_features=256, out_features=256, bias=False)
      (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): ReLU()
    )
    (cls_pred_layer): Linear(in_features=256, out_features=1, bias=True)
    (reg_fc_layers): Sequential(
      (0): Linear(in_features=256, out_features=256, bias=False)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.3, inplace=False)
      (4): Linear(in_features=256, out_features=256, bias=False)
      (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): ReLU()
    )
    (reg_pred_layer): Linear(in_features=256, out_features=7, bias=True)
  )
)
```
报错如下:  
```
NaN or Inf found in input tensor.                      | 11/1856 [00:08<23:07,  1.33it/s, total_it=11]
NaN or Inf found in input tensor.
NaN or Inf found in input tensor.
maxoverlaps:(min=nan, max=nan)
ERROR: FG=0, BG=0
epochs:   0%|                                              | 0/80 [00:09<?, ?it/s, loss=nan, lr=0.001]
Traceback (most recent call last):
  File "/home/xilm/fuxian/SFD/tools/train.py", line 210, in <module>
    main()
  File "/home/xilm/fuxian/SFD/tools/train.py", line 165, in main
    train_model(
  File "/home/xilm/fuxian/SFD/tools/train_utils/train_utils.py", line 86, in train_model
    accumulated_iter = train_one_epoch(
  File "/home/xilm/fuxian/SFD/tools/train_utils/train_utils.py", line 38, in train_one_epoch
    loss, tb_dict, disp_dict = model_func(model, batch)
  File "/home/xilm/fuxian/SFD/pcdet/models/__init__.py", line 30, in model_func
    ret_dict, tb_dict, disp_dict = model(batch_dict)
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/xilm/fuxian/SFD/pcdet/models/detectors/voxel_rcnn.py", line 11, in forward
    batch_dict = cur_module(batch_dict)
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/xilm/fuxian/SFD/pcdet/models/roi_heads/voxelrcnn_head.py", line 227, in forward
    targets_dict = self.assign_targets(batch_dict)
  File "/home/xilm/fuxian/SFD/pcdet/models/roi_heads/roi_head_template.py", line 117, in assign_targets
    targets_dict = self.proposal_target_layer.forward(batch_dict)
  File "/home/xilm/fuxian/SFD/pcdet/models/roi_heads/target_assigner/proposal_target_layer.py", line 32, in forward
    batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
  File "/home/xilm/fuxian/SFD/pcdet/models/roi_heads/target_assigner/proposal_target_layer.py", line 107, in sample_rois_for_rcnn
    sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)
  File "/home/xilm/fuxian/SFD/pcdet/models/roi_heads/target_assigner/proposal_target_layer.py", line 159, in subsample_rois
    raise NotImplementedError
NotImplementedError
```  
原来真的是框架的问题，而不是sfd的问题。  
2. 准备把spconv换成1.1.1了  
## 0715:  
1. 把spconv换成2.x后，果然可以运行了，但是跑到91张的时候又报错了。。。
2.
```
home/xilm/fuxian/SFD/pcdet/models/roi_heads/sfd_head.py:305: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
/home/xilm/fuxian/SFD/pcdet/models/roi_heads/sfd_head.py:306: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
/home/xilm/fuxian/SFD/pcdet/models/roi_heads/sfd_head.py:307: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
/home/xilm/fuxian/SFD/pcdet/models/roi_heads/sfd_head.py:349: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  cur_roi_grid_coords = roi_grid_coords // cur_stride
```
这个错误只要把简单的除号改为torch.div(a, b, rounding_mode='trunc')即可  
3. 然后又报了一个错：
```
Original Traceback (most recent call last):
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/utils/data/_utils/worker.py", line 287, in _worker_loop
    data = fetcher.fetch(index)
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 49, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 49, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/xilm/fuxian/SFD/pcdet/datasets/kitti/kitti_dataset_sfd.py", line 389, in __getitem__
    points_pseudo = self.get_lidar_pseudo(sample_idx)
  File "/home/xilm/fuxian/SFD/pcdet/datasets/kitti/kitti_dataset_sfd.py", line 72, in get_lidar_pseudo
    assert lidar_pseudo_file.exists()
AssertionError
```
经过仔细检查才发现，training文件夹里depth_pseudo_rgbseguv_twise的.bin文件数居然不是7481，而是7470...，原因是我下载的时候曾经电脑卡死，中断了下载，开机后又继续下载，没想到居然少了10个文件。。。泪目
## 0718:  
1. 使用Spconv2.x训练可以无报错运行，前提是需要将pcdet版本修改，使其适应spconv2.x；
2. 跑了40轮，只用了一张显卡，没有修改学习率，获得的结果如下：
```
2023-07-18 05:43:38,688   INFO  *************** EPOCH 31 EVALUATION *****************
eval: 100%|█████████████████████| 3769/3769 [10:50<00:00,  5.80it/s, recall_0.3=(14017, 14015) / 14385]
2023-07-18 05:54:28,858   INFO  *************** Performance of EPOCH 31 *****************
2023-07-18 05:54:28,858   INFO  Run time per sample: 0.1715 second.
2023-07-18 05:54:28,858   INFO  Generate label finished(sec_per_example: 0.1725 second).
2023-07-18 05:54:28,858   INFO  recall_roi_0.3: 0.974418
2023-07-18 05:54:28,858   INFO  recall_rcnn_0.3: 0.974279
2023-07-18 05:54:28,859   INFO  recall_roi_0.5: 0.955509
2023-07-18 05:54:28,859   INFO  recall_rcnn_0.5: 0.963573
2023-07-18 05:54:28,859   INFO  recall_roi_0.7: 0.711992
2023-07-18 05:54:28,859   INFO  recall_rcnn_0.7: 0.841780
2023-07-18 05:54:28,860   INFO  Average predicted number of objects(3769 samples): 5.664
2023-07-18 05:54:41,203   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:97.3984, 90.2031, 89.7454
bev  AP:90.3420, 89.0605, 88.1222
3d   AP:89.4585, 85.0574, 78.9966
aos  AP:97.19, 89.90, 89.31
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.9678, 95.4585, 94.7446
bev  AP:95.8924, 91.9423, 89.3647
3d   AP:92.3953, 85.7993, 82.8595
aos  AP:98.74, 95.08, 94.20
Car AP@0.70, 0.50, 0.50:
bbox AP:97.3984, 90.2031, 89.7454
bev  AP:97.4658, 95.6201, 89.7378
3d   AP:97.3933, 90.1709, 89.7155
aos  AP:97.19, 89.90, 89.31
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.9678, 95.4585, 94.7446
bev  AP:98.9905, 97.2135, 95.0669
3d   AP:98.9662, 95.4487, 94.9493
aos  AP:98.74, 95.08, 94.20

2023-07-18 05:54:41,205   INFO  Result is save to /home/xilm/SFD/output/kitti_models/sfd/default/eval/eval_with_train/epoch_31/val
2023-07-18 05:54:41,205   INFO  ****************Evaluation done.*****************
2023-07-18 05:54:41,225   INFO  Epoch 31 has been evaluated
2023-07-18 05:54:41,243   INFO  ==> Loading parameters from checkpoint /home/xilm/SFD/output/kitti_models/sfd/default/ckpt/checkpoint_epoch_32.pth to GPU
2023-07-18 05:54:41,602   INFO  ==> Checkpoint trained from version: pcdet+0.1.0+03f83de
2023-07-18 05:54:41,619   INFO  ==> Done (loaded 514/514)
2023-07-18 05:54:41,624   INFO  *************** EPOCH 32 EVALUATION *****************
eval: 100%|█████████████████████| 3769/3769 [10:56<00:00,  5.74it/s, recall_0.3=(14053, 14042) / 14385]
2023-07-18 06:05:37,758   INFO  *************** Performance of EPOCH 32 *****************
2023-07-18 06:05:37,759   INFO  Run time per sample: 0.1732 second.
2023-07-18 06:05:37,759   INFO  Generate label finished(sec_per_example: 0.1741 second).
2023-07-18 06:05:37,759   INFO  recall_roi_0.3: 0.976920
2023-07-18 06:05:37,759   INFO  recall_rcnn_0.3: 0.976156
2023-07-18 06:05:37,759   INFO  recall_roi_0.5: 0.957247
2023-07-18 06:05:37,759   INFO  recall_rcnn_0.5: 0.965103
2023-07-18 06:05:37,759   INFO  recall_roi_0.7: 0.722419
2023-07-18 06:05:37,759   INFO  recall_rcnn_0.7: 0.831561
2023-07-18 06:05:37,761   INFO  Average predicted number of objects(3769 samples): 5.530
2023-07-18 06:05:41,206   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:98.9811, 90.1116, 89.7598
bev  AP:90.0488, 88.4313, 87.8269
3d   AP:89.0155, 84.2999, 78.8637
aos  AP:98.86, 89.90, 89.45
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:99.5633, 95.4700, 95.0967
bev  AP:96.0281, 91.3422, 89.0503
3d   AP:92.5179, 85.2054, 82.7775
aos  AP:99.44, 95.19, 94.70
Car AP@0.70, 0.50, 0.50:
bbox AP:98.9811, 90.1116, 89.7598
bev  AP:98.9779, 90.0382, 89.7078
3d   AP:98.9550, 90.0336, 89.6867
aos  AP:98.86, 89.90, 89.45
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:99.5633, 95.4700, 95.0967
bev  AP:99.5392, 95.4196, 95.1365
3d   AP:99.5321, 95.3876, 95.0650
aos  AP:99.44, 95.19, 94.70

2023-07-18 06:05:41,208   INFO  Result is save to /home/xilm/SFD/output/kitti_models/sfd/default/eval/eval_with_train/epoch_32/val
2023-07-18 06:05:41,210   INFO  ****************Evaluation done.*****************
2023-07-18 06:05:41,234   INFO  Epoch 32 has been evaluated
2023-07-18 06:05:41,251   INFO  ==> Loading parameters from checkpoint /home/xilm/SFD/output/kitti_models/sfd/default/ckpt/checkpoint_epoch_33.pth to GPU
2023-07-18 06:05:41,609   INFO  ==> Checkpoint trained from version: pcdet+0.1.0+03f83de
2023-07-18 06:05:41,625   INFO  ==> Done (loaded 514/514)
2023-07-18 06:05:41,630   INFO  *************** EPOCH 33 EVALUATION *****************
eval: 100%|█████████████████████| 3769/3769 [10:53<00:00,  5.77it/s, recall_0.3=(14048, 14045) / 14385]
2023-07-18 06:16:34,815   INFO  *************** Performance of EPOCH 33 *****************
2023-07-18 06:16:34,815   INFO  Run time per sample: 0.1724 second.
2023-07-18 06:16:34,815   INFO  Generate label finished(sec_per_example: 0.1733 second).
2023-07-18 06:16:34,816   INFO  recall_roi_0.3: 0.976573
2023-07-18 06:16:34,816   INFO  recall_rcnn_0.3: 0.976364
2023-07-18 06:16:34,816   INFO  recall_roi_0.5: 0.954675
2023-07-18 06:16:34,816   INFO  recall_rcnn_0.5: 0.964755
2023-07-18 06:16:34,816   INFO  recall_roi_0.7: 0.731665
2023-07-18 06:16:34,816   INFO  recall_rcnn_0.7: 0.828710
2023-07-18 06:16:34,817   INFO  Average predicted number of objects(3769 samples): 6.089
2023-07-18 06:16:38,343   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:97.5589, 95.4639, 89.8681
bev  AP:90.2354, 88.8931, 88.1680
3d   AP:89.3552, 84.0130, 78.9747
aos  AP:97.48, 95.11, 89.50
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:99.1674, 97.2917, 95.1366
bev  AP:95.8094, 91.7298, 89.2455
3d   AP:92.3585, 85.5382, 82.8433
aos  AP:99.10, 96.93, 94.65
Car AP@0.70, 0.50, 0.50:
bbox AP:97.5589, 95.4639, 89.8681
bev  AP:97.6906, 95.3309, 89.8382
3d   AP:97.6327, 95.3450, 89.8243
aos  AP:97.48, 95.11, 89.50
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:99.1674, 97.2917, 95.1366
bev  AP:99.1980, 97.4657, 95.1640
3d   AP:99.1821, 97.4194, 95.1118
aos  AP:99.10, 96.93, 94.65

2023-07-18 06:16:38,345   INFO  Result is save to /home/xilm/SFD/output/kitti_models/sfd/default/eval/eval_with_train/epoch_33/val
2023-07-18 06:16:38,347   INFO  ****************Evaluation done.*****************
2023-07-18 06:16:38,367   INFO  Epoch 33 has been evaluated
2023-07-18 06:16:38,384   INFO  ==> Loading parameters from checkpoint /home/xilm/SFD/output/kitti_models/sfd/default/ckpt/checkpoint_epoch_34.pth to GPU
2023-07-18 06:16:38,743   INFO  ==> Checkpoint trained from version: pcdet+0.1.0+03f83de
2023-07-18 06:16:38,760   INFO  ==> Done (loaded 514/514)
2023-07-18 06:16:38,765   INFO  *************** EPOCH 34 EVALUATION *****************
eval: 100%|█████████████████████| 3769/3769 [09:56<00:00,  6.32it/s, recall_0.3=(14081, 14081) / 14385]
2023-07-18 06:26:34,954   INFO  *************** Performance of EPOCH 34 *****************
2023-07-18 06:26:34,955   INFO  Run time per sample: 0.1572 second.
2023-07-18 06:26:34,955   INFO  Generate label finished(sec_per_example: 0.1582 second).
2023-07-18 06:26:34,955   INFO  recall_roi_0.3: 0.978867
2023-07-18 06:26:34,955   INFO  recall_rcnn_0.3: 0.978867
2023-07-18 06:26:34,955   INFO  recall_roi_0.5: 0.961766
2023-07-18 06:26:34,955   INFO  recall_rcnn_0.5: 0.968648
2023-07-18 06:26:34,955   INFO  recall_roi_0.7: 0.761279
2023-07-18 06:26:34,955   INFO  recall_rcnn_0.7: 0.852207
2023-07-18 06:26:34,957   INFO  Average predicted number of objects(3769 samples): 5.808
2023-07-18 06:26:38,455   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:98.2058, 90.2319, 89.8806
bev  AP:90.4085, 89.2351, 88.6465
3d   AP:89.7178, 85.0874, 84.9339
aos  AP:98.15, 90.07, 89.64
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:99.3320, 95.5544, 95.1660
bev  AP:96.5251, 92.1657, 91.5817
3d   AP:93.2377, 86.4474, 85.4349
aos  AP:99.27, 95.34, 94.84
Car AP@0.70, 0.50, 0.50:
bbox AP:98.2058, 90.2319, 89.8806
bev  AP:98.9245, 95.5942, 89.8394
3d   AP:98.9220, 95.5392, 89.8293
aos  AP:98.15, 90.07, 89.64
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:99.3320, 95.5544, 95.1660
bev  AP:99.5153, 97.3881, 95.1675
3d   AP:99.5146, 97.2101, 95.1131
aos  AP:99.27, 95.34, 94.84

2023-07-18 06:26:38,457   INFO  Result is save to /home/xilm/SFD/output/kitti_models/sfd/default/eval/eval_with_train/epoch_34/val
2023-07-18 06:26:38,459   INFO  ****************Evaluation done.*****************
2023-07-18 06:26:38,477   INFO  Epoch 34 has been evaluated
2023-07-18 06:26:38,493   INFO  ==> Loading parameters from checkpoint /home/xilm/SFD/output/kitti_models/sfd/default/ckpt/checkpoint_epoch_35.pth to GPU
2023-07-18 06:26:38,856   INFO  ==> Checkpoint trained from version: pcdet+0.1.0+03f83de
2023-07-18 06:26:38,872   INFO  ==> Done (loaded 514/514)
2023-07-18 06:26:38,877   INFO  *************** EPOCH 35 EVALUATION *****************
eval: 100%|█████████████████████| 3769/3769 [10:28<00:00,  6.00it/s, recall_0.3=(14086, 14072) / 14385]
2023-07-18 06:37:07,150   INFO  *************** Performance of EPOCH 35 *****************
2023-07-18 06:37:07,151   INFO  Run time per sample: 0.1657 second.
2023-07-18 06:37:07,151   INFO  Generate label finished(sec_per_example: 0.1667 second).
2023-07-18 06:37:07,151   INFO  recall_roi_0.3: 0.979214
2023-07-18 06:37:07,151   INFO  recall_rcnn_0.3: 0.978241
2023-07-18 06:37:07,151   INFO  recall_roi_0.5: 0.963712
2023-07-18 06:37:07,151   INFO  recall_rcnn_0.5: 0.969691
2023-07-18 06:37:07,151   INFO  recall_roi_0.7: 0.757247
2023-07-18 06:37:07,151   INFO  recall_rcnn_0.7: 0.853041
2023-07-18 06:37:07,152   INFO  Average predicted number of objects(3769 samples): 5.879
2023-07-18 06:37:10,660   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:97.6363, 95.9270, 89.9411
bev  AP:90.3385, 89.0230, 88.3678
3d   AP:89.9413, 86.5462, 85.0367
aos  AP:97.45, 95.64, 89.67
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:99.2405, 97.5780, 95.3318
bev  AP:96.2044, 92.1089, 91.3605
3d   AP:95.3178, 88.2680, 85.5414
aos  AP:99.04, 97.29, 94.98
Car AP@0.70, 0.50, 0.50:
bbox AP:97.6363, 95.9270, 89.9411
bev  AP:97.7073, 95.8688, 89.8816
3d   AP:97.6769, 95.8148, 89.8727
aos  AP:97.45, 95.64, 89.67
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:99.2405, 97.5780, 95.3318
bev  AP:99.2504, 97.6701, 95.3092
3d   AP:99.2421, 97.6012, 95.2713
aos  AP:99.04, 97.29, 94.98

2023-07-18 06:37:10,662   INFO  Result is save to /home/xilm/SFD/output/kitti_models/sfd/default/eval/eval_with_train/epoch_35/val
2023-07-18 06:37:10,664   INFO  ****************Evaluation done.*****************
2023-07-18 06:37:10,682   INFO  Epoch 35 has been evaluated
2023-07-18 06:37:10,699   INFO  ==> Loading parameters from checkpoint /home/xilm/SFD/output/kitti_models/sfd/default/ckpt/checkpoint_epoch_36.pth to GPU
2023-07-18 06:37:11,058   INFO  ==> Checkpoint trained from version: pcdet+0.1.0+03f83de
2023-07-18 06:37:11,074   INFO  ==> Done (loaded 514/514)
2023-07-18 06:37:11,079   INFO  *************** EPOCH 36 EVALUATION *****************
eval: 100%|█████████████████████| 3769/3769 [10:20<00:00,  6.08it/s, recall_0.3=(14070, 14063) / 14385]
2023-07-18 06:47:31,263   INFO  *************** Performance of EPOCH 36 *****************
2023-07-18 06:47:31,264   INFO  Run time per sample: 0.1636 second.
2023-07-18 06:47:31,264   INFO  Generate label finished(sec_per_example: 0.1645 second).
2023-07-18 06:47:31,264   INFO  recall_roi_0.3: 0.978102
2023-07-18 06:47:31,264   INFO  recall_rcnn_0.3: 0.977616
2023-07-18 06:47:31,264   INFO  recall_roi_0.5: 0.963990
2023-07-18 06:47:31,264   INFO  recall_rcnn_0.5: 0.969482
2023-07-18 06:47:31,264   INFO  recall_roi_0.7: 0.774974
2023-07-18 06:47:31,264   INFO  recall_rcnn_0.7: 0.863538
2023-07-18 06:47:31,266   INFO  Average predicted number of objects(3769 samples): 5.533
2023-07-18 06:47:34,751   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:98.2307, 96.0440, 89.9152
bev  AP:97.3932, 89.3019, 88.7341
3d   AP:89.7134, 87.3679, 85.3057
aos  AP:98.10, 95.77, 89.67
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:99.3477, 97.6741, 95.3517
bev  AP:98.5487, 94.1753, 91.8271
3d   AP:95.3794, 88.5245, 85.8962
aos  AP:99.20, 97.41, 95.02
Car AP@0.70, 0.50, 0.50:
bbox AP:98.2307, 96.0440, 89.9152
bev  AP:98.2922, 96.0082, 89.8903
3d   AP:98.2494, 95.9474, 89.8690
aos  AP:98.10, 95.77, 89.67
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:99.3477, 97.6741, 95.3517
bev  AP:99.3494, 97.7677, 95.3790
3d   AP:99.3376, 97.7029, 95.3238
aos  AP:99.20, 97.41, 95.02

2023-07-18 06:47:34,754   INFO  Result is save to /home/xilm/SFD/output/kitti_models/sfd/default/eval/eval_with_train/epoch_36/val
2023-07-18 06:47:34,756   INFO  ****************Evaluation done.*****************
2023-07-18 06:47:34,776   INFO  Epoch 36 has been evaluated
2023-07-18 06:47:34,793   INFO  ==> Loading parameters from checkpoint /home/xilm/SFD/output/kitti_models/sfd/default/ckpt/checkpoint_epoch_37.pth to GPU
2023-07-18 06:47:35,154   INFO  ==> Checkpoint trained from version: pcdet+0.1.0+03f83de
2023-07-18 06:47:35,170   INFO  ==> Done (loaded 514/514)
2023-07-18 06:47:35,176   INFO  *************** EPOCH 37 EVALUATION *****************
eval: 100%|█████████████████████| 3769/3769 [10:07<00:00,  6.20it/s, recall_0.3=(14076, 14063) / 14385]
2023-07-18 06:57:42,613   INFO  *************** Performance of EPOCH 37 *****************
2023-07-18 06:57:42,614   INFO  Run time per sample: 0.1602 second.
2023-07-18 06:57:42,614   INFO  Generate label finished(sec_per_example: 0.1612 second).
2023-07-18 06:57:42,614   INFO  recall_roi_0.3: 0.978519
2023-07-18 06:57:42,615   INFO  recall_rcnn_0.3: 0.977616
2023-07-18 06:57:42,615   INFO  recall_roi_0.5: 0.964199
2023-07-18 06:57:42,615   INFO  recall_rcnn_0.5: 0.969760
2023-07-18 06:57:42,615   INFO  recall_roi_0.7: 0.782134
2023-07-18 06:57:42,615   INFO  recall_rcnn_0.7: 0.863260
2023-07-18 06:57:42,616   INFO  Average predicted number of objects(3769 samples): 5.910
2023-07-18 06:57:46,132   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:96.2894, 95.8326, 89.9025
bev  AP:95.7197, 89.3432, 88.8402
3d   AP:89.7937, 87.0789, 85.1095
aos  AP:96.18, 95.58, 89.64
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.8520, 97.4566, 95.2888
bev  AP:98.2702, 94.2782, 91.9344
3d   AP:95.4605, 88.3471, 85.8777
aos  AP:98.73, 97.20, 94.94
Car AP@0.70, 0.50, 0.50:
bbox AP:96.2894, 95.8326, 89.9025
bev  AP:96.3772, 95.7836, 96.0212
3d   AP:96.3376, 95.7334, 89.8514
aos  AP:96.18, 95.58, 89.64
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.8520, 97.4566, 95.2888
bev  AP:98.8645, 97.5681, 97.0463
3d   AP:98.8536, 97.5229, 95.2474
aos  AP:98.73, 97.20, 94.94

2023-07-18 06:57:46,134   INFO  Result is save to /home/xilm/SFD/output/kitti_models/sfd/default/eval/eval_with_train/epoch_37/val
2023-07-18 06:57:46,136   INFO  ****************Evaluation done.*****************
2023-07-18 06:57:46,157   INFO  Epoch 37 has been evaluated
2023-07-18 06:57:46,173   INFO  ==> Loading parameters from checkpoint /home/xilm/SFD/output/kitti_models/sfd/default/ckpt/checkpoint_epoch_38.pth to GPU
2023-07-18 06:57:46,533   INFO  ==> Checkpoint trained from version: pcdet+0.1.0+03f83de
2023-07-18 06:57:46,549   INFO  ==> Done (loaded 514/514)
2023-07-18 06:57:46,555   INFO  *************** EPOCH 38 EVALUATION *****************
eval: 100%|█████████████████████| 3769/3769 [10:06<00:00,  6.21it/s, recall_0.3=(14100, 14096) / 14385]
2023-07-18 07:07:53,024   INFO  *************** Performance of EPOCH 38 *****************
2023-07-18 07:07:53,025   INFO  Run time per sample: 0.1599 second.
2023-07-18 07:07:53,025   INFO  Generate label finished(sec_per_example: 0.1609 second).
2023-07-18 07:07:53,025   INFO  recall_roi_0.3: 0.980188
2023-07-18 07:07:53,025   INFO  recall_rcnn_0.3: 0.979910
2023-07-18 07:07:53,025   INFO  recall_roi_0.5: 0.965172
2023-07-18 07:07:53,025   INFO  recall_rcnn_0.5: 0.971011
2023-07-18 07:07:53,025   INFO  recall_roi_0.7: 0.782343
2023-07-18 07:07:53,025   INFO  recall_rcnn_0.7: 0.858325
2023-07-18 07:07:53,027   INFO  Average predicted number of objects(3769 samples): 5.670
2023-07-18 07:07:56,524   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:96.8965, 95.8547, 89.8874
bev  AP:96.1996, 89.1923, 88.7180
3d   AP:89.7754, 87.3136, 85.3269
aos  AP:96.82, 95.66, 89.70
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.9811, 97.4995, 95.3234
bev  AP:98.1962, 94.1317, 91.8587
3d   AP:95.0839, 88.4505, 85.9941
aos  AP:98.90, 97.31, 95.06
Car AP@0.70, 0.50, 0.50:
bbox AP:96.8965, 95.8547, 89.8874
bev  AP:97.0207, 95.8068, 89.8386
3d   AP:96.9664, 95.7539, 89.8232
aos  AP:96.82, 95.66, 89.70
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.9811, 97.4995, 95.3234
bev  AP:99.0016, 97.6312, 95.3123
3d   AP:98.9867, 97.5946, 95.2705
aos  AP:98.90, 97.31, 95.06

2023-07-18 07:07:56,526   INFO  Result is save to /home/xilm/SFD/output/kitti_models/sfd/default/eval/eval_with_train/epoch_38/val
2023-07-18 07:07:56,528   INFO  ****************Evaluation done.*****************
2023-07-18 07:07:56,549   INFO  Epoch 38 has been evaluated
2023-07-18 07:07:56,565   INFO  ==> Loading parameters from checkpoint /home/xilm/SFD/output/kitti_models/sfd/default/ckpt/checkpoint_epoch_39.pth to GPU
2023-07-18 07:07:56,924   INFO  ==> Checkpoint trained from version: pcdet+0.1.0+03f83de
2023-07-18 07:07:56,940   INFO  ==> Done (loaded 514/514)
2023-07-18 07:07:56,946   INFO  *************** EPOCH 39 EVALUATION *****************
eval: 100%|█████████████████████| 3769/3769 [10:07<00:00,  6.20it/s, recall_0.3=(14090, 14084) / 14385]
2023-07-18 07:18:04,538   INFO  *************** Performance of EPOCH 39 *****************
2023-07-18 07:18:04,539   INFO  Run time per sample: 0.1602 second.
2023-07-18 07:18:04,539   INFO  Generate label finished(sec_per_example: 0.1612 second).
2023-07-18 07:18:04,539   INFO  recall_roi_0.3: 0.979493
2023-07-18 07:18:04,539   INFO  recall_rcnn_0.3: 0.979075
2023-07-18 07:18:04,539   INFO  recall_roi_0.5: 0.965242
2023-07-18 07:18:04,539   INFO  recall_rcnn_0.5: 0.971151
2023-07-18 07:18:04,539   INFO  recall_roi_0.7: 0.786444
2023-07-18 07:18:04,539   INFO  recall_rcnn_0.7: 0.860619
2023-07-18 07:18:04,541   INFO  Average predicted number of objects(3769 samples): 5.833
2023-07-18 07:18:08,038   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:96.8683, 95.9801, 89.8967
bev  AP:96.1489, 89.1522, 88.7044
3d   AP:89.8555, 87.2735, 85.3435
aos  AP:96.78, 95.78, 89.72
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.9912, 97.1957, 95.2927
bev  AP:98.1404, 93.8619, 91.7868
3d   AP:95.1417, 88.6022, 85.9977
aos  AP:98.90, 97.00, 95.04
Car AP@0.70, 0.50, 0.50:
bbox AP:96.8683, 95.9801, 89.8967
bev  AP:96.9686, 95.7999, 89.8622
3d   AP:96.9355, 95.9386, 89.8474
aos  AP:96.78, 95.78, 89.72
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.9912, 97.1957, 95.2927
bev  AP:99.0062, 97.5715, 95.3012
3d   AP:98.9972, 97.5539, 95.2685
aos  AP:98.90, 97.00, 95.04

2023-07-18 07:18:08,044   INFO  Result is save to /home/xilm/SFD/output/kitti_models/sfd/default/eval/eval_with_train/epoch_39/val
2023-07-18 07:18:08,046   INFO  ****************Evaluation done.*****************
2023-07-18 07:18:08,067   INFO  Epoch 39 has been evaluated
2023-07-18 07:18:08,084   INFO  ==> Loading parameters from checkpoint /home/xilm/SFD/output/kitti_models/sfd/default/ckpt/checkpoint_epoch_40.pth to GPU
2023-07-18 07:18:08,430   INFO  ==> Checkpoint trained from version: pcdet+0.1.0+03f83de
2023-07-18 07:18:08,446   INFO  ==> Done (loaded 514/514)
2023-07-18 07:18:08,452   INFO  *************** EPOCH 40 EVALUATION *****************
eval: 100%|█████████████████████| 3769/3769 [10:07<00:00,  6.21it/s, recall_0.3=(14095, 14069) / 14385]
2023-07-18 07:28:15,610   INFO  *************** Performance of EPOCH 40 *****************
2023-07-18 07:28:15,611   INFO  Run time per sample: 0.1602 second.
2023-07-18 07:28:15,611   INFO  Generate label finished(sec_per_example: 0.1611 second).
2023-07-18 07:28:15,611   INFO  recall_roi_0.3: 0.979840
2023-07-18 07:28:15,611   INFO  recall_rcnn_0.3: 0.978033
2023-07-18 07:28:15,611   INFO  recall_roi_0.5: 0.965798
2023-07-18 07:28:15,611   INFO  recall_rcnn_0.5: 0.970247
2023-07-18 07:28:15,611   INFO  recall_roi_0.7: 0.785958
2023-07-18 07:28:15,611   INFO  recall_rcnn_0.7: 0.857143
2023-07-18 07:28:15,612   INFO  Average predicted number of objects(3769 samples): 5.971
2023-07-18 07:28:19,115   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:96.9514, 95.6195, 89.8470
bev  AP:96.1821, 89.0908, 88.5941
3d   AP:89.6907, 87.0173, 84.9417
aos  AP:96.87, 95.42, 89.66
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:99.0013, 97.3292, 95.2137
bev  AP:98.0156, 93.6072, 91.6009
3d   AP:95.2770, 88.3460, 85.7323
aos  AP:98.91, 97.13, 94.95
Car AP@0.70, 0.50, 0.50:
bbox AP:96.9514, 95.6195, 89.8470
bev  AP:97.0581, 95.5724, 89.7876
3d   AP:97.0225, 95.5254, 89.7720
aos  AP:96.87, 95.42, 89.66
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:99.0013, 97.3292, 95.2137
bev  AP:99.0165, 97.4593, 95.1950
3d   AP:99.0067, 97.4062, 95.1521
aos  AP:98.91, 97.13, 94.95

2023-07-18 07:28:19,118   INFO  Result is save to /home/xilm/SFD/output/kitti_models/sfd/default/eval/eval_with_train/epoch_40/val
2023-07-18 07:28:19,120   INFO  ****************Evaluation done.*****************
2023-07-18 07:28:19,138   INFO  Epoch 40 has been evaluated
```
整个训练过程跑了33个半小时。  
## 0722：  
1. 使用demo.py跑了一张效果图：
![](https://github.com/XxxuLimei/SFD-re/blob/main/doc/snapshot.png)

## 0810:  
```
import cv2
import numpy as np
import torch
from torch.nn import functional as F

videoCapture = cv2.VideoCapture('/media/SSD2/personal/xulimei/esrgan/test_gamevideo/跑酷-12帧.mp4')
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (640, 360)
videoWriter = cv2.VideoWriter('/media/SSD2/personal/xulimei/esrgan/test_gamevideo/跑酷-12帧_resize.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    if not ret:
        break
    frame_tensor = np.array(frame)
    frame_tensor = torch.from_numpy(frame_tensor).unsqueeze(0).float()
    frame_tensor = frame_tensor.permute(0, 3, 1, 2)

    # resized_frame = cv2.resize(frame, size)
    out = F.interpolate(frame_tensor, size=(640, 360), mode='bilinear', align_corners=False)
    out = out.squeeze(0).byte().numpy().transpose((1, 2, 0))
    m = videoWriter.write(out)
    print(m)

videoCapture.release()
videoWriter.release()
```
