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
