## 0912:   
1. 最近在学习DLA的网络结构：(dla-34)  
```
DLA(
  (base_layer): Sequential(
    (0): Conv2d(3, 16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (level0): Sequential(
    (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (level1): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (level2): Tree(
    (tree1): BasicBlock(
      (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (tree2): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (root): Root(
      (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (downsample): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (project): Sequential(
      (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (level3): Tree(
    (tree1): Tree(
      (tree1): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (tree2): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (root): Root(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (downsample): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (project): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (tree2): Tree(
      (tree1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (tree2): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (root): Root(
        (conv): Conv2d(448, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (downsample): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (project): Sequential(
      (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (level4): Tree(
    (tree1): Tree(
      (tree1): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (tree2): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (root): Root(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (downsample): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (project): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (tree2): Tree(
      (tree1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (tree2): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (root): Root(
        (conv): Conv2d(896, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (downsample): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (project): Sequential(
      (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (level5): Tree(
    (tree1): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (tree2): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (root): Root(
      (conv): Conv2d(1280, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (downsample): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (project): Sequential(
      (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AvgPool2d(kernel_size=1, stride=1, padding=0)
  (fc): Conv2d(512, 200, kernel_size=(1, 1), stride=(1, 1))
)
```  
模型大小：  
```
                    module name   input shape output shape      params memory(MB)           MAdd         Flops  MemRead(B)  MemWrite(B) duration[%]   MemR+W(B)
0                  base_layer.0     3  32  32   16  32  32      2352.0       0.06    4,800,512.0   2,408,448.0     21696.0      65536.0      14.15%     87232.0
1                  base_layer.1    16  32  32   16  32  32        32.0       0.06       65,536.0      32,768.0     65664.0      65536.0       1.00%    131200.0
2                  base_layer.2    16  32  32   16  32  32         0.0       0.06       16,384.0      16,384.0     65536.0      65536.0       0.36%    131072.0
3                      level0.0    16  32  32   16  32  32      2304.0       0.06    4,702,208.0   2,359,296.0     74752.0      65536.0       3.58%    140288.0
4                      level0.1    16  32  32   16  32  32        32.0       0.06       65,536.0      32,768.0     65664.0      65536.0       0.43%    131200.0
5                      level0.2    16  32  32   16  32  32         0.0       0.06       16,384.0      16,384.0     65536.0      65536.0       0.12%    131072.0
6                      level1.0    16  32  32   32  16  16      4608.0       0.03    2,351,104.0   1,179,648.0     83968.0      32768.0       2.59%    116736.0
7                      level1.1    32  16  16   32  16  16        64.0       0.03       32,768.0      16,384.0     33024.0      32768.0       0.27%     65792.0
8                      level1.2    32  16  16   32  16  16         0.0       0.03        8,192.0       8,192.0     32768.0      32768.0       0.08%     65536.0
9            level2.tree1.conv1    32  16  16   64   8   8     18432.0       0.02    2,355,200.0   1,179,648.0    106496.0      16384.0       1.00%    122880.0
10             level2.tree1.bn1    64   8   8   64   8   8       128.0       0.02       16,384.0       8,192.0     16896.0      16384.0       0.22%     33280.0
11            level2.tree1.relu    64   8   8   64   8   8         0.0       0.02        4,096.0       4,096.0     16384.0      16384.0       0.06%     32768.0
12           level2.tree1.conv2    64   8   8   64   8   8     36864.0       0.02    4,714,496.0   2,359,296.0    163840.0      16384.0       1.34%    180224.0
13             level2.tree1.bn2    64   8   8   64   8   8       128.0       0.02       16,384.0       8,192.0     16896.0      16384.0       0.21%     33280.0
14           level2.tree2.conv1    64   8   8   64   8   8     36864.0       0.02    4,714,496.0   2,359,296.0    163840.0      16384.0       1.96%    180224.0
15             level2.tree2.bn1    64   8   8   64   8   8       128.0       0.02       16,384.0       8,192.0     16896.0      16384.0       0.73%     33280.0
16            level2.tree2.relu    64   8   8   64   8   8         0.0       0.02        4,096.0       4,096.0     16384.0      16384.0       0.06%     32768.0
17           level2.tree2.conv2    64   8   8   64   8   8     36864.0       0.02    4,714,496.0   2,359,296.0    163840.0      16384.0       1.47%    180224.0
18             level2.tree2.bn2    64   8   8   64   8   8       128.0       0.02       16,384.0       8,192.0     16896.0      16384.0       0.24%     33280.0
19             level2.root.conv   128   8   8   64   8   8      8192.0       0.02    1,044,480.0     524,288.0     65536.0      16384.0       1.74%     81920.0
20               level2.root.bn    64   8   8   64   8   8       128.0       0.02       16,384.0       8,192.0     16896.0      16384.0       0.25%     33280.0
21             level2.root.relu    64   8   8   64   8   8         0.0       0.02        4,096.0       4,096.0     16384.0      16384.0       0.08%     32768.0
22            level2.downsample    32  16  16   32   8   8         0.0       0.01        6,144.0       8,192.0     32768.0       8192.0       2.26%     40960.0
23             level2.project.0    32   8   8   64   8   8      2048.0       0.02      258,048.0     131,072.0     16384.0      16384.0       1.35%     32768.0
24             level2.project.1    64   8   8   64   8   8       128.0       0.02       16,384.0       8,192.0     16896.0      16384.0       0.23%     33280.0
25     level3.tree1.tree1.conv1    64   8   8  128   4   4     73728.0       0.01    2,357,248.0   1,179,648.0    311296.0       8192.0       0.91%    319488.0
26       level3.tree1.tree1.bn1   128   4   4  128   4   4       256.0       0.01        8,192.0       4,096.0      9216.0       8192.0       0.22%     17408.0
27      level3.tree1.tree1.relu   128   4   4  128   4   4         0.0       0.01        2,048.0       2,048.0      8192.0       8192.0       0.08%     16384.0
28     level3.tree1.tree1.conv2   128   4   4  128   4   4    147456.0       0.01    4,716,544.0   2,359,296.0    598016.0       8192.0       1.33%    606208.0
29       level3.tree1.tree1.bn2   128   4   4  128   4   4       256.0       0.01        8,192.0       4,096.0      9216.0       8192.0       0.24%     17408.0
30     level3.tree1.tree2.conv1   128   4   4  128   4   4    147456.0       0.01    4,716,544.0   2,359,296.0    598016.0       8192.0       1.45%    606208.0
31       level3.tree1.tree2.bn1   128   4   4  128   4   4       256.0       0.01        8,192.0       4,096.0      9216.0       8192.0       0.23%     17408.0
32      level3.tree1.tree2.relu   128   4   4  128   4   4         0.0       0.01        2,048.0       2,048.0      8192.0       8192.0       0.07%     16384.0
33     level3.tree1.tree2.conv2   128   4   4  128   4   4    147456.0       0.01    4,716,544.0   2,359,296.0    598016.0       8192.0       1.35%    606208.0
34       level3.tree1.tree2.bn2   128   4   4  128   4   4       256.0       0.01        8,192.0       4,096.0      9216.0       8192.0       0.23%     17408.0
35       level3.tree1.root.conv   256   4   4  128   4   4     32768.0       0.01    1,046,528.0     524,288.0    147456.0       8192.0       0.77%    155648.0
36         level3.tree1.root.bn   128   4   4  128   4   4       256.0       0.01        8,192.0       4,096.0      9216.0       8192.0       0.22%     17408.0
37       level3.tree1.root.relu   128   4   4  128   4   4         0.0       0.01        2,048.0       2,048.0      8192.0       8192.0       0.08%     16384.0
38      level3.tree1.downsample    64   8   8   64   4   4         0.0       0.00        3,072.0       4,096.0     16384.0       4096.0       0.39%     20480.0
39       level3.tree1.project.0    64   4   4  128   4   4      8192.0       0.01      260,096.0     131,072.0     36864.0       8192.0       0.38%     45056.0
40       level3.tree1.project.1   128   4   4  128   4   4       256.0       0.01        8,192.0       4,096.0      9216.0       8192.0       0.22%     17408.0
41     level3.tree2.tree1.conv1   128   4   4  128   4   4    147456.0       0.01    4,716,544.0   2,359,296.0    598016.0       8192.0       1.27%    606208.0
42       level3.tree2.tree1.bn1   128   4   4  128   4   4       256.0       0.01        8,192.0       4,096.0      9216.0       8192.0       0.23%     17408.0
43      level3.tree2.tree1.relu   128   4   4  128   4   4         0.0       0.01        2,048.0       2,048.0      8192.0       8192.0       0.06%     16384.0
44     level3.tree2.tree1.conv2   128   4   4  128   4   4    147456.0       0.01    4,716,544.0   2,359,296.0    598016.0       8192.0       1.25%    606208.0
45       level3.tree2.tree1.bn2   128   4   4  128   4   4       256.0       0.01        8,192.0       4,096.0      9216.0       8192.0       0.22%     17408.0
46     level3.tree2.tree2.conv1   128   4   4  128   4   4    147456.0       0.01    4,716,544.0   2,359,296.0    598016.0       8192.0       1.23%    606208.0
47       level3.tree2.tree2.bn1   128   4   4  128   4   4       256.0       0.01        8,192.0       4,096.0      9216.0       8192.0       0.20%     17408.0
48      level3.tree2.tree2.relu   128   4   4  128   4   4         0.0       0.01        2,048.0       2,048.0      8192.0       8192.0       0.06%     16384.0
49     level3.tree2.tree2.conv2   128   4   4  128   4   4    147456.0       0.01    4,716,544.0   2,359,296.0    598016.0       8192.0       1.22%    606208.0
50       level3.tree2.tree2.bn2   128   4   4  128   4   4       256.0       0.01        8,192.0       4,096.0      9216.0       8192.0       0.20%     17408.0
51       level3.tree2.root.conv   448   4   4  128   4   4     57344.0       0.01    1,832,960.0     917,504.0    258048.0       8192.0       0.63%    266240.0
52         level3.tree2.root.bn   128   4   4  128   4   4       256.0       0.01        8,192.0       4,096.0      9216.0       8192.0       0.21%     17408.0
53       level3.tree2.root.relu   128   4   4  128   4   4         0.0       0.01        2,048.0       2,048.0      8192.0       8192.0       0.06%     16384.0
54            level3.downsample    64   8   8   64   4   4         0.0       0.00        3,072.0       4,096.0     16384.0       4096.0       0.92%     20480.0
55             level3.project.0    64   4   4  128   4   4      8192.0       0.01      260,096.0     131,072.0     36864.0       8192.0       0.94%     45056.0
56             level3.project.1   128   4   4  128   4   4       256.0       0.01        8,192.0       4,096.0      9216.0       8192.0       0.21%     17408.0
57     level4.tree1.tree1.conv1   128   4   4  256   2   2    294912.0       0.00    2,358,272.0   1,179,648.0   1187840.0       4096.0       1.37%   1191936.0
58       level4.tree1.tree1.bn1   256   2   2  256   2   2       512.0       0.00        4,096.0       2,048.0      6144.0       4096.0       0.25%     10240.0
59      level4.tree1.tree1.relu   256   2   2  256   2   2         0.0       0.00        1,024.0       1,024.0      4096.0       4096.0       0.06%      8192.0
60     level4.tree1.tree1.conv2   256   2   2  256   2   2    589824.0       0.00    4,717,568.0   2,359,296.0   2363392.0       4096.0       1.98%   2367488.0
61       level4.tree1.tree1.bn2   256   2   2  256   2   2       512.0       0.00        4,096.0       2,048.0      6144.0       4096.0       0.25%     10240.0
62     level4.tree1.tree2.conv1   256   2   2  256   2   2    589824.0       0.00    4,717,568.0   2,359,296.0   2363392.0       4096.0       2.30%   2367488.0
63       level4.tree1.tree2.bn1   256   2   2  256   2   2       512.0       0.00        4,096.0       2,048.0      6144.0       4096.0       0.26%     10240.0
64      level4.tree1.tree2.relu   256   2   2  256   2   2         0.0       0.00        1,024.0       1,024.0      4096.0       4096.0       0.06%      8192.0
65     level4.tree1.tree2.conv2   256   2   2  256   2   2    589824.0       0.00    4,717,568.0   2,359,296.0   2363392.0       4096.0       2.22%   2367488.0
66       level4.tree1.tree2.bn2   256   2   2  256   2   2       512.0       0.00        4,096.0       2,048.0      6144.0       4096.0       0.27%     10240.0
67       level4.tree1.root.conv   512   2   2  256   2   2    131072.0       0.00    1,047,552.0     524,288.0    532480.0       4096.0       1.42%    536576.0
68         level4.tree1.root.bn   256   2   2  256   2   2       512.0       0.00        4,096.0       2,048.0      6144.0       4096.0       0.25%     10240.0
69       level4.tree1.root.relu   256   2   2  256   2   2         0.0       0.00        1,024.0       1,024.0      4096.0       4096.0       0.07%      8192.0
70      level4.tree1.downsample   128   4   4  128   2   2         0.0       0.00        1,536.0       2,048.0      8192.0       2048.0       0.32%     10240.0
71       level4.tree1.project.0   128   2   2  256   2   2     32768.0       0.00      261,120.0     131,072.0    133120.0       4096.0       0.72%    137216.0
72       level4.tree1.project.1   256   2   2  256   2   2       512.0       0.00        4,096.0       2,048.0      6144.0       4096.0       0.23%     10240.0
73     level4.tree2.tree1.conv1   256   2   2  256   2   2    589824.0       0.00    4,717,568.0   2,359,296.0   2363392.0       4096.0       2.28%   2367488.0
74       level4.tree2.tree1.bn1   256   2   2  256   2   2       512.0       0.00        4,096.0       2,048.0      6144.0       4096.0       0.26%     10240.0
75      level4.tree2.tree1.relu   256   2   2  256   2   2         0.0       0.00        1,024.0       1,024.0      4096.0       4096.0       0.06%      8192.0
76     level4.tree2.tree1.conv2   256   2   2  256   2   2    589824.0       0.00    4,717,568.0   2,359,296.0   2363392.0       4096.0       2.14%   2367488.0
77       level4.tree2.tree1.bn2   256   2   2  256   2   2       512.0       0.00        4,096.0       2,048.0      6144.0       4096.0       0.24%     10240.0
78     level4.tree2.tree2.conv1   256   2   2  256   2   2    589824.0       0.00    4,717,568.0   2,359,296.0   2363392.0       4096.0       2.21%   2367488.0
79       level4.tree2.tree2.bn1   256   2   2  256   2   2       512.0       0.00        4,096.0       2,048.0      6144.0       4096.0       0.23%     10240.0
80      level4.tree2.tree2.relu   256   2   2  256   2   2         0.0       0.00        1,024.0       1,024.0      4096.0       4096.0       0.06%      8192.0
81     level4.tree2.tree2.conv2   256   2   2  256   2   2    589824.0       0.00    4,717,568.0   2,359,296.0   2363392.0       4096.0       2.00%   2367488.0
82       level4.tree2.tree2.bn2   256   2   2  256   2   2       512.0       0.00        4,096.0       2,048.0      6144.0       4096.0       0.22%     10240.0
83       level4.tree2.root.conv   896   2   2  256   2   2    229376.0       0.00    1,833,984.0     917,504.0    931840.0       4096.0       1.08%    935936.0
84         level4.tree2.root.bn   256   2   2  256   2   2       512.0       0.00        4,096.0       2,048.0      6144.0       4096.0       0.25%     10240.0
85       level4.tree2.root.relu   256   2   2  256   2   2         0.0       0.00        1,024.0       1,024.0      4096.0       4096.0       0.07%      8192.0
86            level4.downsample   128   4   4  128   2   2         0.0       0.00        1,536.0       2,048.0      8192.0       2048.0       0.78%     10240.0
87             level4.project.0   128   2   2  256   2   2     32768.0       0.00      261,120.0     131,072.0    133120.0       4096.0       0.82%    137216.0
88             level4.project.1   256   2   2  256   2   2       512.0       0.00        4,096.0       2,048.0      6144.0       4096.0       0.23%     10240.0
89           level5.tree1.conv1   256   2   2  512   1   1   1179648.0       0.00    2,358,784.0   1,179,648.0   4722688.0       2048.0       2.90%   4724736.0
90             level5.tree1.bn1   512   1   1  512   1   1      1024.0       0.00        2,048.0       1,024.0      6144.0       2048.0       0.24%      8192.0
91            level5.tree1.relu   512   1   1  512   1   1         0.0       0.00          512.0         512.0      2048.0       2048.0       0.06%      4096.0
92           level5.tree1.conv2   512   1   1  512   1   1   2359296.0       0.00    4,718,080.0   2,359,296.0   9439232.0       2048.0       4.64%   9441280.0
93             level5.tree1.bn2   512   1   1  512   1   1      1024.0       0.00        2,048.0       1,024.0      6144.0       2048.0       0.22%      8192.0
94           level5.tree2.conv1   512   1   1  512   1   1   2359296.0       0.00    4,718,080.0   2,359,296.0   9439232.0       2048.0       3.85%   9441280.0
95             level5.tree2.bn1   512   1   1  512   1   1      1024.0       0.00        2,048.0       1,024.0      6144.0       2048.0       0.25%      8192.0
96            level5.tree2.relu   512   1   1  512   1   1         0.0       0.00          512.0         512.0      2048.0       2048.0       0.06%      4096.0
97           level5.tree2.conv2   512   1   1  512   1   1   2359296.0       0.00    4,718,080.0   2,359,296.0   9439232.0       2048.0       3.74%   9441280.0
98             level5.tree2.bn2   512   1   1  512   1   1      1024.0       0.00        2,048.0       1,024.0      6144.0       2048.0       0.24%      8192.0
99             level5.root.conv  1280   1   1  512   1   1    655360.0       0.00    1,310,208.0     655,360.0   2626560.0       2048.0       1.40%   2628608.0
100              level5.root.bn   512   1   1  512   1   1      1024.0       0.00        2,048.0       1,024.0      6144.0       2048.0       0.23%      8192.0
101            level5.root.relu   512   1   1  512   1   1         0.0       0.00          512.0         512.0      2048.0       2048.0       0.07%      4096.0
102           level5.downsample   256   2   2  256   1   1         0.0       0.00          768.0       1,024.0      4096.0       1024.0       0.80%      5120.0
103            level5.project.0   256   1   1  512   1   1    131072.0       0.00      261,632.0     131,072.0    525312.0       2048.0       0.76%    527360.0
104            level5.project.1   512   1   1  512   1   1      1024.0       0.00        2,048.0       1,024.0      6144.0       2048.0       0.80%      8192.0
105                     avgpool   512   1   1  512   1   1         0.0       0.00          512.0         512.0      2048.0       2048.0       1.01%      4096.0
106                          fc   512   1   1  200   1   1    102600.0       0.00      204,800.0     102,600.0    412448.0        800.0       0.91%    413248.0
total                                                       15373432.0       1.11  126,014,208.0  63,110,344.0    412448.0        800.0     100.00%  63932672.0
===============================================================================================================================================================
Total params: 15,373,432
---------------------------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 1.11MB
Total MAdd: 126.01MMAdd
Total Flops: 63.11MFlops
Total MemR+W: 60.97MB
```
