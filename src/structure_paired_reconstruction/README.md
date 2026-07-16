# torchvision-FasterRCNN
torchvision faster-rcnn例子修改版，考虑到目前大部分目标检测任务使用的数据集制作工具得到的标注文件只有boxes，没有mask和keypoint，
因此在coco_utils.py中简单修改了类ConvertCocoPolysToMask，去除了其中的mask和keypoint部分。
要训练自己的数据集，需要先转换为COCO格式，其组织形式为：
```
├── annotations
  ├── instances_test.json
  ├── instances_train.json
  ├── instances_trainval.json
  ├── instances_trainvaltest.json
  └── instances_val.json
├── test
├── train
├── trainval
├── trainvaltest
└── val
```
Pascal VOC， YOLO， CSV， TXT， COCO等格式的数据集相互转换脚本见笔者的另一个库：
https://github.com/ouening/OD_dataset_conversion_scripts

简单的训练命令为：
```
python3 train.py --data-path datasets/CocoFormat \
--dataset coco \
--num-classes  8 \
--batch-size 4 \
--epochs 50
```
还有其他参数可以自行查看train.py源码。
