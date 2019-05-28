# QGDNet
Quality assessment Guided image De-noising Network

本科毕设《基于质量评价引导的图像去噪算法研究》

## 概述
本研究旨在向彩色数字图像去噪中加入图像质量评价（IQA）的引导，从而得到更符合人眼感知的复原结果。具体地，使用一个预训练好的轻量级全参考彩色图像质量评价网络提取感知特征，计算感知损失，然后将感知损失与像素级的内容损失加权求和，得到联合损失函数，用于指导去噪网络的训练。去噪网络基于改进的U-Net结构。
模型在滑铁卢数据集（Waterloo Exploration Dataset）[1]上训练，在滑铁卢数据库和CBSD68数据库[2]上测试，Waterloo库的level3上部分定性和定量指标如下：

![flag](https://github.com/Bobholamovic/QGDNet/blob/master/flag.bmp)

![flower](https://github.com/Bobholamovic/QGDNet/blob/master/flower.bmp)

模型 | PSNR | SSIM
:-: | :-: | :-:
DNet | 30.5194 | 0.8599
QGDNet-fl23-p | 29.6632 | 0.8460
QGDNet-fl2 | 30.5196 | **0.8619**
QGDNet-fl3 | 30.6311 | 0.8609
QGDNet-fl23 | **30.6736** | 0.8612

IQA网络的代码和预训练模型放在我的另一个仓库[Bobholamovic/CNN-FRIQA](https://github.com/Bobholamovic/CNN-FRIQA)。

## 数据准备
将网络所需经过的“训练”、“验证”和“测试”三个阶段称为3个phases，分别对应洋文train, val, and test. 对于train和val phase都需要给定带标签的样本，test样本可以不带标签，不过这样将无法正确计算测试集上的评价指标。把图像文件的相对路径存在`txt`文件里，如`train_images.txt`、`train_labels.txt`、`val_images.txt`、`val_labels.txt`等等，称为“数据列表”文件。例如，假设数据集目录为`xxx`，而训练样本图像位于`xxx/distorted/train/`文件夹下，则`train_images.txt`的内容应该类似这样：
```
distorted/train/img1.png
distorted/train/img2.bmp
...
```

每一个样本的路径单独占据`txt`文件中的一行，并且每个`{phase}_images.txt`文件与相应的`{phase}_labels.txt`文件的样本数量和顺序都应该是对应的。由于作者太懒，不打算提供划分数据集和生成`txt`文件的脚本。

提供了一个缺省的配置文件`src/config.yaml`，也可以通过命令行参数`--exp-config`指定新的配置文件，这样做对比实验更方便。`DATA_DIR`选项可以指定数据集所在位置，`LIST_DIR`选项指定“数据列表”文件所在位置。

+ 注意，Waterloo数据库的图像是动态生成的，可以先跑一次MATLAB把图存下来。

## 训练模型
在配置文件（默认是`src/config.yaml`）中设置一些实验条件：
- `IQA_MODEL`指定预训练的IQA模型的路径；
- `CROP_SIZE`指定去噪网络输入裁块大小；
- `FEATS`用于指定所选用的特征层和对应的权重大小，比如想用fl2和fl3特征，将`FEATS`修改为：
```yml
FEATS:
 - fl2: 0.05
 - fl3: 0.005
```

**在配置文件中指定特征层的次序一定要和它们在IQA网络中出现的次序一致**，否则可能出现设置的权重和特征未正确配对的情况。

如果当前处在项目根目录，首先：
```bash
cd src
```

在终端输入如下指令进行模型的训练：
```bash
python main.py train --lr 初始学习率大小 --epochs 总epoch数 --batch-size mini-batch大小 --weight-decay L2正则项权重 --workers 加载数据使用进程数 --resume 需要继续训练的checkpoint --store-interval 保存中间模型的频率 --lr-mode 学习率衰减模式 --step step衰减模式下学习率衰减步长
```

最简版（使用默认参数）：
```bash
python main.py train
```

单独验证一次结果：
```bash
python main.py train --evaluate --resume some_checkpoint
```

+ 因为验证集有点大导致验证过程比较耗时，所以我从中截取了一部分，只使用前32个样本。

## 测试模型
配置文件中的`OUT_DIR`可以指定网络输出图像的位置，`TEST_SUFFIX`指定生成图像的后缀（在扩展名前加）。
```bash
python main.py test --resume some_checkpoint
```
如果想在训练集或者验证集上测试，可以通过`--phase`参数指定。如，在训练集数据上测试：
```bash
python main.py test --resume some_checkpoint --phase train
```

## 依赖和环境
略

## 致谢
代码的基本结构基于[fyu/drn](https://github.com/fyu/drn)，UNet模型定义方面使用了来自[milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)的开源代码，向以上作者致以诚挚的感谢。

另：由于这是第一次接触DL和pytorch，代码写得比较乱请见谅。

### 参考文献
> [1] Ma K, Duanmu Z, Wu Q, et al. Waterloo exploration database: New challenges for image quality assessment models[J]. IEEE Transactions on Image Processing, 2017, 26(2):1004–1016.  
[2] Martin D, Fowlkes C, Tal D, et al. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics[C] // IEEE. in Proceedings of IEEE International Conference on Computer Vision, 2001: 416–423.
