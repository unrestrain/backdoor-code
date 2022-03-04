# 后门攻击文章《Invisible Backdoor Attack with Sample-Specific Triggers》的实现代码
该攻击方法的主要内容是实现了Sample-Specific的后门攻击，每张图片的后门触发器都不相同，这使得训练出来的后门模型难以被后门防御算法检测出来。
每张图片的后门触发器是由一个训练好的编码器生成的，该编码器的功能是将一个字符串的信息嵌入到图片当中，嵌入后的图片看起来和原来的图片相同，但是通过解码器可以将嵌入的字符串还原出来。
本文将图像的标签字符串嵌入到图像当中，将嵌入后的图像作为投毒数据。
所以本文的算法程序分为两部分，一部分是生成投毒数据的程序，另一部分是训练后门模型的程序。生
目前整理了训练模型后门的程序，该程序或许也可以用来训练其他算法的后门模型。  

### 训练模型后门模型的程序使用
```
python train.py \
--net=res18 \
--train_batch=128 \
--workers=0 \
--epochs=25 \
--schedule 15 20 \
--bd_label=0 \
--bd_ratio=0.1\
--data_dir=datasets/sub-imagenet-200 \
--bd_data_dir=datasets/sub-imagenet-200-bd/inject_a \
--checkpoint=ckpt/bd/res18_bd_ratio_0.1_inject_a \
--resume=false
```
目前支持resnet18和vgg13模型。

### 生成投毒数据的程序
生成投毒数据的程序来源于另一篇文章《StegaStamp: Invisible Hyperlinks in Physical Photographs》，对应的程序使用tf1.13实现的，在对应版本上可以跑通，该程序可以向一张图片中嵌入一个字符串，图片本身看起来还和原来一样。由于程序使用tf1.13实现，所以目前没有做任何改动，程序的使用方法为
```
python encode_image.py \
--model_path=ckpt/encoder_imagenet \
--image_path=data/imagenet/org/n01770393_12386.JPEG \
--out_dir=data/imagenet/bd/ 
```
其中ckpt/encoder_imagenet文件夹下放置作者已经训练好的模型。
