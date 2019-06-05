y4m 格式介绍：https://wiki.multimedia.cx/index.php/YUV4MPEG2 <br/>
y4m 与 yuv（yuv420 8bit planar） 互转命令：<br/>
    y4mtoyuv: ffmpeg -i xx.y4m -vsync 0 xx.yuv  -y <br/>
    yuvtoy4m: ffmpeg -s 1920x1080 -i xx.yuv -vsync 0 xx.y4m -y <br/>
y4m 与 png 互转命令：<br/>
   y4mtobmp: ffmpeg -i xx.y4m -vsync 0 xx%3d.bmp -y <br/>
   bmptoy4m: ffmpeg -i xx%3d.bmp  -pix_fmt yuv420p  -vsync 0 xx.y4m -y <br/>
y4m 每25帧抽样命令：<br/>
   ffmpeg -i xxx.y4m -vf select='not(mod(n\,25))' -vsync 0  -y xxx_sub25.y4m <br/>

## 初赛训练数据下载链接<br/>
round1_train_input:<br/>
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/input/youku_00000_00049_l.zip<br/>
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/input/youku_00050_00099_l.zip<br/>
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/input/youku_00100_00149_l.zip<br/>

round1_train_label:<br/>
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/label/youku_00000_00049_h_GT.zip<br/>
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/label/youku_00050_00099_h_GT.zip<br/>
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/label/youku_00100_00149_h_GT.zip<br/>

## 初赛验证数据下载链接<br/>
round1_val_input:<br/>
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/input/youku_00150_00199_l.zip<br/>

round1_val_label:<br/>
http://tianchi-media.oss-cn-beijing.aliyuncs.com/231711_youku/round1/train/label/youku_00150_00199_h_GT.zip<br/>

####################<br/>
2019/6/3<br/>
简单转化为bmp格式，上传到百度网盘<br/>
链接：https://pan.baidu.com/s/1y1VbT5GsKIS8CJEO4CWnWQ <br/>
提取码：emw0  

####################<br/>
2019/6/3 因为图片过大 模型改成半精度进行训练<br/>
uploaded latest model trained with automated mixed precision <br/>
install NVIDIA APEX dependency first (https://github.com/nvidia/apex) <br/>
$ git clone https://github.com/NVIDIA/apex <br/>
$ cd apex <br/>
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . <br/>

####################<br/>
TODO 已提issue: 
1.用SSIM loss（已上传 但没merge进模型）替换原有的criterion_content ref:https://github.com/Po-Hsun-Su/pytorch-ssim<br/>
2.简化模型 或者使用小patch（切片等）降低显存占用 使普通GPU也能训练高分辨率图片 trade off between acc and FLOPs <br/>

####################<br/>
2019/6/5<br/>
upload pretrained model and code for prediction <br/>
链接：https://pan.baidu.com/s/1p-RJg8pgjaf88PNL4P8t7g 
提取码：o1ew 


