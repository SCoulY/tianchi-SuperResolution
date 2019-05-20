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

####################
2019/5/20
简单转化为MP4格式，上传到百度网盘
链接：https://pan.baidu.com/s/1jMx8gRJPHTkKiA6iCdr9AQ 
提取码：j84p 
