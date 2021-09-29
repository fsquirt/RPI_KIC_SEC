# 具有环境监测功能的智能厨具
开源精神永流传！！！  
这个作品获得了湖南省的第一，现在被湖南省推荐参加中国国际“互联网 ”大学生创新创业大赛总决赛

## 工作流程
其实就是开三个线程：    
一个线程是用opencv获取摄像头照片,然后使用TensorFlow检测画面中有没有人  
一个是读取温度模块的数据，检测温度是否超过设定值   
一个是读取气体检测模块输出的电平，检测是否为高电平(检测到危险气体)   
如果任意一个条件触发则转动马达来关火，防止发生火灾   

## 对于软件要求
程序在树莓派上运行，使用Python3，需要安装`TensorFlow Lite`，`numpy`，`opencv`

## 对于硬件要求
树莓派3B+，树莓派4B都可以  
在内存低于2G的树莓派上运行，你可能需要启用swap防止内存不足  
如果你不需要增加swap，那需要将17行 `os.system("sudo swapon /swap/swapfile")` 注释掉  
使用的模块:L289N马达控制模块，MQ-5气体检测模块，DS18B20温度检测模块，面包板一个，杜邦线若干，马达一个  
对了还有树莓派的SCI摄像头  

## 仓库文件
`TFlite_Webcam.py`: 程序主文件  
`tf_model\detect.tflite`: 预训练的量化 COCO SSD MobileNet v1 模型  
`tf_model\labelmap.txt`: 模型标签文件  
在导入库的时候会爆警告:`Import "tensorflow.lite.python.interpreter" could not be resolved`，**可以无视**  

## 连线图
晚点发（
