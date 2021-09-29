## 湘乡市东山学校 具有环境监测功能的智能厨具
## 开源地址:https://github.com/fsquirt/RPI_KIC_SEC
import time
print("开始导入裤" +  " " + time.asctime( time.localtime(time.time()) ))
import os,sys
import cv2
import numpy as np
import RPi.GPIO as GPIO
import _thread
from tensorflow.lite.python.interpreter import Interpreter
print("导入裤完成" +  " " + time.asctime( time.localtime(time.time()) ))  #导入TensorFlow Lite比较慢

GPIO.setwarnings(False)
GPIO.cleanup()

# 启用swap防止内存不足
os.system("sudo swapon /swap/swapfile")

# L289N 模块配置
in1 = 18
in2 = 16
en = 22

#GPIO串口相关配置
#GPIO.setmode(GPIO.BCM)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
p=GPIO.PWM(en,1000)
p.start(25)

# MQ-5模块配置
#GPIO.setmode(GPIO.BOARD) 
GPIO.setup(12,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)

# TensorFlow的模型文件相关配置
MODEL_NAME = "tf_model"
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = float(0.5)

CWD_PATH = os.getcwd()
# tf_model模型文件夹路径, 其中包含用于对象检测的模型 
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
# labelmap.txt标签文件路径
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
# detect.tflite模型文件路径
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

del(labels[0])

interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# 获取模型信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

capture = cv2.VideoCapture(0)  #打开摄像头

Now_Time = time.time()
Find_time = time.time() 


def Web_Cam():
    while(True):
        ret, frame = capture.read()  #取一帧拿来用

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        classes = interpreter.get_tensor(output_details[1]['index'])[0] # 检测对象的类索引
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # 检测对象的置信度  

        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                Now_Time = time.time() 
                object_name = labels[int(classes[i])]
                if(object_name == "person"):
                    print(object_name + " " + time.asctime( time.localtime(time.time()) ))
                    Find_time = Now_Time

        if(time.time() - Find_time) > 10:
            print("Nobody")
            _thread.start_new_thread(In_danger,())
            break            

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

def MQ_Get():  #检测是否危险气体
    try:
	    while(True):
		    status=GPIO.input(12)
		    if(status == True):    #有危险气体会输出高电平
		    	print ('正常' + " " + time.asctime( time.localtime(time.time()) ))      
		    else:
		    	print('检测到危险气体' + " " + time.asctime( time.localtime(time.time()) ))
		    	_thread.start_new_thread(In_danger,())
		    	break

		    time.sleep(1)
    except KeyboardInterrupt:
	    GPIO.cleanup()

def Temp_Get(): #检测温度
    try:
        while(True):
            tempture = open("/sys/bus/w1/devices/28-01205a96ef18/temperature","r")  #直接读取串口数据，简单快捷
            lines = tempture.readlines()
            for line in lines:
                temp = int(line) / 1000
            if(temp < 30):
                print(str(temp) + " " + time.asctime( time.localtime(time.time()) ))
            else:
                _thread.start_new_thread(In_danger,())
                break

    except KeyboardInterrupt:
        GPIO.cleanup()

def In_danger():
    p.ChangeDutyCycle(75)
    print("马达启动" + " " + time.asctime( time.localtime(time.time()) ))
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)

    time.sleep(2)  #让马达转2秒钟
    GPIO.cleanup()

    sys.exit()

_thread.start_new_thread(Web_Cam,())
_thread.start_new_thread(MQ_Get,())
_thread.start_new_thread(Temp_Get,())

while 1:
    pass