#coding=utf-8
import threading
import time
import requests
import json
import RPi.GPIO as GPIO
import Adafruit_DHT

apiurl = ["http://api.heclouds.com/devices/606846327/datapoints",
# 用户密码
apiheaders = {'api-key' : 'vhl4ll0Ekjow59REecUKsEAv=8o='}

class checkPermin (threading.Thread):
    def __init__(self, bcmid, name, flaginit):    # 参数列表(BCM接口号，线程名称，是否持续检测)
        threading.Thread.__init__(self)
        self.channel = bcmid
        self.name = name
        self.flaginit = flaginit
    def run(self):
        while self.flaginit:
            print("开始检测")
            rasp_get(self.channel)
            print("马上结束检测")

"""rasp_get函数用于从树莓派中提取DHT11温湿度传感器的数据，每成功检测一次会sleep 60S.
"""
def rasp_get(channel):
    sensor =  Adafruit_DHT.DHT22
    pin = 4 #GPIO4
    humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
    print("温度：", temperature, "摄氏度，湿度：",  humidity, "%")
    data = [0, temperature, humidity]
    setonenet(data)


def setonenet(data):                #完成数据上传与当前数据下载
    print(data[1])
    print(data[2])
    json_data1 = {'value': int(data[1])}
    json_data2 = {'value': int(data[2])}

    data = {'datastreams': [{'id': '1', 'datapoints': [{'value': int(data[1])}]}]}
    jdata = json.dumps(data)

    json_data_turn1 = json.dumps(json_data1)
    json_data_turn2 = json.dumps(json_data2)
    print("开始上传数据")
    time.sleep(12)
    r1 = requests.post(apiurl[0], data=jdata, headers=apiheaders)
    time.sleep(12)
    r2=requests.post(apiurl[1], data=json_data_turn2.encode(), headers=apiheaders)
    print(r1)
    print(r2)
    print("完成上传数据")


if __name__ == "__main__":
    thread1 = checkPermin(4, "检测上传线程", 1)    #创建新线程
    

    # 开启新线程
    thread1.start()
    thread1.join()
    thread2.join()
    print("退出主线程")
