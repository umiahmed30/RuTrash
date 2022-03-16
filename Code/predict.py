import json, os, re, sys
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from picamera import PiCamera
from time import sleep
from PIL import Image
import serial
import board
import busio
import adafruit_sgp30
import time

#Initiate weight sensor
EMULATE_HX711=False
if not EMULATE_HX711:
    import RPi.GPIO as GPIO
    from hx711 import HX711
else:
    from emulated_hx711 import HX711

def cleanAndExit():
    print ("Cleaning...")

    if not EMULATE_HX711:
        GPIO.cleanup()

    print ("Bye!")
    sys.exit()

hx = HX711(5, 6)

if __name__ == '__main__':
    model_path = sys.argv[1]

    print ('Loading serial')
    for x in range(10):
        name = '/dev/ttyACM' + str(x)
        try:
            ser = serial.Serial(name)
        except:
            continue

    print(ser.name)

    print('Loading model:', model_path)
    model = load_model(model_path)
    image2 = 'image2.jpg'
    run= 'y'


    voc = True
    weight_sensor = True

    # Setup i2c
    print('Setting up i2c...')
    try:
        thres_smell = 0
        min_smell = 0
        max_smell = 0
        smell = []
        i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)
        sgp30 = adafruit_sgp30.Adafruit_SGP30(i2c)
        sgp30.iaq_init()
        sgp30.set_iaq_baseline(0x8973, 0x8aae)
        for i in range(40):
            print("eCO2 = %d ppm \t TVOC = %d ppb" % (sgp30.eCO2, sgp30.TVOC))
            sleep(1)
    except:
        print ("VOC failed")
        voc = False

    # Setup weight sensor
    print('Setting up weight sensor...')
    try:
        hx.set_reading_format("MSB", "MSB")
        hx.set_reference_unit(4.09)
        hx.reset()
        hx.tare(50)
        offset_prev = hx.tare(50)
    except:
        print ("weight sensor failed")
        weight_sensor = False
    print ("Place object now...")

    while(run =='y'):
        thres_smell = 0
        camera = PiCamera()
        camera.start_preview()
        sleep(10)
        camera.capture(image2)
        camera.stop_preview()

        if voc:
            smell = []
            for i in range(20):
                eCO2 = sgp30.eCO2
                TVOC = sgp30.TVOC
                print("eCO2 = %d ppm \t TVOC = %d ppb" % (eCO2, TVOC))
                smell.append(TVOC)
                sleep(0.25)
            smell = np.array(smell)
            b = smell[::-1]
            max_smell = len(b) - np.argmax(b) - 1
            if (max_smell == 0 ):
                thres_smell = 0
                min_smell = max_smell
            else:
                min_smell = np.argmin(smell[0:max_smell])
                thres_smell = smell[max_smell] - smell[min_smell]
            print("max", smell[max_smell], "min", smell[min_smell])

        imageCrop = Image.open(image2)
        cropped = imageCrop.crop((570,105,1365,830))
        save = cropped.save('imageC.jpg')
        img2 = image.load_img('imageC.jpg', target_size=(224, 224))
        y = image.img_to_array(img2)

        hx.set_offset_A(offset_prev)
        y = np.expand_dims(y, axis=0)
        preds = model.predict(y)
        print(preds)
        print('Generating predictions on image:', image2)

        category = 4
        for p in range(4):
            if (preds[0][p] >= 0.60):
                category = p


        target_names = [b'glass\n',b'metal\n',b'paper\n',b'plastic\n',b'trash\n']
        print('Predicted Category: ')
        print(target_names[category])
        print('After VOC test')
        if (thres_smell >= 20) and voc:
                category = 4
                print('VOC smelled food')
        print(target_names[category])

        weight = np.zeros(10)
        for w in range(10):
            weight[w] = hx.get_weight(1)
        med_weight = np.median(weight)
        print("Weight (in grams) =", med_weight)
        if (category == 1 or category == 3):
            print('After food weight test')
            if (med_weight >= 200) and weight_sensor:
                    category = 4
        print(target_names[category])
        ser.write(target_names[category])


        run = input('would you like to continue? y/n')
        camera.close()
        offset_prev = hx.tare(50)
        sleep(5)
    ser.close()
