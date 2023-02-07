from tensorflow import keras
import numpy as np
import serial
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import threading
import pyaudio
from time import sleep

def read_data():
    data = ser.readline().decode().strip()
    # data='TOF: 60; Encoder: 700; FSLP1preassure: 34.00; FSLP1position: 300; FSLP2preassure: 50.00; FSLP2position: 200;'
    return (data)

def split_data(data):
    global sensors
    dataSplit = data.split(";")
    sensor_data = []
    if sensors['TOF']:
        sensor_data.append(float(dataSplit[0].split(":")[1]))
    if sensors['ENCODER']:
        sensor_data.append(float(dataSplit[1].split(":")[1]))
    if sensors['FSLP1_PREASSURE']:
        sensor_data.append(float(dataSplit[2].split(":")[1]))
    if sensors['FSLP1_POSITION']:
        sensor_data.append(float(dataSplit[3].split(":")[1]))
    if sensors['FSLP2_PREASSURE']:
        sensor_data.append(float(dataSplit[4].split(":")[1]))
    if sensors['FSLP2_POSITION']:
        sensor_data.append(float(dataSplit[5].split(":")[1]))
    finalData = np.array([sensor_data])
    return finalData

def update_array(new_data):
    global sens_data
    sens_data = np.roll(sens_data, -1, axis=1)
    sens_data[:,-1] = new_data

def fill_data():
    global sens_data
    global normalized
    i=0
    print('loading data')
    while i<timesteps:
        try:
            serial_data=read_data()
            if "TOF" in serial_data and "FSLP2position" in serial_data:
                new_data=split_data(serial_data)
                if normalized:
                    new_data = scaler.transform(new_data)
                sens_data[0,i,:]=new_data
                i=i+1
        except ValueError as ve:
            print("Fehler: ",ve)
            continue
        except IndexError as ie:
            print("Fehler: ",ie)
            continue

def play_sound():
    try:
        p = pyaudio.PyAudio()
        fs = 44100  # sampling rate, Hz, must be integer
        duration = 0.1  # in seconds, may be float
        f = 300.0  # sine frequency, Hz, may be float
        samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)
        stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
        while True:
            volume=predictions
            output_bytes = (volume * samples).tobytes()
            stream.write(output_bytes)
    except Exception as ex:
        print(f"Error: {ex}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def collect_data():
    data=read_data()
    if "TOF" in data and "FSLP2position" in data:
        newData = split_data(data)
        if normalized:
            newData = scaler.transform(newData)
        update_array(newData)

def predict_slip():
    global sens_data
    global model
    global predictions
    while True:
        predictions = model.predict(sens_data,verbose=0)
        print(predictions)

ser = serial.Serial("/dev/ttyUSB0", 115200, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE, None)
if not ser.is_open:
    ser.open()

modelpath='/home/luca/Documents/models'
modelname='25_sensors-TOF-FSLP1_PREASSURE-FSLP2_PREASSURE-FSLP1_POSITION-FSLP2_POSITION_normalized-True_timesteps-60'
sensors = {
    'TOF': 'TOF' in modelname,
    'ENCODER': 'ENCODER' in modelname,
    'FSLP1_PREASSURE': 'FSLP1_PREASSURE' in modelname,
    'FSLP1_POSITION': 'FSLP1_POSITION' in modelname,
    'FSLP2_PREASSURE': 'FSLP1_PREASSURE' in modelname,
    'FSLP2_POSITION': 'FSLP2_POSITION' in modelname
}
model = keras.models.load_model(modelpath + '/' + modelname + '.h5')
normalized=(modelname.split('_normalized-')[1].split('_')[0] == "True")
if normalized:
    scaler = load(modelpath + '/scaler' + modelname + '.joblib')
features=0
for value in sensors.values():
    if value == True:
        features += 1
timesteps= int(modelname.split('_timesteps-')[1])
sens_data=np.zeros((1, timesteps, features))
predictions = 0

def main():
    fill_data()
    predict=threading.Thread(target=predict_slip)
    sound=threading.Thread(target=play_sound)
    sound.setDaemon(True)
    predict.setDaemon(True)
    predict.start()
    sleep(1)
    sound.start()
    try:
        while True:
            try:
                collect_data()
            except ValueError as ve:
                print("Fehler: ",ve)
                continue
            except IndexError as ie:
                print("Fehler: ",ie)
                continue
    finally:
        ser.close()
        print('ser closed')
        
if __name__=='__main__':
    main()
    pass