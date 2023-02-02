from tensorflow import keras
import numpy as np
import serial
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import threading

def read_data():
    data = ser.readline()
    data = data.decode()
    data = data.replace("\n","")
    data = data.replace("\r","")
    return (data)

def split_data(data):
    global sensors
    dataSplit = data.split(";")
    sensor_data = []
    if sensors['TOF']:
        sensor_data.append(float(dataSplit[0].split(":")[1]))
    if sensors['ENCODER']:
        sensor_data.append(float(dataSplit[1].split(":")[1]))
    if sensors['FSLP1_PRESSURE']:
        sensor_data.append(float(dataSplit[2].split(":")[1]))
    if sensors['FSLP1_POSITION']:
        sensor_data.append(float(dataSplit[3].split(":")[1]))
    if sensors['FSLP2_PRESSURE']:
        sensor_data.append(float(dataSplit[4].split(":")[1]))
    if sensors['FSLP2_POSITION']:
        sensor_data.append(float(dataSplit[5].split(":")[1]))
    finalData = np.array([sensor_data])
    return finalData

def update_array(new_data):
    global sens_data
    sens_data = np.roll(sens_data, -1, axis=1)
    sens_data[:,-1] = new_data

def save_sens_data(newData, count):
    global FSLP1preassure_list
    global FSLP2preassure_list
    global FSLP1position_list
    global FSLP2position_list
    global TOF_list
    FSLP1preassure_list.append(newData[0][2])
    FSLP2preassure_list.append(newData[0][4])
    FSLP1position_list.append(newData[0][3])
    FSLP2position_list.append(newData[0][5])
    TOF_list.append(newData[0][0])
    count+=1
    if count > 80:
        FSLP1preassure_list.pop(0)
        FSLP2preassure_list.pop(0)
        FSLP1position_list.pop(0)
        FSLP2position_list.pop(0)
        TOF_list.pop(0)
        count-=1
    return(count)

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

def visualize(predictions_list, FSLP1preassure_list, FSLP2preassure_list, FSLP1pos_list,FSLP2pos_list, TOFlist):
    ax.clear()
    ax.set_ylabel('Prediction Slip in %', size=20, color='r')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)
    ax.plot(predictions_list, linewidth=5, color='red')
    ax2.clear()
    ax2.set_ylabel('FSLP Preassure', size=5, color='b')
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, 100)
    ax2.plot(FSLP1preassure_list, color='orange',linewidth=1.0, linestyle="-", label='FSLP1preassure')
    ax2.plot(FSLP2preassure_list, color='blue',linewidth=1.0, linestyle="-", label='FSLP2preassure')
    ax3.clear()
    ax3.set_ylabel('FSLP Position', size=5, color='b')
    ax3.set_ylim(0, 1)
    ax3.set_xlim(0, 100)
    ax3.plot(FSLP1pos_list, color='orange',linewidth=1.0, linestyle="-", label='FSLP1preassure')
    ax3.plot(FSLP2pos_list, color='blue',linewidth=1.0, linestyle="-", label='FSLP2preassure')
    ax4.clear()
    ax4.set_ylabel('TOF', size=5, color='b')
    ax4.set_ylim(0, 1)
    ax4.set_xlim(0, 100)
    ax4.plot(TOFlist, color='green',linewidth=1.0, linestyle="-", label='FSLP1preassure')
    plt.pause(0.0001)

def collect_data():
    global normalized
    global scaler
    count_sens=0
    while True:
        data=read_data()
        if "TOF" in data and "FSLP2position" in data:
            try:
                newData = split_data(data)
                if normalized:
                    newData = scaler.transform(newData)
                update_array(newData)
                count_sens=save_sens_data(newData, count_sens)
            except ValueError as ve:
                print("Fehler: ",ve)
                continue
            except IndexError as ie:
                print("Fehler: ",ie)
                continue

def predict_slip():
    global sens_data
    global model
    count=0
    while True:
        predictions = model.predict(sens_data,verbose=2)
        predictions_percent=predictions*100
        predictions_list.append(predictions_percent[0][0])
        count+=1
        if count > 80:
            predictions_list.pop(0)
            count-= 1

ser = serial.Serial()
ser.port = "/dev/ttyUSB0"
ser.baudrate = 115200
ser.bytesize = serial.EIGHTBITS 
ser.parity = serial.PARITY_NONE 
ser.stopbits = serial.STOPBITS_ONE 
ser.timeout = None
ser.open()

sensors = {
    'TOF': True,
    'ENCODER': True,
    'FSLP1_PRESSURE': True,
    'FSLP1_POSITION': True,
    'FSLP2_PRESSURE': True,
    'FSLP2_POSITION': True
}

predictions_list = []
FSLP1preassure_list = []
FSLP2preassure_list = []
FSLP1position_list = []
FSLP2position_list = []
TOF_list= []

modelpath='/home/luca/Documents/models'
modelname='16_adam_binary_crossentropy_epoch-300_batch-75_normalized-True_timesteps-60'
normalized=(modelname.split('_normalized-')[1].split('_')[0] == "True")
model = keras.models.load_model(modelpath + '/' + modelname + '.h5')
if normalized:
    scaler = load(modelpath + '/scaler' + modelname + '.joblib')
features=0
for value in sensors.values():
    if value == True:
        features += 1

timesteps= int(modelname.split('_timesteps-')[1])
sens_data=np.zeros((1, timesteps, features))
plt.ion()
fig, ((ax,ax3), (ax2, ax4)) = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(14)

def main():
    fill_data()
    read_Sensdata=threading.Thread(target=collect_data)
    predict=threading.Thread(target=predict_slip)
    read_Sensdata.setDaemon(True)
    predict.setDaemon(True)
    read_Sensdata.start()
    predict.start()
    while True:
            visualize(predictions_list, FSLP1preassure_list, FSLP2preassure_list, FSLP1position_list, FSLP2position_list, TOF_list)
        
if __name__=='__main__':
    main()