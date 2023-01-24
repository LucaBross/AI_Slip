from tensorflow import keras
import numpy as np
import serial
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from collections import deque

def fill_data():
    sens_data=np.zeros((1, timesteps, features))
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
    return(sens_data)

def read_data():
    data = ser.readline()
    data = data.decode()
    data = data.replace("\n","")
    data = data.replace("\r","")
    return (data)
    
def split_data(data):
    dataSplit = data.split(";")
    TOFValue = dataSplit[0].split(":")[1]
    ENCODERValue = dataSplit[1].split(":")[1]
    FSLP1preassure = dataSplit[2].split(":")[1]
    FSLP1position = dataSplit[3].split(":")[1]
    FSLP2preassure = dataSplit[4].split(":")[1]
    FSLP2position = dataSplit[5].split(":")[1]
    Data=np.array([[TOFValue,ENCODERValue,FSLP1preassure,FSLP1position,FSLP2preassure,FSLP2position]])
    finalData=Data.astype(float)
    return(finalData)

def update_array(data_array, new_data):
    data_array = np.roll(data_array, -1, axis=1)
    data_array[:,-1] = new_data
    return(data_array)

def predict_slip(model, predictions_list, count):
    predictions = model.predict(filled_data,verbose=0)
    predictions_percent=predictions*100
    predictions_list.append(predictions_percent[0][0])
    count+=1
    if count > 80:
        predictions_list.pop(0)
        count-= 1
    return(predictions_list, count)

def save_sens_data(sens_data, FSLP1preassure_list, FSLP2preassure_list, FSLP1position_list, FSLP2position_list, TOF_list, count):
    FSLP1preassure_list.append(sens_data[0][2])
    FSLP2preassure_list.append(sens_data[0][4])
    FSLP1position_list.append(sens_data[0][3])
    FSLP2position_list.append(sens_data[0][5])
    TOF_list.append(sens_data[0][0])
    count +=1
    if count > 80:
        FSLP1preassure_list.pop(0)
        FSLP2preassure_list.pop(0)
        FSLP1position_list.pop(0)
        FSLP2position_list.pop(0)
        TOF_list.pop(0)
        count -= 1
    return (FSLP1preassure_list, FSLP2preassure_list,FSLP1position_list,FSLP2position_list, TOF_list, count)

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

COM_Port = "/dev/ttyUSB0"
Baudrate = 115200

ser = serial.Serial()
ser.port = COM_Port
ser.baudrate = Baudrate
ser.bytesize = serial.EIGHTBITS 
ser.parity = serial.PARITY_NONE 
ser.stopbits = serial.STOPBITS_ONE 
ser.timeout = None
ser.open()

modelpath='/home/luca/Documents'
modelname='1_adam_binary_crossentropy_epoch-60_batch-75_normalized-True_timesteps-70'
normalized=True
timesteps=70
features=6

if __name__=="__main__":
    model = keras.models.load_model(modelpath + '/' + modelname + '.h5')
    if normalized:
        scaler = load(modelpath + '/scaler' + modelname + '.joblib')
    plt.ion()
    fig, ((ax,ax3),  (ax2, ax4)) = plt.subplots(2,2)
    fig.set_figheight(10)
    fig.set_figwidth(14)
    count = 0
    count_sens =0 
    i=0
    predictions_list = []
    FSLP1list = []
    FSLP2list = []
    FSLP1pos_list = []
    FSLP2pos_list = []
    TOFlist= []
    filled_data = fill_data()
    try:
        while True:
            serial_data = read_data()
            if "TOF" in serial_data and "FSLP2position" in serial_data:
                try:
                    newData = split_data(serial_data)
                    if normalized:
                        newData = scaler.transform(newData)
                    filled_data = update_array(filled_data, newData)
                    if i ==2:
                        predictions_list, count = predict_slip(model, predictions_list, count)
                        i=0
                    FSLP1list, FSLP2list, FSLP1pos_list,FSLP2pos_list, TOFlist, count_sens = save_sens_data(newData, FSLP1list, FSLP2list, FSLP1pos_list,FSLP2pos_list, TOFlist, count_sens)
                    visualize(predictions_list, FSLP1list, FSLP2list, FSLP1pos_list,FSLP2pos_list, TOFlist)
                    i += 1
                except ValueError as ve:
                    print("Fehler: ",ve)
                    continue
                except IndexError as ie:
                    print("Fehler: ",ie)
                    continue
    except Exception as ex:
        print(type(ex))
        print(ex.args)
        print(ex)
    finally:
        ser.close()
        print('serport closed')
