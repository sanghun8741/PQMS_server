
import numpy as np
import pandas as pd
import psycopg2
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
import os
import json
import sys

def doQuery(conn,subs_id = 1, dl_id = 10) :
    cur = conn.cursor()
    cur.execute('SELECT * FROM data WHERE subs_id=' + str(subs_id) + ' and dl_id=' + str(dl_id) + ';')
    data = pd.DataFrame(cur.fetchall(), columns=['subs_id', 'dl_id', 'load', 'time'])
    conn.close()
    return data


def loadData(link):
    data = pd.read_csv(link)
    datetimeColName = data.columns[0]
    data[datetimeColName] = pd.to_datetime(data[datetimeColName])
    data.set_index(datetimeColName, inplace=True)
    return data


def scale_to_0and1(data):
    scaler = MinMaxScaler(feature_range=(0, 1))  # MinMax Scaler
    data = scaler.fit_transform(data)  # input: ndarray type data
    return(data, scaler)


def inverse_scale_to_0and1(data, scaler):
    data = scaler.inverse_transform(data)  # input: ndarray type data
    return(data)


# split into train and test sets
def splitData(data, trainPercent=0.7, split_by_time=False, split_date=None):
    if split_by_time is False:
        train_size = int(len(data) * trainPercent)
        train_data, test_data = data.iloc[0:train_size,], data.iloc[train_size:,]
        print("\n", "train length:", len(train_data),"\n", "test length:", len(test_data))
        return (train_data, test_data)
    elif split_by_time is True:
        # split_date = pd.Timestamp("01-01-2011")
        split_date = split_date
        train_data = data.ix[:split_date, :]
        train_data.drop(split_date, axis=0, inplace=True)
        test_data = data.ix[split_date:, :]
        return(train_data, test_data)



# --- create dataset with window size --- #
def sequentialize(data, index, sequenceSize=None, to_ndarray=False):
    """
       :param data: intput data (normalized)
       :param index: index of data (before nomalizing) to convert data array to dataframe (in order to use shift() function)
       :param sequenceSize: size of 1 sequence (number of elements in each sequence)
       :param to_ndarray: whether return arrays are converted into narray or not
       : ex:
       :******from*****     data    =   [0,1,2,3,4,5,6,7,8,9....]
       :******to*****       sequentialized = [1,2,3,4,5,6,7,8,9,10,11,12],[2,3,4,5,6,7,8,9,10,11,12,13],[3,4,5....]....
       :return: Array of sequences
       """
    """
    Convert dataset into array of sequence with size of each sequence determined using sequenceSize
    """
    if sequenceSize is None:
        print("\n", "please use 'sequenceSize'...!")
        return(None)
    elif isinstance(sequenceSize, int):
        # change type to use 'shift' of pd.DataFrame
        data = pd.DataFrame(data, columns=["value"], index=index)

        # dataframe which is shifted as many as window size
        for idx in range(1,sequenceSize+1):
            data["before_{}".format(idx)] = data["value"].shift(idx)
            # drop na
            # drop NA to get LSTM input and output sequences
        inputSequence = data.dropna().drop('value', axis=1)
        outputSequence = data.dropna()[['value']]
        # convert to narray if necessary
        if to_ndarray is True:
            inputSequence = inputSequence.values
            outputSequence = outputSequence.values
        return(inputSequence, outputSequence)

def reshapeForLSTM(data, time_steps=None):
    """
    :param data: intput data (do not reshape output data)
    :param time_steps: time steps after
    :return: reshaped data for LSTM
    """
    """
    The LSTM network expects the input data (X) 
    to be provided with 
    a specific array structure in the form of: 
    [samples, time steps, features].
    """
    if time_steps is None:
        print("please denote 'time_steps'...!")
        return(None)
    else:
        data_reshaped = np.reshape(data, (data.shape[0], time_steps, 1))
    return(data_reshaped)

#
class pytorch_lstm(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize=1, numLayers=2):
        super(pytorch_lstm, self).__init__()
        self.hidden_size = hiddenSize
        self.num_layers = numLayers
        self.lstm = nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True)
        self.fc = nn.Linear(hiddenSize, outputSize)


    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


class customLoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(customLoss, self).__init__(size_average)
        self.reduce = reduce
#################################################################
    def forward(self, input, target, coefficient=0.5):
        # todo: Every operating must be done using torch.Variable
        # input1 = input.view(-1).data.numpy()
        # target1 = target.view(-1).data.numpy()
        # loss = 0
        # for i in range(input.shape[0]):
        #         if (input1[i] > target1[i]):
        #             temp = torch.mul(torch.add(input[i], 0-target[i])**2, (1-coefficient)) ### ((1-coefficient) * (abs(input-output))
        #             f = lambda x,y,coefficient: (x-y).sum()
        #         else:
        #             temp = torch.mul(torch.add(input[i], 0 - target[i]).pow(2), (coefficient))

        # todo: how to inspect coefficient into residual based on residual value?
        # todo: respect to the fact that residual is a Tensor (for, if are impossible)
        residualTenSor = torch.add(input, -target) # predict - real
        residualNumpy = residualTenSor.data.numpy()
        for i in range(len(residualNumpy)):
            if residualNumpy[i]>0:
                residualTenSor[i] = torch.mul(residualTenSor[i], coefficient)
            elif residualNumpy[i]<0:
                residualTenSor[i] = torch.mul(residualTenSor[i], 1-coefficient)
        residualRaiseToPower = residualTenSor ** 2  # (predict - real)**2
        totalresidualPower = torch.sum(residualRaiseToPower)  ### sum((predict-real)**2)
        loss = torch.div(totalresidualPower,input.shape[0]) ### sum/n
        #todo: convert loss to FloatTensor before return

        # print('myLoss: %f \n theirLoss: %f' % (loss, F.mse_loss(input, target, size_average=self.size_average, reduce=self.reduce)))
        return loss
    ################################################################
        # return F.mse_loss(input, target, size_average=self.size_average, reduce=self.reduce)
                                    #


def getModelFile(folder, subs_id, dl_id):
    modelList = []
    temp = []
    for file in os.listdir(folder):
        if (file.endswith('.pt')):
            modelList.append(file)
    for m in modelList:
        elements = m.split('_')
        if (elements[1] == '1' and elements[3] == '10'):
            temp.append(folder+'\\'+m)
    if len(temp)!=0:
        return temp[0]
    else:
        print("could not find matched model")


def PQMS_Prediction(subs_id, dl_id,hostname, username, password, database, time,forecastLength = 10):
    ##### create model instance from class
    lstm = pytorch_lstm(1, 12)
    #### mark that this step is not traning step
    flag = lstm.train(False)
    forecastArray = []
    modelPath = getModelFile('D:\\project\\WebstormProjects\\PQMS_server\\model\\algorithm\\pqms\\models', subs_id, dl_id)
    lstm.load_state_dict(torch.load(modelPath))
    lstm.eval()
    conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
    data = doQuery(conn, subs_id, dl_id)
    data1 = data[['time', 'load']].copy()
    data1['time'] = pd.to_datetime(data1['time'])
    data1.set_index('time', inplace=True)
    data1 = data1.resample('T').interpolate().ffill().reindex(pd.date_range(data1.index[0], data1.index[-1], freq='10T'))
    originalData = data1
    data = originalData.loc[originalData.index < pd.to_datetime(time)]
    if len(data)==0:
        return 0,forecastArray
    elif forecastLength>300:
        return 0, forecastArray
    else:
        scaled_data, scaler = scale_to_0and1(data)
        scaled_data = scaled_data.astype('float32')
        ##### forecasting
        window = scaled_data  ################################################## initiating predict input using training data
        for i in range(forecastLength):
            ######## sequentializing, reshaping data to LSTM format
            inputTrain, outputTrain = sequentialize(window, originalData.index[0:len(window)], 12, to_ndarray=True)
            # len(window)
            inputTrain_reshaped = reshapeForLSTM(inputTrain, 12)
            tensor_train_input = torch.from_numpy(inputTrain_reshaped)
            var_train_input = Variable(tensor_train_input)
            predicted = lstm(var_train_input)
            forecastArray = predicted.view(-1).data.numpy()
            window = np.insert(window,0, forecastArray[0])
            window = window.reshape(len(window), 1)
        predicted = forecastArray[-forecastLength:] ##### get predicted part
        predicted = scaler.inverse_transform(predicted.reshape(-1,1))

        predictedDictArray = []
        predictedIndex = originalData.index[len(data):len(data)+forecastLength]
        for i in range(len(predictedIndex)):
            returnDict = {}
            returnDict['time'] = str(predictedIndex[i])
            returnDict['load'] = str(predicted[i,0])
            predictedDictArray.append(returnDict)
        return 1, predictedDictArray


if __name__ == "__main__":
    #hostname = '192.168.1.67'
    #username = 'postgres'
    #password = '1234'
    #database = 'pqms_data'
    #subs_id = 1
    #dl_id = 10
    #time = '2018-02-01 15:00:00'
    #type = 3

    try:
        hostname = sys.argv[1]
        username = sys.argv[2]
        password = sys.argv[3]
        database = sys.argv[4]
        port = sys.argv[5]
        subs_id = sys.argv[6]
        dl_id = sys.argv[7]
        time = sys.argv[8]
        type = sys.argv[9]

        if type == '0':
            flag, predicted_10_minutes = PQMS_Prediction(subs_id, dl_id,hostname, username, password, database, time, forecastLength = 1)
            returnDict = {}
            returnDict['returnCode'] = flag
            returnDict['output'] = predicted_10_minutes
            print(json.dumps(returnDict))
        elif type == '1':
            flag, predicted_60_minutes = PQMS_Prediction(subs_id, dl_id, hostname, username, password, database, time, forecastLength=6)
            returnDict = {}
            returnDict['returnCode'] = flag
            returnDict['output'] = predicted_60_minutes
            print(json.dumps(returnDict))
        elif type == '2':
            flag, predicted_180_minutes = PQMS_Prediction(subs_id, dl_id, hostname, username, password, database, time, forecastLength=18)
            returnDict = {}
            returnDict['returnCode'] = flag
            returnDict['output'] = predicted_180_minutes
            print(json.dumps(returnDict))

    except Exception as ex:
        returnDict = {}
        returnDict['returnCode'] = 0
        returnDict['output'] = []
        print(json.dumps(returnDict))


