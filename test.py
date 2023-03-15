import csv
import numpy as np
from predict import predict

sampling_rate = 128
final_data = {"eeg": {"data":None, "sampling_rate":sampling_rate, "channels":None}}

# read data from file
path = "/media/nastaran/HDD/projects/f2f_remote/preprocessed_data/p01/preprocessed_output/eeg/eeg-01-01-12.csv"

data = []
with open(path, 'r') as file:
    reader = csv.reader(file, delimiter=',')
    i = 0
    for row in reader:
        if i == 0:
            # Skip header line
            i += 1
            continue
        data.append(np.array(row, dtype=np.float32))

data = np.array(data)

#display_signal(data)8

# put 20 seconds of data in data{"gsr":data}
duration = 20
i = 0

while True:
    if (i+1)*duration*sampling_rate > data.shape[0]:
        break
    extracted_data = data[i*duration*sampling_rate:(i+1)*duration*sampling_rate,:]

    #final_data["gsr"]["data"] = extracted_data[:,0]
    #final_data["gsr"]["sampling_rate"] = 128
    final_data["eeg"]["data"] = extracted_data
    final_data["eeg"]["sampling_rate"] = 128
    result_eeg = predict(final_data)
    print(result_eeg)
    i += 1

final_data = {"ppg": {"data":None, "sampling_rate":sampling_rate}}

path = "/media/nastaran/HDD/projects/f2f_remote/preprocessed_data/p01/preprocessed_output/shimmer/ppg/ppg-01-01-12.csv"

data = []
with open(path, 'r') as file:
    reader = csv.reader(file, delimiter=',')
    i = 0
    for row in reader:
        if i == 0:
            # Skip header line
            i += 1
            continue
        data.append(np.array(row, dtype=np.float32))

data = np.array(data)
i=0
while True:
    if (i+1)*duration*sampling_rate > data.shape[0]:
        break
    extracted_data = data[i*duration*sampling_rate:(i+1)*duration*sampling_rate,:]
    print(extracted_data[:,0].shape, data.shape[0])
    #final_data["gsr"]["data"] = extracted_data[:,0]
    #final_data["gsr"]["sampling_rate"] = 128
    final_data["ppg"]["data"] = extracted_data[:,0]
    final_data["ppg"]["sampling_rate"] = 128
    result = predict(final_data)
    print(result)
    i += 1

final_data = {"gsr": {"data":None, "sampling_rate":sampling_rate}}
path = "/media/nastaran/HDD/projects/f2f_remote/preprocessed_data/p01/preprocessed_output/shimmer/gsr/gsr-01-01-12.csv"

data = []
with open(path, 'r') as file:
    reader = csv.reader(file, delimiter=',')
    i = 0
    for row in reader:
        if i == 0:
            # Skip header line
            i += 1
            continue
        data.append(np.array(row, dtype=np.float32))

data = np.array(data)
i=0
while True:
    if (i+1)*duration*sampling_rate > data.shape[0]:
        break
    extracted_data = data[i*duration*sampling_rate:(i+1)*duration*sampling_rate,:]
    print(extracted_data[:,0].shape, data.shape[0])
    #final_data["gsr"]["data"] = extracted_data[:,0]
    #final_data["gsr"]["sampling_rate"] = 128
    final_data["gsr"]["data"] = extracted_data[:,0]
    final_data["gsr"]["sampling_rate"] = 128
    result = predict(final_data)
    print(result)
    i += 1
