import json
import numpy as np
import matplotlib.pyplot as plt

experiment_folder = '/home/group02/week2/results/task_d/to_plot/faster_rcnn_R_50_FPN_3x/lr_0001/batch_size_128/'

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(experiment_folder + 'metrics.json')

train_loss = {}
flag_train = False
for x in experiment_metrics:
    if flag_train:
        if 'total_loss' in x:
                train_loss[x['iteration']] = x['total_loss']

    if 'validation_loss' in x:
        flag_train = True

x1=[]
y1=[]


print(train_loss)
for k, v in train_loss.items():
    x1.append(k)
    y1.append(np.mean(np.array(v)))

print(len(x1))
print(len(y1))
plt.plot(x1,y1, color="blue", label="Train Loss")


validation_loss= {}
flag_val = False
for x in experiment_metrics:
    if flag_val:
        if 'validation_loss' in x:
            validation_loss[x['iteration']] = x['validation_loss']

    if 'validation_loss' in x:
        flag_val = True

x2=[]
y2=[]
for k, v in validation_loss.items():
    x2.append(k)
    y2.append(np.mean(np.array(v)))
print(x2)
print(y2)

plt.plot(x2, y2, color="orange", label="Val Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tick_params(axis='y')
plt.title('Batch Size: 128')
plt.legend(loc='upper right')

plt.savefig(experiment_folder+'bs_128.png')
