import json
import numpy as np
import matplotlib.pyplot as plt

experiment_folder = './results/faster_rcnn_R_50_FPN_3x/lr_0_001_iter_5000_batch_512/0/'

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(experiment_folder + 'metrics.json')

train_loss = {}
for x in experiment_metrics:
    if 'total_loss' in x:
            train_loss[x['iteration']] = x['total_loss']


x1=[]
y1=[]


print(train_loss)
for k, v in train_loss.items():
    x1.append(k)
    y1.append(np.mean(np.array(v)))

print(len(x1))
print(len(y1))
plt.plot(x1,y1, color="blue", label="Faster R-CNN")

plt.savefig(experiment_folder+'total_loss.png')
