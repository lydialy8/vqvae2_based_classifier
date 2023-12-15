import os
import numpy as np
from sklearn import metrics
import pickle

output = './results_18_11_2021/mixed_4'
outputs = ['./results_18_11_2021/scand_4on13','./results_18_11_2021/china_4on13','./results_18_11_2021/lc_4on13']

output = './results_11_11_2021/mixed_4'
outputs = ['./results_11_11_2021/scand_4','./results_11_11_2021/china_4','./results_11_11_2021/LC_4','./results_11_11_2021/alps_4']
channels_nb  = 4


output = './results_23_02_2022/mixed_13'
outputs = ['./results_23_02_2022/LC_13','./results_23_02_2022/scand_13','./results_23_02_2022/china_13']#
channels_nb  = 13


#outputs = ['./results_18_11_2021/scand_4','./results_18_11_2021/china_4','./results_18_11_2021/LC_4','./results_18_11_2021/alps_4']
#channels_nb  = 4

labels=[]
metric=[]
thresholds_mixed=[]

def getThreshold(dir, i):
    with open(os.path.join(dir, f'B{i}', 'pkls', "labels_all.pkl"), "rb") as f:
        labels = np.array(pickle.load(f))
    with open(os.path.join(dir, f'B{i}', 'pkls', "metrics_all.pkl"), "rb") as f:
        metric = np.array(pickle.load(f))
    fpr, tpr, thresholds = metrics.roc_curve(labels, metric, pos_label=0)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    threshold10 = np.interp(0.1, fpr, thresholds)
    threshold5 = np.interp(0.05, fpr, thresholds)
    return optimal_threshold, threshold10, threshold5


Threshold = []
for i in range(0, channels_nb):
    Threshold.append(getThreshold('./results_23_02_2022/snow',i))
    Threshold.append(getThreshold('./results_23_02_2022/vegetation',i))
    Threshold.append(getThreshold('./results_23_02_2022/barren',i))

    labels = []
    metric = []
    labels_test = []
    metrics_test = []
    imgs_test = []
    for j in range(len(outputs)):
        with open(os.path.join(outputs[j], f'B0', 'pkls', "class_all.pkl"), "rb") as f:
            img_class = np.array(pickle.load(f))
        with open(os.path.join(outputs[j], f'B{i}', 'pkls', "labels_all.pkl"), "rb") as f:
            labels.append(pickle.load(f))
        with open(os.path.join(outputs[j],f'B{i}', 'pkls',  "metrics_all.pkl"), "rb") as f:#with open(os.path.join(outputs[j],f'B{i}', 'pkls', "metrics_all.pkl"), "rb") as f:
            metric.append(pickle.load(f))
        threshold_label = np.array(labels[j])
        threshold_metrics = np.array(metric[j])

        labels_test.append(threshold_label)
        metrics_test.append(threshold_metrics)
        img_class = [int(ele) if type(int(ele)) ==int and int(ele) != 3 else 1 for ele in img_class]
        imgs_test.append(img_class)
        predicted_label = list(map(lambda x, y: 0 if x > Threshold[y][0] else 1, threshold_metrics, img_class))
        print(f"accuracy {metrics.accuracy_score(threshold_label, predicted_label)}")
        tn, fp, fn, tp = metrics.confusion_matrix(threshold_label, predicted_label).ravel()
        #print(f"tpr {tp /(tp+fn)}")
        #print(f"fpr {fp /(fp+tn)}")

    labels_test = [item for sublist in labels_test for item in sublist]
    metrics_test = [item for sublist in metrics_test for item in sublist]
    img_classes= [item for sublist in imgs_test for item in sublist]

    predicted_label = list(map(lambda x, y: 0 if x > Threshold[y][0] else 1, metrics_test,img_classes))
    print(f"accuracy {metrics.accuracy_score(labels_test, predicted_label)}")
    tn, fp, fn, tp = metrics.confusion_matrix(labels_test, predicted_label).ravel()
    #print(f"tpr {tp /(tp+fn)}")
    #print(f"fpr {fp /(fp+tn)}")