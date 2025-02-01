import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import matplotlib
plt.rcParams.update({'font.size': 50})
matplotlib.rc('xtick', labelsize=50)
matplotlib.rc('ytick', labelsize=50)

def ccc(gold, pred):
    '''
    Function adapted from https://arxiv.org/abs/2003.10724

    Parameters
    ----------
    gold : array
        True measurements.
    pred : array
        Predicted measurements.

    Returns
    -------
    ccc : float (-1 to 1 )
        correlation score.

    '''
    gold_mean  = np.mean(gold)
    pred_mean  = np.mean(pred)
    covariance = np.mean((gold-gold_mean)*(pred-pred_mean))
    gold_var   = np.mean((gold-gold_mean)**2)
    pred_var   = np.mean((pred-pred_mean)**2)
    ccc        = 2 * covariance / (gold_var + pred_var + (gold_mean - pred_mean)**2 + 1e-16)
    return ccc

wdw = 256

dnn_results = pd.read_csv("optimized-DNN_hoffman_results.csv")
dnn_results_normal = pd.read_csv("DNN_hoffman_results.csv")

vols = dnn_results['vol'].unique()

# Add uma nova linha com o Original e Optimizado


for i in range(len(vols)):
    print("\nVoluntário",vols[i])
    test_vol =  vols[i]
    batch = dnn_results[dnn_results['vol'] == test_vol].copy()
    pred = batch['pred']
    y_test = batch['true']

    rmse = np.sqrt(mse(batch['pred'],batch['true']))
    corr = np.corrcoef((batch[['pred','true']].values.T))[0,1]
    mymae = mae(batch['pred'],batch['true'])

    class_true = np.zeros([len(batch)])
    class_pred = np.zeros([len(batch)])
    class_true[batch['true']<90] = 1
    class_pred[batch['pred']<90] = 1


    print("RMSE:", rmse)
    print("CORRELACAO:", corr)
    print("CCC:", ccc(batch['pred'],batch['true']))
    print("MAE:", mymae)
    print("std", np.std(batch['true'] - batch['pred']))
    print("median", np.median(batch['true'] - batch['pred']))
    print("accuracy_score",accuracy_score(class_true,class_pred))
    print("precision_score",precision_score(class_true,class_pred))
    print("recall_score",recall_score(class_true,class_pred))
    print("f1_score",f1_score(class_true,class_pred))



    fig, axs= plt.subplots( figsize=(18, 12), dpi=100)
    #plt.figure()
    #plt.scatter(pred*100,np.array(y_test)*100)
    plt.plot(np.arange(0, len(pred)*15,15), pred, 'r', label = 'Prediction', linewidth=4)
    plt.plot(np.arange(0, len(y_test)*15,15), y_test, 'b', label = 'Reference', linewidth=4)
    plt.xlim([0,3500])
    plt.ylim([60,105])
    plt.xlabel('Time (s)')
    plt.ylabel(r'SpO$_2$ (%)')
    plt.title("Volunteer " + str(int(vols[i].split('-')[0])%100000))
    plt.tight_layout()
    plt.legend()
    plt.savefig('optimized'+str(int(vols[i].split('-')[0]))+'.png')
    # plt.savefig(str(int(vols[i].split('-')[0]))+'.png')


fig_all, axs= plt.subplots( figsize=(24, 24), dpi=100)
for i in range(len(vols)):
    print("\nVoluntário",vols[i])
    test_vol =  vols[i]
    batch = dnn_results[dnn_results['vol'] == test_vol].copy()
    pred = batch['pred']
    y_test = batch['true']
    #plt.figure()
    #plt.scatter(pred*100,np.array(y_test)*100)
    plt.scatter(pred, y_test, s=200, label = 'vol. '+str(i+1))
    plt.xlim([63,103])
    plt.ylim([63,103])
    plt.xlabel(r'Prediction - SpO$_2$ (%)')
    plt.ylabel(r'Reference - SpO$_2$ (%)')
plt.tight_layout()
plt.legend()
plt.savefig('optimized-scatter_all.png')
# plt.savefig('scatter_all.png')






plt.figure(figsize=(18, 12), dpi=100)
plt.scatter(dnn_results['pred'],dnn_results['true'])
plt.xlabel(r'Prediction - SpO$_2$ (%)')
plt.ylabel(r'Reference - SpO$_2$ (%)')
#plt.title("RMSE: {:.2f}".format(rmse)+" || "+"Pearson Coef.: {:.2f}".format(corr))
plt.tight_layout()
#plt.savefig('scatter_all.png')
plt.show()

rmse = np.sqrt(mse(dnn_results['pred'],dnn_results['true']))
corr = np.corrcoef((dnn_results[['pred','true']].values.T))[0,1]
mymae = mae(dnn_results['pred'],dnn_results['true'])

print("\nTotal")
print("RMSE:", rmse)
print("CORRELACAO:", corr)
print("CCC:", ccc(dnn_results['pred'],dnn_results['true']))
print("MAE:", mymae)
print("std", np.std(dnn_results['true'] - dnn_results['pred']))
print("median", np.median(dnn_results['true'] - dnn_results['pred']))

class_true = np.zeros([len(dnn_results),2])
class_true[dnn_results['true']<90,0] = 1
class_true[dnn_results['pred']<90,1] = 1

print(classification_report(y_true = class_true[:,0], y_pred=class_true[:,1]))
print(confusion_matrix(y_true = class_true[:,0], y_pred=class_true[:,1]))


print("accuracy_score",accuracy_score(y_true = class_true[:,0], y_pred=class_true[:,1]))
print("precision_score",precision_score(y_true = class_true[:,0], y_pred=class_true[:,1]))
print("recall_score",recall_score(y_true = class_true[:,0], y_pred=class_true[:,1]))
print("f1_score",f1_score(y_true = class_true[:,0], y_pred=class_true[:,1]))