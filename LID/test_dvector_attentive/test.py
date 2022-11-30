import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import glob
import os
from speechbrain.pretrained import EncoderClassifier
import torchaudio
from tqdm import tqdm

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

if __name__ == '__main__':
    
    di = {0: "Hindi", 1: "English", 2: "Telugu"}
    l2i = {"en":1, "hi":0, "te":2}
    classifier = EncoderClassifier.from_hparams(source="./best_model", hparams_file='hparams_inference.yaml', savedir="./best_model")
    dev_clean_root = "/home/akhil/models/speechBrain/LID/data/voxLingua/new_dev"
    wav_files = glob.glob(os.path.join(dev_clean_root, "**/*.wav"), recursive=True)
    
    true = []
    pred = []
    x_vecs = []
    
    for utterance in tqdm(wav_files):
        filename = utterance
        lang = splitall(utterance)[9]
        true.append(l2i[lang])
        signal, fs =torchaudio.load(utterance)
        output_probs, score, index, text_lab = classifier.classify_batch(signal)
        pred.append(l2i[text_lab[0]])
        embeddings = classifier.encode_batch(signal)
        x_vecs.append(list(embeddings[0][0].numpy()))
    
    x_vecs = np.array(x_vecs)
    
    confusion = metrics.confusion_matrix(true, pred)
    
    correct = 0 
    for i in range(len(true)):
    	if true[i] == pred[i]:
    		correct += 1

    accuracy = correct/len(true)
    print("Accuracy : ",accuracy)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion, display_labels=["Hindi", "English", "Telugu"])
    display.plot()
    plt.savefig('tdnn_conf.png')
    
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(x_vecs)
    
    df_subset = pd.DataFrame()
    
    df_subset['Dimension 1'] = tsne_results[:,0]
    df_subset['Dimension 2'] = tsne_results[:,1]
    df_subset['True Labels'] = true
    df_subset = df_subset.replace({'True Labels': di})
    
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x='Dimension 1', y='Dimension 2',
        hue='True Labels',
        palette=sns.color_palette("hls", 4),
        data=df_subset,
        legend="full",
        alpha=0.7
    )
    plt.savefig('tdnn_tsne.png')
