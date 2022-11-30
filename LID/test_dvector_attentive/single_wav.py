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
from custom_interface import CustomEncoderClassifier
import torchaudio
from tqdm import tqdm
from spectralcluster import SpectralClusterer

if __name__ == '__main__':
    clusterer = SpectralClusterer(
	    min_clusters=2,
	    max_clusters=3
	    )
    classifier = CustomEncoderClassifier.from_hparams(source="./best_model", hparams_file='hparams_inference.yaml', savedir="./best_model")
    wavfile = "./diarization_test_files/te-hi-en_16k.wav"
    
    #signal, fs = torchaudio.load(wavfile)
    #output_probs, score, index, text_lab = classifier.classify_batch(signal)
    #emb = classifier.encode_batch(signal)
    
    feature = classifier.framelevel_vecs(wavfile).squeeze(0).numpy()[:2000,:]
    print(feature.shape)
    #labels = clusterer.predict(frame_embeddings[:1000,:])
    #print(labels)
    #plt.plot(np.linspace(0,10, len(labels)), labels)
    #plt.show()
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(feature)

    df_subset = pd.DataFrame()
	
    df_subset['Dimension 1'] = tsne_results[:,0]
    df_subset['Dimension 2'] = tsne_results[:,1]
    df_subset['True Labels'] = np.ones(feature.shape[0])
    plt.figure(figsize=(16,10))
    sns.scatterplot(
	    x='Dimension 1', y='Dimension 2',
	    hue='True Labels',
	    palette=sns.color_palette("hls", 4),
	    data=df_subset,
	    legend="full",
	    alpha=0.7
    )
    plt.show()
    

