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
from spectralcluster import SpectralClusterer
                
if __name__ == '__main__':
	
	clusterer = SpectralClusterer(
	    min_clusters=2,
	    max_clusters=4
	    )
    
	wavfile = "./diarization_test_files/te-hi-en_16k.wav"
	classifier = EncoderClassifier.from_hparams(source="./best_model", hparams_file='hparams_inference.yaml', savedir="./best_model")
	signal, fs = torchaudio.load(wavfile)
	
	win_length = int(16000*3) # 1 second segment
	hop = int(16000*0.4) # 0.2 second segment hop
	
	wav = signal
	
	embeddings = []
	data = []
	
	for j in tqdm(range(0, wav.shape[-1]-win_length, hop)):
		inp = wav[:,j:j+win_length]
		output_probs, score, index, text_lab = classifier.classify_batch(inp)
		emb = classifier.encode_batch(inp)[0][0]
		lang_prediction = text_lab[0]
		embeddings.append(list(emb.numpy()))
		timings_start = (j/16000)
		timings_end = ((j+win_length)/16000)
		data.append([timings_start, timings_end, lang_prediction])
	
	diary = pd.DataFrame(data, columns = ['start','end','language'])
	
	embeddings = np.array(embeddings)
	labels = clusterer.predict(embeddings)
		
	print(diary)
