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

if __name__ == '__main__':
    classifier = CustomEncoderClassifier.from_hparams(source="./best_model", hparams_file='hparams_inference.yaml', savedir="./best_model")
    wavfile = "./diarization_test_files/HinEng_Codeswitch.wav"
    
    #signal, fs = torchaudio.load(wavfile)
    #output_probs, score, index, text_lab = classifier.classify_batch(signal)
    #emb = classifier.encode_batch(signal)[0][0]
    
    frame_embeddings = classifier.framelevel_vecs(wavfile).squeeze(0)
    print(frame_embeddings.shape)
