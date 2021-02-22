import re
from  ltp  import  LTP
import time
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pandas as pd
import string
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt') 

def is_one_empty(src_sent: str, trg_sent: str):
    if len(src_sent) == 0 or len(trg_sent) == 0:
        return True
    else:
        return False

def non_aligned_len(src_sent: str, trg_sent: str):
    """
    
    Takes a pair of tokenized/segmented sentences and returns true if the pair should be removed
    ratio of larger sentence to smaller sentence = 3
    
    """

    # first, remove punctiuation
    len_src = len(word_tokenize(src_sent.translate(str.maketrans('', '', string.punctuation))))
    len_trg = len(word_tokenize(trg_sent.translate(str.maketrans('', '', string.punctuation))))

    if ((len_src > len_trg) and ((len_src / (len_trg + 0.1)) >= 3)) or  ((len_trg > len_src) and ((len_trg / (len_src + 0.1)) >= 3)):
        return True
    else:
        return False
    
def lang_detect(src_sent: str, trg_sent: str, src_lang: str, trg_lang: str , fasttext_model):

    """

    Takes a parallel pair of sentences in a parallel corpus and returns whether to remove the pair
    or not based on
        1- A large portion of the source sentence is in a language other than the source language
        2- A large portion of the target sentence is in a language other than the target language

    """

    detected_src, confidence_src = fasttext_model.predict(src_sent, k=1)
    detected_src = detected_src[0].replace("__label__", "")
    confidence_src = confidence_src[0]

    detected_trg, confidence_trg = fasttext_model.predict(trg_sent, k=1)
    detected_trg = detected_trg[0].replace("__label__", "")
    confidence_trg = confidence_trg[0]

    #print(detected_src,"|||||||",detected_src)

    if (not (detected_src == src_lang)) or (not (detected_trg == trg_lang)):
        return True
    elif (confidence_src < 0.5) or (confidence_trg < 0.5):
        return True
    else:
        return False
        
def laser_filter(laser_model, src: str, trg: str, src_lang: str, trg_lang: str):
    """
    
    The laser model tries to embed sentences that are of the same meaning with similar embeddings
    basic experimentation tells that sentences with cosine similarity score >= 0.8 are of acceptable quality
    
    """
    embeddings_a = laser_model.embed_sentences(
        [src],
        lang=src_lang)
    embeddings_b = laser_model.embed_sentences(
        [trg],
        lang=trg_lang)
    similarity_score = cosine_similarity(embeddings_a, embeddings_b).flatten()[0]
    if similarity_score < 0.5:
        return True
    else:
        return False