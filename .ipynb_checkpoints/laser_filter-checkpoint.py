import argparse
from pathlib import Path
import pandas as pd
import os
import time
from utils import bitext_utils as bitext_utils
from utils import sentence_utils as sentence_utils
import fasttext
from laserembeddings import Laser
import camel_tools as camel
from camel_tools.utils.charmap import CharMapper


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--data_dir", 
                            type=str, 
                            help="the director(ies) containing the files to be cleaned (comma-separated, no spaces)")
    parser.add_argument("--batch_size", 
                            type=int,
                            default= 4096, 
                            help="batch size to use with laser filteer")
    parser.add_argument("-s",
                            "--src_lang", 
                            type=str, 
                            help="source language code (default=ar)", 
                            default="ar")
    parser.add_argument("-t", 
                            "--trg_lang", 
                            type=str, 
                            help="target language code (default=zh)", 
                            default="zh")
    parser.add_argument("-p", 
                            "--prefixes", 
                            type=str, 
                            help="file prefixe(s): comma separated without spaces (dafault=data)", 
                            default="data.clean")

    return parser

def batched_laser_filter(laser_model, sources, targets, src_lang: str, trg_lang: str, batch_size=128):
    """
    
    The laser model tries to embed sentences that are of the same meaning with similar embeddings
    basic experimentation tells that sentences with cosine similarity score >= 0.8 are of acceptable quality
    
    """
    start = time.time()
    i = 0
    embeddings_a = []
    with tqdm(total=len(sources) // batch_size) as pbar:
        while i < len(sources):
            embeddings_a.extend(aser_model.embed_sentences(
            sources[i:i+batch_size],
            lang=src_lang))
            i = i + batch_size
            pbar.update(1)
    end = time.time()
    print(f"---- Calculated laser embeddings for {len(sources)} sentences from source language in {elapsed // 60},  mins , {elapsed} % 60,  secs")

    
    similarity_score = cosine_similarity(embeddings_a, embeddings_b).flatten()[0]
    if similarity_score < 0.5:
        return True
    else:
        return False

def main.py():
    laser = Laser()
    for dataset in args.data_dir.split(","):
        start_all = time.time()
        for subset in args.prefixes.split(","):
            print(f'-- Applying laser filter to {dataset}/{subset}')

            data_file_src = open(os.path.join(dataset, subset + "." + args.src_lang), "r")
            data_file_trg = open(os.path.join(dataset, subset + "." + args.trg_lang), "r")

            src_sents = data_file_src.readlines()
            trg_sents = data_file_trg.readlines()

            df = pd.DataFrame(
                {
                    args.src_lang: src_sents,
                    args.trg_lang: trg_sents
                }
            )

            print(f'---- Number of samples in {dataset}/{subset} before cleaning: ', len(df))


