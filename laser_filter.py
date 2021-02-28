import argparse
from pathlib import Path
import pandas as pd
import os
import time
from multiprocessing import Pool
from utils import bitext_utils as bitext_utils
from utils import sentence_utils as sentence_utils
from my_helping_functions import parallelize_np_array, parallelize_list, serialize_list
import fasttext
from laserembeddings import Laser
import camel_tools as camel
from camel_tools.utils.charmap import CharMapper
import argparse
from pathlib import Path
import fasttext
from laserembeddings import Laser
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--data_dir", 
                            type=str, 
                            help="the director(ies) containing the files to be cleaned (comma-separated, no spaces)")
    parser.add_argument("--batch_size", 
                            type=int,
                            default= 4096, 
                            help="batch size to use with laser filter")
    parser.add_argument("--laser_thresh", 
                            type=int, 
                            help="threshold percentage for laser similarity (0-100)")
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
                            default="data")

    return parser

def my_cosine_similarity(embeddings_a, embeddings_b):
    similarities=[]
    for a, b in zip(embeddings_a, embeddings_b):
        similarities.append(cosine_similarity([a], [b]).flatten()[0])
    return similarities

def laser_filter(laser_model, sources, targets, src_lang: str, trg_lang: str, batch_size=2048):
    """
    
    The laser model tries to embed sentences that are of the same meaning with similar embeddings
    basic experimentation tells that sentences with cosine similarity score >= 0.8 are of acceptable quality
    
    """
    start = time.time()
    i = 0
    embeddings_a = []
    #with tqdm(total=len(sources) // batch_size) as pbar:
    while i < len(sources):
        embeddings_a.extend(laser_model.embed_sentences(
        sources[i:i+batch_size],
        lang=src_lang))
        i = i + batch_size
        #    pbar.update(1)
    end = time.time()
    elapsed = end - start
    #print(f"---- Calculated laser embeddings for {len(sources)} sentences from source language in {elapsed // 60},  mins , {elapsed % 60} ,  secs")
    #print(np.array(embeddings_a).shape)
    start = time.time()
    i = 0
    embeddings_b = []
    #with tqdm(total=len(targets) // batch_size) as pbar:
    while i < len(targets):
        embeddings_b.extend(laser_model.embed_sentences(
        targets[i:i+batch_size],
        lang=trg_lang))
        i = i + batch_size
        #    pbar.update(1)
    end = time.time()
    elapsed = end - start
    #print(f"---- Calculated laser embeddings for {len(targets)} sentences from target language in {elapsed // 60},  mins , {elapsed % 60},  secs")
    #print(np.array(embeddings_b).shape)
    similarities = []
    start = time.time()
    for a, b in zip(embeddings_a, embeddings_b):
        similarities.append(cosine_similarity([a], [b]).flatten()[0])
    
    #pool=Pool()
    #embeddings_a, embedding_b=parallelize_np_array(embeddings_a, os.cpu_count()), parallelize_np_array(embeddings_b, os.cpu_count())
    #similarities=serialize_list(list(pool.starmap(my_cosine_similarity, zip(embeddings_a, embeddings_b))))
    #pool.close()
    #pool.join()
    end = time.time()
    elapsed = end - start
    #print(f"---- Calculated cosine for {len(targets)} pairs in {elapsed // 60},  mins , {elapsed} % 60,  secs")
    return similarities

def batched_laser_filter(laser_model, sources, targets, src_lang: str, trg_lang: str, cosine_batch_size=-1,  batch_size=2048):
    """
    This function is made to decrease memory used by laser filtering.
    """
    if cosine_batch_size==-1:
        cosine_batch_size=2*batch_size

    num_batches=len(sources)//cosine_batch_size
    sources, targets = parallelize_list(sources, num_batches), parallelize_list(targets, num_batches)
    similarities=[]

    with tqdm(total=num_batches) as progress_bar:
        for src_batch, trg_batch in zip(sources, targets):
            similarities.append(laser_filter(laser_model, src_batch, trg_batch, src_lang, trg_lang, batch_size))
            progress_bar.update(1)

    return serialize_list(similarities)


def main(args):
    laser = Laser()
    for dataset in args.data_dir.split(","):
        start_all = time.time()
        for subset in args.prefixes.split(","):
            print(f'-- Applying laser filter to {dataset}/{subset}')

            data_file_src = open(os.path.join(dataset, subset + "." + args.src_lang + ".clean"), "r")
            data_file_trg = open(os.path.join(dataset, subset + "." + args.trg_lang + ".clean"), "r")

            src_sents = data_file_src.readlines()
            trg_sents = data_file_trg.readlines()

            df = pd.DataFrame(
                {
                    args.src_lang: src_sents,
                    args.trg_lang: trg_sents,
                    "similarities": batched_laser_filter(laser, src_sents, trg_sents, args.src_lang, args.trg_lang, batch_size=args.batch_size)
                }
            )

            print(f'---- Number of samples in {dataset}/{subset} before cleaning: ', len(df))
            before = len(df)
            start = time.time()
            df = df[df['similarities'] > (args.laser_thresh / 100)]
            df = df[[args.src_lang, args.trg_lang]]
            df = df.reset_index()
            end = time.time()
            after = len(df)
            removed = before - after
            elapsed = end - start
            print("---- Removed ", removed, " pairs that are too different in lengths", elapsed // 60, " mins ", elapsed % 60, " secs")
            dest_file_src = open(os.path.join(dataset, subset + "." + args.src_lang + ".clean_filtered"), "w")
            dest_file_trg = open(os.path.join(dataset, subset + "." + args.trg_lang + ".clean_filtered"), "w")
            dest_file_src.writelines(df[args.src_lang].values.tolist())
            dest_file_trg.writelines(df[args.trg_lang].values.tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Cleaning bilingual corpora', parents=[get_args_parser()])
    args = parser.parse_args()
    
    assert not (args.src_lang == args.trg_lang), "source and target languages should be different"
    main(args)
