import argparse
from pathlib import Path
import pandas as pd
import os
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text  # Needed for loading universal-sentence-encoder-cmlm/multilingual-preprocess



def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--data_dir", 
                            type=str, 
                            help="the director(ies) containing the files to be cleaned (comma-separated, no spaces)")
    parser.add_argument("--batch_size", 
                            type=int,
                            default= 4096, 
                            help="batch size to use with laser filter")
    parser.add_argument("--thresh", 
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

def normalization(embeds):
  norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
  return embeds/norms




def batched_laBSE_filter(preprocessor, encoder, sources, targets, batch_size=50):
    
    start = time.time()
    i = 0
    embeddings_a = []
    with tqdm(total=len(sources) // batch_size) as pbar:
        while i < len(sources):

            embeddings_a.extend(
                encoder(preprocessor(sources[i:i+batch_size]))["default"]
            )
            i = i + batch_size
            pbar.update(1)

    end = time.time()
    elapsed = end - start
    print(f"---- Calculated laBSE embeddings for {len(sources)} sentences from source language in {elapsed // 60},  mins , {elapsed % 60} ,  secs")
    print(np.array(embeddings_a).shape)
    
    start = time.time()
    i = 0
    embeddings_b = []
    with tqdm(total=len(targets) // batch_size) as pbar:
        while i < len(targets):

            embeddings_b.extend(
                encoder(preprocessor(targets[i:i+batch_size]))["default"]
            )
            i = i + batch_size
            pbar.update(1)

    end = time.time()
    elapsed = end - start
    print(f"---- Calculated laBSE embeddings for {len(targets)} sentences from target language in {elapsed // 60},  mins , {elapsed % 60},  secs")
    print(np.array(embeddings_b).shape)
    similarities = []
    start = time.time()
    for a, b in tqdm(zip(embeddings_a, embeddings_b), total=len(embeddings_a)):
        similarities.append(cosine_similarity([a], [b]).flatten()[0])
    end = time.time()
    elapsed = end - start
    print(f"---- Calculated cosine for {len(targets)} pairs in {elapsed // 60},  mins , {elapsed} % 60,  secs")
    return similarities





def main(args):
    preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
    encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")
    for dataset in args.data_dir.split(","):
        start_all = time.time()
        for subset in args.prefixes.split(","):
            print(f'-- Applying laBSE filter to {dataset}/{subset}')

            data_file_src = open(os.path.join(dataset, subset + "." + args.src_lang + ".clean"), "r")
            data_file_trg = open(os.path.join(dataset, subset + "." + args.trg_lang + ".clean"), "r")

            src_sents = data_file_src.readlines()
            trg_sents = data_file_trg.readlines()

            df = pd.DataFrame(
                {
                    args.src_lang: src_sents,
                    args.trg_lang: trg_sents,
                    "similarities": batched_laser_filter(preprocessor, encoder, tf.constant(src_sents), tf.constant(trg_sents), batch_size=args.batch_size)
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
            dest_file_src = open(os.path.join(dataset, subset + "." + args.src_lang + ".clean_filtered_laBSE"), "w")
            dest_file_trg = open(os.path.join(dataset, subset + "." + args.trg_lang + ".clean_filtered_laBSE"), "w")
            dest_file_src.writelines(df[args.src_lang].values.tolist())
            dest_file_trg.writelines(df[args.trg_lang].values.tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Cleaning bilingual corpora', parents=[get_args_parser()])
    args = parser.parse_args()
    
    assert not (args.src_lang == args.trg_lang), "source and target languages should be different"
    main(args)
