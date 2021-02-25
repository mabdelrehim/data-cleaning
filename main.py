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
                            help="batch size to use when using the LTP chinese segmenter (used if one of src_lang or trg_lang is zh)")
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

def main(args):

    fasttext_model = fasttext.load_model('lid.176.bin')
    laser = Laser()
    ar_character_mapper = CharMapper.builtin_mapper('arclean')

    for dataset in args.data_dir.split(","):
        print(f'Cleanining {dataset}')
        start_all = time.time()
        for subset in args.prefixes.split(","):
            print(f'--Cleanining {subset}')


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


            # Apply sentence level cleaning
            start = time.time()
            df[args.src_lang] = df[args.src_lang].apply(
                    lambda sentence: sentence_utils.apply_cleaning(
                        sentence, args.src_lang, ar_character_mapper
                    )
                )
            df[args.trg_lang] = df[args.trg_lang].apply(
                    lambda sentence: sentence_utils.apply_cleaning(
                        sentence, args.trg_lang, ar_character_mapper
                    )
                )
            end = time.time()
            elapsed = end - start
            print("---- Removing unwanted characters for took ", elapsed // 60, " mins ", elapsed % 60, " secs")

            # remove pairs that are empty after applying sentence level
            before = len(df)
            start = time.time()
            df['to_del'] = df.apply(lambda x: bitext_utils.is_one_empty(x[args.src_lang], x[args.trg_lang]), axis=1)
            df = df[df['to_del'] == False]
            df = df[[args.src_lang, args.trg_lang]]
            df = df.reset_index()
            end = time.time()
            after = len(df)
            removed = before - after
            elapsed = end - start
            print("---- Removed ", removed, " pairs that became empty after removing unwanted characters in ", elapsed // 60, " mins ", elapsed % 60, " secs")

            # remove pairs that contain too many tokens from an external language
            before = len(df)
            start = time.time()
            df['to_del'] = df.apply(lambda x: bitext_utils.lang_detect(x[args.src_lang], x[args.trg_lang], args.src_lang, args.trg_lang, fasttext_model), axis=1)
            df = df[df['to_del'] == False]
            df = df[[args.src_lang, args.trg_lang]]
            df = df.reset_index()
            end = time.time()
            after = len(df)
            removed = before - after
            print("---- Removed ", removed, " pairs that contain too many tokens from an external language in ", elapsed // 60, " mins ", elapsed % 60, " secs")
        
            # segmentation and/or tokenization (only non space separated language supported is chinese)
            start = time.time()
            if args.src_lang == "zh":
                df[args.src_lang] = sentence_utils.zh_segment_jieba(df[args.src_lang], batch_size=args.batch_size)
                df[args.trg_lang] = sentence_utils.tokenize(df[args.trg_lang])
            elif args.trg_lang == "zh":
                df[args.trg_lang] = sentence_utils.zh_segment_jieba(df[args.trg_lang], batch_size=args.batch_size)
                df[args.src_lang] = sentence_utils.tokenize(df[args.src_lang])
            else:
                df[args.src_lang] = sentence_utils.tokenize(df[args.src_lang])
                df[args.trg_lang] = sentence_utils.tokenize(df[args.trg_lang])
            end = time.time()
            elapsed = end - start
            print("---- Segmentation and/or tokenization took ", elapsed // 60, " mins ", elapsed % 60, " secs")

            # apply length alignment filter
            before = len(df)
            start = time.time()
            df['to_del'] = df.apply(lambda x: bitext_utils.non_aligned_len(x[args.src_lang], x[args.trg_lang]), axis=1)
            df = df[df['to_del'] == False]
            df = df[[args.src_lang, args.trg_lang]]
            df = df.reset_index()
            end = time.time()
            after = len(df)
            removed = before - after
            elapsed = end - start
            print("---- Removed ", removed, " pairs that are too different in lengths", elapsed // 60, " mins ", elapsed % 60, " secs")

            """
            # apply laser filter
            before = len(df)
            start = time.time()
            df['to_del'] = df.apply(lambda x: bitext_utils.laser_filter(laser, x[args.src_lang], x[args.trg_lang], args.src_lang, args.trg_lang), axis=1)
            df = df[df['to_del'] == False]
            df = df[[args.src_lang, args.trg_lang]]
            df = df.reset_index()
            end = time.time()
            after = len(df)
            removed = before - after
            elapsed = end - start
            print("---- Removed ", removed, " pairs that have low laser cosine similarity in ", elapsed // 60, " mins ", elapsed % 60, " secs")
            """

            # remove duplicates in source language
            before = len(df)
            start = time.time()
            df = df.drop_duplicates(subset=[args.src_lang])
            # remove duplicates in target language
            df = df.drop_duplicates(subset=[args.trg_lang])
            df = df.reset_index()
            end = time.time()
            after = len(df)
            rem_duplicates = before - after
            elapsed = end - start
            print("---- Removed ", rem_duplicates, " pairs that are duplicates in source/target language in ", elapsed // 60, " mins ", elapsed % 60, " secs")
            print("---- Number of samples in ", subset, " after cleaning: ", len(df))
            # add new line character after every sentence
            df[args.src_lang] = df[args.src_lang].apply(lambda x: x + "\n")
            df[args.trg_lang] = df[args.trg_lang].apply(lambda x: x + "\n")
            dest_file_src = open(os.path.join(dataset, subset + "." + args.src_lang + ".clean"), "w")
            dest_file_trg = open(os.path.join(dataset, subset + "." + args.trg_lang + ".clean"), "w")
            dest_file_src.writelines(df[args.src_lang].values.tolist())
            dest_file_trg.writelines(df[args.trg_lang].values.tolist())
        end_all = time.time()
        elapsed = end_all - start_all
        print(f"{dataset} cleaning took ", elapsed // 60, " mins ", elapsed % 60, " secs")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Cleaning bilingual corpora', parents=[get_args_parser()])
    args = parser.parse_args()
    
    assert not (args.src_lang == args.trg_lang), "source and target languages should be different"
    main(args)

