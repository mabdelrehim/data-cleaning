# data-cleaning

Functionality (main):

- Remove unwanted characters and emojis
- Remove pairs that are too different in length (threshold ratio >3)
- Chinese segmentation using jieba (ltp is implemented so you can edit the code to use ltp)
- Remove pairs that are have sentences that are not in the specified language (implemented using a fasttext langid model)

Functionality (laser_filter):

- Remove pairs that do not return a high enough similarity after applying laser embeddings. Laser tries to give sentences of the same meaning the same embeddings. Check this [publication](https://arxiv.org/abs/1812.10464)

Download required data

```bash
$ camel_data full
$ wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
$ python -m laserembeddings download-models
```

Run the help commands

```bash
$ python main.py -h
# usage: Cleaning bilingual corpora [-h] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE] [-s SRC_LANG] [-t TRG_LANG] [-p PREFIXES]

# optional arguments:
#   -h, --help            show this help message and exit
#   --data_dir DATA_DIR   the director(ies) containing the files to be cleaned (comma-separated, no spaces)
#   --batch_size BATCH_SIZE
#                         batch size to use if using the LTP chinese segmenter (otherwise ignored) (used if one of src_lang or trg_lang is zh)
#   -s SRC_LANG, --src_lang SRC_LANG
#                         source language code (default=ar)
#   -t TRG_LANG, --trg_lang TRG_LANG
#                         target language code (default=zh)
#   -p PREFIXES, --prefixes PREFIXES
#                         file prefixe(s): comma separated without spaces (dafault=data)

```

```bash
$ python laser_filter.py -h
# usage: Cleaning bilingual corpora [-h] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE] [--laser_thresh LASER_THRESH] [-s SRC_LANG] [-t TRG_LANG] [-p PREFIXES]

# optional arguments:
#   -h, --help            show this help message and exit
#   --data_dir DATA_DIR   the director(ies) containing the files to be cleaned (comma-separated, no spaces)
#   --batch_size BATCH_SIZE
#                         batch size to use with laser filter
#   --laser_thresh LASER_THRESH
#                         threshold percentage for laser similarity (0-100)
#   -s SRC_LANG, --src_lang SRC_LANG
#                         source language code (default=ar)
#   -t TRG_LANG, --trg_lang TRG_LANG
#                         target language code (default=zh)
#   -p PREFIXES, --prefixes PREFIXES
#                         file prefixe(s): comma separated without spaces (dafault=data)
```

Some parts run on gpu. GPU usage is still not optimal but I'll get to that later when I have the time.
