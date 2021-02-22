import re
from  ltp  import  LTP
import time
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pandas as pd
import string
import nltk
import jieba
import unicodedata, re, itertools, sys
import camel_tools as camel
from camel_tools.utils.stringutils import force_encoding, force_unicode
from camel_tools.utils.charmap import CharMapper
from camel_tools.tokenizers.word import simple_word_tokenize
import demoji
demoji.download_codes()
nltk.download('punkt')

def zh_segment_jieba(chinese_df, batch_size=1024):
    """

    Takes 1 column dataframe containinig chinese sentences to segment and returns a 1 column dataframe
    of segmented chinese senetnces

    """
    ret = []
    for i in tqdm(range(len(chinese_df))):
        toks = jieba.cut(chinese_df.iloc[i])
        ret.append(toks)
    for i in range(len(ret)):
        ret[i] = "".join(word + " " for word in ret[i]).strip()
    # write tokens to dataframe
    chinese_df = pd.DataFrame({"zh": ret})
    return chinese_df

def zh_segment(chinese_df, batch_size=1024):
    """

    Takes 1 column dataframe containinig chinese sentences to segment and returns a 1 column dataframe
    of segmented chinese senetnces

    """
    ltp_model = LTP()
    i = 0
    print("---- tokenizing chinese sentences")

    ret = []
    with tqdm(total=len(chinese_df) // batch_size) as pbar:
        while i < len(chinese_df):
            if (i + batch_size > len(chinese_df)):
                toks = ltp_model.seg(chinese_df.iloc[i:].values.tolist())[0]
            else:
                toks = ltp_model.seg(chinese_df.iloc[i:i + batch_size].values.tolist())[0]
            ret.extend(toks)
            i += batch_size
            pbar.update(1)
    for i in range(len(ret)):
        ret[i] = "".join(word + " " for word in ret[i]).strip()
    # write tokens to dataframe
    chinese_df = pd.DataFrame({"zh": ret})
    return chinese_df

def ar_tokenize(df):
    """

    Takes a column df containing the sentences to tokenize using arabic specific tokenizer
    This is used only for languages that have space separated senteces


    """
    print("---- tokenizing arabic sentences")

    df = df.apply(lambda sentence: "".join(word + " " for word in simple_word_tokenize(sentence)).strip())
    return df

def tokenize(df):
    """

    Takes a column df containing the sentences to tokenize using nltk tokenize
    This is used only for languages that have space separated senteces


    """
    print("---- tokenizing sentences")

    df = df.apply(lambda sentence: "".join(word + " " for word in word_tokenize(sentence)).strip())
    return df

def remove_unwanted_symbols(sent: str):

    """

    remove unwanted symbols

    """
    
    symbols = '[\(\)|/#&@\[\]《》\{\}~<=>`——-]+'
    sent = re.sub(symbols, "", sent)
    return sent.strip()

def ar_clean(mapper, sent: str):
    """

    use cleaning utilities provided by camel-tools for arabic

    """
    sent = force_unicode(sent)
    return mapper.map_string(sent)

def remove_url(line: str):
    """
    
    Remove urls
    
    """
    pattern = re.compile(r'http[a-zA-Z0-9.?/&=:]*')
    return pattern.sub('', line.strip())

def remove_control_characters(s: str):
    """
    
    Remove control characters
    
    """
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

def remove_emoji(line: str):
    """
    
    replace emojis with their description
    
    """
    return demoji.replace_with_desc(line)

def apply_cleaning(sentence: str, lang: str, ar_mapper=None):

    if lang == "ar":
        assert ar_mapper is not None, "specified language is ar but no camel_tools.utils.charmap.CharMapper object given"
        return remove_url(
                    remove_control_characters(
                        remove_emoji(
                            remove_unwanted_symbols(
                                ar_clean(
                                    ar_mapper, sentence
                                )
                            )
                        )
                    )
                )
    else:
        return remove_url(
                    remove_control_characters(
                        remove_emoji(
                            remove_unwanted_symbols(
                                sentence
                            )
                        )
                    )
                )
