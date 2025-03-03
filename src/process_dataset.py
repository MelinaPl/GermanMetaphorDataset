"""
@author: Melina Plakidis
"""
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import os
import spacy
from cassis import *
from pathlib import Path

"""
This script is used to process the chosen text files of gpt3-5, gpt4o and humans
and converts them to uima-cas files containing automatically assigned POS-tags. The 
resulting files can then be used to import them into INCEpTION.
"""

# Load German spacy model
nlp = spacy.load("de_core_news_lg")
# Choose relevant POS-Tags
relevant_pos_tags = ["NN", "ADJA", "ADJD", "VVFIN", "VVIMP", "VVINF", "VVIZU", "VVPP", "TRUNC", "NNE", "PTKVZ"]

# Load type system
with open("full-typesystem.xml", "rb") as f:
  ts = load_typesystem(f)

# Create the CAS
SENTENCE_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
TOKEN_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
POS_TYPE = "webanno.custom.POS"


def write_uima_cas_file(path, inp_filename, out_filename):
    """
    path : str.
        Directory containing the articles in txt format.
    inp_filename : str.
        Input filename of article text file. E.g. '1eeb0768-612d-40c1-8638-91fa6e639696.txt'
    out_filename : str.
        Output filename of uima cas file.
    ----------------------------

    """
    cas = Cas(typesystem=ts)
    with open(f"{path}/{inp_filename}", "r", encoding="utf8") as inp:
        sentences = inp.readlines()
        cas.sofa_string = "".join(sentences)
        Sentence = ts.get_type(SENTENCE_TYPE)
        Token = ts.get_type(TOKEN_TYPE)
        POS = ts.get_type(POS_TYPE)

        offset = 0
        token_offset = 0 
        for sentence in sentences:
            doc = nlp(sentence)
            sent_begin = offset 
            sent_end = offset + len(sentence)
            sent = Sentence(begin=sent_begin, end=sent_end)
            cas.add(sent)
            offset = sent_end 
            for token in doc:
                token_begin = token_offset + token.idx
                token_end = token_begin+ len(token.text)
                cas_token = Token(begin=token_begin, end=token_end)
                cas.add(cas_token)
                if token.tag_ in relevant_pos_tags:
                    cas_pos = POS(begin=token_begin, end=token_end, pos_tag=str(token.tag_))
                    cas.add(cas_pos)
            token_offset = token_end 
        if "humans" in path:
            cas.to_xmi(f"{UIMA_CAS_DIR}/uima_cas_humans/" + out_filename)
        elif "gpt-3.5" in path:
            cas.to_xmi(f"{UIMA_CAS_DIR}/uima_cas_gpt-3.5/" + out_filename)
        elif "gpt-4o" in path:
            cas.to_xmi(f"{UIMA_CAS_DIR}/uima_cas_gpt-4o/" + out_filename)
        else:
            print("ERROR. Please rename your path.")


def write_uima_cas_files(path):
    """
    path : str.
        Path to the directory containing the selected articles in txt format. E.g. 'data/article_texts/'
    ---------------------
    Iterates through files and writes an UIMA CAS file containing
    POS-Tags.
    """
    files = os.listdir(path)
    for file in files:
        out_filename = file.replace(".txt", ".xmi")
        write_uima_cas_file(path, file, out_filename)


if __name__ == '__main__':
    #### Set paths
    REPO_DIR = str(Path().resolve().parents[0])
    DATA_DIR = os.path.join(REPO_DIR, "data")
    UIMA_CAS_DIR = os.path.join(DATA_DIR, "uima_cas")

    #### Create final dataset for annotation
    write_uima_cas_files(f"{DATA_DIR}/article_humans")
    write_uima_cas_files(f"{DATA_DIR}/article_llms/article_gpt-3.5-turbo-0125")
    write_uima_cas_files(f"{DATA_DIR}/article_llms/article_gpt-4o-2024-05-13")

