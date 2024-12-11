"""
@author: Melina Plakidis
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
from nltk import sent_tokenize
import numpy as np
from collections import Counter
from langdetect import detect
from tqdm import tqdm
from collect_data import join_similarities, get_best_article_df, create_gpt_text_files, create_human_text_files

"""
This script processes the prompted data and filters GPT-generated articles 
based on the following criteria:

- similarity score
- whether the human summary is in the GPT-generated articles (--> exclude)
- text length 
- only German language

It additionally provides some descriptive statistics to evaluate which prompt is 
the most suitable.
"""

def detect_summaries_human(human_sample, write_to_file=False):
    """
    human_sample : str.
        File containing human written articles ('human_100n.json')
    write_to_file (optional) : Boolean.
        Whether results should be written to JSON. Default is False.
    ----------------------------
    Searches for the human written summary in the article text.
    Only returns true for exact string matches.
    """
    data = pd.read_json(f"{JSON_DATA_DIR}/{human_sample}")
    contains_summary = []
    for key in data:
        summary = data[key]["summary"]
        text = data[key]["text"]
        sum_sentences = sent_tokenize(summary, "german")
        if summary in text:
            contains_summary.append(key)
        else:
            for sentence in sum_sentences:
                if sentence in text:
                    contains_summary.append(key)
                    break
    summary_dict = {}
    for key in data:
        if key in contains_summary:
            summary_dict[key] = {"contains_summary": True}
        else: 
            summary_dict[key] = {"contains_summary": False}
    summary_df = pd.DataFrame(summary_dict)
    if write_to_file:
        summary_df.to_json(f"{JSON_DATA_DIR}/human_100n_summaries.json", force_ascii=False)
    else:
        return summary_df


def detect_summaries_gpt(human_sample, gpt_sample, write_to_file=False):
    """
    human_sample : str.
        File containing human written articles ('human_100n.json')
    gpt_sample : str.
        File containing prompt data of GPT (either 'gpt-3.5-turbo-0125_100n_170+words.csv'
        or 'gpt-4o-2024-05-13_100n.csv').
    write_to_file (optional) : Boolean.
        Whether results should be written to JSON. Default is False.
    ----------------------------
    Searches for the human written summary in the article text.
    Only returns true for exact string matches.
    """
    data = pd.read_json(f"{JSON_DATA_DIR}/{human_sample}")
    gpt_data = pd.read_csv(f"{PROMPT_DATA_DIR}/{gpt_sample}")
    contains_summary = []
    for p_id, response, article_url in zip(gpt_data["p_id"], gpt_data["response"], gpt_data["article_url"]):
        summary = data[article_url]["summary"]
        sum_sentences = sent_tokenize(summary, "german")
        if summary in response:
            contains_summary.append(f"{article_url}_{p_id}")
        else:
            for sentence in sum_sentences:
                if sentence in response:
                    contains_summary.append(f"{article_url}_{p_id}")
                    break
    summary_dict = {}
    for p_id, response, article_url in zip(gpt_data["p_id"], gpt_data["response"], gpt_data["article_url"]):
        key = f"{article_url}_{p_id}"
        if key in contains_summary:
            summary_dict[key] = {"contains_summary": True, "p_id": p_id, "article_url": article_url}
        else: 
            summary_dict[key] = {"contains_summary": False, "p_id": p_id, "article_url": article_url}
    summary_df = pd.DataFrame(summary_dict)
    if write_to_file:
        summary_df.to_json(f"{JSON_DATA_DIR}/{gpt_sample.replace('.csv', '')}_summaries.json", force_ascii=False)
    else:
        return summary_df
    

def get_average_length_across_prompts(human_sample, version):
    """
    human_sample : str.
        File containing human written articles ('human_100n.json')
    version : str.
        Whether 'human', 'gpt-3.5-turbo-0125_100n_170+words.csv' or 
        'gpt-4o-2024-05-13_100n.csv'
    ----------------------------
    Prints descriptive statistics for text length. 
    """
    if version == "human":
        data = pd.read_json(f"{JSON_DATA_DIR}/{human_sample}")
        word_count_column = [data[key]["word_count"] for key in data]
        df = pd.DataFrame({"word_count": word_count_column})
        print(df.describe().to_latex())
    else:
        gpt_data = pd.read_csv(f"{PROMPT_DATA_DIR}/{version}")
        print(gpt_data["word_count_chatgpt"].describe().T.to_latex())

    
def filter_gpt_articles(gpt_sample, additional_words, write_to_file=True):
    """
    gpt_sample : str.
        File containing prompt data of GPT (either 'gpt-3.5-turbo-0125_100n_170+words.csv'
        or 'gpt-4o-2024-05-13_100n.csv').
    additional_words : Boolean.
        Whether 170 words were added to the word count (concerns GPT-3.5)
    write_to_file (optional) : Boolean.
        Whether results should be written to JSON. Default is True.
    ----------------------------
    Filters GPT articles based on text length, whether the summary is contained, and language.   
    """
    different_length, different_language, contains_summary = [], [], []
    gpt_data = pd.read_csv(f"{PROMPT_DATA_DIR}/{gpt_sample}")
    summary_data = pd.read_json(f"{JSON_DATA_DIR}/{gpt_sample.replace('.csv', '')}_summaries.json")
    output_data = []
    for p_id, prompt, original, word_count_original, word_count_chatgpt, response, article_url, model, experiment_date in zip(gpt_data["p_id"], gpt_data["prompt"], gpt_data["original"], gpt_data["word_count_original"], gpt_data["word_count_chatgpt"], gpt_data["response"], gpt_data["article_url"], gpt_data["model"], gpt_data["experiment_date"]):
        if additional_words:
            word_count_original = word_count_original -170
            diff = word_count_original - word_count_chatgpt
        else:
            diff = word_count_original - word_count_chatgpt
        diff = abs(diff)
        key = f"{article_url}_{p_id}"
        remove = False
        if diff > 50: # control for word length
            different_length.append({"p_id": p_id, "prompt": prompt,"original": original, "word_count_original" : word_count_original, "word_count_chatgpt": word_count_chatgpt, "response": response, "article_url": article_url, "model": model, "experiment_date": experiment_date})
            remove = True
        if summary_data[key]["contains_summary"]: # control for human-written summary
            contains_summary.append({"p_id": p_id, "prompt": prompt,"original": original, "word_count_original" : word_count_original, "word_count_chatgpt": word_count_chatgpt, "response": response, "article_url": article_url, "model": model, "experiment_date": experiment_date})
            remove = True
        if detect(response) != "de": # control for language
            different_language.append({"p_id": p_id, "prompt": prompt,"original": original, "word_count_original" : word_count_original, "word_count_chatgpt": word_count_chatgpt, "response": response, "article_url": article_url, "model": model, "experiment_date": experiment_date})
            remove = True
        if not remove:     
            output_data.append({"p_id": p_id, "prompt": prompt,"original": original, "word_count_original" : word_count_original, "word_count_chatgpt": word_count_chatgpt, "response": response, "article_url": article_url, "model": model, "experiment_date": experiment_date})
    output_df = pd.DataFrame(output_data)
    if write_to_file:
        output_df.to_csv(f"{PROMPT_DATA_DIR}/filtered_{gpt_sample}", index=False)
    print(f"Removed due to word count: {len(different_length)}")
    print(f"Removed due to different language: {len(different_language)}")
    print(f"Removed due to summary: {len(contains_summary)}")

def find_duplicates(human_sample, gpt_sample):
    """
    human_sample : str.
        File containing human written articles ('human_100n.json')
    gpt_sample : str.
        File containing prompt data of GPT (either 'gpt-3.5-turbo-0125_100n_170+words.csv'
        or 'gpt-4o-2024-05-13_100n.csv').
    ----------------------------
    Checks for duplicate sentences between human original and GPT written article.   
    """
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    gpt_data = pd.read_csv(f"{PROMPT_DATA_DIR}/{gpt_sample}")
    human_data = pd.read_json(f"{JSON_DATA_DIR}/{human_sample}")

    all_duplicates = []
    for text_gpt, article_url, p_id in tqdm(zip(gpt_data["response"], gpt_data["article_url"], gpt_data["p_id"])):
        text = human_data[article_url]["text"]
        sents_text = sent_tokenize(text, "german")
        gpt_sents_text = sent_tokenize(text_gpt, "german")
        original_embeddings = model.encode(sents_text)
        gpt_embeddings = model.encode(gpt_sents_text)
        similarities = model.similarity(original_embeddings, gpt_embeddings)
        possible_duplicates = []
        for idx_i, original in enumerate(sents_text):
            for idx_j, gpt in enumerate(gpt_sents_text):
                if similarities[idx_i][idx_j] >= 0.90:
                    possible_duplicates.append({"original" : original, "gpt_duplicate": gpt, "score": similarities[idx_i][idx_j], "article_url": article_url, "p_id": p_id})
                    all_duplicates.append({"original" : original, "gpt_duplicate": gpt, "score": similarities[idx_i][idx_j], "article_url": article_url, "p_id": p_id})
        if not len(possible_duplicates) == 0:
            print("Contains duplicate!")
    print(all_duplicates)
    return all_duplicates


def get_similarity_scores(human_sample, gpt_sample, write_to_file=True): # values for version: gpt-3.5-turbo, gpt-4o
    """
    human_sample : str.
        File containing human written articles ('human_100n.json')
    gpt_sample : str.
        File containing prompt data of GPT (either 'gpt-3.5-turbo-0125_100n_170+words.csv'
        or 'gpt-4o-2024-05-13_100n.csv').
    write_to_file (optional) : Boolean.
        Whether results should be written to JSON. Default is True.
    ----------------------------
    Calculates the similarity scores between human original and GPT written
    equivalent and returns a dataframe containing the article urls, prompt ids
    and similarity scores.
    """
    df_human = pd.read_json(f"{JSON_DATA_DIR}/{human_sample}")
    df_gpt = pd.read_csv(f"{PROMPT_DATA_DIR}/{gpt_sample}")
    model = SentenceTransformer('aari1995/German_Semantic_V3b')
    sim_scores = {}
    for gpt_text, article_url, p_id in tqdm(zip(df_gpt["response"], df_gpt["article_url"], df_gpt["p_id"])):
        embeddings = model.encode([df_human[article_url]["text"], gpt_text])
        sim_score = float(model.similarity(embeddings[0], embeddings[1])[0][0])
        sim_scores[f"{article_url}_{p_id}"] = {"sim_score" : sim_score, "article_url": article_url, "p_id" : p_id}
    sim_scores_df = pd.DataFrame(sim_scores)
    if write_to_file:
        sim_scores_df.to_json(f"{JSON_DATA_DIR}/{gpt_sample.replace('.csv', '')}_similarities.json", force_ascii=False)
    return sim_scores_df


def get_similarity_statistics(gpt_sample, similarity_file):
    """
    gpt_sample : str.
        File containing prompt data of GPT (either 'gpt-3.5-turbo-0125_100n_170+words.csv'
        or 'gpt-4o-2024-05-13_100n.csv').
    similarity_file : str.
        File containing the corresponding similarity scores (either 
        'gpt-3.5-turbo-0125_100n_170+words_similarities.json' or 
        'gpt-4o-2024-05-13_100n_similarities.json')
    ----------------------------
    Merges GPT generated article data with similarity scores and prints
    descriptive statistics.
    """
    df_gpt = pd.read_csv(f"{PROMPT_DATA_DIR}/{gpt_sample}")
    similarities = pd.read_json(f"{JSON_DATA_DIR}/{similarity_file}")
    similarities_column = []
    for article_url, p_id in zip(df_gpt["article_url"], df_gpt["p_id"]):
        similarities_column.append(similarities[f"{article_url}_{p_id}"]["sim_score"])
    similarities_df = pd.DataFrame({"sim_scores": similarities_column})
    df_gpt = df_gpt.join(similarities_df)
    print(df_gpt[["p_id", "sim_scores" ]].groupby("p_id").describe().T.to_latex())
    return df_gpt

def get_overall_similarity_statistics(gpt_sample1, gpt_sample2):
    """
    gpt_sample1 : str.
        First file containing prompt data of GPT (either 'gpt-3.5-turbo-0125_100n_170+words.csv'
        or 'gpt-4o-2024-05-13_100n.csv').
    gpt_sample2 : str.
        Second file containing prompt data of GPT (either 'gpt-3.5-turbo-0125_100n_170+words.csv'
        or 'gpt-4o-2024-05-13_100n.csv').
    ----------------------------
    Prints descriptive statistis for merged (overall) GPT data.
    """
    df_gpt1 = get_similarity_statistics(f"{gpt_sample1}", f"{gpt_sample1.replace('.csv','')}_similarities.json")
    df_gpt2 = get_similarity_statistics(f"{gpt_sample2}", f"{gpt_sample2.replace('.csv','')}_similarities.json")
    merged_df = df_gpt1._append(df_gpt2, ignore_index=True)
    print(merged_df[["p_id", "sim_scores" ]].groupby("p_id").describe().T.to_latex())   

if __name__ == '__main__':
    #### Set paths
    REPO_DIR = str(Path().resolve().parents[0])
    PROMPT_DATA_DIR = os.path.join(REPO_DIR, "prompt_data")
    DATA_DIR = os.path.join(REPO_DIR, "data")
    JSON_DATA_DIR = os.path.join(DATA_DIR, "json_files")    

    #### Write summary files (human, ChatGPT-3.5, ChatGPT-4o)
    detect_summaries_human("human_100n.json", write_to_file=True) # Human
    detect_summaries_gpt("human_100n.json", "gpt-3.5-turbo-0125_100n_170+words.csv",  write_to_file=True) # ChatGPT-3.5
    detect_summaries_gpt("human_100n.json", "gpt-4o-2024-05-13_100n.csv",  write_to_file=True) # ChatGPT-4o

    #### Get word counts (human, ChatGPT-3.5, ChatGPT-4o)
    get_average_length_across_prompts("human_100n.json", version="human") # Human
    get_average_length_across_prompts("human_100n.json", version="gpt-3.5-turbo-0125_100n_170+words.csv") # ChatGPT-3.5
    get_average_length_across_prompts("human_100n.json", version="gpt-4o-2024-05-13_100n.csv") # ChatGPT-4o

    #### Filter ChatGPT articles
    filter_gpt_articles("gpt-3.5-turbo-0125_100n_170+words.csv", additional_words=True) # ChatGPT-3.5
    filter_gpt_articles("gpt-4o-2024-05-13_100n.csv", additional_words=False) # ChatGPT-4o
    
    #### Check for duplicates
    find_duplicates("human_100n.json", "gpt-3.5-turbo-0125_100n_170+words.csv") # ChatGPT-3.5
    find_duplicates("human_100n.json", "gpt-4o-2024-05-13_100n.csv") # ChatGPT-4o
    
    #### Calculate similarity scores between ChatGPT-generated and corresponding human-written article   
    get_similarity_scores("human_100n.json","gpt-3.5-turbo-0125_100n_170+words.csv", write_to_file=True) # ChatGPT-3.5
    get_similarity_scores("human_100n.json","gpt-4o-2024-05-13_100n.csv", write_to_file=True) # ChatGPT-4o

    #### Print similarity scores for one model across prompts
    get_similarity_statistics("gpt-3.5-turbo-0125_100n_170+words.csv", "gpt-3.5-turbo-0125_100n_170+words_similarities.json") # ChatGPT-3.5
    get_similarity_statistics("gpt-4o-2024-05-13_100n.csv", "gpt-4o-2024-05-13_100n_similarities.json") # ChatGPT-4o

    #### Print similarity scores for both models across prompts
    get_overall_similarity_statistics("gpt-4o-2024-05-13_100n.csv", "gpt-3.5-turbo-0125_100n_170+words.csv")

    #### Create dfs containing similarity scores
    gpt3_df = join_similarities("filtered_gpt-3.5-turbo-0125_100n_170+words.csv", "gpt-3.5-turbo-0125_100n_170+words_similarities.json")
    gpt4_df = join_similarities("filtered_gpt-4o-2024-05-13_100n.csv", "gpt-4o-2024-05-13_100n_similarities.json")
    sorted_df = get_best_article_df(gpt3_df, gpt4_df)
    
    ### Choose 10 best articles for each generation type and create a unique ID and mapping file
    create_gpt_text_files(sorted_df, gpt3_df, "gpt-3.5-turbo-0125")
    create_gpt_text_files(sorted_df, gpt4_df, "gpt-4o-2024-05-13")
    create_human_text_files("mapping_gpt-3.5-turbo-0125.csv", "human_100n.json")