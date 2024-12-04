import pandas as pd
from datasets import load_dataset
import spacy
from tqdm import tqdm
import random
import os
from pathlib import Path
import uuid
from nltk import sent_tokenize

"""
This script is used to evaluate the data prompted using six different prompts
to choose the 'best' prompt for further work. 
"""

def create_all_articles_json():
    """
    Creates a JSON file containing all German articles from the
    test split of MLSUM.
    """
    dataset = load_dataset("reciTAL/mlsum", "de", split="test")
    data = {}
    for title, text, sum, date, topic, url in tqdm(zip(dataset["title"], dataset["text"], dataset["summary"], dataset["date"], dataset["topic"], dataset["url"])):
        doc = nlp(text)
        tokens = [token for token in doc]
        data[url] = {"text" : text, "title": title, "summary": sum, "date" : date, "topic" : topic, "word_count": len(tokens)}
    df = pd.DataFrame(data)
    df.to_json(f"{JSON_DATA_DIR}/all_articles.json", force_ascii=False)

def choose_sample_randomly(no_sample=100, write_to_file=False):
    """
    no_sample: int.
        Number of texts that you want to choose randomly.
    write_to_file: Boolean.
        Determines whether the selected sample should be written to
        a JSON file. Default is false.
    --------------------------
    This selects a random sample of texts of the MLSUM dataset.
    """
    df = pd.read_json(f"{JSON_DATA_DIR}/all_articles.json", encoding="utf8")
    sample_data = {}
    random.seed(99)
    list_of_samples = list(df.keys())
    random.shuffle(list_of_samples)
    if write_to_file:
        for i, key in enumerate(list_of_samples):
            if i >= no_sample:
                break
            else:
                text, title, summary, date, topic = df[key]["text"], df[key]["title"], df[key]["summary"], df[key]["date"], df[key]["topic"]
                doc = nlp(text)
                tokens = [token for token in doc]
                if len(tokens) > 350 and len(tokens) < 450:
                    if topic == "politik":
                        if summary in text:
                            continue
                        else:
                            sample_data[key] = {"text" : text, "title": title, "summary": summary, "date" : date, "topic" : topic, "word_count" : len(tokens)}
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_json(f"{JSON_DATA_DIR}/human_{str(no_sample)}n.json", force_ascii=False)          
    return list_of_samples

def join_similarities(gpt_sample, similarity_file):
    """
    gpt_sample: str.
        Filtered GPT dataframe containing the prompted data.
    similarity_file: str.
        File containing the similarity scores for each document ID.
    --------------------------
    This function adds the similarity scores to the existing GPT df.
    """
    df_gpt = pd.read_csv(f"{PROMPT_DIR}/{gpt_sample}")
    similarities = pd.read_json(f"{JSON_DATA_DIR}/{similarity_file}")
    similarities_column = []
    for article_url, p_id in zip(df_gpt["article_url"], df_gpt["p_id"]):
        similarities_column.append(similarities[f"{article_url}_{p_id}"]["sim_score"])
    similarities_df = pd.DataFrame({"sim_scores": similarities_column})
    df_gpt = df_gpt.join(similarities_df)
    df_gpt = df_gpt[df_gpt['p_id'] == "1a"]
    sorted_df = df_gpt.sort_values("sim_scores", ascending=False)
    return sorted_df


def get_best_article_df(gpt3_df, gpt4_df):
    gpt3_final = gpt3_df[gpt3_df['article_url'].isin(gpt4_df['article_url'])]
    gpt4_final = gpt4_df[gpt4_df['article_url'].isin(gpt3_df['article_url'])]
    gpt3_final = gpt3_final.reset_index(drop=True)
    gpt4_final = gpt4_final.reset_index(drop=True)
    urls, scores = [], []
    for article_url, sim_score in zip(gpt3_final["article_url"].values,gpt3_final["sim_scores"].values):
        urls.append(article_url)
        other_score = gpt4_final.loc[gpt4_final['article_url'] == article_url]
        total_score = other_score["sim_scores"].iloc[0] + sim_score
        scores.append(total_score)
    combined_df = pd.DataFrame({"article_url": urls, "sim_scores": scores})
    combined_df = combined_df.reset_index(drop=True)
    sorted_df = combined_df.sort_values("sim_scores", ascending=False)
    return sorted_df


def create_gpt_text_files(sorted_df, gpt_df, chatgpt_version):
    i = 1
    mapping_dict = {}
    for article_url in sorted_df["article_url"]:
        if i > 10:
            break
        gpt_row = gpt_df.loc[gpt_df['article_url'] == article_url]
        gpt_text = gpt_row["response"].iloc[0]
        unique_id = uuid.uuid4()
        mapping_dict[f"{article_url}_1a"] = str(unique_id)
        with open(f"{DATA_DIR}/article_llms/article_{chatgpt_version}/{unique_id}.txt", "w", encoding="utf8") as out:
            sentences = sent_tokenize(gpt_text, "german")
            for sent in sentences:
                out.write(sent + "\n")
        i += 1
    df_mapping = pd.DataFrame({"urls": list(mapping_dict.keys()), "ids" : list(mapping_dict.values())})
    df_mapping.to_csv(f"{DATA_DIR}/mapping/mapping_{chatgpt_version}.csv", encoding="utf8", index=False)
    
def create_human_text_files(mapping_file, human_json):
    mapping = pd.read_csv(f"{MAPPING_DIR}/{mapping_file}", encoding="utf8")
    human = pd.read_json(f"{JSON_DATA_DIR}/{human_json}")
    mapping_dict = {}
    for url in mapping["urls"]:
        idx = url.replace("_1a", "")
        text = human[idx]["text"]
        unique_id = uuid.uuid4()
        mapping_dict[idx] = str(unique_id)
        with open(f"{DATA_DIR}/article_humans/{unique_id}.txt", "w", encoding="utf8") as out:
            sentences = sent_tokenize(text, "german")
            for sent in sentences:
                out.write(sent + "\n")
    df_mapping = pd.DataFrame({"urls": list(mapping_dict.keys()), "ids" : list(mapping_dict.values())})
    df_mapping.to_csv(f"{MAPPING_DIR}/mapping_human.csv", encoding="utf8", index=False)

if __name__ == '__main__':
    #### Set paths
    REPO_DIR = str(Path().resolve().parents[0])
    DATA_DIR = os.path.join(REPO_DIR, "data")
    JSON_DATA_DIR = os.path.join(DATA_DIR, "json_files")
    PROMPT_DIR = os.path.join(REPO_DIR, "prompt_data")
    MAPPING_DIR = os.path.join(DATA_DIR, "mapping")

    #### Load spacy
    nlp = spacy.load("de_core_news_sm")

    #### Create a JSON file containing 100 random articles with topic 'politik' from reciTAL/mlsum
    create_all_articles_json()
    choose_sample_randomly("all_articles.json", no_sample=100, write_to_file=True)
    
    #### Create dfs containing similarity scores
    gpt3_df = join_similarities("filtered_gpt-3.5-turbo-0125_100n_170+words.csv", "gpt-3.5-turbo-0125_100n_170+words_similarities.json")
    gpt4_df = join_similarities("filtered_gpt-4o-2024-05-13_100n.csv", "gpt-4o-2024-05-13_100n_similarities.json")
    sorted_df = get_best_article_df(gpt3_df, gpt4_df)
    
    ### Choose 10 best articles for each generation type and create a unique ID and mapping file
    create_gpt_text_files(sorted_df, gpt3_df, "gpt-3.5-turbo-0125")
    create_gpt_text_files(sorted_df, gpt4_df, "gpt-4o-2024-05-13")
    create_human_text_files("mapping_gpt-3.5-turbo-0125.csv", "human_100n.json")