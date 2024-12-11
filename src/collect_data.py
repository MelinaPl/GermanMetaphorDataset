"""
@author: Melina Plakidis
"""
import pandas as pd
from datasets import load_dataset
import spacy
from tqdm import tqdm
import random
import os
from pathlib import Path
import uuid
from nltk import sent_tokenize
from openai import OpenAI
import datetime

"""
This script is used to evaluate the data prompted using six different prompts
to choose the 'best' prompt for further work. 
"""

#### Set paths
REPO_DIR = str(Path().resolve().parents[0])
DATA_DIR = os.path.join(REPO_DIR, "data")
JSON_DATA_DIR = os.path.join(DATA_DIR, "json_files")
PROMPT_DATA_DIR = os.path.join(REPO_DIR, "prompt_data")
MAPPING_DIR = os.path.join(DATA_DIR, "mapping")

#### Prompts to be tested

prompts = {}
prompts["1"] = "Schreibe einen Online Artikel im Ressort '#ressort#' für die Süddeutsche Zeitung mit dem Titel '#title#' aus dem Jahr #year# im Umfang von genau #numberofwords# Wörtern. Schreibe den Artikel so, dass eine kurze Zusammenfassung des Inhalts wie folgt geschrieben werden könnte: '#summary#'"
prompts["2"] = "Es ist #year#. Du bist ein Redakteur bei der Süddeutschen Zeitung und sollst einen Online Artikel im Ressort '#ressort#' mit dem Titel '#title#' im Umfang von genau #numberofwords# Wörtern schreiben. Schreibe den Artikel so, dass eine kurze Zusammenfassung des Inhalts wie folgt geschrieben werden könnte: '#summary#'"
prompts["3"] = "Es ist #year#. Du bist ein Redakteur bei der Süddeutschen Zeitung und sollst einen Online Artikel im Ressort '#ressort#' mit dem Titel '#title#' im Umfang von genau #numberofwords# Wörtern schreiben. Verwende Metaphern so, wie es ein Redakteur bei der Süddeutschen Zeitung tun würde. Schreibe den Artikel so, dass eine kurze Zusammenfassung des Inhalts wie folgt geschrieben werden könnte: '#summary#'"
prompts["1a"] = "Write a German online article in section '#ressort#' for the Süddeutsche Zeitung with the title '#title#' from the year #year# of exactly #numberofwords# words. Write the article in such a way that a short summary of the content could be written as follows: '#summary#'"
prompts["2a"] = "It is #year#. You are an editor at the Süddeutsche Zeitung and have to write a German online article in section '#ressort#' with the title '#title#' of exactly #numberofwords# words. Write the article in such a way that a short summary of the content could be written as follows: '#summary#'"
prompts["3a"]= "It is #year#. You are an editor at the Süddeutsche Zeitung and have to write a German online article in section '#ressort#' with the title '#title#' of exactly #numberofwords# words. Use metaphors in the way that an editor at the Süddeutsche Zeitung would use them. Write the article in such a way that a short summary of the content could be written as follows: '#summary#'"

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
    df.to_json(f"{JSON_DATA_DIR}/all_articles.json", force_ascii=False, indent=4)

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
        i = 0
        for key in list_of_samples:
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
                            i += 1
                            print(i)
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_json(f"{JSON_DATA_DIR}/human_{str(no_sample)}n.json", force_ascii=False, indent=4)          
    return list_of_samples


def get_response(model, prompt):
    """
    model : str.
        Either 'gpt-3.5-turbo-012' or 'gpt-4o-2024-05-13' 
    prompt : str.
        The prompt used to query ChatGPT.
    ----------------------------
    Returns : str.
    """
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=1
    )
    return completion.choices[0].message.content

def iterate_prompts(prompts, model, df, breaking_point=1):
    """
    prompts : dict.
        A dictionary containing all prompts.
    model : str.
        Either 'gpt-3.5-turbo-012' or 'gpt-4o-2024-05-13' 
    df : pandas Dataframe.
        Dataframe containing 100 human written articles.
    breaking_point (optional): int.
        Default is 1. Determines how many original articles are chosen.
        For six different prompts, value 1 means 1 original article
        and 6 generated articles.
    ----------------------------
    Iterates through all prompts for a given model and then writes the results
    to a CSV file.
    """    
    output_data = []
    for i, key in tqdm(enumerate(df.keys())):
        if i >= breaking_point:
            break
        title, summary, word_count, date, topic = df[key]["title"], df[key]["summary"], df[key]["word_count"], df[key]["date"], df[key]["topic"]
        if model == "gpt-3.5-turbo-012":
            word_count = int(word_count) + 170
        for p_key in prompts.keys():
            final_prompt = prompts[p_key].replace("#title#", title)
            final_prompt = final_prompt.replace("#summary#", summary)
            final_prompt = final_prompt.replace("#numberofwords#", str(word_count))
            final_prompt = final_prompt.replace("#year#", date[-4:])
            final_prompt = final_prompt.replace("#ressort#", topic.capitalize())
            gpt_response = get_response(model, final_prompt)
            gpt_response = gpt_response.replace(f"{title}", "").strip()
            gpt_response = gpt_response.replace('\r', '').replace('\n', '')
            doc = nlp(gpt_response)
            tokens = [token for token in doc]
            today = datetime.date.today()
            output_data.append({"p_id" : p_key, "prompt": final_prompt, "original": df[key]["text"], "word_count_original" : word_count, "word_count_chatgpt": len(tokens), "response": gpt_response, "article_url": key, "model" : model, "experiment_date":today})
            output_df = pd.DataFrame(output_data)
            if model == "gpt-3.5-turbo-012":
                output_df.to_csv(f"{PROMPT_DATA_DIR}/{model}_{str(breaking_point)}n_170+words.csv", encoding="utf8", index=False)
            else:
                output_df.to_csv(f"{PROMPT_DATA_DIR}/{model}_{str(breaking_point)}n.csv", encoding="utf8", index=False)

def join_similarities(gpt_sample, similarity_file):
    """
    gpt_sample: str.
        Filtered GPT dataframe containing the prompted data.
    similarity_file: str.
        File containing the similarity scores for each document ID.
    --------------------------
    This function adds the similarity scores to the existing GPT df and
    sorts the df by similarity scores in descending order.
    """
    df_gpt = pd.read_csv(f"{PROMPT_DATA_DIR}/{gpt_sample}")
    similarities = pd.read_json(f"{JSON_DATA_DIR}/{similarity_file}")
    similarities_column = []
    for article_url, p_id in zip(df_gpt["article_url"], df_gpt["p_id"]):
        similarities_column.append(similarities[f"{article_url}_{p_id}"]["sim_score"])
    similarities_df = pd.DataFrame({"sim_scores": similarities_column})
    df_gpt = df_gpt.join(similarities_df)
    df_gpt = df_gpt[df_gpt['p_id'] == "1a"] # Prompt that was chosen: 1a!
    sorted_df = df_gpt.sort_values("sim_scores", ascending=False)
    return sorted_df


def get_best_article_df(gpt3_df, gpt4_df):
    """
    gpt3_df: pandas Dataframe.
        Filtered GPT3-5 dataframe containing the prompted data and similarity scores.
    gpt4_df: pandas Dataframe.
        Filtered GPT4o dataframe containing the prompted data and similarity scores.
    --------------------------
    This function creates a dataframe containing article urls and their combined similarity
    score and then sorts it by similarity scores in descending order.
    """
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
    """
    sorted_df : pandas Dataframe.
        Dataframe containg articles urls and their combined similarity score, 
        sorted by similarity score in descending order.
    gpt_df: pandas Dataframe.
        Filtered GPT dataframe containing the prompted data and similarity scores.
    chatgpt_version : str.
        Version of ChatGPT which should be used. Either 'gpt-3.5-turbo-0125' or
        'gpt-4o-2024-05-13'
    --------------------------
    This function chooses the final GPT texts, writes them as text files and creates
    a mapping file (containing a unique ID mapped to the url)
    """
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
    df_mapping.to_csv(f"{MAPPING_DIR}/mapping_{chatgpt_version}.csv", encoding="utf8", index=False)
    
def create_human_text_files(mapping_file, human_json):
    """
    mapping_file : str.
        One mapping file (either of ChatGPT-3.5 or ChatGPT-4o) containing the final
        chosen 10 articles.
    human_json : str.
        File 'human_100n.json' containing 100 of the original human written articles.
    --------------------------
    This function chooses the final human written texts, writes them as text files
    and creates a mapping file (containing a unique ID mapped to the url)
    """
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

    #### Load spacy
    nlp = spacy.load("de_core_news_sm")

    #### Create a JSON file containing 100 random articles with topic 'politik' from reciTAL/mlsum
    create_all_articles_json()
    choose_sample_randomly(no_sample=100, write_to_file=True)
    
    #### Set client
    client = OpenAI()

    #### Generate prompt data: Attention - not possible without a fee!
    df = pd.read_json(f"{JSON_DATA_DIR}/human_100n.json")
    iterate_prompts(prompts, "gpt-3.5-turbo-0125", df, breaking_point=100)
    iterate_prompts(prompts, "gpt-4o-2024-05-13", df, breaking_point=100)