"""
@author: Melina Plakidis
"""
import os
import pandas as pd
import spacy
from nltk import sent_tokenize
from pathlib import Path
from openai import OpenAI
import datetime
from tqdm import tqdm
import uuid

# TODO: MISSING - That ChatGPT3-5 was queried with 170 more words. (didn't I have that somewhere?)

"""
This script is used to query chatgpt using six different prompts.
Please keep in mind that collecting data requires an account at OpenAI 
and is currently not for free.
"""

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
    Returns : str.
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
   

#### Prompts to be tested

prompts = {}
prompts["1"] = "Schreibe einen Online Artikel im Ressort '#ressort#' für die Süddeutsche Zeitung mit dem Titel '#title#' aus dem Jahr #year# im Umfang von genau #numberofwords# Wörtern. Schreibe den Artikel so, dass eine kurze Zusammenfassung des Inhalts wie folgt geschrieben werden könnte: '#summary#'"
prompts["2"] = "Es ist #year#. Du bist ein Redakteur bei der Süddeutschen Zeitung und sollst einen Online Artikel im Ressort '#ressort#' mit dem Titel '#title#' im Umfang von genau #numberofwords# Wörtern schreiben. Schreibe den Artikel so, dass eine kurze Zusammenfassung des Inhalts wie folgt geschrieben werden könnte: '#summary#'"
prompts["3"] = "Es ist #year#. Du bist ein Redakteur bei der Süddeutschen Zeitung und sollst einen Online Artikel im Ressort '#ressort#' mit dem Titel '#title#' im Umfang von genau #numberofwords# Wörtern schreiben. Verwende Metaphern so, wie es ein Redakteur bei der Süddeutschen Zeitung tun würde. Schreibe den Artikel so, dass eine kurze Zusammenfassung des Inhalts wie folgt geschrieben werden könnte: '#summary#'"
prompts["1a"] = "Write a German online article in section '#ressort#' for the Süddeutsche Zeitung with the title '#title#' from the year #year# of exactly #numberofwords# words. Write the article in such a way that a short summary of the content could be written as follows: '#summary#'"
prompts["2a"] = "It is #year#. You are an editor at the Süddeutsche Zeitung and have to write a German online article in section '#ressort#' with the title '#title#' of exactly #numberofwords# words. Write the article in such a way that a short summary of the content could be written as follows: '#summary#'"
prompts["3a"]= "It is #year#. You are an editor at the Süddeutsche Zeitung and have to write a German online article in section '#ressort#' with the title '#title#' of exactly #numberofwords# words. Use metaphors in the way that an editor at the Süddeutsche Zeitung would use them. Write the article in such a way that a short summary of the content could be written as follows: '#summary#'"

if __name__ == '__main__':
    #### Set paths
    REPO_DIR = str(Path().resolve().parents[0])
    PROMPT_DATA_DIR = os.path.join(REPO_DIR, "prompt_data")
    DATA_DIR = os.path.join(REPO_DIR, "data")
    JSON_DATA_DIR = os.path.join(DATA_DIR, "json_files")

    #### Load spacy
    nlp = spacy.load("de_core_news_sm") 

    #### Set chatgpt version and set client
    chatgpt_version = "gpt-4o-2024-05-13" # either gpt-3.5-turbo-0125 or gpt-4o-2024-05-13
    client = OpenAI()

    #### Generate prompt data
    df = pd.read_json(f"{JSON_DATA_DIR}/100n_sample_human.json")
    iterate_prompts(prompts, chatgpt_version, df, breaking_point=100)