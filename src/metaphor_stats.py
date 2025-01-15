"""
@author: Melina Plakidis
"""
from zipfile import ZipFile
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from string import punctuation
import re
import ast
import spacy
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression
import json
from scipy.stats import t, ttest_rel, wilcoxon
pd.options.mode.chained_assignment = None  # default='warn'

"""
This script is used to retrieve quantitative results.
"""

def avg_sentence_length(text):
    """
    text : str.
        String containing the article text.
    ----------------------
    Returns the average sentence length per article.
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    if len(sentences) == 0:
        return 0
    return sum(len(sentence) for sentence in sentences) / len(sentences)

def describe_articles():
    """
    Writes two TEX files containing descriptive statistics
    for number of tokens and sentence length across generation
    types.
    """
    files_humans = os.listdir(f"{DATA_DIR}/article_humans")
    files_gpt35 = os.listdir(f"{GPT35_DIR}")
    files_gpt4o = os.listdir(f"{GPT4o_DIR}")
    texts = []
    for human, gpt35, gpt4o in zip(files_humans, files_gpt35, files_gpt4o):
        with open(f"{HUMAN_DIR}/{human}", "r", encoding="utf8") as h, open(f"{GPT35_DIR}/{gpt35}", "r", encoding="utf8") as g3, open(f"{GPT4o_DIR}/{gpt4o}", "r", encoding="utf8") as g4:
            h_text, g3_text, g4_text = h.read(), g3.read(), g4.read()
            h_id, g3_id, g4_id = human.replace(".txt", ""), gpt35.replace(".txt", ""), gpt4o.replace(".txt", "")
            texts.append({"id": h_id, "gen_type": "human", "text": h_text})
            texts.append({"id": g3_id, "gen_type": "gpt3-5", "text": g3_text})
            texts.append({"id": g4_id, "gen_type": "gpt4o", "text": g4_text})
    df = pd.DataFrame(texts)
    df["tokens"] = df["text"].apply(lambda x: len([token for token in nlp(x)]))
    token_count = df.groupby("gen_type")["tokens"].describe()
    df["sentence_length"] = df["text"].apply(avg_sentence_length)
    sentence_length = df.groupby("gen_type")["sentence_length"].describe()
    with open(f"{STATISTICS_DIR}/tex/token_stats.tex", "w") as file:
        file.write(token_count.to_latex(index=True, float_format="%.2f"))
    with open(f"{STATISTICS_DIR}/tex/sentence_stats.tex", "w") as file:
        file.write(sentence_length.to_latex(index=True, float_format="%.2f"))


def add_relative_freq_genversion(df):
    """
    df : pandas Dataframe.
        Pandas dataframe containing metaphor and 
        conventionality annotations.
    -------------------------
    Adds relative frequencies to dataframe.
    """
    gpt35 = list(df["gpt3-5"].values)
    gpt4o = list(df["gpt4o"].values)
    human = list(df["human"].values)
    rel_35, rel_4o, rel_human = [], [], []
    total_35, total_4o, total_human = sum(gpt35), sum(gpt4o), sum(human)
    for gpt3, gpt4, h in zip(gpt35, gpt4o, human):
        rel_35.append(round(((gpt3/total_35)*100),2))
        rel_4o.append(round(((gpt4/total_4o)*100),2))
        rel_human.append(round(((h/total_human)*100),2))
    df.insert(1, "rel_gpt3-5", rel_35)
    df.insert(3, "rel_gpt4o", rel_4o)
    df.insert(5, "rel_human", rel_human)
    return df

def transform_df_to_rel_freq(df):
    """
    df : pandas Dataframe.
        Pandas dataframe containing metaphor and 
        conventionality annotations.
    -------------------------
    Transforms absolute frequencies to relative frequencies.
    """
    rel_df = df.copy()
    for column_name in rel_df.columns:
        rel_column = []
        value_list = list(rel_df[column_name].values)
        total = sum(value_list)
        for value in value_list:
            rel_column.append(round(((value/total)*100),2))
        rel_df[column_name] = rel_column
    return rel_df.transpose()

def descriptive_statistics_mflag(mflags):
    """
    mflags : pandas Dataframe.
        Pandas dataframe containing annotated metaphor 
        flags.
    -------------------------
    Provides descriptive statistics for metaphor flags and writes them
    to a TEX file.
    """
    mflag_df = mflags.groupby(['gen_version', 'MetaphorFlag']).size().unstack(fill_value=0).transpose()
    mflag_df = add_relative_freq_genversion(mflag_df)
    with open(f"{STATISTICS_DIR}/tex/mflags.tex", "w") as file:
        file.write(mflag_df.to_latex(index=True, float_format="%.2f"))

def descriptive_statistics(annotations, no_dfma=True):
    """
    annotations : pandas Dataframe.
        Pandas dataframe containing annotated metaphors
        and conventionality.
    no_dfma (optional) : Boolean.
        Whether to include or discard the category DFMA.
        Default is True (= DFMA discarded).
    -------------------------
    Provides descriptive statistics for metaphors and conventionality
    and writes them to TEX files.
    """
    generations = ["human", "gpt3-5", "gpt4o"]
    if no_dfma:
        annotations = annotations[annotations["metaphor"]!="DFMA"]
    for gen in generations:
        metaphors = annotations[annotations["gen_version"]==gen]
        coarse_metaphors = metaphors.copy()
        coarse_metaphors['metaphor'] = coarse_metaphors['metaphor'].replace(
            {"KOMPL": "METAPHOR", "MRW\_DIR": "METAPHOR", "MRW": "METAPHOR", "PERS": "METAPHOR", "WIDLII": "METAPHOR"}
            )
        coarse_metaphors = coarse_metaphors.groupby(['doc_id','metaphor']).size().unstack(fill_value=0).transpose()
        metaphors = metaphors.groupby(['doc_id','metaphor']).size().unstack(fill_value=0).transpose()
        metaphors_per_doc = transform_df_to_rel_freq(metaphors)
        coarse_metaphors = transform_df_to_rel_freq(coarse_metaphors)
        if no_dfma:
            with open(f"{STATISTICS_DIR}/tex/metaphors_{gen}_per_doc_nodfma.tex", "w") as file:
                file.write(metaphors_per_doc.to_latex(index=True, float_format="%.2f"))
            with open(f"{STATISTICS_DIR}/tex/metaphors_{gen}_describe_nodfma.tex", "w") as file:
                file.write(metaphors_per_doc.describe().transpose().to_latex(index=True, float_format="%.2f"))
            with open(f"{STATISTICS_DIR}/tex/metaphors_{gen}_describe_coarse_nodfma.tex", "w") as file:
                file.write(coarse_metaphors.describe().transpose().to_latex(index=True, float_format="%.2f"))
        else:
            with open(f"{STATISTICS_DIR}/tex/metaphors_{gen}_per_doc.tex", "w") as file:
                file.write(metaphors_per_doc.to_latex(index=True, float_format="%.2f"))
            with open(f"{STATISTICS_DIR}/tex(metaphors_{gen}_describe.tex", "w") as file:
                file.write(metaphors_per_doc.describe().transpose().to_latex(index=True, float_format="%.2f"))
            with open(f"{STATISTICS_DIR}/tex/metaphors_{gen}_describe_coarse.tex", "w") as file:
                file.write(coarse_metaphors.describe().transpose().to_latex(index=True, float_format="%.2f"))

    ### General x metaphors
    metaphors_genversion_fine = annotations.groupby(['gen_version', 'metaphor']).size().unstack(fill_value=0).transpose()
    metaphors_genversion_fine = add_relative_freq_genversion(metaphors_genversion_fine)
    metaphors_genversion_coarse = annotations.copy()
    metaphors_genversion_coarse['metaphor'] = metaphors_genversion_coarse['metaphor'].replace(
        {"KOMPL": "METAPHOR", "MRW\_DIR": "METAPHOR", "MRW": "METAPHOR", "PERS": "METAPHOR", "WIDLII": "METAPHOR"})
    metaphors_genversion_coarse = (metaphors_genversion_coarse.groupby(['gen_version', 'metaphor']).size().unstack(fill_value=0).transpose())
    metaphors_genversion_coarse = add_relative_freq_genversion(metaphors_genversion_coarse)

    ### General x conventionality
    conventionality_genversion = annotations.groupby(['gen_version', 'conventionality']).size().unstack(fill_value=0).transpose()
    conventionality_genversion = add_relative_freq_genversion(conventionality_genversion)

    ### Write files
    if no_dfma:
        with open(f"{STATISTICS_DIR}/tex/metaphors_genversion_fine_nodfma.tex", "w") as file:
            file.write(metaphors_genversion_fine.to_latex(index=True, float_format="%.2f"))
        with open(f"{STATISTICS_DIR}/tex/metaphors_genversion_coarse_nodfma.tex", "w") as file:
            file.write(metaphors_genversion_coarse.to_latex(index=True, float_format="%.2f"))
        with open(f"{STATISTICS_DIR}/tex/conventionality_genversion_nodfma.tex", "w") as file:
            file.write(conventionality_genversion.to_latex(index=True, float_format="%.2f"))
    else:
        with open(f"{STATISTICS_DIR}/tex/metaphors_genversion_fine.tex", "w") as file:
            file.write(metaphors_genversion_fine.to_latex(index=True, float_format="%.2f"))
        with open(f"{STATISTICS_DIR}/tex/metaphors_genversion_coarse.tex", "w") as file:
            file.write(metaphors_genversion_coarse.to_latex(index=True, float_format="%.2f"))
        with open(f"{STATISTICS_DIR}/tex/conventionality_genversion.tex", "w") as file:
            file.write(conventionality_genversion.to_latex(index=True, float_format="%.2f"))



def prepare_counts_across_docs(annotations, write_file=False, no_dfma=True):
    """
    annotations : pandas Dataframe.
        Pandas dataframe containing annotated metaphors
        and conventionality.
    write_file (optional) : Boolean.
        Whether to write the results to file. Default is False.
    no_dfma (optional) : Boolean.
        Whether to include or discard the category DFMA.
        Default is True (= DFMA discarded).
    -------------------------
    Provides descriptive statistics for metaphors and conventionality
    and writes them to TEX files.
    """
    coarse_result_dict, fine_result_dict = {}, {}
    if no_dfma:
        annotations = annotations[annotations["metaphor"]!="DFMA"]
    for version in ["human", "gpt3-5", "gpt4o"]:
        coarse_docs, fine_docs = {}, {}
        fine_metaphors = annotations[annotations["gen_version"]==version]
        coarse_metaphors = fine_metaphors.copy()
        coarse_metaphors['metaphor'] = coarse_metaphors['metaphor'].replace(
            {"KOMPL": "METAPHOR", "MRW\_DIR": "METAPHOR", "MRW": "METAPHOR", "PERS": "METAPHOR", "WIDLII": "METAPHOR"}
            )
        coarse_metaphors = coarse_metaphors.groupby(['doc_id','metaphor']).size().unstack(fill_value=0).transpose()
        fine_metaphors = fine_metaphors.groupby(['doc_id','metaphor']).size().unstack(fill_value=0).transpose()
        # Coarse
        for doc in coarse_metaphors.columns:
            value_list = list(coarse_metaphors[doc].values)
            total = int(sum(value_list))
            metaphor_count = int(coarse_metaphors[doc].loc["METAPHOR"])
            coarse_docs[doc] = {"metaphor_count": metaphor_count, "total": total}
        # Fine
        for doc in fine_metaphors.columns:
            value_list = list(fine_metaphors[doc].values)
            total = int(sum(value_list))
            kompl_count = int(fine_metaphors[doc].loc["KOMPL"])
            try:
                mrwdir_count = int(fine_metaphors[doc].loc["MRW\_DIR"])
            except:
                mrwdir_count = 0
            mrw_count = int(fine_metaphors[doc].loc["MRW"])
            pers_count = int(fine_metaphors[doc].loc["PERS"])
            widlii_count = int(fine_metaphors[doc].loc["WIDLII"])
            fine_docs[doc] = {"kompl_count": kompl_count, "mrwdir_count": mrwdir_count, "mrw_count": mrw_count, "pers_count": pers_count, "widlii_count": widlii_count,"total": total}
        coarse_result_dict[version] = coarse_docs
        fine_result_dict[version] = fine_docs
    if write_file:
        if no_dfma:
            with open(f"{STATISTICS_DIR}/fine_counts_doc_nodfma.json", "w", encoding="utf8") as out:
                json.dump(fine_result_dict, out, ensure_ascii=False, indent=3)
            with open(f"{STATISTICS_DIR}/coarse_counts_doc_nodfma.json.json", "w", encoding="utf8") as out:
                json.dump(coarse_result_dict, out, ensure_ascii=False, indent=3)
        else:
            with open(f"{STATISTICS_DIR}/fine_counts_doc.json", "w", encoding="utf8") as out:
                json.dump(fine_result_dict, out, ensure_ascii=False, indent=3)
            with open(f"{STATISTICS_DIR}/coarse_counts_doc.json.json", "w", encoding="utf8") as out:
                json.dump(coarse_result_dict, out, ensure_ascii=False, indent=3)           
    return coarse_result_dict, fine_result_dict

def visualize_across_docs(no_dfma=True):
    """
    no_dfma (optional) : Boolean.
        Whether to include the category DFMA or not. Default is True.
    ---------------------------
    Creates scatter plots showing the number of metaphors across
    the number of lexical units.
    """
    if no_dfma:
        with open(f"{STATISTICS_DIR}/fine_counts_doc_nodfma.json", "r", encoding="utf8") as inp:
            fine_data = json.load(inp)
        with open(f"{STATISTICS_DIR}/coarse_counts_doc_nodfma.json", "r", encoding="utf8") as inp:
            coarse_data = json.load(inp)
    else:
        with open(f"{STATISTICS_DIR}/fine_counts_doc.json", "r", encoding="utf8") as inp:
            fine_data = json.load(inp)
        with open(f"{STATISTICS_DIR}/coarse_counts_doc.json", "r", encoding="utf8") as inp:
            coarse_data = json.load(inp)
    # Create general metaphor scatter plot
    plot_data = []
    for version, docs in coarse_data.items():
        for doc_id, values in docs.items():
            plot_data.append({"version": version, "total": values["total"], "metaphor_count": values["metaphor_count"]})
    df = pd.DataFrame(plot_data)

    scatter_handles = []
    colors = {"human": "blue", "gpt3-5": "orange", "gpt4o": "green"}
    for version in df["version"].unique():
        subset = df[df["version"] == version]
        scatter = plt.scatter(subset["total"], subset["metaphor_count"], label=f'{version}', color=colors[version], alpha=0.7)
        scatter_handles.append(scatter)

        X = subset[["total"]]
        y = subset["metaphor_count"]
        model = LinearRegression()
        model.fit(X, y)
        
        x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_vals = model.predict(x_vals)
        plt.plot(x_vals, y_vals, color=colors[version], linestyle="--", label=f'{version} regression')

    plt.xlabel("Lexical Units", fontsize=12)
    plt.ylabel("Number of Metaphors", fontsize=12)
    plt.legend(handles=scatter_handles,title="Generation Type", loc="upper left")
    plt.grid(True)
    if no_dfma:
        plt.savefig(f"{IMAGES_DIR}/metaphor_textlength_scatter_nodfma.png", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(f"{IMAGES_DIR}/metaphor_textlength_scatter.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Create fine-grained metaphor scatter plot
    plot_data = []
    for version, docs in fine_data.items():
        for doc_id, values in docs.items():
            plot_data.append({"version": version, "total": values["total"], "kompl_count": values["kompl_count"],
                             "pers_count": values["pers_count"], "widlii_count": values["widlii_count"],
                              "mrw_count": values["mrw_count"], "mrwdir_count": values["mrwdir_count"]})
    df = pd.DataFrame(plot_data)

    colors = {"human": "blue", "gpt3-5": "orange", "gpt4o": "green"}
    for m_type in ["mrw_count", "pers_count", "widlii_count", "kompl_count", "mrwdir_count"]:
        scatter_handles = []
        for version in df["version"].unique():
            subset = df[df["version"] == version]
            scatter = plt.scatter(subset["total"], subset[m_type], label=f'{version} data', color=colors[version], alpha=0.7)
            scatter_handles.append(scatter)

            X = subset[["total"]]
            y = subset[m_type]
            model = LinearRegression()
            model.fit(X, y)
            
            x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_vals = model.predict(x_vals)
            plt.plot(x_vals, y_vals, color=colors[version], linestyle="--", label=f'{version} regression')

        plt.xlabel("Lexical Units", fontsize=12)
        plt.ylabel(f"Number of {m_type.replace('_count','').upper()}", fontsize=12)
        plt.legend(handles=scatter_handles, title="Generation Type", loc="upper left")
        plt.grid(True)
        if no_dfma:
            plt.savefig(f"{IMAGES_DIR}/metaphor_textlength_scatter_{m_type.replace('_count','').upper()}_nodfma.png", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(f"{IMAGES_DIR}/metaphor_textlength_scatter_{m_type.replace('_count','').upper()}.png", dpi=300, bbox_inches="tight")
        plt.show()


def independent_t_test(annotations, no_dfma=True):
    """
    annotations : pandas Dataframe.
        Pandas dataframe containing annotated metaphors
        and conventionality. 
    no_dfma (optional) : Boolean.
        Whether to include the category DFMA or not. Default is True.
    ---------------------------
    Conducts an independent t test and writes the results to files.
    """
    generations = ["human", "gpt3-5", "gpt4o"]
    fine_data, coarse_data = {}, {}
    if no_dfma:
        annotations = annotations[annotations["metaphor"]!= "DFMA"]
    for gen in generations:
        fine_df = annotations[annotations["gen_version"]==gen]
        coarse_df = fine_df.copy()
        coarse_df["metaphor"] = coarse_df["metaphor"].replace(
            {"KOMPL": "METAPHOR", "MRW\_DIR": "METAPHOR", "MRW": "METAPHOR", "PERS": "METAPHOR", "WIDLII": "METAPHOR"}
            )
        coarse_df = coarse_df.groupby(['doc_id','metaphor']).size().unstack(fill_value=0).transpose()
        fine_df = fine_df.groupby(['doc_id','metaphor']).size().unstack(fill_value=0).transpose()
        fine_df = transform_df_to_rel_freq(fine_df)
        coarse_df = transform_df_to_rel_freq(coarse_df)
        coarse_df = coarse_df.describe()
        fine_df = fine_df.describe()
        if "MRW\_DIR" not in fine_df.columns:
            fine_df["MRW\_DIR"] = 0.0
        metaphor_fine = list(fine_df.columns)
        metaphor_coarse = list(coarse_df.columns)
        fine_data["metaphor"] = metaphor_fine
        fine_data[f"mean_{gen}"] = list(fine_df.loc["mean"].values)  
        fine_data[f"std_{gen}"] = list(fine_df.loc["std"].values)
        coarse_data["metaphor"] = metaphor_coarse
        coarse_data[f"mean_{gen}"] = list(coarse_df.loc["mean"].values)  
        coarse_data[f"std_{gen}"] = list(coarse_df.loc["std"].values)  

    fine_df = pd.DataFrame(fine_data)
    coarse_df = pd.DataFrame(coarse_data)
    n = 10

    fine_results, coarse_results = {}, {} 
    for metaphor in fine_df["metaphor"]:
        mean_diff_3_5 = fine_df.loc[fine_df["metaphor"] == metaphor, "mean_human"].values - fine_df.loc[fine_df["metaphor"] == metaphor, "mean_gpt3-5"].values
        mean_diff_4o = fine_df.loc[fine_df["metaphor"] == metaphor, "mean_human"].values - fine_df.loc[fine_df["metaphor"] == metaphor, "mean_gpt4o"].values

        se_3_5 = ((fine_df.loc[fine_df["metaphor"] == metaphor, "std_human"].values ** 2 / n) + (fine_df.loc[fine_df["metaphor"] == metaphor, "std_gpt3-5"].values ** 2 / n)) ** 0.5
        se_4o = ((fine_df.loc[fine_df["metaphor"] == metaphor, "std_human"].values ** 2 / n) + (fine_df.loc[fine_df["metaphor"] == metaphor, "std_gpt4o"].values ** 2 / n)) ** 0.5
        t_stat_3_5 = mean_diff_3_5 / se_3_5
        t_stat_4o = mean_diff_4o / se_4o
        df_t = n - 1
        p_val_3_5 = (1 - t.cdf(abs(t_stat_3_5), df_t)) * 2  # Two-tailed test
        p_val_4o = (1 - t.cdf(abs(t_stat_4o), df_t)) * 2  # Two-tailed test
        fine_results[f"{metaphor}_human_vs_gpt3_5"] = {"t_stat": t_stat_3_5[0], "p_val": p_val_3_5[0]}
        fine_results[f"{metaphor}_human_vs_gpt4o"] = {"t_stat": t_stat_4o[0], "p_val": p_val_4o[0]}

    for metaphor in coarse_df["metaphor"]:
        mean_diff_3_5 = coarse_df.loc[coarse_df["metaphor"] == metaphor, "mean_human"].values - coarse_df.loc[coarse_df["metaphor"] == metaphor, "mean_gpt3-5"].values
        mean_diff_4o = coarse_df.loc[coarse_df["metaphor"] == metaphor, "mean_human"].values - coarse_df.loc[coarse_df["metaphor"] == metaphor, "mean_gpt4o"].values

        se_3_5 = ((coarse_df.loc[coarse_df["metaphor"] == metaphor, "std_human"].values ** 2 / n) + (coarse_df.loc[coarse_df["metaphor"] == metaphor, "std_gpt3-5"].values ** 2 / n)) ** 0.5
        se_4o = ((coarse_df.loc[coarse_df["metaphor"] == metaphor, "std_human"].values ** 2 / n) + (coarse_df.loc[coarse_df["metaphor"] == metaphor, "std_gpt4o"].values ** 2 / n)) ** 0.5
        t_stat_3_5 = mean_diff_3_5 / se_3_5
        t_stat_4o = mean_diff_4o / se_4o
        df_t = n - 1
        p_val_3_5 = (1 - t.cdf(abs(t_stat_3_5), df_t)) * 2  # Two-tailed test
        p_val_4o = (1 - t.cdf(abs(t_stat_4o), df_t)) * 2  # Two-tailed test
        coarse_results[f"{metaphor}_human_vs_gpt3_5"] = {"t_stat": t_stat_3_5[0], "p_val": p_val_3_5[0]}
        coarse_results[f"{metaphor}_human_vs_gpt4o"] = {"t_stat": t_stat_4o[0], "p_val": p_val_4o[0]}

    if no_dfma:
        with open(f"{STATISTICS_DIR}/inferential/independent_t-test_p_values_coarse_nodfma.json", "w", encoding="utf8") as out:
            json.dump(coarse_results, out, ensure_ascii=False, indent=3)
        with open(f"{STATISTICS_DIR}/inferential/independent_t-test_p_values_fine_nodfma.json", "w", encoding="utf8") as out:
            json.dump(fine_results, out, ensure_ascii=False, indent=3)
    else:
        with open(f"{STATISTICS_DIR}/inferential/independent_t-test_p_values_coarse.json", "w", encoding="utf8") as out:
            json.dump(coarse_results, out, ensure_ascii=False, indent=3)
        with open(f"{STATISTICS_DIR}/inferential/independent_t-test_p_values_fine.json", "w", encoding="utf8") as out:
            json.dump(fine_results, out, ensure_ascii=False, indent=3)

def create_paired_data(annotations, no_dfma=True):
    """
    annotations : pandas Dataframe.
        Pandas dataframe containing annotated metaphors
        and conventionality. 
    no_dfma (optional) : Boolean.
        Whether to include the category DFMA or not. Default is True.
    ---------------------------
    Prepares the data for the paired t test and writes
    it to file.
    """
    generations = ["human", "gpt3-5", "gpt4o"]
    coarse_dfs, fine_dfs = [], []
    if no_dfma:
        annotations = annotations[annotations["metaphor"]!="DFMA"]
    for gen in generations:
        metaphors = annotations[annotations["gen_version"]==gen]
        coarse_metaphors = metaphors.copy()
        coarse_metaphors['metaphor'] = coarse_metaphors['metaphor'].replace(
            {"KOMPL": "METAPHOR", "MRW\_DIR": "METAPHOR", "MRW": "METAPHOR", "PERS": "METAPHOR", "WIDLII": "METAPHOR"}
            )
        coarse_metaphors = coarse_metaphors.groupby(['url','metaphor']).size().unstack(fill_value=0).transpose()
        metaphors = metaphors.groupby(['url','metaphor']).size().unstack(fill_value=0).transpose()
        metaphors_per_doc = transform_df_to_rel_freq(metaphors).reset_index()
        coarse_metaphors = transform_df_to_rel_freq(coarse_metaphors).reset_index()
        coarse_metaphors["gen_type"], metaphors_per_doc["gen_type"] = gen, gen
        if gen != "gpt3-5": # Only GPT-3.5 has annotations containing MRW_DIR
            metaphors_per_doc["MRW\_DIR"], metaphors["MRW\_DIR"] = 0.0, 0.0
        coarse_dfs.append(coarse_metaphors)
        fine_dfs.append(metaphors_per_doc)

        coarse_df = pd.concat(coarse_dfs, ignore_index=True)
        fine_df = pd.concat(fine_dfs, ignore_index=True)
        fine_df.reset_index(drop=True, inplace=True)
        fine_df.sort_values(by='url', inplace=True)
        coarse_df.reset_index(drop=True, inplace=True)
        coarse_df.sort_values(by='url', inplace=True)
        if no_dfma:
            coarse_df.to_csv(f"{STATISTICS_DIR}/coarse_pairs_nodfma.csv", index=False)
            fine_df.to_csv(f"{STATISTICS_DIR}/fine_pairs_nodfma.csv", index=False)
        else:
            coarse_df.to_csv(f"{STATISTICS_DIR}/coarse_pairs.csv", index=False)
            fine_df.to_csv(f"{STATISTICS_DIR}/fine_pairs.csv", index=False)            


def paired_tests(no_dfma=True):
    """
    no_dfma (optional) : Boolean.
        Whether to include the category DFMA or not. Default is True.
    ---------------------------
    Conducts a paired t test and wilcoxon signed rank test 
    and writes the results to files.
    """
    if no_dfma:
        coarse_df, fine_df = pd.read_csv(f"{STATISTICS_DIR}/coarse_pairs_nodfma.csv"), pd.read_csv(f"{STATISTICS_DIR}/fine_pairs_nodfma.csv")
    else: 
        coarse_df, fine_df = pd.read_csv(f"{STATISTICS_DIR}/coarse_pairs.csv"), pd.read_csv(f"{STATISTICS_DIR}/fine_pairs.csv")
    coarse_results, fine_results = {}, {}
    if no_dfma:
        cat_list_coarse, cat_list_fine = ["METAPHOR", "O"], ["MRW", "PERS", "KOMPL", "WIDLII", "O"]
    else: 
        cat_list_coarse, cat_list_fine = ["METAPHOR", "DFMA", "O"], ["MRW", "PERS", "KOMPL", "WIDLII", "DFMA", "O"]
    for cat in cat_list_coarse:
        print(f"Category: {cat}")
        gpt35 = coarse_df[coarse_df["gen_type"]=="gpt3-5"]
        gpt35 = gpt35[cat]
        gpt35.reset_index(drop=True, inplace=True)
        human = coarse_df[coarse_df["gen_type"]=="human"]
        human = human[cat]
        human.reset_index(drop=True, inplace=True)
        gpt4o = coarse_df[coarse_df["gen_type"]=="gpt4o"]
        gpt4o = gpt4o[cat]
        gpt4o.reset_index(drop=True, inplace=True)

        h_gpt35_stat, h_gpt35_p_value = wilcoxon(human, gpt35)
        h_gpt4o_stat, h_gpt4o_p_value = wilcoxon(human, gpt4o)
        wilcoxon_gpts_stat, wilcoxon_gpts_p_value = wilcoxon(gpt35, gpt4o)
        h_gpt35_tstat, h_gpt35_p_tvalue = ttest_rel(human, gpt35)
        h_gpt4o_tstat, h_gpt4o_p_tvalue = ttest_rel(human, gpt4o)
        ttest_gpts_stat, ttest_gpts_p_value = ttest_rel(gpt35, gpt4o)

        coarse_results[cat] = {"human_vs_gpt3-5": 
                               {"wilcoxon": 
                                {"p_value": h_gpt35_p_value, "t_stat": h_gpt35_stat}, 
                               "paired_t_test": {"p_value": h_gpt35_p_tvalue, "t_stat": h_gpt35_tstat}
                               },
                               "human_vs_gpt4o": 
                                {"wilcoxon": 
                                 {"p_value": h_gpt4o_p_value, "t_stat": h_gpt4o_stat}, 
                               "paired_t_test": {"p_value": h_gpt4o_p_tvalue, "t_stat": h_gpt4o_tstat}
                                },
                                "gpts":
                                {"wilcoxon":
                                 {"p_value":wilcoxon_gpts_p_value, "t_stat": wilcoxon_gpts_stat},
                                 "paired_t_test": {"p_value": ttest_gpts_p_value, "t_stat": ttest_gpts_stat}}
                                }
    for cat in cat_list_fine:
        print(f"Category: {cat}")
        gpt35 = fine_df[fine_df["gen_type"]=="gpt3-5"]
        gpt35 = gpt35[cat]
        gpt35.reset_index(drop=True, inplace=True)
        human = fine_df[fine_df["gen_type"]=="human"]
        human = human[cat]
        human.reset_index(drop=True, inplace=True)
        gpt4o = fine_df[fine_df["gen_type"]=="gpt4o"]
        gpt4o = gpt4o[cat]
        gpt4o.reset_index(drop=True, inplace=True)
        h_gpt35_stat, h_gpt35_p_value = wilcoxon(human, gpt35)
        h_gpt4o_stat, h_gpt4o_p_value = wilcoxon(human, gpt4o)
        wilcoxon_gpts_stat, wilcoxon_gpts_p_value = wilcoxon(gpt35, gpt4o)
        h_gpt35_tstat, h_gpt35_p_tvalue = ttest_rel(human, gpt35)
        h_gpt4o_tstat, h_gpt4o_p_tvalue = ttest_rel(human, gpt4o)
        ttest_gpts_stat, ttest_gpts_p_value = ttest_rel(gpt35, gpt4o)
        fine_results[cat] = {"human_vs_gpt3-5": 
                               {"wilcoxon": 
                                {"p_value": h_gpt35_p_value, "t_stat": h_gpt35_stat}, 
                               "paired_t_test": {"p_value": h_gpt35_p_tvalue, "t_stat": h_gpt35_tstat}
                               },
                               "human_vs_gpt4o": 
                                {"wilcoxon": 
                                 {"p_value": h_gpt4o_p_value, "t_stat": h_gpt4o_stat}, 
                               "paired_t_test": {"p_value": h_gpt4o_p_tvalue, "t_stat": h_gpt4o_tstat}
                                },
                                "gpts":
                                {"wilcoxon":
                                 {"p_value": wilcoxon_gpts_p_value, "t_stat": wilcoxon_gpts_stat},
                                 "paired_t_test": {"p_value": ttest_gpts_p_value, "t_stat": ttest_gpts_stat}
                                }
                            }
    if no_dfma:
        with open(f"{STATISTICS_DIR}/inferential/paired_tests_coarse_nodfma.json", "w", encoding="utf8") as out:
            json.dump(coarse_results, out, ensure_ascii=False, indent=3)
        with open(f"{STATISTICS_DIR}/inferential/paired_tests_fine_nodfma.json", "w", encoding="utf8") as out:
            json.dump(fine_results, out, ensure_ascii=False, indent=3)
    else:
        with open(f"{STATISTICS_DIR}/inferential/paired_tests_coarse.json", "w", encoding="utf8") as out:
            json.dump(coarse_results, out, ensure_ascii=False, indent=3)
        with open(f"{STATISTICS_DIR}/inferential/paired_tests_fine.json", "w", encoding="utf8") as out:
            json.dump(fine_results, out, ensure_ascii=False, indent=3)  

def describe_pos(annotations, no_dfma=True):
    """
    annotations : pandas Dataframe.
        Pandas dataframe containing annotated metaphors
        and conventionality. 
    no_dfma (optional) : Boolean.
        Whether to include the category DFMA or not. Default is True.
    ---------------------------
    Provides descriptive statistics for POS-Tags and writes the
    results to file.
    """
    if no_dfma:
        annotations = annotations[annotations["metaphor"]!= "DFMA"]
    pos_df = annotations[~annotations["metaphor"].isin(["_", "O", "KOMPL"])]
    compound_df = pos_df[pos_df["compound"]==True]
    compound_df["pos"] = "compound"
    pos_df = pos_df[pos_df["compound"]==False]
    merged_df = pd.concat([pos_df.set_index("gen_version"), compound_df.set_index("gen_version")])
    merged_df = merged_df.groupby(['gen_version', 'pos']).size().unstack(fill_value=0).transpose()
    merged_df = add_relative_freq_genversion(merged_df)
    if no_dfma:
        with open(f"{STATISTICS_DIR}/tex/pos_nodfma.tex", "w") as file:
            file.write(merged_df.to_latex(index=True, float_format="%.2f"))
    else:
        with open(f"{STATISTICS_DIR}/tex/pos.tex", "w") as file:
            file.write(merged_df.to_latex(index=True, float_format="%.2f"))


def create_histograms():
    """
    Creates histograms for paired articles.
    """
    coarse = pd.read_csv(f"{STATISTICS_DIR}/coarse_pairs_nodfma.csv", encoding="utf8")
    fine = pd.read_csv(f"{STATISTICS_DIR}/fine_pairs_nodfma.csv", encoding="utf8")
    coarse["url"] = [url.replace("_1a","") for url in coarse["url"]]
    fine["url"] = [url.replace("_1a","") for url in fine["url"]]
    unique_urls = list(set(coarse["url"]))
    url_to_text_mapping = {url: f"Text {i+1}" for i, url in enumerate(unique_urls)}

    colors = {"human": "blue", "gpt3-5": "orange", "gpt4o": "green"}
    coarse['text_id'] = coarse['url'].map(url_to_text_mapping)
    fine['text_id'] = fine['url'].map(url_to_text_mapping)

    coarse['text_id'] = pd.Categorical(coarse['text_id'], categories=sorted(coarse['text_id'].unique(), key=lambda x: int(x[4:])), 
    ordered=True)
    fine['text_id'] = pd.Categorical(fine['text_id'], categories=sorted(coarse['text_id'].unique(), key=lambda x: int(x[4:])), 
    ordered=True)
    fine = fine.sort_values("text_id")
    coarse = coarse.sort_values('text_id')
    pivot_coarse = coarse.pivot(index="text_id", columns="gen_type", values="METAPHOR")

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    x = np.arange(len(pivot_coarse))

    for i, gen_type in enumerate(pivot_coarse.columns):
        ax.bar(
            x + i * bar_width,
            pivot_coarse[gen_type],
            width=bar_width,
            label=gen_type,
            color= colors[gen_type],        
        )

    # Add labels and formatting
    ax.set_xlabel("Paired Texts", fontsize=14)
    ax.set_ylabel("Relative Frequency of METAPHOR", fontsize=14)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(pivot_coarse.index, fontsize=14)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/metaphor_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

    for cat in ["KOMPL","MRW","O","PERS","WIDLII","MRW\_DIR"]:
        pivot = fine.pivot(index="text_id", columns="gen_type", values=cat)

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.25
        x = np.arange(len(pivot))

        for i, gen_type in enumerate(pivot.columns):
            ax.bar(
                x + i * bar_width,
                pivot[gen_type],
                width=bar_width,
                label=gen_type,
                color= colors[gen_type],        
            )

        ax.set_xlabel("Paired Texts", fontsize=14)
        backslash_char = "\\_"
        ax.set_ylabel(f"Relative Frequency of {cat.replace(backslash_char,' ')}", fontsize=14)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(pivot.index, fontsize=14)
        ax.grid(axis="y", alpha=0.5)
        ax.legend(fontsize=14)

        plt.tight_layout()
        plt.savefig(f"{IMAGES_DIR}/{cat}_histogram.png", dpi=300, bbox_inches="tight")
        plt.show()        

if __name__ == '__main__':
    #### Set paths
    REPO_DIR = str(Path().resolve().parents[0])
    DATA_DIR = os.path.join(REPO_DIR, "data")
    HUMAN_DIR = os.path.join(DATA_DIR, "article_humans")
    GPT35_DIR = os.path.join(DATA_DIR, "article_llms/article_gpt-3.5-turbo-0125")
    GPT4o_DIR = os.path.join(DATA_DIR, "article_llms/article_gpt-4o-2024-05-13")
    STATISTICS_DIR = os.path.join(REPO_DIR, "statistics")
    IMAGES_DIR = os.path.join(STATISTICS_DIR, "images")

    #### Load spacy
    nlp = spacy.load("de_core_news_md")

    #### Descriptive Statistics: Tables
    # all_annotations = pd.read_csv(f"{DATA_DIR}/metaphor_dataset.csv", encoding="utf8")
    # mflags = pd.read_csv(f"{DATA_DIR}/mflag_dataset.csv", encoding="utf8")
    # descriptive_statistics(all_annotations, no_dfma=True)
    # describe_pos(all_annotations, no_dfma=False)
    # describe_articles()

    # #### Descriptive Statistics: Visualization
    # prepare_counts_across_docs(all_annotations, write_file=True, no_dfma=True)
    # visualize_across_docs()
    # create_histograms()

    # #### Inferential Statistics
    # independent_t_test(all_annotations, no_dfma=True)
    # create_paired_data(all_annotations, no_dfma=True)
    paired_tests(no_dfma=True)

    
