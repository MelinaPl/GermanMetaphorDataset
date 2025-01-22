"""
@author: Melina Plakidis
"""
from zipfile import ZipFile
import pandas as pd
import os
from pathlib import Path
import ast
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import json
import networkx as nx
pd.options.mode.chained_assignment = None 

"""
This script is used to retrieve qualitative results.
"""

def write_frequency_lists(df, gen_type):
    """
    df : pandas Dataframe.
        Dataframe containing all annotations.
    gen_type : str.
        Generation type, values either 'human', 'gpt3-5' or 'gpt4o'
    ----------------------------
    Writes frequency lists (both CSV and Tex) of lemmas, tokens, compounds,
    and multiword metaphors for each of the three generation types.
    """
    lemmas, tokens, compounds, multis = [], [], [], []
    for text, compound in zip(df["text"],df["compound"]):
        if not text.startswith("["):
            if not compound == True:
                doc = nlp(text)
                lemma = [token.lemma_ for token in doc]
                lemmas.append(lemma[0])
                tokens.append(text)
            else:
                compounds.append(text)
        else: # More than one token. Includes verbs and complex metaphors
            try:
                multis.append(ast.literal_eval(text))
            except:
                text = text.replace("[","").replace("]","")
                text = text.split(",")
                elements = [t.strip() for t in text]
                multis.append(elements)
    counter_lemmas = Counter(lemmas)
    counter_tokens = Counter(tokens)
    counter_compounds = Counter(compounds)
    lemma_df = pd.DataFrame(counter_lemmas.most_common(), columns=['Word', 'Frequency'])
    token_df = pd.DataFrame(counter_tokens.most_common(), columns=['Word', 'Frequency'])
    compound_df = pd.DataFrame(counter_compounds.most_common(), columns=['Word', 'Frequency'])
    multi_df = pd.DataFrame({'Word': multis,'Frequency': 1})
    # CSV
    with open(f"{QUALI_DIR}/csv/{gen_type}_mostfrequent_lemmas.csv", "w") as file:
        file.write(lemma_df.to_csv(index=False))
    with open(f"{QUALI_DIR}/csv/{gen_type}_mostfrequent_tokens.csv", "w") as file:
        file.write(token_df.to_csv(index=False))
    with open(f"{QUALI_DIR}/csv/{gen_type}_mostfrequent_compounds.csv", "w") as file:
        file.write(compound_df.to_csv(index=False)) 
    with open(f"{QUALI_DIR}/csv{gen_type}_mostfrequent_multis.csv", "w") as file:
        file.write(multi_df.to_csv(index=False))
    # TEX    
    with open(f"{QUALI_DIR}/tex/{gen_type}_mostfrequent_lemmas.tex", "w") as file:
        file.write(lemma_df.to_latex(index=False))
    with open(f"{QUALI_DIR}/tex/{gen_type}_mostfrequent_tokens.tex", "w") as file:
        file.write(token_df.to_latex(index=False))
    with open(f"{QUALI_DIR}/tex/{gen_type}_mostfrequent_compounds.tex", "w") as file:
        file.write(compound_df.to_latex(index=False)) 
    with open(f"{QUALI_DIR}/tex/{gen_type}_mostfrequent_multis.tex", "w") as file:
        file.write(multi_df.to_latex(index=False))

def frequency_main(annotations):   
    """
    annotations : pandas Dataframe.
        Dataframe containing all annotations.
    ----------------------------
    Main function to write frequency lists (both CSV and Tex) of lemmas, tokens, compounds,
    and multiword metaphors for each of the three generation types.    
    """  
    metaphors_human = annotations[annotations["gen_version"]=="human"]
    metaphors_human = metaphors_human[~metaphors_human["metaphor"].isin(["DFMA", "O"])]
    metaphors_gpt35 = annotations[annotations["gen_version"]=="gpt3-5"]
    metaphors_gpt35 = metaphors_gpt35[~metaphors_gpt35["metaphor"].isin(["DFMA", "O"])]
    metaphors_gpt4o = annotations[annotations["gen_version"]=="gpt4o"]
    metaphors_gpt4o = metaphors_gpt4o[~metaphors_gpt4o["metaphor"].isin(["DFMA", "O"])]
    write_frequency_lists(metaphors_human, gen_type="human")
    write_frequency_lists(metaphors_gpt35, gen_type="gpt35")
    write_frequency_lists(metaphors_gpt4o, gen_type="gpt4o")

def create_graph_data(annotations):
    """
    annotations : pandas Dataframe.
        Dataframe containing all annotations.
    ----------------------------
    Prepares the graph data and writes it to a JSON file.   
    """  
    generations = ["human", "gpt3-5", "gpt4o"]
    annotations = annotations[~annotations["metaphor"].isin(["DFMA", "O","_"])]
    data = {}
    for gen in generations:
        gen_annotations = annotations[annotations["gen_version"]==gen]
        lemmas, tokens, lemmas_ordered, counts_ordered = [], [], [], []
        for text, compound in zip(gen_annotations["text"],gen_annotations["compound"]):
            if not text.startswith("["):
                if not compound:
                    doc = nlp(text)
                    lemma = [token.lemma_ for token in doc]
                    lemmas.append(lemma[0])
                    tokens.append(text)
        counter_lemmas = Counter(lemmas)
        counter_tokens = Counter(tokens)
        for lemma, count in counter_lemmas.most_common():
            lemmas_ordered.append(lemma)
            counts_ordered.append(count)
        data[gen] = {"lemmas": lemmas_ordered, "counts": counts_ordered}
    with open(f"{QUALI_DIR}/graph_lemmas.json", "w", encoding="utf8") as out:
        json.dump(data, out, ensure_ascii=False, indent=4)


def create_graph():
    """
    Generates the graphml file.
    """
    # Load and prepare data
    with open(f"{QUALI_DIR}/graph_lemmas.json", "r", encoding="utf8") as inp:
        data = json.load(inp)
        generations = ["human", "gpt3-5", "gpt4o"]
        all_lemmas = []
        all_gens = []
        for gen in generations:
            all_lemmas.extend(data[gen]["lemmas"])
            all_gens.append({"id": f"gen_{gen}", "name": gen})
        unique_lemmas = list(set(all_lemmas))
        all_lemmas = []
        for i, l in enumerate(unique_lemmas):
            all_lemmas.append({"id": f"lemma_{str(i)}", "name": l})

        lemma_dict = {l["name"]: l["id"] for l in all_lemmas}
        G = nx.DiGraph()

        # Add nodes for lemmas
        for lemma in all_lemmas:
            G.add_node(lemma['id'], name=lemma['name'], node_type="lemma")
        # Add nodes for generation types
        for gen in all_gens:
            G.add_node(gen['id'], name=gen['name'], node_type="gen_type")

        # Draw relations
        relations = []
        rel_attributes = {}
        for gen in generations:
            lemmas = data[gen]["lemmas"]
            counts = data[gen]["counts"]
            for l, c in zip(lemmas, counts):
                idx = lemma_dict.get(l)
                relations.append((f"gen_{gen}", idx))
                rel_attributes[(f"gen_{gen}", idx)] = {"count": c}

        G.add_edges_from(relations)
        nx.set_edge_attributes(G, rel_attributes)
        pos = nx.spring_layout(G)
        node_colors = []
        for node, data in G.nodes(data=True):
            if data['node_type'] == 'gen_type':
                node_colors.append('skyblue')
            else:
                node_colors.append('lightgreen')

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
        nx.draw_networkx_edges(G, pos)
        node_labels = {node: data['name'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=node_labels)
        nx.write_graphml(G, f"{QUALI_DIR}/graph_lemmas.graphml")

        print("Graph saved as 'graph_lemmas.graphml'")
        plt.show()
        # Plot was further processed in Gephi. Thus, visual presentation of graph
        # in python not to be used.

def unique_lemmas_tables():
    """
    Writes text files containing lemmas that have either been
    exclusively used by GPT3-5, GPT4o, humans, both GPTs, GPT3-5 and humans, GPT4o
    and humans or all three gen_types. 
    """    
    # Load data
    with open(f"{QUALI_DIR}/graph_lemmas.json", "r", encoding="utf8") as inp:
        data = json.load(inp)

    generations = {"human": [], "gpt3-5": [], "gpt4o": []}
    for gen in generations:
        generations[gen] = set(data[gen]["lemmas"])

    # Restructure the data
    all_lemmas = list(generations["human"] & generations["gpt3-5"] & generations["gpt4o"])
    used_by_all = (generations["human"] & generations["gpt3-5"] & generations["gpt4o"])
    gpts = ((generations["gpt3-5"] & generations["gpt4o"]) - generations["human"]) - used_by_all
    gpt35_human = ((generations["gpt3-5"] & generations["human"]) - generations["gpt4o"]) - used_by_all
    gpt4o_human = ((generations["gpt4o"] & generations["human"]) - generations["gpt3-5"]) - used_by_all

    # List of unique lemmas that only occur in either GPT3-5, GPT4o or human texts
    clean_gpt35 = generations["gpt3-5"] - (gpt35_human | gpts | used_by_all)
    clean_gpt4o = generations["gpt4o"] - (gpt4o_human | gpts | used_by_all)
    clean_human = generations["human"] - (gpt35_human | gpt4o_human | used_by_all)

    # Create a results dict
    result_map = {
        "gpt35": clean_gpt35,
        "gpt4o": clean_gpt4o,
        "human": clean_human,
        "gpt35_human": gpt35_human,
        "gpt4o_human": gpt4o_human,
        "gpts": gpts,
        "used_by_all": all_lemmas,
    }

    # Write results to files
    for filename, lemma_set in result_map.items():
        with open(f"{QUALI_DIR}/exclusive_lemmas/exclusive_lemmas_{filename}.txt", "w", encoding="utf8") as out:
            for word in sorted(lemma_set):
                out.write(word + "\n")


def create_lemma_tables():
    """
    Creates tables with results and writes them to files.
    """
    excl_files = [file for file in os.listdir(f"{QUALI_DIR}/exclusive_lemmas")]
    excl_lemmas = {}
    for file in excl_files:
        filename = file.replace("exclusive_lemmas_","").replace(".txt", "")
        with open(f"{QUALI_DIR}/exclusive_lemmas/{file}", "r", encoding="utf8") as inp:
            data = inp.read()
            data = data.split("\n")
            data = data[:-1]
        excl_lemmas[filename] = {"lemmas": data}
    all, gpts, g35_h, g4o_h =  excl_lemmas["used_by_all"]["lemmas"],  excl_lemmas["gpts"]["lemmas"], excl_lemmas["gpt35_human"]["lemmas"],  excl_lemmas["gpt4o_human"]["lemmas"]
    ordered = sorted([all, gpts, g35_h, g4o_h], key=lambda x: len(x), reverse=True)
    max_len = len(ordered[0])
    for l in ordered[1:]:
        current_length = len(l)
        fill_number = max_len - current_length
        l.extend([""]*fill_number)
    df = pd.DataFrame({"All": all, "GPTs": gpts, "GPT3-5 and Human": g35_h, "GPT4o and Human": g4o_h})
    df.to_latex(f"{QUALI_DIR}/tex/exlusive_shared_lemmas.tex", index=False)
    ### Create multi lemma + compound tables
    # Create multi lemma df
    gpt4o = pd.read_csv(f"{QUALI_DIR}/csv/gpt4o_mostfrequent_multis.csv", encoding="utf8")
    gpt35 = pd.read_csv(f"{QUALI_DIR}/csv/gpt35_mostfrequent_multis.csv", encoding="utf8")
    human = pd.read_csv(f"{QUALI_DIR}/csv/human_mostfrequent_multis.csv", encoding="utf8")
    gpt4o = list(gpt4o["Word"].values)   
    gpt35 = list(gpt35["Word"].values)   
    human = list(human["Word"].values)  
    ordered = sorted([gpt4o, gpt35, human], key=lambda x: len(x), reverse=True) 
    max_len = len(ordered[0])
    for l in ordered[1:]:
        current_length = len(l)
        fill_number = max_len - current_length
        l.extend([""]*fill_number)
    df_multi = pd.DataFrame({"GPT3-5": gpt35, "GPT-4o": gpt4o, "Human": human})
    # Create compound lemma df
    gpt4o = pd.read_csv(f"{QUALI_DIR}/csv/gpt4o_mostfrequent_compounds.csv", encoding="utf8")
    gpt35 = pd.read_csv(f"{QUALI_DIR}/csv/gpt35_mostfrequent_compounds.csv", encoding="utf8")
    human = pd.read_csv(f"{QUALI_DIR}/csv/human_mostfrequent_compounds.csv", encoding="utf8")
    gpt4o = list(gpt4o["Word"].values)   
    gpt35 = list(gpt35["Word"].values)   
    human = list(human["Word"].values)  
    ordered = sorted([gpt4o, gpt35, human], key=lambda x: len(x), reverse=True) 
    max_len = len(ordered[0])
    for l in ordered[1:]:
        current_length = len(l)
        fill_number = max_len - current_length
        l.extend([""]*fill_number)
    
    df_compounds = pd.DataFrame({"GPT3-5": gpt35, "GPT-4o": gpt4o, "Human": human})
    merged_df = pd.concat([df_multi, df_compounds], axis=0)
    merged_df.to_latex(f"{QUALI_DIR}/tex/multilemmas_and_compounds.tex", index=False)

def calculate_simple_LOS():
    """
    Calculates the lexical overlap score.
    LOS: n / A+B-n
    where n = overlapping lemmas, A = lemmas in source A, 
    B = lemmas in source B
    """
    results = {}

    with open(f"{QUALI_DIR}/graph_lemmas.json", "r", encoding="utf8") as inp:
        data = json.load(inp)   
        human_lemmas, gpt35_lemmas, gpt4o_lemmas = data["human"]["lemmas"], data["gpt3-5"]["lemmas"], data["gpt4o"]["lemmas"]
        human_gpt35 = set(data["human"]["lemmas"]) | set(data["gpt3-5"]["lemmas"])
        human_gpt35_overlap = set(data["human"]["lemmas"]) & set(data["gpt3-5"]["lemmas"])
        human_gpt4o = set(data["human"]["lemmas"]) | set(data["gpt4o"]["lemmas"])
        human_gpt4o_overlap = set(data["human"]["lemmas"]) & set(data["gpt4o"]["lemmas"])
        gpt35_gpt4o = set(data["gpt4o"]["lemmas"]) | set(data["gpt3-5"]["lemmas"])
        gpt35_gpt4o_overlap = set(data["gpt4o"]["lemmas"]) & set(data["gpt3-5"]["lemmas"])

        los_human_gpt35 = len(human_gpt35_overlap)/ len(human_gpt35)
        los_human_gpt4o = len(human_gpt4o_overlap)/ len(human_gpt4o)
        los_gpt35_gpt4o = len(gpt35_gpt4o_overlap)/ len(gpt35_gpt4o)
        results["human_gpt35"] = {"overlap": len(human_gpt35_overlap), "total_lemmas": len(human_gpt35), "los":los_human_gpt35}
        results["human_gpt4o"] = {"overlap": len(human_gpt4o_overlap), "total_lemmas": len(human_gpt4o), "los":los_human_gpt4o}
        results["gpts"] = {"overlap": len(gpt35_gpt4o_overlap), "total_lemmas": len(gpt35_gpt4o), "los":los_gpt35_gpt4o}
        results["total_lemmas"] = {"humans": len(human_lemmas), "gpt35": len(gpt35_lemmas), "gpt4o": len(gpt4o_lemmas)}

    with open(f"{QUALI_DIR}/simple_lexicon_overlap_score.json", "w", encoding="utf8") as out:
        json.dump(results, out, ensure_ascii=False, indent=4)
    print(results)

if __name__ == '__main__':
    #### Set paths
    REPO_DIR = str(Path().resolve().parents[0])
    DATA_DIR = os.path.join(REPO_DIR, "data")
    QUALI_DIR = os.path.join(REPO_DIR, "qualitative")

    ####  Load spacy
    nlp = spacy.load("de_core_news_md")

    #### Qualitative analysis
    all_annotations = pd.read_csv(f"{DATA_DIR}/metaphor_dataset.csv", encoding="utf8")
    print(all_annotations)
    frequency_main(all_annotations)

    #### Graph
    create_graph_data(all_annotations)
    create_graph()

    #### Exclusively used lemmas across generation types
    unique_lemmas_tables()
    create_lemma_tables()
    calculate_simple_LOS()
