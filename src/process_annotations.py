"""
@author: Melina Plakidis
"""
from zipfile import ZipFile
import pandas as pd
import os
from pathlib import Path
import ast
pd.options.mode.chained_assignment = None


def annotation_iterator(anno_zipfile): # build iterator
    archive = ZipFile(f"{ANNO_DIR}/{anno_zipfile}", 'r')
    files = [name for name in archive.namelist()]
    archive.extractall(path=ANNO_DIR)
    for file in files:
        if file.endswith(".tsv"):
            print(file)
            annotations = read_webanno(file)
            annotations = handle_compounds(annotations)
            annotations = handle_verbs(annotations)
            annotations = handle_complex_metaphors(annotations)
            mflag_df = extract_mflag(file)
            yield annotations, mflag_df

def extract_mflag(file):
    potentially_nonexistent = ["relation", "rel_id", "MetaphorFlag"]
    doc_id = file.split("/")
    doc_id = doc_id[1].replace(".xmi", "")
    with open(f"{ANNO_DIR}/{file}", "r", encoding="utf8") as inp:
        lines = inp.readlines()
        all_tokens = []
        order_columns = []
        current_sent = ""
        for line in lines:
            if line.strip() == "":
                continue
            elif "#T_SP=webanno.custom.MetaphorFlags|MetaphorFlag" in line:
                order_columns.append("MetaphorFlag")
            elif "#T_SP=webanno.custom.Metaphors|Conventionality|Metaphern" in line:
                order_columns.append("conventionality")
                order_columns.append("metaphor")
            elif "#T_SP=webanno.custom.POS|pos_tag" in line:
                order_columns.append("pos")
            elif "#T_RL=webanno.custom.UnitofAnalysis|bidirectionalrelation|BT_webanno.custom.Metaphors" in line:
                order_columns.append("relation")
            elif line.startswith("#Text=") and len(line.strip())>len("#Text="):
                current_sent = line.replace("#Text=", "").replace("\n", "")
            elif line.startswith("#Text=") and len(line.strip())==len("#Text="):
                continue
            elif "#FORMAT=WebAnno TSV 3.3" in line:
                continue
            else:
                try:
                    columns = line.replace("\n", "").split("\t")
                    if not columns[0] == "" and columns[2]!="":
                        columns.append(current_sent)
                        columns = [column for column in columns if not column == ""]
                        all_tokens.append(columns)
                except: 
                    print(columns)
    if not "MetaphorFlag" in order_columns:
        mflag_df = None
    else:
        pd_columns = ["s_id", "char_id", "text"]
        pd_columns.extend([order_column for order_column in order_columns])
        if "relation" in order_columns:
            pd_columns.append("rel_id")
        pd_columns.append("s_text")
        df = pd.DataFrame(all_tokens, columns=pd_columns)
        df["doc_id"] = [doc_id]*len(df)
        for name in potentially_nonexistent:
            if name in df.columns:
                continue
            else:
                df[name] = ["_"]*len(df)
        mflag_df = df[(df['MetaphorFlag'] == 'MFlag')]
        if doc_id in mapping_gpt35["ids"].values:
            url = mapping_gpt35.loc[mapping_gpt35['ids'] == doc_id, 'urls'].values[0]
            gen_version = "gpt3-5"
        elif doc_id in mapping_gpt4o["ids"].values:
            url = mapping_gpt4o.loc[mapping_gpt4o['ids'] == doc_id, 'urls'].values[0]
            gen_version = "gpt4o"
        elif doc_id in mapping_human["ids"].values:
            url = mapping_human.loc[mapping_human['ids'] == doc_id, 'urls'].values[0]
            gen_version = "human"
        else:
            url = "NONE"
            gen_version = "NONE"
        mflag_df["url"] = [url]*len(mflag_df)
        mflag_df["gen_version"] = [gen_version]*len(mflag_df)
    return mflag_df

def complex_column(char_id, metaphor):
    if metaphor == 'KOMPL':
        try:
            elements = ast.literal_eval(char_id)
            if isinstance(elements, list):
                return len(elements)
        except (ValueError, SyntaxError):
            return 1
    return 0

def handle_compounds(annotation_df):
    whole_compound_ids, compound_ids = [], []
    for idx in annotation_df["s_id"].values: 
        if "." in idx:
            whole_compound = idx.split(".")[0]
            whole_compound_ids.append(whole_compound)
            compound_ids.append(idx)
    whole_compound_ids = list(set(whole_compound_ids))
    for idx in whole_compound_ids:
        pos = annotation_df.loc[annotation_df['s_id'] == idx, 'pos'].values[0]
        for c in compound_ids:
            if c.startswith(idx):
                annotation_df.loc[annotation_df['s_id'] == c, 'pos'] = pos
        annotation_df = annotation_df.loc[annotation_df['s_id'] != idx]
    annotation_df['compound'] = annotation_df['s_id'].apply(lambda x: True if '.' in x else False)
    return annotation_df

def handle_verbs(annotation_df):
    pairs = []
    for idx, relation, rel_id, metaphor in zip(annotation_df["s_id"].values, annotation_df["relation"].values, annotation_df["rel_id"].values, annotation_df["metaphor"].values): 
        if relation == "belong together":
            if not metaphor == "KOMPL":
                pairs.append([idx, rel_id])
    for pair in pairs:
        first_number = int(pair[0].replace("-",""))
        second_number = int(pair[1].replace("-",""))
        if first_number < second_number:
            new_pair = [pair[0], pair[1]]
        else:
            new_pair = [pair[1], pair[0]]
        first_item_text = annotation_df.loc[annotation_df['s_id'] == new_pair[0], 'text'].values[0]
        first_item_pos = annotation_df.loc[annotation_df['s_id'] == new_pair[0], 'pos'].values[0]
        first_item_char_id = annotation_df.loc[annotation_df['s_id'] == new_pair[0], 'char_id'].values[0]
        second_item_text = annotation_df.loc[annotation_df['s_id'] == new_pair[1], 'text'].values[0]
        second_item_pos = annotation_df.loc[annotation_df['s_id'] == new_pair[1], 'pos'].values[0]
        second_item_char_id = annotation_df.loc[annotation_df['s_id'] == new_pair[1], 'char_id'].values[0]
        new_data = {"text": f"[{first_item_text}, {second_item_text}]","pos": f"[{first_item_pos}, {second_item_pos}]", "char_id": f"[{first_item_char_id}, {second_item_char_id}]"}
        annotation_df.loc[annotation_df["s_id"] == new_pair[0], new_data.keys()] = new_data.values()
        annotation_df = annotation_df.loc[annotation_df["s_id"] != new_pair[1]]
    return annotation_df

def handle_complex_metaphors(annotation_df): # have directed relationships (left to right)
    pairs = []
    for idx, relation, rel_id, metaphor in zip(annotation_df["s_id"].values, annotation_df["relation"].values, annotation_df["rel_id"].values, annotation_df["metaphor"].values): 
        if relation == "belong together":
            if metaphor == "KOMPL":
                pairs.append([rel_id, idx])
    previous_pair = [0,0]
    complex_metaphor_pairs = []
    for pair in pairs:
        all_values = []
        if pair[0] == previous_pair[-1]:
            all_values.extend(previous_pair)
            all_values.append(pair[1])
            complex_metaphor_pairs.append(all_values)
            previous_pair = all_values
        else:
            complex_metaphor_pairs.append(pair)
            previous_pair = pair
    cleaned_list = []
    for current_list in sorted(complex_metaphor_pairs, key=len, reverse=True):
        # Check if current_list is a sublist of any list in the result
        if not any(set(current_list).issubset(set(existing)) for existing in cleaned_list):
            cleaned_list.append(current_list)
    for metaphor in cleaned_list:
        new_data = {}
        pos_tags, texts, char_ids = [], [], []
        int_metaphor = [int(ele.replace("-","")) for ele in metaphor]
        assert int_metaphor == sorted(int_metaphor) # Check if order actually is maintained
        for part in metaphor:
            pos = annotation_df.loc[annotation_df['s_id'] == part, 'pos'].values[0]
            text = annotation_df.loc[annotation_df['s_id'] == part, 'text'].values[0]
            char_id = annotation_df.loc[annotation_df['s_id'] == part, 'char_id'].values[0]
            pos_tags.append(pos)
            texts.append(text)
            char_ids.append(char_id)
        new_data["text"] = f"{texts}"
        new_data["pos"] = f"{pos_tags}"
        new_data["char_id"] = f"{char_ids}"
        annotation_df.loc[annotation_df["s_id"] == metaphor[0], new_data.keys()] = new_data.values()
        for part in metaphor[1:]:
            annotation_df = annotation_df.loc[annotation_df["s_id"] != part]
    annotation_df["complex"] = annotation_df.apply(lambda row: complex_column(row['char_id'], row['metaphor']), axis=1)

    return annotation_df


def read_webanno(tsv_file):
    potentially_nonexistent = ["relation", "rel_id", "MetaphorFlag"]
    doc_id = tsv_file.split("/")
    doc_id = doc_id[1].replace(".xmi", "")
    print(doc_id)
    with open(f"{ANNO_DIR}/{tsv_file}", "r", encoding="utf8") as inp:
        lines = inp.readlines()
        all_tokens = []
        order_columns = []
        current_sent = ""
        for line in lines:
            if line.strip() == "":
                continue
            elif "#T_SP=webanno.custom.MetaphorFlags|MetaphorFlag" in line:
                order_columns.append("MetaphorFlag")
            elif "#T_SP=webanno.custom.Metaphors|Conventionality|Metaphern" in line:
                order_columns.append("conventionality")
                order_columns.append("metaphor")
            elif "#T_SP=webanno.custom.POS|pos_tag" in line:
                order_columns.append("pos")
            elif "#T_RL=webanno.custom.UnitofAnalysis|bidirectionalrelation|BT_webanno.custom.Metaphors" in line:
                order_columns.append("relation")
            elif line.startswith("#Text=") and len(line.strip())>len("#Text="):
                current_sent = line.replace("#Text=", "").replace("\n", "")
            elif line.startswith("#Text=") and len(line.strip())==len("#Text="):
                continue
            elif "#FORMAT=WebAnno TSV 3.3" in line:
                continue
            else:
                try:
                    columns = line.replace("\n", "").split("\t")
                    if not columns[0] == "" and columns[2]!="":
                        columns.append(current_sent)
                        columns = [column for column in columns if not column == ""]
                        all_tokens.append(columns)
                except: 
                    print(columns)
    pd_columns = ["s_id", "char_id", "text"]
    pd_columns.extend([order_column for order_column in order_columns])
    if "relation" in order_columns:
        pd_columns.append("rel_id")
    pd_columns.append("s_text")
    df = pd.DataFrame(all_tokens, columns=pd_columns)
    df["doc_id"] = [doc_id]*len(df)
    for name in potentially_nonexistent:
        if name in df.columns:
            continue
        else:
            df[name] = ["_"]*len(df)
    restricted_wordclasses_df = df[(df['pos'] != '_') | (df['metaphor'] != '_')]
    restricted_wordclasses_df = restricted_wordclasses_df[(restricted_wordclasses_df['MetaphorFlag'] != "MFlag") | (restricted_wordclasses_df["metaphor"] != "_")]
    if doc_id in mapping_gpt35["ids"].values:
        url = mapping_gpt35.loc[mapping_gpt35['ids'] == doc_id, 'urls'].values[0]
        gen_version = "gpt3-5"
    elif doc_id in mapping_gpt4o["ids"].values:
        url = mapping_gpt4o.loc[mapping_gpt4o['ids'] == doc_id, 'urls'].values[0]
        gen_version = "gpt4o"
    elif doc_id in mapping_human["ids"].values:
        url = mapping_human.loc[mapping_human['ids'] == doc_id, 'urls'].values[0]
        gen_version = "human"
    else:
        url = "NONE"
        gen_version = "NONE"
    restricted_wordclasses_df["url"] = [url]*len(restricted_wordclasses_df)
    restricted_wordclasses_df["gen_version"] = [gen_version]*len(restricted_wordclasses_df)
    restricted_wordclasses_df = restricted_wordclasses_df.drop(columns=["MetaphorFlag"])
    return restricted_wordclasses_df




if __name__ == '__main__':
    #### Set paths
    REPO_DIR = str(Path().resolve().parents[0])
    ANNO_DIR = os.path.join(REPO_DIR, "annotations")
    DATA_DIR = os.path.join(REPO_DIR, "data")
    MAPPING_DIR = os.path.join(DATA_DIR, "mapping")

    #### Set variables
    mapping_gpt35 = pd.read_csv(f"{MAPPING_DIR}/mapping_gpt-3.5-turbo-0125.csv", sep=",", encoding="utf8")
    mapping_gpt4o =pd.read_csv(f"{MAPPING_DIR}/mapping_gpt-4o-2024-05-13.csv", sep=",", encoding="utf8")
    mapping_human = pd.read_csv(f"{MAPPING_DIR}/mapping_human.csv", sep=",", encoding="utf8")
    
    #### Process files
    read_webanno("curation/d5c938f5-1a53-421f-981d-ac432ef868b3.xmi/CURATION_USER16724183253005124917.tsv")
    all_annotations = []
    all_mflags = []

    # Create dataframes
    for annotations, mflag_df in annotation_iterator("final_curated_docs_tsv.zip"):
        all_annotations.append(annotations)
        if mflag_df is not None:
            all_mflags.append(mflag_df)

    # Write both dataframes to file
    if all_annotations:
        combined_annotations = pd.concat(all_annotations, ignore_index=True)
        combined_annotations.to_csv(f"{DATA_DIR}/my_final_dataset.csv", encoding="utf8")   

    if all_mflags:
        combined_mflag_df = pd.concat(all_mflags, ignore_index=True)
        combined_mflag_df.to_csv(f"{DATA_DIR}/mflag_dataset.csv", encoding="utf8")

        