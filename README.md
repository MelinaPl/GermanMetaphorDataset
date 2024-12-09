# German Metaphor Dataset: Metaphor Production in Large Language Models and Humans

This work presents a new corpus of both ChatGPT-generated texts and human-generated texts, annotated for metaphors in German language to compare the production of metaphors of LLMs with the production of metaphors of humans.

As part of the data collection process, ten German news articles were selected from the *politics* section from [MLSUM](https://huggingface.co/datasets/reciTAL/mlsum) (Scialom et al., 2020), containing between 350 and 450 tokens. Both ChatGPT-3.5 and ChatGPT-4o (`gpt-3.5-turbo-0125`and `gpt-4o-2024-05-13`) were prompted to generate similar articles using [OpenAI's API](https://platform.openai.com/), as follows: 

*Write a German online article in section '{section}' for the Süddeutsche Zeitung with the title '{title}' from the year {year} of exactly {numberofwords} words. Write the article in such a way that a short summary of the content could be written as follows: '{summary}'*


The metaphor annotation scheme is based on the MIPVU (Steen et al. 2010), its German adaptation by Hermann et al. (2019)  and Egg and Kordoni (2022). The annotated word classes are restricted to nouns, adjectives and verbs. More specifically, the following part-of-speech tags were selected: `NN`, `ADJA`, `ADJD`, `VVFIN`, `VVIMP`, `VVINF`, `VVIZU`, `VVPP`, `TRUNC` and `PTKVZ`. The  metaphor classification comprises the following categories:

- Indirect Metaphor-Related Word (MRW)
- Direct Metaphor-Related Word (MRW DIR)
- Complex Metaphor (KOMPL)
- Personification (PERS)
- When in Doubt, Leave it in (WIDLII)
- Discard for Metaphor Analysis (DFMA)
- No Metaphor (O)

As an extension to the MIPVU, the category `KOMPL` for complex metaphors is introduced, i.e. metaphors spanning over more than one word. Moreover, personifications are instantiated as a class of its own. Additionally, following Egg & Kordoni (2022), the annotation scheme also considers conventionality of metaphors based on whether both their contextual and basic meaning appear in the dictionaries or not (using *Duden online* and *Das Wortauskunftssystem zur deutschen Sprache in Geschichte und Gegenwart (DWDS)*). Furthermore, following the MIPVU, metaphor flags are annotated as well. In contrast to other annotations, metaphor flags are not restricted to the selected set of word classes.

## Recreate results

### Set up environment

Please clone this repository and run the following commands:

```
$ conda create -n env python=3.10
$ conda activate env
$ pip install requirements.txt
```

### Collect data

#### Step 1

Create the dataset by selecting human written articles
and collecting GPT data using `src/collect_data.py`.

#### Step 2

Evaluate the different prompts used for GPT to decide on final prompt using `src/evaluate_prompts.py`.

#### Step 3

Prepare data for upload to INCEpTION using `src/process_dataset.py`.

### Process annotations

Run the script `src/process_annotations.py` to process the annotated dataset and create the following two files:

- `data/my_final_dataset_metaphors.csv`
- `data/mflag_dataset.csv`

### Get qualitative annotation results

Run the script `src/metaphor_qualitative.py`. The graph was further processed using the tool [Gephi](https://gephi.org).

### Get quantitative anotation results

Run the script `src/metaphor_stats.py`.

## Repository Structure

### annotations/

Directory containing the annotated data.

### data/


### prompt_data/

### qualitative/

### src/

Directory containing all scripts.

#### collect_data.py

#### evaluate_prompts.py

#### full_typesystem.xml

#### metaphor_qualitative.py

#### metaphor_stats.py

#### process_annotations.py

#### process_dataset.py

### statistics/


## References

Egg, M., & Kordoni, V. (2022, June). Metaphor annotation for German. In Proceedings of the Thirteenth Language Resources and Evaluation Conference (pp. 2556-2562).

Herrmann, J. B. B., Woll, K., & Dorst, A. G. (2019). Linguistic metaphor identification in German. Metaphor identification in multiple languages: MIPVU around the world, Benjamins, Amsterdam, 113-135.

Scialom, T., Dray, P. A., Lamprier, S., Piwowarski, B., & Staiano, J. (2020). MLSUM: The multilingual summarization corpus. arXiv preprint arXiv:2004.14900.

Steen, G. J., Dorst, A. G., Herrmann, J. B., Kaal, A., Krennmayr, T., & Pasma, T. (2010). A method for linguistic metaphor identification: From MIP to MIPVU (Vol. 14). John Benjamins Publishing.