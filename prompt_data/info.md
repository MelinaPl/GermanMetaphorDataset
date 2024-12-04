# File Descriptions

## Final files 

### Unfiltered files containing all 600 articles

- `gpt-3.5.-turbo-0125_100n_170+words.csv`
- `gpt-4o-2024-05-13_100n.csv`

For ChatGPT-3.5, 170 words were added to the word count in the prompt while for ChatGPT-4o, the original word count of the article remained unchanged.

### Filtered files containing only the articles fulfilling the requirements

- `filtered_gpt-3.5-turbo-0125_170+words.csv`
- `filtered_gpt-4o-2024-05-13.csv`

The unfiltered files from above were filtered to meet the requirements. The requirements are as follows:

- All articles have to be in German
- All articles are only allowed to deviate no more than 50 words from the original text length.
- All articles should not contain the human-written summary

## Other experimental files (outdated)

- Files with the ending `preliminary` contain different articles than the final dataset.

This is because initially, I did not check whether the 100 original (human) articles contained their own summary (which they should not, but nevertheless occasionally did). After I controlled for that, the updated 100 human articles did not contain their own summary and I had to query the API again. These results no longer contain the suffix `preliminary`.

- Manual test files were the very first experiment files that I created manually using the demo in the browser.






