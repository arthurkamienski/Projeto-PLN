import os
import re
import nltk
import pandas as pd

# Sets csv file path
dirname = os.path.abspath(os.path.dirname(__file__))
csv_file = os.path.join(dirname, '..\\datasets\\bullying_twitter_preprocessing.csv')
print (csv_file)

# Loads csv dataset file into a dataframe
df = pd.read_csv(csv_file, ',')
print (df.head(10))

# Regex for username, links, and hashtags
patterns = [
    (r"\@\w+", "__USERNAME__"),
    (r"\bhttps?:(\/\/)?\w+\.\w+(\.\w+)*(\/\w+)*\/?", "__HTTP_LINK__"),
    (r"#", "")
]

# Tokenize words from each tweet
for index, rows in df.iterrows() :
    # Preprocessing of mentions, links and hashtags
    for pat, sub in patterns:
        rows['TEXTOS'] = re.sub(pat, sub, rows['TEXTOS'])
    rows['TEXTOS'] = nltk.word_tokenize(rows['TEXTOS'])

    # Prints last tokenized items
    if index > 1500 :
        print(rows['TEXTOS'])

