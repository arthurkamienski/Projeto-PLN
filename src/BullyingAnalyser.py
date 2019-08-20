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
print ("\n")

# Regex for username, links, and hashtags
patterns = [
    (r"\@\w+", "__USERNAME__"),
    (r"\bhttps?:(\/\/)?\w+\.\w+(\.\w+)*(\/\w+)*\/?", "__HTTP_LINK__"),
    (r"#", "")
]

ungrmdf = pd.DataFrame(columns=['TEXTOS', 'y'])
bigrmdf = pd.DataFrame(columns=['TEXTOS', 'y'])

# Tokenize words from each tweet
for index, rows in df.iterrows() :
    # Preprocessing of mentions, links and hashtags
    for pat, sub in patterns:
        rows['TEXTOS'] = re.sub(pat, sub, rows['TEXTOS'])
    rows['TEXTOS'] = nltk.word_tokenize(rows['TEXTOS'])

    ungrmdf = ungrmdf.append(rows, ignore_index=True)
    bigrmdf = bigrmdf.append({'TEXTOS' : list(nltk.bigrams(rows['TEXTOS'])), 'y' : rows['y']}, ignore_index=True)

    # Prints last tokenized items/grams
    if index > 1700 :
        print("Uni:")
        print(ungrmdf.loc[index])
        print("Bi:")
        print(bigrmdf.loc[index])

print ("\n")