import pandas as pd
import random
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns

import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm 

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from scipy.special import softmax

plt.style.use('ggplot')

#source: 
#Dataset from Antoni Sobkowicz. (2017). Steam Review Dataset (2017) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1000885

#inspired by Rob Mulla: 
# https://www.youtube.com/watch?v=QpzMWQvxXWk 

n = sum(1 for line in open('steam-reviews.csv'))-1  # Calculate number of rows in file
s = n//1000 #sample size of 0.1% (abt. 6,000 reviews) for sake of simplicity

#random sample so all reviews are not from same game 
skip = sorted(random.sample(range(1, n+1), n-s))  # n+1 to compensate for header 

#reduces to 0.1% of og size with diff games being reviews
df = pd.read_csv('steam-reviews.csv', skiprows=skip)

#eda
df.insert(0, 'ID', range(0, len(df)))

df["review_score"] = (df["review_score"] == 1).astype(int) #make 1 = pos review 0 = negative review 
df.dropna(inplace = True)


sia = SentimentIntensityAnalyzer()

access_token = ''
model = f'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)
model = AutoModelForSequenceClassification.from_pretrained(model)

# Run Roberta Model

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    scores_dict = { 
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
    }
    return scores_dict

# print(example)
# print("VADER Model: ", sia.polarity_scores(example))
# #column 1 = neg, 2 = neu, 3 = pos
# print('Roberta Model: ', scores)

res = {}
for i, row in tqdm(df.iterrows(), total = len(df)):

    try: 
        review = row['review_text']
        id = row['ID']
        vader_result = sia.polarity_scores(review)
        vader_result_rename = {}

        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value

        roberta_result = polarity_scores_roberta(review)
        both = {**vader_result_rename, **roberta_result}
        res[id] = both

    except RuntimeError: 
        print(f"Excluded Review {id}, too large.")

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'Index': 'ID'})
results_df = results_df.merge(df, left_index=True, right_index=True)

print(results_df.columns)

sns.pairplot(data = results_df, vars=['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'roberta_neg', 'roberta_neu',
       'roberta_pos'], hue = "review_score", palette='tab10')

plt.show()

