# Reddit Sentiment Analysis

## Objective

The goal of this project is to perform sentiment analysis on Reddit posts from a specified subreddit. Sentiment analysis involves determining the emotional tone expressed in textual data, such as comments on Reddit posts.

### Subreddit Selection

You can choose any subreddit of your interest for sentiment analysis. In this example, we have used the "NFTMarketplace" subreddit.

## Data Acquisition

We utilize the PRAW (Python Reddit API Wrapper) library to interact with the Reddit API. PRAW allows us to extract post titles, URLs, top comments, and associated scores from various subreddits related to NFTs and cryptocurrency.

## Reddit API Credentials

To perform sentiment analysis on Reddit, you need to obtain API credentials. Follow these steps:

Create a Reddit Developer Account:

Visit the Reddit Apps page.
Log in or create a Reddit account.
Create a New App:

Scroll down to "Developed Applications."
Click on "Create App" or "Create Another App."
Configure Your App:

Choose a name, select "script" as the app type.
Provide a brief description.
Note Down Your Credentials:

After creating the app, note down client_id and client_secret.
Configure Your Python Script:

Use the obtained credentials in your Python script.
python
Copy code
import praw

# Configure your Reddit API credentials

reddit = praw.Reddit(
client_id='YOUR_CLIENT_ID',
client_secret='YOUR_CLIENT_SECRET',
user_agent='YOUR_USER_AGENT'
)
Now, you can perform sentiment analysis on Reddit data using the PRAW library or any other suitable tool.

# Data Collection

The dataset is collected by scraping the top posts from the chosen subreddit and storing the data in a CSV file (reddit_dataset.csv).

python
Copy code

# Define the subreddit you want to scrape

subreddit_name = 'NFTMarketplace' # Change to your desired subreddit
subreddit = reddit.subreddit(subreddit_name)
top_posts = subreddit.top(limit=None) # You can change the limit as needed

# Create a CSV file for storing the data

with open('reddit_dataset.csv', 'w', newline='', encoding='utf-8') as csv_file:
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Post Title', 'Post URL', 'Top Comment', 'Top Comment Score'])

    # Iterate through the top posts and get their top comment
    for post in top_posts:
        # Code to extract post details and top comment
        ...

print("Data has been saved to 'reddit_dataset.csv'")
Importing Libraries and Reading Data
Libraries like pandas, numpy, matplotlib, seaborn, nltk, emoji, string, and re are imported for data manipulation and analysis. The dataset obtained from Reddit is read into a DataFrame.

python
Copy code

# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import emoji
import string
import re

# Read the dataset into a DataFrame

df = pd.read_csv('reddit_dataset.csv')
print(df.shape)
Data Processing and Cleaning
The data cleaning process involves handling missing values, removing unwanted characters, and preparing the text data for sentiment analysis. NLTK library is used for text processing tasks such as removing emojis, punctuation, and stopwords, as well as tokenization.

python
Copy code

# Function to remove emojis from text

def remove_emojis(text):
...

# Function to remove specific text from a given string

def remove_text(text, text_to_remove):
...

# Function to remove punctuation, links, and HTML tags from text

def remove_puncs_and_link(text):
...

# Tokenization function

def tokenize_text(text):
...

# Stopword removal function

def remove_stopwords(text):
...

# Part-of-speech tagging function

def pos_tagging(tokens):
...

# Named Entity Recognition (NER) function

def named_entity_recognition(tagged_tokens):
...

# Text stemming function

def stem_text(text):
...

# Removing duplicates function

def remove_duplicates(text):
...

# Complete Text Preprocessing Function

def preprocess_data(text):
...
VADER Sentiment Scoring
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a pre-built sentiment analysis tool included in the NLTK library. We use NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text.

python
Copy code

# Import necessary libraries

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

# Download VADER lexicon from NLTK

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Run Sentiment Analysis

tqdm.pandas()
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
text = row['Top Comment']
myid = row['ID']
res[myid] = sia.polarity_scores(text)
Create DataFrame for Sentiment Scores
Create a DataFrame (vaders) with sentiment scores and merge it with the original dataset.

python
Copy code

# Create DataFrame for sentiment scores

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'ID'})
vaders = vaders.merge(df, how='left')
Results Visualization
Matplotlib and Seaborn are used to present the sentiment analysis results in various visualizations.

Compound Sentiment Scores Bar Plot
This plot visualizes the compound sentiment scores for each Reddit post.

python
Copy code
ax = sns.barplot(data=vaders, x='ID', y='compound')
ax.set_title('Compund Score by Reddit Post')
plt.show()
Positive, Neutral, Negative Sentiments Distribution
Visualizations provide insights into the sentiment distribution across the dataset.

python
Copy code
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='ID', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='ID', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='ID', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()
Additional Visualizations
Matplotlib and Seaborn are also used for other visualizations, including a correlation heatmap and pie chart for sentiment distribution.

Correlation Heatmap
This heatmap displays the correlation coefficients between negative, neutral, positive, and compound sentiment scores.

python
Copy code

# Correlation heatmap for sentiment scores

correlation_matrix = vaders[['neg', 'neu', 'pos', 'compound']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap of Sentiment Scores')
plt.show()
Sentiment Distribution Pie Chart
A pie chart summarizes the sentiment distribution across the dataset.

python
Copy code

# Categorize sentiments into Positive, Negative, and Neutral

def categorize_sentiment(compound_score):
if compound_score >= 0.05:
return 'Positive'
elif compound_score <= -0.05:
return 'Negative'
else:
return 'Neutral'

vaders['Sentiment'] = vaders['compound'].apply(categorize_sentiment)

# Create a pie chart

sentiment_counts = vaders['Sentiment'].value_counts()
labels = sentiment_counts.index
sizes = sentiment_counts.values
colors = ['lightgreen', 'coral', 'skyblue']
explode = (0.1, 0, 0) # Explode the 1st slice (Positive)

plt.figure(figsize=(4, 4))
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', shadow=True)
plt.title('Sentiment Distribution')
plt.axis('equal')
plt.show()

# Conclusion

This project demonstrates the process of collecting data from a Reddit subreddit, performing sentiment analysis, and visualizing the results. The visualizations provide valuable insights into the sentiment distribution across the dataset, aiding in understanding community sentiments.

**Feel free to explore the code in the Jupyter Notebook for more detailed information.**
