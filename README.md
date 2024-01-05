# Reddit Sentiment Analysis

## Objective

The goal of this project is to perform sentiment analysis on Reddit posts from a specified subreddit. Sentiment analysis involves determining the emotional tone expressed in textual data, such as comments on Reddit posts.

## Subreddit Selection

You can choose any subreddit of your interest for sentiment analysis. In this example, we have used the "NFTMarketplace" subreddit.

## Sentiment Analysis

Sentiment analysis interprets and categorizes emotions (positive, negative) within textual data. It empowers organizations to discern public sentiment towards specific words or topics.

## Data Acquisition

We utilize the PRAW (Python Reddit API Wrapper) library to interact with the Reddit API. PRAW allows us to extract post titles, URLs, top comments, and associated scores from various subreddits related to NFTs and cryptocurrency.

### Reddit API Credentials

To perform sentiment analysis on Reddit, you need to obtain API credentials. Follow these steps:

1. **Create a Reddit Developer Account:**

   - Visit the [Reddit Apps](https://www.reddit.com/prefs/apps) page.
   - Log in or create a Reddit account.

2. **Create a New App:**
   - Scroll down to "Developed Applications."
   - Click on "Create App" or "Create Another App."
3. **Configure Your App:**

   - Choose a name, select "script" as the app type.
   - Provide a brief description.

4. **Note Down Your Credentials:**

   - After creating the app, note down `client_id` and `client_secret`.

5. **Configure Your Python Script:**
   - Use the obtained credentials in your Python script.

Now, you can perform sentiment analysis on Reddit data using the PRAW library or any other suitable tool.

## Data Collection

The dataset is collected by scraping the top posts from the chosen subreddit and storing the data in a CSV file (`reddit_dataset.csv`).

## Importing Libraries and Reading Data

Libraries like pandas, numpy, matplotlib, seaborn, nltk, emoji, string, and re are imported for data manipulation and analysis. The dataset obtained from Reddit is read into a DataFrame.

## Data Processing and Cleaning

The data cleaning process involves handling missing values, removing unwanted characters, and preparing the text data for sentiment analysis. NLTK library is used for text processing tasks such as removing emojis, punctuation, and stopwords, as well as tokenization.

## VADER Sentiment Scoring

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a pre-built sentiment analysis tool included in the NLTK library. We use NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text.

## Create DataFrame for Sentiment Scores

Create a DataFrame (`vaders`) with sentiment scores and merge it with the original dataset.

## Results Visualization

Matplotlib and Seaborn are used to present the sentiment analysis results in various visualizations.

### Compound Sentiment Scores Bar Plot

This plot visualizes the compound sentiment scores for each Reddit post.

### Positive, Neutral, Negative Sentiments Distribution

Visualizations provide insights into the sentiment distribution across the dataset.

## Additional Visualizations

Matplotlib and Seaborn are also used for other visualizations, including a correlation heatmap and pie chart for sentiment distribution.

### Correlation Heatmap

This heatmap displays the correlation coefficients between negative, neutral, positive, and compound sentiment scores.

### Sentiment Distribution Pie Chart

A pie chart summarizes the sentiment distribution across the dataset.

## Conclusion

This project demonstrates the process of collecting data from a Reddit subreddit, performing sentiment analysis, and visualizing the results. The visualizations provide valuable insights into the sentiment distribution across the dataset, aiding in understanding community sentiments.

**Feel free to explore the code in the Jupyter Notebook for more detailed information.**
