import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load your dataset
df = pd.read_csv('sentiment140.csv', encoding='latin1', header=None)
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'review']

# Select only the necessary columns
df = df[['review', 'sentiment']]

df['sentiment'] = df['sentiment'].map({0: 'negative', 2: 'neutral', 4: 'positive'})

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Function to categorize sentiment
def categorize_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.01:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.:
        return 'negative'
    else:
        return 'neutral'


# Apply sentiment categorization
df['predicted_sentiment'] = df['sentiment'].apply(categorize_sentiment)

# Save the results
df.to_csv('sentiment_categorized.csv', index=False)

print(df.describe())