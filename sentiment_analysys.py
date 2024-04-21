import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

# Add the SpacyTextBlob extension to the pipeline
nlp.add_pipe('spacytextblob')

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase and removing stopwords.
    """
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_stop]
    return " ".join(tokens)

def simple_sentiment_analysis(review):
    """
    Simple sentiment analysis based on spacytextblob library.
    """
    doc = nlp(review)
    polarity = doc._.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def process_chunk(chunk):
    """
    Process a chunk of data.
    """
    chunk.dropna(subset=['reviews.text'], inplace=True)
    chunk['cleaned_text'] = chunk['reviews.text'].apply(preprocess_text)
    chunk['sentiment'] = chunk['cleaned_text'].apply(simple_sentiment_analysis)
    return chunk

def main():
    filename = "amazon_product_reviews.csv"
    chunk_size = 10000
    print("Loading dataset...")
    try:
        chunks = pd.read_csv(filename, chunksize=chunk_size, low_memory=False)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Dataset file '{filename}' not found. Please check the file path.")
        return

    processed_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}")
        processed_chunk = process_chunk(chunk)
        processed_chunks.append(processed_chunk)

    # Concatenate processed chunks into a single DataFrame
    reviews_df = pd.concat(processed_chunks, ignore_index=True)

    # Test the sentiment analysis function on sample reviews
    sample_reviews = [
        "This product is amazing! I love it.",
        "The product quality is terrible. I'm very disappointed.",
        "Neutral review with no strong sentiment."
    ]
    print("Testing sentiment analysis on sample reviews:")
    for review in sample_reviews:
        sentiment = simple_sentiment_analysis(review)
        print(f"Review: {review} | Sentiment: {sentiment}")

    print("Data processing complete.")

if __name__ == "__main__":
    main()

    











