#import necessary libraries

import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from subprocess import run

# Load NLP Models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
analyzer = SentimentIntensityAnalyzer()


# Attempt to load spaCy model, if not found, download it
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Model not found, downloading...")
    run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load a free Hugging Face model
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")


def extract_article_details(url):
    """Extracts Title, Summary, Sentiment, and Topics from a Yahoo Finance article."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title_tag = soup.find("title") or soup.find("h1")
        title = title_tag.get_text().strip() if title_tag else "No title found"

        paragraphs = soup.find_all("p")
        full_text = " ".join([p.get_text() for p in paragraphs if p.get_text().strip()])

        if not full_text:
            return None  # Skip empty articles

        summary = summarizer(full_text[:1024], max_length=75, min_length=10, do_sample=False)[0]['summary_text']
        sentiment_score = analyzer.polarity_scores(full_text)["compound"]
        sentiment = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"

        doc = nlp(full_text)
        topics = list(set([ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "EVENT"]]))

        return {
            "Title": title,
            "Summary": summary,
            "Sentiment": sentiment,
            "Topics": topics,
        }
    except requests.exceptions.RequestException:
        return None


def get_yahoo_news(company):
    """Fetches Yahoo Finance news article links."""
    base_url = f"https://finance.yahoo.com/quote/{company}/news?p={company}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        links = [
            f"https://finance.yahoo.com{a['href']}" if a['href'].startswith("/") else a['href']
            for a in soup.find_all("a", href=True)
            if "/news/" in a["href"] and "finance.yahoo.com" in a["href"] and re.search(r"\d{8}", a["href"])
        ]

        return list(set(links))[:10]  # Limit to 10 links
    except requests.exceptions.RequestException:
        return []


def sentiment_dist(articles):
    """Calculates sentiment distribution."""
    sentiments = [article["Sentiment"] for article in articles if article]
    sentiment_counts = dict(Counter(sentiments))

    return {
        "Positive": sentiment_counts.get("Positive", 0),
        "Negative": sentiment_counts.get("Negative", 0),
        "Neutral": sentiment_counts.get("Neutral", 0)
    }


def get_majority_sentiment(sentiment_distribution, company):
    """Determines the majority sentiment."""
    majority_sentiment = max(sentiment_distribution, key=sentiment_distribution.get)
    messages = {
        "Positive": f"{company} latest news coverage is mostly positive. Potential stock growth expected.",
        "Negative": f"{company} latest news coverage is mostly negative. Stock growth is not expected.",
        "Neutral": f"{company} latest news coverage is neutral. Cannot comment on stock growth!"
    }
    return messages[majority_sentiment]


def generate_impact(comparison_text):
    """Uses Mistral 7B to generate an insightful impact statement."""
    prompt = f"""Analyze the following news article comparison and generate an insightful impact statement:

    {comparison_text}

    Provide a brief analysis of how these differences might affect investors, market perception, or the company's reputation.
    """

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tokenize input and move to correct device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text with controlled randomness
    output = model.generate(
        **inputs, 
        max_new_tokens=150,  # Controls only the generated output length
        repetition_penalty=1.2  # Reduces repetitive outputs
    )
    
    # Decode and return impact statement
    impact_statement = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return impact_statement.strip()


def coverage_difference(articles):
    """Compares article summaries and analyzes coverage differences."""
    # Extract summaries
    summaries = [article["Summary"] for article in articles]

    # Compute TF-IDF similarity between summaries
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(summaries)
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

    coverage_differences = []

    # Compare each pair of articles
    for (idx1, idx2) in combinations(range(len(articles)), 2):
        sim_score = similarity_matrix[idx1][idx2]

        # If similarity is low, articles cover different topics
        if sim_score < 0.5:
            comparison = f"Article {idx1 + 1} focuses on {articles[idx1]['Title']}, while Article {idx2 + 1} discusses {articles[idx2]['Title']}."

            coverage_differences.append({
                "Comparison": comparison,
                "Impact": generate_impact(comparison)
            })
    return coverage_differences


def final_output(company_ticker):
    """Generates final news sentiment analysis."""
    news_links = get_yahoo_news(company_ticker)
    articles = [extract_article_details(url) for url in news_links if extract_article_details(url)]
    sentiment_scores = sentiment_dist(articles)
    difference = coverage_difference(articles)
    final_decision = get_majority_sentiment(sentiment_scores, company_ticker)

    return {
        "Company": company_ticker,
        "Articles": articles,
        "Coverage Difference": difference,
        "Comparative Sentiment Score": sentiment_scores,
        "Final Sentiment Analysis": final_decision
    }
