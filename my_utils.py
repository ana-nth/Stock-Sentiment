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
    # Step 1: Format the company name
    company = company.strip().replace(" ", "+")

    # Step 2: Create the URL for the car company
    URL = f"https://indianexpress.com/about/{company}/"

    # Step 3: Send a request
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(URL, headers=headers)

    # Step 4: Check if the response is successful
    if response.status_code != 200:
        print("Failed to retrieve the page.")
        return []

    # Step 5: Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Step 6: Extract headlines and links
    articles = soup.find_all("h3")

    # Step 7: Collect the first 10 links
    links = []
    for article in articles[:10]:  # Get the first 10 articles
        link_tag = article.find("a", href=True)
        if link_tag:
            link = link_tag["href"]
            # Make the link absolute if it's relative
            if not link.startswith("http"):
                link = "https://indianexpress.com" + link
            links.append(link)

    return links



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

def generate_audio(text, filename="output.mp3"):
    """Generate Hindi text-to-speech audio."""
    translator = Translator()
    translated_text = translator.translate(text, src="en", dest="hi").text
    tts = gTTS(text=text, lang="hi")
    tts.save(filename)
    return filename
