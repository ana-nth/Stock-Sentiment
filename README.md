# News Sentiment Analysis Web Application

This web application analyzes the sentiment of the latest news articles about a company and generates an audio summary based on the analysis. The system fetches relevant news articles, performs sentiment analysis, summarizes the articles, compares the coverage, and provides the final sentiment of the company. Additionally, it produces an audio summary of the sentiment analysis.


# Objective
Develop a web-based application that extracts key details from multiple news articles related
to a given company, performs sentiment analysis, conducts a comparative analysis, and
generates a text-to-speech (TTS) output in Hindi. The tool should allow users to input a
company name and receive a structured sentiment report along with an audio output.
# Requirements
1. News Extraction: Extract and display the title, summary, and other relevant
metadata from at least 10 unique news articles related to the given company.
Consider only non-JS weblinks that can be scraped using BeautifulSoup
(bs4).
2. Sentiment Analysis: Perform sentiment analysis on the article content (positive,
negative, neutral).
3. Comparative Analysis: Conduct a comparative sentiment analysis across the 10
articles to derive insights on how the company's news coverage varies.
4. Text-to-Speech: Convert the summarized content into Hindi speech using an
open-source TTS model.
5. User Interface: Provide a simple web-based interface using Streamlit or Gradio.
Users should input a company name (via dropdown or text input) to fetch news
articles and generate the sentiment report.
6. API Development: The communication between the frontend and backend must
happen via APIs.
7. Deployment: Deploy the application on Hugging Face Spaces for testing.
8. Documentation: Submit a detailed README file explaining implementation,
dependencies, and setup instructions.

## Project Setup

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.8 or higher
- pip (Python package installer)
- Gradio (for the front-end)
- Required libraries and dependencies as outlined in `requirements.txt`

### Steps to Install and Run

1. **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. **Install the required dependencies**:

    Use `pip` to install all the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:

    After setting up the environment and installing the dependencies, you can run the application with the following command:

    ```bash
    python app.py
    ```

4. **Access the application**:

    Once the application is running, it will be available at `http://localhost:7860` on your web browser.

---

## Model Details

### Models Used

This application employs several NLP models for tasks such as summarization, sentiment analysis, and generating the audio summary. The following models are used:

1. **Summarization Model**:
    - **Model**: `sshleifer/distilbart-cnn-12-6`
    - **Description**: A distilled version of BART for text summarization, which generates concise summaries of news articles. It processes the content to provide a shorter, more digestible form of the input text.

2. **Sentiment Analysis Model**:
    - **Model**: `vaderSentiment.VaderSentiment`
    - **Description**: This model is used to analyze the sentiment of the articles. It assigns a sentiment score (positive, negative, or neutral) based on the language used in the article.

3. **Language Model for Impact Generation**:
    - **Model**: `mistralai/Mistral-7B-Instruct-v0.1`
    - **Description**: This large language model is used to generate detailed impact statements from comparative article summaries. It analyzes news article differences and provides insights on potential market or investor impact.

4. **Text-to-Speech (TTS)**:
    - **Model**: Custom function (you can include a model like `pyttsx3` or any other TTS service for generating audio)
    - **Description**: The system generates an audio file based on the sentiment analysis summary.

### Hugging Face Integration

- The **Mistral-7B** model is hosted on Hugging Face and used directly in the backend. Hugging Face provides the infrastructure for running the model, and we interact with it through the `transformers` library.

---

## API Development

The application does not use external third-party APIs but is designed to interact with the models hosted on Hugging Face and Gradio for front-end functionalities. The communication happens directly between the `app.py` and `my_utils.py` modules.

### Endpoints

1. **Sentiment Analysis Endpoint**:
    - **Input**: A company name (stock ticker) from the user.
    - **Process**: Fetches the latest news articles about the company, performs sentiment analysis, generates a summary, and computes the sentiment distribution (positive, negative, neutral).
    - **Output**: A final sentiment message, which will indicate whether the market sentiment for the company is positive, negative, or neutral.

2. **Audio Generation Endpoint**:
    - **Input**: The sentiment analysis result.
    - **Process**: The text of the sentiment result is passed to a text-to-speech engine, which generates an audio file.
    - **Output**: An audio file corresponding to the sentiment analysis result.

### How to Test the Application with Postman or cURL

Although this application does not expose a RESTful API, it can be tested manually via the Gradio interface.

1. **Access the interface**:
    - Open the browser and navigate to `http://localhost:7860`.
    - Enter the company name in the text input field.
    - Click "Analyze Sentiment."

2. **Test the sentiment analysis**:
    - After entering a company name, the application will fetch the latest news, analyze sentiment, and display the result in the "Sentiment Analysis Result" box.
    - The corresponding audio file will be provided as well.

---

## Deployment

This web application has been deployed on [Hugging Face Spaces](https://huggingface.co/spaces/Anaaanthh/AnanthkrishnaDS) for easy access and demonstration. 

### Deployment Process

The application was deployed on Hugging Face Spaces using the following steps:

1. **Prepare the Environment**: Set up a `requirements.txt` file to include all necessary dependencies like Gradio, transformers, and sentiment analysis models.
2. **Push to Hugging Face**: Push the code to the Hugging Face repository by connecting it with your Hugging Face account.
3. **Run on Hugging Face Spaces**: The Hugging Face platform automatically handles the environment and allows easy deployment with minimal configuration.

---

## Assumptions & Limitations

### Assumptions:
- The company name entered is valid and corresponds to a publicly listed company (e.g., a valid Yahoo Finance ticker).
- The news articles fetched from Yahoo Finance are relevant and contain sufficient content for sentiment analysis and summarization.

### Limitations:
- **Web Scraping Limitations**: This application uses web scraping to fetch news articles. This can be subject to changes in the structure of Yahoo Finance, and scraping may break if the layout or structure of the page changes.
- **Model Performance**: While the models used for summarization and sentiment analysis are robust, the quality of output may vary based on the quality and clarity of the input news articles.
- **Sentiment Accuracy**: The sentiment analysis might not always be perfectly accurate, especially for ambiguous or sarcastic content.
- **No Real-time Stock Data**: The application does not integrate real-time stock price or financial data, as it solely focuses on sentiment analysis of news articles.
  
---

## Conclusion

This web application provides an end-to-end solution for sentiment analysis of news articles related to a company. It uses advanced NLP techniques to summarize, analyze, and generate audio-based insights, helping users gauge market sentiment and make informed decisions.

For further development, we may add more features like integrating real-time stock price data, enhancing sentiment analysis with additional models, or incorporating more external news sources.


