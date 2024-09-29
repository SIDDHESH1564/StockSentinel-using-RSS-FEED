# StockSentinel

This project provides a **Streamlit-based dashboard** that allows users to view categorized news based on selected **keywords** and interact with the news data using a **large language model** (LLM). It uses **RSS feeds** from several news sources, automatically filters news by key categories (such as mergers, investments, and partnerships), and offers the ability to ask questions about the news data through an LLM chat interface.

## Features

### 1. **Fetch Latest News from RSS Feeds**
   - The app collects data from various RSS feeds such as Moneycontrol, Economic Times, GlobeNewswire, and others.
   - It processes the news, extracting titles, summaries, and publication times for easy analysis.

### 2. **Categorize News Based on Keywords**
   - News articles are categorized under multiple labels like "merger," "investment," "partnership," and "other" based on pre-defined keywords.
   - The keywords are defined for each category and matched against the news articles' titles and summaries using regular expressions.

### 3. **Named Entity Recognition (NER) with spaCy**
   - The app uses **spaCy** to perform Named Entity Recognition (NER) on the news content, extracting key entities such as organizations, people, products, and locations.
   - It assigns a unique **announcement ID** based on these entities for further tracking and identification.

### 4. **Dynamic Data Storage**
   - The fetched and processed news data is saved into CSV files for each category.
   - The CSV files are updated dynamically with new articles while avoiding duplication.

### 5. **Interactive Chat with News Data**
   - Users can ask questions about the news data via a chat interface.
   - The app uses **LangChain** and **ChatGroq** with a prompt that allows the user to get AI-generated answers based on the latest news articles.

### 6. **Streamlit Web Interface**
   - The app features an intuitive and user-friendly **Streamlit** dashboard, where users can:
     - View the latest news articles categorized by the predefined topics.
     - Interact with the data by asking questions in natural language, and receive AI-generated responses.

## How It Works

### 1. **RSS Feed Fetching**
   The `get_url(i)` function pulls data from a pre-defined set of RSS feed URLs. The content is parsed using **BeautifulSoup** to extract the articles, which are then categorized based on their titles and summaries.

### 2. **News Categorization**
   The script uses a list of keywords defined for different categories like mergers, investments, and partnerships. It matches these keywords against each article using **regular expressions** to categorize them.

### 3. **NER with spaCy**
   Using **spaCy's Named Entity Recognition**, the code extracts key terms such as organizations and people from the news articles, ensuring that important entities are identified and tracked across articles.

### 4. **Storing Data**
   The categorized news is stored in CSV files (`category_news.csv`), and new articles are appended while avoiding duplicates. This allows for long-term tracking of categorized news.

### 5. **AI-Powered Chat**
   With **LangChain** and **ChatGroq**, users can ask questions about the current news. The AI uses the latest fetched news data and responds based on that information.

### 6. **Streamlit Interface**
   The **Streamlit** app displays categorized news data and provides an interactive chat where users can ask questions about the news and receive AI-driven answers.

## Setup

### Prerequisites
- Python 3.10+
- pip
- bs4
- requests
- re
- pandas
- os
- datetime
- spacy
- streamlit

### Libraries

This project uses the following Python libraries:
- `bs4` (BeautifulSoup)
- `requests`
- `pandas`
- `spacy`
- `streamlit`
- `re` (Regular Expressions)
- `langchain_groq` and `langchain_core.prompts`
- `dotenv` (for managing API keys)
