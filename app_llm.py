from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import os
import datetime
import spacy
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load environment variables for API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Load LLM
def load_llm():
    return ChatGroq(temperature=0.4, model_name="llama3-8b-8192", api_key=groq_api_key)

# Function to get RSS feed URLs
def get_url(i):
    urls = [
        'https://www.moneycontrol.com/rss/latestnews.xml',
        'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
        'https://www.tickertape.in/blog/feed/',
        'https://www.newswire.com/newsroom/rss/industries-industry-news',
        'https://www.globenewswire.com/RssFeed/subjectcode/27-Mergers%20and%20Acquisitions/feedTitle/GlobeNewswire%20-%20Mergers%20and%20Acquisitions'
    ]
    return requests.get(urls[i])

# Extract key terms using spaCy NER
def extract_key_terms(text):
    doc = nlp(text)
    key_terms = [ent.text for ent in doc.ents if ent.label_ in {'ORG', 'PERSON', 'GPE', 'PRODUCT'}]
    return key_terms

# Generate announcement ID mapping
announcement_id_map = {}
current_id = 1

# Function to get or create announcement ID
def get_announcement_id(title, summary):
    global current_id
    key_terms = tuple(sorted(set(extract_key_terms(title + ' ' + summary))))
    if key_terms not in announcement_id_map:
        announcement_id_map[key_terms] = current_id
        current_id += 1
    return announcement_id_map[key_terms]

# List of Keywords to match in news
keywords = {
    "merger": ["merger", "merger proposal", "reverse merger", "merge"],
    "investment": ["investment", "equity investment", "capital infusion", "equity stake", "equity transfer"],
    "partnership": ["partnership", "strategic partnership", "strategic alliance", "joint venture", "partnership deal"],
    "other": [
        "acquisition", "definitive agreement", "takeover", "takeover speculations", "nearing deal", 
        "consider sale", "proposal", "proposal to acquire", "non binding offer", "exploring sale", 
        "including sale", "exploring option", "in talks", "potential offer", "acquisition target", 
        "buyout", "buyback", "consolidation", "divestiture", "spin-off", "restructuring", 
        "stake acquisition", "share purchase", "management buyout", "leveraged buyout", 
        "private equity", "recapitalization", "asset purchase", "friendly takeover", "hostile takeover", 
        "asset sale", "take-private deal", "tender offer", "acquire", "acquires"
    ]
}

# Fetch and process data from RSS feeds
def fetch_and_process_data():
    all_results = []  # Collect all results here
    for i in range(5):
        url = get_url(i)
        soup = BeautifulSoup(url.content, 'xml')

        # Get all news of RSS Feed
        entries = soup.find_all('item')

        # Initialize a list to store the filtered entries
        filtered_entries = {category: [] for category in keywords}

        now = datetime.datetime.now()

        # Get Title, Summary, Link, Time of each News
        for entry in entries:
            title = entry.title.text
            summary = entry.description.text
            link = entry.link.text
            time = entry.pubDate.text

            matched_categories = []
            for category, keyword_list in keywords.items():
                if any(re.search(keyword, title.lower()) or re.search(keyword, summary.lower()) for keyword in keyword_list):
                    matched_categories.append(category)

            if matched_categories:
                announcement_id = get_announcement_id(title, summary)
                for category in matched_categories:
                    result = {
                        'Announcement ID': announcement_id,
                        'Time': time,
                        'System Time': now,
                        'Keywords': ', '.join(matched_categories),
                        'Title': title,
                        'Summary': summary,
                        'Link': link
                    }
                    filtered_entries[category].append(result)
                    all_results.append(result)

        # Update dataframes for each category (and save to CSV if needed)
        for category, entries in filtered_entries.items():
            new_df = pd.DataFrame(entries)
            csv_file_path = f'{category}_news.csv'

            if os.path.exists(csv_file_path):
                try:
                    existing_df = pd.read_csv(csv_file_path)
                    combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset='Link')
                except pd.errors.EmptyDataError:
                    combined_df = new_df
            else:
                combined_df = new_df

            combined_df.to_csv(csv_file_path, index=False)

    return all_results  # Return the latest results

# Create prompt for Chat with Results
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions based on current news data.",
        ),
        ("human", "{input}"),
        (
            "system",
            "Here is the current news data: {news_data}. You should provide answers based on this data.Don't give code. Dont generate \n\n like characters."
        ),
    ]
)

# Function for chatting with results
def chat_with_results(all_results, user_question):
    llm = load_llm()
    
    # Format the news data to pass to the LLM
    news_data = "\n".join([f"{i+1}. {res['Title']}: {res['Summary']}" for i, res in enumerate(all_results)])
    
    input_data = {
        "input": user_question,
        "news_data": news_data,
    }
    
    chain = prompt | llm
    response = chain.invoke(input_data)[0]
    print(response)
    
    return response

# Streamlit interface
def display_data_with_chat():
    st.title("StockSentinel")
    st.subheader("Real-time Stock News Analysis for Investment Insights")

    # Fetch and display categorized news
    all_results = fetch_and_process_data()
    
    # Display data in Streamlit
    for category in keywords.keys():
        st.subheader(f"{category.capitalize()} News")
        category_results = [res for res in all_results if category in res['Keywords']]
        df = pd.DataFrame(category_results)
        st.dataframe(df)

    # Chat with results
    st.subheader("Chat with the News Data")
    user_question = st.text_input("Ask a question based on the news data:")
    
    if user_question:
        response = chat_with_results(all_results, user_question)
        st.write(f"LLM Response: {response}")

# Run this function in your main block
if __name__ == "__main__":
    display_data_with_chat()
