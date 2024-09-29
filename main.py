from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import os
import datetime
import spacy
import streamlit as st


# Load spaCy model
# nlp = en_core_web_sm.load()
nlp = spacy.load('en_core_web_sm')

# Fetch data from URL
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

# Generate announcement id mapping
announcement_id_map = {}
current_id = 1

# Function to get or create announcement id
def get_announcement_id(title, summary):
    global current_id
    key_terms = tuple(sorted(set(extract_key_terms(title + ' ' + summary))))
    if key_terms not in announcement_id_map:
        announcement_id_map[key_terms] = current_id
        current_id += 1
    return announcement_id_map[key_terms]

# List of Keywords that we want to match in news
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

# Function to fetch and process data
def fetch_and_process_data():
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
                    filtered_entries[category].append({
                        'Announcement ID': announcement_id,
                        'Time': time,
                        'System Time': now,
                        'Keywords': ', '.join(matched_categories),
                        'Title': title,
                        'Summary': summary,
                        'Link': link
                    })

        # Update dataframes for each category
        for category, entries in filtered_entries.items():
            # Create a DataFrame from the filtered entries
            new_df = pd.DataFrame(entries)

            # Define the CSV file path
            csv_file_path = f'{category}_news.csv'

            # Check if the file exists and is not empty
            if os.path.exists(csv_file_path):
                try:
                    existing_df = pd.read_csv(csv_file_path)
                    # Concatenate the existing DataFrame with the new DataFrame
                    combined_df = pd.concat([existing_df, new_df])
                    # Drop duplicates based on the 'Link' column
                    combined_df = combined_df.drop_duplicates(subset='Link')
                except pd.errors.EmptyDataError:
                    # If the file is empty, just use the new DataFrame
                    combined_df = new_df
            else:
                # If the CSV file doesn't exist, the combined DataFrame is just the new DataFrame
                combined_df = new_df

            # Save the updated DataFrame back to the CSV file
            combined_df.to_csv(csv_file_path, index=False)

# Streamlit interface
def display_data():
    st.title("News Categories Dashboard")
    
    for category in keywords.keys():
        csv_file_path = f'{category}_news.csv'
        if os.path.exists(csv_file_path):
            try:
                df = pd.read_csv(csv_file_path)
                st.subheader(f"{category.capitalize()} News")
                st.dataframe(df)
            except pd.errors.EmptyDataError:
                st.subheader(f"{category.capitalize()} News")
                st.write("No data available")

if __name__ == "__main__":
    # Fetch and process data before displaying Streamlit UI
    fetch_and_process_data()

    # Display the Streamlit interface
    display_data()
