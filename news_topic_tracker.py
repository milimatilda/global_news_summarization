import streamlit as st
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import os
import time
import requests
from llama_index.llms.ollama import Ollama
from keybert import KeyBERT
import spacy
from collections import Counter
from string import punctuation
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import yaml

from sklearn.cluster import KMeans
from codes import country_codes, summary_type_codes

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()
logging.info(".env loaded")




# Initialize chat history in session state if not already present
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load models for topic extraction and embedding
kw_model = KeyBERT()
embedding_model = SentenceTransformer("thenlper/gte-small")
nlp = spacy.load("en_core_web_sm")

# Load summarization prompt templates
with open("prompts.yaml", "r") as file:
    PROMPTS = yaml.safe_load(file)

# Initialize chat history in session state if not already present
if 'messages' not in st.session_state:
    st.session_state.messages = []



# =========================
# NEWS FETCHING FUNCTION
# =========================
    
def get_news_data(country, topic):  
    """
    Scrapes latest news from GNews API based on topic and country.
    """
    base_url = "https://gnews.io/api/v4/search"
    params = {
        "q": topic.replace(" ","%20"),
        "lang": "en",
        "country": country,
        "max": 10,
        "apikey": os.getenv("APIKEY")
    }
    
    news_results = []
    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()

        if "articles" not in data:
            st.error("No articles found. Try a different topic.")
            return []
        
       
        articles = data["articles"]
        # Parse and format each article
        for article in articles: 
            news_results.append(
                {
                    "url" : article['url'],
                    "title" : article['title'],
                    "description" : article['description'],
                    "title_description" : article['title'] + ": " + article['description'],
                    "published_at" : article['publishedAt'] 
                }
            )
                
    
    except Exception as e:
        logging.error(f"Error fetching news: {str(e)}")
        st.error("Failed to fetch news. Please try again later.")
        
    return news_results

# =========================
# KEY TOPIC EXTRACTION
# =========================
def get_keytopics(text):
    """
    Extracts important keyphrases using KeyBERT.
    """
    
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2))
    keyphrases = list(set(phrase for phrase, _ in keywords))

    return keyphrases 


# =========================
# NEWS CLUSTERING FUNCTION
# =========================
def cluster_news(news_data):
    """
    Clusters news articles using embeddings + UMAP + HDBSCAN.
    """
    all_key_topics = []
    # Attach extracted topics to each article
    for news in news_data:
        key_topics = " ".join(set(get_keytopics(news['title_description'])))
        news['key_topics'] = key_topics
        all_key_topics.append(key_topics)
    
    # Generate embeddings
    embeddings = embedding_model.encode(all_key_topics, show_progress_bar=True)
    logging.info("Embedding done")

    # Dimensionality reduction
    umap_model = umap.UMAP(n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    logging.info("Dimension reduced")

    # Clustering
    hdbscan_model = HDBSCAN(min_cluster_size=2, metric="euclidean", cluster_selection_method="eom").fit(reduced_embeddings)
    clusters = hdbscan_model.labels_
    
    # Assign cluster IDs to articles
    if len(clusters) != len(news_data):
        logging.warning("Mismatch in embeddings and dataframe length. Defaulting cluster to -1.")
        
    else:
        for index,news in enumerate(news_data):
            news['cluster'] = clusters[index]

    logging.info("Key topics and cluster id:")
    for item1, item2 in zip(all_key_topics, clusters):
        logging.info(f"{item1:<10} {item2}")
    return news_data

# =========================
# PROMPT CREATION
# =========================
def create_prompt(topic, summary_type):
    try:
        prompt_template = PROMPTS[summary_type]["system_prompt"]
        return prompt_template.format(topic=str(topic).capitalize())
    except KeyError:
        raise ValueError(f"Invalid summary type: {summary_type}")
    

# =========================
# LLM SUMMARIZER
# =========================
def summarizer(text, topic, llm_summary_type):
    """
    Uses LLaMA via Ollama to generate a summary.
    """
    llm_message = [
                ChatMessage(role="system", content=create_prompt(topic, llm_summary_type)),
                ChatMessage(role="user", content=text),
            ]
    try:
        # Initialize the language model with a timeout
        llm = Ollama(model="llama3", request_timeout=120.0)
        # Stream chat responses from the model
        resp = llm.stream_chat(llm_message)
        response = ""
        
        for r in resp:
            response += r.delta
            
        
        return response 
    except Exception as e:
        # Log and re-raise any errors that occur
        logging.error(f"Error during streaming: {str(e)}")
        raise e


# =========================
# SUMMARY GENERATION
# =========================
def generate_summaries(cluster_news_pair, topic):
    """
    Generates summaries for each news cluster.
    """
    summaries = []
    
    for cluster, news in cluster_news_pair.items():
        
        
        combined_news = "\n".join(news)
        if cluster!=-1:            
            summary = summarizer(combined_news, topic, "summary") 
            summaries.append(f"{cluster+1}. {summary}")
        else:            
            summary = summarizer(combined_news, topic, "outlier")
            summaries.append(f"On other news, {summary}")
    st.spinner("Writing...")
    logging.info("Response from LLM received")
    
    return summaries


# =========================
# MAIN STREAMLIT APP
# =========================
def main():
    st.title("Big News in Small Words.") # Set the title of the Streamlit app
    logging.info("App started") # Log that the app has started

    # Sidebar for country selection
    country = country_codes[st.sidebar.selectbox("Choose a Country", list(country_codes.keys()))]
    logging.info(f"Country selected: {country}")
    #Sidebar for summary type selection
    summary_type = summary_type_codes[st.sidebar.selectbox("Choose a summary type", list(summary_type_codes.keys()))]
    logging.info(f"Summary type: {summary_type}")

    user_input = st.chat_input("Type a news topic you want to explore…")

    if 'last_topic' not in st.session_state:
            st.session_state['last_topic'] = ""
    if 'last_country' not in st.session_state:
            st.session_state['last_country'] = country
    if 'last_summary_type' not in st.session_state:
            st.session_state['last_summary_type'] = summary_type
    if 'last_summaries' not in st.session_state:
        st.session_state['last_summaries'] = []
    if 'cluster_news_pair' not in st.session_state:
        st.session_state['cluster_news_pair'] = None

    # Response generation
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        start_time = time.time()

        # Fetch news and process if topic/country changed
        if any([(user_input and (user_input!=st.session_state['last_topic'])), (country!=st.session_state['last_country'])]):
            
            if (user_input and (user_input!=st.session_state['last_topic'])):
                st.session_state['last_topic'] = user_input
            
            if country!=st.session_state['last_country']:
                    st.session_state['last_country'] = country
            with st.spinner(f"Generating the news summary on {st.session_state['last_topic'].capitalize()}..."):
                
                # st.session_state['last_country'] = country
                st.session_state.messages.append({'role':'user', "content": st.session_state['last_topic'] })
                
                url_topic = st.session_state['last_topic'].replace(" ", "%20")
                news_data = get_news_data(st.session_state['last_country'], url_topic) 
                logging.info(f"News collection time: {(time.time() - start_time):.2f}")
                
            
            with st.spinner(f"Latest news on {st.session_state['last_topic'].capitalize()} collected..."):
                logging.info("News collected from GNews")
                clustered_news = cluster_news(news_data)
                logging.info(f"Clustering time: {(time.time() - start_time):.2f}")
            with st.spinner("Similar news are grouped together..."):
                logging.info(f"Clustering news done")
                cluster_news_pair = {}
                for news in clustered_news:
                    cluster = news['cluster']
                    news_description = news['title_description']
                    if cluster in cluster_news_pair:
                        cluster_news_pair[cluster].append(news_description)
                    else:
                        cluster_news_pair[cluster] = [news_description]
                logging.info(f"Cluster news pairing time: {(time.time() - start_time):.2f}")
                st.session_state['cluster_news_pair'] = cluster_news_pair
            logging.info("Generating response")
            with st.spinner("Generating summaries..."):
                st.session_state['last_summaries'] = generate_summaries(cluster_news_pair, st.session_state['last_topic'])
                logging.info(f"Summarizing time: {(time.time() - start_time):.2f}")
        

        # Final summarization
        if st.session_state['last_topic']!= "":
            with st.spinner("Here we go..."):
                st.session_state['last_summary_type'] = summary_type
                logging.info(f"Session Topic: {st.session_state['last_topic']}")    
                logging.info(f"Session country: {st.session_state['last_country']}")  
                logging.info(f"Session Summary type: {st.session_state['last_summary_type']}")   
                logging.info(f"Is last_summary available: {bool(st.session_state.get('last_summaries'))}")
                
         
                final_summary = summarizer(
                            text = "\n".join(st.session_state['last_summaries']), 
                            topic = st.session_state['last_topic'], 
                            llm_summary_type = st.session_state['last_summary_type']
                            )
                logging.info(f"Final summary time: {(time.time() - start_time):.2f}")
            try:    
                st.write(final_summary)
                duration = time.time() - start_time
                logging.info(f"Summaries displayed")
                st.write(f"Duration: {duration:.2f}")

                # Expandable section showing grouped news
                with st.expander("See grouped news"):
                    for cluster, items in st.session_state['cluster_news_pair'].items():
                        cluster_title = f"Cluster {cluster+1}" if cluster != -1 else "Other News"
                        st.markdown(f"**{cluster_title}**")
                        for i, item in enumerate(items, 1):
                            st.markdown(f"{i}. {item}")

            except Exception as e:
                # Handle errors and display an error message
                st.session_state.messages.append({"role": "assistant", "content": str(e)})
                st.error("An error occurred while generating the response.")
                logging.error(f"Error: {str(e)}")

         
   
if __name__ == "__main__":
    main()


    



