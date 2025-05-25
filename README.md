# Big News in Small Words

A Streamlit-based news summarization app that fetches the latest news on any topic by country, clusters similar news articles using advanced NLP and unsupervised learning, and generates concise summaries using a Large Language Model (LLaMA via Ollama).

---

## Features

- Fetches latest news from GNews API by topic and country.
- Extracts key topics from news articles using KeyBERT.
- Generates embeddings for news topics using SentenceTransformer.
- Clusters similar news articles using UMAP (dimensionality reduction) and HDBSCAN (clustering).
- Summarizes clusters and outlier news articles using LLaMA 3 model via Ollama API.
- Interactive Streamlit UI with country and summary type selection.
- Displays grouped news articles for transparency.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [Streamlit](https://streamlit.io/)
- Access to [GNews API](https://gnews.io/) (get an API key)
- Ollama environment setup for LLaMA 3 usage

### Installation

1. Clone the repository

```bash
git clone https://github.com/milimatilda/global_news_summarization.git
cd yourrepo
```
2.	Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3.	Install dependencies
```bash
pip install -r requirements.txt
```
5.  Create a .env file in the root directory with the following variables:
```bash
APIKEY=your_gnews_api_key
```
6.  Make sure you have Ollama installed and configured as per Ollama documentation.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### Running the app

Run the Streamlit app with:
```bash
streamlit run news_topic_tracker.py
```

### Usage
	•	Use the sidebar to select the country and summary type.
	•	Type your desired news topic in the chat input.
	•	The app fetches news, clusters similar articles, generates summaries, and displays them.
	•	Expand the “See grouped news” section to explore clustered articles.

⸻

### Code Structure
	•	news_topic_tracker.py - Main Streamlit app script.
	•	codes.py - Contains country_codes and summary_type_codes dictionaries.
	•	prompts.yaml - Contains prompt templates for LLM summarization.
	•	.env - Environment variables (not committed).

⸻

### Dependencies
	•	streamlit
	•	requests
	•	python-dotenv
	•	llama_index (ollama integration)
	•	keybert
	•	spacy
	•	sentence_transformers
	•	umap-learn
	•	hdbscan
	•	scikit-learn
	•	matplotlib
	•	pyyaml

### Acknowledgments
	•	GNews API for news data
	•	Ollama for LLaMA model hosting and API
	•	KeyBERT and Huggingface for NLP models
