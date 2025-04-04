import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Load models
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Step 1: Search Wikipedia titles
def wiki_search_titles(query, limit=10):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'format': 'json',
        'srlimit': limit
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get('query', {}).get('search', [])
        return [r.get("title", "") for r in results]
    return []

# Step 2: Get full article content
def get_full_wikipedia_text(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
        "redirects": 1
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        pages = response.json().get("query", {}).get("pages", {})
        for page_data in pages.values():
            return page_data.get("extract", "")
    return ""

# Step 3: Summarize each article and combine
def generate_combined_summary_from_article_summaries(articles, article_truncate_limit=3000):
    summaries = []
    for article in articles:
        truncated = article["content"][:article_truncate_limit]
        try:
            summary = summarizer(
                truncated,
                max_length=150,
                min_length=50,
                do_sample=False
            )[0]["summary_text"]
            summaries.append(summary)
        except Exception:
            summaries.append("(Summarization failed for one article)")

    combined_summary_input = " ".join(summaries)
    
    try:
        final_summary = summarizer(
            combined_summary_input,
            max_length=250,
            min_length=100,
            do_sample=False
        )[0]["summary_text"]
        return final_summary
    except Exception:
        return "Failed to generate combined summary."

# Full Wikipedia summarization pipeline
def full_search_pipeline(query, limit=10, top_k=5):
    titles = wiki_search_titles(query, limit=limit)

    articles = []
    for title in titles:
        content = get_full_wikipedia_text(title)
        if content:
            articles.append({"title": title, "content": content})

    if not articles:
        return {"summary": "(No content found)", "sources": []}

    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    for article in articles:
        article_embedding = embedding_model.encode(article["content"], convert_to_tensor=True)
        similarity_score = util.cos_sim(query_embedding, article_embedding).item()
        article["similarity"] = similarity_score

    ranked_articles = sorted(articles, key=lambda x: x["similarity"], reverse=True)[:top_k]
    final_summary = generate_combined_summary_from_article_summaries(ranked_articles)

    return {
        "summary": final_summary,
        "sources": ranked_articles
    }

# Main CSV processing function
def process_claims_csv(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    
    summaries = []
    for claim in df["claim"]:
        print(f"Processing claim: {claim[:60]}...")
        result = full_search_pipeline(claim)
        summaries.append(result["summary"])
    
    df["Additional_Info"] = summaries
    df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Saved output to: {output_csv_path}")

# Example usage
if __name__ == "__main__":
    process_claims_csv("input.csv", "claims_with_summaries.csv")

