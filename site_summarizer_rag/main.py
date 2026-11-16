"""
Website Summarizer using RAG + LLM
----------------------------------
Type a site name (e.g. abc.com) and get a summary of that site.

Pipeline (RAG):
  1. Retrieve  -> crawl the site's pages and split the text into chunks
  2. Index     -> build a TF-IDF vector index of the chunks
  3. Search    -> find the chunks most relevant to the question
  4. Generate  -> send those chunks to an LLM (Claude) to write the summary
"""

import os
import re
import sys
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_PAGES = 5          # how many pages of the site to read
CHUNK_WORDS = 150      # words per chunk
CHUNK_OVERLAP = 30     # overlap between chunks
TOP_K = 6              # chunks passed to the LLM
HEADERS = {"User-Agent": "Mozilla/5.0 (SiteSummarizer RAG demo)"}


# ---------------------------------------------------------------- 1. RETRIEVE
def normalize_url(site: str) -> str:
    site = site.strip()
    if not site.startswith(("http://", "https://")):
        site = "https://" + site
    return site


def fetch_page(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        if "text/html" not in r.headers.get("Content-Type", ""):
            return None
        return r.text
    except requests.RequestException:
        return None


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "svg"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    meta = soup.find("meta", attrs={"name": "description"})
    desc = meta["content"].strip() if meta and meta.get("content") else ""
    body = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", f"{title}. {desc}. {body}")


def find_internal_links(html: str, base_url: str, limit: int):
    soup = BeautifulSoup(html, "html.parser")
    base_host = urlparse(base_url).netloc
    priority = ("about", "product", "service", "solution", "feature", "what", "company")
    links = []
    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a["href"]).split("#")[0]
        p = urlparse(full)
        if p.netloc == base_host and p.scheme in ("http", "https") and full not in links:
            links.append(full)
    links.sort(key=lambda u: 0 if any(k in u.lower() for k in priority) else 1)
    return links[:limit]


def crawl_site(site: str):
    start = normalize_url(site)
    home_html = fetch_page(start)
    if not home_html:
        return []
    pages = [(start, extract_text(home_html))]
    for link in find_internal_links(home_html, start, MAX_PAGES - 1):
        html = fetch_page(link)
        if html:
            pages.append((link, extract_text(html)))
    return pages


def chunk_text(text: str, source: str):
    words = text.split()
    step = CHUNK_WORDS - CHUNK_OVERLAP
    chunks = []
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i + CHUNK_WORDS])
        if len(piece.split()) > 20:
            chunks.append({"text": piece, "source": source})
    return chunks


# ------------------------------------------------------------------ 2 & 3. INDEX + SEARCH
class VectorIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def search(self, query: str, k: int = TOP_K):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        top = scores.argsort()[::-1][:k]
        return [self.chunks[i] for i in top]


# ---------------------------------------------------------------- 4. GENERATE
def ask_llm(question: str, context_chunks, site: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks)

    if not api_key:
        return ("(No ANTHROPIC_API_KEY found, showing the most relevant extracts instead.)\n\n"
                + "\n\n".join(f"- {c['text'][:400]}..." for c in context_chunks[:3]))

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        f"You are given extracts from the website {site}. "
        f"Using ONLY these extracts, answer the request below. "
        f"If the extracts don't contain the answer, say so.\n\n"
        f"<extracts>\n{context}\n</extracts>\n\n"
        f"Request: {question}"
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


SUMMARY_QUESTION = (
    "Give a clear summary of this website: what it is, who it is for, "
    "its main products/services/content, and any key details. "
    "Use a short paragraph followed by 3-5 bullet points."
)


def main():
    print("=" * 55)
    print("  Website Summarizer  (RAG + LLM)")
    print("  Type a site like  abc.com   |  type 'exit' to quit")
    print("=" * 55)

    while True:
        site = input("\nEnter site name: ").strip()
        if site.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            sys.exit(0)
        if not site:
            continue

        print("Reading the website...")
        pages = crawl_site(site)
        if not pages:
            print("Could not open that site. Check the name or your internet.")
            continue
        print(f"Fetched {len(pages)} page(s).")

        chunks = []
        for url, text in pages:
            chunks.extend(chunk_text(text, url))
        if not chunks:
            print("Not enough readable text on this site (it may be JavaScript-only).")
            continue

        index = VectorIndex(chunks)
        top_chunks = index.search("what is this website about, products, services, purpose")

        print("Generating summary...\n")
        print("SUMMARY of", site)
        print("-" * 55)
        print(ask_llm(SUMMARY_QUESTION, top_chunks, site))
        print("-" * 55)

        # Bonus: ask follow-up questions about the same site (RAG Q&A)
        while True:
            q = input("\nAsk a question about this site (or press Enter for a new site): ").strip()
            if not q:
                break
            print("\n" + ask_llm(q, index.search(q), site))


if __name__ == "__main__":
    main()
