# Website Summarizer (RAG + LLM)

Enter a site name like `abc.com` and get a summary of the site.

## How it works (RAG)
1. **Retrieve**: crawls the home page + a few internal pages
2. **Index**: splits text into chunks and builds a TF-IDF vector index
3. **Search**: finds the chunks most relevant to the question
4. **Generate**: sends those chunks to an LLM (Claude) to write the summary

## Run in VS Code
1. Unzip the folder, then in VS Code: File > Open Folder > `site_summarizer_rag`
2. Open terminal (Ctrl + `) and run:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and paste your Anthropic API key
   (without a key, the program still runs and shows the most relevant extracts)
4. Run:
   ```
   python main.py
   ```
5. Type a site name, e.g. `python.org`
