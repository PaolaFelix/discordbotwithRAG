# Langchain RAG 

Install dependencies.

```python
pip install -r requirements.txt
```

Create the Chroma DB.

```python
python3 create_database.py
```

Query the Chroma DB.

```python
python3 query_data.py "What are tortillas?"
```

You'll also need to set up an OpenAI account (and set the OpenAI key in your environment variable) for this to work.
