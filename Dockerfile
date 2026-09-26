# Demo image for Hugging Face Spaces (Docker SDK). Serves the Streamlit app on port 7860.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/app/.venv/bin:$PATH \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed PYTHONUNBUFFERED=1 \
    LANGFUSE_HOST=https://cloud.langfuse.com
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN uv sync --no-dev --extra app
# Bake the embedding model into the image so the first question isn't slow.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY --chown=user config ./config
COPY --chown=user app ./app

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
