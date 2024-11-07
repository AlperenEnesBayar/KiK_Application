# Set base image
ARG BASE_IMAGE=neuml/txtai-cpu
FROM $BASE_IMAGE

# Copy RAG application
COPY requirements.txt  .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

RUN \
    # Install Java (for Apache Tika)
    apt-get update && \
    apt-get -y --no-install-recommends install default-jre-headless && \
    rm -rf /var/lib/apt/lists && \
    apt-get -y autoremove && \
    \
    # Install base requirements
    python -m pip install -r requirements.txt

COPY rag.py Data zemberek-full.jar tika.log .

# Start streamlit application
ENTRYPOINT ["streamlit", "run", "rag.py"]
