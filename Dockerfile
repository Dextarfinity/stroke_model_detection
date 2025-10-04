FROM python:3.11-slim

WORKDIR /app

# Install only absolutely essential dependencies
RUN apt-get update && apt-get install -y libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY main.py demo.html ./

EXPOSE 8000

CMD ["python", "main.py"]