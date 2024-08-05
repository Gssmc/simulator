FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8080

# Command to run the application
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.enableCORS=false"]
