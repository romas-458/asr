# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get install -y build-essential

RUN pip install --no-cache-dir packages
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r pre-requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn transformers soundfile

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
