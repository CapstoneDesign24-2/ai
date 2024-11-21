# Base image: Use Python 3.9 or compatible version
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install pybind11 before installing other dependencies
RUN pip install pybind11>=2.12

# Install flask-cors and other dependencies
RUN pip install flask-cors

# Install dependencies
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]