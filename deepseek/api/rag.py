import os
from glob import glob
from openai import OpenAI

# 从环境变量获取 DeepSeek API Key
api_key = os.getenv("DEEPSEEK_API_KEY")
api_key = "sk-b83549cba7314c989037edc883ef2fe3"
print(f"API Key: {'***' + api_key[-4:] if api_key else 'Not set'}")

if not api_key:
    print("Warning: DEEPSEEK_API_KEY environment variable is not set.")
    print("You can set it by running: $env:DEEPSEEK_API_KEY='your-api-key-here'")
    print("Continuing with demo without API calls...")
    deepseek_client = None
else:
    deepseek_client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",  # DeepSeek API 的基地址
    )

text_lines = []

for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()

    text_lines += file_text.split("# ")

print(f"Found {len(text_lines)} text segments")

# Using OpenAI client for embeddings instead of pymilvus DefaultEmbeddingFunction
# Note: DeepSeek doesn't have embedding API, so we'll use a simple alternative approach
# For demonstration, let's use a basic embedding approach
def simple_text_embedding(text, dim=384):
    """
    Create a simple embedding representation of text
    This is a placeholder - in real scenarios you'd want to use proper embedding models
    """
    import hashlib
    import numpy as np
    
    # Create a hash-based embedding for demo purposes
    text_hash = hashlib.md5(text.encode()).hexdigest()
    # Convert hash to numerical values
    hash_values = [ord(c) for c in text_hash]
    # Pad or truncate to desired dimension
    if len(hash_values) < dim:
        hash_values.extend([0] * (dim - len(hash_values)))
    else:
        hash_values = hash_values[:dim]
    
    # Normalize to create a proper embedding vector
    embedding = np.array(hash_values, dtype=np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()

# Test the embedding function
test_embedding = simple_text_embedding("This is a test")
embedding_dim = len(test_embedding)
print(f"Embedding dimension: {embedding_dim}")
print(f"First 10 values: {test_embedding[:10]}")

print("\nScript completed successfully without ONNX Runtime dependency!")

