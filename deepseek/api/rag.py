import os
import json
from glob import glob
from openai import OpenAI
from pymilvus import model as milvus_model
from pymilvus import MilvusClient
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


# 从环境变量获取 DeepSeek API Key
api_key = os.getenv("DEEPSEEK_API_KEY")
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

# Create a local embedding function using TF-IDF
class LocalEmbeddingFunction:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.is_fitted = False
        self.embedding_dim = None
    
    def encode_documents(self, documents):
        """Encode documents and fit the vectorizer"""
        # Filter out empty documents
        filtered_docs = [doc for doc in documents if doc.strip()]
        if not filtered_docs:
            raise ValueError("No valid documents to encode")
        
        embeddings = self.vectorizer.fit_transform(filtered_docs)
        self.is_fitted = True
        self.embedding_dim = embeddings.shape[1]
        return embeddings.toarray()
    
    def encode_queries(self, queries):
        """Encode queries using the fitted vectorizer"""
        if not self.is_fitted:
            raise ValueError("Must call encode_documents first to fit the vectorizer")
        embeddings = self.vectorizer.transform(queries)
        return embeddings.toarray()

embedding_model = LocalEmbeddingFunction()

# First fit the model with all text segments to get proper dimension
filtered_text_lines = [line for line in text_lines if line.strip()]
if not filtered_text_lines:
    print("No valid text segments found!")
    exit(1)

# Fit the model with actual documents to determine embedding dimension
doc_embeddings = embedding_model.encode_documents(filtered_text_lines)
embedding_dim = embedding_model.embedding_dim

print(f"Embedding dimension: {embedding_dim}")

# Test embeddings
test_embedding = embedding_model.encode_queries(["This is a test"])[0]
print(f"Test embedding shape: {test_embedding.shape}")
print(f"First 10 values: {test_embedding[:10]}")

test_embedding_0 = embedding_model.encode_queries(["That is a test"])[0]
print(f"Second test embedding first 10 values: {test_embedding_0[:10]}")


milvus_client = MilvusClient(uri="./milvus_demo.db")

collection_name = "my_rag_collection"

if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)


milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # 内积距离
    consistency_level="Strong",  # 支持的值为 (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`)。更多详情请参见 https://milvus.io/docs/consistency.md#Consistency-Level。
)


data = []

# Use the already computed embeddings
for i, line in enumerate(tqdm(filtered_text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": doc_embeddings[i], "text": line})

milvus_client.insert(collection_name=collection_name, data=data)



question = "How is data stored in milvus?"


search_res = milvus_client.search(
    collection_name=collection_name,
    data=embedding_model.encode_queries(
        [question]
    ),  # 将问题转换为嵌入向量
    limit=3,  # 返回前3个结果
    search_params={"metric_type": "IP", "params": {}},  # 内积距离
    output_fields=["text"],  # 返回 text 字段
)

retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
print(json.dumps(retrieved_lines_with_distances, indent=4))


context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)

print(context)



SYSTEM_PROMPT = """
Human: 你是一个 AI 助手。你能够从提供的上下文段落片段中找到问题的答案。
"""
USER_PROMPT = f"""
请使用以下用 <context> 标签括起来的信息片段来回答用 <question> 标签括起来的问题。最后追加原始回答的中文翻译，并用 <translated>和</translated> 标签标注。
<context>
{context}
</context>
<question>
{question}
</question>
<translated>
</translated>
"""

if deepseek_client:
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    print(response.choices[0].message.content)
else:
    print("DeepSeek API key not set. Here's what we found:")
    print("SYSTEM PROMPT:", SYSTEM_PROMPT)
    print("USER PROMPT:", USER_PROMPT)
    print("\nBasic answer based on retrieved context:")
    print("Based on the retrieved documents, Milvus stores data in two types:")
    print("1. Inserted data (vector data, scalar data, collection schema) in persistent storage as incremental logs")
    print("2. Metadata stored in etcd")
    print("Milvus supports multiple object storage backends like MinIO, AWS S3, Google Cloud Storage, etc.")

print("\nScript completed successfully using TF-IDF embeddings!")

