import chromadb

client = chromadb.Client()

# 컬렉션 생성
collection = client.create_collection("test_collection")

# 더미 벡터 추가
collection.add(
    embeddings=[[0.1]*512],
    ids=["test_1"],
    metadatas=[{"video_id": "v1", "start_time": 0, "end_time": 5}]
)
print("벡터 추가 완료!")

# 유사도 검색
results = collection.query(
    query_embeddings=[[0.1]*512],
    n_results=1
)
print("검색 결과:", results)