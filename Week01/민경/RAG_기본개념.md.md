## RAG 파이프라인

```python
사용자 질문 입력
    ↓
1. 문서 로드 (PDF, TXT, 웹페이지 등)
    ↓
2. 텍스트 분할 (Chunking)
    ↓
3. 임베딩 변환 (텍스트 → 벡터)
    ↓
4. 벡터 스토어 저장
    ↓
==================== 검색 단계 ====================
    ↓
5. 질문 임베딩 → 리트리버로 유사 문서 검색
    ↓
6. 검색된 문서 + 질문 → 프롬프트 구성
    ↓
7. LLM이 답변 생성
    ↓
8. 체인으로 전체 프로세스 연결
```

## 개념 정리

### **임베딩(Embedding)이란?**

텍스트를 숫자 배열(벡터)로 변환하는 것

```python
"강아지" → [0.2, 0.8, 0.1, 0.9, ...]
"개" → [0.3, 0.7, 0.2, 0.8, ...]  # 강아지와 비슷한 벡터
"자동차" → [0.9, 0.1, 0.8, 0.2, ...]  # 전혀 다른 벡터

```

의미가 비슷하면 벡터도 비슷하다

---

### 텍스트 분할

너무 크면 검색 불가, 너무 작으면 맥락 손실

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 한 조각의 크기
    chunk_overlap=200  # 문맥 보존을 위한 겹침 (이전 청크와 얼마나 겹칠지)
)
chunks = splitter.split_text(document)
```

문서가 너무 크면 LLM 컨텍스트 한계를 초과할 수 있기 때문에 적절한 크기로 나눠야 관련 부분만 검색할 수 있다. 

---

### **벡터 DB의 역할**

만들어진 벡터들을 저장하고, 빠르게 유사한 벡터를 찾아주는 데이터베이스

```python
# 문서 저장 단계
documents = [
    "Spring Boot는 Java 기반 프레임워크입니다",
    "JPA는 ORM 기술입니다",
    "React는 프론트엔드 라이브러리입니다"
]

# 각 문서를 임베딩으로 변환해서 벡터 DB에 저장
vectorDB.add(
    text="Spring Boot는 Java 기반 프레임워크입니다",
    embedding=[0.1, 0.8, 0.3, ...]  # 임베딩 모델이 생성
)

```

## RAG 구성 요소

**`RAG = Retrieval + Generation`**

Retrieval(검색)

- 관련 문서 찾기.
- 질문과 유사한 문서를 벡터 DB에서 검색
- **좋은 검색이 더 중요하다!**
    - 잘못된 문서를 찾으면 → LLM이 그럴듯한 거짓말을 만들어냄

Generation(생성)

- 찾은 문서를 기반으로 LLM이 답변 작성
- 환각(hallucination) 방지

## **검색 과정**

```python
# 1. 사용자 질문
query = "Java 백엔드 프레임워크 알려줘"

# 2. 질문도 똑같은 방식으로 임베딩
query_embedding = embedding_model.embed(query)
# → [0.2, 0.7, 0.4, ...]

# 3. 벡터 DB에서 가장 유사한 문서 검색
# 코사인 유사도, 유클리드 거리 등으로 계산
results = vectorDB.similarity_search(
    query_embedding,
    k=3  # 상위 3개 문서 반환
)

# 4. 결과
# "Spring Boot는 Java 기반 프레임워크입니다" ← 가장 유사
# "JPA는 ORM 기술입니다" ← 두번째
# "React는 프론트엔드 라이브러리입니다" ← 세번째

```

임베딩 모델이 학습할 때:

- "Java", "백엔드", "프레임워크", "Spring Boot" 같은 단어들이 자주 함께 나타남
- 그래서 이런 단어들의 벡터가 가까운 공간에 위치하게 됨
- 질문에 "Java 백엔드 프레임워크"가 들어있으면 자동으로 Spring Boot 관련 문서와 가까워짐

## 일반 DB vs 벡터 DB

```sql
-- 일반 DB: 정확히 일치하는 것만 찾음
SELECT * FROM documents
WHERE title = 'Spring Boot'  -- 'Spring' 또는 'Boot'는 못 찾음

-- 벡터 DB: 의미적으로 유사한 것을 찾음
vectorDB.search("스프링부트")
-- → "Spring Boot" 문서도 찾아냄! (의미가 같으니까)
```

| 구분 | 일반 DB | 벡터 DB |
| --- | --- | --- |
| 검색 방식 | 정확한 일치 | 의미적 유사도 |
| 저장 형태 | 텍스트, 숫자 | 벡터 (수치 배열) |
| 장점 | 정확한 검색 | 유연한 검색, 동의어/유사어 처리 |
| 단점 | 융통성 없음 | 계산 비용 높음 |