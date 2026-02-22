# 마사학 RAG

## 1. Supabase에서 어떤 형식으로 데이터를 가져올지

### 일기 데이터

```python
{
  "user_id": "123456789",
  "diary_id": "uuid",
  "date": "2026-02-22",
  "content": "오늘 미적분 극값 개념 공부했는데...",
  "language": "ko",        # 다국어 유저 대응용
  "diary_reply": {...}
}
```

### 투두 데이터

```python
{
  "user_id": "123456789",
  "todo_id": "uuid",
  "date": "2026-02-22",
  "content": "미적분 연습문제 3번",
  "is_completed": true
}질문
```

### 질문

- `language` 필드가 필요할지? 한다면 자동 감지?
- 투두에서 `is_completed: false`인 항목도 RAG에 넣을지

---

## 2. 답장 청크 처리

### 방식 1: 원본 일기에 합치기

**장점** — 일기와 답장 맥락이 함께 검색됨. 한 번에 풍부한 답변 생성 가능.

**단점** — 청크가 길어짐. 답장이 많으면 토큰 초과 위험.

---

### 방식 2: 메타데이터로 연결

```python
# 일기 청크
{
  "content": "...",
  "metadata": { "type": "diary", "id": "diary_001" }
}

# 답장 청크
{
  "content": "...",
  "metadata": { "type": "reply", "parent_diary_id": "diary_001" }
}
```

검색 결과에 답장이 걸리면 `parent_diary_id`로 원본 일기를 같이 가져와서 컨텍스트에 붙여주는 방식.

**→ 분리의 검색 정밀도 + 합치기의 맥락 보존을 둘 다 챙길 수 있음.**

---

## 3. 청킹 전략

### 방식 1: RecursiveCharacterTextSplitter (일반적인 방식)

**장점** — LangChain 기본 제공.

**단점** — 일기/답장 경계를 무시하고 문장 중간에서 잘릴 수 있음.

---

### 방식 2: 커스텀 (구조 기반)

일기 1개 = 청크 1개, 투두는 날짜+과목 단위로 묶기.

```python
	Document(
        page_content=diary["content"],
        metadata={
            "type": "diary",
            "discord_client_id": diary["discord_client_id"],
            "date": diary["date"],
            "language": diary["language"],
            "group_id": diary["group_id"]
        }
    )
```

**장점** — 의미 단위가 깨지지 않음.

**단점** — 일기가 너무 길면 추가 분할 로직 필요.

---

## 4. 임베딩 모델 선택

### OpenAI text-embedding-3-small

- 다국어 자체 지원 (한국어, 영어, 스페인어 등 혼재해도 의미 기반으로 매핑됨)
- 유료지만 비용 저렴한 편
- 관리 단순

### HuggingFace 한국어 특화 모델

- 한국어만 있으면 품질이 더 좋을 수 있음
- 다국어 환경에선 오히려 역효과 가능성 있음
- 무료지만 서버 직접 운영 필요

---

## 5. 벡터 저장소 선택

### Supabase pgvector

- https://supabase.com/docs/guides/database/extensions/pgvector
- 이미 Supabase를 쓰고 있어서 별도 벡터 DB 추가 불필요
- 인프라 관리 포인트 최소화
- 그룹 ID, 언어 등 메타데이터 필터링 쉬움

### Pinecone / Chroma 등 별도 벡터 DB

- 벡터 검색 성능 더 좋을 수 있음
- 별도 서비스 관리 필요