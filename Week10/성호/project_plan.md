# 내 데이터 RAG 챗봇 프로젝트

## 1. 프로젝트 개요

### 무엇을 만드는가

Supabase에서 특정 사용자의 **일기 / 투두 데이터**를 가져와서,
"요즘 나 뭐했지?", "투두 달성률 어때?", "지난주 일기 요약해줘" 같은 질문에
**실제 데이터 근거로 답변하는 Streamlit 웹 챗봇**

### 왜 이렇게 하는가

- 단순 LLM에게 물어보면 내 데이터를 모름
  → RAG로 **내 일기/투두를 검색해서 컨텍스트로 넘겨야** 실제 근거 있는 답변 가능
- 11주간 배운 RAG 파이프라인(문서로드 → 분할 → 임베딩 → 벡터스토어 → 리트리버 → 프롬프트 → LLM)을
  **처음부터 끝까지** 직접 구현해보는 경험

### 개인별 맞춤 RAG

이 프로젝트는 **사용자별 전용 인덱스**를 만드는 방식이다.
질문하면 **내 데이터에서만** 검색해서 답변한다. 다른 사람 데이터는 절대 섞이지 않는다.

```
Streamlit에서 user_id 선택/입력
    ↓
loader.py: WHERE user_id = :user_id 로 해당 사용자 데이터만 가져옴
    ↓
indexer.py: 그 사용자 전용 FAISS 인덱스 생성 (faiss_index/{user_id}/)
    ↓
rag_chain.py: 해당 인덱스에서만 검색 → 답변 생성
```

**왜 이 방식을 선택했는가:**

| 방식 | 장점 | 단점 |
|------|------|------|
| **사용자별 인덱스 분리 (채택)** | 다른 사람 데이터 절대 안 섞임. 구현 단순 | 사용자마다 FAISS 파일이 따로 생김 |
| 전체 인덱스 + metadata 필터 | 인덱스 1개로 관리 | 검색 시 필터링 필요. 실수로 다른 사람 데이터 노출 가능성 |

2주 프로젝트에서는 **사용자별 분리**가 가장 단순하고 확실하다.
사용자 수가 많아지면(수백~수천명) 전체 인덱스 + 필터 방식을 고려해야 하지만,
프로토타입에서는 분리 방식이 적합하다.

---

## 2. 공부한 기술 → 프로젝트 적용 맵

```
[Supabase SQL]
    ↓
커스텀 DocumentLoader ─────────────── W6 문서로더
    ↓
RecursiveCharacterTextSplitter ────── W6 텍스트분할 (긴 일기만)
    ↓
OpenAI text-embedding-3-small ─────── W7 임베딩
    ↓
FAISS ─────────────────────────────── W8 벡터스토어
    ↓
EnsembleRetriever ─────────────────── W9 리트리버
  BM25(Kiwi 형태소) + FAISS
    ↓
ChatPromptTemplate ────────────────── W3 프롬프트
  + MessagesPlaceholder                W5 메모리 (대화 히스토리)
    ↓
ChatOpenAI gpt-4.1-mini ──────────── W5 모델
    ↓
StrOutputParser ───────────────────── W4 출력파서
    ↓
Streamlit 채팅 UI
```

### 기술별 선택 이유

| 공부한 것 | 프로젝트 적용 | 왜 이것을 선택했는가 |
|-----------|-------------|---------------------|
| **커스텀 DocumentLoader** (W6) | Supabase SQL → Document 변환 | W6에서 배운 `Document(page_content, metadata)` 구조를 그대로 활용. `load()` 인터페이스를 맞추면 LangChain 파이프라인에 바로 연결 |
| **RecursiveCharacterTextSplitter** (W6) | 긴 일기(500자+)만 분할 | `["\n\n", "\n", " ", ""]` 순서로 자연스럽게 분할. 대부분 일기는 짧아서 분할 불필요하지만, 긴 일기는 의미 단위로 쪼개야 검색 정확도 상승 |
| **OpenAI text-embedding-3-small** (W7) | 일기/투두 벡터화 | 가성비 최고 (62,500 pages/$1). 프로토타입에 적합 |
| **FAISS** (W8) | 벡터 저장/검색 | `save_local()`/`load_local()`로 인덱스 파일 관리 간편 |
| **EnsembleRetriever** (W9) | BM25(Kiwi) + FAISS | 한국어에서 BM25+형태소 분석이 키워드 매칭에 강함. 의미 검색은 FAISS가 우위. 둘을 결합해야 최적 |
| **Kiwi 형태소 분석기** (W9) | BM25 한국어 토크나이저 | 한국어는 공백 기반 토큰화가 안 됨. Kiwi가 속도와 정확도 모두 우수 |
| **ChatPromptTemplate + MessagesPlaceholder** (W3, W5) | 시스템 프롬프트 + 대화 히스토리 | 역할 부여 + 이전 대화 맥락 유지로 자연스러운 멀티턴 대화 가능 |
| **gpt-4.1-mini** (W5) | 답변 생성 | 빠르고 저렴하면서 충분한 품질 |

---

## 3. Supabase 데이터 가져오기 설계

### 3-1. 일기 데이터 (diaries + users)

```sql
SELECT
    d.diary_id,
    d.user_id,
    d.guild_id,
    d.diary_date,
    d.content,
    u.nickname
FROM diaries d
JOIN users u ON d.user_id = u.user_id AND d.guild_id = u.guild_id
WHERE d.user_id = :user_id AND d.guild_id = :guild_id
ORDER BY d.diary_date DESC;
```

**Document 변환:**

```python
Document(
    page_content=f"[{d.diary_date} 일기] {d.content}",
    metadata={
        "source": "diary",
        "user_id": d.user_id,
        "guild_id": d.guild_id,
        "date": str(d.diary_date),
        "nickname": u.nickname,
    }
)
```

- `date`를 metadata에 넣으면 → "지난주 일기", "2월 일기" 같은 날짜 기반 검색 가능
- `page_content`에 날짜를 포함하면 → BM25에서도 날짜 키워드 매칭 가능

### 3-2. 투두 데이터 (todos, 날짜별 묶음)

```sql
SELECT
    t.user_id,
    t.guild_id,
    t.todo_date,
    t.content,
    t.is_completed,
    t.sort_order
FROM todos t
WHERE t.user_id = :user_id AND t.guild_id = :guild_id
ORDER BY t.todo_date DESC, t.sort_order;
```

**Document 변환 (같은 날짜를 하나로 묶음):**

```python
Document(
    page_content=(
        "[2026-02-20 투두리스트]\n"
        "[완료] 알고리즘 3문제 풀기\n"
        "[미완료] RAG 프로젝트 설계\n"
        "[완료] 운동 30분"
    ),
    metadata={
        "source": "todo",
        "user_id": user_id,
        "guild_id": guild_id,
        "date": "2026-02-20",
        "total": 3,
        "completed": 2,
        "rate": 0.67,
    }
)
```

- 투두 하나하나는 너무 짧음("운동 30분") → 임베딩 품질 떨어짐
- 날짜별로 묶으면 "그 날 뭘 했는지" 전체 맥락이 하나의 벡터에 담김
- `rate`를 metadata에 넣으면 → "달성률 낮은 날" 같은 필터 가능

---

## 4. RAG 파이프라인 상세 설계

### 4-1. 인덱싱 파이프라인 (1회 실행)

```python
# 1. 데이터 로드
documents = supabase_loader.load()    # 일기 + 투두 Document 리스트

# 2. 텍스트 분할 (긴 일기만)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# 3. 임베딩
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 4. FAISS 저장
vectorstore = FAISS.from_documents(split_docs, embedding)
vectorstore.save_local("./faiss_index")
```

### 4-2. 검색 파이프라인

```python
# BM25 리트리버 (Kiwi 한국어 형태소)
kiwi = Kiwi()
def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]

bm25 = BM25Retriever.from_documents(split_docs, preprocess_func=kiwi_tokenize)
bm25.k = 3

# FAISS 리트리버
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 앙상블 (BM25 70% + FAISS 30%)
ensemble = EnsembleRetriever(
    retrievers=[bm25, faiss_retriever],
    weights=[0.7, 0.3],
)
```

### 4-3. 생성 파이프라인

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 사용자의 개인 데이터를 기반으로 답변하는 AI 비서야.
아래 검색된 데이터를 참고해서 답변해. 데이터에 없는 내용은 추측하지 마.

[검색된 데이터]
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

chain = (
    {
        "context": ensemble | format_docs,
        "question": RunnablePassthrough(),
        "chat_history": lambda x: chat_history,
    }
    | prompt
    | ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
    | StrOutputParser()
)
```

---

## 5. 예상 질문 & 답변 예시

| 질문 | 검색되는 데이터 | 답변 예시 |
|------|---------------|----------|
| "지난주 뭐했어?" | 지난주 일기 + 투두 | "지난주에 알고리즘 스터디 2번, 운동 3번 하셨고 투두 달성률은 평균 73%였습니다." |
| "요즘 투두 달성률 어때?" | 최근 2주 투두 | "최근 2주 평균 달성률이 65%예요. 특히 월요일 투두 완료율이 낮은 편이에요." |
| "내가 제일 자주 쓰는 투두는?" | 전체 투두 | "반복 투두 기준으로 '운동', '알고리즘 풀기'가 가장 많이 등록되었어요." |
| "지난달 일기 요약해줘" | 지난달 일기 | "1월에는 주로 프로젝트 스트레스, 스터디 준비, 운동 루틴에 대해 쓰셨어요..." |

---

## 6. 파일 구조

```
rag-project/
├── app.py              # Streamlit 메인 (채팅 UI, user_id 선택)
├── loader.py           # Supabase → Document 변환 로더
├── indexer.py          # 임베딩 + FAISS 인덱스 생성 (사용자별)
├── rag_chain.py        # 리트리버 + 프롬프트 + LLM 체인
├── faiss_index/        # 사용자별 인덱스 저장 디렉토리
│   ├── {user_id_A}/    #   사용자 A 전용 인덱스
│   └── {user_id_B}/    #   사용자 B 전용 인덱스
├── .env                # OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY
└── requirements.txt    # langchain, faiss-cpu, streamlit, kiwipiepy, supabase 등
```

---

## 7. 2주 일정 (2명, 둘 다 RAG 담당)

### 팀 구성

- **2명** 모두 RAG를 공부하는 입장 → 둘 다 RAG 파이프라인에 직접 관여
- Streamlit UI는 AI로 빠르게 만들어서 시간 절약 → **RAG + 실험에 집중**

### 역할 분담

| | A (데이터 → 검색) | B (검색 → 생성) |
|--|---|---|
| **담당 파일** | `loader.py` + `indexer.py` | `rag_chain.py` |
| **RAG 경험 범위** | Document 변환, 임베딩, FAISS 인덱싱, BM25+Kiwi 구성 | EnsembleRetriever 조합, 프롬프트 설계, LLM 체인, 메모리 |
| **관련 스터디 범위** | W6(문서로더, 텍스트분할), W7(임베딩), W8(벡터스토어) | W3(프롬프트), W5(메모리), W9(리트리버) |
| **실험 담당** | 실험 2: Kiwi 유무 (BM25 담당이니까) | 실험 1: 앙상블 비율 (Retriever 담당이니까) |
| **Streamlit** | 같이 AI로 빠르게 | 같이 AI로 빠르게 |

**핵심:** 둘 다 RAG 파이프라인의 서로 다른 절반을 맡아서, 합치면 전체가 완성되는 구조.

### 일정

| 기간 | A (데이터 → 검색) | B (검색 → 생성) |
|------|---|---|
| **Day 1-3** | Supabase → Document 로더 + FAISS + BM25(Kiwi) | 프롬프트 설계 + LLM 체인 구성 |
| **Day 4** | **합치기 (end-to-end 연결 테스트)** | **합치기** |
| **Day 5-7** | 실험: Kiwi 유무 비교 | 실험: 앙상블 비율 비교 |
| **Day 8-9** | 같이 프롬프트 튜닝 + 추가 실험 (k값, chunk_size 등) | 같이 프롬프트 튜닝 + 추가 실험 |
| **Day 10** | Streamlit (AI로 빠르게) | Streamlit (AI로 빠르게) |
| **Day 11-14** | 코드 정리 + 발표 준비 | 코드 정리 + 발표 준비 |

> **Day 4가 핵심이다.** 여기서 합쳐서 end-to-end로 질문→답변이 돌아가야 이후에 여유가 생긴다.
>
> Streamlit을 AI로 빠르게 처리하면 **Day 8-9에 실험을 3~4개까지 늘릴 여유**가 생긴다.

---

## 8. RAG 하이퍼파라미터 정리 (토의용)

RAG 파이프라인의 각 단계에는 **성능에 직접 영향을 주는 수치(하이퍼파라미터)**가 있다.
같은 데이터, 같은 질문이라도 이 수치를 바꾸면 답변 품질이 크게 달라진다.

### 8-1. 텍스트 분할 수치 (W6)

#### `chunk_size` — 청크 하나의 최대 크기

```python
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
```

| chunk_size | 장점 | 단점 |
|-----------|------|------|
| 작게 (200) | 검색 정확도 높음 (핀포인트) | 맥락이 잘림, 답변에 충분한 정보가 안 담길 수 있음 |
| 크게 (1000) | 맥락이 풍부함 | 불필요한 정보도 포함, 검색 정확도 저하 |
| **300~500 (권장)** | 정확도와 맥락의 균형 | - |

#### `chunk_overlap` — 청크 간 겹치는 부분

- 겹침이 크면 → 맥락 연결이 좋지만, 인덱스 크기 증가 + 중복 검색 가능성
- 겹침이 작으면 → 저장 효율적이지만, 청크 경계에서 맥락이 끊김
- 일반적으로 chunk_size의 **10~20%** 권장 (500이면 50~100)

#### 우리 프로젝트에서의 고민

일기는 보통 짧음 (100~300자). 이 경우:
- **분할 안 하는 게 나을 수 있음** → 일기 1건 = Document 1개
- 긴 일기(500자+)만 선별적으로 분할하는 전략이 필요
- 투두는 날짜별 묶음이라 길이가 들쭉날쭉 → 분할 기준을 어디에 둘지?

### 8-2. 임베딩 모델 & 차원 수 (W7)

| 모델 | 차원 수 | 가격 (pages/$1) | MTEB 성능 |
|------|--------|----------------|-----------|
| text-embedding-3-small | 1536 | 62,500 | 62.3% |
| text-embedding-3-large | 3072 | 9,615 | 64.6% |
| text-embedding-ada-002 | 1536 | 12,500 | 61.0% |
| BAAI/bge-m3 (로컬) | 1024 | 무료 | 다국어 강점 |

- 프로토타입에서는 **small(1536)**로 충분
- 한국어 일기에 OpenAI 임베딩이 충분한가? BGE-M3가 더 나을 수 있지만 로컬 GPU 필요

### 8-3. 리트리버 수치들 (W9) — 가장 중요

#### `k` — 검색 결과 개수

| k 값 | 장점 | 단점 |
|------|------|------|
| 작게 (1~2) | 가장 관련 높은 것만, 토큰 절약 | 정보 부족할 수 있음 |
| 중간 (3~5) | 균형 잡힌 컨텍스트 | - |
| 크게 (7~10) | 풍부한 정보 | 노이즈 증가, 토큰 비용 증가, **Lost in the Middle** 문제 |

> **Lost in the Middle** (W9): LLM은 긴 컨텍스트의 **처음과 끝은 잘 보지만, 중간은 놓치는** 경향이 있음.

#### `weights` — 앙상블 가중치 (EnsembleRetriever)

```python
EnsembleRetriever(
    retrievers=[bm25, faiss],
    weights=[0.7, 0.3],   # ← 이 비율이 성능을 좌우
)
```

| BM25 : FAISS | 특성 |
|-------------|------|
| 9:1 | 키워드 매칭 위주. "알고리즘" 검색 시 정확히 "알고리즘" 들어간 문서만 |
| **7:3 (한국어 권장)** | 키워드 + 약간의 의미 검색. 한국어에서 가장 안정적 |
| 5:5 | 균형. 키워드와 의미 검색 반반 |
| 3:7 | 의미 검색 위주. "피곤하다" 검색 시 "지쳤다", "힘들었다"도 나옴 |

#### `lambda_mult` — MMR 다양성 조절

| lambda_mult | 동작 |
|------------|------|
| 0.0 | 최대 다양성 (비슷한 문서 배제) |
| **0.25~0.5 (권장)** | 유사도와 다양성의 균형 |
| 1.0 | 순수 유사도 (다양성 무시) |

#### `score_threshold` — 유사도 임계값

| score_threshold | 동작 |
|----------------|------|
| 높게 (0.9) | 매우 관련 높은 문서만. 결과가 0개일 수 있음 |
| **0.7~0.8 (권장)** | 적당한 관련성 필터 |
| 낮게 (0.5) | 느슨한 필터. 관련 약한 문서도 포함 |

#### `decay_rate` — 시간 가중치 (TimeWeightedRetriever)

```python
# 최종 점수 = 의미유사도 + (1.0 - decay_rate) ^ 경과시간(hours)
```

| decay_rate | 동작 |
|-----------|------|
| 0에 가까움 | 시간 거의 무관, 유사도만으로 검색 |
| **0.01 (권장)** | 최근 2주 내 데이터에 약간의 가중치 |
| 0.999 | 거의 최근 데이터만 나옴 |

### 8-4. LLM 파라미터 (W5)

| temperature | 용도 |
|------------|------|
| 0.0 | 사실 기반 질의응답. 같은 질문에 항상 같은 답 |
| **0.0~0.3 (우리 프로젝트)** | 데이터 기반 답변이므로 낮게 설정 |
| 0.7~1.0 | 창의적 글쓰기, 다양한 표현 |

---

## 9. 실험 계획

### 필수 실험 (Day 5~7, 각자 1개씩)

아래 2개는 **반드시** 진행. 가장 결과 차이가 눈에 잘 보이고, 토의하기 좋다.

### 실험 1: 앙상블 가중치 비교

**왜 이 실험을 하는가:** 같은 EnsembleRetriever인데 비율만 바꿔도 검색 결과가 완전히 달라진다.
키워드 질문과 의미 질문에서 어떤 비율이 최적인지 우리 데이터로 직접 확인.

| 설정 | 키워드 질문 ("알고리즘 스터디") | 의미 질문 ("요즘 힘든 일 있었어?") |
|------|-------------------------------|--------------------------------|
| BM25:FAISS = **9:1** | ? | ? |
| BM25:FAISS = **7:3** | ? | ? |
| BM25:FAISS = **5:5** | ? | ? |
| BM25:FAISS = **3:7** | ? | ? |

**관찰 포인트:**
- 키워드 질문: BM25 비중이 높을수록 정확한가?
- 의미 질문: FAISS 비중이 높을수록 유사 문서를 잘 찾는가?
- 한국어 일기에서 최적 비율은?

### 실험 2: Kiwi 형태소 분석 유무

**왜 이 실험을 하는가:** W9에서 배운 것 중 가장 체감이 큰 부분.
한국어 BM25는 형태소 분석 없이는 제대로 작동하지 않는다는 것을 직접 증명.

| 설정 | "공부" 검색 | "운동" 검색 | "스트레스" 검색 |
|------|-----------|-----------|---------------|
| BM25 기본 (공백 분리) | ? | ? | ? |
| BM25 + **Kiwi** | ? | ? | ? |

**관찰 포인트:**
- Kiwi 없이: "공부했다", "공부하면서" 같은 활용형이 매칭되는가?
- Kiwi 있으면: "공부했다" → ["공부", "하", "었", "다"]로 분리되어 "공부" 검색에 매칭
- 검색 결과 개수와 관련성이 얼마나 차이나는가?

### 추가 실험 (Day 8~9, 시간 여유 있으면 같이)

Streamlit을 AI로 빠르게 끝내면 아래 실험도 가능하다.

#### 실험 3: k 값 비교 (A+B 같이)

| 설정 | 관찰 포인트 |
|------|------------|
| k=2 | 답변이 너무 짧거나 정보 부족한가? |
| k=5 | 적절한 양인가? |
| k=10 | 답변에 관련 없는 내용이 섞이는가? 토큰 비용은? |

#### 실험 4: chunk_size 비교 (A+B 같이)

| 설정 | 관찰 포인트 |
|------|------------|
| 분할 안함 (일기 1건 = 1 Document) | 짧은 일기에서 검색 정확도는? |
| chunk_size=300 | 긴 일기가 적절히 쪼개지는가? |
| chunk_size=500 | 맥락 유지와 정확도의 균형은? |

### 실험 결과 기록 양식

```markdown
## 실험: [실험 이름]
- **변경한 수치**: weights [0.7, 0.3] → [0.3, 0.7]
- **테스트 질문**: "요즘 힘든 일 있었어?"
- **변경 전 검색 결과**: (검색된 문서 제목/내용 요약)
- **변경 후 검색 결과**: (검색된 문서 제목/내용 요약)
- **변경 전 답변**: (복붙)
- **변경 후 답변**: (복붙)
- **결론**: 우리 일기 데이터에서는 7:3이 최적.
           의미 질문에서 3:7이 약간 나았지만,
           키워드 질문 정확도가 크게 떨어져서 7:3이 균형 잡힘.
```

---

## 10. 참고: 빼도 되지만 추가하면 좋은 것

| 우선순위 | 추가 기능 | 소요 | 효과 |
|---------|----------|------|------|
| 1 | 활동 데이터 (voice_sessions 등) | 반나절 | "공부 많이 했다" 주장을 실제 데이터로 검증 |
| 2 | CacheBackedEmbeddings | 1시간 | 재실행 시 임베딩 비용 절감 |
| 3 | FlashRank Reranker | 반나절 | 검색 품질 향상 (데이터 많을수록 효과 큼) |
