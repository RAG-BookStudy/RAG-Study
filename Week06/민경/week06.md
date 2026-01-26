## 청킹

### 1. CharacterTextSplitter (고정 구분자)

```python
from langchain.text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",  # 이 구분자로 분할
    chunk_size=1000,
    chunk_overlap=200
)

text = """
첫 번째 문단입니다.
내용이 계속됩니다.

두 번째 문단입니다.
또 다른 내용.

세 번째 문단입니다.
"""

chunks = splitter.split_text(text)
# → "\n\n"로만 나눔

```

- 명확한 단락 구분이 있는 문서
- Markdown 문서 (`\n\n`로 단락 구분)
- 정형화된 포맷

**장점**

- 간단하고 예측 가능
- 빠른 처리 속도
- 명확한 구조가 있는 문서에 효과적

**단점**

- 유연성 부족
- 구분자가 없으면 제대로 분할 안 됨
- chunk_size를 초과해도 강제로 자르지 않음

---

### 2. RecursiveCharacterTextSplitter (계층적 구분자)

**동작 방식**

```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # 순서대로 시도
)

# 작동 원리:
# 1. "\n\n"로 분할 시도
# 2. 청크가 여전히 크면 "\n"로 분할
# 3. 그래도 크면 ". "로 분할
# 4. 그래도 크면 " " (공백)로 분할
# 5. 최후에는 ""(문자 단위)로 분할

```

- CharacterTextSplitter보다 약간 느림 (큰 차이 없음)
- 구분자 순서 설정이 중요
- 다양한 문서 타입에 범용적으로 사용 가능

**장점**

- 가장 자연스러운 분할 지점 찾음
- 문맥 보존 우수

---

### 3. TokenTextSplitter (토큰 기반)

**동작 방식**

```python
from langchain.text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=1000,  # 1000 토큰
    chunk_overlap=200
)

# 내부적으로 토크나이저 사용
# tiktoken (OpenAI) 기본값

```

- 토큰 제한이 엄격한 경우
- API 비용 최적화가 중요한 경우

**단점**

- 문맥 경계 무시 (문장 중간에서 자를 수 있음)
- 토큰화 오버헤드

**부적합한 문서**

- 문맥 보존이 중요한 교육 자료
- 자연스러운 읽기 흐름이 중요한 문서

RecursiveCharacterTextSplitter로 분할 

→ 토큰 수가 너무 많으면 TokenTextSplitter로 재분할

---

### 4. SpacyTextSplitter (NLP 기반)

**동작 방식**

```python
from langchain.text_splitters import SpacyTextSplitter

splitter = SpacyTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    pipeline="ko_core_news_sm" 
)

# Spacy의 문장 경계 인식 활용
# 문장이 완전히 끝나는 지점에서만 분할

```

- 문장 구조가 복잡한 문학 작품
- 문장 단위 분석이 중요한 경우

**장점**

- 문장 경계를 정확하게 인식함. 자연스러운 분할
- 다국어 지원

**단점**

- Spacy 모델 설치 필요
- 처리 속도 느림 (NLP 분석 수행)

**부적합한 문서**

- 대용량 데이터 (속도 문제)
- 실시간 처리 필요한 경우

---

### 5. MarkdownHeaderTextSplitter (구조 기반)

**동작 방식**

```python
from langchain.text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

markdown_text = """
# Chapter 1
내용...

## Section 1.1
내용...

### Subsection 1.1.1
내용...
"""

chunks = markdown_splitter.split_text(markdown_text)

# 각 청크에 계층 정보 메타데이터 포함
# chunk.metadata = {
#     "Header 1": "Chapter 1",
#     "Header 2": "Section 1.1",
#     "Header 3": "Subsection 1.1.1"
# }

```

- Markdown 기술 문서
- 위키, 블로그 포스트
- 구조화된 학습 자료

**장점**

- 문서 구조 보존
- 계층 정보를 메타데이터로 활용
- 검색 시 섹션 필터링 가능

**단점**

- Markdown 전용

---

## 시맨틱 청킹(Semantic Chunking)

### 동작 원리

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"  # 또는 "standard_deviation", "interquartile"
)

text = """
객체지향 프로그래밍은 소프트웨어 설계 패러다임입니다.
이는 데이터와 기능을 객체로 묶는 것을 의미합니다.

캡슐화는 OOP의 핵심 원칙 중 하나입니다.
데이터를 외부로부터 보호하고 접근을 제어합니다.
"""

chunks = semantic_splitter.split_text(text)

# 작동 방식:
# 1. 문장별로 임베딩 생성
# 2. 인접 문장 간 유사도 계산
# 3. 유사도가 급격히 떨어지는 지점에서 분할
# 4. 주제가 바뀌는 자연스러운 경계 찾음

```

**예시**

```python
# 문장별 유사도 분석
문장1: "객체지향 프로그래밍은..." → 임베딩 A
문장2: "이는 데이터와 기능을..." → 임베딩 B
# 유사도(A, B) = 0.85 (높음) → 같은 청크

문장3: "캡슐화는 OOP의..." → 임베딩 C
# 유사도(B, C) = 0.45 (낮음) → 여기서 분할!

# 결과:
청크1: 문장1 + 문장2 (OOP 일반 개념)
청크2: 문장3 + 문장4 (캡슐화 구체적 설명)

```

---

### ⭕ 적합

1. 주제가 자주 바뀌는 문서(뉴스 기사)

2. 서술적이고 연속적인 문서(소설, 에세이, 기행문)

3. 다양한 관점이 섞인 문서(토론, 인터뷰, 대화록)

4. 학술 논문

---

**❌ 부적합**

1. 이미 구조가 명확한 문서(Markdown, HTML)

2. 짧고 독립적인 항목들(FAQ, 용어집, 단어장)

3. 코드 파일

- 함수/클래스 단위로 분할해야 함
- CodeTextSplitter 사용

4. 표, 데이터, 수치 중심 문서

---

시맨틱 청킹은 주력으로 쓰기엔 오버헤드가 크지만, 

**특정 고품질 문서에 선택적으로 사용**하면 좋은 결과를 얻을 수 있음!