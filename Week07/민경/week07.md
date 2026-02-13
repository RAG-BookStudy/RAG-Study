# 임베딩

텍스트 → 숫자 벡터로 변환

```python
text1 = "강아지"
embedding1 = [0.2, 0.8, 0.1, 0.9, ...]

text2 = "개"
embedding2 = [0.3, 0.7, 0.2, 0.8, ...]  # 의미 비슷 → 벡터 비슷

text3 = "자동차"
embedding3 = [0.9, 0.1, 0.8, 0.2, ...]  # 의미 다름 → 벡터 다름

# 코사인 유사도
similarity(embedding1, embedding2) = 0.95  # 높음
similarity(embedding1, embedding3) = 0.12  # 낮음
```

### 의미를 포착하는 방법

같이 나타나는 단어는 비슷한 의미를 가진다.

```python
# 학습 데이터
"""
강아지는 귀엽다
개는 충성스럽다
고양이는 귀엽다
자동차는 빠르다
"""

# 학습 과정
# "강아지" 주변에 나타나는 단어: [귀엽다, 는, ...]
# "개" 주변에 나타나는 단어: [충성스럽다, 는, ...]
# "고양이" 주변에 나타나는 단어: [귀엽다, 는, ...]

# 결과: "강아지", "개", "고양이"의 문맥이 비슷함
# → 임베딩 벡터도 비슷하게 학습됨

# 실제 임베딩
embedding_dog = [0.8, 0.2, 0.1, ...]     # "귀여움" 차원이 높음
embedding_cat = [0.7, 0.3, 0.1, ...]     # "귀여움" 차원이 높음
embedding_car = [0.1, 0.1, 0.9, ...]     # "속도" 차원이 높음

# 입력: "강아지는 [MASK] 동물이다"
# 학습 목표: [MASK] 자리에 올 단어 예측

# 모델이 학습하는 것:
# 1. "강아지"라는 단어의 의미
# 2. "동물" 앞에 올 수 있는 형용사
# 3. "는"이라는 조사의 역할

# 이 과정을 수십억 개 문장으로 반복
# → **의미**가 벡터 공간에 인코딩됨

# 결과:
# - 비슷한 의미 단어 → 가까운 벡터
# - 다른 의미 단어 → 먼 벡터
```

### 벡터 공간에서의 의미

```python
# 유명한 예시: 단어 연산
king - man + woman ≈ queen

# 실제로는
king_vector = [0.1, 0.9, 0.8, ...]
man_vector = [0.2, 0.7, 0.1, ...]
woman_vector = [0.2, 0.1, 0.1, ...]

result = king_vector - man_vector + woman_vector
# result와 가장 가까운 단어 찾기
# → queen!

# "왕"과 "남성"의 관계 = "여왕"과 "여성"의 관계
# → 벡터 공간에서 이 관계가 "방향"으로 표현됨
```

---

## **임베딩 차원**

```
OpenAI text-embedding-3-small: 1536차원
OpenAI text-embedding-3-large: 3072차원
BERT: 768차원
sentence-transformers: 384~768차원
```

## 차원의 의미

각 차원은 **의미적/**독립적인 **특징을 가진다**

```python
# 단순화된 예시 (5차원 임베딩)
word_embeddings = {
    "강아지": [
        0.9,  # 차원 0: "동물성"
        0.8,  # 차원 1: "애완동물"
        0.7,  # 차원 2: "작음"
        0.2,  # 차원 3: "야생"
        0.1   # 차원 4: "위험"
    ],
    "호랑이": [
        0.9,  # "동물성" 높음
        0.1,  # "애완동물" 낮음
        0.3,  # "작음" 낮음
        0.9,  # "야생" 높음
        0.9   # "위험" 높음
    ]
}

# 각 차원이 복잡한 의미 조합을 표현
# 인간이 해석 불가능한 추상적 특징, 수학적으로는 의미 있는 패턴
```

### 왜 1536차원일까

너무 적으면 (예: 50차원)

- 표현력 부족
- "강아지"와 "개"를 구분하기 어려움
- "배(과일)"과 "배(신체)"를 구분 못함

적당하면 (예: 768차원 - BERT)

- 대부분의 의미 구분 가능
- 계산 효율적

많으면 (예: 1536차원 - OpenAI)

- 미묘한 뉘앙스까지 표현
- 계산 비용 증가
- 과적합 위험

---

## 동음이의어는 어떻게 구분할까

```python
sentences = [
    "배가 고프다",      # 신체(stomach)
    "배를 먹었다",      # 과일(pear)
    "배를 탔다",       # 탈것(ship)
]
```

### 문맥 임베딩 (Contextual Embedding)

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 문장 임베딩
emb1 = embeddings.embed_query("배가 고프다")
emb2 = embeddings.embed_query("배를 먹었다")
emb3 = embeddings.embed_query("배를 탔다")

# 유사도
sim_1_2 = cosine_similarity(emb1, emb2)  # 0.45 (조금 비슷 - 둘 다 "배" 포함)
sim_1_3 = cosine_similarity(emb1, emb3)  # 0.35 (다름)
sim_2_3 = cosine_similarity(emb2, emb3)  # 0.30 (다름)

# 확장 문맥
emb1_full = embeddings.embed_query("점심을 못 먹어서 배가 고프다")
emb2_full = embeddings.embed_query("과일 가게에서 배를 먹었다")
emb3_full = embeddings.embed_query("제주도 가는 배를 탔다")

# 더 명확한 구분
sim_1_2_full = cosine_similarity(emb1_full, emb2_full)  # 0.25 (더 다름)
sim_1_3_full = cosine_similarity(emb1_full, emb3_full)  # 0.20 (더 다름)

# 문맥이 길수록 구분 잘 된다.
```

"배가 고프다" 처리 시

- "배" 토큰이 다른 토큰들을 참조:
- "고프다" → 강한 연관성 (배 = 신체)
- "가" → 주격 조사

"배를 먹었다" 처리 시

- "배" 토큰이 다른 토큰들을 참조:
- "먹었다" → 강한 연관성 (배 = 음식)
- "를" → 목적격 조사

같은 "배"지만

- 주변 단어가 다름
- 가중치가 다름
- 최종 임베딩이 다름