## 모델 선택 기준

#### 성능:

- 정확도: 복잡한 질문 vs 단순 질문
- 응답 속도: 실시간성 필요 여부
- 컨텍스트 길이: 긴 문서 처리 필요성

#### 비용:

- Input 토큰 비용
- Output 토큰 비용
- 월간 예상 사용량

#### 기술적 제약:

- API 속도 제한
- 배치 처리 지원 여부
- 로컬 실행 가능성

#### 어떤 경우에 폴백할까?

- API 속도 제한 초과
- 타임아웃 (>30초)
- 특정 에러 코드 (429, 500, 503)
- 비용 임계값 초과

---

## 메모리 타입별 활용

### ConversationBufferMemory

**특징**: 모든 대화를 그대로 저장

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

**적합한 경우**:

- 짧은 대화 세션 (< 10회 왕복)
- 모든 컨텍스트가 중요한 경우

**단점**: 토큰 사용량 ⏫

---

### ConversationBufferWindowMemory

**특징**: 최근 N개 대화만 유지

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=5,  # 최근 5개 대화쌍
    memory_key="chat_history",
    return_messages=True
)

```

**적합한 경우**:

- 긴 대화에서 최근 맥락만 필요

**장점**: 토큰 사용량 예측 가능

---

### ConversationSummaryMemory

**특징**: 이전 대화를 요약해서 저장

```python
from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI

memory = ConversationSummaryMemory(
    llm=OpenAI(temperature=0),
    memory_key="chat_history"
)

```

**적합한 경우**:

- 매우 긴 대화 세션
- 전체 맥락은 필요하지만 토큰 절약 필요

**단점**: 요약 시 정보 손실 가능, 요약 비용 발생

---

### ConversationSummaryBufferMemory

**특징**: 최근 대화 + 이전 요약 조합

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=OpenAI(temperature=0),
    max_token_limit=500,  # 토큰 제한
    memory_key="chat_history",
    return_messages=True
)

```

**적합한 경우**:

- 대부분의 프로덕션 환경

**장점**: 비용과 성능의 최적 균형

---

## Temperature 설정

### Temperature 범위별 특성

```
0.0 - 0.3: 결정적, 일관성 높음
→ FAQ, 정확한 정보 제공, 코드 생성, 사실 기반 답변

0.4 - 0.7: 균형적, 약간의 창의성
→ 일반 대화, 설명, 요약

0.8 - 1.0: 창의적, 다양성 높음
→ 아이디어 생성, 브레인스토밍, 창작 콘텐츠

1.0+: 매우 랜덤 (비추)

```

---

## 컨텍스트 압축

### 무엇을 요약하고 무엇을 유지할까?

**유지해야 할 정보**

```python
critical_info = [
    "사용자 명시적 지시사항",
    "진행 중인 작업 컨텍스트",
    "최근 3-5개 대화",
    "오류 및 수정 내역",
]
```

**요약 가능한 정보**

```python
can_summarize = [
    "이전 세션 내역",
    "일반적인 배경 대화",
    "성공적으로 완료된 작업",
]
```