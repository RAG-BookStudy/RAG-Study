## 스키마 검증 실패한다면?

### 1. 기본 예외 처리 (Try-Catch)

```python
parser = PydanticOutputParser(pydantic_object=EvaluationResult)

try:
    result = parser.parse(llm_output)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    
    # 예외 처리(ex: 기본값 반환, 재시도...)
    ...

```

### 2. 자동 수정: OutputFixingParser

```python
# LLM이 잘못된 형식을 출력했을 때 자동으로 수정 시도
fixing_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=ChatOpenAI(model="gpt-4", temperature=0)
)

try:
    # 기본 파서로 먼저 시도
    result = parser.parse(llm_output)
except Exception:
    # 실패하면 fixing_parser가 LLM을 한 번 더 호출해서 수정
    result = fixing_parser.parse(llm_output)

```

**장점**: LLM이 형식을 약간 틀렸을 때 자동 복구

**단점**: 추가 API 호출 비용, 시간 증가

### 3. 재시도: RetryOutputParser

```python
retry_parser = RetryOutputParser.from_llm(
    parser=parser,
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    max_retries=2  # 최대 재시도 횟수
)

# 원본 프롬프트와 함께 전달
result = retry_parser.parse_with_prompt(llm_output, prompt)

```

`RetryOutputParser` 동작 방식:

1. 기본 파서로 파싱 시도
2. 실패 시 원본 프롬프트 + 에러 정보를 LLM에게 다시 전달 / 재요청

```
	1단계: 기본 파싱 
		→ 2단계: 자동 수정 파서 
			→ 3단계: 수동 재시도 (프롬프트가 있는 경우)
				→ 4단계: 기본값 반환
```