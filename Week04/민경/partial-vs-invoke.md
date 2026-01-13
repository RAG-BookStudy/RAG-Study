**1. `partial()` - 고정값 바인딩**

```python
# format_instructions는 잘 변하지 않는 형식 지시사항
prompt = prompt.partial(format_instructions=parser.get_format_instructions())
```

- 값 결정 시점: 프롬프트 생성 시점 (한 번만)
- 용도: 실행할 때마다 변하지 않는 고정 값

**2. `invoke()` - 동적 입력**

```python
# question은 매번 다른 값이 입력되는 사용자 질문
chain.invoke({"question": question})
```

- 값 결정 시점: 체인 실행 시점 (매번)
- 용도: 실행할 때마다 달라지는 동적 값

### **`partial`과 `invoke`를** 구분해서 사용하면

```python
# format_instructions는 항상 같음
format_instructions = parser.get_format_instructions()
# 출력: "The output should be formatted as a JSON instance..."

# 하지만 question은 매번 다름
question1 = "자바 컬렉션에 대해 알려주세요"
question2 = "인공지능의 미래는?"
question3 = "파이썬 리스트 설명해줘"

```

- **성능**: `format_instructions`를 매번 전달하지 않아도 됨
- **편의성**: 반복되는 값을 한 번만 설정
- **명확성**: 고정값과 동적값의 구분이 명확

### 둘 다 invoke로 넘길 경우

```python
chain.invoke({
    "question": question,
    "format_instructions": parser.get_format_instructions()
})
```

- 매번 `format_instructions`를 전달해야 함
- 의도 불명확: `format_instructions`가 변경 가능한 것처럼 보임