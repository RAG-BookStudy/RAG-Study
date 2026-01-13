### 커스텀 파서가 필요한가?

1. 기존 파서로 처리 불가능한 형식 (Markdown, 수식 등)
2. 특수한 후처리 로직 필요
3. 성능 최적화 필요
4. 도메인 특화 파싱

### 커스텀 1: 기존 파서 조합

```python
from langchain.output_parsers import (
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
    OutputFixingParser
)

# 여러 파서를 순차적으로 적용
class CompositeParser:
    def __init__(self):
        self.primary = PydanticOutputParser(pydantic_object=MyModel)
        self.fallback = CommaSeparatedListOutputParser()
        self.fixer = OutputFixingParser.from_llm(
            parser=self.primary,
            llm=ChatOpenAI()
        )

    def parse(self, text: str):
        ... 

```

### 커스텀 2: 새로 만들기

```python
# 1. 성능이 중요한 경우 (파서 체인 오버헤드 제거)
class FastCustomParser(BaseOutputParser):
    """불필요한 검증 생략, 빠른 파싱에 집중"""
    pass

# 2. 도메인 특화 로직
class ChemistryEquationParser(BaseOutputParser):
    """화학식 파싱, 균형 맞추기 등 특수 로직"""
    pass

# 3. 레거시 시스템 통합
class LegacyFormatParser(BaseOutputParser):
    """기존 시스템 형식에 맞춤"""
    pass

```

기존 조합으로 80% 해결할 수 있음!