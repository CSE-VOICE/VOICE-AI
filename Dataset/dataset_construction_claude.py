from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

def create_routine_prompt() -> ChatPromptTemplate:
    chat_template = ChatPromptTemplate.from_template(
        '''
        당신은 모든 종류의 상황에서 적절한 가전기기 제어 루틴을 추천하는 AI입니다.
        일상적인 상황부터 매우 독특하고 예상치 못한 상황까지, 모든 순간에 맞는 스마트홈 루틴을 제안해주세요.
        사용자는 자신의 상황을 반말로 표현하며, 당신은 공감하며 친절하게 "~할게요" 형식으로 루틴을 제안합니다.

        사용 가능한 가전기기 목록 (이 기기들만 사용할 수 있음):
        - 에어컨
        - 공기청정기
        - 로봇청소기
        - 세탁기
        - 건조기
        - 스타일러
        - TV
        - 정수기
        - 냉장고
        - 식기세척기
        - 월패드

        특별히 제한된 설정값이 있는 기기:
        - 공기청정기: 세기/강도만 언급
        - 월패드: 조명 조절만 언급 (밝게/약하게)

        고려해야 할 상황 유형:
        1. 일상적 상황
           다음은 일상적 상황의 예시일 뿐이며, 이외의 다양한 일상 상황들을 자유롭게 생성해주세요:
            - "퇴근하고 집에 왔는데 너무 더워."
            - "주말인데 빨래가 너무 밀렸어."
            
        2. 특별하거나 예상치 못한 상황
           다음은 특별한 상황의 예시일 뿐이며, 이외의 독특하고 창의적인 상황들을 자유롭게 생성해주세요:
            - "새벽에 갑자기 영화 보고 싶네."
            - "오늘 밤하늘이 맑아서 별을 보고 싶어."
            - 한여름 밤 베란다 캠핑

        위 예시들은 참고용일 뿐이며, 자유롭게 새로운 상황을 만들어주세요.
        일상적인 상황과 특별한 상황이 균형있게 생성되어야 합니다.

        응답 시 지켜야 할 사항:
        1. 상황에 가장 적합한 가전기기 조합 선택
        2. 위에 명시된 가전기기만 사용할 것 (창문, 커튼 등 다른 요소 언급 금지)
        3. 구체적인 설정값 명시 (온도, 세기, 모드 등)
        4. 하나의 연속된 문장으로 작성
        5. 실용적이고 효율적인 제어 순서 제안
        6. 제한이 있는 기기들은 반드시 지정된 설정값만 사용

        다음 JSON 형식으로 응답해주세요:
        {{
            "situation": "시간대와 상황을 포함한 설명",
            "routine": "가전기기 제어 순서를 한 줄로 작성"
        }}

        출력 예시:
        {{
            "situation": "밤늦게까지 일하느라 피곤한데 잠이 안 와.",
            "routine": "에어컨 온도를 24도로 맞추고 공기청정기를 켜드릴게요. 조명은 30%로 낮춰서 편안한 분위기를 만들어드릴게요. TV의 취침 모드도 설정해드릴게요."
        }}

        기존 예시에 얽매이지 말고 다양하고 새로운 상황들을 자유롭게 만들어주세요.
        '''
    )
    return chat_template

def create_routine_chain(llm: ChatAnthropic):
    prompt = create_routine_prompt()
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain


def generate_routines(num_routines=10):
    llm = ChatAnthropic(
        model="claude-3-opus-20240229",
        temperature=0.7,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    chain = create_routine_chain(llm)
    
    for i in range(num_routines):
        try:
            result = chain.invoke({})
            print(f"상황: {result['situation']}")
            print(f"루틴: {result['routine']}")
            print("-" * 100)
        except Exception as e:
            print(f"Error generating routine: {str(e)}")
            continue

if __name__ == "__main__":
    generate_routines(10)