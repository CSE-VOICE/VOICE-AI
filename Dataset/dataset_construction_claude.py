from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Any, Tuple

load_dotenv()

def create_routine_prompt() -> ChatPromptTemplate:
    chat_template = ChatPromptTemplate.from_template(
        '''
        당신은 모든 종류의 상황에서 적절한 가전기기 제어 루틴을 추천하는 AI입니다.
        일상적인 상황부터 매우 독특하고 예상치 못한 상황까지, 모든 순간에 맞는 스마트홈 루틴을 제안해주세요.
        사용자는 자신의 상황을 반말로 표현하며, 당신은 공감하며 친절하게 "~할게요" 형식으로 루틴을 제안합니다.

        사용 가능한 가전기기 목록 (이 기기들만 사용할 수 있음):
        - 에어컨: 온도, 세기, 모드 설정값 명시
        - 공기청정기: 모드(켜기/끄기 중 선택)만 명시
        - 로봇청소기: 모드(청소 시작/청소 중지 중 선택)만 명시
        - 세탁기
        - 건조기
        - 스타일러
        - TV: 모드(켜기/끄기 중 선택), 
        - 정수기: 온수/냉수/정수 중 선택
        - 식기세척기
        - 조명: 밝게/어둡게 중 선택

        루틴 작성 시 중요 규칙:
        1. 가전기기가 직접 실행할 수 있는 동작만 포함할 것
        2. 사람이 직접 해야 하는 행동은 제외할 것 (예: 빨래 널기, 식기 정리하기 등)
        3. 각 기기의 설정값을 구체적으로 명시할 것
        4. 목적이나 의도는 자유롭게 포함 가능

        좋은 예시:
        - "편안한 취침을 위해 에어컨을 26도로 설정하고 월패드로 조명을 어둡게 하고 TV를 끌게요."
        - "상쾌한 아침을 위해 로봇청소기 청소를 시작하고 공기청정기를 강하게 켤게요."
        - "영화 감상을 위해 에어컨을 24도로 맞추고 TV를 켜고 월패드로 조명을 어둡게 설정할게요."

        나쁜 예시:
        - "정수기에서 온수를 받아서 라면을 끓일게요." (사람이 직접 하는 행동 포함)
        - "세탁기를 돌리고 빨래를 널어둘게요." (빨래 널기는 기기가 할 수 없는 동작)
        - "식기세척기를 비우고 설거지를 시작할게요." (식기 정리는 사람의 행동)

        고려해야 할 상황 유형:
        1. 일상적 상황
           다음은 일상적 상황의 예시일 뿐이며, 이외의 다양한 일상 상황들을 자유롭게 생성해주세요:
            - "퇴근하고 집에 왔는데 너무 더워."
            - "주말인데 빨래가 너무 밀렸어."
            
        2. 특별하거나 예상치 못한 상황
           다음은 특별한 상황의 예시일 뿐이며, 이외의 독특하고 창의적인 상황들을 자유롭게 생성해주세요:
            - "새벽에 갑자기 영화 보고 싶네."
            - "오늘 밤하늘이 맑아서 별을 보고 싶어."

        상황 생성 가이드:
        - 현재 시점의 상황과 니즈만 포함
        - 미래의 상황이나 조건부 상황 포함하지 않기
        - 일상적인 상황과 특별한 상황이 균형있게 포함
        - 이전에 생성된 상황과 유사하거나 중복되지 않도록 할 것
        - 매우 독특하고 창의적인 상황도 생성 가능

        다음 JSON 형식으로 응답해주세요:
        {{
            "situation": "시간대와 상황을 포함한 설명",
            "routine": "가전기기가 직접 실행 가능한 동작들만 포함한 한 문장"
        }}

        올바른 출력 예시:
        {{
            "situation": "한밤중에 무서운 영화가 보고 싶어졌어.",
            "routine": "무서운 분위기를 위해 TV를 켜고 월패드로 조명을 어둡게 설정하고 에어컨을 25도로 맞출게요."
        }}

        잘못된 출력 예시:
        {{
            "situation": "아침에 일어나자마자 커피가 마시고 싶어.",
            "routine": "정수기에서 온수를 받아 커피를 타서 마시면서 TV를 볼게요."
        }}

        기존 예시에 얽매이지 말고 다양하고 새로운 상황들을 자유롭게 만들되, 반드시 가전기기가 직접 실행 가능한 동작만 포함하여 루틴을 만들어주세요.
        '''
    )
    return chat_template

def create_routine_chain(llm: ChatAnthropic):
    prompt = create_routine_prompt()
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain

def remove_duplications(routines: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    상황(situation)이 완전히 동일한 루틴을 제거합니다.
    
    Args:
        routines: 생성된 루틴 딕셔너리의 리스트
        
    Returns:
        Tuple[고유한 루틴 리스트, 제거된 중복 루틴 리스트]
    """
    unique_situations = set()
    unique_routines = []
    duplicate_routines = []
    
    for routine in routines:
        situation = routine['situation']
        if situation not in unique_situations:
            unique_situations.add(situation)
            unique_routines.append(routine)
        else:
            duplicate_routines.append(routine)
    
    return unique_routines, duplicate_routines

def generate_routines(num_routines=10000, excel_path='smart_home_routines.xlsx'):
    llm = ChatAnthropic(
        model="claude-3-opus-20240229",
        temperature=0.7
    )
    chain = create_routine_chain(llm)
    
    all_routines = []
    attempts = 0
    max_attempts = num_routines * 2
    
    while len(all_routines) < num_routines and attempts < max_attempts:
        try:
            result = chain.invoke({})
            all_routines.append(result)
            print(f"상황: {result['situation']}")
            print(f"루틴: {result['routine']}")
            print("-" * 100)
            
        except Exception as e:
            print(f"Error generating routine: {str(e)}")
        
        attempts += 1
    
    # 중복 제거 및 중복된 항목 확인
    unique_routines, duplicate_routines = remove_duplications(all_routines)
    
    # DataFrame 생성 및 엑셀 저장
    df = pd.DataFrame(unique_routines)
    df.columns = ['상황', '루틴']
    df.to_excel(excel_path, index=False)
    
    print(f"\n=== 생성 결과 ===")
    print(f"요청한 루틴 수: {num_routines}")
    print(f"생성된 총 루틴 수: {len(all_routines)}")
    print(f"중복 제거 후 루틴 수: {len(unique_routines)}")
    print(f"제거된 중복 항목 수: {len(duplicate_routines)}")
    
    if duplicate_routines:
        print("\n=== 제거된 중복 항목 ===")
        for idx, routine in enumerate(duplicate_routines, 1):
            print(f"\n중복 항목 {idx}:")
            print(f"상황: {routine['situation']}")
            print(f"루틴: {routine['routine']}")
            print("-" * 50)
    
    print(f"\n최종 결과가 {excel_path}에 저장되었습니다.")
    
    return unique_routines

if __name__ == "__main__":
    routines = generate_routines(10)