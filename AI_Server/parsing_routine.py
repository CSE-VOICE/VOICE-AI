from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import json

load_dotenv()

class LowercaseBooleanJsonParser(JsonOutputParser):
    def parse(self, text):
        parsed = super().parse(text)
        result = json.loads(
            json.dumps(parsed, default=str).replace('True', 'true').replace('False', 'false')
        )
        
        for update in result["updates"]:
            if update["onoff"].upper() == "OFF":
                update["state"] = "대기"
                update["is_active"] = False
                update["onoff"] = "off"
            else:
                update["onoff"] = "on"
                
        return result

def create_device_state_prompt() -> ChatPromptTemplate:
    chat_template = ChatPromptTemplate.from_template(
        '''
        당신은 자연어로 된 스마트홈 기기 제어 문장을 파싱하여 각 기기의 상태를 JSON 형태로 변환하는 AI입니다.
        
        기기 ID 정보:
        1. 에어컨 id: 1
        2. 공기청정기 id: 2
        3. 로봇청소기 id: 3
        4. TV id: 4
        5. 조명 id: 5
        6. 정수기 id: 6
        7. 세탁기 id: 7
        8. 건조기 id: 8
        9. 식기세척기 id: 9
        10. 스타일러 id: 10

        각 기기별 권장 상태값:
        - 에어컨: ["{{temperature}}°C", "냉방 모드", "제습 모드", "송풍 모드", "자동 모드", "파워 바람", "취침 모드"]
        - 공기청정기: ["무풍", "약풍", "중풍", "강풍"]
        - 로봇청소기: ["청소 모드", "충전 모드", "청소 완료", "빠른 청소"]
        - TV: ["음악 재생", "영화 모드"]
        - 조명: ["밝게", "어둡게"]
        - 정수기: ["냉수 준비", "온수 준비"]
        - 세탁기: ["표준 세탁", "급속 세탁", "섬세 세탁", "세탁 완료", "탈수 중", "헹굼 중"]
        - 건조기: ["표준 건조", "강력 건조", "섬세 건조", "건조 완료"]
        - 식기세척기: ["살균 건조", "강력", "일반", "세척 완료"]
        - 스타일러: ["표준 스타일링", "급속 스타일링", "강력 스타일링", "위생살균 모드", "관리 완료"]

        다음 문장을 파싱해주세요: {text}

        출력 형식:
        {{
          "updates": [
            {{
              "appliance_id": 기기ID,
              "user_id": 1,
              "name": "기기명",
              "onoff": "ON"/"OFF",
              "state": "현재 상태",
              "is_active": true/false
            }}
          ]
        }}

        규칙:
        1. 문장에서 언급된 기기만 포함할 것
        2. appliance_id는 위 ID 목록에서 매핑
        3. user_id는 항상 6로 설정
        4. 기기를 켜는 동작의 경우: onoff="ON", is_active=true
        5. 기기를 끄는 동작의 경우: onoff="OFF", is_active=false
        6. state는 가능한 한 권장 상태값 중에서 선택하되, 적절한 값이 없는 경우 표현 그대로 사용
        7. 각 기기별로 state는 1개만 설정할 것 ((중복된 상태 업데이트 금지, 에어컨의 경우 온도와 모드를 함께 표현)
        8. 명시적으로 끄라는 지시가 없는 경우 모든 기기는 "ON" 상태로 간주
        '''
    )
    return chat_template

def create_parser_chain(llm: ChatAnthropic):
    prompt = create_device_state_prompt()
    parser = LowercaseBooleanJsonParser()
    chain = prompt | llm | parser
    return chain

def parse_device_control(control_text: str):
    """기기 제어 문장을 파싱하여 JSON 형태로 변환합니다."""
    llm = ChatAnthropic(
        model="claude-3-opus-20240229",
        temperature=0.3
    )
    
    try:
        parser_chain = create_parser_chain(llm)
        result = parser_chain.invoke({"text": control_text})
        return result
    
    except Exception as e:
        print(f"Error parsing device control: {str(e)}")
        return None