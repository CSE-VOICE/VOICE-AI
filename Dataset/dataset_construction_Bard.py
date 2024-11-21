from google.cloud import bard
from dotenv import load_dotenv
import pandas as pd
import hashlib
from datetime import datetime

load_dotenv()

bard_client = bard.BardClient(api_key="YOUR_BARD_API_KEY")


def create_routine_prompt() -> str:
    return '''
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
    - TV: 모드(켜기/끄기 중 선택)
    - 정수기: 온수/냉수/정수 중 선택
    - 식기세척기
    - 조명: 밝게/어둡게 중 선택

    출력 형식(JSON):
    {{
        "situation": "시간대와 상황을 포함한 설명",
        "routine": "가전기기가 직접 실행 가능한 동작들만 포함한 한 문장"
    }}
    '''


def generate_routine(bard_client, prompt: str):
    response = bard_client.ask(prompt)
    try:
        content = response.get("content", {}).get("text", {})
        return eval(content) if isinstance(content, str) else content
    except Exception:
        return {}


def log_routine(routine):
    print(f"상황: {routine.get('situation', 'N/A')}")
    print(f"루틴: {routine.get('routine', 'N/A')}")
    print("-" * 100)


def remove_duplications(routines):
    unique_situations = set()
    unique_routines = []
    duplicate_routines = []

    for routine in routines:
        situation_hash = hashlib.sha256(routine['situation'].encode()).hexdigest()
        if situation_hash not in unique_situations:
            unique_situations.add(situation_hash)
            unique_routines.append(routine)
        else:
            duplicate_routines.append(routine)

    return unique_routines, duplicate_routines


def generate_routines(num_routines=100, excel_path=None):
    prompt = create_routine_prompt()
    all_routines = []
    attempts = 0
    max_attempts = num_routines * 2
    failed_attempts = 0

    while len(all_routines) < num_routines and attempts < max_attempts:
        try:
            result = generate_routine(bard_client, prompt)
            if result and 'situation' in result and 'routine' in result:
                all_routines.append(result)
                log_routine(result)
            else:
                failed_attempts += 1
        except Exception:
            failed_attempts += 1

        attempts += 1

    unique_routines, duplicate_routines = remove_duplications(all_routines)

    if not excel_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"smart_home_routines_{timestamp}.xlsx"

    df = pd.DataFrame(unique_routines)
    df.columns = ['상황', '루틴']
    df.to_excel(excel_path, index=False)

    print(f"\n=== 생성 결과 ===")
    print(f"요청한 루틴 수: {num_routines}")
    print(f"생성된 총 루틴 수: {len(all_routines)}")
    print(f"중복 제거 후 루틴 수: {len(unique_routines)}")
    print(f"제거된 중복 항목 수: {len(duplicate_routines)}")
    print(f"API 호출 실패 횟수: {failed_attempts}")
    print(f"결과가 {excel_path}에 저장되었습니다.")

    return unique_routines


if __name__ == "__main__":
    routines = generate_routines(10)
