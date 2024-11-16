import os
import pandas as pd
from tqdm import tqdm

original_dataset = pd.read_excel("../dataset.xlsx")

data = {"id" : [], "situation" : [], "routine" : []}

situations = original_dataset['situation']
routines = original_dataset['routine']

input_template = '''
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

        응답 시 지켜야 할 사항:
        1. 상황에 가장 적합한 가전기기 조합 선택
        2. 위에 명시된 가전기기만 사용할 것 (창문, 커튼 등 다른 요소 언급 금지)
        3. 구체적인 설정값 명시 (온도, 세기, 모드 등)
        4. 하나의 연속된 문장으로 작성
        5. 실용적이고 효율적인 제어 순서 제안
        6. 제한이 있는 기기들은 반드시 지정된 설정값만 사용

        현재 상황 정보 : {}
        '''

id_counter = 1
for idx in tqdm(range(len(situations))) :
    data["id"].append(id_counter)
    data["situation"].append(input_template.format(situations[idx]))
    data["routine"].append(routines[idx])
    id_counter += 1


# 데이터 전처리 한 파일 output
df = pd.DataFrame(data)
output_dir = "../preprocessed_dataset.csv"
df.to_csv(output_dir, index=False, encoding="utf-8")


