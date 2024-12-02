from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch
import parsing_routine
import audio_analysis
import tempfile
import os
import httpx
import emotion_mapping

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# # Load model 
# Cuda GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Apple Silicon GPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = T5TokenizerFast.from_pretrained("paust/pko-chat-t5-large")
model = T5ForConditionalGeneration.from_pretrained("VOICE_model")
model.to(device)

# Create FastAPI app
app = FastAPI()

input_template = '''
당신은 모든 종류의 상황에서 적절한 가전기기 제어 루틴을 추천하는 AI입니다.
일상적인 상황부터 매우 독특하고 예상치 못한 상황까지, 모든 순간에 맞는 스마트홈 루틴을 제안해주세요.
사용자는 자신의 상황을 반말로 표현하며, 당신은 공감하며 친절하게 "~할게요" 형식으로 루틴을 제안합니다.

사용 가능한 가전기기 목록 (이 기기들만 사용할 수 있음):
- 에어컨: 온도, 세기, 모드 설정값 명시
- 공기청정기
- 로봇청소기: 모드(청소 시작/청소 중지 중 선택)만 명시
- 세탁기
- 건조기
- 스타일러
- TV
- 정수기: 온수/냉수/정수 중 선택
- 식기세척기
- 월패드: 조명 조절(밝게/어둡게)만 가능

응답 시 지켜야 할 사항:
1. 가전기기가 직접 실행할 수 있는 동작만 포함할 것
2. 사람이 직접 해야 하는 행동은 제외할 것 (예: 빨래 널기, 식기 정리하기 등)
3. 각 기기의 설정값을 구체적으로 명시할 것
4. 목적이나 의도는 자유롭게 포함 가능
5. 위에 명시된 가전기기만 사용할 것 (창문, 커튼 등 다른 요소 언급 금지)

좋은 예시:
- "편안한 취침을 위해 에어컨을 26도로 설정하고 월패드로 조명을 어둡게 하고 TV를 끌게요."
- "상쾌한 아침을 위해 로봇청소기 청소를 시작하고 공기청정기를 강하게 켤게요."
- "영화 감상을 위해 에어컨을 24도로 맞추고 TV를 켜고 월패드로 조명을 어둡게 설정할게요."

나쁜 예시:
- "정수기에서 온수를 받아서 라면을 끓일게요." (사람이 직접 하는 행동 포함)
- "세탁기를 돌리고 빨래를 널어둘게요." (빨래 널기는 기기가 할 수 없는 동작)
- "식기세척기를 비우고 설거지를 시작할게요." (식기 정리는 사람의 행동)

현재 상황 정보 : {}
'''

input_template = input_template.replace("\n", " ").strip()

class InputData(BaseModel):
    situation: str

# handling text based routine recommendation
@app.post('/recommend_routine/')
async def recommend_routine(data: InputData):
    prev_response = dict()
    input_text = data.situation
    print(input_text)
    situation = input_template.format(input_text)
    # situation = input_text
    # convert to token & load to device
    input_ids = tokenizer(situation, return_tensors="pt").input_ids.to(device)

    while True:
        # model prediction
        outputs = model.generate(
            input_ids,
            max_length=1024,
            temperature=0.5,
            no_repeat_ngram_size=6,
            do_sample=True,
            num_return_sequences=1,
        )

        routine = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # remove duplicated routine recommendation
        if prev_response.get(input_text) == None or prev_response[input_text] != routine :
            prev_response[input_text] = routine
            break

    # routine = "시원해지시게끔 에어컨을 24도로 설정하고, 정수기에서 냉수 준비해드릴게요."

    routine_output = parsing_routine.parse_device_control(routine)

    routine_output["routine"] = routine
    
    return routine_output

# handling voice audio file analysis & convert it into input text
@app.post('/voice_analysis/')
async def voice_analysis(audio: UploadFile = File(...)):
    if audio.content_type != "audio/wave":
        raise HTTPException(status_code=400, detail="Only .wav file format is supported!");

    # save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file_path = tmp.name
        tmp.write(await audio.read())

    try:
        # Google STT analysis
        google_results = audio_analysis.analyze_with_google(file_path)
        audio_analysis.print_google_results(google_results)

        # Hume AI audio file analysis (extract extra verbal details)
        hume_results = audio_analysis.analyze_with_hume(file_path)
        audio_analysis.print_hume_results(hume_results)

        if google_results and hume_results and 'error' not in google_results: 
            print("\n=== 최종 분석 결과 ===")
            print("-" * 40)
            text = google_results['text']
            top_emotion = audio_analysis.get_top_emotion(hume_results)
            print(f"{text} ({top_emotion})")
            top_emotion = emotion_mapping.map_emotion(top_emotion)
            final_input = f"{text} ({top_emotion})"

        # Send POST request to `/recommend_routine/`
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    # "http://3.133.23.226:8000/recommend_routine/",
                    "http://127.0.0.1:8000/recommend_routine/",
                    json={"situation": final_input},
                    timeout=60.0
                )

            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to get routine: {response.text}"
                )
    finally:
        # delete temporarily save file
        if os.path.exists(file_path):
                    os.remove(file_path)


    