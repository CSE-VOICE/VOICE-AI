import json
import io
import sys
from ibm_watson import SpeechToTextV1, NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

stt_authenticator = IAMAuthenticator('u7QmlPO4j0OWHPu4_sVW7GiYDU1qyJoDKuw2RYCFqBWo')
speech_to_text = SpeechToTextV1(
    authenticator=stt_authenticator
)
speech_to_text.set_service_url('https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/1e2083b7-b3bc-48ef-9a1c-4713d53c10b7')

nlu_authenticator = IAMAuthenticator('Bpe9CHTZ6D1LCsa3fTKR1MIEIA_fZkph0b8C2ZW0XPmg')
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=nlu_authenticator
)
nlu.set_service_url('https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/3620d1db-53ca-4ff7-9ccf-2c69b7f3bc8d')

def analyze_audio(audio_file_path):
    try:
        print("=== STT 분석 시작 ===")
        with open(audio_file_path, 'rb') as audio_file:
            stt_result = speech_to_text.recognize(
                audio=audio_file,
                content_type='audio/flac',
                model='ko-KR_BroadbandModel' 
            ).get_result()

        # 2. 변환된 텍스트 모음
        all_texts = []
        for item in stt_result['results']:
            text = item['alternatives'][0]['transcript']
            confidence = item['alternatives'][0]['confidence']
            print(f"인식된 텍스트: {text}")
            print(f"신뢰도: {confidence:.2f}\n")
            all_texts.append(text)

        # 3. 감성 분석
        print("=== 감성 분석 시작 ===")
        for text in all_texts:
            try:
                response = nlu.analyze(
                    text=text,
                    features=Features(
                        sentiment=SentimentOptions(
                            document=True
                        )
                    ),
                    language='ko'
                ).get_result()

                sentiment = response['sentiment']['document']
                print(f"텍스트: {text}")
                print(f"감성 점수: {sentiment['score']}")
                print(f"감성 라벨: {sentiment['label']}\n")

            except Exception as e:
                print(f"해당 텍스트 감성 분석 실패: {text}")
                print(f"에러 메시지: {str(e)}\n")

    except Exception as e:
        print(f"에러 발생: {str(e)}")

# 실행
if __name__ == "__main__":
    audio_file_path = "C:/Users/poket/OneDrive/Desktop/Bangbaeseonhaeng-gil.flac"
    analyze_audio(audio_file_path)
    