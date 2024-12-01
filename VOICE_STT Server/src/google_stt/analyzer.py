from google.cloud import speech
from google.cloud.language_v1 import LanguageServiceClient, Document
import os
import ffmpeg
from typing import Optional, Dict, Any

class GoogleVoiceSentimentAnalyzer:
    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials/service_account.json"

        self.speech_client = speech.SpeechClient()
        self.language_client = LanguageServiceClient()

    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """음성 파일을 텍스트로 변환"""
        try:
            with open(audio_file_path, 'rb') as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="ko-KR",
                enable_automatic_punctuation=True
            )

            response = self.speech_client.recognize(config=config, audio=audio)

            transcribed_text = ""
            for result in response.results:
                transcribed_text += result.alternatives[0].transcript

            return transcribed_text

        except Exception as e:
            print(f"음성 변환 중 에러 발생: {str(e)}")
            return None

    def analyze_sentiment(self, text: str) -> Optional[Dict[str, Any]]:
        """텍스트의 감정 분석"""
        try:
            document = Document(
                content=text,
                type_=Document.Type.PLAIN_TEXT,
                language="ko"
            )

            sentiment = self.language_client.analyze_sentiment(
                request={'document': document}
            )

            return {
                'text': text,
                'sentiment_score': sentiment.document_sentiment.score,
                'sentiment_magnitude': sentiment.document_sentiment.magnitude,
                'sentences': [{
                    'text': sentence.text.content,
                    'score': sentence.sentiment.score,
                    'magnitude': sentence.sentiment.magnitude
                } for sentence in sentiment.sentences]
            }

        except Exception as e:
            print(f"감정 분석 중 에러 발생: {str(e)}")
            return None

    def analyze_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """음성 파일 분석 (변환 + STT + 감정 분석)"""

        # 1. 음성을 텍스트로 변환
        text = self.transcribe_audio(audio_file_path)
        if not text:
            return {'error': '음성 변환 실패'}

        # 2. 텍스트 감정 분석
        sentiment_result = self.analyze_sentiment(text)
        if not sentiment_result:
            return {'error': '감정 분석 실패'}

        return sentiment_result
