import os
import json
import requests
from dotenv import load_dotenv

class HumeBatchAPI:
    def __init__(self):
        # .env 파일에서 API 키 로드
        load_dotenv()
        self.api_key = os.getenv("HUME_API_KEY")
        
        self.base_url = "https://api.hume.ai/v0/batch/jobs"
        
    def analyze_audio(self, file_path, granularity="utterance"):
        """로컬 음성 파일 분석"""
        headers = {
            "X-Hume-Api-Key": self.api_key
        }
        
        # 모델 설정
        json_data = {
            "models": {
                "prosody": {
                    "granularity": granularity
                }
            },
            "notify": True
        }

        # 파일 준비
        files = {
            'file': ('audio.wav', open(file_path, 'rb'), 'audio/wav'),
            'json': ('json', json.dumps(json_data), 'application/json')
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, files=files)
            response.raise_for_status()
            return response.json()["job_id"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error starting job: {str(e)}")
            print(f"Response content: {response.text}")
            return None
    
    def get_job_status(self, job_id):
        """작업 상태 확인"""
        headers = {
            "X-Hume-Api-Key": self.api_key
        }
        
        try:
            response = requests.get(f"{self.base_url}/{job_id}", headers=headers)
            response.raise_for_status()
            return response.json()["state"]["status"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error checking job status: {str(e)}")
            return None
    
    def get_predictions(self, job_id):
        """분석 결과 가져오기"""
        headers = {
            "X-Hume-Api-Key": self.api_key
        }
        
        try:
            response = requests.get(f"{self.base_url}/{job_id}/predictions", headers=headers)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting predictions: {str(e)}")
            return None