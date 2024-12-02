from src.google_stt.analyzer import GoogleVoiceSentimentAnalyzer
from src.hume.client import HumeBatchAPI
import time
from operator import itemgetter

def analyze_with_google(file_path: str) -> dict:
    """Google STT와 감정 분석 실행"""
    analyzer = GoogleVoiceSentimentAnalyzer()
    return analyzer.analyze_audio(file_path)

def print_google_results(results: dict):
    """Google 분석 결과 출력"""
    if 'error' in results:
        print(f"Error: {results['error']}")
        return

    print("\nRunning Google Analysis...")
    print("\n=== Google STT & Sentiment Analysis ===")
    print("-" * 40)
    print(f"{results['text']}\n")

def analyze_with_hume(file_path: str) -> dict:
    """Hume AI 분석 실행"""
    client = HumeBatchAPI()
    job_id = client.analyze_audio(file_path)
    
    if not job_id:
        return {"error": "Failed to start Hume analysis"}
    
    print("Running Hume Analysis...")
    print(f"Hume analysis started - Job ID: {job_id}")
    
    while True:
        status = client.get_job_status(job_id)
        print(f"Status: {status}")
        
        if status == "COMPLETED":
            return client.get_predictions(job_id)
        elif status == "FAILED":
            return {"error": "Hume analysis failed"}
            
        time.sleep(5)

def print_hume_results(predictions: dict):
    """Hume 분석 결과 출력"""
    try:
        prediction = predictions[0]['results']['predictions'][0]
        prosody_data = prediction['models']['prosody']
        speech_data = prosody_data['grouped_predictions'][0]['predictions'][0]
        
        print(f"\n=== 음성 정보 ===")
        print("-" * 40)
        print(f"음성 인식 신뢰도: {speech_data['confidence']:.2%}\n")
        
        print("=== 감정 분석 (상위 10개) ===")
        print("-" * 40)
        print(f"{'감정':25} 점수")
        print("-" * 40)
        
        sorted_emotions = sorted(speech_data['emotions'], 
                               key=itemgetter('score'), 
                               reverse=True)
        
        for emotion in sorted_emotions[:10]:
            print(f"{emotion['name']:25} {emotion['score']:.4f} ({emotion['score']:.2%})")
            
    except Exception as e:
        print(f"Error parsing Hume results: {str(e)}")

def get_top_emotion(predictions: dict) -> str:
    """Hume 분석 결과에서 가장 높은 감정 추출"""
    try:
        prediction = predictions[0]['results']['predictions'][0]
        prosody_data = prediction['models']['prosody']
        emotions = prosody_data['grouped_predictions'][0]['predictions'][0]['emotions']
        sorted_emotions = sorted(emotions, key=itemgetter('score'), reverse=True)
        return sorted_emotions[0]['name']
    except Exception as e:
        print(f"Error extracting emotion: {str(e)}")
        return None