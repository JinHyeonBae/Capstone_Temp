import speech_recognition as sr
from socketClient import socketClient
sr.__version__ # 3.8.1

import time, random, json


class speechRecognition : 

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.speechSocket = socketClient()
        

        # response object returns 3 parts - error message boolean for success and transcribed text if completed
        self.response = {
            "success": random.choice([True, False]),
            "error": None,
            "transcription": None,
            "type" : 1,
            "cheating_word": [],
            'type' : 'sound'              
        }
        # 이건 다른 데이터랑 겹쳐서 판단하기로 하자
        self.cheating_word = [
            '보여',
            '답',
            '넘겨봐',
            '답뭐지',
            '공유좀'
            '공유',
            '해줘',
            '알고있어',
            '보여',
            '시험지',
            '보여 줘'
        ]
        self.cheating = False

    def send(self):
        self.speechSocket.send(self.response, 'localhost', 8081)

    def get(self):
        return self.response
        
            
    def word_catch(self):
        speak_value = self.response["transcription"]
        res = self.response
        print(speak_value)
        if speak_value is None:
            return 

        for value in self.cheating_word:
            print("hello")
            for sp in speak_value.split(' '):
                print("sp :", sp)
                print("value :", value)
                if sp == value:
                    res['cheating_word'].append(speak_value)

    def recognize_speech_from_mic(self):
        
        print("hello")
        # check that recognizer and microphone arguments are appropriate type
        if not isinstance(self.recognizer, sr.Recognizer):
            raise TypeError("`recognizer` must be `Recognizer` instance")
        if not isinstance(self.microphone, sr.Microphone):
            raise TypeError("`microphone` must be `Microphone` instance")

        with self.microphone as source:
            self.recognizer.pause_threshold = 1
            # duration이 높을 수록 답변 늦어짐
            self.recognizer.adjust_for_ambient_noise(source, duration=3)
            print("Say :")
            audio = self.recognizer.listen(source, timeout=3)
    
        try:
            self.response["transcription"] = self.recognizer.recognize_google(audio, language='ko')
            
            print(self.response["transcription"] )
        except sr.RequestError:
            self.response["success"] = False
            self.response["error"] = "API unavailable"
        except sr.UnknownValueError:
            self.response["error"] = "Unable to recognize speech"
        except TypeError:
            print("no voice")
            pass


    def run(self):

        while True:

            self.recongnize_voice()
            self.word_catch()

            self.speechSocket.send(self.response, 'localhost', 8081)

        
    def test(self):
        print("this is speech process")


if __name__ == '__main__':
    sp = speechRecognition()
    #sp.recognize_speech_from_mic()
   
    while True:
        # print(i)
        speech = sp.recognize_speech_from_mic()
        sp.word_catch()
        sp.send()