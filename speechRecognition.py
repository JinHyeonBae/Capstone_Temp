import speech_recognition as sr
from socketClient import socketConnect
from type import systemType
sr.__version__ # 3.8.1

import time, random


class speechRecognition : 

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # response object returns 3 parts - error message boolean for success and transcribed text if completed
        self.response = {
            "success": True,
            "error": None,
            "transcription": None,
            "type" : systemType.sound,
            "cheating_word": []              
        }
        # 이건 다른 데이터랑 겹쳐서 판단하기로 하자
        self.cheating_word = [
            '답 좀 보여줘',
            '보여줘',
            '종이 넘겨봐',
            '답 뭐지',
            '답 알고 있어',
            '안 보여'
        ]
        self.cheating = False


    def get(self):
        return self.response

    def send(self):
        # try:
        time.sleep(5)
        ch = random.choice([True, False])
        print("ch :", ch)
        testObj = {
            'success': True,
            'error': None,
            'transcription': f"this is test {ch}",
            'type' : 1,
            'cheating_word' : ['test', 'is']
        }
        print(testObj)
        
        socketConnect(testObj)

        # except KeyboardInterrupt :
        #     print("Bye :)")
        #     break
            
    def word_catch(self, speak_value):
        res = self.response

        for value in self.cheating_word:
            if value.find(speak_value) != -1:
                res['cheating_word'].append(speak_value)

    def recognize_speech_from_mic(self):

        # check that recognizer and microphone arguments are appropriate type
        if not isinstance(self.recognizer, sr.Recognizer):
            raise TypeError("`recognizer` must be `Recognizer` instance")
        if not isinstance(self.microphone, sr.Microphone):
            raise TypeError("`microphone` must be `Microphone` instance")

        while True :
            with self.microphone as source:
                self.recognizer.pause_threshold = 1
                # duration이 높을 수록 답변 늦어짐
                self.recognizer.adjust_for_ambient_noise(source, duration=5)
                print("Say :")
                audio = self.recognizer.listen(source)
        
            try:
                self.response["transcription"] = self.recognizer.recognize_google(audio, language='ko')
                print(self.recognizer.recognize_google(audio, show_all=True, language='ko'))
            except sr.RequestError:
                self.response["success"] = False
                self.response["error"] = "API unavailable"
            except sr.UnknownValueError:
                self.response["error"] = "Unable to recognize speech"


if __name__ == '__main__':
    sp = speechRecognition()
    #sp.recognize_speech_from_mic()
    while True:
        #speech = sp.recognize_speech_from_mic()
        sp.send()