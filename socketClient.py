from socket import *

import json


# udp
class socketClient:

    def __init__(self):
        self.clientSock = socket(AF_INET, SOCK_DGRAM)

    def send(self, value, host, port):
        print('연결 확인 됐습니다.')
        print("type :",type(value))

        self.clientSock.sendto(value.encode('utf-8'), (host, port))

        print('메시지를 전송했습니다.')

    def receive(self):
        recvMsg, addr = self.clientSock.recvfrom(2048)
        print(recvMsg.decode())


    def connect(self, queue):

        while True:

            caliValue = queue.get()
            print("connect :", caliValue)
            print("connect type :", type(caliValue))
            if caliValue is None:
                return
            
            caliValue_json = json.dumps(caliValue)

            try:
                if caliValue['type'] == 'pupil':
                    pupilSocket.send(caliValue_json, 'localhost', 8080)
                    #pupilSocket.receive()

                if caliValue['type'] == 'sound':
                    soundSocket.send(caliValue_json, 'localhost', 8081)
                    #soundSocket.receive()

                if caliValue['type'] == 'person':
                    personSocket.send(caliValue_json, 'localhost', 8082)
                    #personSocket.receive()

            except KeyboardInterrupt:
                break
        
pupilSocket = socketClient()
soundSocket = socketClient()
personSocket = socketClient()