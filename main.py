import traceback, cv2, time
from threading import Thread

from manager import Manager
from socketClient import socketClient
from speechRecognition import speechRecognition

from multiprocessing import Queue
from productor import Productor
from consumer import Consumer

if __name__ == "__main__":
    
    managerObj = Manager() #pupil
    speechObj  = speechRecognition() #speechRecognition
    socket = socketClient()
    system_queue =  Queue()
    # pro = Productor(system_queue)
    # con = Consumer(system_queue)
 
    try:
        managerThread = Thread(name="producer1", target=managerObj.start, args=(system_queue,))
        speechObj = Thread(name="producer2", target=speechObj.run, args=(system_queue,))
        socket = Thread(name="socket consumer", target=socket.connect, args=(system_queue,))

        managerThread.start()
        speechObj.start()
        socket.start()
        # pro.start()

    except KeyboardInterrupt :
        print("Bye :)")
        exit()
    

    except ConnectionResetError:
        print("ConnectionResetError")
        pass

    # while True:
    #     key = input()

    #     if key == 'q':
    #         managerObj.run()
    #     elif key == 'w':
    #         speechObj.send()
    #     else:
    #         break

    # print("while문 탈출")