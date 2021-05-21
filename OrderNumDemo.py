import socket
import time
import random

host, port = "127.0.0.1", 9090
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))
sendNum = 0

while True:
    time.sleep(3) #sleep 3sec
    randNum = random.randrange(0, 5)
    print(randNum)
    sendNum = str(randNum) #increase x by one

    sock.sendall(sendNum.encode("UTF-8")) #Converting string to Byte, and sending it to C#
    receivedData = sock.recv(1024).decode("UTF-8") #receiveing data in Byte fron C#, and converting it to String
    print(receivedData)