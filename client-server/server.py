### The code is replicated from thepythoncode.com

import socket
import tqdm
import os
import threading



class MultiClient(threading.Thread):
    def __init__ (self, address, client_socket):
        threading.Thread.__init__(self)
        self.csocket = client_socket
    def run(self):
        print ("Connection from : ", address)
        # receive the file infos
        # receive using client socket, not server socket
        received = client_socket.recv(BUFFER_SIZE).decode()
        filename, filesize = received.split(SEPARATOR)
        # remove absolute path if there is
        filename = os.path.basename(filename)
        # convert to integer
        filesize = int(filesize)


        progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)
        with open(filename, "wb") as f:
            for _ in progress:
                # read 1024 bytes from the socket (receive)
                bytes_read = client_socket.recv(BUFFER_SIZE)
                if not bytes_read:
                    # nothing is received
                    # file transmitting is done
                    break
                # write to the file the bytes we just received
                f.write(bytes_read)
                # update the progress bar
                progress.update(len(bytes_read))




SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5001


# receive 4096 bytes each time
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

s = socket.socket()
s.bind((SERVER_HOST, SERVER_PORT))


while True:
    s.listen(25)
    print(f"[*] Listening as {SERVER_HOST}:{SERVER_PORT}")

    # accept connection if there is any
    client_socket, address = s.accept()
    newthread = MultiClient(address, client_socket)
    newthread.start()
    # if below code is executed, that means the sender is connected
    print(f"[+] {address} is connected.")







