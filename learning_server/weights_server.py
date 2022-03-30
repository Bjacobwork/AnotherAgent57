import socket
from multiprocessing import Process, Value, Lock
import os


def worker(socket, checkpoint_dir, dqn_checkpoint, im_checkpoint):
    import datetime
    while True:
        try:
            client, address = socket.accept()
            with dqn_checkpoint.get_lock():
                dqn_path = checkpoint_dir + f"/agent57_{dqn_checkpoint.value}_dqn.h5"
            print(f"Sending {dqn_path}", end='')
            with open(dqn_path, 'rb') as file:
                client.send(bytes(str(os.path.getsize(dqn_path)), 'utf-8'))
                client.sendall(file.read())
            print(f"\rSent {dqn_path}\n {datetime.datetime.now()}\n")
            with im_checkpoint.get_lock():
                im_path = checkpoint_dir + f"/agent57_{dqn_checkpoint.value}_im.h5"
            print(f"Sending {im_path}", end='')
            with open(im_path, 'rb') as file:
                client.send(bytes(str(os.path.getsize(im_path)), 'utf-8'))
                client.sendall(file.read())
            client.close()
            print(f"\rSent {im_path}\n {datetime.datetime.now()}\n")
        except Exception as e:
            print(e)


def server(params, checkpoint_dir):
    import time
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((params['Misc']['weights_ip'], params['Misc']['weights_port']))
    serversocket.listen(5)

    dqn = -1
    im = -1
    for obj in os.listdir(checkpoint_dir):
        if os.path.isfile(checkpoint_dir + "/" + obj):
            if obj.endswith(".h5"):
                tokens = obj.split("_")
                if tokens[0] == 'agent57':
                    if tokens[-1] == "dqn.h5":
                        dqn = max(dqn, int(tokens[1]))
                    elif tokens[-1] == "im.h5":
                        im = max(im, int(tokens[1]))
    if dqn < 0 or im < 0:
        raise Exception("cannot find valid checkpoints")

    dqn_checkpoint = Value('i', dqn)
    im_checkpoint = Value('i', im)

    workers = [Process(target=worker, args=(serversocket, checkpoint_dir, dqn_checkpoint, im_checkpoint)) for i in
               range(params['Misc']['weights_workers'])]
    for p in workers:
        p.daemon = True
        p.start()

    download_period = params['Misc']['download_period']
    while True:
        time.sleep(download_period)
        for obj in os.listdir(checkpoint_dir):
            if os.path.isfile(checkpoint_dir + "/" + obj):
                if obj.endswith(".h5"):
                    tokens = obj.split("_")
                    if tokens[0] == 'agent57':
                        if tokens[-1] == "dqn.h5":
                            dqn = max(dqn, int(tokens[1]))
                        elif tokens[-1] == "im.h5":
                            im = max(im, int(tokens[1]))
        with dqn_checkpoint.get_lock():
            dqn_checkpoint.value = dqn
        with im_checkpoint.get_lock():
            im_checkpoint.value = im


if __name__ == "__main__":
    import yaml

    with open('../params.yml', 'r') as file:
        params = yaml.full_load(file)
    # server = WeightsServer(params,"../weights/agent57_{}.h5")
    # server.run()
    server(params, "../weights/checkpoints")
