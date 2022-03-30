import socket
from bandit.multi_armed_bandit import MultiArmedBandit


def worker(socket, params, lock, address, root_dir):
    import datetime
    import traceback
    mab = MultiArmedBandit(params, lock, address, root_dir)
    while True:
        try:
            client, address = socket.accept()
            data = client.recv(1024)
            tokens = data.decode('utf-8').split(" ")
            if tokens[0] == 'get_j':
                j = mab.get_j()
                print(f"\nSending bandit {j}\n {datetime.datetime.now()}")
                client.send(j.to_bytes(2, 'big'))
            elif tokens[0] == 'greedy_j':
                j = mab.greed()
                print(f"\nSending greedy bandit {j}\n {datetime.datetime.now()}")
                client.send(j.to_bytes(2, 'big'))
            elif tokens[0] == "non_greedy_j":
                if mab.k < mab.N:
                    j = int(mab.k[0])
                else:
                    j = mab.ucb()
                print(f"\nSending non-greedy bandit {j}\n {datetime.datetime.now()}")
                client.send(j.to_bytes(2, 'big'))
            elif tokens[0] == 'update':
                arm = int(tokens[1])
                reward = float(tokens[2])
                score = float(tokens[3])
                mab.update_reward(arm, reward, score)
                print(f"\nUpdated {arm} with {reward}\n High Score: {mab.high_score[0]}\n {datetime.datetime.now()}")
                client.send(b'done')
            client.close()
        except Exception as e:
            print(traceback.print_exc())
            print(e)
            print(datetime.datetime.now())


def server(params, root_dir):
    from multiprocessing import Lock, Process
    mab_lock = Lock()
    mab = MultiArmedBandit(params, mab_lock)
    mab.load(root_dir)
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((params['Misc']['bandit_ip'], params['Misc']['bandit_port']))
    serversocket.listen(5)

    workers = [Process(target=worker, args=(serversocket, params, mab_lock, mab.mem.name, root_dir))]
    for w in workers:
        w.daemon = True
        w.start()
    while True:
        pass


if __name__ == "__main__":
    import yaml

    with open('../params.yml', 'r') as file:
        params = yaml.full_load(file)
    server(params, "../bandit_config")
