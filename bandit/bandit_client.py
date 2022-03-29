import socket

class BanditClient:

    def __init__(self, params):
        self.ip = params['Misc']['bandit_ip']
        self.port = params['Misc']['bandit_port']

    def get_policy_index(self, byte_code):
        s = socket.socket()
        s.connect((self.ip, self.port))
        s.send(byte_code)
        j = s.recv(1024)
        j = int.from_bytes(j, 'big')
        s.close()
        return j

    def get_j(self):
        return self.get_policy_index(b'get_j')

    def get_non_greedy_j(self):
        return self.get_policy_index(b'non_greedy_j')

    def get_greedy_j(self):
        return self.get_policy_index(b'greedy_j')

    def update_reward(self, arm, reward, score):
        s = socket.socket()
        msg = f"update {arm} {reward} {score}"
        s.connect((self.ip, self.port))
        s.send(bytes(msg, 'utf-8'))
        msg = s.recv(1024)
        s.close()


if __name__ == "__main__":
    import yaml
    with open('../actors/params.yml', 'r') as file:
        params = yaml.full_load(file)
    client = BanditClient(params)
    N = 32
    for i in range(N*2):
        j = client.get_j()
        print(f"recieved arm {j}")
        client.update_reward(j, i/N)
        print(f"updating {j} with {i/N}")