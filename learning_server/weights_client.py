import socket


def download_files(params, path_template):
    s = socket.socket()
    s.connect((params['Misc']['weights_ip'], params['Misc']['weights_port']))
    file_size = int(s.recv(1024).decode('utf-8'))
    dqn_file = s.recv(file_size)
    file_size = int(s.recv(1024).decode('utf-8'))
    im_file = s.recv(file_size)
    s.close()
    with open(path_template.format("dqn"), 'wb') as file:
        file.write(dqn_file)
    with open(path_template.format("im"), 'wb') as file:
        file.write(im_file)


if __name__ == "__main__":
    import yaml

    with open('../params.yml', 'r') as file:
        params = yaml.full_load(file)
    # client = WeightsClient(params)
    # client.get_weights('../weights/model_download.h5')
    # client.get_weights('../weights/okay.h5')
    download_files(params, "../weights/test_template_{}.h5")
