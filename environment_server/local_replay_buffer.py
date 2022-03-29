import time
from replay_buffer import replay_client, database

def transition_upload_process(params, input_queue, replay_lock, config=None):
    cm = database.ConnectionManager(config)
    transitions = []
    resets = []
    while True:
        while input_queue.empty():
            pass
        for _ in range(input_queue.qsize()):
            for episode_id, step, extrinsic_reward, intrinsic_reward, action, observation, hidden, mu, value, discounted, reset in zip(*input_queue.get()):
                transitions.append([int(episode_id),
                                        int(step),
                                        float(extrinsic_reward),
                                        float(intrinsic_reward),
                                        int(action),
                                        observation.tobytes(),
                                        hidden.tobytes(),
                                        float(mu),
                                        float(value),
                                        float(discounted)])
                resets.append([episode_id, reset])
        if len(transitions) >= 256:
            with replay_lock:
                allowed_to_upload = replay_client.request_transition_upload(params, len(transitions))
            if allowed_to_upload > 0:
                packets_1024 = allowed_to_upload//1024
                final_packet = allowed_to_upload-(packets_1024*1024)
                packets = [1024 for _ in range(packets_1024)]
                if final_packet:
                    packets.append(final_packet)
                for upload in packets:
                    uploading = transitions[:upload]
                    resetting = resets[:upload]
                    transitions = transitions[upload:]
                    resets = resets[upload:]
                    success = cm.upload_transitions(uploading)
                    if success:
                        for r in resetting:
                            if r[1]:
                                with replay_lock:
                                    replay_client.signal_episode_end(params, int(r[0]))
                    else:
                        transitions += uploading
                        resets += resetting