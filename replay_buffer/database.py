import datetime
import time
import mariadb
import traceback

DEFAULT_CONFIG = {
    'host': '127.0.0.1',
    'port': 3307,
    'user': 'root',
    'password': ''
}
class ConnectionManager:

    def setup_database(self):
        with open("../replay_buffer/agent57.sql", 'r') as file:
            for query in file.readlines():
                query = query.strip("\n")
                self.cur.execute(query)

    def __init__(self, config=None):
        if not config:
            config = DEFAULT_CONFIG
        self.config = config
        self.conn = mariadb.connect(**config)
        self.cur = self.conn.cursor()
        self.setup_database()

    def init_episode(self, episode_id, policy):
        try:
            query = "REPLACE INTO episode (episode_id, policy) VALUES (?, ?)"
            self.cur.execute(query, (episode_id, policy))
            return True
        except Exception as e:
            print("init_episode")
            print(e)
            print(traceback.print_exc())
            return False

    def get_trace_batch_ids(self, min_required, batches, batch_lock, batch_index):
        try:
            query = "SELECT COUNT(trace_id) FROM trace"
            self.cur.execute(query)
            for r in self.cur:
                if r[0] < min_required:
                    return False
                break
            query = "SELECT trace_id FROM trace ORDER BY priority DESC"
            self.cur.execute(query)
            trace_id = enumerate(batches[batch_index])
            batch_lock.acquire()
            for r in self.cur:
                if r[0] not in batches:
                    try:
                        next_index, value = next(trace_id)
                        while value >= 0:
                            next_index, value = next(trace_id)
                        batches[batch_index, next_index] = r[0]
                    except StopIteration:
                        break
            batch_lock.release()
            return True
        except Exception as e:
            print("get_trace_batch_ids")
            print(e)
            print(traceback.print_exc())
            return False

    def get_database_size(self):
        try:
            query = "SELECT SUM(data_length + index_length) FROM information_schema.tables WHERE table_schema = ?"
            self.cur.execute(query, ('agent57',))
            for r in self.cur:
                bytes_occupied = r[0]
                break
            return bytes_occupied
        except Exception as e:
            print("get_database_size")
            print(e)
            print(traceback.print_exc())
            return 0

    def get_episode_count(self):
        try:
            query = "SELECT COUNT(*) FROM episode"
            self.cur.execute(query)
            for r in self.cur:
                count = int(r[0])
                break
            return count
        except Exception as e:
            print(e)
            print(traceback.print_exc())

    def remove_all_but(self, num_episodes_allowed, in_progress, progress_lock, batches, batch_lock):
        saving = []
        removing = []
        with progress_lock:
            with batch_lock:
                for eid in in_progress:
                    if eid >= 0:
                        saving.append(eid)
                for c in batches:
                    for eid in c:
                        if eid >= 0:
                            saving.append(eid)
            try:
                query = "SELECT episode_id FROM episode ORDER BY priority DESC"
                self.cur.execute(query)
                count = 0
                for r in self.cur:
                    eid = r[0]
                    count+= 1
                    if eid not in saving:
                        if len(saving) < num_episodes_allowed:
                            saving.append(eid)
                        else:
                            removing.append(eid)
            except Exception as e:
                print(e)
                print(traceback.print_exc())
        print(len(saving))
        print(len(removing))
        try:
            id_to_remove = iter(removing)
            clearing = True
            remove_episode = "DELETE FROM episode "
            remove_trace = "DELETE FROM trace "
            remove_transition = "DELETE FROM transition "
            while clearing:
                these = []
                try:
                    for i in range(16):
                        eid = next(id_to_remove)
                        these.append(eid)
                except StopIteration:
                    clearing = False
                if len(these) > 0:
                    where = f"WHERE (episode_id) IN ({','.join(str(eid) for eid in these)})"
                    self.cur.execute(remove_episode+where)
                    self.cur.execute(remove_trace+where)
                    self.cur.execute(remove_transition+where)
        except Exception as e:
            print(e)
            print(traceback.print_exc())


    def allocate_space(self,bytes_per_transition, byte_limit, target_free_space, in_progress, batches):
        try:
            query = "SELECT SUM(data_length + index_length) FROM information_schema.tables WHERE table_schema = ?"
            self.cur.execute(query, ('agent57',))
            for r in self.cur:
                bytes_occupied = r[0]
                break
            bytes_to_allocate = target_free_space - (byte_limit-bytes_occupied)
            if bytes_to_allocate <= 0:
                return byte_limit-bytes_occupied
            print(f"\nALLOCATING {bytes_to_allocate} BYTES", end="")
            time.sleep(10)
            query = "SELECT episode.episode_id,(SELECT COUNT(step) FROM transition WHERE transition.episode_id = episode.episode_id) as counts FROM episode ORDER BY episode.priority ASC, counts DESC"
            self.cur.execute(query)
            conn_1 = mariadb.connect(**self.config)
            cur_1 = conn_1.cursor()
            trace = "SELECT trace_id FROM trace WHERE episode_id = ?"
            remove_episode = "DELETE FROM episode WHERE episode_id = ?"
            remove_trace = "DELETE FROM trace WHERE episode_id = ?"
            remove_transition = "DELETE FROM transition WHERE episode_id = ?"
            for r in self.cur:
                if r[0] not in in_progress:
                    working = False
                    cur_1.execute(trace, (r[0],))
                    for q in cur_1:
                        if q[0] in batches:
                            working = True
                            break
                    if not working:
                        bytes_to_allocate -= r[1]*bytes_per_transition
                        cur_1.execute(remove_episode, (r[0],))
                        cur_1.execute(remove_trace, (r[0],))
                        cur_1.execute(remove_transition, (r[0],))
                if bytes_to_allocate <= 0:
                    break
            cur_1.close()
            conn_1.close()
            return target_free_space-bytes_to_allocate
        except Exception as e:
            print("allocate_space")
            print(e)
            print(traceback.print_exc())
            return 0

    def upload_transitions(self, transitions):
        import itertools
        try:
            query = "REPLACE INTO transition (episode_id,step,prev_extrinsic_reward,prev_intrinsic_reward,action,observation,hidden_state,mu,q_value,discounted_q)" \
                  " VALUES "+",".join("(?,?,?,?,?,?,?,?,?,?)" for _ in range(len(transitions)))
            transitions = list(itertools.chain(*transitions))
            self.cur.execute(query, transitions)
            return True
        except Exception as e:
            print("upload_transitions")
            print(e)
            print(traceback.print_exc())
            return False

    def calculate_priorities(self, trace_length, replay_period, N, episode_id, training_splits, eta=.9):
        try:
            query = "SELECT COUNT(step) FROM transition WHERE episode_id = ?"
            self.cur.execute(query, (episode_id,))
            count = 0
            for r in self.cur:
                count = r[0]
                break
            if count < trace_length+1:
                remove_episode = "DELETE FROM episode WHERE episode_id = ?"
                remove_trace = "DELETE FROM trace WHERE episode_id = ?"
                remove_transition = "DELETE FROM transition WHERE episode_id = ?"
                self.cur.execute(remove_episode, (episode_id,))
                self.cur.execute(remove_trace, (episode_id,))
                self.cur.execute(remove_transition, (episode_id,))
                return
            from bandit import policies
            import math
            query = "SELECT policy FROM episode WHERE episode_id = ?"
            self.cur.execute(query, (episode_id,))
            for r in self.cur:
                j = r[0]
                break
            beta, gamma = policies.get_policy(j, N)
            training_block = (count-replay_period-1)
            training_len = (trace_length-replay_period)//training_splits
            trace_count = training_block/training_len
            regular_position = math.floor(trace_count)
            offset_position = math.ceil(trace_count-regular_position)

            query = "SELECT prev_extrinsic_reward, prev_intrinsic_reward, q_value, discounted_q FROM transition WHERE episode_id = ? ORDER BY step ASC"
            self.cur.execute(query, (episode_id,))
            transition = iter(self.cur)
            temporal_differences = []
            _, _, v, _ = next(transition)
            while True:
                try:
                    re, ri, nv, gpq = next(transition)
                    temporal_differences.append(abs(re+(beta*ri)+gpq-v))
                    v = nv
                except StopIteration:
                    break
            # Removing replay from temporal_differences
            priorities = []
            init_steps = []
            for i in range(regular_position):
                init_step = i*training_len
                init_steps.append(init_step)
                mean = sum(temporal_differences[init_step+replay_period:init_step+trace_length])/trace_length
                highest = max(temporal_differences[init_step+replay_period:init_step+trace_length])
                priorities.append((eta*highest)+((1-eta)*mean))
            if offset_position:
                init_step = count-trace_length-1
                init_steps.append(init_step)
                mean = sum(temporal_differences[init_step+replay_period:init_step+trace_length])/trace_length
                highest = max(temporal_differences[init_step+replay_period:init_step+trace_length])
                priorities.append((eta*highest)+((1-eta)*mean))
            mean = sum(priorities)/len(priorities)
            highest = max(priorities)
            episode_priority = (eta*highest)+((1-eta)*mean)
            query = "REPLACE INTO trace (episode_id, initial_step, priority) VALUES "
            query += ",".join("(?, ?, ?)" for _ in range(len(init_steps)))
            traces = []
            for i, p in zip(init_steps, priorities):
                traces.append(episode_id)
                traces.append(i)
                traces.append(p)
            self.cur.execute(query, traces)
            query = "UPDATE episode SET priority = ? WHERE episode_id = ?"
            self.cur.execute(query, (episode_priority, episode_id))
        except Exception as e:
            print("calculate_priorities")
            print(e)
            print(traceback.print_exc())

    def get_batch_by_trace_ids(self, learner_data, trace_length):
        import numpy as np
        try:
            trace_query = "SELECT episode_id, initial_step FROM trace WHERE trace_id = ?"
            episode_query = "SELECT policy FROM episode WHERE episode_id = ?"
            transition_query = "SELECT prev_extrinsic_reward, prev_intrinsic_reward, action, observation, hidden_state, mu, discounted_q FROM transition WHERE episode_id = ? AND step >= ? AND step <= ? ORDER BY step ASC"
            batch_size = sum([1 if tid >= 0 else 0 for tid in learner_data.trace_ids])
            first = 0
            last = batch_size-1
            index = 0
            for i, tid in enumerate(learner_data.trace_ids):
                if tid < 0:
                    break
                self.cur.execute(trace_query, (int(tid),))
                for r in self.cur:
                    episode_id = r[0]
                    initial_step = int(r[1])
                    if initial_step == 0:
                        index = first
                        first += 1
                    else:
                        index = last
                        last -= 1
                    break
                self.cur.execute(episode_query, (episode_id,))
                for r in self.cur:
                    learner_data.j[index] = int(r[0])
                    break
                self.cur.execute(transition_query, (episode_id, initial_step, initial_step+trace_length))
                for j, r in enumerate(self.cur):
                    if j == 0:
                        learner_data.hidden[index] = np.frombuffer(r[4], dtype=np.float32).reshape(learner_data.hidden.shape[-1])
                    learner_data.prev_extrinsic_rewards[j][index] = r[0]
                    learner_data.prev_intrinsic_rewards[j][index] = r[1]
                    learner_data.actions[j][index] = r[2]
                    learner_data.observations[j][index] = np.frombuffer(r[3], dtype=np.uint8).reshape(learner_data.observations.shape[-3:])/255
                    # i and j are swapped to avoid transpose
                    learner_data.mu[index][j] = r[5]
                    learner_data.lost_life[index][j] = True if r[6] == 0. else False
                learner_data.episode_ids[index] = episode_id
            learner_data.init_step_count[0] = first
        except Exception as e:
            print(e)
            print(traceback.print_exc())

    def update_priorities(self, episode_ids, trace_ids, priorities, eta=.9):
        update_query = "UPDATE trace SET priority = ? WHERE trace_id = ?"
        for p,tid in zip(priorities,trace_ids):
            try:
                self.cur.execute(update_query, (float(p), int(tid)))
            except Exception as e:
                print(e)
                print(traceback.print_exc())
        updated = []
        select_query = "SELECT priority FROM trace WHERE episode_id = ?"
        update_query = "UPDATE episode SET priority = ? WHERE episode_id = ?"
        for eid in episode_ids:
            if eid not in updated:
                try:
                    self.cur.execute(select_query, (int(eid),))
                    priorities = [float(r[0]) for r in self.cur]
                    p = eta*max(priorities)+(1-eta)*(sum(priorities)/len(priorities))
                    self.cur.execute(update_query, (p, int(eid)))
                    updated.append(eid)
                except Exception as e:
                    print(e)
                    print(traceback.print_exc())

if __name__ == "__main__":
    import numpy as np
    import random
    obs = np.zeros((1,210, 160,1), dtype=np.uint8)
    h = np.zeros((1,512*4), dtype=np.float32)
    episode_id = 0
    j = 20
    cm = ConnectionManager(DEFAULT_CONFIG)
    cm.init_episode(DEFAULT_CONFIG, episode_id, j)
    in_progress = np.array([-1,-1,-1,-1])
    batches = np.array([[-1,-1],[-1,-1]])
    offset = 0
    for i in range(250):
        transitions = []
        for i in range(100):
            transitions.append([episode_id,i+offset,random.randrange(100),random.randrange(100),0,obs.tobytes(), h.tobytes(), random.random(),random.randrange(100),random.randrange(100)])
        offset += 100
        can_upload = cm.allocate_space(DEFAULT_CONFIG,960, 100, in_progress, batches)
        if can_upload:
            cm.upload_transitions(DEFAULT_CONFIG, transitions)
    cm.calculate_priorities(DEFAULT_CONFIG, 80,40,32,episode_id)