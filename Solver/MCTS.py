import threading
import multiprocessing
from random import random

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from copy import copy
import os
from time import sleep

from Problem.IRMCSP import Value

MAX_GLOBAL_T = 1e5
SIMULATION_BUDGET = 250

MAX_EPISODE_LENGTH = 8
BATCH_SIZE = 32

GAMMA = .99  # discount rate for advantage estimation and reward discounting

LSTM_SIZE = 256
CLIP_BY_NORM = 40.0

LEARNING_RATE = 1e4

# EPS_START_RANGE = [0.4, 0.3, 0.3]
# EPS_STOP_RANGE = [0.1, 0.01, 0.5]
#
# # EPS_START_RANGE = [0]
# # EPS_STOP_RANGE = [0]
#
# EPS_STEPS = MAX_GLOBAL_T
#
# MAX_STAGNATION_RANGE = [int(x * MAX_GLOBAL_T) for x in [0.01, 0.05, 0.10, 0.20, 0.25, 0.33, 0.50]]

SOLUTION_FIRST_CRITERION = "ratio"
# SOLUTION_FIRST_CRITERION = "value"

# # REPORT_LEVEL = "learning"
# # REPORT_LEVEL = "episodes"
# REPORT_LEVEL = "steps"

STATE_SIZE = None
ACTION_SIZE = None
DOMAIN_SHAPE = None

MODEL_PATH = './model'

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class AC_Network:
    def __init__(self, domain_shape, s_size, a_size, scope, trainer):
        global DOMAIN_SHAPE
        global STATE_SIZE
        global ACTION_SIZE

        DOMAIN_SHAPE = domain_shape
        STATE_SIZE = s_size
        ACTION_SIZE = a_size

        self.trainer = trainer

        self.saved_solutions = {}
        self.global_best_solution = 0

        with tf.variable_scope(scope):
            # Input layers
            self.inputs = tf.placeholder(shape=[None, STATE_SIZE], dtype=tf.float32)

            self.sequences = tf.reshape(self.inputs, shape=[-1, STATE_SIZE])

            hidden = slim.fully_connected(slim.flatten(self.sequences), LSTM_SIZE, activation_fn=tf.nn.relu)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.sequences)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, LSTM_SIZE])

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, ACTION_SIZE,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_one_hot = tf.one_hot(self.actions, ACTION_SIZE, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_one_hot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, CLIP_BY_NORM)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class Node:
    def __init__(self, parent, meeting, action):
        self.parent = parent
        self.children = {}
        self.meeting = meeting
        self.action = action


class Tree:
    def __init__(self):
        self.root = Node(None, None, None)


class Worker:
    def __init__(self, worker_id, global_tree, env, global_network, trainer, model_path, global_episodes):
        self.id = worker_id
        self.name = "worker_" + str(worker_id)
        self.tree = global_tree
        self.env = env
        self.env.worker = self
        self.global_network = global_network
        self.model_path = model_path
        self.trainer = trainer
        self.global_t = global_episodes
        self.total_steps = 0
        self.episode_buffer = []
        self.episode_values = []
        self.episode_step_count = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.id))

        # eps_rnd = choice(range(len(EPS_START_RANGE)))
        # self.eps_start = EPS_START_RANGE[eps_rnd]
        # self.eps_end = EPS_STOP_RANGE[eps_rnd]
        # self.eps_steps = EPS_STEPS

        # self.max_stagnation = 0

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(DOMAIN_SHAPE, STATE_SIZE, ACTION_SIZE, self.name, self.trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = self.actions = np.identity(ACTION_SIZE, dtype=bool).tolist()

    # def getEpsilon(self):
    #     if self.total_steps >= self.eps_steps:
    #         return self.eps_end
    #     else:
    #         return self.eps_start + self.total_steps * (
    #         self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def work(self, sess, coord, saver):
        print("Starting worker " + str(self.id))

        global_t = sess.run(self.global_t)
        with sess.as_default(), sess.graph.as_default():
            while global_t < MAX_GLOBAL_T:
                env = copy(self.env)

                rnn_state = self.local_AC.state_init

                current_node = self.tree.root
                has_selected = False
                done = False

                meeting, state = env.get_state()
                while not has_selected:
                    action, value, rnn_state = self.predict(sess, state, rnn_state)
                    action, meeting_value = env.action_to_value(meeting, action)

                    env.take_action(meeting, meeting_value)
                    env.evaluate_solution()

                    new_meeting, new_state = env.get_state()
                    # CHANGE!!!
                    reward = 1

                    if action not in current_node.children:
                        current_node.children[action] = Node(current_node, meeting, action)
                        has_selected = True

                    current_node = current_node.children[action]

                    self.save_step_data(sess, state, action, reward, new_state, done, value[0, 0], rnn_state)

                    state = new_state
                    meeting = new_meeting

                self.simulate_from_node(copy(env), sess, coord, saver)

                self.episode_rewards.append(self.episode_reward)
                self.episode_lengths.append(self.episode_step_count)
                self.episode_mean_values.append(np.mean(self.episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(self.episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(self.episode_buffer, sess, GAMMA, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if global_t % 5 == 0 and global_t != 0:
                    # if self.name == 'worker_0' and episode_count % 25 == 0:
                    #     time_per_step = 0.05
                    #     images = np.array(episode_frames)
                    #     make_gif(images, './frames/image' + str(episode_count) + '.gif',
                    #              duration=len(images) * time_per_step, true_image=True, salience=False)
                    if global_t % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(global_t) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, global_t)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.global_t.assign_add(1))
                global_t += 1

    def predict(self, sess, s, rnn_state):
        # Take an action using probabilities from policy network output.
        a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                        feed_dict={self.local_AC.inputs: [s],
                                                   self.local_AC.state_in[0]: rnn_state[0],
                                                   self.local_AC.state_in[1]: rnn_state[1]})
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)
        return a, v, rnn_state

    def save_step_data(self, sess, state, action, reward, new_state, done, value, rnn_state):
        self.episode_buffer.append([state, action, reward, new_state, done, value])
        self.episode_values.append(value)

        self.episode_reward += reward
        self.episode_step_count += 1

        # If the episode hasn't ended, but the experience buffer is full, then we
        # make an update step using that experience rollout.
        if len(self.episode_buffer) == BATCH_SIZE and done != True and self.episode_step_count != MAX_EPISODE_LENGTH - 1:
            # Since we don't know what the true final return is, we "bootstrap" from our current
            # value estimation.
            v1 = sess.run(self.local_AC.value,
                          feed_dict={self.local_AC.inputs: [state],
                                     self.local_AC.state_in[0]: rnn_state[0],
                                     self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
            v_l, p_l, e_l, g_n, v_n = self.train(self.episode_buffer, sess, GAMMA, v1)
            episode_buffer = []
            sess.run(self.update_local_ops)

    def simulate_from_node(self, current_env, sess, coord, saver):
        best_env = copy(current_env)
        global_t = sess.run(self.global_t)

        with sess.as_default(), sess.graph.as_default():
            sess.run(self.update_local_ops)

            d = False

            m, s = current_env.get_state()
            rnn_state = self.local_AC.state_init

            while self.episode_step_count < SIMULATION_BUDGET:
                action, value, rnn_state = self.predict(sess, state, rnn_state)
                action, meeting_value = env.action_to_value(meeting, action)

                env.take_action(meeting, meeting_value)
                env.evaluate_solution()

                new_meeting, new_state = env.get_state()
                # CHANGE!!!
                reward = 1

                if action not in current_node.children:
                    current_node.children[action] = Node(current_node, meeting, action)
                    has_selected = True

                current_node = current_node.children[action]

                self.save_step_data(sess, state, action, reward, new_state, done, value[0, 0], rnn_state)

                state = new_state
                meeting = new_meeting


    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n


class Solution:
    uid = 1

    def __init__(self):

        self.id = self.next_uid()
        self.worker = None
        self.title = ""
        self.notes = ""
        self.semester = ""
        self.nr_meetings = 0
        self.nr_groups = 0

        self.domains_as_categories = None
        self.domains_as_calendar = None

        self.durations = {}
        self.durations_in_slots = {}

        self.constraints = None

        self.groups_per_meeting = {}
        self.groups_by_meeting = {}

        self.int_to_indices = {}
        self.time_by_indices = {}

        self.placed_ratio = 0.0
        self.pref_value = 0

        self.iteration = 0
        self.stagnation = 0

        self.meeting_state = {}

        self.calendar = None
        self.placed_values = {}

    def next_uid(self):
        Solution.uid += 1
        return Solution.uid

    def get_state(self):
        meetings = np.zeros(self.nr_meetings)

        s = None
        for m in range(self.nr_meetings):
            h = np.hstack(self.domains_as_categories[m].values())
            meetings[m] = np.sum(h) / np.size(h)

            if m in self.placed_values:
                meetings[m] -= 1

                rooms = np.zeros((DOMAIN_SHAPE["rooms"]))
                weeks = np.zeros((DOMAIN_SHAPE["weeks"]))
                days = np.zeros((DOMAIN_SHAPE["days"]))
                slots = np.zeros((DOMAIN_SHAPE["slots"]))

                indices = self.placed_values[m].indices

                rooms[indices[0]] = meetings[m]
                weeks[indices[1]] = meetings[m]
                days[indices[2]] = meetings[m]
                slots[indices[3]] = meetings[m]

                h = np.hstack([rooms, weeks, days, slots])
            else:
                h[h == 1] = meetings[m]

            if s is None:
                s = h
            else:
                s = np.vstack((s, h))

        m_random = np.random.random(self.nr_meetings)
        m_max_shuffle = np.where(meetings == np.max(meetings), m_random, 0)
        m = np.argmax(m_max_shuffle)

        s = np.reshape(s, (1, STATE_SIZE, -1))

        return m, s

    def action_to_value(self, m, a):
        d = self.domains_as_calendar[m]
        eps = self.worker.getEpsilon()

        if random() < eps:
            random_a = np.random.random(d.shape)
            a = np.argmax(d * random_a)

        else:
            max_a = np.max(a)
            random_a = np.random.random(d["rooms"].shape)
            random_max_a = np.where(a == max_a, random_a, 0)
            a = np.argmax(random_max_a)

        if d[a] == 1:
            indices = self.int_to_indices[a]

            duration_in_slots = self.durations_in_slots[m]
            m_v = Value(m, indices, duration_in_slots)

        else:
            m_v = None

        return m_v

    def take_action(self, meeting, value):
        for slot in range(value.start, value.end + 1):
            conflict = self.calendar[value.room, value.week, value.day, slot]
            if conflict > 0:
                self.remove_meeting(conflict)

            for room in range(DOMAIN_SHAPE["rooms"]):
                conflict = self.calendar[room, value.week, value.day, slot]
                if conflict > 0:
                    for group in self.groups_by_meeting[meeting]:
                        if group in self.groups_by_meeting[conflict]:
                            self.remove_meeting(conflict)
                            break

        value.pref_value = self.groups_per_meeting[meeting]
        # TODO check constraints and reduce pref_value if necessary
        for constraint in self.constraints:
            conflicts = constraint.check_consistency(self, meeting, value)
            if conflicts:
                for conflict in conflicts:
                    self.remove_meeting(conflict)

        # TODO quick and dirty implementation for prefererring earlier timeslots
        # if index_time_end > ((16 - 8) * 4) - 1:
        pref_late_end_cost_factor = max(0.01, 1 - (value.end / (self.calendar.shape[3] - 1)))
        value.pref_value *= pref_late_end_cost_factor

        self.place_meeting(meeting, value)

    def place_meeting(self, meeting, value):
        self.calendar[value.room, value.week, value.day, value.start: value.end + 1] = meeting
        self.placed_values[meeting] = value

    def remove_meeting(self, meeting):
        self.calendar[self.calendar == meeting] = 0
        self.placed_values.pop(meeting)

    def evaluate_solution(self):
        self.placed_ratio = len(self.placed_values) / self.nr_meetings

        self.pref_value = 0
        for placed_value in self.placed_values.values():
            self.pref_value += placed_value.pref_value

    def is_better_solution(self, current_best, first_criterion="ratio"):
        if first_criterion == "ratio":
            criterion = {0: {"s": self.placed_ratio, "cb": current_best.placed_ratio},
                         1: {"s": self.pref_value, "cb": current_best.pref_value}}
        elif first_criterion == "value":
            criterion = {0: {"s": self.pref_value, "cb": current_best.pref_value},
                         1: {"s": self.placed_ratio, "cb": current_best.placed_ratio}}

        if criterion[0]["s"] > criterion[0]["cb"]:
            return True
        elif criterion[0]["s"] == criterion[0]["cb"]:
            if criterion[1]["s"] > criterion[1]["cb"]:
                return True
            else:
                return False
        else:
            return False

    def is_stagnant(self, limit):
        if self.stagnation > 0 and self.stagnation >= limit:
            return True
        else:
            return False

    def reset_solution(self):
        self.iteration = 0
        self.stagnation = 0
        self.placed_values = {}
        self.placed_ratio = 0.0
        self.pref_value = 0
        self.meeting_state = {}
        self.calendar_state = np.zeros_like(self.calendar_state)
        self.calendar = np.zeros_like(self.calendar)


class MonteCarloTreeSearch:
    def __init__(self):
        self.tree = Tree()

    def run(self):
        load_model = False
        tf.reset_default_graph()

        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)

        # Create a directory to save episode playback gifs to
        if not os.path.exists('./frames'):
            os.makedirs('./frames')

        with tf.device("/cpu:0"):
            global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
            master_network = AC_Network(STATE_SIZE, ACTION_SIZE, 'global', None)  # Generate global network
            num_workers = multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
            workers = []
            # Create worker classes
            for i in range(num_workers):
                workers.append(Worker(i, STATE_SIZE, ACTION_SIZE, trainer, MODEL_PATH, global_episodes))
            saver = tf.train.Saver(max_to_keep=5)

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            if load_model == True:
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            # This is where the asynchronous magic happens.
            # Start the "work" process for each worker in a separate threat.
            worker_threads = []
            for worker in workers:
                worker_work = lambda: worker.work(MAX_EPISODE_LENGTH, GAMMA, sess, coord, saver)
                t = threading.Thread(target=(worker_work))
                t.start()
                sleep(0.5)
                worker_threads.append(t)
            coord.join(worker_threads)