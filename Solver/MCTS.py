import datetime
import random
import threading
import multiprocessing
from time import sleep

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from copy import copy, deepcopy
import os
import math

from Problem.IRMCSP import Value

# import networkx as nx
# import matplotlib.pyplot as plt

MAX_GLOBAL_T = 1e4
SIMULATION_BUDGET = 256

# THREADS = 1
THREADS = multiprocessing.cpu_count()

LOAD_MODEL = False
TRAIN_ON_PLAYOUT = True

BATCH_SIZE = 16

UCT_C = math.sqrt(2)
RAVE_B = 0.01

GAMMA = 0.99  # discount rate for advantage estimation and reward discounting
LAMBDA = 0.9

LSTM_SIZE = 128
CLIP_BY_NORM = 5

LEARNING_RATE = 1e4

SOLUTION_FIRST_CRITERION = "ratio"
# SOLUTION_FIRST_CRITERION = "value"

# # REPORT_LEVEL = "learning"
# # REPORT_LEVEL = "episodes"
# REPORT_LEVEL = "steps"

STATE_SIZE = None
STATE_SHAPE = None
ACTION_SIZE = None

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
    def __init__(self, state_shape, state_size, action_size, scope, trainer):
        global STATE_SHAPE
        global STATE_SIZE
        global ACTION_SIZE

        STATE_SHAPE = state_shape
        STATE_SIZE = state_size
        ACTION_SIZE = action_size

        self.trainer = trainer

        self.saved_solutions = {}
        self.global_best_solution = 0

        with tf.variable_scope(scope):
            # Input layers
            self.inputs = tf.placeholder(shape=[None, *STATE_SHAPE], dtype=tf.float32)

            self.sequences = tf.reshape(self.inputs, shape=[-1, STATE_SHAPE[0] * STATE_SHAPE[1], STATE_SHAPE[2] * STATE_SHAPE[3], 1])

            self.conv1 = tf.layers.conv2d(inputs=self.sequences,
                                          filters=LSTM_SIZE,
                                          kernel_size=[STATE_SHAPE[0] * STATE_SHAPE[1], STATE_SHAPE[2] * STATE_SHAPE[3]],
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          padding="valid",
                                          activation=tf.nn.relu)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(slim.flatten(self.conv1), [0])
            step_size = tf.shape(self.sequences)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, LSTM_SIZE])

            # Output layers for policy and value estimations
            self.policy_meeting = slim.fully_connected(rnn_out, ACTION_SIZE["meeting"],
                                                      activation_fn=tf.nn.softmax,
                                                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                      biases_initializer=None)

            self.policy_action = slim.fully_connected(rnn_out, ACTION_SIZE["action"],
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=None)

            self.policy = tf.concat([self.policy_meeting, self.policy_action], axis=1)

            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.meetings = tf.placeholder(shape=[None], dtype=tf.int32)
                self.meeting_one_hot = tf.one_hot(self.meetings, ACTION_SIZE["meeting"], dtype=tf.float32)
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_one_hot = tf.one_hot(self.actions, ACTION_SIZE["action"], dtype=tf.float32)
                self.one_hot_concat = tf.concat([self.meeting_one_hot, self.actions_one_hot], axis=1)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.one_hot_concat, [1])

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
    def __init__(self, parent_node, meeting, action):
        self.parent_node = parent_node
        if parent_node is None:
            self.layer = 0
        else:
            self.layer = parent_node.layer + 1

        self.child_nodes = {}

        self.meeting = meeting
        self.action = action

        self.current_score = 0.0

        self.prior_value = 0.0

        self.visit_count = 0
        self.total_reward = 0.0
        self.mean_reward = 0.0

        self.rave_visit_count = 0
        self.rave_total_reward = 0.0
        self.rave_mean_reward = 0.0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Node: {}, Layer: {}, Visits: {}, Mean: {}, Child Nodes: {}"\
               .format((self.meeting, self.action),
                       self.layer,
                       self.visit_count,
                       self.mean_reward,
                       len(self.child_nodes))

class Tree:
    def __init__(self):
        self.root = Node(None, None, None)


class Worker(threading.Thread):
    def __init__(self, worker_id, mcts, env, sess, saver, global_network, trainer, model_path, global_episodes):
        threading.Thread.__init__(self)
        self.id = worker_id
        self.name = "worker_" + str(worker_id)
        self.mcts = mcts
        self.tree = mcts.tree
        self.env = env
        self.env.worker = self
        self.best_env = None
        self.sess = sess
        self.saver = saver
        self.global_network = global_network
        self.model_path = model_path
        self.trainer = trainer
        self.global_t = global_episodes
        self.episode_buffer = []
        self.episode_values = []
        self.episode_step_count = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.id))
        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(STATE_SHAPE, STATE_SIZE, ACTION_SIZE, self.name, self.trainer)
        self.update_local_ops = update_target_graph('global', self.name)

    def run(self):
        print("Starting worker " + str(self.id))

        if self.id == THREADS - 1:
            print()
            self.print_stat_headers()

        global_t = self.sess.run(self.global_t)
        with self.sess.as_default(), self.sess.graph.as_default():

            while global_t < MAX_GLOBAL_T:
                start = datetime.datetime.now()

                self.sess.run(self.update_local_ops)
                rnn_state = self.local_AC.state_init

                self.env.reset()
                current_node = self.tree.root

                has_selected = False
                done = False

                selection_step = 0
                simulation_step = 0
                rave_list = []

                state = self.env.get_state()
                while not done:
                    m_dist, a_dist, value, rnn_state = self.predict(self.sess, state, rnn_state)
                    meeting, action = self.env.validate_prediction(m_dist, a_dist)

                    if not has_selected:
                        with self.mcts.lock:
                            if not current_node.child_nodes:
                                expansion = "{}_{}".format(current_node.layer, current_node.meeting)
                                if current_node.parent_node is None:
                                    node_list = range(self.env.nr_meetings)
                                else:
                                    child_keys = list(current_node.parent_node.child_nodes.keys())
                                    node_list = [m for m in child_keys if m != current_node.meeting]

                                    if len(node_list) != (len(child_keys) - 1):
                                        print()

                                for m in node_list:
                                    if m == meeting:
                                        node_to_add = Node(current_node, meeting, action)
                                        node_to_add.prior_value = max(min(value[0, 0], 1), 0)
                                    else:
                                        random_action = random.randrange(ACTION_SIZE["action"])
                                        random_action = self.env.validate_action(m, random_action)
                                        node_to_add = Node(current_node, m, random_action)
                                        node_to_add.prior_value = random.random()

                                    self.update_tree_stats(current_node, node_to_add)
                                    current_node.child_nodes[m] = node_to_add

                                # self.mcts.graph.add_edge(current_node, node_to_add)

                                if meeting not in current_node.child_nodes:
                                    current_node = self.evaluate_nodes(list(current_node.child_nodes.values()))
                                else:
                                    current_node = current_node.child_nodes[meeting]
                                current_node.visit_count += 1

                                if current_node.meeting is None or current_node.action is None:
                                    print()

                                meeting = current_node.meeting
                                action = current_node.action
                                has_selected = True

                                selection_step += 1
                            else:
                                while len(current_node.child_nodes) + current_node.layer != self.env.nr_meetings:
                                    sleep(0.001)

                                if meeting in current_node.child_nodes:
                                    if current_node.child_nodes[meeting].action == action:
                                        current_node.child_nodes[meeting].prior_value = max(min(value[0, 0], 1), 0)

                                current_node = self.evaluate_nodes(list(current_node.child_nodes.values()))
                                current_node.visit_count += 1

                                if current_node.meeting is None or current_node.action is None:
                                    print()

                                meeting = current_node.meeting
                                action = current_node.action

                                selection_step += 1
                    else:
                        rave_list.append((meeting, action))
                        simulation_step += 1

                    self.env.take_action(meeting, action)
                    self.env.evaluate_solution()

                    reward = self.env.placed_ratio

                    new_state = self.env.get_state()

                    # if self.id == 0:
                    #     print(str(global_t).rjust(6),
                    #           str(self.id).rjust(4),
                    #           str(current_node.layer).rjust(4),
                    #           str(simulation_step).rjust(4),
                    #           str(meeting).rjust(4),
                    #           str(action).rjust(5),
                    #           "{:.4f}".format(self.env.placed_ratio).replace(".", ",").rjust(9),
                    #           str(int(self.env.pref_value)).rjust(5),
                    #           str(datetime.datetime.now() - start)[5:].rjust(10), sep="\t")

                    if not simulation_step < SIMULATION_BUDGET or current_node.layer == self.env.nr_meetings or self.env.placed_ratio == 1:
                        done = True

                    if TRAIN_ON_PLAYOUT or simulation_step == 0:
                        self.save_step_data(self.sess, state, meeting, action, reward, new_state, done, value[0, 0], rnn_state)

                    state = new_state

                evaluated_reward = self.update_nodes(current_node, rave_list, value[0, 0], reward)

                # Update the network using the experience buffer at the end of the episode.
                if self.episode_buffer:
                    v_l, p_l, e_l, g_n, v_n = self.train(self.episode_buffer, self.sess, GAMMA, 0.0)
                else:
                    v_l = p_l = e_l = g_n = v_n = 0

                    # self.episode_rewards.append(self.episode_reward)
                # self.episode_lengths.append(self.episode_step_count)
                # self.episode_mean_values.append(np.mean(self.episode_values))
                # mean_reward = np.mean(self.episode_rewards[-5:])
                # mean_length = np.mean(self.episode_lengths[-5:])
                # mean_value = np.mean(self.episode_mean_values[-5:])
                # summary = tf.Summary()
                # summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                # summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                # summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                # summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                # summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                # summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                # summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                # summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                # self.summary_writer.add_summary(summary, global_t)
                # self.summary_writer.flush()

                if self.id == 0:
                    self.sess.run(self.global_t.assign_add(1))

                if self.id == 0:
                    # nx.draw(self.mcts.graph)
                    # plt.show()

                    if global_t > 0 and global_t % 10 == 0:
                        self.mcts.best_env = self.get_most_visited_path()
                        self.print_stat_headers()

                print(str(global_t).rjust(6),
                      str(self.id).rjust(4),
                      str(self.mcts.layer_count).rjust(5),
                      str(self.mcts.node_count).rjust(8),
                      expansion.rjust(8),
                      str(len(self.mcts.best_env.placed_values)).rjust(7),
                      "{:.4f}".format(self.mcts.best_env.placed_ratio).replace(".", ",").rjust(9),
                      str(int(self.mcts.best_env.pref_value)).rjust(5),
                      "{:.4f}".format(max(min(value[0, 0], 1), 0)).replace(".", ",").rjust(9),
                      "{:.4f}".format(reward).replace(".", ",").rjust(9),
                      "{:.4f}".format(evaluated_reward).replace(".", ",").rjust(9),
                      "{:.4f}".format(p_l).replace(".", ",").rjust(9),
                      "{:.4f}".format(v_l).replace(".", ",").rjust(9),
                      "{:.4f}".format(e_l).replace(".", ",").rjust(9),
                      "{:.4f}".format(g_n).replace(".", ",").rjust(9),
                      "{:.4f}".format(v_n).replace(".", ",").rjust(9),
                      str(datetime.datetime.now() - start)[5:].rjust(10), sep="\t")

                # Periodically save model parameters, and summary statistics.
                if global_t > 0 and global_t % 100 == 0 and self.id == 0:
                    self.saver.save(self.sess, self.model_path + '/model-' + str(global_t) + '.cptk')
                    print("Saved Model")

                global_t += 1

        self.mcts.best_env = self.get_most_visited_path()


    def update_tree_stats(self, current_node, node_to_add):
        self.mcts.layer_count = max(self.mcts.layer_count, node_to_add.layer)
        self.mcts.node_count += 1
        if node_to_add.layer not in self.mcts.nodes_by_layer:
            self.mcts.nodes_by_layer[node_to_add.layer] = 0
        self.mcts.nodes_by_layer[node_to_add.layer] += 1

    def print_stat_headers(self):
        print("t".rjust(6),
              "id".rjust(4),
              "tlyr".rjust(5),
              "tnds".rjust(8),
              "cpath".rjust(8),
              "placed".rjust(7),
              "ratio".rjust(9),
              "pref".rjust(5),
              "value".rjust(9),
              "reward".rjust(9),
              "evl_rew".rjust(9),
              "policy".rjust(9),
              "value".rjust(9),
              "entropy".rjust(9),
              "grad".rjust(9),
              "var".rjust(9),
              "time".rjust(10), sep="\t")

    def predict(self, sess, s, rnn_state):
        # Take an action using probabilities from policy network output.
        m_dist, a_dist, v, rnn_state = sess.run([self.local_AC.policy_meeting, self.local_AC.policy_action, self.local_AC.value, self.local_AC.state_out],
                                        feed_dict={self.local_AC.inputs: [s],
                                                   self.local_AC.state_in[0]: rnn_state[0],
                                                   self.local_AC.state_in[1]: rnn_state[1]})

        return m_dist, a_dist, v, rnn_state

    def evaluate_nodes(self, child_nodes):
        for child_node in child_nodes:
            if child_node.visit_count == 0 or child_node.parent_node.visit_count == 0:
                beta = 1
                uct = 1000
            else:
                if child_node.rave_visit_count == 0:
                    beta = 0
                else:
                    beta = child_node.rave_visit_count / \
                           (child_node.visit_count + child_node.rave_visit_count +
                            4 * math.pow(RAVE_B, 2) * child_node.visit_count * child_node.rave_visit_count)

                uct = UCT_C * math.sqrt(math.log(child_node.parent_node.visit_count) / child_node.visit_count)

            prior = child_node.prior_value / (1 + child_node.parent_node.visit_count)

            child_node.current_score = (1 - beta) * child_node.mean_reward + beta * child_node.rave_mean_reward + prior # + uct

        child_nodes.sort(key=lambda x: x.current_score, reverse=True)
        return child_nodes[0]

    def update_nodes(self, leaf_node, rave_list, value, reward):
        value = max(min(value, 1), 0)
        evaluated_reward = (1 - LAMBDA) * value + LAMBDA * reward

        current_node = leaf_node
        while current_node.parent_node is not None:
            current_node.total_reward += evaluated_reward
            current_node.mean_reward = current_node.total_reward / current_node.visit_count
            current_node = current_node.parent_node

            if current_node.parent_node is not None:
                for sibling_node in list(current_node.parent_node.child_nodes.values()):
                    if sibling_node is not current_node:
                        if (sibling_node.meeting, sibling_node.action) in rave_list:
                            sibling_node.rave_visit_count += 1
                            sibling_node.rave_total_reward += evaluated_reward
                            sibling_node.rave_mean_reward = sibling_node.rave_total_reward / sibling_node.rave_visit_count

        return evaluated_reward

    def get_most_visited_path(self):
        self.env.reset()
        current_node = self.tree.root

        while current_node.child_nodes:
            node_list = list(current_node.child_nodes.values())
            node_list = sorted(node_list, key=lambda x: (x.visit_count, x.mean_reward), reverse=True)
            # visit_count = np.array([child_node.visit_count for child_node in node_list])
            # n = np.random.choice(np.where(visit_count == visit_count.max())[0])
            # most_visited_child_node = node_list[n]
            current_node = node_list[0]
            self.env.take_action(current_node.meeting, current_node.action)

        self.env.evaluate_solution()
        return copy(self.env)

    def save_step_data(self, sess, state, meeting, action, reward, new_state, done, value, rnn_state):
        self.episode_buffer.append([state, meeting, action, reward, new_state, done, value])
        self.episode_values.append(value)

        self.episode_reward += reward
        self.episode_step_count += 1

        # If the episode hasn't ended, but the experience buffer is full, then we
        # make an update step using that experience rollout.
        if len(self.episode_buffer) >= BATCH_SIZE and not done:
            # Since we don't know what the true final return is, we "bootstrap" from our current
            # value estimation.
            v1 = sess.run(self.local_AC.value,
                          feed_dict={self.local_AC.inputs: [state],
                                     self.local_AC.state_in[0]: rnn_state[0],
                                     self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
            v_l, p_l, e_l, g_n, v_n = self.train(self.episode_buffer, sess, GAMMA, v1)
            self.episode_buffer = []
            sess.run(self.update_local_ops)

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        for idx in range(len(observations)):
            observations[idx] = np.reshape(observations[idx], [1, *STATE_SHAPE])
        meetings = rollout[:, 1]
        actions = rollout[:, 2]
        rewards = rollout[:, 3]
        next_observations = rollout[:, 4]
        values = rollout[:, 6]

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
                     self.local_AC.meetings: meetings,
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

        state = np.zeros_like(self.calendar, dtype=np.float64)
        for meeting in range(self.nr_meetings):
            domain = self.domains_as_calendar[meeting]
            meeting_evaluation = np.zeros_like(state, dtype=np.float64)
            meetings[meeting] = np.sum(domain) / np.size(domain)

            if meeting in self.placed_values:
                meetings[meeting] -= 1

                idx_room = self.placed_values[meeting].indices[0]
                idx_week = self.placed_values[meeting].indices[1]
                idx_day = self.placed_values[meeting].indices[2]
                idx_slot_start = self.placed_values[meeting].indices[3]
                idx_slot_end = idx_slot_start + self.durations_in_slots[meeting] + 1

                meeting_evaluation[idx_room, idx_week, idx_day, idx_slot_start:idx_slot_end] = meetings[meeting]

            else:
                meeting_evaluation[domain == 1] = meetings[meeting]

            state += meeting_evaluation

        # m_random = np.random.random(self.nr_meetings)
        # m_max_shuffle = np.where(meetings == np.max(meetings), m_random, 0)
        # meeting = np.argmax(m_max_shuffle)

        state -= np.mean(state)
        state /= np.std(state)

        if state.shape != STATE_SHAPE:
            state = np.reshape(state, STATE_SHAPE)

        return state

    def validate_prediction(self, m_dist, a_dist):
        meeting_dist = list(np.argsort(m_dist)[0])
        meeting_dist = list(reversed(meeting_dist))

        action_dist = list(np.argsort(a_dist)[0])
        action_dist = list(reversed(action_dist))

        for meeting in meeting_dist:
            domain = self.domains_as_calendar[meeting].ravel()
            if np.sum(domain) == 0:
                continue
            for action in action_dist:
                if domain[action] == 1:
                    return meeting, action

        print()

    def validate_action(self, meeting, action):
        # idx = np.random.choice(predictions[0], p=predictions[0])
        # idx = np.argmax(predictions == idx)

        domain = self.domains_as_calendar[meeting].ravel()
        assert np.sum(domain) > 0, "Unable to validate action for meeting {}, empty domain".format(meeting)

        if domain[action] == 1:
            return action
        else:
            possible_actions = np.nonzero(domain)[0]
            earlier_actions = possible_actions[possible_actions < action]
            later_actions = possible_actions[possible_actions > action]

            if len(later_actions) == 0:
                return earlier_actions[-1]
            elif len(earlier_actions) == 0:
                return later_actions[0]
            else:
                if (action - earlier_actions[-1]) <= (later_actions[0] - action):
                    return earlier_actions[-1]
                else:
                    return later_actions[0]


    def take_action(self, meeting, action):
        indices = self.int_to_indices[action]
        duration_in_slots = self.durations_in_slots[meeting]
        value = Value(meeting, indices, duration_in_slots)

        for slot in range(value.start, max(value.end + 1, self.calendar.shape[3])):
            conflict = self.calendar[value.room, value.week, value.day, slot]
            if conflict > 0:
                self.remove_meeting(conflict)

            for room in range(self.calendar.shape[0]):
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

    def reset(self):
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
        self.lock = threading.RLock()
        self.tree = Tree()
        # self.graph = nx.DiGraph()
        self.best_env = None
        self.layer_count = 0
        self.node_count = 0
        self.nodes_by_layer = {0: 0}
        self.saved_solutions = {}

    def run(self, env, state_shape, state_size, action_size):
        self.best_env = env

        tf.reset_default_graph()

        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)

        with tf.Session() as sess:
            global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
            master_network = AC_Network(state_shape, state_size, action_size, 'global', None)  # Generate global network
            saver = tf.train.Saver(max_to_keep=5)
            workers = [Worker(i, self, deepcopy(env), sess, saver, master_network, trainer, MODEL_PATH, global_episodes)
                       for i in range(THREADS)]
            if LOAD_MODEL == True:
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            for worker in workers:
                worker.start()

            for worker in workers:
                worker.join()

        self.saved_solutions[0] = self.best_env