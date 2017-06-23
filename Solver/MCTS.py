import datetime
import threading
import multiprocessing
from random import random

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from copy import copy, deepcopy
import os
from time import sleep

from Problem.IRMCSP import Value

MAX_GLOBAL_T = 1e5
SIMULATION_BUDGET = 250

THREADS = 1
# THREADS = multiprocessing.cpu_count()

LOAD_MODEL = False
TRAIN_ON_PLAYOUT = False

BATCH_SIZE = 16

GAMMA = 0.99  # discount rate for advantage estimation and reward discounting
LAMBDA = 0.75

LSTM_SIZE = 128
CLIP_BY_NORM = 5

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
STATE_SHAPE = None

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
    def __init__(self, state_shape, s_size, a_size, scope, trainer):
        global STATE_SHAPE
        global STATE_SIZE
        global ACTION_SIZE

        STATE_SHAPE = state_shape
        STATE_SIZE = s_size
        ACTION_SIZE = a_size

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

            # hidden = slim.fully_connected(slim.flatten(self.sequences), LSTM_SIZE, activation_fn=tf.nn.relu)

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
            self.policy = slim.fully_connected(rnn_out, ACTION_SIZE,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
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
    def __init__(self, parent_node, meeting, action):
        self.parent_node = parent_node
        if parent_node is None:
            self.layer = 0
        else:
            self.layer = parent_node.layer + 1

        self.child_nodes = {}

        self.meeting = meeting
        self.action = action

        self.visit_count = 0
        self.prior_value = 0.0
        self.total_reward = 0.0
        self.mean_reward = 0.0


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
        self.total_steps = 0
        self.episode_buffer = []
        self.episode_values = []
        self.episode_step_count = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.id))

        # self.max_stagnation = 0

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(STATE_SHAPE, STATE_SIZE, ACTION_SIZE, self.name, self.trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = self.actions = np.identity(ACTION_SIZE, dtype=bool).tolist()

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

                train = True
                has_selected = False
                done = False

                simulation_step = 0

                meeting, state = self.env.get_state()
                while not done:
                    action, value, rnn_state = self.predict(self.sess, state, rnn_state)
                    action = self.env.validate_action(meeting, action)

                    if not has_selected:
                        if current_node.child_nodes:
                            if (meeting, action) not in current_node.child_nodes:
                                node_to_add = Node(current_node, meeting, action)
                                node_to_add.prior_value = value[0, 0]
                                self.update_tree_stats(current_node, node_to_add)
                                current_node.child_nodes[(meeting, action)] = node_to_add
                            else:
                                node_to_update = current_node.child_nodes[(meeting, action)]
                                node_to_update.prior_value = value[0, 0]

                            best_child_node = self.evaluate_child_nodes(current_node)
                            current_node = best_child_node
                            meeting = best_child_node.meeting
                            action = best_child_node.action
                        else:
                            node_to_add = Node(current_node, meeting, action)
                            node_to_add.prior_value = value[0, 0]
                            self.update_tree_stats(current_node, node_to_add)
                            current_node.child_nodes[(meeting, action)] = node_to_add
                            current_node = node_to_add
                            has_selected = True
                    else:
                        simulation_step += 1

                    self.env.take_action(meeting, action)
                    self.env.evaluate_solution()

                    reward = self.env.placed_ratio

                    new_meeting, new_state = self.env.get_state()

                    if not simulation_step < SIMULATION_BUDGET or self.env.placed_ratio == 1:
                        done = True

                    if TRAIN_ON_PLAYOUT or simulation_step == 0:
                        self.save_step_data(self.sess, state, action, reward, new_state, done, value[0, 0], rnn_state)


                    state = new_state
                    meeting = new_meeting


                self.update_nodes(current_node, value[0, 0], reward)

                # Update the network using the experience buffer at the end of the episode.
                if self.episode_buffer:
                    v_l, p_l, e_l, g_n, v_n = self.train(self.episode_buffer, self.sess, GAMMA, 0.0)

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
                    if global_t % 10 == 0:
                        self.mcts.best_env = self.get_most_visited_path()
                print(str(global_t).rjust(6),
                      str(self.id).rjust(4),
                      str(self.mcts.layer_count).rjust(5),
                      str(self.mcts.node_count).rjust(8),
                      str(current_node.layer).rjust(5),
                      str(self.mcts.nodes_by_layer[current_node.layer]).rjust(8),
                      "{:.4f}".format(self.mcts.best_env.placed_ratio).replace(".", ",").rjust(9),
                      str(int(self.mcts.best_env.pref_value)).rjust(5),
                      "{:.4f}".format(p_l).replace(".", ",").rjust(9),
                      "{:.4f}".format(v_l).replace(".", ",").rjust(9),
                      "{:.4f}".format(e_l).replace(".", ",").rjust(9),
                      "{:.4f}".format(g_n).replace(".", ",").rjust(9),
                      "{:.4f}".format(v_n).replace(".", ",").rjust(9),
                      str(datetime.datetime.now() - start)[5:].rjust(10), sep="\t")

                # Periodically save model parameters, and summary statistics.
                if global_t > 0 and global_t % 100 == 0 and self.id == 0:
                    # self.saver.save(self.sess, self.model_path + '/model-' + str(global_t) + '.cptk')
                    # print("Saved Model")
                    self.print_stat_headers()
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
              "clyr".rjust(5),
              "clyrnds".rjust(8),
              "ratio".rjust(9),
              "pref".rjust(5),
              "policy".rjust(9),
              "value".rjust(9),
              "entropy".rjust(9),
              "grad".rjust(9),
              "var".rjust(9),
              "time".rjust(10), sep="\t")

    def predict(self, sess, s, rnn_state):
        # Take an action using probabilities from policy network output.
        a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                        feed_dict={self.local_AC.inputs: [s],
                                                   self.local_AC.state_in[0]: rnn_state[0],
                                                   self.local_AC.state_in[1]: rnn_state[1]})

        a = np.random.choice(np.where(a_dist[0] == a_dist[0].max())[0])

        # a = np.random.choice(a_dist[0], p=a_dist[0])
        # a = np.argmax(a_dist == a)

        return a, v, rnn_state

    def evaluate_child_nodes(self, parent_node):
        evaluated_child_nodes = []
        for child_node in list(parent_node.child_nodes.values()):
            score = child_node.mean_reward + (child_node.prior_value / (1 + parent_node.visit_count))
            evaluated_child_nodes.append({"node":child_node, "score": score})
        evaluated_child_nodes.sort(key=lambda x: x["score"], reverse=True)
        return evaluated_child_nodes[0]["node"]

    def update_nodes(self, leaf_node, value, reward):
        current_node = leaf_node
        evaluated_reward = (1 - LAMBDA) * value + LAMBDA * reward
        while current_node.parent_node is not None:
            current_node.visit_count += 1
            current_node.total_reward += evaluated_reward
            current_node.mean_reward = current_node.total_reward / current_node.visit_count
            current_node = current_node.parent_node

    def get_most_visited_path(self):
        self.env.reset()
        current_node = self.tree.root

        while current_node.child_nodes:
            node_list = list(current_node.child_nodes.values())
            visit_count = np.array([child_node.visit_count for child_node in node_list])
            n = np.random.choice(np.where(visit_count == visit_count.max())[0])
            most_visited_child_node = node_list[n]
            current_node = most_visited_child_node
            self.env.take_action(current_node.meeting, current_node.action)

        self.env.evaluate_solution()
        return copy(self.env)

    def save_step_data(self, sess, state, action, reward, new_state, done, value, rnn_state):
        self.episode_buffer.append([state, action, reward, new_state, done, value])
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

        s = np.zeros_like(self.calendar, dtype=np.float64)
        for m in range(self.nr_meetings):
            d = self.domains_as_calendar[m]
            h = np.zeros_like(s, dtype=np.float64)
            meetings[m] = np.sum(d) / np.size(d)

            if m in self.placed_values:
                meetings[m] -= 1

                idx_room = self.placed_values[m].indices[0]
                idx_week = self.placed_values[m].indices[1]
                idx_day = self.placed_values[m].indices[2]
                idx_slot_start = self.placed_values[m].indices[3]
                idx_slot_end = idx_slot_start + self.durations_in_slots[m] + 1

                h[idx_room, idx_week, idx_day, idx_slot_start:idx_slot_end] = meetings[m]

            else:
                h[h == 1] = meetings[m]

            s += h

        m_random = np.random.random(self.nr_meetings)
        m_max_shuffle = np.where(meetings == np.max(meetings), m_random, 0)
        m = np.argmax(m_max_shuffle)

        # s -= np.mean(s)
        # s /= np.std(s)

        if s.shape != STATE_SHAPE:
            s = np.reshape(s, STATE_SHAPE)

        return m, s

    def validate_action(self, meeting, action):
        domain = self.domains_as_calendar[meeting]
        assert np.sum(domain) > 0, "Unable to validate action for meeting {}, empty domain".format(meeting)

        if domain[self.int_to_indices[action]] == 1:
            return action
        else:
            action_inc = action + 1
            action_dec = action - 1
            while True:
                if action_inc < ACTION_SIZE:
                    if domain[self.int_to_indices[action_inc]] == 1:
                        return action_inc
                    else:
                        action_inc += 1
                elif action_dec >= 0:
                    if domain[self.int_to_indices[action_dec]] == 1:
                        return action_dec
                    else:
                        action_dec -= 1

    def take_action(self, meeting, action):
        indices = self.int_to_indices[action]
        duration_in_slots = self.durations_in_slots[meeting]
        value = Value(meeting, indices, duration_in_slots)

        for slot in range(value.start, value.end + 1):
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
        self.tree = Tree()
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