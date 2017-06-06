# iRM CSP OR+ML Hybrid Proof of Concept; based on:
#
# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
#
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import tensorflow as tf
import numpy as np

import copy, datetime, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K

from IRMCSP import Value

# -- constants


MAX_GLOBAL_T = 1e5

THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

N_STEP_RETURN = 8

GAMMA = 0.99
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START_RANGE = [0.4, 0.3, 0.3]
# EPS_START_RANGE = [0.99, 0.95, 0.9]
EPS_STOP_RANGE = [0.1, 0.01, 0.5]
EPS_STEPS = 1e5

MAX_STAGNATION_RANGE = range(7, 10)
SOLUTION__FIRST_CRITERION = "ratio"

MIN_BATCH = 64

LR_LOW = 1e-6
LR_HIGH = 1e-4
LR_DECAY_RATE = 0.96
LR_DECAY_STEP = 5000

CLIP_BY_NORM = 0.0

BETA_1 = 0.9
BETA_2 = 0.999

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient

STATE_SHAPE = None
BATCH_SHAPE = None
NONE_STATE = None
NUM_MEETINGS = 0
NUM_CALENDAR = 0

SUMMARY_DIR = "./Summaries"

# ---------
class Brain:
    train_queue = [[], [], [], [], [], []]  # s, m, c, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self, state_shape, num_meetings, num_calendar):
        global STATE_SHAPE
        global BATCH_SHAPE
        global NONE_STATE
        global NUM_MEETINGS
        global NUM_CALENDAR

        STATE_SHAPE = state_shape
        BATCH_SHAPE = [MIN_BATCH, *STATE_SHAPE]
        NUM_MEETINGS = num_meetings
        NUM_CALENDAR = num_calendar
        NONE_STATE = np.zeros(STATE_SHAPE)

        self.saved_solutions = {}
        self.global_best_solution = 0

        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)
        self.tr_loss = None

        self.not_enough_optimizers = False

        self.merged_summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(SUMMARY_DIR + '/train', self.session.graph)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()  # avoid modifications

    def _build_model(self):
        input_layer = Input(shape=STATE_SHAPE,
                            name="input")

        conv0 = Convolution2D(filters=1,
                              kernel_size=(1, 1),
                              strides=(1, 1),
                              padding="same",
                              data_format="channels_last",
                              activation="relu",
                              use_bias=True,
                              kernel_initializer='glorot_normal',
                              bias_initializer='zeros',
                              name="conv0")(input_layer)

        conv1 = Convolution2D(filters=1,
                              kernel_size=(16, 1),
                              strides=(1, 1),
                              padding="same",
                              data_format="channels_last",
                              activation="relu",
                              use_bias=True,
                              kernel_initializer='glorot_normal',
                              bias_initializer='zeros',
                              name="conv1")(conv0)

        conv2 = Convolution2D(filters=1,
                              kernel_size=(12, 1),
                              strides=(1, 1),
                              padding="same",
                              data_format="channels_last",
                              activation="relu",
                              use_bias=True,
                              kernel_initializer='glorot_normal',
                              bias_initializer='zeros',
                              name="conv2")(conv1)

        conv3 = Convolution2D(filters=1,
                              kernel_size=(8, 1),
                              strides=(1, 1),
                              padding="same",
                              data_format="channels_last",
                              activation="relu",
                              use_bias=True,
                              kernel_initializer='glorot_normal',
                              bias_initializer='zeros',
                              name="conv3")(conv2)

        conv_flat = Flatten()(conv3)

        dense = Dense(units=128,
                            activation='relu',
                            use_bias=True,
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros',
                            name="dense")(conv_flat)

        out_meetings = Dense(units=NUM_MEETINGS,
                            activation='softmax',
                            use_bias=True,
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros',
                            name="out_meetings")(dense)

        out_calendar = Dense(units=NUM_CALENDAR,
                            activation='softmax',
                            use_bias=True,
                            kernel_initializer='glorot_normal',
                            bias_initializer='zeros',
                            name="out_calendar")(dense)

        out_value = Dense(units=1,
                          activation='linear',
                          use_bias=True,
                          kernel_initializer='glorot_normal',
                          bias_initializer='zeros',
                          name="out_value")(dense)

        model = Model(inputs=[input_layer], outputs=[out_meetings, out_calendar, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, *STATE_SHAPE))
        m_t = tf.placeholder(tf.float32, shape=(None, NUM_MEETINGS))
        c_t = tf.placeholder(tf.float32, shape=(None, NUM_CALENDAR))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p_m, p_c, v = model(s_t)

        advantage = r_t - v

        log_prob_m = tf.log(tf.reduce_sum(p_m * m_t, axis=1, keep_dims=True) + 1e-10)
        log_prob_c = tf.log(tf.reduce_sum(p_c * c_t, axis=1, keep_dims=True) + 1e-10)

        loss_policy_m = - log_prob_m * tf.stop_gradient(advantage)  # maximize policy
        loss_policy_c = - log_prob_c * tf.stop_gradient(advantage)  # maximize policy

        loss_value = LOSS_V * tf.square(advantage)  # minimize value error

        entropy_m = LOSS_ENTROPY * tf.reduce_sum(p_m * tf.log(p_m + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)
        entropy_c = LOSS_ENTROPY * tf.reduce_sum(p_c * tf.log(p_c + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy_m + loss_policy_c + loss_value + entropy_m + entropy_c)

        # tf.summary.scalar("policy loss", loss_policy)
        # tf.summary.scalar("value_loss", loss_value)
        # tf.summary.scalar("entropy", entropy)
        # tf.summary.scalar("total_loss", loss_total)
        #
        # self.tr_loss = loss_policy

        global_step = tf.Variable(0, trainable=False)
        lr = np.exp(random.uniform(np.log(LR_LOW), np.log(LR_HIGH)))
        learning_rate = tf.train.exponential_decay(lr, global_step, LR_DECAY_STEP, LR_DECAY_RATE, staircase=False)
        print()
        print("learning rate: ", "{: .2e}".format(lr).replace(".", ","))

        optimizer = tf.train.AdamOptimizer(learning_rate, BETA_1, BETA_2)
        grad_and_vars = optimizer.compute_gradients(loss_total)
        if CLIP_BY_NORM != 0:
            grad_and_vars = [(tf.clip_by_norm(grad, CLIP_BY_NORM), var) for grad, var in grad_and_vars]
        minimize = optimizer.apply_gradients(grad_and_vars, global_step=global_step)

        return s_t, m_t, c_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            s, m, c, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], [], []]

        s = np.vstack(s)
        m = np.vstack(m)
        c = np.vstack(c)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH:
            # print("Optimizer alert! Minimizing batch of %d" % len(s))
            self.not_enough_optimizers = True

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, m_t, c_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, m_t: m, c_t: c, r_t: r})
        # summary = self.session.run(self.merged_summaries, feed_dict={s_t: s, a_t: a, r_t: r})
        # self.train_writer.add_summary(summary)

    def train_push(self, s, m, c, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(m)
            self.train_queue[2].append(c)
            self.train_queue[3].append(r)

            if s_ is None:
                self.train_queue[4].append(NONE_STATE)
                self.train_queue[5].append(0.)
            else:
                self.train_queue[4].append(s_)
                self.train_queue[5].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p_m, p_c, v = self.model.predict(s)
            return p_m, p_c, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p_m, p_c, v = self.model.predict(s)
            return p_m, p_c

    def predict_p_m(self, s):
        with self.default_graph.as_default():
            p_m, p_c, v = self.model.predict(s)
            return p_m

    def predict_p_c(self, s):
        with self.default_graph.as_default():
            p_m, p_c, v = self.model.predict(s)
            return p_c

    def predict_v(self, s):
        with self.default_graph.as_default():
            p_m, p_c, v = self.model.predict(s)
            return v


# ---------
global_t = 0


class Agent:
    def __init__(self, brain, solution, eps_start, eps_end, eps_steps):
        self.brain = brain
        self.solution = solution

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        self.memory = []  # used for n_step return
        self.R = 0.0

    def getEpsilon(self):
        if global_t >= self.eps_steps:
            return self.eps_end
        else:
            return self.eps_start + global_t * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s):
        global global_t
        eps = self.getEpsilon()
        global_t = global_t + 1

        p_m, p_c = self.brain.predict_p(s)

        if random.random() < eps:
            m = random.randrange(0, NUM_MEETINGS)
            d = np.ravel(self.solution.domains[m])
            c = random.randrange(0, len(d))

        else:
            max_m_value = np.max(p_m)
            m_max = np.where(p_m == max_m_value, random.random(), 0)
            m = np.argmax(m_max)

            d = np.ravel(self.solution.domains[m])
            c = np.argmax(p_c * d)

        if d[c] == 1:
            indices = self.solution.int_to_indices[c]
            duration = self.solution.meeting_durations[m]
            m_v = Value(m, c, indices, duration, self.solution.time_by_indices)
        else:
            m_v = None

        return m, c, m_v

    def train(self, s, m, c, r, s_):

        def get_sample(memory, n):
            s, m, c, _, _ = memory[0]
            _, _, _, _, s_ = memory[n - 1]

            return s, m, c, self.R, s_

        m_hot = np.zeros(NUM_MEETINGS)  # turn action into one-hot representation
        m_hot[m] = 1

        c_hot = np.zeros(NUM_CALENDAR)  # turn action into one-hot representation
        c_hot[c] = 1

        self.memory.append((s, m_hot, c_hot, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, m, c, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, m, c, r, s_)

                self.R = (self.R - self.memory[0][3]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, m, c, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, m, c, r, s_)

            self.R = self.R - self.memory[0][3]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect


# ---------
class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, brain, solution, id, render=False, eps_start=EPS_START_RANGE, eps_end=EPS_STOP_RANGE, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)

        self.brain = brain
        self.id = id
        self.agent_t = 0

        self.current_solution = solution
        self.best_solution = copy.copy(solution)
        self.brain.saved_solutions[self.id] = copy.copy(solution)

        self.max_stagnation = 0

        eps_rnd = random.choice(range(len(eps_start)))
        self.agent = Agent(self.brain, self.current_solution, eps_start[eps_rnd], eps_end[eps_rnd], eps_steps)

        self.render = render

    def runEpisode(self):
        done = False
        self.max_stagnation = 2 ** random.choice(MAX_STAGNATION_RANGE)
        s = self.current_solution.get_state()
        R = 0

        while True:
            time.sleep(THREAD_DELAY)  # yield

            # if self.render: self.env.render()

            start = datetime.datetime.now()
            self.agent_t += 1
            self.current_solution.iteration += 1
            m, c, m_v = self.agent.act(s)

            if m_v is None:
                r = -1
                self.current_solution.stagnation += 1
            else:
                self.current_solution.take_action(m, m_v)
                self.current_solution.evaluate_solution()

                if self.current_solution.is_better_solution(self.best_solution, SOLUTION__FIRST_CRITERION):
                    r = 1  # self.current_solution.placed_ratio
                    self.current_solution.stagnation = 0
                    self.best_solution = copy.copy(self.current_solution)
                    if self.best_solution.is_better_solution(self.brain.saved_solutions[self.id], SOLUTION__FIRST_CRITERION):
                        self.brain.saved_solutions[self.id] = copy.copy(self.best_solution)
                        if self.brain.saved_solutions[self.id].is_better_solution(self.brain.saved_solutions[self.brain.global_best_solution], SOLUTION__FIRST_CRITERION):
                            self.brain.global_best_solution = self.id
                else:
                    r = 0
                    self.current_solution.stagnation += 1

            s_ = self.current_solution.get_state()

            self.agent.train(s, m, c, r, s_)

            if self.agent_t % N_STEP_RETURN == 0:
                self.print_episode_stats(start)

                done = self.current_solution.is_stagnant(self.max_stagnation)

                if done:  # terminal state
                    s_ = None

            s = s_
            R += r

            if done:
                self.current_solution.reset_solution()
                self.best_solution = copy.copy(self.current_solution)

            if done or self.stop_signal:
                break

    def run(self):
        if self.id == 0:
            print()
            print("glb_t".rjust(6),
                  "ag".rjust(4),
                  "ag_t".rjust(5),
                  "iter".rjust(5),
                  "stg".rjust(4),
                  "b_r".rjust(7),
                  "b_v".rjust(4),
                  "p_r".rjust(7),
                  "p_v".rjust(4),
                  "g_r".rjust(7),
                  "g_v".rjust(4),
                  "time".rjust(10),
                  sep="\t")

        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True

    def print_episode_stats(self, start):
        print(str(global_t).rjust(6),
              str(self.id).rjust(4),
              str(self.agent_t).rjust(5),
              str(self.current_solution.iteration).rjust(5),
              str(self.current_solution.stagnation).rjust(4),
              "{:.4f}".format(self.current_solution.placed_ratio).replace(".", ",").rjust(7),
              str(self.current_solution.pref_value).rjust(4),
              "{:.4f}".format(self.brain.saved_solutions[self.id].placed_ratio).replace(".", ",").rjust(7),
              str(self.brain.saved_solutions[self.id].pref_value).rjust(4),
              "{:.4f}".format(self.brain.saved_solutions[self.brain.global_best_solution].placed_ratio).replace(".", ",").rjust(7),
              str(self.brain.saved_solutions[self.brain.global_best_solution].pref_value).rjust(4),
              str(datetime.datetime.now() - start)[5:].rjust(10), sep="\t")


# ---------
class Optimizer(threading.Thread):
    stop_signal = False
    brain = None

    def __init__(self, brain):
        self.brain = brain
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            self.brain.optimize()

    def stop(self):
        self.stop_signal = True
