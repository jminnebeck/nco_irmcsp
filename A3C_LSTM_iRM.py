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

import copy, math, datetime, time, random, threading

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

from IRMCSP import Value

# -- constants
MAX_GLOBAL_T = 1e5

THREADS = 12
OPTIMIZERS = 2
THREAD_DELAY = 0.001

NUM_LSTM_CELLS = 128

global_t = 0
N_STEP_RETURN = 8

GAMMA = 0.99
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START_RANGE = [0.4, 0.3, 0.3]
EPS_STOP_RANGE = [0.1, 0.01, 0.5]

# EPS_START_RANGE = [0]
# EPS_STOP_RANGE = [0]

EPS_STEPS = MAX_GLOBAL_T

MAX_STAGNATION_RANGE = [int(x * MAX_GLOBAL_T) for x in [0.01, 0.05, 0.10, 0.20, 0.25, 0.33, 0.50]]

SOLUTION_FIRST_CRITERION = "ratio"
# SOLUTION_FIRST_CRITERION = "value"

REPORT_LEVEL = "learning"
# REPORT_LEVEL = "episodes"
# REPORT_LEVEL = "steps"

MIN_BATCH = 32

LR_LOW = 1e-6
LR_HIGH = 1e-5
LR_DECAY_RATE = 0.96
LR_DECAY_STEP = 1000

CLIP_BY_NORM = 0

BETA_1 = 0.9
BETA_2 = 0.999

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient

STATE_SHAPE = None
BATCH_SHAPE = None
NONE_STATE = None
NUM_OUTPUTS = None

SUMMARY_DIR = "./Summaries"


# ---------
class Brain:
    train_queue = [[], [], [], [], [], [], [], []]  # s, r_c, w_, d_c, s_c, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self, state_shape, num_outputs):
        global STATE_SHAPE
        global BATCH_SHAPE
        global NONE_STATE
        global NUM_OUTPUTS

        STATE_SHAPE = state_shape
        BATCH_SHAPE = [MIN_BATCH, *STATE_SHAPE]
        NUM_OUTPUTS = num_outputs
        NONE_STATE = np.zeros_like(STATE_SHAPE)

        self.global_t = 0
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
        self.train_writer = tf.summary.FileWriter(SUMMARY_DIR + '/train')

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()  # avoid modifications

        if REPORT_LEVEL == "steps" or REPORT_LEVEL == "episodes":
            print("\nglb_t".rjust(6),
                  "ag".rjust(4),
                  "ag_t".rjust(5),
                  "iter".rjust(5),
                  "stg".rjust(4),
                  "b_ratio".rjust(9),
                  "b_val".rjust(5),
                  "p_ratio".rjust(9),
                  "p_val".rjust(5),
                  "g_ratio".rjust(9),
                  "g_val".rjust(5),
                  "time".rjust(10),
                  sep="\t")

        elif REPORT_LEVEL == "learning":
            print("\nglb_t".rjust(6),
                  "batch".rjust(6),
                  "queue".rjust(6),
                  "policy".rjust(10),
                  "value".rjust(10),
                  "entropy".rjust(10),
                  "total".rjust(10),
                  "g_ratio".rjust(9),
                  "g_val".rjust(5),
                  sep="\t")

    def _build_model(self):
        input_layer = Input(STATE_SHAPE)

        conv_input = Conv2D(filters=NUM_LSTM_CELLS,
                      kernel_size=(STATE_SHAPE[0], STATE_SHAPE[1]),
                      activation="relu")(input_layer)

        conv_reshaped = Reshape((NUM_LSTM_CELLS, 1))(conv_input)

        lstm_in = LSTM(units=NUM_LSTM_CELLS,
                       return_sequences=True)(conv_reshaped)
        lstm_2 = LSTM(units=NUM_LSTM_CELLS,
                      return_sequences=True)(lstm_in)
        lstm_out = LSTM(units=NUM_LSTM_CELLS)(lstm_2)

        lstm_out_reshaped = Reshape((NUM_LSTM_CELLS, 1, 1))(lstm_out)

        conv_rooms = Conv2D(filters=NUM_OUTPUTS["rooms"],
                      kernel_size=(NUM_LSTM_CELLS, 1),
                      activation='softmax')(lstm_out_reshaped)
        out_rooms = Reshape((NUM_OUTPUTS["rooms"],))(conv_rooms)

        conv_weeks = Conv2D(filters=NUM_OUTPUTS["weeks"],
                            kernel_size=(NUM_LSTM_CELLS, 1),
                            activation='softmax')(lstm_out_reshaped)
        out_weeks = Reshape((NUM_OUTPUTS["weeks"],))(conv_weeks)

        conv_days = Conv2D(filters=NUM_OUTPUTS["days"],
                            kernel_size=(NUM_LSTM_CELLS, 1),
                            activation='softmax')(lstm_out_reshaped)
        out_days = Reshape((NUM_OUTPUTS["days"],))(conv_days)

        conv_slots = Conv2D(filters=NUM_OUTPUTS["slots"],
                            kernel_size=(NUM_LSTM_CELLS, 1),
                            activation='softmax')(lstm_out_reshaped)
        out_slots = Reshape((NUM_OUTPUTS["slots"],))(conv_slots)

        # out_rooms = Dense(units=NUM_OUTPUTS["rooms"], activation='softmax')(lstm_out)
        # out_weeks = Dense(units=NUM_OUTPUTS["weeks"], activation='softmax')(lstm_out)
        # out_days = Dense(units=NUM_OUTPUTS["days"], activation='softmax')(lstm_out)
        # out_slots = Dense(units=NUM_OUTPUTS["slots"], activation='softmax')(lstm_out)

        out_value = Dense(units=1,
                          activation='linear')(lstm_out)

        model = Model(inputs=[input_layer], outputs=[out_rooms, out_weeks, out_days, out_slots, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        state_t = tf.placeholder(tf.float32, shape=(None, *STATE_SHAPE))
        rooms_t = tf.placeholder(tf.float32, shape=(None, NUM_OUTPUTS["rooms"]))
        weeks_t = tf.placeholder(tf.float32, shape=(None, NUM_OUTPUTS["weeks"]))
        days_t = tf.placeholder(tf.float32, shape=(None, NUM_OUTPUTS["days"]))
        slots_t = tf.placeholder(tf.float32, shape=(None, NUM_OUTPUTS["slots"]))
        reward_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p_rooms, p_weeks, p_days, p_slots, value = model(state_t)

        advantage = reward_t - value

        log_prob_rooms = tf.log(tf.reduce_sum(p_rooms * rooms_t, axis=1, keep_dims=True) + 1e-10)
        log_prob_weeks = tf.log(tf.reduce_sum(p_weeks * weeks_t, axis=1, keep_dims=True) + 1e-10)
        log_prob_days = tf.log(tf.reduce_sum(p_days * days_t, axis=1, keep_dims=True) + 1e-10)
        log_prob_slots = tf.log(tf.reduce_sum(p_slots * slots_t, axis=1, keep_dims=True) + 1e-10)

        # maximize policy
        loss_policy_rooms = - log_prob_rooms * tf.stop_gradient(advantage)
        loss_policy_weeks = - log_prob_weeks * tf.stop_gradient(advantage)
        loss_policy_days = - log_prob_days * tf.stop_gradient(advantage)
        loss_policy_slots = - log_prob_slots * tf.stop_gradient(advantage)
        loss_policy_total = tf.reduce_mean(loss_policy_rooms + loss_policy_weeks + loss_policy_days + loss_policy_slots)

        # minimize value error
        loss_value = tf.reduce_mean(LOSS_V * tf.square(advantage))

        # maximize entropy (regularization)
        entropy_rooms = tf.reduce_sum(p_rooms * tf.log(p_rooms + 1e-10), axis=1, keep_dims=True)
        entropy_weeks = tf.reduce_sum(p_rooms * tf.log(p_rooms + 1e-10), axis=1, keep_dims=True)
        entropy_days = tf.reduce_sum(p_rooms * tf.log(p_rooms + 1e-10), axis=1, keep_dims=True)
        entropy_slots = tf.reduce_sum(p_rooms * tf.log(p_rooms + 1e-10), axis=1, keep_dims=True)
        entropy_total = tf.reduce_mean(LOSS_ENTROPY * (entropy_rooms + entropy_weeks + entropy_days + entropy_slots))

        loss_total = tf.reduce_mean(loss_policy_total + loss_value + entropy_total)

        tf.summary.scalar(loss_total.op.name, loss_total)

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

        return state_t, rooms_t, weeks_t, days_t, slots_t, reward_t, loss_policy_total, loss_value, entropy_total, loss_total, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(THREAD_DELAY)  # yield
            return 0, 0, 0, 0, 0, 0

        with self.lock_queue:
            s, r_c, w_c, d_c, s_c, r, s_, s_mask = self.train_queue[:MIN_BATCH]
            del self.train_queue[:MIN_BATCH]
            if not self.train_queue:
                self.train_queue = [[], [], [], [], [], [], [], []]
            batch_length = len(s)
            queue_length = len(self.train_queue[0])


        s = np.vstack(s)
        r_c = np.vstack(r_c)
        w_c = np.vstack(w_c)
        d_c = np.vstack(d_c)
        s_c = np.vstack(s_c)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(self.train_queue[0]) > MIN_BATCH:
            # print("Optimizer alert! Minimizing batch of %d" % len(s))
            self.not_enough_optimizers = True

        v = self.predict_v(s_)
        r += GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        state_t, rooms_t, weeks_t, days_t, slots_t, reward_t, \
        loss_policy_total, loss_value, entropy_total, loss_total, minimize = self.graph

        loss_policy_total, loss_value, entropy_total, loss_total, _ = self.session.run([loss_policy_total,
                                                                                        loss_value,
                                                                                        entropy_total,
                                                                                        loss_total,
                                                                                        minimize],
                                                                                       feed_dict={state_t: s,
                                                                                                  rooms_t: r_c,
                                                                                                  weeks_t: w_c,
                                                                                                  days_t: d_c,
                                                                                                  slots_t: s_c,
                                                                                                  reward_t: r})

        # self.train_writer.add_summary(summary)
        # self.train_writer.flush()

        return batch_length, queue_length, loss_policy_total, loss_value, entropy_total, loss_total

    def train_push(self, s, r_c, w_c, d_c, s_c, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(r_c)
            self.train_queue[2].append(w_c)
            self.train_queue[3].append(d_c)
            self.train_queue[4].append(s_c)
            self.train_queue[5].append(r)

            if s_ is None:
                self.train_queue[6].append(NONE_STATE)
                self.train_queue[7].append(0.)
            else:
                self.train_queue[6].append(s_)
                self.train_queue[7].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p_r, p_w, p_d, p_s, v = self.model.predict(s)
            return p_r, p_w, p_d, p_s, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p_r, p_w, p_d, p_s, v = self.model.predict(s)
            return p_r, p_w, p_d, p_s

    def predict_v(self, s):
        with self.default_graph.as_default():
            p_r, p_w, p_d, p_s, v = self.model.predict(s)
            return v


# ---------
class Optimizer(threading.Thread):
    stop_signal = False
    brain = None

    def __init__(self, brain):
        self.brain = brain
        threading.Thread.__init__(self)

    def run(self):
        global global_t
        while not self.stop_signal:
            batch_length, queue_length, loss_policy_total, loss_value, entropy_total, loss_total = self.brain.optimize()
            if REPORT_LEVEL == "learning":
                if batch_length > 0:
                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                          .format(str(self.brain.global_t).rjust(6),
                              str(batch_length).rjust(6),
                              str(queue_length).rjust(6),
                              "{: .2e}".format(loss_policy_total).replace(".", ",").rjust(10),
                              "{: .2e}".format(loss_value).replace(".", ",").rjust(10),
                              "{: .2e}".format(entropy_total).replace(".", ",").rjust(10),
                              "{: .2e}".format(loss_total).replace(".", ",").rjust(10),
                              "{:.4f}".format(self.brain.saved_solutions[self.brain.global_best_solution].placed_ratio)
                                  .replace(".",",").rjust(9),
                              str(int(self.brain.saved_solutions[self.brain.global_best_solution].pref_value))
                                  .rjust(5)))

    def stop(self):
        self.stop_signal = True


# ---------
class Agent:
    def __init__(self, env_id, brain, solution, eps_start, eps_end, eps_steps):
        self.brain = brain

        self.solution = solution
        self.solution.actor_id = env_id
        self.solution.domains = self.solution.domains_as_categories

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

    def act(self, m, s):
        eps = self.getEpsilon()
        self.brain.global_t += 1

        p_r, p_w, p_d, p_s = self.brain.predict_p(s)
        d = self.solution.domains[m]

        if random.random() < eps:
            random_r = np.random.random(d["rooms"].shape)
            random_w = np.random.random(d["weeks"].shape)
            random_d = np.random.random(d["days"].shape)
            random_s = np.random.random(d["slots"].shape)

            r_c = np.argmax(d["rooms"] * random_r)
            w_c = np.argmax(d["weeks"] * random_w)
            d_c = np.argmax(d["days"] * random_d)
            s_c = np.argmax(d["slots"] * random_s)

        else:
            max_r = np.max(p_r)
            max_w = np.max(p_w)
            max_d = np.max(p_d)
            max_s = np.max(p_s)

            random_r = np.random.random(d["rooms"].shape)
            random_w = np.random.random(d["weeks"].shape)
            random_d = np.random.random(d["days"].shape)
            random_s = np.random.random(d["slots"].shape)

            random_max_r = np.where(p_r == max_r, random_r, 0)
            random_max_w = np.where(p_w == max_w, random_w, 0)
            random_max_d = np.where(p_d == max_d, random_d, 0)
            random_max_s = np.where(p_s == max_s, random_s, 0)

            r_c = np.argmax(random_max_r)
            w_c = np.argmax(random_max_w)
            d_c = np.argmax(random_max_d)
            s_c = np.argmax(random_max_s)

        if d["rooms"][r_c] == d["weeks"][w_c] == d["days"][d_c] == d["slots"][s_c] == 1:
            indices = [r_c, w_c, d_c, s_c]

            duration_in_slots = self.solution.durations_in_slots[m]
            m_v = Value(m, indices, duration_in_slots)

            assert m_v.end < d["slots"].size, "Calculated index_end_time {} exceeds max slots {}"\
                                              .format(m_v.end, d["slots"].size - 1)

        else:
            m_v = None

        return r_c, w_c, d_c, s_c, m_v

    def train(self, s, r_c, w_c, d_c, s_c, r, s_):

        def get_sample(memory, n):
            s, r_c, w_c, d_c, s_c, _, _ = memory[0]
            _, _, _, _, _, _, s_ = memory[n - 1]

            return s, r_c, w_c, d_c, s_c, self.R, s_

        # turn action into one-hot representation
        r_hot = np.zeros(NUM_OUTPUTS["rooms"])
        r_hot[r_c] = 1

        w_hot = np.zeros(NUM_OUTPUTS["weeks"])
        w_hot[w_c] = 1

        d_hot = np.zeros(NUM_OUTPUTS["days"])
        d_hot[d_c] = 1

        s_hot = np.zeros(NUM_OUTPUTS["slots"])
        s_hot[s_c] = 1

        self.memory.append((s, r_hot, w_hot, d_hot, s_hot, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, r_c, w_c, d_c, s_c, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, r_c, w_c, d_c, s_c, r, s_)

                self.R = (self.R - self.memory[0][-2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, r_c, w_c, d_c, s_c, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, r_c, w_c, d_c, s_c, r, s_)

            self.R = self.R - self.memory[0][-2]
            self.memory.pop(0)


# ---------
class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, brain, solution, env_id, render=False, eps_start=EPS_START_RANGE, eps_end=EPS_STOP_RANGE, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)

        self.brain = brain
        self.id = env_id
        self.agent_t = 0

        self.current_solution = solution
        self.best_solution = copy.copy(solution)
        self.brain.saved_solutions[self.id] = copy.copy(solution)

        self.max_stagnation = 0

        eps_rnd = random.choice(range(len(eps_start)))
        self.agent = Agent(self.id, self.brain, self.current_solution, eps_start[eps_rnd], eps_end[eps_rnd], eps_steps)

        self.render = render

    def runEpisode(self):
        done = False
        self.max_stagnation = random.choice(MAX_STAGNATION_RANGE)
        m, s = self.current_solution.get_state()
        R = 0

        while True:
            time.sleep(THREAD_DELAY)  # yield

            # if self.render: self.env.render()

            start = datetime.datetime.now()
            self.agent_t += 1
            self.current_solution.iteration += 1
            r_c, w_c, d_c, s_c, m_v = self.agent.act(m, s)

            if m_v is None:
                r = -1
                self.current_solution.stagnation += 1
            else:
                self.current_solution.take_action(m, m_v)
                self.current_solution.evaluate_solution()

                if self.current_solution.is_better_solution(self.best_solution, SOLUTION_FIRST_CRITERION):
                    r = 1  # self.current_solution.placed_ratio
                    self.current_solution.stagnation = 0
                    self.best_solution = copy.copy(self.current_solution)
                    if self.best_solution.is_better_solution(self.brain.saved_solutions[self.id], SOLUTION_FIRST_CRITERION):
                        self.brain.saved_solutions[self.id] = copy.copy(self.best_solution)
                        if self.brain.saved_solutions[self.id].is_better_solution(self.brain.saved_solutions[self.brain.global_best_solution], SOLUTION_FIRST_CRITERION):
                            self.brain.global_best_solution = self.id
                else:
                    r = 0
                    self.current_solution.stagnation += 1

            m_, s_ = self.current_solution.get_state()

            self.agent.train(s, r_c, w_c, d_c, s_c, r, s_)

            if REPORT_LEVEL == "steps":
                self.print_step_stats(m, r_c, w_c, d_c, s_c)

            if self.agent_t % N_STEP_RETURN == 0:
                if REPORT_LEVEL == "episodes":
                    self.print_episode_stats(start)

                done = self.current_solution.is_stagnant(self.max_stagnation)

                if done:  # terminal state
                    s_ = None

            m = m_
            s = s_
            R += r

            if done:
                self.current_solution.reset_solution()
                self.best_solution = copy.copy(self.current_solution)

            if done or self.stop_signal:
                break

    def run(self):
        while not self.stop_signal or self.brain.global_t < MAX_GLOBAL_T:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True

    def print_step_stats(self, m, r, w, d, s):
        print(str(self.brain.global_t).rjust(6),
              str(self.id).rjust(4),
              str(self.agent_t).rjust(5),
              str(self.current_solution.iteration).rjust(5),
              str(self.current_solution.stagnation).rjust(4),
              str(m).rjust(4), str(r).rjust(4), str(w).rjust(4), str(d).rjust(4), str(s).rjust(4),
              "{:.4f}".format(self.current_solution.placed_ratio).replace(".", ",").rjust(9),
              str(int(self.current_solution.pref_value)).rjust(5),
              "{:.4f}".format(self.brain.saved_solutions[self.id].placed_ratio).replace(".", ",").rjust(9),
              str(int(self.brain.saved_solutions[self.id].pref_value)).rjust(5),
              "{:.4f}".format(self.brain.saved_solutions[self.brain.global_best_solution].placed_ratio).replace(".", ",").rjust(9),
              str(int(self.brain.saved_solutions[self.brain.global_best_solution].pref_value)).rjust(5), sep="\t")

    def print_episode_stats(self, start):
        print(str(self.brain.global_t).rjust(6),
              str(self.id).rjust(4),
              str(self.agent_t).rjust(5),
              str(self.current_solution.iteration).rjust(5),
              str(self.current_solution.stagnation).rjust(4),
              "{:.4f}".format(self.current_solution.placed_ratio).replace(".", ",").rjust(9),
              str(int(self.current_solution.pref_value)).rjust(5),
              "{:.4f}".format(self.brain.saved_solutions[self.id].placed_ratio).replace(".", ",").rjust(9),
              str(int(self.brain.saved_solutions[self.id].pref_value)).rjust(5),
              "{:.4f}".format(self.brain.saved_solutions[self.brain.global_best_solution].placed_ratio).replace(".", ",").rjust(9),
              str(int(self.brain.saved_solutions[self.brain.global_best_solution].pref_value)).rjust(5),
              str(datetime.datetime.now() - start)[5:].rjust(10), sep="\t")


# ---------
class Solution:

    uid = 1

    def __init__(self):

        self.id = self.next_uid()
        self.actor_id = 0
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
            h = np.hstack(self.domains[m].values())
            meetings[m] = np.sum(h) / np.size(h)

            if m in self.placed_values:
                meetings[m] -= 1

                rooms = np.zeros((NUM_OUTPUTS["rooms"]))
                weeks = np.zeros((NUM_OUTPUTS["weeks"]))
                days = np.zeros((NUM_OUTPUTS["days"]))
                slots = np.zeros((NUM_OUTPUTS["slots"]))

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

        s = np.reshape(s, (1, *STATE_SHAPE))

        return m, s

    def take_action(self, meeting, value):
        for slot in range(value.start, value.end + 1):
            conflict = self.calendar[value.room, value.week, value.day, slot]
            if conflict > 0:
                self.remove_meeting(conflict)

            for room in range(NUM_OUTPUTS["rooms"]):
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

