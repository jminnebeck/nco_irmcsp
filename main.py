import multiprocessing
import os
import pickle
# from A3C_LSTM_iRM import *
# from Gurobi_iRM import Flex_CSP_Model
import threading
from time import sleep

from A3C_LSTM_new import *
from IRMCSP import *


def main():

    print("Suche bestehende Problemdefiniton")
    try:
        irmcsp = pickle.load(open(".\Pickle\\irmcsp.p", "rb"))
        initial_solution = pickle.load(open(".\Pickle\\initial_solution.p", "rb"))
    except (pickle.PickleError, FileNotFoundError, EOFError):
        print("Lese neue Daten aus DB")
        irmcsp = IRMCSP(instance=1)
        initial_solution = Solution()

        import pypyodbc
        conn = pypyodbc.win_connect_mdb(".\Database\BE_iRMCSP.accdb")

        irmcsp.read_data(conn, initial_solution)

        # print("Pickele Problemdefiniton")
        # pickle.dump(irmcsp, open(".\Pickle\\irmcsp.p", "wb"))
        # pickle.dump(initial_solution, open(".\Pickle\\initial_solution.p", "wb"))

    domain_shape = {"rooms": len(irmcsp.rooms),
                    "weeks": irmcsp.nr_weeks,
                    "days": irmcsp.nr_days,
                    "slots": irmcsp.nr_slots}

    state_size = len(irmcsp.rooms) + irmcsp.nr_weeks + irmcsp.nr_days + irmcsp.nr_slots

    action_size = len(irmcsp.rooms) * irmcsp.nr_weeks * irmcsp.nr_days * irmcsp.nr_slots


    tf.reset_default_graph()

    load_model = False
    model_path = './model'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create a directory to save episode playback gifs to
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        async_rl = AC_Network(domain_shape, state_size, action_size, 'global', None)  # Generate global network
        trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        # Create worker classes
        workers = []
        num_workers = multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
        for i in range(num_workers):
            workers.append(Worker(async_rl, initial_solution, i, state_size, action_size, trainer, model_path, global_episodes))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(sess, coord, saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)

    # try:
    #     async_rl = pickle.load(open(".\Pickle\\A3C_brain.p", "rb"))
    #     print("A3C Netz geladen")
    # except (pickle.PickleError, FileNotFoundError, EOFError):
    #     num_outputs = {"rooms": len(irmcsp.rooms),
    #                    "weeks": irmcsp.nr_weeks,
    #                    "days": irmcsp.nr_days,
    #                    "slots": irmcsp.nr_slots}
    #
    #     state_shape = [len(irmcsp.rooms) + irmcsp.nr_weeks + irmcsp.nr_days + irmcsp.nr_slots,
    #                    1,
    #                    irmcsp.nr_meetings]
    #
    #     print("Erstelle neues Netz für A3C")
    #     async_rl = Brain(state_shape, num_outputs)
    #
    # irmcsp.current_version_note = "actors: {}, global_max_t: {}"\
    #                               .format(THREADS, MAX_GLOBAL_T)
    #
    # envs = [Environment(async_rl, deepcopy(initial_solution), i) for i in range(THREADS)]
    # opts = [Optimizer(async_rl) for i in range(OPTIMIZERS)]
    #
    # for o in opts:
    #     o.start()
    #
    # for e in envs:
    #     e.start()
    #
    # while async_rl.global_t < MAX_GLOBAL_T:
    #     time.sleep(1)  # yield
    #
    #     e_alive = 0
    #     o_alive = 0
    #
    #     for e in envs:
    #         if e.isAlive:
    #             e_alive += 1
    #         else:
    #             print("toter enviroment/agent thread gefunden")
    #             envs.remove(e)
    #     if e_alive == 0:
    #         break
    #
    #     for o in opts:
    #         if o.isAlive:
    #             o_alive += 1
    #         else:
    #             print("toter optimzer thread gefunden")
    #             opts.remove(o)
    #     if o_alive == 0:
    #         break
    #
    #     if async_rl.not_enough_optimizers:
    #         with threading.Lock():
    #             opts.append(Optimizer(async_rl))
    #             print("starte zusätzlichen optimizer thread, nun aktiv:", len(opts))
    #             async_rl.not_enough_optimizers = False
    #
    # for e in envs:
    #     e.stop()
    # for e in envs:
    #     e.join()
    #
    # for o in opts:
    #     o.stop()
    # for o in opts:
    #     o.join()

    # flex_csp = Flex_CSP_Model(irmcsp)
    # print("Starte Optimierung")
    # flex_csp.model.optimize()
    # flex_csp.model.getAttr("X")
    # flex_csp.model.write("flex_csp.sol")

    if async_rl.saved_solutions:
        print("Schreibe Lösungen in DB")
        for solution in async_rl.saved_solutions.values():
            irmcsp.nr_saved_solutions += 1
            solution.id = irmcsp.nr_saved_solutions
            conn = pypyodbc.win_connect_mdb(".\Database\BE_iRMCSP.accdb")
            irmcsp.write_solution(conn, solution)

        # print("Pickele Lösungen")
        # for solution in async_rl.saved_solutions.values():
        #     pickle.dump(solution, open(".\Pickle\\solution_" + str(solution.id) + ".p", "wb"))
    else:
        print("Keine Lösungen übermittelt" + "\n")

    conn.close()

    # save_net = input("Soll das Netz zur weiteren Verwendung gepickelt werden? (j/n)")
    # if save_net == ("j"):
    #     print("Pickele bestehendes Netz")
    #     pickle.dump(async_rl, open(".\Pickle\\async_rl.p", "wb"))
    # else:
    #     print("Netz verworfen")


if __name__ == "__main__":
    main()