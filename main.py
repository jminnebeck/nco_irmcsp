import pickle
import pypyodbc
from Problem.IRMCSP import *
from Solver.MCTS import *

def main():

    print("Suche bestehende Problemdefiniton")
    try:
        irmcsp = pickle.load(open("./Pickle/irmcsp.p", "rb"))
        initial_solution = pickle.load(open("./Pickle/initial_solution.p", "rb"))
    except (pickle.PickleError, FileNotFoundError, EOFError):
        print("Lese neue Daten aus DB")
        irmcsp = IRMCSP(instance=1)
        initial_solution = Solution()

        conn = pypyodbc.win_connect_mdb("./Database/BE_iRMCSP.accdb")

        irmcsp.read_data(conn, initial_solution)

        print("Pickele Problemdefiniton")
        if not os.path.exists("./Pickle/"):
            os.makedirs("./Pickle/")
        pickle.dump(irmcsp, open("./Pickle/irmcsp.p", "wb"))
        pickle.dump(initial_solution, open("./Pickle/initial_solution.p", "wb"))

    irmcsp.current_version_note = "actors: {}, global_max_t: {}"\
                                  .format(THREADS, MAX_GLOBAL_T)

    action_size = len(irmcsp.rooms) * irmcsp.nr_weeks * irmcsp.nr_days * irmcsp.nr_slots
    state_size = action_size
    state_shape = [len(irmcsp.rooms), irmcsp.nr_weeks, irmcsp.nr_days, irmcsp.nr_slots]

    print("Starte Neuronale Monte Carlo Tree Search")
    mcts = MonteCarloTreeSearch()
    mcts.run(initial_solution, state_shape, state_size, action_size)

    # if mcts.saved_solutions:
    #     print("Schreibe Lösungen in DB")
    #     for solution in mcts.saved_solutions.values():
    #         irmcsp.nr_saved_solutions += 1
    #         solution.id = irmcsp.nr_saved_solutions
    #         conn = pypyodbc.win_connect_mdb(".\Database\BE_iRMCSP.accdb")
    #         irmcsp.write_solution(conn, solution)
    # else:
    #     print("Keine Lösungen übermittelt" + "\n")
    #
    # conn.close()

if __name__ == "__main__":
    main()