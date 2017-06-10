import pickle
from Problem.IRMCSP import *
from Solver.MCTS import *

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



    # print("Erstelle neues Netz für A3C")
    # print("Suche bestehendes Neuronales Netz")

    irmcsp.current_version_note = "actors: {}, global_max_t: {}"\
                                  .format(THREADS, MAX_GLOBAL_T)

    state_size = irmcsp.nr_meetings * (len(irmcsp.rooms) + irmcsp.nr_weeks + irmcsp.nr_days + irmcsp.nr_slots)
    action_size = len(irmcsp.rooms) * irmcsp.nr_weeks * irmcsp.nr_days * irmcsp.nr_slots
    state_shape = [state_size]

    print("Starte Neuronale Monte Carlo Tree Search")
    mcts = MonteCarloTreeSearch()
    mcts.run(initial_solution, state_shape, state_size, action_size)

    if mcts.saved_solutions:
        print("Schreibe Lösungen in DB")
        for solution in mcts.saved_solutions.values():
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