import random
import numpy as np
from math import exp
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd

N_QUEENS = 32
TEMPERATURE = 40


def threat_calculate(n):
    '''Combination formular. It is choosing two queens in n queens'''
    if n < 2:
        return 0
    if n == 2:
        return 1
    return (n - 1) * n / 2


def create_board(n):
    '''Create a chess boad with a queen on a row'''
    chess_board = {}
    temp = list(range(n))
    random.shuffle(temp)  # shuffle to make sure it is random
    column = 0

    while len(temp) > 0:
        row = random.choice(temp)
        chess_board[column] = row
        temp.remove(row)
        column += 1
    del temp
    return chess_board


def cost(chess_board):
    '''Calculate how many pairs of threaten queen'''
    threat = 0
    m_chessboard = {}
    a_chessboard = {}

    for column in chess_board:
        temp_m = column - chess_board[column]
        temp_a = column + chess_board[column]
        if temp_m not in m_chessboard:
            m_chessboard[temp_m] = 1
        else:
            m_chessboard[temp_m] += 1
        if temp_a not in a_chessboard:
            a_chessboard[temp_a] = 1
        else:
            a_chessboard[temp_a] += 1

    for i in m_chessboard:
        threat += threat_calculate(m_chessboard[i])
    del m_chessboard

    for i in a_chessboard:
        threat += threat_calculate(a_chessboard[i])
    del a_chessboard

    return threat


def one_search_run(n, method='HC', temperature=40, schedule=0.99, t_min=1e-7):
    '''Single run of HC or SA from one random start state.'''
    start_time = time.time()
    answer = create_board(n)
    cost_answer = cost(answer)
    costs = [cost_answer]

    t = float(temperature)
    while t > t_min:
        t *= schedule
        successor = deepcopy(answer)

        while True:
            index_1 = random.randrange(0, n)
            index_2 = random.randrange(0, n)
            if index_1 != index_2:
                break

        successor[index_1], successor[index_2] = successor[index_2], successor[index_1]
        successor_cost = cost(successor)
        delta = successor_cost - cost_answer

        if method == 'HC':
            accept = (delta < 0)
        elif method == 'SA':
            accept = (delta < 0) or (random.uniform(0, 1) < exp(-delta / t))
        else:
            raise ValueError("method must be 'HC' or 'SA'")

        if accept:
            answer = deepcopy(successor)
            cost_answer = successor_cost
            costs.append(cost_answer)

        if cost_answer == 0:
            return True, costs, (time.time() - start_time)

    return False, costs, (time.time() - start_time)


def run_until_success(n, method='HC', temperature=40):
    '''Repeat independent runs until first success.'''
    total_start = time.time()
    repetitions = 0
    success_costs = []

    while True:
        repetitions += 1
        success, costs, _ = one_search_run(n=n, method=method, temperature=temperature)
        if success:
            success_costs = costs
            break

    return repetitions, (time.time() - total_start), success_costs
    

def print_chess_board(board):
    '''Print the chess board'''
    showBoard = np.zeros([N_QUEENS,N_QUEENS],dtype = int)
    for column, row in board.items():
        showBoard[row][column]=1
        #print("{} => {}".format(column, row))
    for i in range(N_QUEENS):
        print(showBoard[i])


def benchmark_hc_sa(n_values=(8, 16, 32, 64), repeats=10, temperature=40):
    rows = []
    for n in n_values:
        for method in ('HC', 'SA'):
            runtimes = []
            repetitions_before_success = []
            success_rates = []

            for _ in range(repeats):
                reps, runtime, _ = run_until_success(n=n, method=method, temperature=temperature)
                repetitions_before_success.append(reps)
                runtimes.append(runtime)
                success_rates.append(1.0 / reps)

            rows.append({
                'N': n,
                'Method': method,
                'Avg Runtime (s)': float(np.mean(runtimes)),
                'Avg Repetitions Before Success': float(np.mean(repetitions_before_success)),
                'Avg Success Rate (1/reps)': float(np.mean(success_rates)),
            })

    return pd.DataFrame(rows).sort_values(['N', 'Method']).reset_index(drop=True)


def benchmark_sa_temperature(n=64, temperatures=(4000, 400, 40, 4, 0.4), repeats=10):
    rows = []
    for temp in temperatures:
        runtimes = []
        repetitions_before_success = []
        success_rates = []

        for _ in range(repeats):
            reps, runtime, _ = run_until_success(n=n, method='SA', temperature=temp)
            repetitions_before_success.append(reps)
            runtimes.append(runtime)
            success_rates.append(1.0 / reps)

        rows.append({
            'N': n,
            'Temperature': temp,
            'Avg Runtime (s)': float(np.mean(runtimes)),
            'Avg Repetitions Before Success': float(np.mean(repetitions_before_success)),
            'Avg Success Rate (1/reps)': float(np.mean(success_rates)),
        })

    return pd.DataFrame(rows).sort_values('Temperature', ascending=False).reset_index(drop=True)


def plot_cost_traces(n=64, temperature=40):
    _, _, costs_hc = run_until_success(n=n, method='HC', temperature=temperature)
    _, _, costs_sa = run_until_success(n=n, method='SA', temperature=temperature)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(costs_hc)
    plt.title(f'HC Cost Trace (N={n})')
    plt.xlabel('Board updates (accepted moves)')
    plt.ylabel('# of attacked Q pairs')

    plt.subplot(1, 2, 2)
    plt.plot(costs_sa)
    plt.title(f'SA Cost Trace (N={n}, T={temperature})')
    plt.xlabel('Board updates (accepted moves)')
    plt.ylabel('# of attacked Q pairs')

    plt.tight_layout()
    plt.show()


def main():
    print('=== (a) HC vs SA benchmark for N = 8, 16, 32, 64 (10 repeats) ===')
    results_a = benchmark_hc_sa(n_values=(8, 16, 32, 64), repeats=10, temperature=TEMPERATURE)
    print(results_a)

    print('\n=== (b) Cost traces for HC vs SA ===')
    plot_cost_traces(n=64, temperature=TEMPERATURE)

    print('\n=== (c) SA temperature sweep for N=64 ===')
    results_c = benchmark_sa_temperature(n=64, temperatures=(4000, 400, 40, 4, 0.4), repeats=10)
    print(results_c)


if __name__ == "__main__":
    main()