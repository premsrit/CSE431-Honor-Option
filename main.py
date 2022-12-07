from AVLTree import AVLTree
from RbTree import RBtree
from SplayTree import SplayTree, Node
from BSTree import BSTree
import random
from random import shuffle
import argparse
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt


def created_random_operation(size, adding_rate, search_rate, *, repeated_rate=0.01, recent_threshold=5, seed=431):
    """
    Creating random operations of adding, searching, and deleting with difference valuee.
    Notice that, deleting rate = 1 - (adding_rate + searching_rate)
    :param size: size of generating operation sequence
    :param adding_rate: rate of adding to happen
    :param search_rate: rate of searching to happen
    :param repeated_rate: rate of repeat the recent access value
    :param recent_threshold: defined how many recent value can be accessed
    :param seed: random seed to be used for random generator
    :return: sequence of operations
    """
    random.seed(seed)
    operations = []
    all_values = []
    sizes = 0
    while size:
        rate = random.random()
        if rate < adding_rate:
            operate = "add"
        elif rate < adding_rate + search_rate:
            operate = "search"
        else:
            operate = "remove"

        rate = random.random()
        if operate == "remove":
            if sizes == 0:
                continue
            recent_size = len(all_values) if rate < repeated_rate else min(recent_threshold, len(all_values))
            value = all_values[-random.randint(1, recent_size)]

        elif rate < repeated_rate and len(all_values) > 0:

            recent_size = min(recent_threshold, len(all_values))
            value = all_values[-random.randint(1, recent_size)]

        else:
            value = random.randint(0, 10000)
        operations.append((operate, value))
        if operate != "remove":
            all_values.append(value)
        sizes += 1 if operate == "add" else -1 if operate == "remove" else 0
        size -= 1
    return operations


def plot_figure(running_time, sizes, title, xlabel, ylable, save_name='figure.png'):
    """

    :param running_time: Dictionary of algorithm name as key and time to be plots
    :param sizes: sizes of input correspond to time
    :param title: Title of the figure
    :param xlabel: label of x-axis
    :param ylable: label of y-axis
    :return: None
    """
    plt.figure(figsize=(12, 8), dpi=100)
    for algorithm, runtime in running_time.items():
        runtime = np.array(runtime) * 1000  # Changing from sec to ms
        plt.plot(sizes, runtime, label=algorithm)
    plt.legend()
    plt.ylabel(ylable)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(save_name, format='png')
    plt.show()


def measure_runtime(data_structure, sizes, trial: int, adding_rate, searching_rate, repeated_rate=0.01):
    """
    Measure running time of the given algorithm with various input sizes, repeated several times based on trail
    :param data_structure: Data structure to measure running time
    :param sizes: Size that input need to be generated
    :param trial: Repeated number of experiment
    :return: None
    """

    def perform_operations(current_tree, orders):
        for op, value in orders:
            if op == "add":
                current_tree.insert(current_tree.root, value)
            elif op == "search":
                current_tree.search(current_tree.root, value)
            else:
                current_tree.remove(current_tree.root, value)

    average_times = {}
    for tree_name, tree in data_structure.items():
        average_times[tree_name] = []
        for size in sizes:
            total_time = 0
            random.seed(431)
            for _ in tqdm(range(trial), desc=str(tree_name) + " of size " + str(size)):
                operations = created_random_operation(size, adding_rate=adding_rate, search_rate=searching_rate,
                                                      seed=random.randint(1, 9999), repeated_rate=repeated_rate)
                cur_tree = tree()
                start = time.perf_counter()
                perform_operations(cur_tree, operations)
                end = time.perf_counter()
                total_time += end - start
            average_times[tree_name].append(total_time / trial)
    return average_times


def parse_argument():
    """
    Creating the argument parse for setting up the experiment
    :return:
    """
    parser = argparse.ArgumentParser(description="Run NLI Learning Code")

    parser.add_argument('--trial', default=10, help='number of trial of experiment', type=int)
    parser.add_argument('--adding_rate', default=0.3, help='rate of adding operation to happen', type=float)
    parser.add_argument('--searching_rate', default=0.4, help='rate of searching operation to happen', type=float)
    parser.add_argument('--repeating_rate', default=0.01, help='rate of repeating recently access value', type=float)
    parser.add_argument('--figure_title', default="Comparison of data structure", help='Title of plot figure', type=str)
    parser.add_argument('--figure_name', default="figure", help='Name of saving figure', type=str)

    args = parser.parse_args()
    return args


def main(args):

    tree = {
        "Binary Search Tree": BSTree,
        "Splay Tree": SplayTree,
        "AVL Tree": AVLTree,
        "Red Black Tree": RBtree,
    }
    sizes = [100, 500, 1000, 5000, 10000]
    result = measure_runtime(tree, sizes, args.trial,
                             adding_rate=args.adding_rate,
                             searching_rate=args.searching_rate,
                             repeated_rate=args.repeating_rate)
    plot_figure(result, sizes, args.figure_title, "Size of operations", "Running Time(ms)",
                save_name=args.figure_name + ".png")


if __name__ == "__main__":
    args = parse_argument()
    main(args)
