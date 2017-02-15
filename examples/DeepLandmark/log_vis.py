import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize a log file')
    parser.add_argument('--log', dest='log_file',
                        help='log file to visualize',
                        default=None, type=str)
    parser.add_argument('--start', dest='start_index',
                        help='start index of the iteration (included)',
                        default=0, type=int)
    parser.add_argument('--end', dest='end_index',
                        help='end index of the iteration (excluded)',
                        default=1, type=int)
    parser.add_argument('--step', dest='step_size',
                        help='step size',
                        default=1, type=int)
    parser.add_argument('--loss', dest='loss_pattern',
                        help='loss pattern to extract loss information, if it is none, then total loss is analysized',
                        default=None, type=str)
    parser.add_argument('--after', dest='after_pattern',
                        help='the loss information after this pattern will be processed',
                        default='Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model',
                        type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def total_loss_analysis(matched_text):
    words = matched_text.split(' ')
    num_word = len(words)
    loss = 0
    iteration = 0
    for i in np.arange(num_word):
        if words[i] == 'loss':
            loss = float(words[i+2])
        if words[i] == 'Iteration':
            iteration = int(words[i+1][0:-1])
    return iteration, loss

def total_loss_plot(log_file, start_index=0, end_index=1, step_size=1,
                 after_pattern='Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model*'):
    iteration_array = np.zeros((0), dtype=np.int32)
    loss_array = np.zeros((0), dtype=np.float32)
    with open(log_file, mode='r') as logs:
        after_pattern_matched = False
        index = 0
        skipped = 0
        for line in logs:
            if not after_pattern_matched:
                search_obj = re.search(after_pattern, line)
                if search_obj:
                    after_pattern_matched = True
            else:
                search_obj = re.search(r'Iteration (.*), loss = (.*)', line, re.M|re.I)
                if search_obj:
                    if skipped < step_size - 1:
                        skipped += 1
                    else:
                        if index >= start_index and index < end_index:
                            iteration, loss = total_loss_analysis(search_obj.group())
                            iteration_array = np.hstack((iteration_array, iteration))
                            loss_array = np.hstack((loss_array, loss))
                            skipped = 0
                    index += 1
            if index >= end_index:
                break

    if len(iteration_array) != len(loss_array) or len(iteration_array) == 0:
        print 'No loss information matched'
    else:
        iteration_max = np.amax(iteration_array)
        iteration_min = np.amin(iteration_array)
        loss_max = np.amax(loss_array)
        loss_min = np.amin(loss_array)

        axis = [iteration_min, iteration_max, np.floor(loss_min), np.ceil(loss_max)]
        plt.plot(iteration_array, loss_array, 'r^--')
        plt.axis(axis)
        plt.ylabel('Total Loss')
        plt.xlabel('Iteration')
        plt.show()

def specified_loss_analysis(matched_text, loss_pattern):
    words = matched_text.split(' ')
    num_word = len(words)
    loss = 0
    for i in np.arange(num_word):
        if words[i] == loss_pattern:
            loss = float(words[i+2])
    return loss

def specified_loss_plot(log_file, start_index=0, end_index=1, step_size=1,
                        loss_pattern = None,
                        after_pattern='Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model*'):
    iteration_array = np.zeros((0), dtype=np.int32)
    loss_array = np.zeros((0), dtype=np.float32)
    with open(log_file, mode='r') as logs:
        after_pattern_matched = False
        iteration_pattern_matched = False
        index = 0
        skipped = 0
        for line in logs:
            if not after_pattern_matched:
                search_obj = re.search(after_pattern, line)
                if search_obj:
                    after_pattern_matched = True
            else:
                search_obj = re.search(r'Iteration (.*), loss = (.*)', line, re.M | re.I)
                if search_obj:
                    if skipped < step_size - 1:
                        skipped += 1
                    else:
                        if index >= start_index and index < end_index:
                            iteration, _ = total_loss_analysis(search_obj.group())
                            iteration_array = np.hstack((iteration_array, iteration))
                            iteration_pattern_matched = True
                            skipped = 0
                    index += 1
                if iteration_pattern_matched:
                    search_obj = re.search(loss_pattern + ' = (.*) \(',
                                           line, re.M|re.I)
                    if search_obj:
                        loss = specified_loss_analysis(search_obj.group(),
                                                       loss_pattern)
                        loss_array = np.hstack((loss_array, loss))
                        iteration_pattern_matched = False
            if index >= end_index and len(iteration_array) == len(loss_array):
                break

    if len(iteration_array) != len(loss_array) or len(iteration_array) == 0:
        print 'No loss information matched'
    else:
        iteration_max = np.amax(iteration_array)
        iteration_min = np.amin(iteration_array)
        loss_max = np.amax(loss_array)
        loss_min = np.amin(loss_array)

        axis = [iteration_min, iteration_max, np.floor(loss_min), np.ceil(loss_max)]
        plt.plot(iteration_array, loss_array, 'r^--')
        plt.axis(axis)
        plt.ylabel(loss_pattern)
        plt.xlabel('Iteration')
        plt.show()



if __name__ == '__main__':
    args = parse_args()
    if args.end_index - args.start_index / args.step_size < 1:
        print 'no useful loss information'
    if args.loss_pattern:
        specified_loss_plot(args.log_file, args.start_index, args.end_index, args.step_size,
                            args.loss_pattern,
                            args.after_pattern)
    else:
        total_loss_plot(args.log_file, args.start_index, args.end_index, args.step_size,
                        args.after_pattern)