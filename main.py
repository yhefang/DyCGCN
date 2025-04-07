#!/usr/bin/env python
import argparse
import sys

# torchlight
import torch
from torchlight import import_class



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    # arg = parser.parse_args()
    # arg = parser.parse_args(['recognition', '-c', 'config/rt_gcn/HKEXtoSSE/train_4test.yaml'])
    # arg = parser.parse_args(['recognition', '-c', 'config/rt_gcn/NASDAQ/train.yaml'])
    # arg = parser.parse_args(['recognition', '-c', 'config/rt_gcn/NASDAQ/test.yaml'])
    # arg = parser.parse_args(['recognition', '-c', 'config/rt_gcn/HKEXtoSSE/train.yaml'])
    # arg = parser.parse_args(['recognition', '-c', 'config/rt_gcn/HKEXtoSSE/test.yaml'])
    # arg = parser.parse_args(['recognition', '-c', 'config/rt_gcn/HKEXtoSZSE/train.yaml'])
    # arg = parser.parse_args(['recognition', '-c', 'config/rt_gcn/HKEXtoSZSE/test.yaml'])
    # arg = parser.parse_args(['recognition', '-c', 'config/rt_gcn/sp500/train.yaml'])
    arg = parser.parse_args(['recognition', '-c', 'config/rt_gcn/sp500/test.yaml'])


    # start
    Processor = processors[arg.processor]

    # p = Processor(['-c', 'config/rt_gcn/HKEXtoSSE/train_4test.yaml'])
    # p = Processor(['-c', 'config/rt_gcn/NASDAQ/train.yaml'])
    # p = Processor(['-c', 'config/rt_gcn/NASDAQ/test.yaml'])
    # p = Processor(['-c', 'config/rt_gcn/HKEXtoSSE/train.yaml'])
    # p = Processor(['-c', 'config/rt_gcn/HKEXtoSSE/test.yaml'])
    # p = Processor(['-c', 'config/rt_gcn/HKEXtoSZSE/train.yaml'])
    # p = Processor(['-c', 'config/rt_gcn/HKEXtoSZSE/test.yaml'])
    # p = Processor(['-c', 'config/rt_gcn/sp500/train.yaml'])
    p = Processor(['-c', 'config/rt_gcn/sp500/test.yaml'])

    p.start()
    #
    # arg = parser.parse_args(['recognition', '-c', 'config/rt_gcn/sp500/test.yaml'])
    # p = Processor(['-c', 'config/rt_gcn/sp500/test.yaml'])
    # p.start()