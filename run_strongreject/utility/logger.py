#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import sys
import os.path
import logging
import subprocess
import re
import datetime


def initialize_exp(args):
    '''
    args.dump_path : dump folder.
    args.exp_name : experiment name.
    args.exp_id : one experiment (share the same experiment name) for many times by different experiment id.
    '''

    ###############* Implement args.exp_name here *###############
    args.exp_name = args.exp_name.replace('exp_name_placeholder', f'{args.input_dataset}_[{args.victim_model_name}]') # jailbreak dataset - victim model
    
    # Dump parameters
    create_dump_path(args) 

    # Get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            command.append(x)
        else:
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)

    # Check experiment name
    assert len(args.exp_name.strip()) > 0
    
    # Create an ID for the job if it is not given in the parameters.
    # args.exp_id : one experiment (share the same experiment name) for many times by different experiment id
    if args.exp_id == '' or args.exp_id is None:        
        now = datetime.datetime.now()
        args.exp_id = exp_date = now.strftime("%y-%m-%d_%H:%M:%S")  # date time as exp_id
        
    # Create a logger / get a logger    
    logfile_name = f'run_{args.exp_id}__.log' if args.defense is None else f'{args.defense}-run_{args.exp_id}__.log'
    print_fancy_box(message, os.path.join(args.dump_path, logfile_name))
    logger = get_color_logger(os.path.join(args.dump_path, logfile_name))
    logger.info("\n"+"\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(args)).items())))
    logger.info("The experiment will be stored in %s" % args.dump_path)     # print experimental settings
    logger.info("Running command: %s" % command)                            # print running command

    return logger

def get_logger(filename, verbosity=1, name=None):
    
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s] %(message)s","%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def create_dump_path(args):

    # Create the sweep path if it does not exist
    sweep_path = os.path.join(args.dump_path, args.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # Create the dump folder
    args.dump_path = os.path.join(sweep_path)
    if not os.path.isdir(args.dump_path):
        subprocess.Popen("mkdir -p %s" % args.dump_path, shell=True).wait()
        
    # Add to args
    logfile_name = f'run_{args.exp_id}__.log' if args.defense is None else f'{args.defense}-run_{args.exp_id}__.log'
    args.logfile = os.path.join(args.dump_path, logfile_name)
        
def print_fancy_box(message, log_path):
    lines = message.split('\n')
    max_length = max(len(line) for line in lines)
    with open(log_path, 'a') as fw:
        print('╔' + '═' * (max_length * 5) + '╗', file=fw)
        for line in lines:
            # use rjust() and ljust() to ensure alignment in formatted string
            print(f'║ {line.ljust(max_length *5 - 2)} ║', file=fw)
        print('╚' + '═' * (max_length * 5) + '╝', file=fw)

# demo message
message = """
WARNING:
Initialized logger
"""

# define NOTICE log level
NOTICE_LEVEL_NUM = 25
logging.addLevelName(NOTICE_LEVEL_NUM, "NOTICE")

def notice(self, message, *args, **kws):
    if self.isEnabledFor(NOTICE_LEVEL_NUM):
        self._log(NOTICE_LEVEL_NUM, message, args, **kws)

# add to Logger
logging.Logger.notice = notice





# colorful log
def get_color_logger(filename, verbosity=1, name=None):
    log_colors = {
        'DEBUG':    'cyan',
        'NOTICE':   'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
    }
    import colorlog
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s %(levelname)s] %(message)s","%Y-%m-%d %H:%M:%S",
        log_colors=log_colors
    )
    formatter_ = logging.Formatter(
        "[%(asctime)s %(levelname)s] %(message)s","%Y-%m-%d %H:%M:%S"
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter_)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
