# Utility file for project miscaleneous 

import argparse
from concurrent_log_handler import ConcurrentRotatingFileHandler
from datetime import datetime
import json
import logging
import nilearn.image
import numpy as np
import os
import platform
import sys

#----------------------------------------------------
#                General utilities
#----------------------------------------------------


def today():
    return datetime.now().strftime('_%Y%m%d')


def get_working_dir():
    """ get computer name to set working path """
    if platform.node()=='qimr18844':
        working_dir = '/home/sebastin/working/'
    elif platform.node()=='qimr17596':
        working_dir = '/home/sebastin/working/'
    elif platform.node()=='neurosrv01':
        working_dir = '/home/sebastienn/working/'
    elif platform.node()=='NEUROv02':
        working_dir = '/home/sebastin/working/'
    elif 'hpc' in platform.node():
        working_dir = '/mnt/lustre/working/'
    #else:
        #print('Setting working dir as /working')
    #    working_dir = '/data/working/'
    return working_dir


def create_log(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(10)
    #file_handler = logging.FileHandler(os.path.join(proj_dir, 'logs', logname+".log"), mode="a", encoding="utf-8")
    logfile = os.path.join(proj_dir, 'logs', logname+".log")
    file_handler = ConcurrentRotatingFileHandler(logfile, "a", maxBytes=64 * 1024 * 1024, backupCount=1) # 64Mb 
    file_handler.setLevel(10)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("{asctime} - {levelname} - {message}", style="{", datefmt="%Y-%m-%d %H:%M")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(10)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    """logging.basicConfig(
        filename=os.path.join(proj_dir, 'logs', logname+".log"),
        encoding="utf-8",
        filemode="a",
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M")"""
    return logger

# default proj_dir if no config file is given
working_dir = get_working_dir()
proj_dir = os.path.join(get_working_dir(), 'lab_lucac/sebastiN/projects/OCD_Symphony')

# when config file is given, this gets overwritten
cfg = argparse.Namespace()
cfg.proj_dir = proj_dir



#-----------------------------------------------------------------------------------
#                           Brain Symphony utilities
#-----------------------------------------------------------------------------------

def get_BrainSymphony_atlas_labels():
    """ Load atlas id and labels for Brain Symphony's Schaefer400_TianS3 """
    # load atlas img and id/labels pairs
    with open(os.path.join(proj_dir, 'parcellations', 'atlas_config.json'), 'r') as f:
        atlas_config = json.load(f)

    # re-org ROI indices from Schaefer to Brain Symphony: ctx_L, sub-ctx_L, ctx_R, sub-ctx_R --> sub-ctx_R, sub-ctx_L, ctx_L, ctx_R 
    roi_inds = np.concatenate([np.arange(425,450), np.arange(200,225), np.arange(0,200), np.arange(225, 425)])

    # get significant regions names and ids from Brain Symph to Schaefer    
    node_names = np.array(atlas_config['schaefer400_tianS3']['node_names'])[roi_inds]
    node_ids = np.array(atlas_config['schaefer400_tianS3']['node_ids'])[roi_inds]

    return node_ids, node_names
