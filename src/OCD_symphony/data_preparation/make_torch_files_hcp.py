# Script to make torch files from fMRI preprocessed data

import argparse
from joblib import Parallel, delayed
import logging
import nilearn
from nilearn.input_data import NiftiLabelsMasker
import numpy as np
import os 
from time import time
import torch 

from OCD_symphony.utils import working_dir, proj_dir, create_log, today

data_dir = working_dir+'joint_projects/bmh_bigdata/HCP_YA/'
parc_dir = proj_dir+'/parcellations/'

# task
task_list = ['REST']

# preprocessed image space
img_space = '' # HCP was registered in MNI152NLin2009cAsym by default

# preprocessing label (no smoothing needed)
preproc_label = {
    'REST': 'hp2000_clean_rclean_tclean'}

# parcellation nifti files used in timeseries extraction
ts_parc_dict = {'schaefer400_tianS3': parc_dir+'schaefer400_tianS3MNI_lps_mni.nii.gz'}

smoothing_fwhm = {'schaefer400_tianS3': 0}


def get_subj_data(subj:str, task='REST') -> list:
    """ Get directory of data from subject """ 
    return [os.path.join(data_dir, subj, 'MNINonLinear/Results/rfMRI_REST/', 
                        'rfMRI_REST_'+preproc_label[task]+'.nii.gz')]


def extract_timeseries(subj, window_size=200, logger=logging.getLogger("debug")):
    # Extracts atlas based timeseries using nilearn
    t0 = time()
    for task in task_list:
        preproc = preproc_label[task]
        for parc in ts_parc_dict.keys():
            logger.info(f"Extracting timeseries for \t {subj} \t {task} : {preproc} \t {parc}")

            bold_files = get_subj_data(subj, task=task)

            pt_tensor = dict()

            for i,bold_file in enumerate(bold_files):

                # use nilearn to extract timeseries
                masker = NiftiLabelsMasker(ts_parc_dict[parc], smoothing_fwhm=smoothing_fwhm[parc], \
                                t_r=0.72, #low_pass=0.1, high_pass=0.01, \
                                memory='nilearn_cache', memory_level=1, verbose=0)
                
                # make timeseries and reshape (1, n_regions, n_timepoints)
                try:
                    time_series = masker.fit_transform(bold_file).T
                except Exception as e:
                    logger.info(f"{type(e).__name__} - {e}")
                    logger.info(f"{subj} : could not extract time series from file {bold_file}, discarding.")
                    continue
                
                # create samples
                n_samples = time_series.shape[-1] // window_size
                if n_samples==0:
                    logger.info(f"{subj} time serie of file {i} < window size, so discarded.")
                    continue
                time_series = time_series[:,:n_samples*window_size].reshape(n_samples,time_series.shape[0], window_size)
                fc = np.array([np.corrcoef(ts) for ts in time_series])
                
                # convert to torch arrays
                time_series = torch.tensor(time_series, dtype=torch.float32)[None, :, :]
                fc = torch.tensor(fc, dtype=torch.float32)[None, :, :]
                session = torch.tensor(np.repeat(i, n_samples), dtype=torch.int64)[None, :]

                # concatenate pre and post timeseries along first dimension
                if not 'fmri_data' in pt_tensor.keys():
                    pt_tensor['fmri_data'] = time_series
                    pt_tensor['fc_data'] = fc
                    pt_tensor['session'] = session
                else:
                    pt_tensor['fmri_data'] = torch.concat([pt_tensor['fmri_data'], time_series])
                    pt_tensor['fc_data'] = torch.concat([pt_tensor['fc_data'], fc])
                    pt_tensor['session'] = torch.concat([pt_tensor['session'], session])

            # save timeseries out as torch tensor file
            if pt_tensor != {}:
                out_file = os.path.join(proj_dir, 'data/hcp', subj+'.pt')
                torch.save(pt_tensor, out_file)
            else:
                logger.info(f"Subject {subj} was discarded, no data could be extracted.")
            
    t1 = time()
    logger.info(f"Extracted timeseries for {subj} in {t1-t0:.1f}s")


if __name__ == "__main__":
    logger = create_log("make_torch_files_hcp"+today())
    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', action='store', default=None, help='subject to process')
    args = parser.parse_args()

    if (args.subj == None):
        # loop through everyone and run:
        subj_list = [d for d in os.listdir(data_dir) if os.path.isdir(data_dir+d)]
        Parallel(n_jobs=1)(delayed(extract_timeseries)(subj, logger=logger) for subj in subj_list)
    else:
        extract_timeseries(args.subj, logger=logger)
