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
from OCD_symphony.data_preparation.fmri_cleaning import (
    CleaningConfig, cleaning_metadata,
)
from OCD_symphony.data_preparation.paper_extraction import (
    extract_subject_paper, parcellate, prepare_output_dir, update_metadata_dropped,
)

#data_dir = working_dir+'lab_lucac/sebastiN/projects/OCDbaseline/'
#deriv_dir = data_dir+'data/derivatives/fmriprep-fix/'
parc_dir = proj_dir+'/parcellations/'

# task
task_list = ['rest']

# preprocessed image space
img_space = 'MNI152NLin2009cAsym_res-2'

# preprocessing label (no smoothing needed)
preproc_label = {
    'rest': ['preproc_bold']}

# parcellation nifti files used in timeseries extraction
ts_parc_dict = {'schaefer400_tianS3': parc_dir+'schaefer400_tianS3MNI_lps_mni.nii.gz'}

smoothing_fwhm = {'schaefer400_tianS3': 0}


def get_subj_data(subj:str, task='rest', preproc='preproc_bold') -> list:
    """ Get directory of data from subject according to whether control or OCD patient """
    #if 'control' in subj:
    return [os.path.join(#working_dir,
                            #'lab_lucac/sebastiN/projects/OCDbaseline/data/derivatives/fmriprep-fix/',
                            #'/home/sebastin/LabData/Lab_LucaC/Luke/Archive/OCDbaseline/data/derivatives/fmriprep/',
                            '/home/sebastin/LabData/Lab_LucaC/Luke/Backups/hpc_backups/projects/fMRIPrep_23_2_0/data/OCDbaseline/derivatives/fmriprep',
                            subj, 'func', 
                            subj+'_task-'+task+'_space-'+img_space+'_desc-'+preproc+'.nii.gz')]
                            #proj_dir, 'data', 'derivatives', 'fmriprep',
                            #subj, 'func', 
                            #subj+'_task-'+task+'_desc-'+preproc+'.nii.gz')]
    """elif 'patient' in subj:
        return [os.path.join(#working_dir,
                            #'lab_lucac/sebastiN/projects/OCD_clinical_trial/data/derivatives/fmriprep-fix/',
                            #'/home/sebastin/LabData/Lab_LucaC/Luke/Archive/ocd_clinical_trial/data/derivatives/fmriprep/',
                            proj_dir, 'data', 'derivatives', 'fmriprep',
                            subj, ses, 'func',
                            #subj+'_ses-pre_task-'+task+'_space-'+img_space+'_desc-'+preproc+'.nii.gz'),
                            subj+'_'+ses+'_task-'+task+'_desc-'+preproc+'.nii.gz') for ses in ['ses-pre', 'ses-post']]
    """

def extract_timeseries(subj, window_size=200, subdir = '', logger=logging.getLogger("debug")):
    # Extracts atlas based timeseries using nilearn
    t0 = time()
    for task in task_list:
        for preproc in preproc_label[task]:
            for parc in ts_parc_dict.keys():
                logger.info(f"Extracting timeseries for \t {subj} \t {task} : {preproc} \t {parc}")

                bold_files = get_subj_data(subj, task=task, preproc=preproc)

                pt_tensor = dict()

                for i,bold_file in enumerate(bold_files):

                    # use nilearn to extract timeseries
                    masker = NiftiLabelsMasker(ts_parc_dict[parc], smoothing_fwhm=smoothing_fwhm[parc], \
                                    t_r=0.81, #low_pass=0.1, high_pass=0.01, \
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
                    #pt_tensor['subj'] = subj
                    #pt_tensor['cohort'] = subj.split('-')[-1][:-2] #extract "control" or "patient" from subject id

                # save as torch tensor file
                if pt_tensor != {}:
                    out_dir = os.path.join(proj_dir, 'data', subdir)
                    os.makedirs(out_dir, exist_ok=True)
                    out_file = os.path.join(out_dir, subj+'.pt')
                    torch.save(pt_tensor, out_file)
                else:
                    logger.info(f"Subject {subj} was discarded, no data could be extracted.")
                
    t1 = time()
    logger.info(f"Extracted timeseries for {subj} in {t1-t0:.1f}s")


# ---------------------------------------------------------------------------
# Paper-mode extraction (default). Source: fMRIPrep-FIX derivative, native MNI resolution.
# Per run: parcellate -> detrend -> band-pass -> motion regression (6 confounds) ->
# FD flagging (fMRIPrep framewise_displacement) -> interpolation -> 200-TR windows.
# ---------------------------------------------------------------------------

PAPER_OUT_DIR = os.path.join(proj_dir, 'data', 'OCD_paper')
DERIV_FIX_DIR = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCDbaseline/data/derivatives/fmriprep-fix/')
DEMOGRAPHICS_FILE = os.path.join(proj_dir, 'data/demographics.csv')
PAPER_SPACE = 'MNI152NLin2009cAsym'
MOTION_COLUMNS = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']


def paper_bold_paths(subj: str, task: str = 'rest') -> dict:
    func = os.path.join(DERIV_FIX_DIR, subj, 'func')
    stem = os.path.join(func, f"{subj}_task-{task}_space-{PAPER_SPACE}_desc-preproc_bold")
    return dict(bold=stem + '.nii.gz', sidecar=stem + '.json',
                confounds=os.path.join(func, f"{subj}_task-{task}_desc-confounds_timeseries.tsv"))


def read_tr(sidecar: str) -> float:
    import json
    with open(sidecar) as f:
        return float(json.load(f)['RepetitionTime'])


def read_confounds(path: str):
    """Return (motion (T, 6), fd (T,)) from an fMRIPrep confounds TSV."""
    import pandas as pd
    df = pd.read_csv(path, sep='\t', usecols=['framewise_displacement'] + MOTION_COLUMNS)
    motion = df[MOTION_COLUMNS].to_numpy(dtype=np.float64)
    fd = pd.to_numeric(df['framewise_displacement'], errors='coerce').to_numpy(dtype=np.float64, copy=True)
    fd[0] = 0.0  # first volume has no previous frame; fMRIPrep writes n/a
    return motion, fd


def ocd_subject_list() -> list:
    import pandas as pd
    subjs = pd.read_csv(DEMOGRAPHICS_FILE)['subj'].astype(str).str.strip().tolist()
    return sorted(subjs)


def load_ocd_paper_runs(subj: str, logger=logging.getLogger("debug")) -> list:
    paths = paper_bold_paths(subj)
    for key in ('bold', 'sidecar', 'confounds'):
        if not os.path.exists(paths[key]):
            logger.info(f"{subj}: missing {key} file {paths[key]}")
            return []
    tr = read_tr(paths['sidecar'])
    ts = parcellate(paths['bold'], ts_parc_dict['schaefer400_tianS3'], tr)
    motion, fd = read_confounds(paths['confounds'])
    if motion.shape[0] != ts.shape[1]:
        logger.info(f"{subj}: confounds have {motion.shape[0]} rows, series {ts.shape[1]} TR")
        return []
    return [dict(ts=ts, tr=tr, fd=fd, confounds=motion, session=0)]


def ocd_paper_metadata(cfg: CleaningConfig, tr_note: str) -> dict:
    meta = cleaning_metadata(cfg, float('nan'), sources=dict(
        derivative=DERIV_FIX_DIR, bold=f"*_task-rest_space-{PAPER_SPACE}_desc-preproc_bold.nii.gz",
        confounds="*_task-rest_desc-confounds_timeseries.tsv", atlas=ts_parc_dict['schaefer400_tianS3']))
    meta['tr_seconds'] = tr_note
    meta['dataset'] = 'ocd'
    meta['fd_radius_mm'] = None
    meta['fd_note'] = "framewise_displacement column from fMRIPrep confounds (Power definition)"
    return meta


def extract_paper(subj: str, out_dir: str, cfg: CleaningConfig, overwrite: bool = False,
                  logger=logging.getLogger("debug")) -> None:
    t0 = time()
    logger.info(f"Extracting paper-mode timeseries for {subj} -> {out_dir}")
    runs = load_ocd_paper_runs(subj, logger=logger)
    if not runs:
        logger.info(f"{subj}: no usable run, discarded")
        return
    extract_subject_paper(subj, runs, out_dir, runs[0]['tr'], cfg, overwrite, logger)
    logger.info(f"Extracted {subj} in {time()-t0:.1f}s")


if __name__ == "__main__":
    logger = create_log("make_torch_files_ocd"+today())
    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', action='store', default=None, help='subject to process')
    parser.add_argument('--subdir', action='store', default='', help='Sub-directory in proj_dir/data/ to save the torch files (legacy)')
    parser.add_argument('--out_dir', action='store', default=PAPER_OUT_DIR,
                        help='paper-mode output folder (default data/OCD_paper)')
    parser.add_argument('--n_jobs', action='store', type=int, default=16, help='number of parallel processes')
    parser.add_argument('--overwrite', action='store_true', help='recompute and replace existing outputs')
    parser.add_argument('--legacy', action='store_true',
                        help='run the original extraction (unchanged; source path no longer exists)')
    args = parser.parse_args()

    if args.legacy:
        if (args.subj == None):
            # loop through everyone and run:
            subj_list = list(np.loadtxt(os.path.join(proj_dir, 'data/subject_list.txt'), dtype='str'))
            Parallel(n_jobs=16)(delayed(extract_timeseries)(subj, subdir=args.subdir, logger=logger) for subj in subj_list)
        else:
            extract_timeseries(args.subj, subdir=args.subdir, logger=logger)
    else:
        cfg = CleaningConfig(min_windows=2)  # N=2 for OCD: a single run yields at most 4 windows
        tr_note = 'per subject, from JSON RepetitionTime'
        prepare_output_dir(args.out_dir, ocd_paper_metadata(cfg, tr_note), overwrite=args.overwrite)
        subj_list = [args.subj] if args.subj is not None else ocd_subject_list()
        if len(subj_list) == 1:
            extract_paper(subj_list[0], args.out_dir, cfg, args.overwrite, logger=logger)
        else:
            Parallel(n_jobs=args.n_jobs)(
                delayed(extract_paper)(subj, args.out_dir, cfg, args.overwrite, logger)
                for subj in subj_list)
        update_metadata_dropped(args.out_dir)
