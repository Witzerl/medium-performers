import os
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
import random
import numpy as np


def list_case_folders(root_dir):
    """
    List all case folder names in a directory.
    Ignores files; only folders are listed.

    Args:
        root_dir (str): Path to root directory (e.g., validation or training folder).

    Returns:
        list: Sorted list of case folder names.
    """
    case_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    case_folders.sort()
    return case_folders


def inspect_case_contents(root_dir):
    """
    Inspects contents of each case folder in the given directory.

    Args:
        root_dir (str): Path to root directory (e.g., validation or training folder).

    Returns:
        dict: Dictionary where keys are case folder names and values are lists of file names inside.
    """
    case_folders = list_case_folders(root_dir)
    contents = {}

    for case in case_folders:
        case_path = os.path.join(root_dir, case)
        files = sorted(os.listdir(case_path))  # sorted for consistent viewing
        contents[case] = files

    print(f"Total cases: {len(contents)}")
    return contents

def load_nifti(file_path):
    """
    Load a NIfTI file, reorient to RAS+, standardize shape to (Height, Width, Slices) based on shape symmetry.
    Only fix if first dimension is clearly different from the second and third.
    Returns:
        data: np.ndarray, (H, W, Slices)
        info: dict, containing shape, suspicious flag
    """

    img = nib.load(file_path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine

    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    origin = affine[:3, 3]

    shape = data.shape

    suspicious = False

    # Check if first dimension is much different from second and third
    # Assume if dim0 differs a lot, it was wrongly ordered (wrong orientation)
    if not (np.isclose(shape[0], shape[1], rtol=0.1) or np.isclose(shape[0], shape[2], rtol=0.1)):
        suspicious = True
        data = np.transpose(data, (1, 2, 0))

    info = {
        'original_shape': shape,
        'corrected_shape': data.shape,
        'suspicious_orientation': suspicious,
        'spacing': spacing,
        'origin': origin
    }

    return data, info

def load_case(case_dir, case_files):
    """
    Load all files from a SPECIFIC case given the case directory and case file names.

    Args:
        case_dir (str): Path to the root directory (e.g., /path/to/validation/BraTS-MET-00833-000)
        case_files (list): List of filenames available inside that case.

    Returns:
        dict: Dictionary of modality name -> numpy array, plus 'seg' if available.
        dict: Single correction/info dictionary for the case
    """
    suffixes = ['t1n', 't1c', 't2w', 't2f']
    data = {suffix: None for suffix in suffixes}
    data['seg'] = None  # segmentation slot

    spacing_info = None  # Will hold correction info for the case

    for file in case_files:
        lower_file = file.lower()
        if lower_file.endswith(".nii.gz"):
            filepath = os.path.join(case_dir, file)

            if "-seg" in lower_file:
                seg_volume, _ = load_nifti(filepath)
                data['seg'] = seg_volume.astype(np.uint8)  # convert to integers

            else:
                for suffix in suffixes:
                    if f"-{suffix}" in lower_file:
                        volume, corr_info = load_nifti(filepath)
                        data[suffix] = volume

                        if spacing_info is None:
                            spacing_info = corr_info  # Save only once
                        break

    return data, spacing_info


def load_all_cases(root_dir, case_contents, limit=None, seed=None):
    """
    Loads all (or a limited number of) cases from root_dir with optional random selection and progress bar.

    Args:
        root_dir (str): Path to the folder containing case folders.
        case_contents (dict): Dictionary {case_name: [file_list]} from inspect_case_contents.
        limit (int, optional): Maximum number of cases to load. If None, loads all.
        seed (int, optional): Random seed for reproducible random selection if limit is set.

    Returns:
        dict: {case_name: {'data': {modality: numpy_array}, 'info': case_info_dict}}
    """
    loaded_cases = {}

    # Prepare list of cases
    case_items = list(case_contents.items())

    if limit is not None:
        if seed is not None:
            random.seed(seed)
        case_items = random.sample(case_items, k=limit)

    # Progress bar setup
    pbar = tqdm(case_items, desc="Loading cases", unit="case")

    for case_name, file_list in pbar:
        case_dir = os.path.join(root_dir, case_name)
        case_data, case_info = load_case(case_dir, file_list)

        loaded_cases[case_name] = {
            'data': case_data,
            'info': case_info
        }

    return loaded_cases


def filter_suspicious_cases(root_dir, case_contents):
    """
    Removes cases with suspicious orientation by checking the T1c modality.
    """
    filtered = {}
    removed_count = 0
    missing_t1c = []

    for case_name, file_list in case_contents.items():
        case_dir = os.path.join(root_dir, case_name)
        t1c_file = next(
            (f for f in file_list if '-t1c' in f.lower() and f.endswith('.nii.gz')), None
        )

        if not t1c_file:
            missing_t1c.append(case_name)
            continue

        filepath = os.path.join(case_dir, t1c_file)
        _, info = load_nifti(filepath)

        if info.get('suspicious_orientation', True):
            removed_count += 1
            continue

        filtered[case_name] = file_list

    print(f"Excluded {removed_count} cases with suspicious orientation.")
    print(f"Skipped {len(missing_t1c)} cases without T1c.")
    return filtered

