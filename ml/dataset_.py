from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torchio as tio
from tqdm import tqdm
from utils.loading_utils import load_case
from typing import Tuple

class RandomCropOrPad(tio.Transform):
    def __init__(self, target_shape, padding_mode='constant', padding_value=0, p=1):
        super().__init__(p=p)
        self.target_shape = np.array(target_shape)
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def apply_transform(self, subject):
        for image in subject.get_images(intensity_only=False):
            data = image.data
            current_shape = np.array(data.shape[1:])  # exclude channel
            diff = self.target_shape - current_shape

            padding = [[0, 0]]  # channel dimension not padded
            crop = [[0, 0]]  # for channel dimension

            for d in diff:
                if d > 0:
                    pad_before = np.random.randint(0, d + 1)
                    pad_after = d - pad_before
                    padding.append([pad_before, pad_after])
                    crop.append([0, 0])
                elif d < 0:
                    crop_before = np.random.randint(0, -d + 1)
                    crop_after = -d - crop_before
                    padding.append([0, 0])
                    crop.append([crop_before, crop_after])
                else:
                    padding.append([0, 0])
                    crop.append([0, 0])

            # Apply crop if needed
            if any(c[0] > 0 or c[1] > 0 for c in crop[1:]):
                slices = tuple(
                    slice(c[0], data.shape[i + 1] - c[1]) for i, c in enumerate(crop[1:])
                )
                data = data[:, slices[0], slices[1], slices[2]]

            # Apply padding if needed
            if any(p[0] > 0 or p[1] > 0 for p in padding[1:]):
                pad_flat = [val for pair in reversed(padding[1:]) for val in pair]  # reverse to match torch padding order
                data = torch.nn.functional.pad(
                    data, pad_flat, mode=self.padding_mode, value=self.padding_value
                )

            image.set_data(data)

        return subject

# ... BrainMetDataset, BrainMetDatasetPreloaded
#       -> returns torchio.Subject; needs a torchio.SubjectsLoader
#       -> patch based sampling (e.g. 128x128x128) with overlap (e.g. 32x32x32);
#
#       -> probably the better solution...

#       -> non-proloaded: lazy variant loads on getitem()
#       -> Preloaded: loads the complete dataset into memory on init() (only if hw allows)

class BrainMetDataset(tio.SubjectsDataset):
    def __init__(self, root_dir, transform=None):
        # lazy loading
        self.root_dir = root_dir
        self.transform = transform

        self.subjects = self._create_subjects_list()
        super(BrainMetDataset, self).__init__(self.subjects, transform=self.transform)

    def __len__(self):
        return len(self.subjects)

    def _create_subjects_list(self):
        patient_dirs = [os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir)
                        if os.path.isdir(os.path.join(self.root_dir, d)) and d.startswith('BraTS-MET')]

        subjects = []
        for patient_path in tqdm(patient_dirs, 'Creating subjects'):
            patient_id = os.path.basename(patient_path)

            subject = tio.Subject(
                t1c=tio.ScalarImage(os.path.join(patient_path, f"{patient_id}-t1c.nii.gz")),
                t1n=tio.ScalarImage(os.path.join(patient_path, f"{patient_id}-t1n.nii.gz")),
                t2f=tio.ScalarImage(os.path.join(patient_path, f"{patient_id}-t2f.nii.gz")),
                t2w=tio.ScalarImage(os.path.join(patient_path, f"{patient_id}-t2w.nii.gz")),
                seg=tio.LabelMap(os.path.join(patient_path, f"{patient_id}-seg.nii.gz")),
                patient_id=patient_id  # Optional metadata
            )

            subjects.append(subject)
        return subjects


class BrainMetDatasetPreloaded(Dataset):
    def __init__(self, root_dir, case_contents, with_segmentation=True, transforms=None, to_device=None):
        self.subjects = []

        print("Preloading dataset into memory...")
        for case_name, case_files in tqdm(case_contents.items(), desc="Loading cases"):
            case_dir = os.path.join(root_dir, case_name)
            case_data, _ = load_case(case_dir, case_files)

            ref_mod = next((case_data[m] for m in ['t1n', 't1c', 't2w', 't2f'] if case_data[m] is not None), None)
            input_tensors = {}
            for mod in ['t1n', 't1c', 't2w', 't2f']:
                vol = case_data.get(mod)
                if vol is None:
                    vol = np.zeros_like(ref_mod)
                tensor = torch.from_numpy(vol).unsqueeze(0).float()
                input_tensors[mod] = tensor

            subject = tio.Subject(
                t1n=tio.ScalarImage(tensor=input_tensors['t1n']),
                t1c=tio.ScalarImage(tensor=input_tensors['t1c']),
                t2w=tio.ScalarImage(tensor=input_tensors['t2w']),
                t2f=tio.ScalarImage(tensor=input_tensors['t2f']),
                name=case_name
            )

            if with_segmentation and case_data.get('seg') is not None:
                seg_tensor = torch.from_numpy(case_data['seg']).unsqueeze(0).to(torch.long)
                subject.add_image(image_name='seg', image=tio.LabelMap(tensor=seg_tensor))

            if transforms:
                subject = transforms(subject)

            if to_device:
                for img in subject.get_images(intensity_only=False):
                    img.set_data(img.data.to(to_device))

            self.subjects.append(subject)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return self.subjects[idx]


class GridSamplerWrapper(tio.data.PatchSampler):
    def __init__(self, patch_size, patch_overlap=(0, 0, 0), padding_mode=None):
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.padding_mode = padding_mode

    def __call__(self, subject):
        return tio.GridSampler(
            subject=subject,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap,
            padding_mode=self.padding_mode
        )