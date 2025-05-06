from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torchio as tio
from tqdm import tqdm
from utils.loading_utils import load_case

# ... BrainMetDataset, BrainMetDatasetPreloaded
#       -> returns torchio.Subject; needs a torchio.SubjectsLoader
#       -> patch based sampling (e.g. 128x128x128) with overlap (e.g. 32x32x32);
#
#       -> probably the better solution...

#       -> non-proloaded: lazy variant loads on getitem()
#       -> Preloaded: loads the complete dataset into memory on init() (only if hw allows)

class BrainMetDataset(Dataset):
    def __init__(self, root_dir, case_contents, with_segmentation=True, transforms=None, to_device=None):
        # lazy loading
        self.root_dir = root_dir
        self.case_contents = case_contents
        self.with_segmentation = with_segmentation
        self.transforms = transforms
        self.to_device = to_device
        self.case_names = list(case_contents.keys())

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        case_name = self.case_names[idx]
        case_dir = os.path.join(self.root_dir, case_name)
        case_files = self.case_contents[case_name]
        case_data, _ = load_case(case_dir, case_files)

        input_tensors = {}
        ref_mod = next((case_data[m] for m in ['t1n', 't1c', 't2w', 't2f'] if case_data[m] is not None), None)

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
        )

        if self.with_segmentation and case_data.get('seg') is not None:
            seg_tensor = torch.from_numpy(case_data['seg']).unsqueeze(0).to(torch.long)
            subject.add_image(image_name='seg', image=tio.LabelMap(tensor=seg_tensor))

        if self.transforms:
            subject = self.transforms(subject)

        if self.to_device:
            for img in subject.get_images(intensity_only=False):
                img.set_data(img.data.to(self.to_device))

        return subject


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