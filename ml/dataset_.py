from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torchio as tio
import nibabel as nib
from tqdm import tqdm
import scipy.ndimage as ndi

from utils.loading_utils import load_case
from utils.preprocessing import (z_score_normalization, random_crop_3d, resample_to_uniform)

class RandomCropOrPad(tio.Transform):
    """ Random Crop or Pad for tio.Compose
     Note: This is still a bit buggy and unstable

     """
    def __init__(self, target_shape, padding_mode='constant', padding_value=0, p=1):
        super().__init__(p=p)
        self.target_shape = np.array(target_shape)
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def apply_transform(self, subject):
        for image in subject.get_images(intensity_only=False):
            data = image.data
            current_shape = np.array(data.shape[1:])  # exclude channel

            # Crop first
            for i in range(3):
                excess = current_shape[i] - self.target_shape[i]
                if excess > 0:
                    crop_start = np.random.randint(0, excess + 1)
                    crop_end = crop_start + self.target_shape[i]
                    slicer = [slice(None)]
                    slicer += [slice(None)] * i + [slice(crop_start, crop_end)]
                    slicer += [slice(None)] * (2 - i)
                    data = data[tuple(slicer)]
                    current_shape[i] = self.target_shape[i]

            # Pad if needed
            pad_dims = []
            for i in range(3):
                diff = self.target_shape[i] - current_shape[i]
                if diff > 0:
                    pad_before = diff // 2
                    pad_after = diff - pad_before
                else:
                    pad_before = pad_after = 0
                pad_dims.append((pad_before, pad_after))

            pad_flat = [p for pair in reversed(pad_dims) for p in pair]
            if any(p > 0 for p in pad_flat):
                data = torch.nn.functional.pad(
                    data,
                    pad=pad_flat,
                    mode=self.padding_mode,
                    value=self.padding_value,
                )

            # Final safety check
            assert list(data.shape[1:]) == list(self.target_shape), \
                f"Final shape {data.shape[1:]} does not match target {self.target_shape}"

            image.set_data(data)

        return subject

def apply_random_affine_3d(image, seg, max_rot=10, max_trans=5):
    """
    Applies a small random 3D affine transformation to a multi-channel image and label volume.
    Assumes image shape (C, H, W, D) and seg shape (1, H, W, D).
    """
    angle = np.deg2rad(np.random.uniform(-max_rot, max_rot))
    tx, ty, tz = np.random.uniform(-max_trans, max_trans, size=3)

    # Simple rotation around z axis + translation
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    affine_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])

    affine_offset = [-tx, -ty, -tz]

    # Apply to each image channel
    for c in range(image.shape[0]):
        image[c] = ndi.affine_transform(
            image[c],
            matrix=affine_matrix,
            offset=affine_offset,
            order=1,
            mode='nearest'
        )

    # Apply to segmentation (nearest-neighbor to preserve labels)
    seg[0] = ndi.affine_transform(
        seg[0],
        matrix=affine_matrix,
        offset=affine_offset,
        order=0,
        mode='nearest'
    )

    return image, seg



class BrainMetPytorchDataset(Dataset):
    """ Pytorch Dataset for loading BraTS datapoints.
    Based on https://github.com/KurtLabUW/brats2023_updated/tree/master

    Args:
        root_dir: Root directory of the dataset.
        patch_size: Patch size (ph, pw, pd) to randomly crop images for.
        img_pad_value: Const value with which the input images are padded.
        seg_pad_value: Const value with which the segmentations are padded.
    """
    def __init__(self, root_dir, patch_size=(128,128, 96), img_pad_value=0, seg_pad_value=0, do_raffine=False):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.img_pad_value = img_pad_value
        self.seg_pad_value = seg_pad_value
        self.random_affine = do_raffine

        self.datapoints = []

        # Top-level BraTS-MET folders
        for d in os.listdir(root_dir):
            path = os.path.join(root_dir, d)
            if os.path.isdir(path) and d.startswith("BraTS-MET"):
                self.datapoints.append(path)

        # UCSD-Training subfolders (if present)
        ucsd_path = os.path.join(root_dir, "UCSD - Training")
        if os.path.isdir(ucsd_path):
            print(f'Found UCSD-Training subfolder: {ucsd_path}')
            for d in os.listdir(ucsd_path):
                path = os.path.join(ucsd_path, d)
                if os.path.isdir(path) and d.startswith("BraTS-MET"):
                    self.datapoints.append(path)
        print(f'Total # samples: {len(self.datapoints)} in {self.root_dir}\n')


    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        """ Loads the datapoint at index idx and applies a z-score normalization over the layers
        and a random cropping into the whole volume before returning.

        Returns:
            images: Torch tensor 3d image of shape (4, ph, ph, pd)
            segmentation: Torch tensor 3d segmentation of shape (1, ph, ph, pd)
        """

        data_point = self.datapoints[idx]
        images, segmentation = self._prepare_datapoint(data_point)

        # Layer-wise transformations
        images = [np.ascontiguousarray(x, dtype=np.float32) for x in images]
        images = [z_score_normalization(x) for x in images]

        segmentation = np.ascontiguousarray(segmentation, dtype=np.float32)

        # Assemble volumes
        images = np.stack(images)               # (4, H, W, D)
        segmentation = segmentation[None, ...]  # (1, H, W, D)
        # segmentation = segmentation_to_channels(segmentation)

        # Content-aware random crop
        MAX_ATTEMPTS = 5
        for _ in range(MAX_ATTEMPTS):
            cropped_img, cropped_seg = random_crop_3d(
                images,
                segmentation,
                crop_size=self.patch_size,
                img_pad_value=self.img_pad_value,
                seg_pad_value=self.seg_pad_value,
            )
            if cropped_seg.sum() > 0:
                break
        else:
            # fallback: just take any patch (may contain only background)
            cropped_img, cropped_seg = random_crop_3d(
                images,
                segmentation,
                crop_size=self.patch_size,
                img_pad_value=self.img_pad_value,
                seg_pad_value=self.seg_pad_value,
            )

        if self.random_affine:
            cropped_img, cropped_seg = apply_random_affine_3d(cropped_img, cropped_seg)

        # Volume transformations
        #images, segmentation = random_crop_3d(images, segmentation, crop_size=self.patch_size, img_pad_value=self.img_pad_value, seg_pad_value=self.seg_pad_value)
        #return torch.from_numpy(images), torch.from_numpy(segmentation)
        return torch.from_numpy(cropped_img), torch.from_numpy(cropped_seg)

    def _prepare_datapoint(self, datapoint):
        """ Loads the  different layers and segmentations """

        layer_data = list()
        for suffix in ['t1n', 't1c', 't2w', 't2f']:
            data = self._load(datapoint, suffix)
            layer_data.append(data)

        segmentation_data = self._load(datapoint, 'seg')
        return layer_data, segmentation_data

    def _load(self, datapoint, suffix):
        """ Loads full 3d vol using nibabel """

        folder_name = os.path.basename(datapoint)
        filename = f'{folder_name}-{suffix}.nii.gz'
        path = os.path.join(datapoint, filename)

        img = nib.load(path)

        # assert RAS orientation for all samples
        orig_ornt = nib.io_orientation(img.affine)
        targ_ornt = nib.orientations.axcodes2ornt("RAS")
        transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
        img = img.as_reoriented(transform)

        data = img.get_fdata(dtype=np.float32)
        spacing = img.header.get_zooms()[:3]

        target_spacing = (1.0, 1.0, 1.0)
        if not np.allclose(spacing, (1.0, 1.0, 1.0), atol=1e-3):
            data = resample_to_uniform(data, spacing, target_spacing)
        return data

# ... BrainMetDataset, BrainMetDatasetPreloaded
#       -> returns torchio.Subject; needs a torchio.SubjectsLoader
#       -> patch based sampling (e.g. 128x128x128) with overlap (e.g. 32x32x32);
#
#       -> probably the better solution...

#       -> non-proloaded: lazy variant loads on getitem()
#       -> Preloaded: loads the complete dataset into memory on init() (only if hw allows)

class BrainMetFullVolumeDataset(Dataset):
    """
    Dataset for generating predictions on BraTS-MET validation/test set.
    Only loads the 4 image modalities (no segmentation).
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.datapoints = [d for d in os.listdir(self.root_dir)
                           if os.path.isdir(os.path.join(self.root_dir, d)) and d.startswith('BraTS-MET')]

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        datapoint = self.datapoints[idx]
        images = [self._load(datapoint, suffix) for suffix in ['t1n', 't1c', 't2w', 't2f']]
        images = [z_score_normalization(np.ascontiguousarray(x, dtype=np.float32)) for x in images]
        images = np.stack(images)  # (4, H, W, D)
        return torch.from_numpy(images), datapoint  # return ID for naming

    def _load(self, datapoint, suffix):
        path = os.path.join(self.root_dir, datapoint, f"{datapoint}-{suffix}.nii.gz")
        img = nib.load(path)

        # assert RAS orientation for all samples
        orig_ornt = nib.io_orientation(img.affine)
        targ_ornt = nib.orientations.axcodes2ornt("RAS")
        transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
        img = img.as_reoriented(transform)

        data = img.get_fdata(dtype=np.float32)
        spacing = img.header.get_zooms()[:3]

        if not np.allclose(spacing, (1.0, 1.0, 1.0), atol=1e-3):
            data = resample_to_uniform(data, spacing, target_spacing=(1.0, 1.0, 1.0))
        return data

class BrainMetPytorchDatasetValidation(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.datapoints = []

        # Top-level BraTS-MET folders
        for d in os.listdir(root_dir):
            path = os.path.join(root_dir, d)
            if os.path.isdir(path) and d.startswith("BraTS-MET"):
                self.datapoints.append(path)

        # UCSD-Training subfolders (if present)
        ucsd_path = os.path.join(root_dir, "UCSD - Training")
        if os.path.isdir(ucsd_path):
            print(f'Found UCSD-Training subfolder: {ucsd_path}')
            for d in os.listdir(ucsd_path):
                path = os.path.join(ucsd_path, d)
                if os.path.isdir(path) and d.startswith("BraTS-MET"):
                    self.datapoints.append(path)
        print(f'Total # samples: {len(self.datapoints)} in {self.root_dir}\n')


    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        """ Loads the datapoint at index idx and applies a z-score normalization over the layers
        and a random cropping into the whole volume before returning.

        Returns:
            images: Torch tensor 3d image of shape (4, ph, ph, pd)
            segmentation: Torch tensor 3d segmentation of shape (1, ph, ph, pd)
        """

        data_point = self.datapoints[idx]
        images, segmentation = self._prepare_datapoint(data_point)

        # Layer-wise transformations
        images = [np.ascontiguousarray(x, dtype=np.float32) for x in images]
        images = [z_score_normalization(x) for x in images]

        segmentation = np.ascontiguousarray(segmentation, dtype=np.float32)

        # Assemble volumes
        images = np.stack(images)               # (4, H, W, D)
        segmentation = segmentation[None, ...]  # (1, H, W, D)

        images = torch.from_numpy(images)
        segmentation = torch.from_numpy(segmentation)

        import torch.nn.functional as F
        def pad_to_next_multiple(x, multiple=16):
            # x: (C, H, W, D)
            padding = []
            for dim in reversed(x.shape[1:]):  # skip channel
                remainder = dim % multiple
                pad = (0, 0) if remainder == 0 else (0, multiple - remainder)
                padding.extend(pad)
            return F.pad(x, padding, mode="constant", value=0)

        images = pad_to_next_multiple(images, 16)
        segmentation = pad_to_next_multiple(segmentation, 16)

        return images, segmentation

    def _prepare_datapoint(self, datapoint):
        """ Loads the  different layers and segmentations """

        layer_data = list()
        for suffix in ['t1n', 't1c', 't2w', 't2f']:
            data = self._load(datapoint, suffix)
            layer_data.append(data)

        segmentation_data = self._load(datapoint, 'seg')
        return layer_data, segmentation_data

    def _load(self, datapoint, suffix):
        """ Loads full 3d vol using nibabel """

        folder_name = os.path.basename(datapoint)
        filename = f'{folder_name}-{suffix}.nii.gz'
        path = os.path.join(datapoint, filename)

        img = nib.load(path)

        # assert RAS orientation for all samples
        orig_ornt = nib.io_orientation(img.affine)
        targ_ornt = nib.orientations.axcodes2ornt("RAS")
        transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
        img = img.as_reoriented(transform)

        data = img.get_fdata(dtype=np.float32)
        spacing = img.header.get_zooms()[:3]

        target_spacing = (1.0, 1.0, 1.0)
        if not np.allclose(spacing, (1.0, 1.0, 1.0), atol=1e-3):
            data = resample_to_uniform(data, spacing, target_spacing)
        return data


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