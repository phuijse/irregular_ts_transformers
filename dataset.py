from typing import Optional
import torch
from torch import Tensor
from torch.utils.data import Dataset


def collate_light_curves(batch: list[dict],
                         target_length: Optional[int] = None):
    """
    batch is a list of dicts obtained using the __getitem__() method of
    the datasets. The elements in the dict include light curves, features,
    labels and other metadata. The collate function concats the light curves
    in the batch by padding/cropping to a target length. If the target length
    is not specified, it will be set to the maximum length within the batch.
    A padding mask is created and included in the batch.
    """
    collated_batch = {}
    labels = [sample['label'] for sample in batch]
    collated_batch['label'] = torch.LongTensor(labels)
    collated_batch['light_curve'] = {}
    if target_length is None:
        max_length = 0
        for sample in batch:
            seq_len = sample['light_curve'].shape[1]
            if seq_len > max_length:
                max_length = seq_len
        target_length = max_length
    
    lcs, masks = [], []
    for sample in batch:
        data = sample['light_curve']
        padded_data, mask = resize_tensor(target_length, data)
        lcs.append(padded_data)
        masks.append(mask)
    collated_batch['light_curve'] = (torch.stack(lcs), torch.stack(masks))
    return collated_batch

def resize_tensor(target_length: int,
                  tensor: Tensor) -> tuple[Tensor, Tensor]:
    """
    Resizes the tensor so that it matches the target length.
    If they are longer than target length they are truncated.
    If they are shorted than target length they are zero-padded.
    A zero padding mask is returned as the second output.
    """
    mask = torch.ones(target_length, dtype=torch.bool)
    dim, current_length = tensor.shape
    if current_length > target_length:
        tensor = tensor[:, :target_length]
    elif current_length < target_length:
        pad = torch.zeros(dim, target_length - current_length)
        tensor = torch.cat([tensor, pad], dim=-1)
        mask[current_length:] = False
    return tensor, mask


class FourierSeries(Dataset):

    def __init__(self, 
                 samples_per_class, 
                 min_seq_len=30, 
                 max_seq_len=100,
                 time_span=1000,
                 rseed=1234):

        torch.manual_seed(rseed)
        data = []
        self.time_span = time_span
        # Class 1 is sine waves with random lengths, freqs and phases
        lengths = (torch.rand(samples_per_class)*(max_seq_len - min_seq_len)).to(torch.int) + min_seq_len
        freqs = torch.rand(samples_per_class)*10/time_span + 1/time_span
        phases = torch.pi*torch.rand(samples_per_class)
        for length, freq, phase in zip(lengths, freqs, phases):
            data.append(self._synthesize_fourier_series(length, [1], freq, phase))
        # Class 2 is a three harmonic fourier
        lengths = (torch.rand(samples_per_class)*(max_seq_len - min_seq_len)).to(torch.int) + min_seq_len
        freqs = torch.rand(samples_per_class)*10/time_span + 1/time_span
        phases = torch.pi*torch.rand(samples_per_class)
        for length, freq, phase in zip(lengths, freqs, phases):
            data.append(self._synthesize_fourier_series(length, [1, 0.5, 0.25], freq, phase))

        self.data = data
        self.label = [0]*samples_per_class + [1]*samples_per_class

    def _synthesize_fourier_series(self, length, amplitudes, fundamental_freq, phase):
        time = torch.rand(length).sort()[0]*self.time_span
        value = 0.0
        for k, A in enumerate(amplitudes):
            value += amplitudes[k]*torch.sin(2*torch.pi*time*(k+1)*fundamental_freq + phase)
        value = (value - value.mean())/value.std()
        return torch.stack([time, value])
    
    def __getitem__(self, idx):
        sample = {}
        sample['label'] = self.label[idx]
        sample['light_curve'] = self.data[idx]
        return sample

    def __len__(self):
        return len(self.data)