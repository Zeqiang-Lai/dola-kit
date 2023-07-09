from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import functools
import numpy as np

FORMAT_HWC = 'hwc'
FORMAT_CHW = 'chw'
DATA_FORMAT = FORMAT_HWC

__all__ = [
    'psnr',
    'ssim',
    'sam',
    'mpsnr',
    'mssim'
]

def set_data_format(format):
    if format.lower() != FORMAT_HWC and format.lower() != FORMAT_CHW:
        raise ValueError('Invalid data format, choose from '
                         'torchlight.metrics.HWC or torchlight.metrics.CHW')
    global DATA_FORMAT
    DATA_FORMAT = format


def CHW2HWC(func):
    @functools.wraps(func)
    def warpped(output, target, *args, **kwargs):
        if DATA_FORMAT == FORMAT_CHW:
            output = output.transpose(1, 2, 0)
            target = target.transpose(1, 2, 0)
        return func(output, target, *args, **kwargs)
    return warpped


def torch2numpy(func):
    @functools.wraps(func)
    def warpped(output, target, *args, **kwargs):
        if not isinstance(output, np.ndarray):
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
        return func(output, target, *args, **kwargs)
    return warpped


def bandwise(func):
    @functools.wraps(func)
    def warpped(output, target, *args, **kwargs):
        if DATA_FORMAT == FORMAT_CHW:
            C = output.shape[-3]
            total = 0
            for ch in range(C):
                x = output[ch, :, :]
                y = target[ch, :, :]
                total += func(x, y, *args, **kwargs)
            return total / C
        else:
            C = output.shape[-1]
            total = 0
            for ch in range(C):
                x = output[:, :, ch]
                y = target[:, :, ch]
                total += func(x, y, *args, **kwargs)
            return total / C
    return warpped


def enable_batch_input(reduce=True):
    def inner(func):
        @functools.wraps(func)
        def warpped(output, target, *args, **kwargs):
            if len(output.shape) == 4:
                b = output.shape[0]
                out = [func(output[i], target[i]) for i in range(b)]
                if reduce:
                    return sum(out) / len(out)
                return out
            return func(output, target, *args, **kwargs)
        return warpped
    return inner


# raw psnr, ssim, sam assume HWC format


@torch2numpy
@enable_batch_input()
@CHW2HWC
def psnr(output, target, data_range=1):
    return peak_signal_noise_ratio(target, output, data_range=data_range)


@torch2numpy
@enable_batch_input()
@CHW2HWC
def ssim(img1, img2, **kwargs):
    return structural_similarity(img1, img2, channel_axis=2, **kwargs)


@torch2numpy
@enable_batch_input()
@CHW2HWC
def sam(img1, img2, eps=1e-8):
    """
    Spectral Angle Mapper which defines the spectral similarity between two spectra
    """
    tmp1 = np.sum(img1 * img2, axis=2) + eps
    tmp2 = np.sqrt(np.sum(img1**2, axis=2)) + eps
    tmp3 = np.sqrt(np.sum(img2**2, axis=2)) + eps
    tmp4 = tmp1 / tmp2 / tmp3
    angle = np.arccos(tmp4.clip(-1, 1))
    return np.mean(np.real(angle))


@torch2numpy
@enable_batch_input()
@bandwise
def mpsnr(output, target, data_range=1):
    return peak_signal_noise_ratio(target, output, data_range=data_range)


@torch2numpy
@enable_batch_input()
@bandwise
def mssim(img1, img2, **kwargs):
    return structural_similarity(img1, img2, **kwargs)
