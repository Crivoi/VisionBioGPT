import logging
import numpy
import random
import torch
from enum import Enum

logger = logging.getLogger(__name__)


class Splits(Enum):
    train = 'train'
    dev = 'dev'
    test = 'test'


class DataSamples(Enum):
    sample = 'sample'
    top50 = '50'
    full = 'full'


class ImgSizes(Enum):
    large = (384, 384)
    small = (224, 224)


class MimicCXRLabels(Enum):
    atelectasis = 'Atelectasis'
    cardiomegaly = 'Cardiomegaly'
    consolidation = 'Consolidation'
    edema = 'Edema'
    enlarged_cardiomediastinum = 'Enlarged Cardiomediastinum'
    fracture = 'Fracture'
    lung_lesion = 'Lung Lesion'
    lung_opacity = 'Lung Opacity'
    no_finding = 'No Finding'
    pleural_effusion = 'Pleural Effusion'
    pleural_other = 'Pleural Other'
    pneumonia = 'Pneumonia'
    pneumothorax = 'Pneumothorax'
    support_devices = 'Support Devices'


class MimicCXRViewPositions(Enum):
    postero_anterior = 'PA'
    lateral = 'LATERAL'
    antero_posterior = 'AP'
    left_lateral = 'LL'
    nan = 'NaN'
    left_anterior_oblique = 'LAO'
    right_anterior_oblique = 'RAO'
    ap_axial = 'AP AXIAL'
    swimmers = 'SWIMMERS'
    pa_lld = 'PA LLD'


def pad_sequences(sequences, max_length=None, dtype="int32", value=0):
    """[2022-Feb-17] https://github.com/LorrinWWW/Pyramid/blob/master/utils/data.py#L327"""
    batch_size = len(sequences)
    if max_length is None:
        max_length = max([len(s) for s in sequences])

    sample_shape = numpy.asarray(sequences[0]).shape[1:]
    padded_sequences = numpy.full((batch_size, max_length) + sample_shape, value, dtype=dtype)
    for i, s in enumerate(sequences):
        assert len(s) > 0
        trunc_s = numpy.asarray(s[:max_length], dtype=dtype)
        padded_sequences[i, :len(trunc_s)] = trunc_s

    return padded_sequences


def set_seed(seed=52):
    """[2022-Feb-17] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_utils.py#L50"""
    """Fix the random seed for reproducibility"""
    if seed < 0: return
    logger.debug(f"Random seed: {seed}")
    # os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cuda.matmul.allow_tf32 = False


if __name__ == '__main__':
    pass
