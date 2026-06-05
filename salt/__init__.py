from salt.dataset import *
from salt.metrics import *
from salt.preprocessing import *
from salt.utils import *

from salt.dataset import __all__ as _dataset_all
from salt.metrics import __all__ as _metrics_all
from salt.preprocessing import __all__ as _preprocessing_all
from salt.utils import __all__ as _utils_all

__version__ = "0.1.1"

__all__ = [
    "__version__",
    *_dataset_all,
    *_metrics_all,
    *_preprocessing_all,
    *_utils_all,
]
