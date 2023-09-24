import lazy_loader as lazy

# this assumes there is a `.pyi` file adjacent to this module
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

from .convert import *
from .download import *
from .examples import *
from .io import *
from .metrics import *
from .os import *
from .plot import *
from .utils import *
