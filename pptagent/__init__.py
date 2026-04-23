"""PPTAgent: Generating and Evaluating Presentations Beyond Text-to-Slides.

This package provides tools to automatically generate presentations from documents,
following a two-phase approach of Analysis and Generation.

For more information, visit: https://github.com/icip-cas/PPTAgent
"""

__version__ = "0.1.0"
__author__ = "Hao Zheng"
__email__ = "wszh712811@gmail.com"


# Check the version of python and python-pptx
import asyncio
import sys

# if sys.version_info < (3, 10):
#     raise ImportError("You should use Python 3.10 or higher for this project.")

# Polyfill asyncio.TaskGroup for Python < 3.11 so the async pipeline (induct,
# pptgen, multimodal, document, presentation.layout) works on 3.10.
if not hasattr(asyncio, "TaskGroup"):

    class _TaskGroup:
        def __init__(self):
            self._tasks: list[asyncio.Task] = []
            self._entered = False

        async def __aenter__(self):
            self._entered = True
            return self

        async def __aexit__(self, exc_type, exc, tb):
            if exc is not None:
                for t in self._tasks:
                    if not t.done():
                        t.cancel()
                return False
            if not self._tasks:
                return False
            try:
                await asyncio.gather(*self._tasks)
            except BaseException:
                for t in self._tasks:
                    if not t.done():
                        t.cancel()
                raise
            return False

        def create_task(self, coro, *, name=None):
            assert self._entered, "TaskGroup must be entered with `async with`"
            task = (
                asyncio.create_task(coro, name=name)
                if name is not None
                else asyncio.create_task(coro)
            )
            self._tasks.append(task)
            return task

    asyncio.TaskGroup = _TaskGroup

from packaging.version import Version
from pptx import __version__ as PPTXVersion

try:
    PPTXVersion, Mark = PPTXVersion.split("+")
    assert Version(PPTXVersion) >= Version("1.0.4") and Mark == "PPTAgent"
except:
    raise ImportError(
        "You should install the customized `python-pptx` for this project: Force1ess/python-pptx, but got %s."
        % PPTXVersion
    )

# Import main modules to make them directly accessible when importing the package
from .agent import *
from .apis import *
from .document import *
from .induct import *
from .llms import *
from .model_utils import *
from .multimodal import *
from .pptgen import *
from .presentation import *
from .utils import *

# Define the top-level exports
__all__ = [
    "agent",
    "pptgen",
    "document",
    "llms",
    "presentation",
    "utils",
    "apis",
    "model_utils",
    "multimodal",
    "induct",
]
