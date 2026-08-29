from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent / "src"))

from examples.mock_router_demo import main


if __name__ == "__main__":
    main()
