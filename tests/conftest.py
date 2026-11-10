"""Force a headless Matplotlib backend before any test imports pyplot."""

import matplotlib

matplotlib.use("Agg")
