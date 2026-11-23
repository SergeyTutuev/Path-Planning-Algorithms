"""plot_overlay.py — оверлей кругов-аппроксимаций препятствий поверх ir-sim."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection


class CirclesOverlay:
    """Один EllipseCollection для быстрой отрисовки N окружностей.

    Использование:

        overlay = CirclesOverlay(plt.gca())
        ...
        world.step()
        overlay.update(world.objects[0]._debug_circles)
        world.render()
    """

    def __init__(self, ax, color="orange", alpha=0.40, linewidth=0.6, zorder=10):
        self.ax = ax
        self._coll = EllipseCollection(
            widths=[],
            heights=[],
            angles=[],
            units="xy",
            offsets=[],
            transOffset=ax.transData,
            facecolors="none",
            edgecolors=color,
            alpha=alpha,
            linewidths=linewidth,
            zorder=zorder,
        )
        ax.add_collection(self._coll)

    def clear(self):
        self._coll.set_offsets(np.empty((0, 2)))
        self._coll.set_widths([])
        self._coll.set_heights([])
        self._coll.set_angles([])

    def update(self, circles):
        """circles: iterable of (x, y, r) or (x, y, vx, vy, r)."""
        if not circles:
            self.clear()
            return

        arr = np.asarray(
            [[c[0], c[1], c[-1] if len(c) >= 5 else c[2]] for c in circles],
            dtype=float,
        )
        n = arr.shape[0]
        self._coll.set_offsets(arr[:, :2])
        self._coll.set_widths(2.0 * arr[:, 2])
        self._coll.set_heights(2.0 * arr[:, 2])
        self._coll.set_angles(np.zeros(n))