"""
MetAIsAFe — utils/plot_utils.py
=================================
Shared figure utilities: saving and interactive environment detection.

Public API
----------
    is_interactive()              -> bool
    save_figure(fig, path, ...)   -> Path

Author : MetAIsAFe team
Version: 3.2
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt

from config import FIGURE_DPI, FIGURE_FORMAT


def is_interactive() -> bool:
    """
    Detect whether code is running in an interactive environment.

    Returns ``True`` for Jupyter notebooks and IPython terminals,
    ``False`` for scripts and pipeline runs.

    Returns
    -------
    bool
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in (
            "ZMQInteractiveShell",     # Jupyter notebook / lab
            "TerminalInteractiveShell", # IPython terminal
        )
    except ImportError:
        return False


def save_figure(
    fig: plt.Figure,
    save_path: Path,
    show: bool | None = None,
    dpi: int = FIGURE_DPI,
    fmt: str = FIGURE_FORMAT,
    close: bool = True,
) -> Path:
    """
    Save a matplotlib figure to disk, then optionally display or close it.

    Parameters
    ----------
    fig : plt.Figure
    save_path : Path
        If suffix is missing, ``FIGURE_FORMAT`` is appended automatically.
    show : bool or None
        ``True``  → always call ``plt.show()``
        ``False`` → never display
        ``None``  → auto: display only in interactive environments
    dpi : int
    fmt : str
        Format string (e.g. ``"png"``, ``"pdf"``).
    close : bool
        If ``True`` and ``show=False``, call ``plt.close(fig)`` to free memory.

    Returns
    -------
    Path
        Resolved save path.
    """
    save_path = Path(save_path)

    if not save_path.suffix:
        save_path = save_path.with_suffix(f".{fmt}")
    elif save_path.suffix.lstrip(".").lower() != fmt.lower():
        warnings.warn(
            f"save_figure: path extension '{save_path.suffix}' differs from "
            f"fmt='{fmt}'. File saved in '{save_path.suffix.lstrip('.')}' format.",
            UserWarning,
            stacklevel=2,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"  💾 Saved: {save_path}")

    _show = is_interactive() if show is None else show
    if _show:
        plt.show()
    elif close:
        plt.close(fig)

    return save_path
