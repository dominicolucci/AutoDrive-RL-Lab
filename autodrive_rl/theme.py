"""Visual theme for the launcher.

The colours are imported from the renderer rather than redefined, so the
launcher and the simulation it launches are literally the same palette. A
duplicated hex table would drift the first time either side was retouched.

Everything here is `ttk` styling. Tk's stock widgets look like 1998 on every
platform; `clam` is the one built-in theme whose elements accept full colour
control, so it is used as the base and then almost entirely repainted.
"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from .renderer import (
    ACCENT,
    ACCENT_DEEP,
    BG,
    CARD_BG,
    CARD_EDGE,
    PANEL_BG,
    TEXT_DIM,
    TEXT_MAIN,
)

# Derived tones the launcher needs that the renderer has no use for.
FIELD_BG = "#101823"
FIELD_EDGE = "#26364a"
HOVER = "#1b2836"
SELECT_BG = "#1d3348"
OK = "#57cc99"
WARN = "#e9a13b"
ERROR = "#e5383b"
LOG_BG = "#080c12"

__all__ = [
    "ACCENT", "ACCENT_DEEP", "BG", "CARD_BG", "CARD_EDGE", "PANEL_BG",
    "TEXT_DIM", "TEXT_MAIN", "FIELD_BG", "FIELD_EDGE", "HOVER", "SELECT_BG",
    "OK", "WARN", "ERROR", "LOG_BG", "apply_theme", "enable_dpi_awareness",
]


def enable_dpi_awareness() -> None:
    """Stop Windows from bitmap-scaling the window into a blurry mess.

    On a high-DPI display Windows will happily upscale a non-aware application,
    which makes every glyph soft. Harmless and silent everywhere else.
    """
    if sys.platform != "win32":
        return
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001 - cosmetic only, never worth crashing over
        pass


def _fonts(root: tk.Misc) -> dict[str, tkfont.Font]:
    """A small type scale, using whatever the platform actually has."""
    families = set(tkfont.families(root))

    def pick(*candidates: str, fallback: str) -> str:
        for name in candidates:
            if name in families:
                return name
        return fallback

    ui = pick("Segoe UI", "Inter", "Helvetica Neue", "DejaVu Sans", fallback="TkDefaultFont")
    mono = pick("Cascadia Mono", "Consolas", "JetBrains Mono", "DejaVu Sans Mono",
                fallback="TkFixedFont")

    return {
        "title": tkfont.Font(root=root, family=ui, size=16, weight="bold"),
        "tab": tkfont.Font(root=root, family=ui, size=10, weight="bold"),
        "label": tkfont.Font(root=root, family=ui, size=10),
        "hint": tkfont.Font(root=root, family=ui, size=9),
        "button": tkfont.Font(root=root, family=ui, size=10, weight="bold"),
        "mono": tkfont.Font(root=root, family=mono, size=9),
        "status": tkfont.Font(root=root, family=ui, size=10, weight="bold"),
    }


def apply_theme(root: tk.Tk) -> dict[str, tkfont.Font]:
    """Repaint ttk to match the simulation. Returns the font scale."""
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    fonts = _fonts(root)
    root.configure(background=BG)

    # ── surfaces ─────────────────────────────────────────────────────────────
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=CARD_BG, relief="flat")
    style.configure("Bar.TFrame", background=PANEL_BG)

    # ── text ─────────────────────────────────────────────────────────────────
    style.configure("TLabel", background=BG, foreground=TEXT_MAIN, font=fonts["label"])
    style.configure("Card.TLabel", background=CARD_BG, foreground=TEXT_MAIN, font=fonts["label"])
    style.configure("Hint.TLabel", background=CARD_BG, foreground=TEXT_DIM, font=fonts["hint"])
    style.configure("Intro.TLabel", background=CARD_BG, foreground=TEXT_DIM, font=fonts["label"])
    style.configure("Title.TLabel", background=PANEL_BG, foreground=TEXT_MAIN, font=fonts["title"])
    style.configure("Subtitle.TLabel", background=PANEL_BG, foreground=TEXT_DIM, font=fonts["hint"])
    style.configure("Caption.TLabel", background=BG, foreground=TEXT_DIM, font=fonts["hint"])
    style.configure("Status.TLabel", background=BG, foreground=TEXT_DIM, font=fonts["status"])

    # ── tabs ─────────────────────────────────────────────────────────────────
    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(2, 4, 2, 0))
    style.configure(
        "TNotebook.Tab",
        background=BG,
        foreground=TEXT_DIM,
        font=fonts["tab"],
        padding=(20, 10),
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", CARD_BG), ("active", HOVER)],
        foreground=[("selected", ACCENT), ("active", TEXT_MAIN)],
        # clam lifts the selected tab by default, which clips it against the
        # notebook edge. Keep every tab on the same baseline.
        expand=[("selected", [0, 0, 0, 0])],
    )

    # ── buttons ──────────────────────────────────────────────────────────────
    style.configure(
        "TButton",
        background=FIELD_BG,
        foreground=TEXT_MAIN,
        font=fonts["button"],
        borderwidth=1,
        focusthickness=0,
        padding=(14, 8),
        relief="flat",
    )
    style.map(
        "TButton",
        background=[("active", HOVER), ("disabled", PANEL_BG)],
        foreground=[("disabled", "#3d4a58")],
        bordercolor=[("!disabled", FIELD_EDGE)],
    )
    # The primary action carries the accent so the eye lands on it first.
    style.configure(
        "Accent.TButton",
        background=ACCENT_DEEP,
        foreground="#f6fbff",
        font=fonts["button"],
        borderwidth=0,
        padding=(20, 9),
        relief="flat",
    )
    style.map(
        "Accent.TButton",
        background=[("active", ACCENT), ("disabled", "#20303f")],
        foreground=[("disabled", "#54636f")],
    )

    # ── inputs ───────────────────────────────────────────────────────────────
    style.configure(
        "TEntry",
        fieldbackground=FIELD_BG,
        background=FIELD_BG,
        foreground=TEXT_MAIN,
        insertcolor=ACCENT,
        bordercolor=FIELD_EDGE,
        lightcolor=FIELD_EDGE,
        darkcolor=FIELD_EDGE,
        borderwidth=1,
        padding=6,
        relief="flat",
    )
    style.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)])

    style.configure(
        "TCombobox",
        fieldbackground=FIELD_BG,
        background=FIELD_BG,
        foreground=TEXT_MAIN,
        arrowcolor=TEXT_DIM,
        bordercolor=FIELD_EDGE,
        lightcolor=FIELD_EDGE,
        darkcolor=FIELD_EDGE,
        borderwidth=1,
        padding=5,
        relief="flat",
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", FIELD_BG)],
        foreground=[("readonly", TEXT_MAIN)],
        bordercolor=[("focus", ACCENT)],
        arrowcolor=[("active", ACCENT)],
    )
    # The dropdown list is a classic Tk listbox and ignores ttk styling.
    root.option_add("*TCombobox*Listbox.background", FIELD_BG)
    root.option_add("*TCombobox*Listbox.foreground", TEXT_MAIN)
    root.option_add("*TCombobox*Listbox.selectBackground", SELECT_BG)
    root.option_add("*TCombobox*Listbox.selectForeground", ACCENT)
    root.option_add("*TCombobox*Listbox.borderWidth", "0")

    # clam draws the tick with `indicatorforeground` on an `indicatorbackground`
    # fill — `indicatorcolor` (the default theme's option) is ignored here, which
    # is why a checked box can otherwise look identical to an unchecked one.
    style.configure(
        "TCheckbutton",
        background=CARD_BG,
        foreground=TEXT_MAIN,
        font=fonts["label"],
        indicatorbackground=FIELD_BG,
        indicatorforeground=BG,
        indicatorsize=12,
        indicatormargin=(0, 0, 8, 0),
        upperbordercolor=FIELD_EDGE,
        lowerbordercolor=FIELD_EDGE,
        focusthickness=0,
        padding=(0, 3),
    )
    style.map(
        "TCheckbutton",
        background=[("active", CARD_BG)],
        foreground=[("active", ACCENT)],
        indicatorbackground=[
            ("selected", "!disabled", ACCENT),
            ("active", "!selected", HOVER),
            ("!selected", FIELD_BG),
        ],
        indicatorforeground=[("selected", BG)],
        upperbordercolor=[("selected", ACCENT), ("active", ACCENT)],
        lowerbordercolor=[("selected", ACCENT), ("active", ACCENT)],
    )

    # ── scrollbar ────────────────────────────────────────────────────────────
    style.configure(
        "Vertical.TScrollbar",
        background=FIELD_EDGE,
        troughcolor=LOG_BG,
        bordercolor=LOG_BG,
        arrowcolor=TEXT_DIM,
        borderwidth=0,
        relief="flat",
        gripcount=0,  # clam draws grip ridges on the thumb by default
        arrowsize=12,
    )
    style.map("Vertical.TScrollbar", background=[("active", HOVER)])

    style.configure("Separator.TFrame", background=CARD_EDGE)

    return fonts


def style_listbox(widget: tk.Listbox, fonts: dict[str, tkfont.Font]) -> None:
    """Listbox is a classic widget — it has to be coloured by hand."""
    widget.configure(
        background=FIELD_BG,
        foreground=TEXT_MAIN,
        selectbackground=SELECT_BG,
        selectforeground=ACCENT,
        highlightthickness=1,
        highlightbackground=FIELD_EDGE,
        highlightcolor=ACCENT,
        borderwidth=0,
        relief="flat",
        activestyle="none",
        font=fonts["label"],
    )


def style_text(widget: tk.Text, fonts: dict[str, tkfont.Font], *, log: bool = False) -> None:
    widget.configure(
        background=LOG_BG if log else FIELD_BG,
        foreground=TEXT_MAIN,
        insertbackground=ACCENT,
        selectbackground=SELECT_BG,
        highlightthickness=1,
        highlightbackground=FIELD_EDGE,
        highlightcolor=FIELD_EDGE,
        borderwidth=0,
        relief="flat",
        font=fonts["mono"],
        padx=10,
        pady=8,
    )
