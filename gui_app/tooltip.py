"""
Всплывающие подсказки при наведении мыши на виджет
"""
import tkinter as tk


class ToolTip:
    """Всплывающая подсказка при наведении мыши на виджет."""

    def __init__(self, widget, text, delay=400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window = None
        self.after_id = None
        widget.bind('<Enter>', self._schedule)
        widget.bind('<Leave>', self._hide)
        widget.bind('<ButtonPress>', self._hide)

    def _schedule(self, event=None):
        self._hide()
        self.after_id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(bg='#333333')
        label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background='#ffffee', foreground='#222222',
            relief=tk.SOLID, borderwidth=1,
            font=('Segoe UI', 9), padx=8, pady=6,
            wraplength=380,
        )
        label.pack()
        self.tip_window = tw

    def _hide(self, event=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None