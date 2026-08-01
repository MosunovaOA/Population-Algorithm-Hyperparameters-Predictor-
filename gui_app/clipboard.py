"""
Обработка буфера обмена для текстовых полей
"""
import tkinter as tk
from tkinter import ttk


class ClipboardManager:
    """Менеджер для работы с буфером обмена"""

    @staticmethod
    def _clip_copy(w):
        try:
            if isinstance(w, tk.Text):
                sel = w.get(tk.SEL_FIRST, tk.SEL_LAST)
                w.clipboard_clear()
                w.clipboard_append(sel)
            elif isinstance(w, (tk.Entry, ttk.Entry)):
                if w.selection_present():
                    sel = w.selection_get()
                    w.clipboard_clear()
                    w.clipboard_append(sel)
        except (tk.TclError, Exception):
            pass
        return 'break'

    @staticmethod
    def _clip_paste(w):
        if isinstance(w, tk.Text):
            try:
                if w.cget('state') == tk.DISABLED:
                    return 'break'
            except Exception:
                pass
        try:
            clip = w.clipboard_get()
            if isinstance(w, tk.Text):
                try:
                    w.delete(tk.SEL_FIRST, tk.SEL_LAST)
                except tk.TclError:
                    pass
                w.insert(tk.INSERT, clip)
            elif isinstance(w, (tk.Entry, ttk.Entry)):
                try:
                    if w.selection_present():
                        w.delete(tk.SEL_FIRST, tk.SEL_LAST)
                except Exception:
                    pass
                w.insert(tk.INSERT, clip)
        except (tk.TclError, Exception):
            pass
        return 'break'

    @staticmethod
    def _clip_cut(w):
        if isinstance(w, tk.Text):
            try:
                if w.cget('state') == tk.DISABLED:
                    return ClipboardManager._clip_copy(w)
            except Exception:
                pass
        try:
            if isinstance(w, tk.Text):
                sel = w.get(tk.SEL_FIRST, tk.SEL_LAST)
                w.clipboard_clear()
                w.clipboard_append(sel)
                w.delete(tk.SEL_FIRST, tk.SEL_LAST)
            elif isinstance(w, (tk.Entry, ttk.Entry)):
                if w.selection_present():
                    sel = w.selection_get()
                    w.clipboard_clear()
                    w.clipboard_append(sel)
                    w.delete(tk.SEL_FIRST, tk.SEL_LAST)
        except (tk.TclError, Exception):
            pass
        return 'break'

    @staticmethod
    def _clip_select_all(w):
        if isinstance(w, tk.Text):
            w.tag_add(tk.SEL, '1.0', tk.END)
            w.mark_set(tk.INSERT, tk.END)
        elif isinstance(w, (tk.Entry, ttk.Entry)):
            w.select_range(0, tk.END)
            w.icursor(tk.END)
        return 'break'

    @classmethod
    def setup_clipboard_handlers(cls, root):
        """Настройка обработчиков для буфера обмена"""

        def _handle_key(event):
            if not (event.state & 0x4):
                return

            w = event.widget
            kc = event.keycode
            ks = event.keysym.lower()

            is_copy = (ks == 'c' or kc == 67)
            is_paste = (ks == 'v' or kc == 86)
            is_cut = (ks == 'x' or kc == 88)
            is_selall = (ks == 'a' or kc == 65)

            if is_copy:
                return cls._clip_copy(w)
            elif is_paste:
                return cls._clip_paste(w)
            elif is_cut:
                return cls._clip_cut(w)
            elif is_selall:
                return cls._clip_select_all(w)

        root.bind_all('<Key>', _handle_key)