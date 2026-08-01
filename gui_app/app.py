"""
Основной класс GUI-приложения
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import threading
import os

from Population_Algorithm import PopulationAlgorithm
from visualizer import show_prediction_visualization

from .function_parser import FunctionParser
from .function_utils import make_safe_function, register_custom_function
from .predictor import load_predictor, predict_params
from .feature_info import FEATURE_INFO
from .tooltip import ToolTip
from .clipboard import ClipboardManager


def resource_path(relative_path):
    """Путь к ресурсу — работает и в .exe, и при обычном запуске."""
    import sys
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)


class population_algorithmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parameters Predictor for Population Algorithm")
        self.root.resizable(True, True)

        self.predictor = None
        self.parsed_features = None

        self._build_ui()
        ClipboardManager.setup_clipboard_handlers(root)
        self._load_model()

        self.root.update_idletasks()
        screen_h = self.root.winfo_screenheight()
        win_h = min(960, screen_h - 80)
        self.root.geometry(f"960x{win_h}")

    def _build_ui(self):
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'))
        style.configure('Result.TLabel', font=('Consolas', 10))
        style.configure('Big.TButton', font=('Segoe UI', 10, 'bold'), padding=6)
        style.configure('Hint.TLabel', font=('Segoe UI', 8), foreground='#666666')

        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        main = ttk.Frame(canvas, padding=10)
        win_id = canvas.create_window((0, 0), window=main, anchor=tk.NW)

        def _on_configure(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_resize(event):
            canvas.itemconfig(win_id, width=event.width)

        main.bind('<Configure>', _on_configure)
        canvas.bind('<Configure>', _on_canvas_resize)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all('<MouseWheel>', _on_mousewheel)

        self._canvas = canvas
        self._main = main

        frame1 = ttk.LabelFrame(main, text="  1. Функция  ", padding=10)
        frame1.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(frame1, text="f(x) =").pack(anchor=tk.W)
        self.formula_text = scrolledtext.ScrolledText(
            frame1, height=3, font=('Consolas', 10), wrap=tk.WORD
        )
        self.formula_text.pack(fill=tk.X, pady=(2, 6))
        self.formula_text.insert(
            tk.END,
            "10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))"
        )

        ex_frame = ttk.Frame(frame1)
        ex_frame.pack(fill=tk.X)
        ttk.Label(ex_frame, text="Примеры:", font=('Segoe UI', 8)).pack(side=tk.LEFT)

        examples = {
            "Sphere":
                ("sum([x[i]**2 for i in range(len(x))])", -100, 100),

            "Rosenbrock":
                ("sum([100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1)])",
                 -30, 30),

            "Rastrigin":
                ("10*len(x) + sum([x[i]**2 - 10*cos(2*pi*x[i]) for i in range(len(x))])",
                 -5.12, 5.12),

            "Griewank":
                ("sum([x[i]**2 for i in range(len(x))])/4000 - prod([cos(x[i]/(i+1)**0.5) for i in range(len(x))]) + 1",
                 -600, 600),

            "Ackley":
                ("-20*exp(-0.2*(sum([x[i]**2 for i in range(len(x))])/len(x))**0.5) - exp(sum([cos(2*pi*x[i]) for i in range(len(x))])/len(x)) + 20 + e",
                 -32.768, 32.768),

            "Levy":
                ("sin(pi*(1+(x[0]-1)/4))**2 + sum([((x[i]-1)/4)**2*(1+10*sin(pi*(1+(x[i]-1)/4)+1)**2) for i in range(len(x)-1)]) + ((x[len(x)-1]-1)/4)**2*(1+sin(2*pi*(1+(x[len(x)-1]-1)/4))**2)",
                 -10, 10),

            "Damped Schwefel":
                ("sum([abs(x[i]) - x[i]*sin(abs(x[i])**0.5) for i in range(len(x))])",
                 -500, 500),

            "Michalewicz":
                ("-sum([sin(x[i])*sin((i+1)*x[i]**2/pi)**20 for i in range(len(x))])",
                 0, 3.1416),

            "Eggcrate":
                ("sum([x[i]**2 + 25*sin(x[i])**2 for i in range(len(x))])",
                 -5, 5),
        }

        for name, (formula, lb, ub) in examples.items():
            btn = ttk.Button(
                ex_frame, text=name,
                command=lambda f=formula, l=lb, u=ub: self._set_example(f, l, u)
            )
            btn.pack(side=tk.LEFT, padx=2)

        frame2 = ttk.LabelFrame(main, text="  2. Параметры задачи и признаки  ", padding=10)
        frame2.pack(fill=tk.X, pady=(0, 8))

        row = ttk.Frame(frame2)
        row.pack(fill=tk.X, pady=(0, 4))

        lbl_dim = ttk.Label(row, text="Размерность D:")
        lbl_dim.pack(side=tk.LEFT)
        ToolTip(lbl_dim,
                "Количество переменных в векторе x.\n"
                "Например, D=3 означает x = [x₁, x₂, x₃].\n"
                "Чем выше D, тем сложнее задача оптимизации.")
        self.dim_var = tk.StringVar(value="3")
        dim_combo = ttk.Combobox(
            row, textvariable=self.dim_var,
            values=["3", "4", "5"], width=4, state="readonly"
        )
        dim_combo.pack(side=tk.LEFT, padx=(4, 20))

        lbl_bounds = ttk.Label(row, text="Границы: от")
        lbl_bounds.pack(side=tk.LEFT)
        ToolTip(lbl_bounds,
                "Область поиска [lb, ub] для каждой переменной.\n"
                "Все частицы инициализируются внутри этих границ.\n"
                "Примеры:\n"
                "  Rastrigin: [-5.12, 5.12]\n"
                "  Schwefel:  [-500, 500]\n"
                "  Ackley:    [-32.768, 32.768]")
        self.lb_var = tk.StringVar(value="-5.12")
        ttk.Entry(row, textvariable=self.lb_var, width=8).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Label(row, text="до").pack(side=tk.LEFT)
        self.ub_var = tk.StringVar(value="5.12")
        ttk.Entry(row, textvariable=self.ub_var, width=8).pack(side=tk.LEFT, padx=(2, 20))

        lbl_gmin = ttk.Label(row, text="Global min (опц.):")
        lbl_gmin.pack(side=tk.LEFT)
        ToolTip(lbl_gmin,
                "Известное минимальное значение функции.\n"
                "Используется ТОЛЬКО для подсчёта Success Rate\n"
                "(доли запусков, сошедшихся к глобальному минимуму).\n\n"
                "Если не задано — популяционный алгоритм просто\n"
                "ищет минимум, SR не вычисляется.\n\n"
                "Примеры: Sphere→0, Rastrigin→0, Ackley→0")
        self.gmin_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.gmin_var, width=8).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Label(
            frame2,
            text="Global min нужен только для оценки точности. "
                 "Если неизвестен — оставьте пустым.",
            style='Hint.TLabel'
        ).pack(anchor=tk.W, pady=(0, 6))

        ttk.Button(
            frame2,
            text="▶  Распознать признаки функции и предсказать оптимальные параметры",
            style='Big.TButton', command=self._on_parse
        ).pack(fill=tk.X, pady=(2, 6))

        ttk.Button(
            frame2,
            text="▶  Показать 3D-визуализацию предсказаний",
            style='Big.TButton', command=self.show_visualization
        ).pack(fill=tk.X, pady=(2, 6))

        self.features_frame = ttk.Frame(frame2)
        self.features_frame.pack(fill=tk.X, anchor=tk.W)

        frame3 = ttk.LabelFrame(
            main,
            text="  3. Параметры популяционного алгоритма (можно править вручную)  ",
            padding=10
        )
        frame3.pack(fill=tk.X, pady=(0, 8))

        self.params_label = ttk.Label(
            frame3, text="", style='Result.TLabel', justify=tk.LEFT
        )
        self.params_label.pack(anchor=tk.W, pady=(0, 6))

        population_algorithm_row1 = ttk.Frame(frame3)
        population_algorithm_row1.pack(fill=tk.X, pady=(0, 4))

        w_box = ttk.Frame(population_algorithm_row1)
        w_box.pack(side=tk.LEFT, padx=(0, 24))
        lbl_w = ttk.Label(w_box, text="bI (инерциальный):")
        lbl_w.pack(anchor=tk.W)
        ToolTip(lbl_w,
                "Коэффициент инерции — вес предыдущей скорости.\n\n"
                "• bI > 1.0 — частицы разгоняются, широкий поиск\n"
                "• bI ≈ 0.7–0.9 — баланс разведки и сходимости\n"
                "• bI < 0.4 — локальная доводка")
        self.w_var = tk.StringVar(value="0.5")
        ttk.Entry(w_box, textvariable=self.w_var, width=8).pack(anchor=tk.W)

        c1_box = ttk.Frame(population_algorithm_row1)
        c1_box.pack(side=tk.LEFT, padx=(0, 24))
        lbl_c1 = ttk.Label(c1_box, text="bC (когнитивный):")
        lbl_c1.pack(anchor=tk.W)
        ToolTip(lbl_c1,
                "Когнитивный коэффициент — притяжение к личному\n"
                "лучшему положению частицы (pbest).\n\n"
                "• Большое bC — частица доверяет своему опыту\n"
                "• Малое bC — частица больше полагается на коллективный опыт")
        self.c1_var = tk.StringVar(value="1.5")
        ttk.Entry(c1_box, textvariable=self.c1_var, width=8).pack(anchor=tk.W)

        c2_box = ttk.Frame(population_algorithm_row1)
        c2_box.pack(side=tk.LEFT)
        lbl_c2 = ttk.Label(c2_box, text="bS (социальный):")
        lbl_c2.pack(anchor=tk.W)
        ToolTip(lbl_c2,
                "Социальный коэффициент — притяжение к лучшему\n"
                "положению (лидера) во всей популяции (gbest).\n\n"
                "• Большое bS — быстрая сходимость к лидеру\n"
                "• Малое bS — сходимость медленнее, но лучше разведка")
        self.c2_var = tk.StringVar(value="1.5")
        ttk.Entry(c2_box, textvariable=self.c2_var, width=8).pack(anchor=tk.W)

        population_algorithm_row2 = ttk.Frame(frame3)
        population_algorithm_row2.pack(fill=tk.X, pady=(6, 0))

        pop_box = ttk.Frame(population_algorithm_row2)
        pop_box.pack(side=tk.LEFT, padx=(0, 24))
        lbl_pop = ttk.Label(pop_box, text="Размер популяции:")
        lbl_pop.pack(anchor=tk.W)
        ToolTip(lbl_pop,
                "Размер популяции — количество частиц в популяции.\n\n"
                "Больше частиц = лучше покрытие, но медленнее алгоритм.")
        self.pop_var = tk.StringVar(value="30")
        ttk.Entry(pop_box, textvariable=self.pop_var, width=6).pack(anchor=tk.W)

        iter_box = ttk.Frame(population_algorithm_row2)
        iter_box.pack(side=tk.LEFT, padx=(0, 24))
        lbl_iter = ttk.Label(iter_box, text="Макс. число итераций:")
        lbl_iter.pack(anchor=tk.W)
        ToolTip(lbl_iter,
                "Максимальное число итераций популяционного алгоритма.\n\n"
                "• Может остановиться раньше по стагнации\n"
                "  (30 итераций без улучшения)")
        self.iter_var = tk.StringVar(value="300")
        ttk.Entry(iter_box, textvariable=self.iter_var, width=6).pack(anchor=tk.W)

        runs_box = ttk.Frame(population_algorithm_row2)
        runs_box.pack(side=tk.LEFT)
        lbl_runs = ttk.Label(runs_box, text="Повторные запуски:")
        lbl_runs.pack(anchor=tk.W)
        ToolTip(lbl_runs,
                "Количество повторных запусков популяционного\n"
                "алгорима со случайной инициализацией.")
        self.runs_var = tk.StringVar(value="10")
        ttk.Entry(runs_box, textvariable=self.runs_var, width=6).pack(anchor=tk.W)

        frame4 = ttk.LabelFrame(main, text="  4. Поиск минимума  ", padding=10)
        frame4.pack(fill=tk.X, pady=(0, 8))

        btn_row = ttk.Frame(frame4)
        btn_row.pack(fill=tk.X, pady=(0, 6))

        self.btn_run = ttk.Button(
            btn_row, text="▶  Запустить популяционный алгоритм",
            style='Big.TButton', command=self._on_run_population_algorithm
        )
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress = ttk.Progressbar(btn_row, length=150, mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=(10, 0))

        self.log_text = scrolledtext.ScrolledText(
            frame4, height=16, font=('Consolas', 9),
            state=tk.DISABLED, bg='#1e1e1e', fg='#d4d4d4'
        )
        self.log_text.pack(fill=tk.X)

    def _set_example(self, formula, lb, ub):
        self.formula_text.delete('1.0', tk.END)
        self.formula_text.insert(tk.END, formula)
        self.lb_var.set(str(lb))
        self.ub_var.set(str(ub))
        self.gmin_var.set("")

    def _log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + '\n')
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def _clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _load_model(self):
        try:
            self.predictor = load_predictor(resource_path('saved_model_v10_simple'))
            self._log("   Модель загружена из saved_model_v10_simple/")
            self._log(f"   Компоненты: scaler_X, cls_alive, cls_good, "
                      f"reg_sr, reg_iter, meta_regressor")
            self._log(f"   Признаков: {self.predictor.scaler_X.n_features_in_}")
        except Exception as e:
            self._log(f"  Не удалось загрузить модель: {e}")
            self._log("   Предсказание недоступно — задавайте параметры вручную.")

    def _show_features_detailed(self, features):
        for widget in self.features_frame.winfo_children():
            widget.destroy()

        feature_order = [
            'is_multimodal', 'is_periodic', 'is_separable',
            'has_product_terms', 'trig_nesting_depth', 'trig_frequency',
        ]

        for key in feature_order:
            value = features[key]
            info = FEATURE_INFO[key]

            row = ttk.Frame(self.features_frame)
            row.pack(fill=tk.X, pady=1)

            icon = ""
            val_str = f"{value:.2f}" if isinstance(value, float) else str(value)

            lbl_main = ttk.Label(
                row,
                text=f"  {icon} {info['title']}: {val_str}",
                font=('Consolas', 9, 'bold')
            )
            lbl_main.pack(side=tk.LEFT)

            lbl_hint = ttk.Label(
                row,
                text=f"  — {info['short']}",
                style='Hint.TLabel'
            )
            lbl_hint.pack(side=tk.LEFT)

            ToolTip(lbl_main, f"{info['title']}\n\n{info['full']}")
            ToolTip(lbl_hint, f"{info['title']}\n\n{info['full']}")

        self._main.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_parse(self):
        formula = self.formula_text.get('1.0', tk.END).strip()
        if not formula:
            messagebox.showwarning("Ошибка", "Введите формулу функции!")
            return

        try:
            parser = FunctionParser(formula)
            features = parser.parse_all()
            self.parsed_features = features
        except Exception as e:
            messagebox.showerror("Ошибка парсинга", str(e))
            return

        self._show_features_detailed(features)

        dim = int(self.dim_var.get())
        try:
            test_func = make_safe_function(formula)
            test_val = test_func(np.zeros(dim))
            self._log(f"   Функция корректна.")
        except Exception as e:
            messagebox.showerror("Ошибка функции", str(e))
            return

        if self.predictor is not None and self.predictor.is_loaded:
            try:
                feat_list = [
                    features['is_multimodal'],
                    features['is_periodic'],
                    features['is_separable'],
                    features['has_product_terms'],
                    features['trig_nesting_depth'],
                    features['trig_frequency'],
                ]
                params, sr, iter_pred = predict_params(self.predictor, feat_list, dim)
                w, c1, c2 = params

                self.w_var.set(f"{w:.4f}")
                self.c1_var.set(f"{c1:.4f}")
                self.c2_var.set(f"{c2:.4f}")

                self.params_label.config(
                    text=(f"  bI = {w:.1f}     bC = {c1:.1f}    bS = {c2:.1f}\n"
                          f"  Ожидаемый SR: {sr:.4f}\n"
                          f"  Ожидаемое число итераций: {round(iter_pred * 300)}")
                )
                self._log(f"   SR = {sr:.4f}, итераций = {round(iter_pred * 300)}")
            except Exception as e:
                self._log(f"    Ошибка предсказания: {e}")
                import traceback
                self._log(traceback.format_exc())
                self.params_label.config(text=f"Ошибка: {e}")
        else:
            self.params_label.config(
                text="Модель не загружена — задайте параметры вручную"
            )
            self._log("  Признаки распознаны. Задайте bI, bC, bS вручную.")

    def _on_run_population_algorithm(self):
        formula = self.formula_text.get('1.0', tk.END).strip()
        if not formula:
            messagebox.showwarning("Ошибка", "Введите формулу!")
            return

        try:
            dim    = int(self.dim_var.get())
            lb     = float(self.lb_var.get())
            ub     = float(self.ub_var.get())
            w      = float(self.w_var.get())
            c1     = float(self.c1_var.get())
            c2     = float(self.c2_var.get())
            n_pop  = int(self.pop_var.get())
            n_iter = int(self.iter_var.get())
            n_runs = int(self.runs_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте числовые поля!")
            return

        gmin_str = self.gmin_var.get().strip()
        gmin = None
        if gmin_str:
            try:
                gmin = float(gmin_str)
            except ValueError:
                messagebox.showerror("Ошибка", "Global min должен быть числом или пустым!")
                return

        try:
            func_name = register_custom_function(formula, (lb, ub), name="custom_gui")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Невалидная функция:\n{e}")
            return

        self._clear_log()
        self._log("=" * 60)
        self._log(f"  population_algorithm: D={dim}, bounds=[{lb}, {ub}]")
        if gmin is not None:
            self._log(f"  Global min (для SR): {gmin}")
        else:
            self._log(f"  Global min: не задан (SR не считается)")
        self._log(f"  bI={w:.3f}, bC={c1:.3f}, bS={c2:.3f}")
        self._log(f"  Размер популяции: {n_pop}, Количество итераций: {n_iter}, Число повторных запусков: {n_runs}")
        self._log("=" * 60)

        self.btn_run.config(state=tk.DISABLED)
        self.progress['value'] = 0

        def run_thread():
            try:
                successes = 0
                all_vals = []
                all_iters = []
                best_overall_pos = None
                best_overall_val = float('inf')

                for run in range(n_runs):
                    population_algorithm = PopulationAlgorithm(
                        func_name=func_name,
                        dim=dim,
                        population_size=n_pop,
                        max_iter=n_iter,
                        w=w,
                        c_cog=c1,
                        c_soc=c2,
                        c_attr=0,
                        c_hist=0,
                        c_worst=0,
                        tolerance=1e-10,
                        stagnation_limit=30,
                        stop_by_stagnation=True
                    )

                    best_pos, best_val = population_algorithm.optimize()
                    iterations = len(population_algorithm.history)

                    all_vals.append(best_val)
                    all_iters.append(iterations)

                    if best_val < best_overall_val:
                        best_overall_val = best_val
                        best_overall_pos = best_pos.copy()

                    if gmin is not None:
                        converged = abs(best_val - gmin) < 1e-4
                        if converged:
                            successes += 1
                        status = "✅" if converged else "❌"
                    else:
                        status = "▪️"

                    self._log(f"  {status} Run {run+1:2d}/{n_runs}: "
                              f"f = {best_val:.8f}, iter = {iterations}")

                    self.progress['value'] = (run + 1) / n_runs * 100
                    self.root.update_idletasks()

                self._log(f"\n{'=' * 60}")

                if gmin is not None:
                    sr = successes / n_runs
                    self._log(f"  SR = {sr:.1%} ({successes}/{n_runs})")

                self._log(f"  Лучшее значение:  {best_overall_val:.10f}")
                self._log(f"  Лучшая точка:     {np.round(best_overall_pos, 6)}")
                self._log(f"  Среднее значение: {np.mean(all_vals):.10f}")
                self._log(f"  Медиана значений: {np.median(all_vals):.10f}")
                self._log(f"  Среднее итераций: {np.mean(all_iters):.1f}")
                self._log(f"{'=' * 60}")

            except Exception as e:
                self._log(f"\n Ошибка: {e}")
                import traceback
                self._log(traceback.format_exc())
            finally:
                self.btn_run.config(state=tk.NORMAL)
                self.progress['value'] = 100

        thread = threading.Thread(target=run_thread, daemon=True)
        thread.start()

    def show_visualization(self):
        if self.predictor is None or not self.predictor.is_loaded:
            messagebox.showwarning("Ошибка", "Модель не загружена!")
            return

        if self.parsed_features is None:
            messagebox.showwarning("Ошибка", "Сначала распознайте признаки функции!")
            return

        try:
            dim = int(self.dim_var.get())
            formula = self.formula_text.get('1.0', tk.END).strip()

            feat_list = [
                self.parsed_features['is_multimodal'],
                self.parsed_features['is_periodic'],
                self.parsed_features['is_separable'],
                self.parsed_features['has_product_terms'],
                self.parsed_features['trig_nesting_depth'],
                self.parsed_features['trig_frequency'],
            ]

            import matplotlib.pyplot as plt
            plt.close('all')

            show_prediction_visualization(self.predictor, feat_list, dim, formula)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось запустить визуализацию:\n{e}")