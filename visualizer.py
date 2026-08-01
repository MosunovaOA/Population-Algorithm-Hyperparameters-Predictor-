"""
Модуль визуализации предсказаний модели
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

MAX_ITERATIONS = 300.0

# Непрерывная цветовая шкала для SR
SR_CMAP = plt.cm.RdYlGn
SHOW_THRESHOLD_SR = 0.05

# Непрерывная цветовая шкала для итераций
ITER_CMAP = plt.cm.viridis


def render_scatter_sr(ax, grid, sr_values, title):
    """
    Отрисовка точек для Success Rate
    Показывает все точки с SR > 0
    """
    threshold = 0.0
    visible = sr_values > threshold
    indices = np.where(visible)[0]

    if len(indices) == 0:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        ax.text2D(0.5, 0.5, "нет данных выше порога",
                  transform=ax.transAxes, fontsize=11,
                  ha='center', va='center', color='#999999')
        return

    coords = grid[indices]
    sr_vis = sr_values[indices]

    # Сортируем для правильного наложения
    order = np.argsort(sr_vis)
    coords = coords[order]
    sr_vis = sr_vis[order]

    # Используем непрерывную цветовую шкалу (как в voxel_plots.py)
    colors = SR_CMAP(sr_vis)

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=colors, s=35, edgecolors='none', depthshade=True)

    _format_3d_axes(ax, title)


def render_scatter_iter(ax, grid, iter_values, sr_values, title):
    """
    Отрисовка точек для числа итераций
    """
    threshold = 0.0
    # Фильтр: только точки, где SR > 0 И итерации > 0
    visible = (sr_values > threshold) & (iter_values > 0)
    indices = np.where(visible)[0]

    if len(indices) == 0:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        ax.text2D(0.5, 0.5, f"нет данных (SR ≤ {threshold * 100:.0f}%)",
                  transform=ax.transAxes, fontsize=11,
                  ha='center', va='center', color='#999999')
        return 0

    coords = grid[indices]
    iter_vis = iter_values[indices]
    sr_vis = sr_values[indices]

    # Сортировка по итерациям
    order = np.argsort(iter_vis)
    coords = coords[order]
    iter_vis = iter_vis[order]
    sr_vis = sr_vis[order]

    # Динамическая верхняя граница для цвета
    max_iter_display = np.max(iter_vis)

    # Нормализация для цветовой шкалы
    if max_iter_display > 0:
        iter_norm = iter_vis / max_iter_display
    else:
        iter_norm = iter_vis
    colors = ITER_CMAP(iter_norm)

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=colors, s=35, edgecolors='none', depthshade=True)

    _format_3d_axes(ax, title)

    return max_iter_display


def _format_3d_axes(ax, title):
    """Форматирование 3D осей"""
    ax.set_xlim(0, 2.1)
    ax.set_ylim(0, 2.1)
    ax.set_zlim(0, 2.1)

    tick_vals = [0.2, 0.5, 1.0, 1.5, 2.0]
    tick_labels = [f'{v:.1f}' for v in tick_vals]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_zticks(tick_vals)
    ax.set_zticklabels(tick_labels, fontsize=7)

    ax.set_xlabel('$b_I$', fontsize=11, labelpad=12)
    ax.set_ylabel('$b_C$', fontsize=11, labelpad=12)
    ax.set_zlabel('$b_S$', fontsize=11, labelpad=12)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.view_init(elev=25, azim=45)


def add_sr_legend(fig):
    """Добавление непрерывной цветовой шкалы для SR"""
    import matplotlib.colorbar as cbar

    # Позиционирование (слева, до кубов)
    cax = fig.add_axes([0.01, 0.2, 0.02, 0.6])

    norm = plt.Normalize(0, 1)
    label_text = 'Success Rate'

    cb = cbar.ColorbarBase(cax, cmap=SR_CMAP, norm=norm,
                           orientation='vertical')

    # Подписи от 0 до 1
    tick_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    tick_labels = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
    cb.set_ticks(tick_values)
    cb.set_ticklabels(tick_labels)

    cb.set_label(label_text, fontsize=10)
    cb.ax.tick_params(labelsize=8)

def add_iter_legend(fig, max_iter_display):
    """Добавление цветовой шкалы для итераций (справа, после кубов)"""
    import matplotlib.colorbar as cbar

    # Позиционирование (справа)
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])

    if max_iter_display > 0:
        norm = plt.Normalize(0, max_iter_display)
        label_text = f'Итерации'
    else:
        norm = plt.Normalize(0, MAX_ITERATIONS)
        label_text = f'Итерации\n(нет данных)'

    cb = cbar.ColorbarBase(cax, cmap=ITER_CMAP, norm=norm,
                           orientation='vertical')
    cb.set_label(label_text, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def show_prediction_visualization(predictor, features_6, dim, formula_str=""):
    """
    Показать 3D-визуализацию предсказаний модели
    """
    GRID_VALUES = np.round(np.arange(0.1, 2.01, 0.1), 2)

    # Строим сетку параметров
    grid = np.array(list(product(GRID_VALUES, GRID_VALUES, GRID_VALUES)))
    n = len(grid)

    # Формируем признаки для сетки
    X = np.column_stack([
        np.full(n, features_6[0]), np.full(n, features_6[1]),
        np.full(n, features_6[2]), np.full(n, features_6[3]),
        np.full(n, features_6[4]), np.full(n, features_6[5]),
        np.full(n, int(dim == 3)), np.full(n, int(dim == 4)),
        np.full(n, int(dim == 5)),
        grid[:, 0], grid[:, 1], grid[:, 2],
    ])

    # Предсказания модели
    X_scaled = predictor.scaler_X.transform(X)
    p = predictor._predict_full(X_scaled)

    sr_values = p['sr_median']
    iter_values = p['iter_pred'] * MAX_ITERATIONS

    # Создаём фигуру с двумя графиками
    fig = plt.figure(figsize=(22, 10), facecolor='white')

    # График 1: SR (левый)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    render_scatter_sr(ax1, grid, sr_values, 'Success Rate')

    # График 2: Итерации (правый)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    max_iter = render_scatter_iter(ax2, grid, iter_values, sr_values,
                                   'Среднее число итераций')

    # Легенды (положения сохранены как в исходном коде)
    add_sr_legend(fig)
    add_iter_legend(fig, max_iter)

    # Заголовок
    if formula_str:
        func_display = formula_str[:60] + ('...' if len(formula_str) > 60 else '')
    else:
        func_display = f"Функция {dim}D"

    func_name_formatted = func_display.replace('_', ' ').title()

    plt.suptitle(f'Предсказания модели для функции: \n{func_name_formatted}\nРазмерность D={dim}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.show()