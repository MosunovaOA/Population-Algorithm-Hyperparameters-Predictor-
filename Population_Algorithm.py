import numpy as np
import matplotlib.pyplot as plt
from math import prod

# ======================= КЛАСС ТЕСТОВЫХ ФУНКЦИЙ =======================
class TestFunctions:
    # разрешены sin, cos, exp, abs, sum, prod, +, -, *, /, **, x[i], len(x), range(), pi, e
    _registry = {
        "sphere": {
            "func": lambda x: sum([x[i] ** 2 for i in range(len(x))]),
            "bounds": (-100.0, 100.0),
            "global_min": 0.0
        },

        "rastrigin": {
            "func": lambda x: 10 * len(x) + sum([x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]) for i in range(len(x))]),
            "bounds": (-5.12, 5.12),
            "global_min": 0.0
        },

        "ackley": {
            "func": lambda x: (
                    -20 * np.exp(-0.2 * (sum([x[i] ** 2 for i in range(len(x))]) / len(x)) ** 0.5)
                    - np.exp(sum([np.cos(2 * np.pi * x[i]) for i in range(len(x))]) / len(x))
                    + 20 + np.e
            ),
            "bounds": (-32.768, 32.768),
            "global_min": 0.0
        },

        "eggcrate": {
            "func": lambda x: sum([x[i] ** 2 + 25 * np.sin(x[i]) ** 2 for i in range(len(x))]),
            "bounds": (-10.0, 10.0),
            "global_min": 0.0
        },

        "rosenbrock": {
            "func": lambda x: sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)]),
            "bounds": (-30.0, 30.0),
            "global_min": 0.0
        },

        "griewank": {
            "func": lambda x: (
                    sum([x[i] ** 2 for i in range(len(x))]) / 4000
                    - prod([np.cos(x[i] / (i + 1) ** 0.5) for i in range(len(x))])
                    + 1
            ),
            "bounds": (-600.0, 600.0),
            "global_min": 0.0
        },

        "damped_schwefel": {
            "func": lambda x: sum([abs(x[i]) - x[i] * np.sin(abs(x[i]) ** 0.5) for i in range(len(x))]),
            "bounds": (-500.0, 500.0),
            "global_min": 0.0
        },

        "levy": {
            "func": lambda x: (
                    np.sin(np.pi * (1 + (x[0] - 1) / 4)) ** 2
                    + sum([(((x[i] - 1) / 4) ** 2 * (1 + 10 * np.sin(np.pi * (1 + (x[i] - 1) / 4) + 1) ** 2)) for i in
                           range(len(x) - 1)])
                    + ((x[len(x) - 1] - 1) / 4) ** 2 * (1 + np.sin(2 * np.pi * (1 + (x[len(x) - 1] - 1) / 4)) ** 2)
            ),
            "bounds": (-10.0, 10.0),
            "global_min": 0.0
        },

        "michalewicz": {
            "func": lambda x: -sum([np.sin(x[i]) * np.sin((i + 1) * x[i] ** 2 / np.pi) ** 20 for i in range(len(x))]),
            "bounds": (0.0, np.pi),
            "global_min": {
                2: -1.801303,
                3: -2.760395,
                4: -3.698857,
                5: -4.687658,
                10: -8.665502
            }
        },

        "bohachevsky": {
            "func": lambda x: sum([
                x[i] ** 2 + 2 * x[i + 1] ** 2
                - 0.3 * np.cos(3 * np.pi * x[i])
                - 0.4 * np.cos(4 * np.pi * x[i + 1])
                + 0.7
                for i in range(len(x) - 1)
            ]),
            "bounds": (-100.0, 100.0),
            "global_min": 0.0
        },

        "xin_she_yang": {
            "func": lambda x: (
                    (sum([np.sin(x[i]) ** 2 for i in range(len(x))]) - np.exp(-sum([x[i] ** 2 for i in range(len(x))])))
                    * np.exp(-sum([np.sin(abs(x[i]) ** 0.5) ** 2 for i in range(len(x))]))
            ),
            "bounds": (-10.0, 10.0),
            "global_min": -1.0
        },

        "sum_of_powers": {
           "func": lambda x: sum([(x[i]**2)**(i+1) for i in range(len(x))]),
            "bounds": (-1.0, 1.0),
            "global_min": 0.0
        },

        "schaffer_n2": {
            "func": lambda x: sum([
                0.5 + (np.sin(x[i] ** 2 - x[i + 1] ** 2) ** 2 - 0.5)
                / (1 + 0.001 * (x[i] ** 2 + x[i + 1] ** 2)) ** 2
                for i in range(len(x) - 1)
            ]),
            "bounds": (-100.0, 100.0),
            "global_min": 0.0
        },

        "alpine_n1": {
            "func": lambda x: sum([abs(x[i] * np.sin(x[i]) + 0.1 * x[i]) for i in range(len(x))]),
            "bounds": (-10.0, 10.0),
            "global_min": 0.0
        },
    }

    @classmethod
    def get_function(cls, name: str):
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Функция '{name}' не найдена! Доступные: {list(cls._registry.keys())}")
        return cls._registry[name]["func"]

    @classmethod
    def get_bounds(cls, name: str):
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Функция '{name}' не найдена!")
        return cls._registry[name]["bounds"]

    @classmethod
    def list_functions(cls):
        return list(cls._registry.keys())

    @classmethod
    def list_global_mins(cls):
        # Возвращает список кортежей (имя_функции, global_min)
        return [(name, info["global_min"]) for name, info in cls._registry.items()]


# ========================= КЛАСС ЧАСТИЦЫ =========================
class Particle:
    def __init__(self, dim: int, bounds):
        self.dim = dim
        self.bounds = bounds
        low, high = bounds
        self.position = np.random.uniform(low, high, dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_fitness = np.inf


# ========================= ПОПУЛЯЦИОННЫЙ АЛГОРИТМ =========================
class PopulationAlgorithm:
    def __init__(self,
                 func_name: str = "rastrigin",
                 dim: int = 3,
                 population_size: int = 40,
                 max_iter: int = 500,
                 w: float = 0.8,
                 c_cog: float = 1.5,
                 c_soc: float = 1.5,
                 c_attr: float = 0.5,
                 c_hist: float = 0.5,
                 c_worst: float = 0.5,
                 #attractor: np.ndarray = None,
                 tolerance: float = 1e-6,
                 stagnation_limit: int = 60,
                 stop_by_stagnation: bool = True):

        self.func_name = func_name.lower()
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter

        # Параметры тестируемой функции из TestFunctions
        self.objective = TestFunctions.get_function(self.func_name)
        self.bounds = TestFunctions.get_bounds(self.func_name)

        self.center = np.mean(self.bounds) * np.ones(dim)
        #self.attractor = attractor if attractor is not None else self.center.copy()
        # Инициализируем аттрактор как None, будет вычисляться в процессе оптимизации
        self.attractor = None

        # Коэффициенты для 6 компонент
        self.w = w
        self.c_cog = c_cog
        self.c_soc = c_soc
        self.c_attr = c_attr
        self.c_hist = c_hist
        self.c_worst = c_worst

        # Параметры остановки
        self.tolerance = tolerance
        self.stagnation_limit = stagnation_limit
        self.stop_by_stagnation = stop_by_stagnation
        self.stagnation_counter = 0
        self.previous_gbest = np.inf

        # Рой
        low, high = self.bounds
        self.particles = [Particle(dim, self.bounds) for _ in range(self.population_size)]

        # Вычисляем fitness для всех частиц
        for p in self.particles:
            p.best_fitness = self.objective(p.position)

        # Глобальные значения
        best_idx = np.argmin([p.best_fitness for p in self.particles])
        self.gbest_position = self.particles[best_idx].best_position.copy()
        self.gbest_fitness = self.particles[best_idx].best_fitness
        self.hgbest_position = self.gbest_position.copy()

        # Худшая позиция
        worst_idx = np.argmax([p.best_fitness for p in self.particles])
        self.gworst_position = self.particles[worst_idx].best_position.copy()

        self.history = []

    def _calculate_swarm_center(self):
        # Вычисление центра тяжести роя: среднее арифметическое всех позиций частиц
        positions = np.array([p.position for p in self.particles])
        return np.mean(positions, axis=0)

    def optimize(self):
        for it in range(self.max_iter):
            # Обновляем аттрактор как центр тяжести роя
            self.attractor = self._calculate_swarm_center()

            # Обновляем позиции частиц
            for p in self.particles:
                r = np.random.random((5, self.dim))

                inertia = self.w * p.velocity
                cognitive = self.c_cog * r[0] * (p.best_position - p.position)
                social = self.c_soc * r[1] * (self.gbest_position - p.position)
                attractor_term = self.c_attr * r[2] * (self.attractor - p.position)
                historical = self.c_hist * r[3] * (self.hgbest_position - p.position)
                worst_term = -self.c_worst * r[4] * (self.gworst_position - p.position)

                p.velocity = inertia + cognitive + social + attractor_term + historical + worst_term
                # Ограничение скорости
                #p.velocity = np.clip(p.velocity, -0.2 * (self.bounds[1] - self.bounds[0]),
                #                     0.2 * (self.bounds[1] - self.bounds[0]))

                p.position += p.velocity
                # Ограничение позиций
                #p.position = np.clip(p.position, self.bounds[0], self.bounds[1])

                fit = self.objective(p.position)
                if fit < p.best_fitness:
                    p.best_position = p.position.copy()
                    p.best_fitness = fit

            # Обновляем глобальные значения после обработки всех частиц
            for p in self.particles:
                if p.best_fitness < self.gbest_fitness:
                    self.gbest_position = p.best_position.copy()
                    self.gbest_fitness = p.best_fitness
                    self.hgbest_position = p.best_position.copy()

            # Обновляем худшую позицию
            worst_idx = np.argmax([p.best_fitness for p in self.particles])
            self.gworst_position = self.particles[worst_idx].best_position.copy()

            self.history.append(self.gbest_fitness)

            if self.stop_by_stagnation:
                if self.previous_gbest > self.gbest_fitness:
                    improvement = self.previous_gbest - self.gbest_fitness
                else:
                    improvement = self.gbest_fitness - self.previous_gbest
                if improvement < self.tolerance:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0
                self.previous_gbest = self.gbest_fitness

                if self.stagnation_counter >= self.stagnation_limit:
                    #print(f"\nСТОП: стагнация {self.stagnation_counter} итераций")
                    self.stagnation_counter = 0
                    break

            # if it % 50 == 0 or it == self.max_iter-1:
            #     print(f"Итерация {it:4d} → f = {self.gbest_fitness:.10e}")

        return self.gbest_position, self.gbest_fitness

    def plot_surface_3d(self, resolution=150):
        # 3D-визуализация только при dim == 2
        if self.dim != 2:
            print(f"3D-график рисуется только при dim=2 (текущая: {self.dim})")
            return

        x = np.linspace(self.bounds[0], self.bounds[1], resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self.objective(np.array([X[i,j], Y[i,j]]))
                       for j in range(resolution)] for i in range(resolution)])

        fig = plt.figure(figsize=(16,12))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, linewidth=0, antialiased=True)

        pos = np.array([p.position for p in self.particles])
        z_vals = np.array([self.objective(p.position) for p in self.particles])

        ax.scatter(pos[:,0], pos[:,1], z_vals, c='red', s=100, edgecolors='white',
                   linewidth=2, depthshade=False, zorder=10)
        ax.scatter(self.gbest_position[0], self.gbest_position[1], self.gbest_fitness,
                   c='gold', s=700, marker='*', edgecolors='black', linewidth=3, zorder=100,
                   label=f'Best: {self.gbest_fitness:.2e}')

        ax.set_title(f'Population Algorithm — {self.func_name.capitalize()} (D=2)', fontsize=16)
        ax.set_xlabel('X1'); ax.set_ylabel('X2'); ax.set_zlabel('f(X)')
        ax.legend()
        plt.colorbar(surf, shrink=0.6)
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
        plt.show()
