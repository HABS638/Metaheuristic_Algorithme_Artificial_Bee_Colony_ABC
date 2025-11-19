import numpy as np
import math
from mealpy import FloatVar
from scipy import stats
from tabulate import tabulate


#!/usr/bin/env python
# Created by "Thieu" at 09:57, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
from mealpy.optimizer import Optimizer


class OriginalABC(Optimizer):
    """
    The original version of: Artificial Bee Colony (ABC)

    Links:
        1. https://www.sciencedirect.com/topics/computer-science/artificial-bee-colony

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_limits (int): Limit of trials before abandoning a food source, default=25

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ABC
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = ABC.OriginalABC(epoch=1000, pop_size=50, n_limits = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] B. Basturk, D. Karaboga, An artificial bee colony (ABC) algorithm for numeric function optimization,
    in: IEEE Swarm Intelligence Symposium 2006, May 12–14, Indianapolis, IN, USA, 2006.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_limits: int = 25, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size = onlooker bees = employed bees, default = 100
            n_limits: Limit of trials before abandoning a food source, default=25
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_limits = self.validator.check_int("n_limits", n_limits, [1, 1000])
        self.is_parallelizable = False
        self.set_parameters(["epoch", "pop_size", "n_limits"])
        self.sort_flag = False

def initialize_variables(self):
        self.trials = np.zeros(self.pop_size)


def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            # Choose a random employed bee to generate a new solution
            rdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            pos_new = self.pop[idx].solution + phi * (self.pop[rdx].solution - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1
        # Onlooker bees phase
        # Calculate the probabilities of each employed bee
        employed_fits = np.array([agent.target.fitness for agent in self.pop])
        # probabilities = employed_fits / np.sum(employed_fits)
        for idx in range(0, self.pop_size):
            # Select an employed bee using roulette wheel selection
            selected_bee = self.get_index_roulette_wheel_selection(employed_fits)
            # Choose a random employed bee to generate a new solution
            rdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx, selected_bee}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            pos_new = self.pop[selected_bee].solution + phi * (self.pop[rdx].solution - self.pop[selected_bee].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[selected_bee].target, self.problem.minmax):
                self.pop[selected_bee] = agent
                self.trials[selected_bee] = 0
            else:
                self.trials[selected_bee] += 1
        # Scout bees phase
        # Check the number of trials for each employed bee and abandon the food source if the limit is exceeded
        abandoned = np.where(self.trials >= self.n_limits)[0]
        for idx in abandoned:
            self.pop[idx] = self.generate_agent()
            self.trials[idx] = 0

# Définir la fonction Rosenbrock
def rosenbrock(solution):
    d = len(solution)
    sum_val = 0
    for i in range(d - 1):  # Parcours de 0 à d-2
        xi = solution[i]
        xnext = solution[i + 1]
        sum_val += 100 * (xnext - xi**2)**2 + (xi - 1)**2
    return sum_val

# Définir la fonction Rastrigin
def rastrigin(solution):
    d = len(solution)
    sum = 0.0
    for s in solution:
        sum += (s * s - 10 * math.cos(2 * math.pi * s))
    return 10 * d + sum

# Définir la fonction Ackley
def ackley(solution):
    a = 20
    b = 0.2
    c = 2 * np.pi

    d = len(solution)
    sum1 = np.sum(solution ** 2)
    sum2 = np.sum(np.cos(c * solution))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)


# Définition du problème de la fonction Rosenbrock
dimension = 10

problem_dict_Rosenbrock = {
    "bounds": FloatVar(lb=(-2.048,) * dimension, ub=(2.048,) * dimension, name="delta"),
    "minmax": "min",
    "obj_func": rosenbrock
}

# Définition du problème de la fonction Rastrigin
problem_dict_Rastringin = {
    "bounds": FloatVar(lb=(-5.12,) * dimension, ub=(5.12,) * dimension, name="delta"),
    "minmax": "min",
    "obj_func": rastrigin
}

# Définition du problème de la fonction Ackley
problem_dict_Ackley = {
    "bounds": FloatVar(lb=(-32.768,) * dimension, ub=(32.768,) * dimension, name="delta"),
    "minmax": "min",
    "obj_func": ackley
}


# Stockage des meilleures fitness pour chaque fonction objectif
best_fitnesses_Rosenbrock = []
best_fitnesses_Rastrigin = []
best_fitnesses_Ackley = []

# Nombre d'exécutions
run = 10

# Exécution de l'algorithme 10 fois
for i in range(run):
    print(f"Exécution {i + 1}...")

    # n_limits qui prend comme valeur : (10,50,100)
    model = OriginalABC(epoch=5000, pop_size=30, n_limits=30)

    g_best_Rosenbrock = model.solve(problem_dict_Rosenbrock)
    g_best_Rastringin = model.solve(problem_dict_Rastringin)
    g_best_Ackley = model.solve(problem_dict_Ackley)

    best_fitnesses_Rosenbrock.append(g_best_Rosenbrock.target.fitness)
    best_fitnesses_Rastrigin.append(g_best_Rastringin.target.fitness)
    best_fitnesses_Ackley.append(g_best_Ackley.target.fitness)

# Moyenne et écart-type avec numpy
results = {
    "Rosenbrock": (np.mean(best_fitnesses_Rosenbrock), np.std(best_fitnesses_Rosenbrock)),
    "Rastrigin": (np.mean(best_fitnesses_Rastrigin), np.std(best_fitnesses_Rastrigin)),
    "Ackley": (np.mean(best_fitnesses_Ackley), np.std(best_fitnesses_Ackley)),
}

# Affichage des résultats
for name, (mean, std) in results.items():
    print(f"\n{name}")
    print(f"Moyenne : {mean}")
    print(f"Écart type : {std}")

# TESTS STATISTIQUES WILCOXON


# Résultats Python
python_results = {
    30: {"Rosenbrock": 6970, "Rastrigin": 449, "Ackley": 20},
    50: {"Rosenbrock": 14489, "Rastrigin": 768, "Ackley": 20},
    100: {"Rosenbrock": 36454, "Rastrigin": 1650, "Ackley": 21}
}

# Résultats C++
cpp_results = {
    30: {"Rosenbrock": 1876.71, "Rastrigin": 331.586, "Ackley": 16.6436},
    50: {"Rosenbrock": 7830.03, "Rastrigin": 645.013, "Ackley": 20.086},
    100: {"Rosenbrock": 30769, "Rastrigin": 1509.63, "Ackley": 20.8629}
}

# Add Python standard deviations
python_std = {
    30: {"Rosenbrock": 433.73719648279695, "Rastrigin": 19.008853470382412, "Ackley": 0.14214250358465116},
    50: {"Rosenbrock": 1108.358114017656, "Rastrigin": 42.979930556562614, "Ackley": 0.04508707523690753},
    100: {"Rosenbrock": 2840.138044218718, "Rastrigin": 42.997933746142316, "Ackley": 0.0630472338527251}
}

# C++ standard deviations
cpp_standard_dev = {
    30: {"Rosenbrock": 397.63, "Rastrigin": 18.9776, "Ackley": 1.13235},
    50: {"Rosenbrock": 1173.89, "Rastrigin": 24.2568, "Ackley": 0.274255},
    100: {"Rosenbrock": 1823.47, "Rastrigin": 31.4353, "Ackley": 0.0674136}
}

# Mise à jour des en-têtes pour inclure les écart-types
headers = ["Dimension", "Function", "Moyenne Python", "Écart-type Python", "Moyenne C++", "Écart-type C++", "Diff (%)", "Better Implementation"]
table_data = []

# Création du tableau avec les écart-types
for dim in [30, 50, 100]:
    for func in ["Rosenbrock", "Rastrigin", "Ackley"]:
        py_val = python_results[dim][func]
        cpp_val = cpp_results[dim][func]
        py_std = python_std[dim][func]
        cpp_std = cpp_standard_dev[dim][func]
        
        # Calcul de la différence en pourcentage
        diff_percent = ((py_val - cpp_val) / max(py_val, cpp_val)) * 100
        
        # Déterminer la meilleure implémentation
        better = "C++" if cpp_val < py_val else "Python" if py_val < cpp_val else "Equal"
        
        table_data.append([
            dim,
            func,
            f"{py_val:.4f}",
            f"{py_std:.4f}",
            f"{cpp_val:.4f}",
            f"{cpp_std:.4f}",
            f"{diff_percent:.2f}%",
            better
        ])


# Tests de Wilcoxon
print("Résultats comparatifs Python vs C++:")
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# Préparation des données pour les tests de Wilcoxon
functions = ["Rosenbrock", "Rastrigin", "Ackley"]
dimensions = [30, 50, 100]

print("\nTests statistiques de Wilcoxon:")
print("-" * 50)

for func in functions:
    py_values = [python_results[dim][func] for dim in dimensions]
    cpp_values = [cpp_results[dim][func] for dim in dimensions]
    
    # Test de Wilcoxon
    statistic, p_value = stats.wilcoxon(py_values, cpp_values)
    
    print(f"\nFonction {func}:")
    print(f"Statistique de Wilcoxon: {statistic:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    # Interprétation du test
    alpha = 0.05
    if p_value < alpha:
        print("Conclusion: Il existe une différence statistiquement significative entre Python et C++")
    else:
        print("Conclusion: Pas de différence statistiquement significative entre Python et C++")






















