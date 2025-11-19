#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <numeric>
#include "OriginalABC.h" // Inclure le fichier de l'algorithme ABC

// Fonctions d'optimisation
double rosenbrock(const std::vector<double>& solution) {
    double sum_val = 0.0;
    for (size_t i = 0; i < solution.size() - 1; ++i) {
        double xi = solution[i];
        double xnext = solution[i + 1];
        sum_val += 100.0 * std::pow(xnext - xi * xi, 2) + std::pow(xi - 1, 2);
    }
    return sum_val;
}

double rastrigin(const std::vector<double>& solution) {
    double sum = 0.0;
    for (double s : solution) {
        sum += s * s - 10.0 * std::cos(2.0 * M_PI * s);
    }
    return 10.0 * solution.size() + sum;
}

double ackley(const std::vector<double>& solution) {
    const double a = 20.0;
    const double b = 0.2;
    const double c = 2.0 * M_PI;

    double sum1 = 0.0, sum2 = 0.0;
    for (double x : solution) {
        sum1 += x * x;
        sum2 += std::cos(c * x);
    }
    size_t d = solution.size();
    return -a * std::exp(-b * std::sqrt(sum1 / d)) - std::exp(sum2 / d) + a + std::exp(1);
}

// Fonction pour calculer la moyenne d'un vecteur
double mean(const std::vector<double>& data) {
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

// Fonction pour calculer l'écart type d'un vecteur
double stddev(const std::vector<double>& data, double mean_val) {
    double sum = 0.0;
    for (double val : data) {
        sum += (val - mean_val) * (val - mean_val);
    }
    return std::sqrt(sum / data.size());
}

int main() {
    // Paramètres des fonctions
    const int dimension = 100;
    const int epochs = 5000;
    const int population_size = 30;
    const int n_limit = 25; // 10 ou 50 ou 100
    const int runs = 10;

    // Bornes des problèmes
    std::vector<std::pair<double, double>> bounds_rosenbrock( dimension, {-2.048, 2.048});
    std::vector<std::pair<double, double>> bounds_rastrigin(dimension, {-5.12, 5.12});
    std::vector<std::pair<double, double>> bounds_ackley(dimension, {-32.768, 32.768});

    // Vecteurs pour stocker les meilleurs fitness
    std::vector<double> best_fitness_rosenbrock;
    std::vector<double> best_fitness_rastrigin;
    std::vector<double> best_fitness_ackley;

    // Exécuter l'algorithme plusieurs fois
    for (int run = 0; run < runs; ++run) {

        // Initialiser l'algorithme
        OriginalABC model(epochs, population_size, n_limit);

        // Résoudre chaque problème
        model.initialize_variables(bounds_rosenbrock);
        model.solve(rosenbrock);
        best_fitness_rosenbrock.push_back(model.get_best_fitness());

        model.initialize_variables(bounds_rastrigin);
        model.solve(rastrigin);
        best_fitness_rastrigin.push_back(model.get_best_fitness());

        model.initialize_variables(bounds_ackley);
        model.solve(ackley);
        best_fitness_ackley.push_back(model.get_best_fitness());
    }

    // Calculer les moyennes et écarts types
    double mean_rosenbrock = mean(best_fitness_rosenbrock);
    double std_rosenbrock = stddev(best_fitness_rosenbrock, mean_rosenbrock);

    double mean_rastrigin = mean(best_fitness_rastrigin);
    double std_rastrigin = stddev(best_fitness_rastrigin, mean_rastrigin);

    double mean_ackley = mean(best_fitness_ackley);
    double std_ackley = stddev(best_fitness_ackley, mean_ackley);


    // Afficher les résultats à l'écran
std::cout << "Rosenbrock\n";
std::cout << "Moyenne : " << mean_rosenbrock << "\n";
std::cout << "Ecart type : " << std_rosenbrock << "\n\n";

std::cout << "Rastrigin\n";
std::cout << "Moyenne : " << mean_rastrigin << "\n";
std::cout << "Ecart type : " << std_rastrigin << "\n\n";

std::cout << "Ackley\n";
std::cout << "Moyenne : " << mean_ackley << "\n";
std::cout << "Ecart type : " << std_ackley << "\n\n";

std::cout << "Les moyennes et les ecarts types ont ete affiches a l'ecran." << std::endl;

    return 0;
}
