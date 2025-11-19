#include "OriginalABC.h"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>

Agent::Agent(int dim) : solution(dim, 0.0), fitness(std::numeric_limits<double>::max()) {}

OriginalABC::OriginalABC(int epoch, int pop_size, int n_limits)
    : epoch(epoch), pop_size(pop_size), n_limits(n_limits), problem_dim(0) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
}

void OriginalABC::initialize_variables(const std::vector<std::pair<double, double>>& bounds) {
    this->bounds = bounds;
    this->problem_dim = bounds.size();
    population = std::vector<Agent>(pop_size, Agent(problem_dim));
    trials = std::vector<int>(pop_size, 0);

    for (Agent& agent : population) {
        for (int i = 0; i < problem_dim; ++i) {
            agent.solution[i] = uniform(bounds[i].first, bounds[i].second);
        }
        agent.fitness = std::numeric_limits<double>::max(); // Initial fitness
    }
}

void OriginalABC::solve(std::function<double(const std::vector<double>&)> obj_func) {
    for (Agent& agent : population) {
        agent.fitness = obj_func(agent.solution);
    }

    for (int epoch_iter = 0; epoch_iter < epoch; ++epoch_iter) {
        for (int i = 0; i < pop_size; ++i) {
            int neighbor_idx = std::rand() % pop_size;
            while (neighbor_idx == i) {
                neighbor_idx = std::rand() % pop_size;
            }

            std::vector<double> new_solution = generate_new_solution(population[i].solution, population[neighbor_idx].solution);
            new_solution = correct_solution(new_solution);

            double new_fitness = obj_func(new_solution);
            if (compare_target(new_fitness, population[i].fitness, true)) {
                population[i].solution = new_solution;
                population[i].fitness = new_fitness;
                trials[i] = 0;
            } else {
                ++trials[i];
            }
        }

        for (int i = 0; i < pop_size; ++i) {
            if (trials[i] >= n_limits) {
                for (int j = 0; j < problem_dim; ++j) {
                    population[i].solution[j] = uniform(bounds[j].first, bounds[j].second);
                }
                population[i].fitness = obj_func(population[i].solution);
                trials[i] = 0;
            }
        }
    }
}

std::vector<double> OriginalABC::generate_new_solution(const std::vector<double>& solution, const std::vector<double>& neighbor_solution) {
    std::vector<double> new_solution(solution.size());
    for (size_t i = 0; i < solution.size(); ++i) {
        double phi = uniform(-1.0, 1.0);
        new_solution[i] = solution[i] + phi * (solution[i] - neighbor_solution[i]);
    }
    return new_solution;
}

double clamp(double value, double min, double max) {
    return std::max(min, std::min(value, max));
}

std::vector<double> OriginalABC::correct_solution(const std::vector<double>& solution) {
    std::vector<double> corrected_solution = solution;
    for (size_t i = 0; i < corrected_solution.size(); ++i) {
        corrected_solution[i] = clamp(corrected_solution[i], bounds[i].first, bounds[i].second);
    }
    return corrected_solution;
}

double OriginalABC::uniform(double min, double max) {
    return min + (max - min) * (static_cast<double>(std::rand()) / RAND_MAX);
}

bool OriginalABC::compare_target(double target1, double target2, bool minimize) {
    return minimize ? target1 < target2 : target1 > target2;
}

double OriginalABC::get_best_fitness() const {
    double best_fitness = std::numeric_limits<double>::max();
    for (const Agent& agent : population) {
        if (agent.fitness < best_fitness) {
            best_fitness = agent.fitness;
        }
    }
    return best_fitness;
}
