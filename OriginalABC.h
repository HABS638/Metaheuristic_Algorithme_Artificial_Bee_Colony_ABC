#ifndef ORIGINALABC_H
#define ORIGINALABC_H

#include <vector>
#include <functional>
#include <iostream>
#include <random>
#include <utility>

class Agent {
public:
    Agent(int dim);
    std::vector<double> solution;
    double fitness;
};

class OriginalABC {
public:
    OriginalABC(int epoch, int pop_size, int n_limits);

    void initialize_variables(const std::vector<std::pair<double, double>>& bounds);
    void solve(std::function<double(const std::vector<double>&)> obj_func);
    double get_best_fitness() const;

private:
    int epoch;
    int pop_size;
    int n_limits;
    int problem_dim;
    std::vector<Agent> population;
    std::vector<int> trials;
    std::vector<std::pair<double, double>> bounds;

    std::vector<double> generate_new_solution(const std::vector<double>& solution, const std::vector<double>& neighbor_solution);
    std::vector<double> correct_solution(const std::vector<double>& solution);
    double uniform(double min, double max);
    bool compare_target(double target1, double target2, bool minimize);
};

#endif // ORIGINALABC_H
