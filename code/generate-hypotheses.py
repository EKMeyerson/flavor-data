#
# Generate experiment suggestions using parsimonious
# symbolic regresion surrogates.
#

import sys
import json
import operator
import math
import random
import itertools

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp



class SuggestionModel:
    """
    Class for managing the construction and optimization
    of symbolic regression surrogates.
    """

    def __init__(self):

        # Input specs
        self.training_data = None
        self.constraints = None

        # Model-fitting toolbox
        self.toolbox = None

        # Flag for identifying invalid functions
        self.invalid = False

        return


    def load_training_data(self, training_file):
        """
        Load training data from file without dependencies.
        """

        training_data = {}
        with open(training_file, 'r') as infile:
            lines = infile.readlines()
            vars = lines[0].strip().split(',')
            training_data['output var'] = vars[0]
            training_data['input vars'] = vars[1:]
            training_data['samples'] = []
            for line in lines[1:]:
                training_data['samples'].append([float(f) for f in line.split(',')])
        self.training_data = training_data

        return training_data


    def load_constraints(self, constraints_file):
        """
        Load constraints json.
        """

        with open(constraints_file) as cfile:
            constraints = json.load(cfile)
        self.constraints = constraints

        return constraints


    def generate_hypotheses(self, training_data):
        """
        Generate Pareto front of parsimonious symbolic regression models.
        """

        NGEN = 1000
        MU = 2000
        LAMBDA = 2000
        XOVER_PROB = 0.5
        MUT_PROB = 0.5

        # Specify algorithm.
        self.toolbox = self.build_toolbox(training_data)

        # Initialize population.
        pop = self.toolbox.population(n=MU)
        hof = tools.ParetoFront()

        # Set stats to track during evolution.
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # Run evolutionary model-fitting.
        algorithms.eaMuPlusLambda(pop, self.toolbox, MU, LAMBDA, XOVER_PROB,
                                  MUT_PROB, NGEN, stats, halloffame=hof)

        return hof


    def build_toolbox(self, training_data):
        """
        Set up symbolic regression tools.
        """

        # Set primitives.
        pset = gp.PrimitiveSet("MAIN", len(training_data['input vars']))
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(self.protectedDiv, 2)
        pset.addPrimitive(operator.neg, 1)
        pset.addPrimitive(math.cos, 1)
        pset.addEphemeralConstant("rand01", lambda: random.random())

        # Rename variables to those in training data.
        for i in range(len(pset.arguments)):
            old_name = pset.arguments[i]
            new_name = training_data['input vars'][i]
            pset.arguments[i] = new_name
            pset.mapping[new_name] = pset.mapping[old_name]
            pset.mapping[new_name].value = new_name
            del pset.mapping[old_name]

        # Set objectives.
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))

        # Set representation.
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        # Set EA operators.
        toolbox.register("evaluate", self.evalSymbReg)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        return toolbox


    def protectedDiv(self, left, right):
        """
        Wrapper to avoid numerical errors in division.
        """
        eps = 1e-08
        if abs(right) < eps:
            self.invalid = True
            return 1 / eps
        else:
            return left / right


    def evalSymbReg(self, individual, view=False):
        """
        Compute error and size of symbolic regression model.
        """

        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)

        # Pull out samples.
        samples = self.training_data['samples']

        # Option to visualize function along first two dimensions.
        if view:
            Y = [d[0] for d in samples]
            X = [d[1] for d in samples]
            plt.scatter(X,Y)
            Xf = [i/float(10) for i in range(0,200)]
            Yf = [func(x) for x in Xf]
            plt.plot(Xf,Yf)
            plt.ylim(0, 500)
            plt.show()

        # Evaluate the mean squared error between the expression
        # and the real data
        sqerror = 0.0
        for d in samples:
            sqerror += (func(*d[1:]) - d[0])**2

        return sqerror / len(samples), float(len(individual))


    def filter_hypotheses(self, hypotheses, num_suggestions):
        """
        Return the top <num_suggestions> non-trivial hypotheses.
        """

        # Ignore hypotheses that are not a function of the input
        non_trivial_hypotheses = []
        for h in hypotheses:
            used_vars = set()
            for i,v in enumerate(self.training_data['input vars']):
                if v in str(h):
                    used_vars.add(i)
            if len(used_vars) > 0:
                non_trivial_hypotheses.append(h)

        # Select top <num_suggestions> hypotheses
        top_hypotheses = non_trivial_hypotheses[::-1][:num_suggestions][::-1]

        return top_hypotheses


    def generate_suggestions(self, training_data, constraints, top_hypotheses):
        """
        Generate suggestions from <top_hypotheses>.
        """

        # Set search ranges for each variable.
        SPACE_SIZE = 100
        searchspaces = []
        for i,v in enumerate(training_data['input vars']):
            if constraints['resources'][v] is None:
                hardmin, hardmax = constraints['ranges'][v]
                delta = constraints['max delta'][v]
                samplevals = [s[i + 1] for s in training_data['samples']]
                minval = max(hardmin, min(samplevals) - delta)
                maxval = min(hardmax, max(samplevals) + delta)
                searchspaces.append(list(np.linspace(minval, maxval, SPACE_SIZE)))
            else:
                searchspaces.append(constraints['resources'][v])
                random.shuffle(searchspaces[-1])
        output_min, output_max = constraints['ranges'][constraints['output var']]

        # Generate top suggestion for each hypothesis.
        suggestions = []
        for h in top_hypotheses:
            func = self.toolbox.compile(expr=h)

            # Identify optimal score and configuration.
            best_predicted_score = -float('inf')
            best_suggestion = None
            for input_setting in itertools.product(*searchspaces):
                self.invalid = False
                score = func(*input_setting)
                if (score > best_predicted_score) and not self.invalid \
                        and (score >= output_min) and (score <= output_max):
                    best_predicted_score = score
                    best_suggestion = list(input_setting[:])

            # Detect unused variables.
            unused_vars = set()
            for i,v in enumerate(training_data['input vars']):
                if v not in str(h):
                    unused_vars.add(i)

            # Set unused variables to be optimally diverse.
            for i in unused_vars:
                samplevals = [s[i + 1] for s in training_data['samples']] \
                           + [s[i + 1] for s in suggestions]
                best_setting = -1
                max_sparsity = -1
                for j in searchspaces[i]:
                    sparsity = min(abs(j - k) for k in samplevals)
                    if sparsity > max_sparsity:
                        best_setting = j
                        max_sparsity = sparsity
                best_suggestion[i] = best_setting

            # Add suggestion to list of suggestions.
            suggestions.append([best_predicted_score] + best_suggestion[:])

            # Remove any limited resource the suggestion uses.
            for i,v in enumerate(training_data['input vars']):
                if constraints['resources'][v] != []:
                    searchspaces[i].remove(best_suggestion[i])

        return suggestions


    def write_suggestions(self, top_hypotheses, suggestions, suggestions_out_file):
        """
        Write suggestions and corresponding hypotheses to file.
        """

        with open(suggestions_out_file, 'w') as outfile:
            outfile.write('hypothesis;error;prediction;' + \
                          ';'.join(self.training_data['input vars']) + '\n')
            for h, s in zip(top_hypotheses, suggestions):
                error = self.evalSymbReg(h)
                outfile.write(str(h) + ';' + str(error) + ';' + \
                              ';'.join(["%.2f" % f for f in s]) + '\n')
        print("Output written to", suggestions_out_file)

        return



if __name__=='__main__':

    """
    Usage: python generate-hypotheses.py <training_file> \
                                         <constraints_file> \
                                         <num_suggestions> \
                                         <suggestions_output_file>
    """

    # Path to file containing data for training surrogates.
    # training_file should be a csv, where the first column
    # is the target variable, and remaining columns are input
    # variables. See sample-training-data.dat for an example.
    training_file = sys.argv[1]

    # Path to file specifying any constraints that must be
    # satisfied by the suggestions. Constraints can include
    # fixed variable ranges, fixed resource sets, and max
    # value deltas from previous values for each variable.
    # See sample-constraints.json for an example.
    constraints_file = sys.argv[2]

    # Number of suggestions to generate. The number of
    # suggestions cannot be greater than the number of
    # resources available for any input variable.
    num_suggestions = int(sys.argv[3])

    # Unique path to output file that will be created, and
    # to which generated suggestions will be written.
    suggestions_out_file = sys.argv[4]

    # Run suggestion generation.
    sm = SuggestionModel()
    training_data = sm.load_training_data(training_file)
    constraints = sm.load_constraints(constraints_file)
    hypotheses = sm.generate_hypotheses(training_data)
    top_hypotheses = sm.filter_hypotheses(hypotheses, num_suggestions)
    suggestions = sm.generate_suggestions(training_data, constraints, top_hypotheses)
    sm.write_suggestions(top_hypotheses, suggestions, suggestions_out_file)

    print('Done. Thank you.')
