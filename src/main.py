def calculate_statistics(data):
    best = np.min(data)
    worst = np.max(data)
    mean = np.mean(data)
    variance = np.var(data)
    return best, worst, mean, variance

def evaluate(individual, x_train, y_train, x_test, y_test):
    func = gp.compile(expr=individual, pset=pset)
    y_pred_train = [func(T, current) for T, current in x_train]
    y_pred_test = [func(T, current) for T, current in x_test]
    mae_train = np.mean(np.abs(np.array(y_pred_train) - y_train))
    mae_test = np.mean(np.abs(np.array(y_pred_test) - y_test))
    mse_train = np.mean((np.array(y_pred_train) - y_train) ** 2)
    mse_test = np.mean((np.array(y_pred_test) - y_test) ** 2)
    return mae_train, mse_train, mae_test, mse_test

def generate_data(num_samples=1000):
    current_range = range(20, 900)
    train_currents = random.sample(current_range, int(0.8 * len(current_range))) # 80% for training
    test_currents = random.sample(current_range, int(0.2 * len(current_range))) # 20% for testing
    X_train, Y_train, X_test, Y_test = [], [], [], []

    for T in temperature_values:
        for current in train_currents:
            # Introduce randomness by perturbing temperature and current
            perturbed_T = T + random.uniform(-1, 1) # Add random noise to temperature
            perturbed_current = current + random.uniform(-10, 10) # Add random noise to current

            # Use the perturbed values in your calculations
            Enernst = E0 + (((R * perturbed_T) / (2 * F)) * (2.303 * (math.log((PH2 * PO2) / (PH2O)))))
            i_limit = ilimit_den / A
            activation_loss = ((R * perturbed_T) / (alpha * n * F)) * (2.303 * (math.log((i_limit / i0_den))))
            concentration_loss = - (((R * perturbed_T) / (n * F)) * (2.303 * (math.log(1 - (i_limit / ilimit_den)))))
            VStack = Enernst - activation_loss - concentration_loss

            X_train.append((perturbed_T, perturbed_current))
            Y_train.append(VStack)

        for current in test_currents:
            # Introduce randomness by perturbing temperature and current
            perturbed_T = T + random.uniform(-1, 1) # Add random noise to temperature
            perturbed_current = current + random.uniform(-10, 10) # Add random noise to current

            # Use the perturbed values in your calculations
            Enernst = E0 + (((R * perturbed_T) / (2 * F)) * (2.303 * (math.log((PH2 * PO2) / (PH2O)))))
            i_limit = ilimit_den / A
            activation_loss = ((R * perturbed_T) / (alpha * n * F)) * (2.303 * (math.log((i_limit / i0_den))))
            concentration_loss = - (((R * perturbed_T) / (n * F)) * (2.303 * (math.log(1 - (i_limit / ilimit_den)))))
            VStack = Enernst - activation_loss - concentration_loss

            X_test.append((perturbed_T, perturbed_current))
            Y_test.append(VStack)

        if len(X_train) >= num_samples:
            break

    random.shuffle(X_train)
    random.shuffle(X_test)
    random.shuffle(Y_train)
    random.shuffle(Y_test)

    return X_train, Y_train, X_test, Y_test

pset = gp.PrimitiveSet("MAIN", arity=2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)
pset.addTerminal(1.0)
pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))

max_depth = 3

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)

random.seed(142)

runs = 50
pop_size = 100
crossover_prob = 0.8
mutation_prob = 0.01
num_generations = 50

mae_train_list, mae_test_list = [], []
mse_train_list, mse_test_list = [] , []

best_individuals, best_equations = [], []

for run in range(runs):
    X_train, Y_train, X_test, Y_test = generate_data()
    toolbox.register("evaluate", evaluate, x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test)

    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min_mae_train", np.min)
    stats.register("min_mse_train", np.min)
    stats.register("min_mae_test", np.min)
    stats.register("min_mse_test", np.min)

    algorithms.eaMuPlusLambda(population, toolbox, mu=pop_size, lambda_=2 * pop_size, cxpb=crossover_prob, mutpb=mutation_prob, ngen=num_generations, stats=stats, halloffame=hof, verbose=False)

    best_individual = hof[0]
    best_individuals.append(best_individual)
    mae_train, mse_train, mae_test, mse_test = best_individual.fitness.values

    mae_train_list.append(mae_train)
    mse_train_list.append(mse_train)
    mae_test_list.append(mae_test)
    mse_test_list.append(mse_test)

    best_individual_idx = np.argmin(mae_test_list)
    best_individual_test = best_individuals[best_individual_idx]

    best_equation = str(best_individual_test)
    best_equations.append(best_equation)

all_mae_train, all_mse_train = np.array(mae_train_list), np.array(mse_train_list)
all_mae_test, all_mse_test = np.array(mae_test_list), np.array(mse_test_list)

best_mae_train, worst_mae_train, mean_mae_train, variance_mae_train = calculate_statistics(all_mae_train)
best_mse_train, worst_mse_train, mean_mse_train, variance_mse_train = calculate_statistics(all_mse_train)

best_mae_test, worst_mae_test, mean_mae_test, variance_mae_test = calculate_statistics(all_mae_test)
best_mse_test, worst_mse_test, mean_mse_test, variance_mse_test = calculate_statistics(all_mse_test)

best_individual_test = best_individuals[best_individual_idx]
best_equation = str(best_individual_test)
compiled_function = gp.compile(expr=best_individual_test, pset=pset)

best_result = None
best_T_value = None
best_current_value = None

for T_value, current_value in zip(X_test, Y_test):
    result = compiled_function(T_value, current_value)
    if best_result is None or result < best_result:
        best_result = result
        best_T_value = T_value
        best_current_value = current_value

print("MAE on Training Data \n\n Best: {}\n Worst: {}\n Mean: {}\n Variance: {}\n".format(best_mae_train, worst_mae_train, mean_mae_train, variance_mae_train))
print("MSE on Training Data \n\n Best: {}\n Worst: {}\n Mean: {}\n Variance: {}\n".format(best_mse_train, worst_mse_train, mean_mse_train, variance_mse_train))

print("MAE on Test Data \n\n Best: {}\n Worst: {}\n Mean: {}\n Variance: {}".format(best_mae_test, worst_mae_test, mean_mae_test, variance_mae_test))
print("\nMSE on Test Data \n\n Best: {}\n Worst: {}\n Mean: {}\n Variance: {}".format(best_mse_test, worst_mse_test, mean_mse_test, variance_mse_test))

print("\n Best Value (with respect to test data): {}\n Best Equation (with respect to test data): {}\n".format(best_result, best_equations[np.argmin(mae_test_list)]))
