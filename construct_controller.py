import itertools
import numpy as np
import sympy as sym


def construct_poles_polynomial(alpha, kbeta):
    n = len(alpha)
    coeff = np.empty(n+1)
    coeff[0] = 1
    for i in range(n-1):
        coeff[i+1] = kbeta[i] + alpha[i+1] - alpha[i]
    coeff[n] = kbeta[n-1] - alpha[n-1]
    return coeff


def get_zeros(coeff):
    return np.roots(coeff)


def are_zeros_inside_unit_circle(zeros):
    zeros_inside = True
    num_zeros_outside = 0
    for zero in zeros:
        if abs(zero) >= 1:
            zeros_inside = False
            num_zeros_outside += 1
    return zeros_inside, num_zeros_outside


def get_parameter_equations(expression, variable, parameters, order, max_combinations):
    equations = []
    combinations = []
    for i in range(2, max_combinations+1):
        temp_combinations = list(itertools.combinations(parameters, i))
        combinations += temp_combinations
    for i in range(order):
        equation = 0
        temp_expression = sym.poly(sym.expand(expression / variable**i))
        equation += temp_expression.coeff_monomial(1)
        for parameter in parameters:
            equation += temp_expression.coeff_monomial(parameter) * parameter
        for combination in combinations:
            param_combo = 1
            for param in combination:
                param_combo *= param
            equation += temp_expression.coeff_monomial(param_combo) * param_combo
        equations.append(equation)
    return equations


def get_variables(equation, x):
    temp_variables = equation.free_symbols
    if x in temp_variables:
        temp_variables.remove(x)
    return temp_variables


def print_parameters(solution, parameters):
    if not solution:
        print("No solution")
        return
    if isinstance(solution, list):
        solution = solution[0]
    for parameter in parameters:
        print(str(parameter) + ": " + str(solution.get(parameter)))


def is_stable(solution, parameters, order):
    if not solution:
        return
    if isinstance(solution, list):
        solution = solution[0]
    
    if not all([type(solution.get(parameter)) == sym.Float for parameter in parameters]):
        return
    
    alpha_list = [1]
    beta_list = []
    for i, parameter in enumerate(parameters):
        if i < order - 1:
            alpha_list.append(solution.get(parameter))
        else:
            beta_list.append(solution.get(parameter))
    is_inside = are_zeros_inside_unit_circle(get_zeros(construct_poles_polynomial(alpha_list, beta_list)))[0]
    location = ["outside", "inside"]
    print("The parameters are: " + location[is_inside])


def construct_controller(order_controller, order_adaptivity_extra, order_stepsize_filter, order_error_filter, pole_placements=[]):
    if order_stepsize_filter > 0 and order_error_filter > 0:
        raise Exception("Cannot apply both stepsize and error filter!")
    
    x = sym.Symbol("x")
    alpha = [sym.Symbol("alpha%d"%(i+1)) for i in range(order_controller)]
    kbeta = [sym.Symbol("kbeta%d"%(i+1)) for i in range(order_controller)]
    
    equations = []
    variables = set([])
    
    P = 0
    Q = 0
    for i in range(order_controller):
        P += kbeta[i] * x**(order_controller-1-i)
        if i == 0:
            Q += x**(order_controller-1-i)
        else:
            Q += alpha[i] * x**(order_controller-1-i)
    
    Qzeros = [sym.Symbol("Qzero%d"%(i+1)) for i in range(order_controller-1)]
    temp_Q = (x-1)**order_adaptivity_extra * (x+1)**order_error_filter
    extra_df = order_controller-order_adaptivity_extra-order_error_filter-1
    for i in range(extra_df):
        temp_Q *= (x - Qzeros[i])
    Q_new = sym.poly(sym.expand(Q - temp_Q))
    temp_variables = get_variables(Q_new, x)
    equations += get_parameter_equations(Q_new, x, temp_variables, order_controller, extra_df)
    variables.update(temp_variables)
    
    c = sym.Symbol("c")
    Pzeros = [sym.Symbol("Pzero%d"%(i+1)) for i in range(order_controller-1)]
    temp_P = (x+1)**order_stepsize_filter
    extra_df = order_controller-order_stepsize_filter-1
    for i in range(extra_df):
        temp_P *= (x - Pzeros[i])
    P_new = sym.poly(sym.expand(P - c*temp_P))
    temp_variables = get_variables(P_new, x)
    equations += get_parameter_equations(P_new, x, temp_variables, order_controller, extra_df+1)
    variables.update(temp_variables)
    
    fixed_poles = len(pole_placements)
    poles = [sym.Symbol("pole%d"%(i+1)) for i in range(order_controller-fixed_poles)]
    temp_characteristic_equation = 1
    for pole in pole_placements:
        temp_characteristic_equation *= (x - pole)
    for pole in poles:
        temp_characteristic_equation *= (x - pole)
    characteristic_equation = sym.poly(sym.expand((x-1)*Q + P) - temp_characteristic_equation)
    temp_variables = get_variables(characteristic_equation, x)
    equations += get_parameter_equations(characteristic_equation, x, temp_variables, order_controller+1, order_controller-fixed_poles)
    variables.update(temp_variables)
    
    solution = sym.solve(equations, variables)
    
    return solution, kbeta + alpha[1:]


def main():
    order_controller = 3
    order_adaptivity_extra = 1
    order_stepsize_filter = 1
    order_error_filter = 0
    pole_placements = [0, 0, 0]

    solution, parameters = construct_controller(order_controller, order_adaptivity_extra, order_stepsize_filter, order_error_filter, pole_placements=pole_placements)

    print_parameters(solution, parameters)
    
    is_stable(solution, parameters, order_controller)
    

if __name__ == "__main__":
    main()