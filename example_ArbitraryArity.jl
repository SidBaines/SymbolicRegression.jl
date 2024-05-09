using SymbolicRegression
using DynamicExpressions.UtilsModule: FuncArityPair

function myfunc(x,y,z,a)
    return x * cos(y) * sin(z*a)
end
myfuncpair = FuncArityPair(myfunc, 4)

X = randn(Float32, 5, 100)
y = 2 .* cos.(X[4, :]) .* sin.(X[1, :]) + 0.3 .* cos.(X[2, :]) .* sin.(X[3, :] .* 2.1)

options = SymbolicRegression.Options(;
    anyary_operators=[myfuncpair], binary_operators=[+, *], unary_operators=[cos, exp], populations=20,#, should_simplify=true, crossover_probability=0.5,
)

hall_of_fame = equation_search(
    X, y; niterations=40, options=options, parallelism=:multithreading
)

dominating = calculate_pareto_frontier(hall_of_fame)

trees = [member.tree for member in dominating]

tree = trees[end]
output, did_succeed = eval_tree_array(tree, X, options)

println("Complexity\tMSE\tEquation")

for member in dominating
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)

    println("$(complexity)\t$(loss)\t$(string)")
end
