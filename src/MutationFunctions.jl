module MutationFunctionsModule

using Random: default_rng, AbstractRNG, shuffle
using DynamicExpressions:
    AbstractExpressionNode,
    AbstractNode,
    NodeSampler,
    constructorof,
    copy_node,
    set_node!,
    count_nodes,
    has_constants,
    has_operators
using Compat: Returns, @inline
using ..CoreModule: Options, DATA_TYPE

"""
    random_node(tree::AbstractNode; filter::F=Returns(true))

Return a random node from the tree. You may optionally
filter the nodes matching some condition before sampling.
"""
function random_node(
    tree::AbstractNode, rng::AbstractRNG=default_rng(); filter::F=Returns(true)
) where {F<:Function}
    Base.depwarn(
        "Instead of `random_node(tree, filter)`, use `rand(NodeSampler(; tree, filter))`",
        :random_node,
    )
    return rand(rng, NodeSampler(; tree, filter))
end

"""Swap operands in binary operator for ops like pow and divide"""
function swap_operands(tree::AbstractNode, rng::AbstractRNG=default_rng())
    if !any(node -> node.degree > 1, tree)
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 1))
    perm = shuffle(rng,1:node.degree)
    node.children = [node.children[i] for i in perm]
    return tree
end

"""Randomly convert an operator into another one (binary->binary; unary->unary)"""
function mutate_operator(
    tree::AbstractExpressionNode{T}, options::Options, rng::AbstractRNG=default_rng()
) where {T}
    if !(has_operators(tree))
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree != 0))
    if node.degree == 1
        node.op = rand(rng, 1:(options.nuna))
    elseif node.degree == 2
        node.op = rand(rng, 1:(options.nbin))
    else
        node.op = rand(rng, 1:(options.nany))
    end
    return tree
end

"""Randomly perturb a constant"""
function mutate_constant(
    tree::AbstractExpressionNode{T},
    temperature,
    options::Options,
    rng::AbstractRNG=default_rng(),
) where {T<:DATA_TYPE}
    # T is between 0 and 1.

    if !(has_constants(tree))
        return tree
    end
    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && t.constant)))

    bottom = 1//10
    maxChange = options.perturbation_factor * temperature + 1 + bottom
    factor = T(maxChange^rand(rng, T))
    makeConstBigger = rand(rng, Bool)

    if makeConstBigger
        node.val *= factor
    else
        node.val /= factor
    end

    if rand(rng) > options.probability_negate_constant
        node.val *= -1
    end

    return tree
end

"""Add a random unary/binary operation to the end of a tree"""
function append_random_op(
    tree::AbstractExpressionNode{T},
    options::Options,
    nfeatures::Int,
    rng::AbstractRNG=default_rng();
    maxOperatorDegree::Union{UInt,Nothing}=nothing,
) where {T<:DATA_TYPE}
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree == 0))

    if maxOperatorDegree === nothing
        ch = rand(1:sum([options.nuna, options.nbin, options.nany]))
        typeOfNewOp = findfirst(cumsum([options.nuna, options.nbin, options.nany]).>=ch)
    end

    if typeOfNewOp==1
        newnode = constructorof(typeof(tree))(
            rand(rng, 1:(options.nuna)), make_random_leaf(nfeatures, T, typeof(tree), rng)
        )
    elseif typeOfNewOp==2
        newnode = constructorof(typeof(tree))(
            rand(rng, 1:(options.nbin)),
            make_random_leaf(nfeatures, T, typeof(tree), rng),
            make_random_leaf(nfeatures, T, typeof(tree), rng),
        )
    else
        choice = rand(rng,1:(options.nany))
        arity=options.operators.anynops[choice].arity
        newnode = constructorof(typeof(tree))(
            choice,
            [make_random_leaf(nfeatures, T, typeof(tree), rng) for i = 1:arity],
        )
    end

    set_node!(node, newnode)

    return tree
end

"""Insert random node"""
function insert_random_op(
    tree::AbstractExpressionNode{T},
    options::Options,
    nfeatures::Int,
    rng::AbstractRNG=default_rng(),
) where {T<:DATA_TYPE}
    node = rand(rng, NodeSampler(; tree))
    ch = rand(1:sum([options.nuna, options.nbin, options.nany]))
    typeOfNewOp = findfirst(cumsum([options.nuna, options.nbin, options.nany]).>=ch)
    left = copy_node(node)

    if typeOfNewOp == 1
        newnode = constructorof(typeof(tree))(rand(rng, 1:(options.nuna)), left)
    elseif typeOfNewOp == 2
        right = make_random_leaf(nfeatures, T, typeof(tree), rng)
        newnode = constructorof(typeof(tree))(rand(rng, 1:(options.nbin)), left, right)
    else
        choice = rand(rng, 1:(options.nany))
        arity = options.operators.anyops[choice].arity
        pos_of_old_node = rand(rng, 1:arity)
        children = [(pos_of_old_node == n) ? left : make_random_leaf(nfeatures, T, typeof(tree), rng) for n in 1:arity]
        newnode = constructorof(typeof(tree))(choice, children)
    end
    set_node!(node, newnode)
    return tree
end

"""Add random node to the top of a tree"""
function prepend_random_op(
    tree::AbstractExpressionNode{T},
    options::Options,
    nfeatures::Int,
    rng::AbstractRNG=default_rng(),
) where {T<:DATA_TYPE}
    node = tree
    ch = rand(1:sum([options.nuna, options.nbin, options.nany]))
    typeOfNewOp = findfirst(cumsum([options.nuna, options.nbin, options.nany]).>=ch)
    left = copy_node(tree)

    if typeOfNewOp == 1
        newnode = constructorof(typeof(tree))(rand(rng, 1:(options.nuna)), left)
    elseif typeOfNewOp == 2
        right = make_random_leaf(nfeatures, T, typeof(tree), rng)
        newnode = constructorof(typeof(tree))(rand(rng, 1:(options.nbin)), left, right)
    else
        choice = rand(rng, 1:(options.nany))
        arity = options.operators.anyops[choice].arity
        children = [(n == 1) ? left : make_random_leaf(nfeatures, T, typeof(tree), rng) for n in 1:arity]
        newnode = constructorof(typeof(tree))(choice, children)
    end
    set_node!(node, newnode)
    return node
end

function make_random_leaf(
    nfeatures::Int, ::Type{T}, ::Type{N}, rng::AbstractRNG=default_rng()
) where {T<:DATA_TYPE,N<:AbstractExpressionNode}
    if rand(rng, Bool)
        return constructorof(N)(; val=randn(rng, T))
    else
        return constructorof(N)(T; feature=rand(rng, 1:nfeatures))
    end
end

"""Return a random node from the tree with parent, and the index of the node in the parent's list of children (-1 for no parent)"""
function random_node_and_parent(tree::AbstractNode, rng::AbstractRNG=default_rng())
    if tree.degree == 0
        return tree, tree, -1
    end
    parent = rand(rng, NodeSampler(; tree, filter=t -> t.degree != 0))
    idx = rand(rng, 1:length(parent.children))
    return (parent.children[idx], parent, idx)
end

"""Select a random node, and splice it out of the tree."""
function delete_random_op!(
    tree::AbstractExpressionNode{T},
    options::Options,
    nfeatures::Int,
    rng::AbstractRNG=default_rng(),
) where {T<:DATA_TYPE}
    node, parent, side = random_node_and_parent(tree, rng)
    isroot = side == -1

    if node.degree == 0
        # Replace with new constant
        newnode = make_random_leaf(nfeatures, T, typeof(tree), rng)
        set_node!(node, newnode)
    elseif node.degree == 1
        # Join one of the children with the parent
        if isroot
            return node.children[1]
        else
            # TODO this seems like a mess, can we make it better (may involve refactoring whole function)? I think this originally had to look like this because it's possible a parent has multiple identical children (eg. a tree that looks like '(1 + (x^x)', we might want to merge either x)
            prt_ch_change = rand(rng, findall(t->t==node, parent.children)) # The index of the parents children (necessary for the case where parent has multiple identical children) that we'll merge
            parent.children = [i!=prt_ch_change ? parent.children[i] : parent.children[i].children[1] for i in 1:length(parent.children)]
        end
    else
        # Join one of the children with the parent
        node_ch_change = rand(rng, 1:length(node.children)) # The index of the childs children that we'll merge
        if isroot
            return node.children[node_ch_change]
        else
            prt_ch_change = rand(rng, findall(t->t==node, parent.children)) # The index of the parents children (necessary for the case where parent has multiple identical children) that we'll merge
            parent.children = [i!=prt_ch_change ? parent.children[i] : parent.children[i].children[node_ch_change] for i in 1:length(parent.children)]
        end
    end
    return tree
end

"""Create a random equation by appending random operators"""
function gen_random_tree(
    length::Int, options::Options, nfeatures::Int, ::Type{T}, rng::AbstractRNG=default_rng()
) where {T<:DATA_TYPE}
    # Note that this base tree is just a placeholder; it will be replaced.
    tree = constructorof(options.node_type)(T; val=convert(T, 1))
    for i in 1:length
        # TODO: This can be larger number of nodes than length.
        tree = append_random_op(tree, options, nfeatures, rng)
    end
    return tree
end

function gen_random_tree_fixed_size(
    node_count::Int,
    options::Options,
    nfeatures::Int,
    ::Type{T},
    rng::AbstractRNG=default_rng(),
) where {T<:DATA_TYPE}
    tree = make_random_leaf(nfeatures, T, options.node_type, rng)
    cur_size = count_nodes(tree)
    while cur_size < node_count
        maxOperatorDegree=cur_size-node_count
        canFindOps = false # Check if there are any operators with max arity or less, if this doesn't become true will go over the requested amount, so we must break.
        if maxOperatorDegree <= 0
        end
        if maxOperatorDegree >= 1
            canFindOps = options.nuna != 0
        end
        if maxOperatorDegree >= 2
            canFindOps = canFindOps || (options.nbin != 0)
        end
        if maxOperatorDegree >= 3
            for op in options.operators.anyops
                if op.arity <= maxOperatorDegree
                    canFindOps = true
                end
            end
        end
        (!canFindOps) && break
        tree = append_random_op(tree, options, nfeatures, rng; maxOperatorDegree=maxOperatorDegree)
        cur_size = count_nodes(tree)
    end
    return tree
end

"""Crossover between two expressions"""
function crossover_trees(
    tree1::AbstractExpressionNode{T},
    tree2::AbstractExpressionNode{T},
    rng::AbstractRNG=default_rng(),
) where {T}
    tree1 = copy_node(tree1)
    tree2 = copy_node(tree2)

    node1, parent1, side1 = random_node_and_parent(tree1, rng)
    node2, parent2, side2 = random_node_and_parent(tree2, rng)

    node1 = copy_node(node1)


    if side1 >= 1
        parent1.children = [(side1 != n) ? parent1.children[n] : copy_node(node2) for n in 1:length(parent1.children)]
        # tree1 now contains this.
    else # 'n'
        # This means that there is no parent2.
        tree1 = copy_node(node2)
    end

    if side2 >= 1
        parent2.children = [(side2 != n) ? parent2.children[n] : copy_node(node1) for n in 1:length(parent2.children)]
    else # 'n'
        tree2 = copy_node(node1)
    end
    return tree1, tree2
end

function form_random_connection!(tree::AbstractNode, rng::AbstractRNG=default_rng())
    if length(tree) < 5
        return tree
    end

    local parent, new_child, would_form_loop

    attempt_number = 0
    max_attempts = 10

    while true
        parent = rand(rng, NodeSampler(; tree, filter=t -> t.degree != 0))
        new_child = rand(rng, NodeSampler(; tree, filter=t -> t !== tree))
        attempt_number += 1
        would_form_loop = any(t -> t === parent, new_child)
        if would_form_loop && attempt_number <= max_attempts
            continue
        else
            break
        end
    end
    if would_form_loop
        return tree
    end
    # Set one of the children to be this new child:
    if parent.degree <= 2
        side = (parent.degree == 1 || rand(rng, Bool)) ? :l : :r
        setproperty!(parent, side, new_child)
    else
        # TODO Check if this is correctly translated into the arbitrary arity case?
        side = rand(rng, 1:parent.degree)
        parent.children[side] = new_child
    end
    return tree
end
function break_random_connection!(tree::AbstractNode, rng::AbstractRNG=default_rng())
    tree.degree == 0 && return tree
    parent = rand(rng, NodeSampler(; tree, filter=t -> t.degree != 0))
    if parent.degree <= 2
        side = (parent.degree == 1 || rand(rng, Bool)) ? :l : :r
        unshared_child = copy(getproperty(parent, side))
        setproperty!(parent, side, unshared_child)
    else
        # TODO Check if this is correctly translated into the arbitrary arity case?
        side = rand(rng, 1:parent.degree)
        unshared_child = copy(parent.children[side])
        parent.children[side] = unshared_child
    end
    return tree
end

end
