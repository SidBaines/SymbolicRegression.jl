# The Val{i} optimizes it into a branching statement (https://discourse.julialang.org/t/meta-programming-an-if-else-statement-of-user-defined-length/53525)
function BINOP!(x::Array{Float32, 1}, y::Array{Float32, 1}, ::Val{i}, ::Val{clen}, options::Options) where {i,clen}
    op = options.binops[i]
    # broadcast!(op, x, x, y)
    @inbounds @simd for j=1:clen
        x[j] = op(x[j], y[j])
    end
end

function UNAOP!(x::Array{Float32, 1}, ::Val{i}, ::Val{clen}, options::Options) where {i,clen}
    op = options.unaops[i]
    @inbounds @simd for j=1:clen
        x[j] = op(x[j])
    end
end

# Evaluate an equation over an array of datapoints
function evalTreeArray(tree::Node, cX::Array{Float32, 2}, options::Options)::Union{Array{Float32, 1}, Nothing}
    clen = size(cX)[1]
    if tree.degree == 0
        if tree.constant
            return fill(tree.val, clen)
        else
            return copy(cX[:, tree.val])
        end
    elseif tree.degree == 1
        cumulator = evalTreeArray(tree.l, cX, options)
        if cumulator === nothing
            return nothing
        end
        op_idx = tree.op
        UNAOP!(cumulator, Val(op_idx), Val(clen), options)
        @inbounds for i=1:clen
            if isinf(cumulator[i]) || isnan(cumulator[i])
                return nothing
            end
        end
        return cumulator
    else
        cumulator = evalTreeArray(tree.l, cX, options)
        if cumulator === nothing
            return nothing
        end
        array2 = evalTreeArray(tree.r, cX, options)
        if array2 === nothing
            return nothing
        end
        op_idx = tree.op
        BINOP!(cumulator, array2, Val(op_idx), Val(clen), options)
        @inbounds for i=1:clen
            if isinf(cumulator[i]) || isnan(cumulator[i])
                return nothing
            end
        end
        return cumulator
    end
end
