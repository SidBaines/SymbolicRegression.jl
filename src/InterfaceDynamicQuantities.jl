module InterfaceDynamicQuantitiesModule

import DynamicQuantities:
    AbstractDimensions,
    AbstractQuantity,
    Dimensions,
    SymbolicDimensions,
    Quantity,
    uparse,
    sym_uparse,
    DEFAULT_DIM_BASE_TYPE

"""
    get_units(T, D, x, f)

Gets unit information from a vector or scalar. The first two
types are the default numeric type and dimensions type, respectively.
The third argument is the value to get units from, and the fourth
argument is a function for parsing strings (in case a string is passed)
"""
function get_units(args...)
    return error(
        "Unit information must be passed as one of `AbstractDimensions`, `AbstractQuantity`, `AbstractString`, `Real`.",
    )
end
function get_units(_, _, ::Nothing, ::Function)
    return nothing
end
function get_units(::Type{T}, ::Type{D}, x::AbstractString, f::Function) where {T,D}
    isempty(x) && return one(Quantity{T,D})
    return convert(Quantity{T,D}, f(x))
end
function get_units(::Type{T}, ::Type{D}, x::Quantity, ::Function) where {T,D}
    return convert(Quantity{T,D}, x)
end
function get_units(::Type{T}, ::Type{D}, x::AbstractDimensions, ::Function) where {T,D}
    return convert(Quantity{T,D}, Quantity(one(T), x))
end
function get_units(::Type{T}, ::Type{D}, x::Real, ::Function) where {T,D}
    return Quantity(convert(T, x), D)::Quantity{T,D}
end
function get_units(::Type{T}, ::Type{D}, x::AbstractVector, f::Function) where {T,D}
    return Quantity{T,D}[get_units(T, D, xi, f) for xi in x]
end

"""
    get_si_units(::Type{T}, units)

Gets the units with Dimensions{DEFAULT_DIM_BASE_TYPE} type from a vector or scalar.
"""
function get_si_units(::Type{T}, units) where {T}
    return get_units(T, Dimensions{DEFAULT_DIM_BASE_TYPE}, units, uparse)
end

"""
    get_sym_units(::Type{T}, units)

Gets the units with SymbolicDimensions{DEFAULT_DIM_BASE_TYPE} type from a vector or scalar.
"""
function get_sym_units(::Type{T}, units) where {T}
    return get_units(T, SymbolicDimensions{DEFAULT_DIM_BASE_TYPE}, units, sym_uparse)
end

"""
    get_dimensions_type(A, default_dimensions)

Recursively finds the dimension type from an array, or,
if no quantity is found, returns the default type.
"""
function get_dimensions_type(
    A::Union{AbstractMatrix,AbstractVector{<:AbstractVector}}, ::Type{D}
) where {D}
    rows = eachrow(A)
    if length(rows) == 1
        return get_dimensions_type(rows[1], D)
    else
        return get_dimensions_type(rows[1], rows[2:end], D)
    end
end
function get_dimensions_type(::AbstractVector, tail, ::Type{D}) where {D}
    return get_dimensions_type(tail, D)
end
function get_dimensions_type(::AbstractVector, ::Type{D}) where {D}
    return D
end
function get_dimensions_type(
    ::AbstractVector{Q}, _, ::Type{D}
) where {Dout,Q<:AbstractQuantity{<:Any,Dout},D}
    return Dout
end
function get_dimensions_type(
    ::AbstractMatrix{Q}, ::Type{D}
) where {Dout,Q<:AbstractQuantity{<:Any,Dout},D}
    return Dout
end
function get_dimensions_type(
    ::AbstractVector{Q}, ::Type{D}
) where {Dout,Q<:AbstractQuantity{<:Any,Dout},D}
    return Dout
end

end
