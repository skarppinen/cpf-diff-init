include("../../../config.jl");

using DataStructures: OrderedDict
using DataFrames: DataFrame
using StaticArrays

"""
A type representing floating point scalar parameters of a statistical model.
The type makes a distinction between "estimated" and "fixed" parameters,
that can be specified in run time.
"""
struct Parameter{ParamType <: AFloat}
    p::OrderedDict{Symbol, ParamType}
    est::Vector{Symbol}
    fix::Vector{Symbol}
    function Parameter(p::OrderedDict{Symbol, T}, est::AVec{Symbol}) where {T <: AFloat}
        @assert length(est) > 0 "length of `est` (estimated) must be > 0."
        @assert length(p) >= length(est) "length of `est` (estimated) must be less than or equal to 'length(p)'"
        param_names = collect(keys(p));
        @assert (map(x -> x in param_names, est) |> all) "all parameter names in 'est' (estimated) must be in 'p'"
        fix = setdiff(param_names, est); # Fixed parameters are those that weren't given.
        new{T}(p, est, fix);
    end
end

function Parameter(p::NamedTuple{X, NTuple{N, T}},
                   estimated::AVec{Symbol}) where {X, N, T <: AFloat}
    Parameter(OrderedDict(zip(keys(p), values(p))), estimated);
end

function Parameter(p::NamedTuple; estimated::AVec{Symbol})
    Parameter(p, estimated);
end

function Parameter(p::NamedTuple)
    Parameter(p, collect(keys(p)));
end

function Vector(θ::Parameter{T}, syms::Vector{Symbol}) where {T <: AFloat}
    r = Vector{T}(undef, length(syms));
    copy!(r, θ, syms);
    r;
end

function Vector(θ::Parameter, ::Val{:estimated})
    Vector(θ, θ.est);
end

function Vector(θ::Parameter, ::Val{:fixed})
    Vector(θ, θ.fix);
end

"""
Construct SVector from a Parameter. The N will simply have to match
with the number of estimated parameters.
"""
function SVector{N, T}(θ::Parameter{T}) where {N, T <: AFloat}
    SVector{N, T}(values(estimated(θ)));
end

function estimated(θ::Parameter)
    NamedTuple{Tuple(θ.est)}(Vector(θ, Val(:estimated)));
end

function fixed(θ::Parameter)
    NamedTuple{Tuple(θ.fix)}(Vector(θ, Val(:fixed)));
end

import Base.getindex, Base.length
function getindex(θ::Parameter, s::Symbol)
    θ.p[s];
end

function length(θ::Parameter)
    length(θ.p);
end

import Base.setindex!
function setindex!(θ::Parameter{<: AFloat}, val::Float64, s::Symbol)
    θ.p[s] = val;
    nothing;
end

import Base.copy!
function copy!(θ::Parameter, x::AVec{<: AFloat}, syms)
    for (i, s) in enumerate(syms)
        θ.p[s] = x[i];
    end
    nothing
end

## Copy to Parameter from Vector.
function copy!(θ::Parameter, x::AVec{<: AFloat}, ::Val{:estimated})
    copy!(θ, x, θ.est);
end

function copy!(θ::Parameter, x::AVec{<: AFloat})
    copy!(θ, x, Val(:estimated));
end

set_param!(θ::Parameter, x::AVec{<: AFloat}) = copy!(θ, x);

function copy!(x::AVec{<: AFloat}, θ::Parameter, syms::AVec{Symbol})
    for (i, s) in enumerate(syms)
        x[i] = θ[s];
    end
    nothing
end

## Copy to vector from Parameter.
function copy!(x::AVec{<: AFloat}, θ::Parameter, ::Val{:fixed})
    copy!(x, θ, θ.fix);
end
function copy!(x::AVec{<: AFloat}, θ::Parameter, ::Val{:estimated})
    copy!(x, θ, θ.est);
end
function copy!(x::AVec{<: AFloat}, θ::Parameter)
    copy!(x, θ, Val(:estimated));
end

## Copy to Parameter from Parameter.
function copy!(dest::Parameter, src::Parameter, syms)
    for p in syms
        dest[p] = src[p];
    end
    nothing;
end

## Make a DataFrame out of a Parameter.
function DataFrame(θ::Parameter)
    DataFrame(; estimated(θ)..., fixed(θ)...);
end

## Miscellanneous.
import Base.names
function names(θ::Parameter)
    vcat(θ.est, θ.fix);
end

import Base.show
function Base.show(io::IO, θ::Parameter)
    strtyp = string(typeof(θ));
    println(io, string(strtyp, " with values:"));
    for s in θ.est
        println(io, string(s, " => ", θ[s], " (e)"));
    end
    for s in θ.fix
        println(io, string(s, " => ", θ[s], " (f)"));
    end
    nothing;
end
