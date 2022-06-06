module Tensor_Package

using Base.Threads
using Distances
using LightGraphs
using LinearAlgebra
using PyCall
using Random
using ScikitLearn
using SparseArrays
using Statistics
#using UCIData
#using DataFrames
using CSV



include("similarity_knn.jl")
include("tensors.jl")
include("labelspreading.jl")
include("utils.jl")
include("crossval.jl")


end
