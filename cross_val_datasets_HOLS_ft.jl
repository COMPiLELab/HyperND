include("Tensor_Package/Tensor_Package.jl")
include("competitors/scripts/utils.jl")

using .Tensor_Package.Tensor_Package
using Base.Threads
using BSON
using CSV
using DataFrames
using DelimitedFiles
using LinearAlgebra
using Random
using SparseArrays
using Statistics
using Dates


include("CV_helpers_HOLS_ft.jl")
# include("wrap_functions.jl")

function read_hols_datasets(dataset)
    IJ = readdlm("./data/fb100/edges-$dataset.txt")
    Y = readdlm("./data/fb100/labels-$dataset.txt")
    n = length(Y)
    A = sparse(IJ[:,1], IJ[:,2], 1, n, n)
    A = max.(A, A')
    A = min.(A, 1)
    return A, convert(Vector{Int64}, vec(Y))
end

function main(dataset, percentage_of_known_labels; kn=10, order=3, type="points" )
    Random.seed!(1234)

    # higher-order mixing functions
    ps = [-1,0.1,1,2,10]
    methods = []
    for p in ps
        push!(methods, (p, string("incidence_", p), :HOLS_ft))
    end

    num_rand_trials = 5

    ls_grid = 0.1:0.1:0.9
    balanced = true
    max_iterations = 25
    num_CV_trials = 5
    tolerance = 5*1e-2


    results = DataFrame()


    X, true_labels, INC, W, D_n, D_e, edges_ = Tensor_Package.prepare_incidence_data(dataset, order, kn, type=type)
    new_edges = inc_to_edges(INC)
    hypergraph = edges_to_graph(new_edges)
    # Save training / validation splits for external codes
    all_train_inds = Dict{Float64, Vector{Vector{Int64}}}()
    for p in percentage_of_known_labels
         all_train_inds[p] = Vector{Vector{Int64}}(undef, num_rand_trials)
    end

    if !isdir(joinpath(pwd(), "competitors", "results"))
                mkdir(joinpath(pwd(), "competitors", "results"))
    end

    if !isdir(joinpath(pwd(), "competitors", "results", "hypergcn_comparison"))
                mkdir(joinpath(pwd(), "competitors", "results", "hypergcn_comparison"))
    end

    for trial in 1:num_rand_trials
        for (j, percentage) in enumerate(percentage_of_known_labels)
            df, training_inds = CV_binary(X, INC, hypergraph, W, D_n, D_e, true_labels,
                                          methods,
                                          num_CV_trials,
                                          percentage,
                                          balanced=balanced,
                                          ε=1e-6,
                                          ls_search_params=ls_grid,
                                          max_iterations=max_iterations,
                                          tolerance=tolerance,
                                          verbose=true)


            all_train_inds[percentage][trial] = vcat(training_inds...)

            df[!, :trial] .= trial

            results = [results;df]
        end
        # Write out results
        CSV.write(joinpath(pwd(), "competitors", "results", "hypergcn_comparison", "final_results_cv_$dataset.csv"), results)
        println(results)
        println("Results written in CSV file for Dataset = $dataset, Trial = $trial")
    end

    # Write out results
    dt = today()
    CSV.write(joinpath(pwd(), "competitors", "results", "hypergcn_comparison", "final_results_cv_$dataset.csv"), results)

    # Write out reproducibility info
    bson(joinpath(pwd(), "competitors", "results", "hypergcn_comparison", "train_val_inds_$dataset.bson"),
         Dict("A"              => INC,
             "y"              => vec(true_labels),
             "features"       => "binary",
             "all_train_ind" => all_train_inds
            ))
end

datasets = ["cora-cit", "dblp", "pubmed"]
datasets2 = ["citeseer", "cora-author"]


datasets_all = ["citeseer", "cora-cit", "cora-author", "dblp", "pubmed"]
percentage_of_known_labels_array = [0.041666666666666664,0.051698670605613,0.051698670605613,0.042128710474069055,0.0039559770756200235]


for (dataset, percentage_of_known_labels) in zip(datasets_all[4:5], percentage_of_known_labels_array[4:5])
    main(dataset,[percentage_of_known_labels]; type="network")
end
