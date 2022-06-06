# using Revise

include(joinpath(pwd(),"Tensor_Package", "Tensor_Package.jl"))
include(joinpath(pwd(), "competitors", "scripts", "utils.jl"))

using   .Tensor_Package.Tensor_Package,
        Random,
        Base.Threads,
        SparseArrays,
        MLDataUtils,
        PyCall,
        DataStructures,
        CSV,
        StatsBase,
        InvertedIndices,
        DataFrames,
        LinearAlgebra

np = pyimport("numpy")
sp = pyimport("scipy.sparse")
pickle = pyimport("pickle")
main_jl = pyimport("appnp.main_jl")
# @pyimport scipy.sparse as sp
# @pyimport pickle
# @pyimport appnp.main_jl as main_jl

function APPNP_test(dataset_name, order, kn, perc_train, perc_val; balanced=true, type="network", features=false)

    args = Dict("model" => "exact",
                "epochs"=> 2000,
                "seed" => 42,
                "iterations" => 10,
                "early_stopping_rounds" => 500,
                "dropout" => 0.5,
                "alpha" => 0.1,
                "learning_rate" => 0.01,
                "lambd" => 0.005,
                "layers" => [64, 64])

    X, y, B, W, _, _, edges_ = Tensor_Package.prepare_incidence_data(dataset_name, order, kn; type=type, features=features)
    A = clique_adj(W, B)
    clique_edges = adj_to_edges_list(A)
    train_inds, test_inds, val_inds = Tensor_Package.train_test_val_split_indices(y, perc_train, perc_val; balanced=false)
    features = sparse_to_dic(X)
    test_inds = collect(setdiff(Set(1:length(y)), Set(train_inds)))
    y = np.array(y)
    pred = main_jl.main(args, clique_edges, features, y, train_inds .- 1, test_inds .- 1, val_inds .- 1)

    return Tensor_Package.accuracy(vec(pred[test_inds]), y[test_inds])
        end


function APPNP(args, X, y, clique_edges, train_inds, test_inds, val_inds)

        epochs, lr, wd = args


        args = Dict("model" => "exact",
                    "epochs"=> Int(epochs),
                    "seed" => 42,
                    "iterations" => 10,
                    "early_stopping_rounds" => 500,
                    "dropout" => 0.5,
                    "alpha" => 0.1,
                    "learning_rate" => lr,
                    "lambd" => wd,
                    "layers" => [64, 64])

        features = sparse_to_dic(X)
        y = np.array(y)
        pred = main_jl.main(args, clique_edges, features, y, train_inds, test_inds, val_inds)

        return Tensor_Package.calc_metrics(vec(pred[test_inds.+ 1]), y[test_inds .+ 1])

end

#pred = APPNP_test("cora-cit", 3, 15, .05, .05; type="network", features=true)
