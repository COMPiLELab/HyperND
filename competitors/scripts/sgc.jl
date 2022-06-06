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
        Suppressor,
        UCIData,
        DataFrames,
        LinearAlgebra

np = pyimport("numpy")
sp = pyimport("scipy.sparse")
pickle = pyimport("pickle")
citation_jl = pyimport("sgc.citation_jl")

export SGC

function SGC_test(dataset_name, order, kn, perc_train, perc_val;
                    balanced=true, type="network", features=true)

    args = Dict("no_cuda" => false,
                "seed"=> 2000,
                "epochs" => 100,
                "lr" => 0.2,
                "weight_decay" => 5e-6,
                "hidden" => 0,
                "dropout" => 0,
                "model" => "SGC", #can be also GCN (but they
                                  #have no code to obtain test results when this
                                  #parameter is set to GCN, so I assume it's not
                                  #working well)
                "feature" => "mul", #can be cat and adj
                "normalization" => "AugNormAdj",
                "degree" => 2,
                "per" => -1,
                "experiment" => "base-experiment",
                "tuned" => false
                )



    X, y, B, W, _, _, edges_ = Tensor_Package.prepare_incidence_data(dataset_name, order, kn; type=type, features=features)
    A = clique_adj(W, B)
    graph = edges_to_graph(edges_)
    Y_train, Y_test, Y_val, train_mask, test_mask, val_mask = Tensor_Package.train_test_val_split(y, perc_train, perc_val, balanced=balanced)
    idx_train, idx_test, idx_val = findall(train_mask) .- 1, findall(test_mask) .- 1, findall(val_mask) .- 1
    features = np.array(X)
    Y = labels_to_bin(y)
    labels = np.array(Y)
    A = np.array(A)
    pred = citation_jl.main(args, A, features,
     labels, idx_train, idx_val, idx_test)
    final_pred = map(x->x[2], argmax(pred,dims=2))
    return Tensor_Package.accuracy(vec(final_pred), y[idx_test .+ 1])
end


function SGC(args, A, X, y, train_inds, test_inds, val_inds)
        epochs, lr, wd = args

        args = Dict("no_cuda" => false,
                    "seed"=> 2000,
                    "epochs" => Int(epochs),
                    "lr" => lr,
                    "weight_decay" => wd,
                    "hidden" => 0,
                    "dropout" => 0,
                    "model" => "SGC", #can be also GCN (but they
                                      #have no code to obtain test results when this
                                      #parameter is set to GCN, so I assume it's not
                                      #working well)
                    "feature" => "mul", #can be cat and adj
                    "normalization" => "AugNormAdj",
                    "degree" => 2,
                    "per" => -1,
                    "experiment" => "base-experiment",
                    "tuned" => false
                    )


        features = np.array(X)
        labels = np.array(labels_to_bin(y))
        A = np.array(A)
        pred = citation_jl.main(args, A, features,
         labels, train_inds, val_inds, test_inds)
        final_pred = map(x->x[2], argmax(pred,dims=2))
        return Tensor_Package.calc_metrics(vec(final_pred), y[test_inds .+ 1])

end



#pred = SGC_test("cora-cit", 3, 15, .01, .02, type="network", features=true)
