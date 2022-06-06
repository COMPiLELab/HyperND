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
sce_train = pyimport("sce.train_jl")



function SCE_test(dataset_name, order, kn, perc_train, perc_val;
                    balanced=true, type="network", features=false)

    args = Dict("no_cuda" => false,
                "seed"=> 123,
                "nhid" => 512,
                "output" => 512,
                "lr" => 0.01,
                "weight_decay" => 0,
                "epochs" => 30,
                "sample" => 5,
                "alpha" => 100000
                )



    X, y, B, W, _, _, edges_ = Tensor_Package.prepare_incidence_data(dataset_name, order, kn; type=type, features=features)
    A = clique_adj(W, B)
    graph = edges_to_graph(edges_)
    Y_train, Y_test, Y_val, train_mask, test_mask, val_mask = Tensor_Package.train_test_val_split(y, perc_train, perc_val, balanced=balanced)
    idx_train, idx_test, idx_val = findall(train_mask) .- 1, findall(test_mask) .- 1, findall(val_mask) .- 1
    features = np.array(X)
    labels = np.array(y)
    A = np.array(A)
    pred = sce_train.train(args, A, features,
     labels, idx_train, idx_val, idx_test)
    return Tensor_Package.accuracy(pred, y[idx_test .+ 1])
end


function SCE(args, A, X, y, train_inds, test_inds, val_inds)




        args = Dict("no_cuda" => false,
                    "seed"=> 123,
                    "nhid" => 512,
                    "output" => 512,
                    "lr" => 0.01,
                    "weight_decay" => 0,
                    "epochs" => 30,
                    "sample" => 5,
                    "alpha" => 100000
                    )



        features = np.array(X)
        labels = np.array(y)
        A = np.array(A)
        pred = sce_train.train(args, A, features,
         labels, train_inds, val_inds, test_inds)
        return Tensor_Package.calc_metrics(vec(pred), y[test_inds .+ 1])

end

#pred = SCE_test("cora-cit", 3, 15, .01, .02, type="network", features=true)
