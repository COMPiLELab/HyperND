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
        InvertedIndices

np = pyimport("numpy")
sp = pyimport("scipy.sparse")
pickle = pyimport("pickle")
hypergcn_jl = pyimport("hypergcn.hypergcn_jl")


function HyperGCN_test(dataset_name, order, kn, perc_train; balanced=true, type="network", features=true)

    args = Dict("dataset" => dataset_name,
                "mediators" => false,
                "fast" => false,
                "split" => 1,
                "gpu" => 3,
                "cuda" => true,
                "seed" => 5,
                "depth" => 2,
                "dropout" => 0.5,
                "epochs" => 200,
                "rate" => 0.01,
                "decay" => 0.0005)

    X, y, B, _, _, _, edges_ = Tensor_Package.prepare_incidence_data(dataset_name, order, kn; type=type, features=features)
    new_edges = inc_to_edges(B)
    hypergraph = edges_to_graph(new_edges)
    X_features, X_labels, train_inds = Tensor_Package.prepare_features_and_labels(X,y,
    perc_train; percentage_of_known_features=1, feature_remove_style="rows")
    test_inds = collect(setdiff(Set(1:length(y)), Set(train_inds)))
    classes = Set(y)
    n = length(y)
    Y = zeros(n, length(classes))
    for label in classes
        Y[y .== label, label] .= 1.0
    end
    X_features = sp.csr_matrix(X_features)

    test_acc, H, Z = hypergcn_jl.train(args, X_features, Y, hypergraph, train_inds .- 1, test_inds .- 1)

    final_pred = map(x->x[2], argmax(H,dims=2))

    #return test_acc, H, Z
    print(test_inds)
    return Tensor_Package.accuracy(vec(final_pred[test_inds]), y[test_inds]),
           Tensor_Package.recall(vec(final_pred[test_inds]), y[test_inds]),
           Tensor_Package.precision(vec(final_pred[test_inds]), y[test_inds])
end


function HyperGCN(args, X_features, y, hypergraph, train_inds, test_inds)

    epochs, lr, wd = args

    args = Dict("dataset" => "",
                "mediators" => false,
                "fast" => false,
                "split" => 1,
                "gpu" => 3,
                "cuda" => true,
                "seed" => 5,
                "depth" => 2,
                "dropout" => 0.5,
                "epochs" => Int(epochs),
                "rate" => lr,
                "decay" => wd)


    X_features = sp.csr_matrix(X_features)
    Y = labels_to_bin(y)
    test_acc, H, Z = hypergcn_jl.train(args, X_features, Y, hypergraph, train_inds, test_inds)
    final_pred = map(x->x[2], argmax(H,dims=2))
    return Tensor_Package.calc_metrics(vec(final_pred[test_inds.+ 1]), y[test_inds .+ 1])

end

#pred = HyperGCN_test("cora-cit", 3, 15, .06)
