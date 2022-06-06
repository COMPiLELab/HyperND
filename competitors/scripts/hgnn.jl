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
        DataFrames

np = pyimport("numpy")
sp = pyimport("scipy.sparse")
pickle = pyimport("pickle")
train_jl = pyimport("hgnn.train_jl")


function HGNN_test(dataset_name, order, kn, perc_train; balanced=true, type="network", features=true)

    args = Dict("K_neigs"=> [10],
                "max_epoch" => 600,
                "n_hid" => 128,
                "lr" => 0.001,
                "milestones" => [100],
                "gamma" => 0.9,
                "drop_out" => 0.5,
                "print_freq" => 50,
                "weight_decay" => 0.0005,
                "decay_step" => 200,
                "decay_rate" => 0.7)

    X, y, _, _, _, _ = Tensor_Package.prepare_incidence_data(dataset_name, order, kn; type=type, features=features)
    X_features, X_labels, train_inds = Tensor_Package.prepare_features_and_labels(X,y,
    perc_train; percentage_of_known_features=1, feature_remove_style="rows")
    test_inds = collect(setdiff(Set(1:length(y)), Set(train_inds)))

    X_features = np.array(X_features)
    y = np.array(y)


    pred = train_jl.train(args, X_features, y, train_inds .- 1, test_inds .- 1)


    return Tensor_Package.accuracy(vec(pred)[test_inds], y[test_inds])
end


function HGNN(args, X_features, y, train_inds, test_inds)

    epochs, lr, wd = args

    args =     Dict("K_neigs"=> [10],
                    "max_epoch" => Int(epochs),
                    "n_hid" => 128,
                    "lr" => lr,
                    "milestones" => [100],
                    "gamma" => 0.9,
                    "drop_out" => 0.5,
                    "print_freq" => 50,
                    "weight_decay" => wd,
                    "decay_step" => 200,
                    "decay_rate" => 0.7)

    X_features = np.array(X_features)
    y = np.array(y)
    pred = train_jl.train(args, X_features, y, train_inds, test_inds)

    pred_unknown = pred[test_inds .+ 1]
    y_unknown = y[test_inds .+ 1]
    return Tensor_Package.calc_metrics(vec(pred_unknown), y_unknown)
end


#pred = HGNN_test("cora-cit", 3, 15, .05, type="network", features=true)
