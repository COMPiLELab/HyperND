using PyCall
# I had to do this to make pytorch work with the new numpy library, it's optional
#os = pyimport("os")
#os.add_dll_directory(raw"E:\pytorch_venv\Library\bin")


include(joinpath(pwd(),"Tensor_Package", "Tensor_Package.jl"))
include(joinpath(pwd(), "competitors", "scripts", "utils.jl"))


using   .Tensor_Package.Tensor_Package,
        Random,
        Base.Threads,
        SparseArrays,
        MLDataUtils,
        DataStructures,
        CSV,
        StatsBase,
        InvertedIndices,
        DataFrames,
        LinearAlgebra

np = pyimport("numpy")
sp = pyimport("scipy.sparse")
pickle = pyimport("pickle")
pd  = pyimport("pandas")

#first 5 are the competitors. all of them require pytorch and torch_sparse, the installation guide can be found online, it is very straightforward
#I recommend a clean virtual environment if you don't already have pytorch set up
include("appnp.jl")
include("hgnn.jl")
include("sgc.jl")
include("sce.jl")
include("hypergcn.jl")
include("hols.jl")


num_splits = 5
# order is the size of hyperedge (we have 2, 3, 4, that is only for data that doesn't have hypergraph ready)
function main(dataset_name, order, kn, train_percentages; max_trial_number=5, seed=123,
                                type="network", balanced=true, features=true)
    results = []
    # creates incidence matrix from dataset considering only the largest
    # connected components
    X, y, B, W, D_n, D_e, _ = Tensor_Package.prepare_incidence_data(dataset_name, order,
                                                kn; type=type, features=features)
    # this is creating clique expanded graph adjacency matrix from the hypergraph

    A = clique_adj(W, B)
    # converting the adj matrix to an edge list
    clique_edges = adj_to_edges_list(A)
    # converting the incidence matrix to an edge list
    new_edges = inc_to_edges(B)
    # converting to format of dictionary for hypergcn
    hypergraph = edges_to_graph(new_edges)
    for trial_number in 1:max_trial_number
            for train_perc in train_percentages
                print(train_perc)
                train_splits, train_inds, test_inds, val_inds, y_onehot = Tensor_Package.train_test_val_split_indices(y, train_perc,
                                                        train_perc; balanced=balanced)
                train_data = vcat([train_inds, val_inds]...)
                train_val_splits = [vcat([t, v]...) for (t, v) in zip(train_inds, val_inds)]
                #python indexing starts from 0, julia - from 1
                train_inds, test_inds, val_inds = train_inds .- 1, test_inds .- 1, val_inds .- 1


                #case when features=false
                if X == nothing
                        # this is made for APPNP, because it doesn't work with zero rows (y_onehot has 1 only on rows of the input train indices)
                        fake_feature = ones(length(y))
                        X = [y_onehot fake_feature]
                end

                p = [1, 2, 3, 5, 10]
                α = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

                splits = CV_splits(train_val_splits, num_splits, split=0.5)
                p_best, α_best = CV_HOLS(X, B, W, D_n, D_e, y, p, α, splits)
                metrics = HOLS(X, B, W, D_n, D_e, y, p_best, α_best, train_data .+ 1, test_inds .+ 1)

                #the first 3 values are the default values for epochs, learning rate and weight decay in these algorithms
                appnp_metrics = APPNP((2000, 0.01, 0.005), X, y, clique_edges, train_inds, test_inds, val_inds)
                print("APPNP finished\n")
                hypergcn_metrics = HyperGCN((200, 0.01, 0.0005), X, y, hypergraph, vcat([train_inds, val_inds]...), test_inds)
                print("HyperGCN finished\n")
                sce_metrics = SCE((100, 0.2, 5e-6), A, X, y, train_inds, test_inds, val_inds)
                print("SCE finished\n")
                hgnn_metrics = HGNN((600, 0.001, 0.0005), X, y, vcat([train_inds, val_inds]...), test_inds)
                print("HGNN finished\n")
                sgc_metrics = SGC((30, 0.01, 0.0), A, X, y, train_inds, test_inds, val_inds)
                print("HGNN finished\n")
                push!(results, vcat(dataset_name, kn, train_perc, trial_number, appnp_metrics, hgnn_metrics, hypergcn_metrics, sgc_metrics, sce_metrics, metrics, p_best, α_best))
                fixed_results = present_results(results)
                fixed_results.to_csv(joinpath(pwd(), "competitors", "results", string(dataset_name,".csv")))

             end
    end
    return results
end


function present_results(results)
        metrics = []
        for method in ["appnp", "hgnn", "hypergcn", "sgc", "sce", "hols_labels_features"]
            for metric in ["acc", "rec", "prec"]
                push!(metrics, string(metric, "_", method))
            end
        end
        columns = vcat(["dataset_name", "neighbors", "train_perc", "trial_number"], metrics, ["p", "α"])
        return pd.DataFrame(results, columns=columns)
end

Random.seed!(15)


percentages = [0.042,0.052,0.052,0.04,0.008]
datasets  = ["citeseer", "cora-cit", "cora-author", "dblp", "pubmed"]
for (dataset,train_percentage) in zip(datasets, percentages)
        if !isdir(joinpath(pwd(), "competitors", "results"))
                mkdir(joinpath(pwd(), "competitors", "results"))
        end
        # 3 is the order of the hypergraph in case you want to build it (for point-cloud datasets)
        # 7 is the maximum number of neighbors, that is used while building a distance matrix for the data (for point-cloud datasets)
        # use type="points" for point-cloud dataset
        results = main(dataset, 3, 7, [train_percentage], max_trial_number=5, features=true, type="network")
        results = present_results(results)
        results.to_csv(joinpath(pwd(), "competitors", "results", string(dataset,".csv")))
end

# the competitors couldn't handle mag-dual and trivago-clicks, so we are not showing results for them
datasets  = ["foodweb", "contact-high-school", "mag-dual", "trivago-clicks"]
for dataset in datasets
        if !isdir(joinpath(pwd(), "competitors", "results"))
                mkdir(joinpath(pwd(), "competitors", "results"))
        end
        if dataset == "foodweb"
                results = main(dataset, 3, 15, [0.05, 0.1], max_trial_number=5, features=false)
        else
                results = main(dataset, 3, 15, [0.01, 0.05], max_trial_number=5, features=false)
        end
        results = present_results(results)
        results.to_csv(joinpath(pwd(), "competitors", "results", string(dataset,".csv")))
        break
end
