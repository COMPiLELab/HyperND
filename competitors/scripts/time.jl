using PyCall
# I had to do this to make pytorch work with the new numpy library, it's optional
os = pyimport("os")
os.add_dll_directory(raw"E:\pytorch_venv\Library\bin")


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

include("appnp.jl")
include("hgnn.jl")
include("sgc.jl")
include("sce.jl")
include("hypergcn.jl")
include("hols.jl")

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
                train_splits, train_inds, test_inds, val_inds = Tensor_Package.train_test_val_split_indices(y, train_perc,
                                                        train_perc; balanced=balanced)
                train_data = vcat([train_inds, val_inds]...)
                train_val_splits = [vcat([t, v]...) for (t, v) in zip(train_inds, val_inds)]
                train_inds, test_inds, val_inds = train_inds .- 1, test_inds .- 1, val_inds .- 1

                p = [1, 2, 3, 5, 10]
                α = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
                hols_time = @timed(HOLS(X, B, W, D_n, D_e, y, rand(p), rand(α), train_data .+ 1, test_inds .+ 1))

                #for (i, vals) in combinations
                appnp_time = @timed(APPNP((2000, 0.01, 0.005), X, y, clique_edges, train_inds, test_inds, val_inds))
                hgnn_time = @timed(HGNN((600, 0.001, 0.0005), X, y, vcat([train_inds, val_inds]...), test_inds))
                hypergcn_time = @timed(HyperGCN((200, 0.01, 0.0005), X, y, hypergraph, vcat([train_inds, val_inds]...), test_inds))
                sgc_time = @timed(SGC((30, 0.01, 0.0), A, X, y, train_inds, test_inds, val_inds))
                sce_time = @timed(SCE((100, 0.2, 5e-6), A, X, y, train_inds, test_inds, val_inds))
                results = [dataset_name, appnp_time[2], hgnn_time[2], hypergcn_time[2], sgc_time[2], sce_time[2], hols_time[2]]
             end
    end
    return results
end




function present_results(results)
        columns = ["dataset_name", "appnp", "hgnn", "hypergcn", "sgc", "sce", "hols_labels_features"]
        return pd.DataFrame(results, columns=columns)
end

Random.seed!(15)

percentages = [0.042,0.052,0.052,0.04,0.008, 0.05, 0.05, 0.05, 0.05, 0.05]
datasets  = ["citeseer", "cora-cit", "cora-author", "dblp", "pubmed", "senate-bills", "pendigits", "optdigits", "contact-high-school", "foodweb"]
all_results = []
for dataset in datasets
        if !isdir(joinpath(pwd(), "competitors", "results"))
                mkdir(joinpath(pwd(), "competitors", "results"))
        end
        if dataset in ["pendigits", "optdigits"]
                results = main(dataset, 3, 7, [0.05], max_trial_number=1, features=false, type="points")
        elseif dataset in ["senate-bills", "contact-high-school", "foodweb"]
                results = main(dataset, 3, 7, [0.05], max_trial_number=1, features=false, type="network")
        else
                results = main(dataset, 3, 7, [0.05], max_trial_number=1, features=true, type="network")
        end
        push!(all_results, results)
        results = present_results(all_results)
        results.to_csv(joinpath(pwd(), "competitors", "results", string("time.csv")))
end
