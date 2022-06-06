export accuracy, precision, recall, train_test_val_split, train_test_split

using CSV, ScikitLearn, DataFrames, StatsBase, DataStructures, MLDatasets

@sk_import linear_model: LogisticRegression
@sk_import exceptions : ConvergenceWarning
pyimport("warnings").simplefilter("ignore", category=ConvergenceWarning)

@pyimport pickle
@pyimport re



struct SuperSparse3Tensor
    I::Vector{Int64}
    J::Vector{Int64}
    K::Vector{Int64}
    V::Vector{Float64}
    n::Int64
end


function prepare_uci_data(dataset_name)
   dataset = UCIData.dataset(dataset_name)
   titles = unique(dataset.target)
   d = Dict(title => i  for (i, title) in zip(1:length(titles), titles))
   n_dataset = DataFrame(replace!(convert(Matrix, dataset), d...))
   rename!(n_dataset, names(dataset))
   y = n_dataset.target
   X = convert(Matrix, dataset[!, 2:end-1])
   X = convert(Array{Float64,2}, X)
   return X, y
end


function accuracy(y_predicted, y_actual)
    return ( sum(y_predicted .== y_actual) ./ length(y_actual) )*100
end

function precision(y_predicted, y_actual; method="mean")
    if method=="mean"
        p = 0
        for label in unique(y_predicted)
            p += sum((y_predicted .== label) .* (y_actual .== label)) / sum(y_predicted .== label)
        end
        p = p/length(unique(y_predicted)) *100
    elseif method=="min"
        p = Inf
        for label in unique(y_predicted)
            p = min(p, sum((y_predicted .== label) .* (y_actual .== label)) / sum(y_predicted .== label) ) *100
        end
    else
        @assert false
    end
    return p
end

function recall(y_predicted, y_actual; method="mean")
    if method=="mean"
        p = 0
        for label in unique(y_predicted)
            p += sum((y_predicted .== label) .* (y_actual .== label)) / sum(y_actual .== label)
        end
        p = p/length(unique(y_predicted)) *100
    elseif method=="min"
        p = Inf
        for label in unique(y_predicted)
            p = min(p, sum((y_predicted .== label) .* (y_actual .== label)) / sum(y_actual .== label) ) *100
        end
    else
        @assert false
    end
    return p
end

function train_test_val_split(y, perc_train, perc_val; balanced=false)
    n = length(y)
    num_classes = length(unique(y))
    Y_test = zeros(n, num_classes)
    Y_train = zeros(n, num_classes)
    Y_val = zeros(n, num_classes)
    test_mask = zeros(n)
    train_mask = zeros(n)
    val_mask = zeros(n)
    num_train_per_class = Tensor_Package.generate_known_labels(perc_train, balanced, y)
    print(num_train_per_class)
    num_val_per_class = Tensor_Package.generate_known_labels(perc_val, balanced, y)
    for (label, num_train, num_val) in zip(1:num_classes, num_train_per_class, num_val_per_class)
        class_inds = findall(y .== label)
        print(length(class_inds))
        shuffle!(class_inds)

        train_indices = class_inds[1:num_train]
        Y_train[train_indices, label] .= 1
        train_mask[train_indices] .= 1

        test_indices = class_inds[num_train+num_val+1:end]
        Y_test[test_indices, label] .= 1
        test_mask[test_indices] .= 1

        val_indices = class_inds[num_train+1:num_train+num_val]
        Y_val[val_indices, label] .= 1
        val_mask[val_indices] .= 1

    end
    return Y_train, Y_test, Y_val, Bool.(train_mask), Bool.(test_mask), Bool.(val_mask)
end

function train_test_val_split_indices(y, perc_train, perc_val; balanced=false)
    n = length(y)
    num_classes = length(unique(y))
    num_train_per_class = Tensor_Package.generate_known_labels(perc_train, balanced, y)
    num_val_per_class = Tensor_Package.generate_known_labels(perc_val, balanced, y)
    y_onehot = spzeros(n, num_classes)
    @show size(y_onehot)
    train, test, val = [], [], []
    for (label, num_train, num_val) in zip(1:num_classes, num_train_per_class, num_val_per_class)
        class_inds = findall(y .== label)
        shuffle!(class_inds)
        y_onehot[class_inds[1:num_train], label] .= 1

        train_indices = class_inds[1:num_train]
        push!(train, train_indices)

        val_indices = class_inds[num_train+1:num_train+num_val]
        push!(val, val_indices)

        test_indices = class_inds[num_train+num_val+1:end]
        push!(test, test_indices)
    end
    @show size(y_onehot)
    return train, vcat(train...), vcat(test...), vcat(val...), y_onehot
end

function fill_array(arr, inds1, inds2)
    for (ind1, ind2) in zip(inds1, inds2)
        arr[ind1, ind2] = 1
    end
    return arr
end


function train_test_split(X, y, perc_train; balanced=false)
    n = length(y)
    num_classes = length(unique(y))
    num_train_per_class = Tensor_Package.generate_known_labels(perc_train, balanced, y)
    num_train_total = sum(num_train_per_class)
    Y_test = zeros(n - num_train_total, num_classes)
    Y_train = zeros(num_train_total, num_classes)
    full_train_indices = []
    full_test_indices = []
    for (label, num_train) in zip(1:num_classes, num_train_per_class)
        class_inds = findall(y .== label)
        shuffle!(class_inds)

        train_indices = class_inds[1:num_train]
        push!(full_train_indices, train_indices...)
        #Y_train[(label-1)*num_train + 1:label*num_train, label] .= 1

        num_test = length(class_inds) - num_train
        test_indices = class_inds[num_train+1:end]
        push!(full_test_indices, test_indices...)
        #Y_test[(label-1)*num_test + 1:label*num_test, label] .= 1

    end
    Y_train = fill_array(Y_train, 1:size(Y_train)[1], y[full_train_indices])
    Y_test = fill_array(Y_test, 1:size(Y_test)[1], y[full_test_indices])
    return Y_train, Y_test, X[full_train_indices, :], X[full_test_indices, :], X[vcat(full_train_indices, full_test_indices), :], y[vcat(full_train_indices, full_test_indices)]
end


function train_test_split(X, y, perc_train; balanced=false)
#minus(indx, x) = setdiff(1:length(y), indx)
        n = length(y)
        num_classes = length(unique(y))
        num_train_per_class = Tensor_Package.generate_known_labels(perc_train, balanced, y)
        num_train_total = sum(num_train_per_class)
        Y_test = zeros(n - num_train_total, num_classes)
        Y_train = zeros(num_train_total, num_classes)
        full_train_indices = []
        full_test_indices = []
        for (label, num_train) in zip(1:num_classes, num_train_per_class)
           class_inds = findall(y .== label)
           shuffle!(class_inds)

           train_indices = class_inds[1:num_train]
           push!(full_train_indices, train_indices...)
           Y_train[(label-1)*num_train + 1:label*num_train, label] .= 1

           num_test = length(class_inds) - num_train
           test_indices = class_inds[num_train+1:end]
           push!(full_test_indices, test_indices...)
           Y_test[(label-1)*num_test + 1:label*num_test, label] .= 1

        end
        return Y_train, Y_test, X[full_train_indices, :], X[full_test_indices, :], X[vcat(full_train_indices, full_test_indices), :]
end

function CV_splits(training_inds, num_splits, split=0.5)
    all_splits = []
    for _ in 1:num_splits
        trial_splits = []
        for inds in training_inds
            split_ind = Int64.(ceil(length(inds) * split))
            class_splits = []
            shuffle!(inds)
            push!(trial_splits, (inds[1:split_ind], inds[(split_ind + 1):end]))
        end
        push!(all_splits, trial_splits)
    end
    return all_splits
end




function read_text_hypergraph_data(folder; features=true)
    edges_ = []
    @pywith pybuiltin("open")("./data/$folder/hyperedges-$folder.txt", "r") as f begin
        edges_ = [[parse(Int64, a) for a in re.split("[,\\t]", strip(d))] for d in f.readlines()]
    end


    labels = Int64[]
    open("./data/$folder/node-labels-$folder.txt") do f
        for line in eachline(f)
            push!(labels, parse(Int64, line))
        end
    end

    if features == true
        features = scipy2julia_sparse(myunpickle("./data/$folder/features.pickle"))
    else
        features = nothing
    end

    return edges_, labels, features
end

function load_data(dataset_name, kn)
    #### UCI datasets ###################
    try
        X, y = Tensor_Package.prepare_uci_data(dataset_name)
        adj_matrix = Tensor_Package.distance_matrix(X, kn, mode="connectivity")
        return X, y, adj_matrix
    catch KeyErrror
        println("$dataset_name is not a part of UCI datasets or the specified name is spelled wrong.")
    end
    ##################################

   #### Matlab datasets ###################
   mat_dataset_names = ["3sources","BBC4view_685","BBCSport2view_544","cora","UCI_mfeat", "citeseer", "WikipediaArticles"]
   #there are not features here, so it return adj_matrix twice
   if dataset_name in mat_dataset_names
       data = MAT.matread("./data/matlab_multilayer_data/"*dataset_name*"/knn_10.mat")
       y = data["labels"][:]
       adj_matrix= data["W_cell"][1]
       return adj_matrix, y, adj_matrix
   else
      println("$dataset_name is not a part of our matlab multilayer data.")
   end

   if dataset_name in ["BASEHOCK", "COIL20", "lung", "PCMAC"]
       vars = matread("data/mat/$dataset_name.mat")
       X, y = vars["X"], vars["Y"]
       adj_matrix = Tensor_Package.distance_matrix(X, kn, mode="connectivity")
       return X, y, adj_matrix
   end


   ##################################

   #### Pendigits ###################
   if dataset_name == "pendigits"
        train = Array(CSV.read("./data/pendigits.csv", DataFrame))
        X = train[:,1:end-1]
        adj_matrix = distance_matrix(X, kn, mode="connectivity")
        y = train[:,end] .+ 1
        return X, y, adj_matrix
   ##################################
   #### Optdigits ###################
   elseif dataset_name == "optdigits"
       train = Array(CSV.read("./data/optdigits.csv", DataFrame))
       X = train[:,1:end-1]
       # adj_matrix = Tensor_Package.distance_matrix(X, kn, mode="connectivity")
       adj_matrix = nothing
       y = train[:,end] .+ 1
       return X, y, adj_matrix
   elseif dataset_name == "skdigits"
       data = load_digits()
       X, y = data["data"], data["target"]
       adj_matrix = Tensor_Package.distance_matrix(X, kn, mode="connectivity")
       return X, y .+ 1, adj_matrix
   ##################################
   #### F-MNIST #####################
   elseif dataset_name == "f-mnist"
       train_x, train_y = FashionMNIST.traindata()
       rows, cols, num = size(train_x)
       X = reshape(train_x, (rows*cols, num))'
       X = convert(Array{Float64,2}, X)
       adj_matrix = distance_matrix(X, kn, mode="connectivity")
       y = train_y
       return X, y, adj_matrix
   #### MNIST #######################
   elseif dataset_name == "mnist"
       train_x, train_y = MNIST.traindata()
       rows, cols, num = size(train_x)
       X = reshape(train_x, (rows*cols, num))'
       X = convert(Array{Float64,2}, X)
       # adj_matrix = distance_matrix(X, kn, mode="connectivity")
       adj_matrix = nothing
       y = train_y
       return X, y, adj_matrix
   ##################################
   else
       println("$dataset_name is not one of the digits datasets.")
   end

   if dataset_name in readdir("data/custom/")
       files = readdir("./data/custom/$dataset_name")
       if length(files) == 1
           data_file = files[1]
           if endswith(data_file, ".csv")
               data = Array(CSV.read("./data/custom/$dataset_name/$data_file"))
               X = data[:,1:end-1]
               y = data[:,end] .+ 1
               adj_matrix = distance_matrix(X, kn, mode="connectivity")
               return X, y, adj_matrix
           elseif endswith(data_file, ".mat")
               data = MAT.matread("./data/custom/$dataset_name/$data_file")
               y = data["labels"][:]
               adj_matrix= data["W_cell"][1]
               return adj_matrix, y, adj_matrix
           elseif endswith(data_file, r".xls[x]?")
               data = convert(Array, DataFrame(load("./data/custom/$dataset_name/$data_file", split(data_file, '.')[1])))
               X = data[:,1:end-1]
               y = data[:,end] .+ 1
               adj_matrix = distance_matrix(X, kn, mode="connectivity")
               return X, y, adj_matrix
           end
       elseif length(files) == 2
           features_file = filter(x -> startswith(x, 'X'), files)[1]
           labels_file = filter(x -> startswith(x, 'y'), files)[1]
           if endswith(features_file, ".npy")
               X = npzread("./data/custom/$dataset_name/$features_file")
               y = npzread("./data/custom/$dataset_name/$labels_file")
               adj_matrix = distance_matrix(X, kn, mode="connectivity")
               return X, y, adj_matrix
           end
       end
   else
       println("$dataset_name is not one of the datasets you provided.")
   end
   return nothing, nothing
end

function remove!(a, item)
    deleteat!(a, findall(x->x==item, a))
end


function remove_unconnected_nodes(edges, inds, n)
    new_edges = []
    new_node_set = 1:length(inds)
    new_edges_dict = Dict( old => new for (old, new) in zip(inds, new_node_set) )
    inds = Set(inds)
    for edge in edges
        push!(new_edges, collect(setdiff(Set(edge), inds)))
    end
    return new_edges
end



function prepare_incidence_data(dataset_name, order, kn; features=true, type="network")
    if type == "points"
        X, y, _ = load_data(dataset_name, kn)
        K = distance_matrix(X, kn; mode="distance")
        INC, w = incid_mat(K, order, Tensor_Package.area_triangle)
        dw = spdiagm(0=>Float64.(w))
        edges_ = nothing
    elseif type == "network"
        edges_, y, X = read_text_hypergraph_data(dataset_name, features=features)
        @show size(X)
        inds = find_rows_not_in_graph(edges_,)
        INC = network_to_INC(edges_, length(y))
        n = length(y)
        if X != nothing
            INC, y, X =  INC[inds, :], y[inds], X[inds, :]
        else
            INC, y =  INC[inds, :], y[inds]
        end
        dw = spdiagm(0 => ones(Float64, size(INC, 2)))
    end
    KW = INC*dw
    D_n = spdiagm(0 => vec(1.0 ./ sqrt.(sum(KW,dims=2))) )
    D_e = spdiagm(0 => vec(1.0 ./ sum(INC,dims=1)) )
    return X, y, INC, dw, D_n, D_e, edges_
end

function load_HyperGCN_data(dataset_name)
    edges_, y, X = read_text_hypergraph_data(dataset_name)
    inds = find_rows_not_in_graph(edges_)
    new_inds = 1:length(inds)
    inds_dict = Dict(zip(inds, new_inds))
    hypergraph = Dict()
    for (i, edge) in enumerate(edges_)
        new_edge = [inds_dict[e] for e in edge]
        hypergraph[i] = new_edge .- 1
    end
    return hypergraph, y, X
end


function prepare_features(X, features_weight, combined_train_inds; throw_perc=0.5, throw="initial")

    X = X .* features_weight
    test_inds = setdiff(Set(1:size(X,1)),Set(combined_train_inds))

    if throw == "initial"
        X[Not(combined_train_inds), :] .= 0.0
    elseif (throw == "nodes") || (throw == "rows")
        amount_of_thrown_labels = Int(round(throw_perc * length(test_inds)))
        thrown_labels = StatsBase.sample(test_inds, amount_of_thrown_labels,replace=false)
        #if amount_of_thrown_labels == size(X, 1)
        X[thrown_labels, :] .= 0.0
        #end
    elseif throw == "elements"
        amount_of_thrown_labels = Int(round(throw_perc * length(test_inds)))
        thrown_labels_rows = StatsBase.sample(test_inds, amount_of_thrown_labels,replace=false)
        _,js,_ = findnz(X[thrown_labels_rows,:])
        amount_of_thrown_labels_cols = Int(round(throw_perc * length(js)))
        thrown_labels_cols = StatsBase.sample(js, amount_of_thrown_labels_cols,replace=false)
        X[thrown_labels_rows, thrown_labels_cols] .= 0.0
    end

    return X
end

function myunpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end


function make_classification(X, c)
    Y_pred = map(x->x[2], argmax(X[:, 1:c], dims=2))
    return Y_pred
end

function choose_replacement(list_of_changes, label, labels_by_class, finished_labels)
    if argmin([length(a) for a in labels_by_class]) == argmax([length(a) for a in labels_by_class])
        repl = rand(setdiff(Set(list_of_changes[label]), finished_labels))
    else
        repl = argmax([length(a) for a in labels_by_class])
    end
end


function swap_labels(num_of_classes, labels, labels_by_class)
    list_of_changes = reverse(collect((combinations(1:num_of_classes, num_of_classes-1))))
    replacements = []
    finished_labels = Set()
    for label in labels
        replacement = choose_replacement(list_of_changes, label, labels_by_class, finished_labels)
        try
            pop!(labels_by_class[replacement])
        catch ArgumentError
            push!(finished_labels, replacement)
            replacement = choose_replacement(list_of_changes, label, labels_by_class, finished_labels)
        end
        push!(replacements, replacement)
    end
end

function random_perturbation(num_of_classes, inds)
    list_of_changes = reverse(collect((combinations(1:5, 4))))
end

function LogReg(X, y, train_inds; C=10, class_weights="balanced")
        # ε = 1e-4
        # X = (1 - ε) .* X .+ ε
        pyimport("warnings").simplefilter("ignore", category=ConvergenceWarning)

        X_train = X[train_inds, :]
        y_train = y[train_inds]
        clf = LogisticRegression(C=C, verbose=0, tol=1e-4, max_iter = 100, class_weight=class_weights) # default values of scikitlearn logreg
        clf = clf.fit(X_train, y_train)
        return clf
end

function calc_metrics(Y_pred, Y_real)

    return [Tensor_Package.accuracy(Y_pred, Y_real),
        Tensor_Package.recall(Y_pred, Y_real),
        Tensor_Package.precision(Y_pred, Y_real)]
end

function subsample_features(X, combined_train_inds; throw_perc=0.5, throw="initial")

    test_inds = setdiff(Set(1:size(X,1)),Set(combined_train_inds))
    test_inds = collect(test_inds)
    if throw == "initial"
        X[Not(combined_train_inds), :] .= 0.0
    elseif (throw == "nodes") || (throw == "rows")
        amount_of_thrown_labels = Int(round(throw_perc * length(test_inds)))
        thrown_labels = StatsBase.sample(test_inds, amount_of_thrown_labels,replace=false)
        X[thrown_labels, :] .= 0.0
    elseif throw == "elements"
        amount_of_thrown_labels = Int(round(throw_perc * length(test_inds)))
        thrown_labels_rows = StatsBase.sample(test_inds, amount_of_thrown_labels,replace=false)
        _,js,_ = findnz(X[thrown_labels_rows,:])
        amount_of_thrown_labels_cols = Int(round(throw_perc * length(js)))
        thrown_labels_cols = StatsBase.sample(js, amount_of_thrown_labels_cols,replace=false)
        X[thrown_labels_rows, thrown_labels_cols] .= 0.0
    end

    return X
end

function prepare_train_inds(y, known_labels_per_class)
        n = length(y)
        training_inds = []
        for (label, num) in enumerate(known_labels_per_class)
                class_inds = findall(y .== label)
                shuffle!(class_inds)
                push!(training_inds, class_inds[1:num])
        end

        combined_training_inds = vcat(training_inds...)
        return combined_training_inds, training_inds
end

function prepare_features_and_labels(X,y,percentage_of_known_labels; percentage_of_known_features=1, feature_remove_style="rows")

        balanced = true
        known_labels_per_class = Tensor_Package.generate_known_labels(percentage_of_known_labels, balanced, y)
        num_of_classes = length(known_labels_per_class)
        n = length(y)


        combined_train_inds, train_inds = prepare_train_inds(y, known_labels_per_class)

        Y = zeros(n, num_of_classes)
        for (i, train_batch) in enumerate(train_inds)
                Y[train_batch, i] .= 1.0
        end

        X_f =  subsample_features(X, combined_train_inds, throw_perc=1-percentage_of_known_features, throw=feature_remove_style)
        X_features =  X_f # [Y features_weight.*X_f]
        X_labels = Y
        return X_features, X_labels, combined_train_inds, train_inds
end
