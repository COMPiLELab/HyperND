using StatsBase
export edges_to_graph, inc_to_edges, clique_adj, sparse_to_dic, labels_to_bin,
        adj_to_edge_list, CV_splits, swap_train_splits


function edges_to_graph(edges)
    dict = OrderedDict()
    for edge in edges
        dict[edge[1]] = edge[2:end]
    end
    return dict
end

function inc_to_edges(INC)
    edges = []
    for col in eachcol(INC)
        push!(edges,  findall(x -> x != 0, col) .- 1)
    end
    return edges
end

function clique_adj(W, B)
    A = B * W * B'
    A = A - Diagonal(A)
    return A
end

function sparse_to_dic(M)
    I, J, V = findnz(M)
    dic = OrderedDict()
    for (i, j) in zip(I, J)
        i = i - 1
        j = j - 1
        if i in keys(dic)
            push!(dic[i], j)
        else
            dic[i] = [j]
        end
    end
    return dic
end


function labels_to_bin(labels)
    classes = Set(labels)
    print(classes)
    Y = zeros(length(labels), length(classes))
    for i in classes
        class_indices = findall(labels .== i)
        Y[class_indices, i] .= 1
    end
    return Y
end


function adj_to_edges_list(A)
    I, J, _ = findnz(A)
    return [I.-1 J.-1]
end

function CV_splits(training_inds, num_splits; split=0.5)
    all_splits = []
    for _ in 1:num_splits
        trial_train_splits = []
        trial_val_splits = []
        for inds in training_inds
            split_ind = Int64.(ceil(length(inds) * split))
            shuffle!(inds)
            append!(trial_train_splits, inds[1:split_ind])
            append!(trial_val_splits, inds[(split_ind + 1):end])
        end
        push!(all_splits, [trial_train_splits, trial_val_splits])
    end
    return all_splits
end

# train_splits = [[1674, 1178, 1200, 1882, 361, 1898, 886, 1381, 1796, 1826, 1678, 1237,1357, 1714, 2093, 1409, 1505, 300, 120], [1896, 2060, 104, 317, 732, 1457, 7, 196, 109, 2067, 735, 512, 829, 778, 69, 2123, 478, 826, 643], [1944, 1756, 1230, 2007, 102, 1043,
#  1026, 962, 2008, 596, 989, 1713, 1533], [31, 1185, 2105, 1646, 1728, 1587, 875, 1851, 2038, 1562, 1226, 1673, 1504, 1689], [605, 880, 408, 681, 278, 1715, 563, 541, 285, 447, 1611,
#   70, 811, 181, 413, 435, 273, 340, 1098, 27, 198],
#  [2111, 9, 1625, 1069, 229, 2023]]

function swap_train_splits(train_splits, swap_perc)
    len = length.(train_splits)
    n_per_class = Int64.(ceil.(swap_perc.*len))
    new_train_splits = copy(train_splits)
    for i in 1:length(train_splits)
        if i == length(train_splits)
            next_ind = 1
        else
            next_ind = i + 1
        end
        split_cur = shuffle(new_train_splits[i])
        swap_subset_cur = StatsBase.sample(split_cur, n_per_class[i]; replace=false)
        filtered_split_cur = filter(e->e∉swap_subset_cur,split_cur)
        new_train_splits[next_ind] = vcat(new_train_splits[next_ind], swap_subset_cur)
        split_next = shuffle(new_train_splits[next_ind])
        swap_subset_next = StatsBase.sample(split_next, n_per_class[next_ind]; replace=false)
        filtered_split_next = filter(e->e∉swap_subset_next,split_next)
        new_train_splits[i] = vcat(filtered_split_cur, swap_subset_next)
        new_train_splits[next_ind] = filtered_split_next
    end
    return new_train_splits, sum(n_per_class)
end

function update_y(y, swapped_train_splits)
    new_y = copy(y)
    for (i, split) in enumerate(swapped_train_splits)
        new_y[split] .= i
    end
    return new_y
end

# sw = swap_train_splits(train_splits, 0.1)
#
# sort(vcat(sw...))
# sort(vcat(train_splits...))
