

function CV_splits(training_inds, num_splits, split=0.5)
    all_splits = []
    for _ in 1:num_splits
        trial_train_splits = []
        trial_val_splits = []
        for inds in training_inds
            split_ind = Int64.(ceil(length(inds) * split))
            class_splits = []
            shuffle!(inds)
            push!(trial_train_splits, inds[1:split_ind])
            push!(trial_val_splits, inds[(split_ind + 1):end])
        end
        push!(all_splits, (vcat(trial_train_splits...), vcat(trial_val_splits...)))
    end
    return all_splits
end


function CV_run(args, method, X, y, train_inds, test_inds,
                   num_CV_trials;
                   balanced=true)
    splits = CV_splits(train_inds, num_CV_trials)
    df = DataFrame

    accs = Float64[]
    for split in splits
        train_inds = split[1]
        test_inds = collect(setdiff(Set(1:length(y)), Set(train_inds)))
        acc, _, _ = HGNN(args, X, y, train_inds, test_inds)
        push!(accs, acc)
    end

    return mean(accs)

    for split in splits
        val_inds = split[2]
        test_inds = collect(setdiff(Set(1:length(y)), Set(train_inds)))
        acc, _, _ = HGNN(args, X, y, train_inds, test_inds)
        push!(accs, acc)
    end

    #acc, rec, prec = validation_metrics(y, training_inds, A, DG_isqrt, T, DH_isqrt, B,
    #                                     α_best, β_best, f, ε, max_iterations)
end
