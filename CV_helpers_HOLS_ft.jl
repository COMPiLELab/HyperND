using Base.Threads

hypergcn_jl = pyimport("hypergcn.hypergcn_jl")
sp = pyimport("scipy.sparse")

function CV_HyperLS(Kfun, φ, Input_features, INC, hypergraph, y, splits, grid, num_CV_trials, ε;
               max_iterations=max_iterations, tolerance=tolerance, verbose=verbose)
    @show grid
    num_grid_pts = length(grid)
    CV_accs = zeros(Float64, (num_grid_pts, 8))

    args = Dict("dataset" => "",
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

    n = length(y)
    for i = 1:num_grid_pts
            α = grid[i]
            # Get accuracy over each split for this hyperparameter setting
            hyperparam_accs = zeros(length(splits), 8)
            for (split_number, split) in enumerate(splits)
                # Run label spreading for each class
                X_HyperLS_labels = zeros(n, length(split))
                Input_labels = zeros(Float64, n)
                combined_train_inds = []
                for (class, class_split) in enumerate(split)
                    train_inds = class_split[1]
                    Input_labels[train_inds] .= 1.0
                    append!(combined_train_inds, train_inds)
                end

                combined_test_inds = []
                for (class, class_split) in enumerate(split)
                    test_inds = class_split[2]
                    append!(combined_test_inds, test_inds)
                end
                classes = Set(y)
                Y = zeros(n, length(classes))
                for label in classes
                    Y[y .== label, label] .= 1.0
                end
                X_features = sp.csr_matrix(Input_features)
                _, H, Z  = hypergcn_jl.train(args, X_features, Y, hypergraph, combined_train_inds .- 1, combined_test_inds .- 1)

                # Evaluate accuracy on this split
                ## 1. Only labels ---------------------------------------------------------
                X0 = (1 - ε) .* Input_labels .+ ε
                X_learned_from_labels, err = Tensor_Package.HOLS_ft(Kfun, φ, X0, α,
                                                        max_iterations=max_iterations, tolerance=tolerance, verbose=verbose, normalize=true)
                d = Diagonal(1 ./ vec(maximum(X_learned_from_labels, dims=2)))
                dY = d * X_learned_from_labels
                clf_logreg = Tensor_Package.LogReg(dY, y, combined_train_inds, C=10)
                acc = Tensor_Package.accuracy(clf_logreg.predict(dY[combined_test_inds,:]), y[combined_test_inds])
                hyperparam_accs[split_number, 1] = acc
                # -----------------------------------------------------------------------

                method_ind = 1
                ## 2,3. Only labels + H/Z ---------------------------------------------------------
                for (ind, matrix) in enumerate([[dY H],[dY Z]])
                    clf_logreg = Tensor_Package.LogReg(matrix, y, combined_train_inds, C=10)
                    acc = Tensor_Package.accuracy(clf_logreg.predict(matrix[combined_test_inds,:]), y[combined_test_inds])
                    hyperparam_accs[split_number, ind+method_ind] = acc
                end
                # -----------------------------------------------------------------------

                method_ind = 4


                ## 2.Labels and features --------------------------------------------------
                X0 = (1 - ε) .* [Input_labels Input_features] .+ ε
                X_learned_from_features, err = Tensor_Package.HOLS_ft(Kfun, φ, X0, α,
                                                        max_iterations=max_iterations, tolerance=tolerance, verbose=verbose, normalize=true)

                d_2 = Diagonal(1 ./ vec(maximum(X_learned_from_features, dims=2)))
                dYdX = d_2 * X_learned_from_features
                clf_logreg = Tensor_Package.LogReg(dYdX, y, combined_train_inds, C=10)
                acc = Tensor_Package.accuracy(clf_logreg.predict(dYdX[combined_test_inds,:]), y[combined_test_inds])
                hyperparam_accs[split_number, method_ind] = acc

                ## Labels + features + H/Z
                for (ind, matrix) in enumerate([[dYdX H],
                                                [dYdX Z],
                                                [dYdX[:, 1:length(split)] H],
                                                [dYdX[:, 1:length(split)] Z]])
                    clf_logreg = Tensor_Package.LogReg(matrix, y, combined_train_inds, C=10)
                    acc = Tensor_Package.accuracy(clf_logreg.predict(matrix[combined_test_inds,:]), y[combined_test_inds])
                    hyperparam_accs[split_number, ind+method_ind] = acc
                end


            end
            # Just record average accuracy
            CV_accs[i, :] = mean(hyperparam_accs, dims=1)
    end

    # Now return best parameters (best mean accuracy)
    return [grid[argmax(CV_accs[:, i])] for i in 1:8]
end


function validation_metrics(Kfun, φ, Input_features, INC, y, training_inds, α, ε;
                                    max_iterations=max_iterations, tolerance=tolerance, verbose=false)

    args = Dict("dataset" => "",
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
    n = length(y) # number of points
    K = length(training_inds) # number of classes

    # X_HOLS_ft_labels = zeros(n, length(training_inds))
    Input_labels = zeros(n, K)
    num_methods = length(α)
    whole_metrics = zeros(num_methods, 3)
    combined_train_inds = []
    for (class, class_inds) in enumerate(training_inds)
        Input_labels[class_inds, class] .= 1.0
        append!(combined_train_inds, class_inds)
    end

    X0 = (1 - ε) .* Input_labels .+ ε
    combined_test_inds = setdiff(1:length(y), combined_train_inds)
    _, H, Z  = HyperGCN(args, Input_features, y, INC, combined_train_inds .- 1, combined_test_inds .- 1 )
    # X_learned_from_labels, err = Tensor_Package.HOLS_ft(Kfun, φ, X0, α[1],
    #                                         max_iterations=max_iterations, tolerance=tolerance, verbose=verbose, normalize=true)
    # d = Diagonal(1 ./ vec(maximum(X_learned_from_labels, dims=2)))
    # XX = d * X_learned_from_labels
    # clf_logreg = Tensor_Package.LogReg(XX, y, combined_train_inds, C=10)
    # metrics = Tensor_Package.calc_metrics(clf_logreg.predict(XX[Not(combined_train_inds),:]), y[Not(combined_train_inds)])
    # whole_metrics[1, :] = metrics
    #push!(df, [dataset, percentage_of_known_labels,p,α,"labels-log-reg", acc,t])

    # -----------------------------------------------------------------------

    ## 2.Labels and features --------------------------------------------------
    XX_s = []
    X0 = (1 - ε) .* Input_labels  .+ ε
    for α_var in α[1:3]
        X_learned_from_features, err = Tensor_Package.HOLS_ft(Kfun, φ, X0, α_var,
                                            max_iterations=max_iterations, tolerance=tolerance, verbose=verbose, normalize=true)

        d_2 = Diagonal(1 ./ vec(maximum(X_learned_from_features, dims=2)))
        XX = d_2 * X_learned_from_features
        push!(XX_s, XX)
    end

    X0 = (1 - ε) .* [Input_labels Input_features] .+ ε
    for α_var in α[4:end]
        X_learned_from_features, err = Tensor_Package.HOLS_ft(Kfun, φ, X0, α_var,
                                            max_iterations=max_iterations, tolerance=tolerance, verbose=verbose, normalize=true)

        d_2 = Diagonal(1 ./ vec(maximum(X_learned_from_features, dims=2)))
        XX = d_2 * X_learned_from_features
        push!(XX_s, XX)
    end


    method_matrix_list = [ XX_s[1], [XX_s[2] H], XX_s[4], [XX_s[5] H] ]
    method_name_list = ["l", "l+H", "l+f", "l+f+H"]

    for (ind, matrix) in enumerate(method_matrix_list)

        clf_logreg = Tensor_Package.LogReg(matrix, y, combined_train_inds, C=10)
        metrics = Tensor_Package.calc_metrics(clf_logreg.predict(matrix[Not(combined_train_inds),:]), y[Not(combined_train_inds)])
        whole_metrics[ind, :] = metrics
    end

    return whole_metrics,method_name_list
end

# default 50/50 split
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

function Kf_v2(K, D_n, W, x, ϕ, ψ)
    return D_n*(K * (W * ψ(K'*ϕ(D_n * x))))
end

function φ_incidence_ft(K, D_n, W, x, ϕ, ψ)
    μ = ψ(K'*ϕ(D_n*sparse(x)))
    μ_v = []
    for i in 1:size(μ, 1)
        push!(μ_v, norm(μ[i, :], 2)^2)
    end
    φ = 0.5 * sqrt(sum(K*W*Float64.(μ_v)))
    return φ
end

function CV_binary(X, INC, hypergraph, W, D_n, D_e, y,
                   methods,
                   num_CV_trials,
                   percentage_of_known_labels;
                   balanced=true,
                   ε=1e-2,
                   ls_search_params = 0.1:0.1:.9,
                   max_iterations = 100,
                   tolerance=tolerance,
                   verbose=verbose)
    num_per_class = Tensor_Package.generate_known_labels(percentage_of_known_labels, balanced, y)
    training_inds = []
    for (label, num) in enumerate(num_per_class)
        class_inds = findall(y .== label)
        shuffle!(class_inds)
        push!(training_inds, class_inds[1:num])
    end
    splits = CV_splits(training_inds, num_CV_trials)

    df = DataFrame

    αs = Float64[]
    accs = Float64[]
    recs = Float64[]
    precs = Float64[]
    names = []
    variations = []
    ϕ(x,p) = x.^p
    ψ(x,p) = 2 * (D_e * x).^(1/p)
    @time begin
        for (p, method_name, method_type) in methods
            Kfun(x) = Kf_v2(INC, D_n, W, x, u->ϕ(u,p), u->ψ(u,p))
            φ(x) = φ_incidence_ft(INC, D_n, W, x, u->ϕ(u,p), u->ψ(u,p))
            println("$method_name...")
            α_best = CV_HyperLS(Kfun, φ, X, INC, hypergraph, y, splits, ls_search_params, num_CV_trials, ε;
                           max_iterations=max_iterations, tolerance=tolerance, verbose=verbose)
            # Now evaluate on the entire data
            metrics, method_variation = validation_metrics(Kfun, φ, X, hypergraph, y, training_inds, α_best, ε,
                                                max_iterations=max_iterations, tolerance=tolerance, verbose=false)
            print(metrics)
            for (i, variation) in enumerate(method_variation) #(["l", "l+H", "l+f", "l+f+H"]) #"l+f", "l+H", "l+Z", "l+f+H", "l+f+Z", "l+f_sub_H", "l+f_sub_Z"])
                push!(αs, α_best[i])
                push!(accs, metrics[i, 1])
                push!(precs, metrics[i, 2])
                push!(recs, metrics[i, 3])
                push!(names, method_name)
                push!(variations, variation)
            end
        end
    end

    final_data = DataFrame()
    final_data[!, :α] = αs
    final_data[!, :acc] = accs
    final_data[!, :rec] = recs
    final_data[!, :prec] = precs
    final_data[!, :method_name] = names
    final_data[!, :size] .= length(y)
    final_data[!, :percentage_of_known_labels] .= percentage_of_known_labels
    final_data[:, :balanced] .= balanced
    final_data[!, :variation] = variations
    return (final_data, training_inds)
end
