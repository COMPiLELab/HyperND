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
        ScikitLearn


py"""
from scipy.sparse import csr_matrix, find
from scipy.sparse.linalg import norm

def create_matrix(I, J, V, n, m):
    return csr_matrix((V, (I - 1, J - 1)), (n, m))

def calculate_norm(M):
    return norm(M, axis=1)

"""

@sk_import utils: class_weight

function Kf_v2(K, D_n, D_e, W, x, p)

    return D_n*(K * (W * (2 * (D_e * (K'*(D_n * sparse(x)).^p)).^(1/p))))
end


function φ_incidence_ft(K, D_n, D_e, W, x, p)

    μ = 2 .* (D_e * (SparseMatrixCSC(K')*(D_n*sparse(x)).^p)).^(1/p)

    I, J, V = findnz(μ)
    n, m = size(μ)
    μ_v = py"calculate_norm"(py"create_matrix"(I, J, V, n, m))
    return 0.5 * sqrt(sum(K*W*μ_v))
end



function HOLS(X, B, W, D_n, D_e, y, p, α, train_inds, test_inds; balanced=true, ε=1e-6, max_iterations=100, tolerance=1e-3, verbose=false)
        Kfun(x) = Kf_v2(B, D_n, D_e, W, x, p)
        φ(x) = φ_incidence_ft(B, D_n, D_e, W, x, p)
        X_learned_from_features, err = Tensor_Package.HOLS_ft(Kfun, φ, X, α,
                                        max_iterations=max_iterations, tolerance=tolerance, verbose=verbose, normalize=true)
        dYdX = Diagonal(1 ./ vec(maximum(X_learned_from_features, dims=2))) * X_learned_from_features
        weight = class_weight.compute_class_weight("balanced", classes=np.unique(y[train_inds]),
                        y=y[train_inds])
        weights = Dict(zip(np.unique(y[train_inds]), weight))
        clf_logreg = Tensor_Package.LogReg(dYdX, y, train_inds, C=0.1, class_weights=weights)
        metrics_labels_f =  Tensor_Package.calc_metrics(clf_logreg.predict(dYdX[test_inds,:]), y[test_inds])
        return vcat([collect(metrics_labels_f)]...)


end


function CV_HOLS(X, B, W, D_n, D_e, y, ps, alphas, cv_splits)
        parameters = []
        accuracies = []
        for p in ps
                for alpha in alphas
                        split_accs = []
                        for (i, split) in enumerate(cv_splits)
                                @show p
                                @show alpha
                                acc_CV, _, _ = HOLS(X, B, W, D_n, D_e, y, p, alpha, split[1], split[2])
                                push!(split_accs, acc_CV)
                                @show acc_CV
                        end
                        push!(accuracies, mean(split_accs))
                        push!(parameters, [p, alpha])
                end
        end
        ind = argmax(accuracies)
        best_p, best_alpha = parameters[ind]
        # take the best p and alpha over all averages
        return best_p, best_alpha
end


function HOLS_HyperGCN(Input_features, B, W, D_n, D_e, H, y, p, α, train_inds, test_inds; balanced=true, ε=1e-6, max_iterations=25, tolerance=5*1e-2, verbose=true)
        Input_labels = zeros(Float64, length(y))
        Input_labels[train_inds] .= 1.0
        ϕ(x,p) = x.^p
        ψ(x,p) = 2 * (D_e * x).^(1/p)
        Kfun(x) = Kf_v2(B, D_n, W, x, u->ϕ(u,p), u->ψ(u,p))
        φ(x) = φ_incidence_ft(B, D_n, W, x, u->ϕ(u,p), u->ψ(u,p))
        accs = []

        ## 2. Only labels ---------------------------------------------------------
        X0 = (1 - ε) .* Input_labels .+ ε
        X_learned_from_labels, err = Tensor_Package.HOLS_ft(Kfun, φ, X0, α,
                                max_iterations=max_iterations, tolerance=tolerance, verbose=verbose, normalize=true)
        d = Diagonal(1 ./ vec(maximum(X_learned_from_labels, dims=2)))
        dY = d * X_learned_from_labels
        clf_logreg = Tensor_Package.LogReg(dY, y, train_inds, C=10)
        acc = Tensor_Package.accuracy(clf_logreg.predict(dY[test_inds,:]), y[test_inds])
        push!(accs, acc)
        # -----------------------------------------------------------------------

        ## 2. Only labels + H ---------------------------------------------------------
        clf_logreg = Tensor_Package.LogReg([dY H], y, train_inds, C=10)
        acc = Tensor_Package.accuracy(clf_logreg.predict([dY H][test_inds,:]), y[test_inds])
        push!(accs, acc)
        # -----------------------------------------------------------------------



        ## 2.Labels and features --------------------------------------------------
        X0 = (1 - ε) .* [Input_labels Input_features] .+ ε
        X_learned_from_features, err = Tensor_Package.HOLS_ft(Kfun, φ, X0, α,
                                max_iterations=max_iterations, tolerance=tolerance, verbose=verbose, normalize=true)

        d_2 = Diagonal(1 ./ vec(maximum(X_learned_from_features, dims=2)))
        dYdX = d_2 * X_learned_from_features
        clf_logreg = Tensor_Package.LogReg(dYdX, y, train_inds, C=10)
        acc = Tensor_Package.accuracy(clf_logreg.predict(dYdX[test_inds,:]), y[test_inds])
        push!(accs, acc)

        ## Labels + features + H
        clf_logreg = Tensor_Package.LogReg([dYdX H], y, train_inds, C=10)
        acc = Tensor_Package.accuracy(clf_logreg.predict([dYdX H][test_inds,:]), y[test_inds])
        push!(accs, acc)
        return accs


end

function HOLS_HyperGCN_validate(Input_features, B, W, D_n, D_e, H, y, ps, αs, train_inds, test_inds; balanced=true, ε=1e-6, max_iterations=25, tolerance=5*1e-2, verbose=true)
        Input_labels = zeros(Float64, length(y))
        Input_labels[train_inds] .= 1.0
        p = ps[1]
        ϕ(x,p) = x.^p
        ψ(x,p) = 2 * (D_e * x).^(1/p)
        Kfun(x) = Kf_v2(B, D_n, W, x, u->ϕ(u,p), u->ψ(u,p))
        φ(x) = φ_incidence_ft(B, D_n, W, x, u->ϕ(u,p), u->ψ(u,p))
        accs = []

        ## 2. Only labels ---------------------------------------------------------
        X0 = (1 - ε) .* Input_labels .+ ε
        X_learned_from_labels, err = Tensor_Package.HOLS_ft(Kfun, φ, X0, αs[1],
                                max_iterations=max_iterations, tolerance=tolerance, verbose=verbose, normalize=true)
        d = Diagonal(1 ./ vec(maximum(X_learned_from_labels, dims=2)))
        dY = d * X_learned_from_labels
        clf_logreg = Tensor_Package.LogReg(dY, y, train_inds, C=10)
        acc = Tensor_Package.accuracy(clf_logreg.predict(dY[test_inds,:]), y[test_inds])
        push!(accs, acc)
        # -----------------------------------------------------------------------

        ## 2. Only labels + H ---------------------------------------------------------
        clf_logreg = Tensor_Package.LogReg([dY H], y, train_inds, C=10)
        acc = Tensor_Package.accuracy(clf_logreg.predict([dY H][test_inds,:]), y[test_inds])
        push!(accs, acc)
        # -----------------------------------------------------------------------



        ## 2.Labels and features --------------------------------------------------
        p = ps[2]
        @show p
        ϕ(x,p) = x.^p
        ψ(x,p) = 2 * (D_e * x).^(1/p)
        Kfun(x) = Kf_v2(B, D_n, W, x, u->ϕ(u,p), u->ψ(u,p))
        φ(x) = φ_incidence_ft(B, D_n, W, x, u->ϕ(u,p), u->ψ(u,p))
        X0 = (1 - ε) .* [Input_labels Input_features] .+ ε
        X_learned_from_features, err = Tensor_Package.HOLS_ft(Kfun, φ, X0, αs[2],
                                max_iterations=max_iterations, tolerance=tolerance, verbose=verbose, normalize=true)

        d_2 = Diagonal(1 ./ vec(maximum(X_learned_from_features, dims=2)))
        dYdX = d_2 * X_learned_from_features
        clf_logreg = Tensor_Package.LogReg(dYdX, y, train_inds, C=10)
        acc = Tensor_Package.accuracy(clf_logreg.predict(dYdX[test_inds,:]), y[test_inds])
        push!(accs, acc)

        ## Labels + features + H
        clf_logreg = Tensor_Package.LogReg([dYdX H], y, train_inds, C=10)
        acc = Tensor_Package.accuracy(clf_logreg.predict([dYdX H][test_inds,:]), y[test_inds])
        push!(accs, acc)
        return accs


end

function CV_HOLS_HyperGCN(X, B, W, D_n, D_e, H, y, ps, alphas, cv_splits)
        parameters = []
        accuracies = []
        for p in ps
                for alpha in alphas
                        split_accs = []
                        for (i, split) in enumerate(cv_splits)

                                acc_CV = HOLS_HyperGCN(X, B, W, D_n, D_e, H, y, p, alpha, split[1], split[2])
                                push!(split_accs, acc_CV)
                                @show acc_CV
                        end
                        push!(accuracies, mean(hcat(split_accs...), dims=2))
                        push!(parameters, [p, alpha])
                end
        end
        #print(accuracies)
        ind =  map(x->x[2], argmax(hcat(accuracies...), dims=2))
        parameters[ind]
        # take the best p and alpha over all avereges
        return parameters[ind]
end
