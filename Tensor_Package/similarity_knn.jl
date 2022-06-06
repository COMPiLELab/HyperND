@sk_import neighbors: NearestNeighbors
const scipy_sparse_find = pyimport("scipy.sparse")."find"

export SuperSparse3Tensor,
       cosine_angle,
       area_triangle,
       mean_square_distance,
       mean_distance_squared,
       max_square_distance,
       scipy2julia_sparse,
       distance_matrix,
       time_distance_matrix,
       rbf_similarity_weights_tensor,
       rbf_similarity_weights

struct SuperSparse3Tensor
    I::Vector{Int64}
    J::Vector{Int64}
    K::Vector{Int64}
    V::Vector{Float64}
    n::Int64
end


###### Weight functions for similarity tensor
function cosine_angle(a,b,c)
    dif1 = (a - b)
    dif2 = (c - b)
    den = (norm(dif1)*norm(dif2))
    if den > 1e-20 cosine = dot(dif1, dif2) / den else cosine = 0 end
    return 1-cosine
end

function area_triangle(a,b,c)
    p = (a+b+c)/2
    if p*(p-a)*(p-b)*(p-c) < 0
        return 1
    else
        return sqrt(p*(p-a)*(p-b)*(p-c))
    end
end

function mean_square_distance(a,b,c)
    p = (a^2+b^2+c^2)/3
    return p
end

function mean_distance_squared(a,b,c)
    p = ((a+b+c)/3)^2
    return p
end

function max_square_distance(a,b,c)
    p = max(a,b,c) ^ 2
    return p
end



######## KNN matrix from dataset
function scipy2julia_sparse(Apy::PyObject)
    IA, JA, SA = scipy_sparse_find(Apy)
    return sparse(Int[i+1 for i in IA], Int[i+1 for i in JA], SA)
end

function distance_matrix(X, kn; mode="distance")
    nn = NearestNeighbors(n_neighbors=kn, p=2, n_jobs=-1)
    nn.fit(X)
    A = nn.kneighbors_graph(X, mode=mode)
    K = scipy2julia_sparse(A)
    K = max.(K,K')
    return K
end



function time_distance_matrix(X,kn; mode="distance")
    print("distance matrix:\t")
    @time distance_matrix(X,kn,mode=mode)
end


function rbf_similarity_weights(T::SuperSparse3Tensor; fast = true)
    valsT = T.V
    valsTT = copy(valsT)
    if fast
        valsTT = exp.(- ((valsT.^2) ./ 4) )
    else
        σ = zeros(Float64,T.n)
        for i in 1:T.n
            σ[i] = maximum(valsT[T.I .== i])
        end

        for (h , (i,j,k,v)) in enumerate(zip(T.I,T.J,T.K,T.V))
            valsTT[h] = exp(-4*v^2 / min(σ[i],σ[j],σ[k])^2 )
        end

    end
    return SuperSparse3Tensor(T.I,T.J,T.K,valsTT,T.n)
end


function rbf_similarity_weights(K::SparseArrays.SparseMatrixCSC; fast=true)
    I, J, valsK = findnz(K)
    n = size(K,1)
    furthest = maximum(K, dims=1)

    if fast
        # W_ij = - ||x_i -x_j||^2 / 2σ^2, for σ = √2
        valsTT = exp.(- ((valsK.^2 ) ./ 4) )
        W = sparse(I,J,valsTT,n,n)
    else
        # S_ij = exp(- 4 ||x_i -x_j||^2 / σ^2), for σ = distance xi to its k-th neighbor
        # W_ij = max(S_ij,S_ji)
        W = spzeros(n,n)
        for (i,j,v) in zip(I,J,valsK)
            W[i,j] = exp(-4*v^2 / min(furthest[i],furthest[j])^2 )
        end
    end
    return W
end
