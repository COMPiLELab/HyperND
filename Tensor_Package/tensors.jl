export compute_matrix_and_tensor, Tf, Tf_matrix, Ax, B_matrix, neighbor_neighbor_tensor, unfolded_triangle_tensor_2, row_normalize, triangle_tensor

struct SuperSparse3Tensor
    I::Vector{Int64}
    J::Vector{Int64}
    K::Vector{Int64}
    V::Vector{Float64}
    n::Int64
end

function Tf(T::SuperSparse3Tensor, DH_isqrt::Vector{Float64}, f, x)
    x = DH_isqrt .* x
    n = length(x)

    y = zeros(Float64, n)
    for (i, j, k, v) in zip(T.I, T.J, T.K, T.V)
        @inbounds y[i] += v * f(x[j], x[k])
    end
    return DH_isqrt .* y
end

function Kf(K, DH_isqrt, f, x)
    x = DH_isqrt .* x
    n = length(x)

    y = zeros(Float64, n)
    for (i, j, k, v) in zip(T.I, T.J, T.K, T.V)
        @inbounds y[i] += v * f(x[j], x[k])
    end
    return DH_isqrt .* y
end

function Tf_matrix(T::SuperSparse3Tensor, DH_isqrt, f, x)
    x = DH_isqrt * x

    y = zeros(Float64, size(x))
    for (i, j, k, v) in zip(T.I, T.J, T.K, T.V)
        @inbounds y[i, :] += v * f(x[j, :], x[k, :])
    end
    return DH_isqrt * y
end

function Ax(A, DG_isqrt, x)
    x = DG_isqrt .* x[:]
    y = A * x
    return DG_isqrt .* y
end

function Px(A, DG_isqrt, x)
    y = A * x
    return DG_isqrt .* (DG_isqrt .* y)
end

function B_matrix(T::SuperSparse3Tensor)
    # Returns B_jk = sum_i T_ijk, for a row-normalized tensor T
    return sparse(T.J, T.K, T.V, T.n, T.n)
end

function neighbor_neighbor_tensor(A; weight_function = weight_function)
    # INPUT A = adjacency matrix (knn matrix) weighted with distances (ie. sparsified distance matrix)
    Is = []
    Js = []
    Ks = []
    Vs = []

    # Incidence matrix and degree vector for Matlab (Matthias code)
    INC_I = []
    INC_J = []
    w = []

    II, JJ = findnz(A)
    c = 1 # counter for INC_J
    for (i, j) in zip(II, JJ)
        indices = filter!(e->e≠i, findall(A[j, :] .!= 0))
        for k in indices
            v = weight_function(A[i, j], A[j, k], A[k, i])
            push!(Is, i)
            push!(Js, j)
            push!(Ks, k)
            push!(Vs, v)

            push!(INC_I, i,j,k)
            push!(INC_J, c,c,c)
            push!(w, c)
            c += 1
        end
    end

    T = SuperSparse3Tensor(Is,Js,Ks,Vs,size(A,1))
    INC = sparse(INC_I,INC_J,1)
    return T, INC, w
end

function unfolded_triangle_tensor_2(A::SparseArrays.SparseMatrixCSC{Float64,Int64}; weight_function = (x,y,z)-> 1)
    # input : distance matrix A[i,j] = ||xi - xj||
    n = size(A, 1)  # number of nodes
    d = vec(sum(A, dims=2))  # degree vector
    deg_order = zeros(Int64, n)
    deg_order[sortperm(d)] = 1:n
    #     triangles = []
    rows = []
    cols = []
    vals = []
    ϕ(i,j) = tensor2matrix_index(i,j,n)


    for i = 1:n
        N_i = findnz(A[:, i])[1]  # neighbors of node i
        # only look over pairs of neighbors with higher degree order
        N_i_keep = [j for j in N_i if deg_order[j] > deg_order[i]]
        for jj = 1:length(N_i_keep)
            for kk = (jj + 1):length(N_i_keep)
                j = N_i_keep[jj]
                k = N_i_keep[kk]
                # check for triangle
                if A[j, k] > 0
                    # triangle (i, j, k)
                    #push!(triangles, (i, j, k))
                    rows = [rows; [i;i]]; cols = [cols; ϕ(j,k)]; cols = [cols; ϕ(k,j)];
                    rows = [rows; [j;j]]; cols = [cols; ϕ(i,k)]; cols = [cols; ϕ(k,i)];
                    rows = [rows; [k;k]]; cols = [cols; ϕ(j,i)]; cols = [cols; ϕ(i,j)];
                    #@show A[i,j],A[j,k],A[i,k]
                    #@show X[i, :], X[j, :], X[k, :]
                    #@show i, j, k
                    for _ in 1:6 push!(vals, weight_function(A[i,j],A[j,k],A[i,k])) end
                    #for _ in 1:6 push!(vals, 1) end
                end
            end
        end
    end
    T = sparse(rows,cols,Float64.(vals),n,n*n)
    return T
end


# this is the main function, as it states in the name, calculates everything,
#you can change here which function to use for creating the T
function compute_matrix_and_tensor(X, kn, weight_function)
    n = size(X, 1)
    K = distance_matrix(X, kn)
    W = rbf_similarity_weights(K,fast=false)
    g = SimpleGraph(W);
    #if !is_connected(g) println("the graph is NOT connected") end

    # T = unfolded_triangle_tensor_2(K) # for a binary triangle tensor

    # T = unfolded_triangle_tensor_2(K, weight_function=weight_function)  # for a weighted triangle tensor
    T,INC,w = neighbor_neighbor_tensor(K, weight_function=weight_function)
    T = rbf_similarity_weights(T,fast=false)

    T, DH_isqrt = row_normalize(T, W)
    A, DG_isqrt, = row_normalize(W)
    B = B_matrix(T)

    return A, DG_isqrt, T, DH_isqrt, B, INC', w
end

# Return row-scaled version of T
function row_normalize(T::SuperSparse3Tensor, A::SparseArrays.SparseMatrixCSC)
    sums = zeros(Float64, T.n)
    for (i, v) in zip(T.I, T.V)
        sums[i] += v
    end

    # deal with zero sums
    new_I = copy(T.I)
    new_J = copy(T.J)
    new_K = copy(T.K)
    new_V = copy(T.V)
    for i in findall(sums .== 0)
        nbrs = findnz(A[:, i])[1]
        if length(nbrs) == 0
            throw("NO DATA ERROR: $i")
        end
        for nbr in nbrs
            push!(new_I, i,   i)
            push!(new_J, nbr, i)
            push!(new_K, i,   nbr)
            push!(new_V, 1.0, 1.0)
            sums[i] += 2.0
        end
    end

    DH_isqrt = vec(1.0 ./ sqrt.(sums))
    return (SuperSparse3Tensor(new_I, new_J, new_K, new_V, T.n), DH_isqrt)
end

function row_normalize(A::SparseArrays.SparseMatrixCSC)
    d = vec(sum(A,dims=2))
    DG_isqrt = 1.0 ./ sqrt.(d);
    if sum(DG_isqrt .== Inf) > 0
        error("Adjacency matrix has a zero row, can't form normalized adjacency")
    end
    return (A, DG_isqrt)
end

function compute_matrix_and_tensor_binary(W::SparseArrays.SparseMatrixCSC;  noise = 0)
    n = size(W,1)
    #println("n,nnz,nnz/n:\t\t"string(n)", "string(nnz(W))", "*string(nnz(W)/n))
    if noise > 0
        W = sparse(W + Int64.(sprand(Bool,n,n,noise)) )
    end
    W[W.!=0].=1
    W = max.(W,W')
    # println(unique(W))
    #println("+ noise:\t\t"string(n)", "string(nnz(W))", "*string(nnz(W)/n))

    g = SimpleGraph(W);
    if !is_connected(g) println("the graph is NOT connected") end

    # T,INC,w = time_neighbor_neighbor_tensor(W)
    T,INC,w = triangle_tensor(W)
    println(size(INC))
    #println("n,nnz,nnz/n:\t\t"string(n)", "string(length(T.V))", "*string(length(T.V)/n))

    T, DH_isqrt = row_normalize(T, W)
    A, DG_isqrt = row_normalize(W)
    B = B_matrix(T)

    return A, DG_isqrt, T, DH_isqrt, B, INC', w
end


function compute_matrix_and_tensor_binary(X, kn;  noise = 0)
    n = size(X, 1)
    W = time_distance_matrix(X, kn, mode="connectivity")

    if noise > 0
        W = sparse(W + Int64.(sprand(Bool,n,n,noise)) )
    end
    #println("n,nnz,nnz/n:\t\t"string(n)", "string(nnz(W))", "*string(nnz(W)/n))
    W[W.!=0] .= 1 # = min.(W,1)
    W = max.(W,W')
    # println(unique(W))
    #println("+ noise:\t\t"string(n)", "string(nnz(W))", "*string(nnz(W)/n))

    g = SimpleGraph(W);
    if !is_connected(g) println("the graph is NOT connected") end

    # T,INC,w = time_neighbor_neighbor_tensor(W)
    T,INC,w = triangle_tensor(W)
    # println(unique(T.V))
    #println("n,nnz,nnz/n:\t\t"string(n)", "string(length(T.V))", "*string(length(T.V)/n))

    T, DH_isqrt = row_normalize(T, W)
    A, DG_isqrt = row_normalize(W)
    B = B_matrix(T)

    ### cols(INC)= no of edges
    return A, DG_isqrt, T, DH_isqrt, B, INC', w
end

function triangle_tensor(A::SparseArrays.SparseMatrixCSC{Float64,Int64}; weight_function = (x,y,z)-> 1)
    # INPUT A = adjacency matrix (knn matrix) weighted with distances (ie. sparsified distance matrix)

    n = size(A, 1)  # number of nodes
    d = vec(sum(A, dims=2))  # degree vector
    deg_order = zeros(Int64, n)
    deg_order[sortperm(d)] = 1:n

    Is = []
    Js = []
    Ks = []
    Vs = []

    # Incidence matrix and degree vector for Matlab (Matthias code)
    INC_I = []
    INC_J = []
    w = []
    c = 1 # counter for INC_J

    for i = 1:n
        N_i = findnz(A[:, i])[1]  # neighbors of node i
        # only look over pairs of neighbors with higher degree order
        N_i_keep = [j for j in N_i if deg_order[j] > deg_order[i]]
        for jj = 1:length(N_i_keep)
            for kk = (jj + 1):length(N_i_keep)
                j = N_i_keep[jj]
                k = N_i_keep[kk]
                # check for triangle
                if A[j, k] > 0
                    # triangle (i, j, k)
                    #push!(triangles, (i, j, k))
                    v = weight_function(A[i, j], A[j, k], A[k, i])
                    push!(Is, i,i,j,j,k,k)
                    push!(Js, j,k,i,k,i,j)
                    push!(Ks, k,j,k,i,j,i)
                    push!(Vs, v,v,v,v,v,v)

                    push!(INC_I, i,j,k)
                    push!(INC_J, c,c,c)
                    push!(w, v)
                    c += 1

                end
            end
        end
    end
    T = SuperSparse3Tensor(Is,Js,Ks,Vs,n)
    INC = sparse(INC_I,INC_J,1, n, length(w))
    return T, INC, w
end


#INCIDENCE FUNCTIONS

function quadruple_tensor(A::SparseArrays.SparseMatrixCSC{Float64,Int64}; weight_function = (x,y,z)-> 1)
    # INPUT A = adjacency matrix (knn matrix) weighted with distances (ie. sparsified distance matrix)
    w = []
    n = size(A, 1)  # number of nodes
    d = vec(sum(A, dims=2))  # degree vector
    deg_order = zeros(Int64, n)
    deg_order[sortperm(d)] = 1:n

    # Incidence matrix and degree vector for Matlab (Matthias code)
    INC_I = []
    INC_J = []
    w = []
    c = 1 # counter for INC_J

    for i = 1:n
        N_i = findnz(A[:, i])[1]  # neighbors of node i
        # only look over pairs of neighbors with higher degree order
        N_i_keep = [j for j in N_i if deg_order[j] > deg_order[i]]
        for jj = 1:length(N_i_keep)
            for kk = (jj + 1):length(N_i_keep)
                for hh = (kk+1):length(N_i_keep)
                    j = N_i_keep[jj]
                    k = N_i_keep[kk]
                    h = N_i_keep[hh]
                    # check for triangle
                    if A[j, k] > 0
                        # triangle (i, j, k)
                        #push!(triangles, (i, j, k)
                        v = weight_function(A[i, j], A[j, k], A[k, i])
                        push!(INC_I, i,j,k)
                        push!(INC_J, c,c,c)
                        c += 1
                        push!(w, v)
                        if A[k, h] > 0
                            push!(INC_I, i, j, k, h)
                            push!(INC_J, c, c, c, c)
                            v = weight_function(A[i, j], A[j, k], A[k, i]) + weight_function(A[i, k], A[k, h], A[h, i])
                            c += 1
                            push!(w, v)
                        end
                    end
                end
            end
        end
    end
    INC = sparse(INC_I,INC_J,1, n, length(w))
    return INC, w
end


function incid_mat(A, order, weight_function=(x,y,z)->1)
    I, J, V = findnz(A)
    INC_adj = spzeros(size(A)[1], length(V))
    W = rbf_similarity_weights(A, fast=false)
    w_adj = []
    for (ind, (i, j)) in enumerate(zip(I, J))
        INC_adj[[i, j], ind] .= 1
        push!(w_adj, W[i, j])
    end
    if order == 1
        return INC_adj, w_adj
    elseif order == 2
        _, INC, w = triangle_tensor(W, weight_function=weight_function)
        return hcat(INC_adj, INC), vcat(w_adj, w)
    elseif order == 3
        INC, w = quadruple_tensor(W, weight_function=weight_function)
        return hcat(INC_adj, INC), vcat(w_adj, w)
    end
end

function Kf_v2(K, D_n, W, x, φ, ψ)
    return D_n*(K * (W * ψ(K'*φ(D_n * x))))

end

# function D(K, w)
#     matr = K*w
#     return spdiagm(0 => vec(1.0 ./ sqrt.(sum(matr,dims=2)))),
#             spdiagm(0 => vec(1.0 ./ sum(matr,dims=1)))
# end


function network_to_INC(edge_list, node_number)
    INC = spzeros(node_number, length(edge_list))
    A = Int64[]
    for (i, edge_inds) in enumerate(edge_list)
        append!(A, repeat([i], length(edge_inds)))
        #INC[edge_inds, i] .= 1
    end
    vals = ones(length(vcat(edge_list...)))
    INC = sparse(vcat(edge_list...), A, vals )
    return INC
end


function T_to_INC_weights(T)
    B = spzeros(length(y), length(T.V))

    for (ind, (i, j, k, v)) in enumerate(zip(T.I, T.J, T.K, T.V))
        B[[i, j, k], ind] .= v
    end

    return B
end

function find_rows_not_in_graph(h_edges)
    return sort(unique(vcat(h_edges...)))
end
