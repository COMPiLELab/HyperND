function read_single(filename::String)
    ret = Int64[]
    open(filename) do f
        for line in eachline(f)
            push!(ret, parse(Int64, line))
        end
    end
    return ret
end

function read_node_map(filename::String)
    nm = Dict{Int64,Int64}()
    open(filename) do f
        for (i, line) in enumerate(eachline(f))
            if i == 1; continue; end
            new, orig = [parse(Int64, v) for v in split(line)]
            nm[orig] = new
        end
    end
    return nm
end

function read_labels(filename::String, nm, n::Int64)
    labels = zeros(Int64, n)
    label_map = Dict{AbstractString,Int64}()

    open(filename) do f
        for line in eachline(f)
            data = split(line)
            orig_id = parse(Int64, data[1])
            # Some nodes do not participate in any edge
            if !haskey(nm, orig_id); continue; end
            node = nm[orig_id]
            label = data[2]
            if !haskey(label_map, label)
                label_map[label] = length(label_map) + 1
            end
            labels[node] = label_map[label]
        end
    end

    return labels, label_map
end

function main()
    nverts = read_single("raw/contact-high-school-nverts.txt")
    simplices = read_single("raw/contact-high-school-simplices.txt")
    nm = read_node_map("raw/contact-high-school-nodemap.txt")
    labels, label_map = read_labels("raw/metadata_2013.txt", nm, maximum(simplices))

    hedges = Set{Set{Int64}}()

    let curr_ind = 0
	for nvert in nverts
    	    simplex = simplices[(curr_ind + 1):(curr_ind + nvert)]
            push!(hedges, Set{Int64}(simplex))
	    curr_ind += nvert
        end
    end

    open("hyperedges-contact-high-school.txt", "w") do f
        for hedge in hedges
            write(f, join(sort(collect(hedge)), ','))
            write(f, "\n")
        end
    end

    open("label-names-contact-high-school.txt", "w") do f
        output = sort([(v, k) for (k, v) in label_map])
        for (v, k) in output
            write(f, "$k\n")
        end
    end

    open("node-labels-contact-high-school.txt", "w") do f
        for l in labels
            write(f, "$l\n")
        end
    end
end
