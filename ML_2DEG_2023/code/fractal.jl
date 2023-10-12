using Distributed
@everywhere using Images

# Power-law fitting
@everywhere function pfit(x,y)
    n = length(x)
    logx = log.(x)
    logy = log.(y)

    n1 = sum(logx .* logy)
    n2 = sum(logx)
    n3 = sum(logy)
    n4 = sum(logx .^ 2)

    b = (n*n1 - n2*n3)/(n*n4 - n2^2)

    return -b
end

# Load and image and normalize it
@everywhere function loadImg(filename)
    img = convert(Array{Float64},Gray.(load(filename)))
    Lx,Ly = size(img)

    # Normalize
    ma = maximum(img)
    mi = minimum(img)

    img[img .≠ ma] .= (img[img .≠ ma] .- mi)/(ma - mi)
    img[img .== ma] .= 1.0

    return img,Lx,Ly
end

# Measure for renormalization group
@everywhere function μ(matrix)
    #any_zeros = Int(any(x -> x == 1, matrix))
    any_zeros = min(1,sum(matrix))

    return any_zeros
end

# Count number of boxes
@everywhere function count(n, Lx, Ly, data)
    submatrices = [μ(data[i:min(i+n-1,Lx), j:min(j+n-1,Ly)]) for i in 1:n:size(data, 1), j in 1:n:size(data, 2)]
    return sum(submatrices)
end

# Calculate box dimension for file
@everywhere function process_file(filename)
    img,Lx,Ly = loadImg(filename)
    x = []
    y = []
    for r in [2^k for k in 1:5]
        c = count(r,Lx,Ly,img)
        if c > 0
            push!(x,r)
            push!(y,c)
        end
    end

    return pfit(x,y)
end

#--- Distributed Processing ---
# Add worker processes
addprocs(32)

filenames = []
for i in 1:32
    push!(filenames,"/scratch/cc3682/data/test_"*string(i)*".png")
end

# Spawn tasks
tasks = [@spawn process_file(filename) for filename in filenames]

# Collect results
results = fetch.(tasks)

# Wait for all tasks to complete
for task in tasks
    wait(task)
end

# Print results
open("batch.out","w") do file
    for (n,result) in enumerate(results)
        println(file,"Result of task $n: $result")
    end
end
