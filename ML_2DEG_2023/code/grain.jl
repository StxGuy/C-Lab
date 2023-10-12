using Distributed

@everywhere using Images
@everywhere using Statistics

# Dilate or erode an image
@everywhere function dilateErode!(N,M,s,f)
    Lx,Ly = size(M)

    @inbounds for j in 1:Ly
        yi = max(j-s÷2,1)
        ya = min(j+s÷2,Ly)
        for i in 1:Lx
            xi = max(i-s÷2,1)
            xa = min(i+s÷2,Lx)

            if f
                N[i,j] = maximum(@view M[xi:xa,yi:ya])
            else
                N[i,j] = minimum(@view M[xi:xa,yi:ya])
            end
        end
    end
end

# Opening operation on an image
# Source: M
# Target: N
# Kernel size: s
@everywhere function opening!(N,M,s)
    K = similar(M)
    dilateErode!(K,M,s,false)
    dilateErode!(N,K,s,true)
end

# Load images
@everywhere function loadImg(filename)
    img = convert(Array{Float64},Gray.(load(filename)))
    Lx,Ly = size(img)

    # Normalize
    ma = maximum(img)
    mi = minimum(img)

    img[img .≠ ma] .= (img[img .≠ ma] .- mi)/(ma - mi)
    img[img .== ma] .= 1.0

    return img
end

# Compute granulometry
# v: maximum kernel size
@everywhere function granulometry(filename,v)
    img = loadImg(filename)
    Lx,Ly = size(img)

    granu = Vector{Float64}(undef,v+1)
    N = similar(img)

    for k in 1:2:(v+1)
        opening!(N,img,k)
        granu[k] = mean(N)  # <- Granulometry function = cardinality of the opening
    end

    return -100*diff(granu)
end

#---- Distributed Processing ----#
addprocs(6)

filenames = []
for i in 1:6
    push!(filenames,"../Results_ES/test_"*string(i)*".png")
end

# Spawn tasks
tasks = [@spawn granulometry(filename,50) for filename in filenames]

# Collect results
results = fetch.(tasks)

# Wait for all tasks to complete
for task in tasks
    wait(task)
end

# Save results
open("results.out","w") do file
    for (n,result) in enumerate(results)
        print(file,"x"*string(n)*"=[")
        for g in result
            print(file,g,",")
        end
        println(file,"]")
        println(file,"")
    end
end

