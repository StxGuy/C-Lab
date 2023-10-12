using Images
using StatsBase
using PyPlot

T = [
3.15414778668526,
3.07668177065179,
2.89125130707564,
2.68447194144301,
2.54356918787034,
2.27518299058906,
2.13672011153712,
2.0257058208435,
1.94701986754967,
1.81221680027884,
1.66643429766469,
1.61702683861973,
1.59872777971419,
1.34681073544789,
1.10526315789474,
1.03938654583479,
1.01132798884629,
0.927152317880795,
0.861885674451028,
0.790519344719414,
0.713663297316138,
0.700243987452074,
0.568490763332172,
0.497124433600558,
0.484925060996863,
0.397089578250262,
0.303154409201812,
0.2757058208435,
0.220808644126874,
0.166521436040432,
0.102474729871035,
0.019518996165912]

function Roughness(filename)
    img = load(filename)
    gray_img = Gray.(img)
    M = [x.val for x in gray_img]
        
    μ = mean(M)
    R = sqrt(mean((M .- μ).^2))
    
    return R
end
    
function ImgEntropy(filename)
    img = load(filename)
    gray_img = Gray.(img)
    vals = vec([x.val for x in gray_img])
        histogram = fit(Histogram,vals,nbins=256)
        probabilities = histogram.weights/sum(histogram.weights)
        S = -sum(probabilities .* log2.(probabilities .+ 1E-10))
        
        return S
end

function Box(filename,q)
    # Load image and convert it to grayscale
    img = load(filename)
    gray_img = Gray.(img)

    # Find spatial and color limits
    Lx,Ly = size(gray_img)
    Ci = minimum(gray_img)
    Ca = maximum(gray_img)
    Lc = Ca - Ci

    # Estimate minimum ϵ
    ϵᵢ = maximum([1.0/Lx, 1.0/Ly])

    # Main loop
    fⱼ = []
    αⱼ = []
    x = []
    for it in 1:5
        ϵ = ϵᵢ*it

        # Find infinitesimal displacements
        ϵx = Int(round(Lx*ϵ))
        ϵy = Int(round(Ly*ϵ))
        ϵc = Lc*ϵ

        # Initial conditions
        xᵢ = 1
        yᵢ = 1
        cᵢ = Ci
        stop = false
        mass = []
        Mϵ = 0

        # Raster image
        while(stop == false)
            xₐ = xᵢ + ϵx - 1
            yₐ = yᵢ + ϵy - 1
            cₐ = cᵢ + ϵc

            # Mass mᵢ in box i, and total mass Mϵ
            # Remove zero measures.
            mᵢ = sum(cₐ .≥ gray_img[xᵢ:xₐ,yᵢ:yₐ] .> cᵢ)
            if (mᵢ > 0)
                push!(mass,mᵢ)
                Mϵ += mᵢ
            end

            # Boundary conditions
            xᵢ += ϵx
            if (xᵢ + ϵx > Lx)
                xᵢ = 1
                yᵢ += ϵy
                if (yᵢ + ϵy > Ly)
                    yᵢ = 1
                    cᵢ += ϵc
                    if (cᵢ + ϵc > Ca)
                        stop = true
                    end
                end
            end
        end

        # Find probabilities and canonical measure
        Pᵢ = mass/Mϵ
        Pᵢq = Pᵢ.^q
        μᵢ = Pᵢq/sum(Pᵢq)

        # Find spectra
        push!(fⱼ, sum(μᵢ.*log.(μᵢ)))
        push!(αⱼ, sum(μᵢ.*log.(Pᵢ)))
        push!(x, log(ϵ))
    end

    v = var(x)
    f = cov(x,fⱼ)/v
    α = cov(x,αⱼ)/v

    return α,f
end

function plotBox(filename)
    ϵ = 1E-2

    q_space = LinRange(-10,10,30)

    α = []
    F = []
    for q in q_space
        a,f = Box(filename,q)
        push!(α,a)
        push!(F,f)
    end

    plot(α,F,"o-")
end


function plotRoughness()
    R1 = []
    for i in 1:32
        filename = "../Results_Yu/"*string(i-1)*".png"
        R = Roughness(filename)
        push!(R1,R)
    end

    R2 = []
    for i in 1:32
        filename = "../Results_CNN/Pr"*string(i)*".png"
        R = Roughness(filename)
        push!(R2,R)
    end

    R3 = []
    for i in 1:32
        filename = "../Results_ES/test_"*string(i)*".png"
        R = Roughness(filename)
        push!(R3,R)
    end

    plot(T,R1,"o")
    plot(T,R2,"s")
    plot(T,R3,"*")
    ylabel("RMS Roughness")
    xlabel("Transmission")
    legend(["Pix2Pix","CNN","ES"])
    show()
end

plotBox("../Results_CNN/Pr2.png")
plotBox("../Results_CNN/Pr10.png")
plotBox("../Results_CNN/Pr20.png")
plotBox("../Results_CNN/Pr30.png")
xlabel("α")
ylabel("f(α)")
show()
