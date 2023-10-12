using Images
using StatsBase

function ImgEntropy(filename)
    img = load(filename)
    gray_img = Gray.(img)
    vals = vec([x.val for x in gray_img])
    histogram = fit(Histogram,vals,nbins=256)
    probabilities = histogram.weights/sum(histogram.weights)
    S = -sum(probabilities .* log2.(probabilities .+ 1E-10))
    
    return S
end

function computeEntropies()
    Sav = 0
    Smi = 1E10
    Sma = -1E10
    
    for i in 1:32
        filename = "../Results_Yu/"*string(i-1)*".png"
        
        S = ImgEntropy(filename)
        if (S < Smi)
            Smi = S
        end
        if (S > Sma)
            Sma = S
        end
        Sav += S
    end
    
    return Sav,Smi,Sma
end

Sav,Smin,Smax = computeEntropies()

println("Average entropy: ",Sav/32)
println("Minimum entropy: ",Smin)
println("Maximum entropy: ",Smax)
    

    


