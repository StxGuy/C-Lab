using JLD
using CUDA
using PyPlot
using Statistics

#---------------------------------------------------------#
#                      FEED FORWARD                       #
#---------------------------------------------------------#

# Feedforward kernel
function feed5!(A,B,X,U,dX)               
    j = (blockIdx().x-1)*blockDim().x + threadIdx().x
    i = (blockIdx().y-1)*blockDim().y + threadIdx().y
    
    # Fixed kernel size
    if (i >= 3 && i <= 602 && j >= 3 && j <= 602)
        ξ = 1.0
        for l in 1:5
            vj = j + l - 3
            for k in 1:5
                vi = i + k - 3
                @inbounds ξ += A[i,j,k,l]*tanh(X[vi,vj]) + B[i,j,k,l]*U[vi,vj]
            end
        end
        
        dX[i,j] = ξ 
    end
    
    return nothing
end

function feed7!(A,B,X,U,dX)               
    j = (blockIdx().x-1)*blockDim().x + threadIdx().x
    i = (blockIdx().y-1)*blockDim().y + threadIdx().y
    
    # Fixed kernel size
    if (i >= 4 && i <= 603 && j >= 4 && j <= 603)
        ξ = 1.0
        for l in 1:7
            vj = j + l - 4
            for k in 1:7
                vi = i + k - 4
                @inbounds ξ += A[i,j,k,l]*tanh(X[vi,vj]) + B[i,j,k,l]*U[vi,vj]
            end
        end
        
        dX[i,j] = ξ 
    end
    
    return nothing
end

function feed9!(A,B,X,U,dX)               
    j = (blockIdx().x-1)*blockDim().x + threadIdx().x
    i = (blockIdx().y-1)*blockDim().y + threadIdx().y
    
    # Fixed kernel size
    if (i >= 5 && i <= 604 && j >= 5 && j <= 604)
        ξ = 1.0
        for l in 1:9
            vj = j + l - 5
            for k in 1:9
                vi = i + k - 5
                @inbounds ξ += A[i,j,k,l]*tanh(X[vi,vj]) + B[i,j,k,l]*U[vi,vj]
            end
        end
        
        dX[i,j] = ξ 
    end
    
    return nothing
end

function feed21!(A,B,X,U,dX)               
    j = (blockIdx().x-1)*blockDim().x + threadIdx().x
    i = (blockIdx().y-1)*blockDim().y + threadIdx().y
    
    # Fixed kernel size
    if (i >= 11 && i <= 610 && j >= 11 && j <= 610)
        ξ = 1.0
        for l in 1:21
            vj = j + l - 11
            for k in 1:21
                vi = i + k - 11
                @inbounds ξ += A[i,j,k,l]*tanh(X[vi,vj]) + B[i,j,k,l]*U[vi,vj]
            end
        end
        
        dX[i,j] = ξ 
    end
    
    return nothing
end

# Feed forward
function feedforward(X,U,A,B,F)
    y,x = size(U)
    dX = CUDA.zeros(y,x)
    Δ = 0.01
    
    er = 100
    while(er > 0.01)
        if (F == 5)
            CUDA.@sync begin
                @cuda threads=(30,30) blocks=(21,21) feed5!(A,B,X,U,dX)
            end
        elseif (F == 7)
            CUDA.@sync begin
                @cuda threads=(30,30) blocks=(21,21) feed7!(A,B,X,U,dX)
            end
        elseif (F == 9)
            CUDA.@sync begin
                @cuda threads=(30,30) blocks=(21,21) feed9!(A,B,X,U,dX)
            end
        else
            CUDA.@sync begin
                @cuda threads=(30,30) blocks=(21,21) feed21!(A,B,X,U,dX)
            end
        end
        
        er = sum(map(abs,dX-X))/(x*y)
        X = X.*(1-Δ) + dX.*Δ
    end
    
    return X
end

#---------------------------------------------------------#
#                   BACK PROPAGATION                      #
#---------------------------------------------------------#

# Backpropagation kernel
function back5!(D,X,U,dA,dB)
    j = (blockIdx().x-1)*blockDim().x + threadIdx().x
    i = (blockIdx().y-1)*blockDim().y + threadIdx().y
        
    if (i >= 3 && i <= 602 && j >= 3 && j <= 602)
        for l in 1:5
            vj = j + l - 3
            for k in 1:5
                vi = i + k - 3
                @inbounds dA[i,j,k,l] = D[i,j]*tanh(X[vi,vj])
                @inbounds dB[i,j,k,l] = D[i,j]*U[vi,vj]
            end
        end
    end
     
    return nothing
end

function back7!(D,X,U,dA,dB)
    j = (blockIdx().x-1)*blockDim().x + threadIdx().x
    i = (blockIdx().y-1)*blockDim().y + threadIdx().y
        
    if (i >= 4 && i <= 603 && j >= 4 && j <= 603)
        for l in 1:7
            vj = j + l - 4
            for k in 1:7
                vi = i + k - 4
                @inbounds dA[i,j,k,l] = D[i,j]*tanh(X[vi,vj])
                @inbounds dB[i,j,k,l] = D[i,j]*U[vi,vj]
            end
        end
    end
     
    return nothing
end

function back9!(D,X,U,dA,dB)
    j = (blockIdx().x-1)*blockDim().x + threadIdx().x
    i = (blockIdx().y-1)*blockDim().y + threadIdx().y
        
    if (i >= 5 && i <= 604 && j >= 5 && j <= 604)
        for l in 1:9
            vj = j + l - 5
            for k in 1:9
                vi = i + k - 5
                @inbounds dA[i,j,k,l] = D[i,j]*tanh(X[vi,vj])
                @inbounds dB[i,j,k,l] = D[i,j]*U[vi,vj]
            end
        end
    end
     
    return nothing
end

function back21!(D,X,U,dA,dB)
    j = (blockIdx().x-1)*blockDim().x + threadIdx().x
    i = (blockIdx().y-1)*blockDim().y + threadIdx().y
        
    if (i >= 11 && i <= 610 && j >= 11 && j <= 610)
        for l in 1:21
            vj = j + l - 11
            for k in 1:21
                vi = i + k - 11
                @inbounds dA[i,j,k,l] = D[i,j]*tanh(X[vi,vj])
                @inbounds dB[i,j,k,l] = D[i,j]*U[vi,vj]
            end
        end
    end
     
    return nothing
end

function backprop!(X,U,P,A,B,F)
    D = dloss(X,P)
    
    x,y,z,t = size(A)    
    dA = CUDA.zeros(x,y,z,t)
    dB = CUDA.zeros(x,y,z,t)
    
    if (F == 5)
        CUDA.@sync begin
            @cuda threads=(30,30) blocks=(21,21) back5!(D,X,U,dA,dB)
        end
    elseif (F == 7)
        CUDA.@sync begin
            @cuda threads=(30,30) blocks=(21,21) back7!(D,X,U,dA,dB)
        end
    elseif (F == 9)
        CUDA.@sync begin
            @cuda threads=(30,30) blocks=(21,21) back9!(D,X,U,dA,dB)
        end
    else
        CUDA.@sync begin
            @cuda threads=(30,30) blocks=(21,21) back21!(D,X,U,dA,dB)
        end
    end

    return dA,dB
end 

#---------------------------------------------------------#
#                          Loss                           #
#---------------------------------------------------------#
function loss(Ŷ,Y)
    uŶ = mean(Ŷ)
    uY = mean(Y)
    σŶ = std(Ŷ)
    σY = std(Y)
    
    r = sum((Ŷ.-uŶ).*(Y.-uY))/(σŶ*σY*(length(Y)-1))
    
    return r
end

function dloss(Ŷ,Y)
    uŶ = mean(Ŷ)
    uY = mean(Y)
    σŶ = std(Ŷ)
    σY = std(Y)
    
    η = 0.001
    
    r = loss(Ŷ,Y)
    
    drdX = ((Y.-uY) - (Ŷ.-uŶ).*(((σY/σŶ)*r)))./(σŶ*σY)
                
    return drdX
end

#---------------------------------------------------------#
#             Load a PNG image and zero pad it            #
#---------------------------------------------------------#
function load(fname, m, F, cond)
    w = F >> 1

    # Read
    X_host = imread(fname)
    X_host = X_host[:,:,1]
    xi = minimum(X_host)
    xa = maximum(X_host)
    
    # Normalization
    Y_host = zeros(600+F-1,600+F-1)
    Y_host[(w+1):(w+600),(w+1):(w+600)] = ((X_host.-xi)./(xa-xi).-0.5).*2
    
    #*** Image augmentation ***
    #- Flip horizonal
    if (m == 2 || m == 5 || m == 7 || m == 8)
        for i in 1:((600+F)>>1)
            t = Y_host[:,i]
            Y_host[:,i] = Y_host[:,(602+w)-i]
        end
    end
    #- Flip vertical
    if (m == 3 || m == 6 || m == 7 || m == 8)
        for i in 1:((600+F)>>1)
            t = Y_host[i,:]
            Y_host[i,:] = Y_host[(602+w)-i,:]
        end
    end
    #- Salt & Pepper noise
    if ((m == 4 || m == 5 || m == 6 || m == 8) && cond == true)
        R = rand(600+F-1,600+F-1).*1E-2
        Y_host += R
    end        
       
    # Transfer to device
    X_dev = CUDA.zeros(600+F-1,600+F-1)
    copyto!(X_dev,Y_host)
    
    return X_dev
end 

#---------------------------------------------------------#
#                PRINTing and PLOTing                     #
#---------------------------------------------------------#
function tshow(D_dev,F)
    w = F >> 1
    X = zeros(600,600)
    
    figure()
    copyto!(X,D_dev[(1+w):(w+600),(w+1):(w+600)])
    imshow(X)

    show()

    return nothing
end

# Save image
function saveImg(fname,P_d,F)
    w = F >> 1
    P = zeros(600+F-1,600+F-1)
    copyto!(P,P_d)
    
    ioff()
    fig = figure(frameon=false,figsize=(600+175,600+180),dpi=10)
    axis(false)
    imshow(P[(1+w):(w+600),(w+1):(w+600)],aspect="auto",cmap="gray")
    savefig(fname,bbox_inches="tight",pad_inches=0,dpi=1)
    close(fig)
end

#---------------------------------------------------------#
#                       TRAINING                          #
#---------------------------------------------------------#
function train(F)
    w = F >> 1
    η = 1E-2
    η_max = 1E-5
    η_min = 1E-8
    Nepoch = 14
    TrainingLength = 50
    β1 = 0.9
    β2 = 0.999
    ε = 1E-8    
    
    # Load filters
    A = zeros(600+F-1,600+F-1,F,F)
    #A = load("FilterA.jld")["data"]
    #copyto!(A_dev,A)
    #A = load("FilterB.jld")["data"]
    #copyto!(B_dev,B)    
    
    A_dev = (CUDA.rand(600+F-1,600+F-1,F,F).-0.5)./100
    B_dev = (CUDA.rand(600+F-1,600+F-1,F,F).-0.5)./100
    X_dev = CUDA.zeros(600+F-1,600+F-1)
    
    mA = CUDA.zeros(600+F-1,600+F-1,F,F)
    mB = CUDA.zeros(600+F-1,600+F-1,F,F)
    vA = CUDA.zeros(600+F-1,600+F-1,F,F)
    vB = CUDA.zeros(600+F-1,600+F-1,F,F)
    
    println("** TRAINING **")
    
    L = []
    for l in 1:TrainingLength
        # Training Schedule
        if l < 10
            η = l*η_max
        else
            η = η_min + 0.5*(η_max-η_min)*(1+cos(π*(l-10)/(TrainingLength-10)))
        end           
    
        # Epoch
        ddA = CUDA.zeros(600+F-1,600+F-1,F,F)
        ddB = CUDA.zeros(600+F-1,600+F-1,F,F)
        lx = 0.0
        for k in 1:Nepoch
            # Load image pairs
            i = rand(1:49)
            method = rand(8)
            
            U_dev = load("SGM/S"*string(i)*".png",method,F,true)
            P_dev = load("POT/P"*string(i)*".png",method,F,false)

            # Training itself
            X_dev = feedforward(X_dev,U_dev,A_dev,B_dev,F)
            dA,dB = backprop!(X_dev,U_dev,P_dev,A_dev,B_dev,F)
            ddA += dA
            ddB += dB
            
            # Loss
            lx += loss(P_dev[(w+1):(w+600),(w+1):(w+600)],X_dev[(w+1):(w+600),(w+1):(w+600)])
        end
        lx /= Nepoch
        
        # Gradient
        A_dev += η.*ddA
        B_dev += η.*ddB
        #=mA = β1 .* mA + (1-β1).*ddA
        vA = β2 .* vA + (1-β2).*ddA.*ddA
        
        mA = mA./(1-β1^l)
        vA = vA./(1-β2^l)
        
        A_dev += η.*mA./(map(sqrt,vA).+ε)
        
        
        mB = β1 .* mB + (1-β1).*ddB
        vB = β2 .* vB + (1-β2).*ddB.*ddB
        
        mB = mB./(1-β1^l)
        vB = vB./(1-β2^l)
        
        =#B_dev += η.*mB./(map(sqrt,vB).+ε)
        
        
        # Loss
        push!(L,lx)
        println(l,": ",lx)
    end
    
    plot(L)
    show()
    
    for i in 20:32
        U_dev = load("EXP/F"*string(i)*".png",1,F,false)
        X_dev = feedforward(X_dev,U_dev,A_dev,B_dev,F)
        saveImg("Pr"*string(i)*".png",X_dev,F)
    end
    
    # Save filters
    #copyto!(A,A_dev)
    #save("FilterA.jld","data",A)
    #copyto!(A,B_dev)
    #save("FilterB.jld","data",A)

    return X_dev
end

#---------------------------------------------------------#
#                          MAIN                           #
#---------------------------------------------------------#
X_dev = train(7)


