using JLD
using PyPlot
using Base.Threads
using StatsBase
using LinearAlgebra

mutable struct GreenParameters
    nrows   :: Integer
    ncols   :: Integer
    dx      :: Real
    dy      :: Real
    hex     :: Real
    hey     :: Real
    Pot     :: Matrix{Float64}
    iη      :: ComplexF64
end


#-----------------------------#
# Initialize Green's function #
#-----------------------------#
function Green(Pot,height,width,effective_mass)
    ħ = 1.054571817E-34     # J.s
    q = 1.602176634E-19     # C
    mo = 9.1093837015E-31   # kg
    
    rows,cols = size(Pot)
    
    dx = width/cols
    dy = height/rows
    
    hex = ((ħ/dx^2)*(ħ/(effective_mass*mo)))/q
    hey = ((ħ/dy^2)*(ħ/(effective_mass*mo)))/q
    
    G = GreenParameters(rows,cols,dx,dy,hex,hey,Pot,1E-9*im)
    
    #println("Hopping energy x: ",1E3*hex," [meV]")
    #println("Hopping energy y: ",1E3*hey," [meV]")
    
    return G
end 

#-----------------------#
# Lead Green's function #
#-----------------------#
function leadGF(E,par::GreenParameters)
    π = 3.14159265358979323
    
    G = zeros(ComplexF64,par.nrows,par.nrows)
    U = zeros(ComplexF64,par.nrows,par.nrows)
    
    dL = 1.0/(par.nrows+1)
    Vx = -par.hex
    Vy = -par.hey
    q = 2*abs(Vx)
            
    for i in 1:par.nrows
        p = E + 2*(Vx+Vy) + 2*abs(Vy)*cos(Complex(π*i*dL)) + par.iη;
        G[i,i] = (2*p/q^2)*(1.0 - sqrt(Complex(1.0 - (q/p)^2)));
        
        for j in 1:par.nrows
            U[i,j] = sqrt(Complex(2*dL))*sin((j*pi*dL)*i);
        end
    end
        
    return U*G*U';    
end

#-------------#
# Hamiltonian #
#-------------#
function Hamiltonian(n,par::GreenParameters)
    P = par.Pot[:,n]
    v = -par.hey*ones(par.nrows - 1)
    H = Tridiagonal(v, 2*(par.hex + par.hey)*ones(par.nrows) + P, v)
    
    return H
end

#-----#
# SGM #
#-----#
function SGM(Energy, par::GreenParameters)
    D = DOS(Energy, par)
    S = zeros(par.nrows, par.ncols)
    w = 20
    δ = 100   
    
    for j in 1:par.ncols
        west = max(j-w,1)
        east = min(j+w,par.ncols)
        for i in 1:par.nrows
            up = max(i-w,1)
            dn = min(i+w,par.nrows)
            
            for h in west:east
            for v in up:dn
                γ = (δ^2)/((h-j)^2+(v-i)^2+δ^2)
                S[i,j] = S[i,j] + D[v,h]*γ
            end
            end
            S[i,j] /= (east-west+1)*(dn-up+1)
        end
    end
    
    return S
end 

#-------------------#
# Density of States #
#-------------------#
function DOS(Energy, par::GreenParameters)
    M = zeros(ComplexF64,par.nrows,par.nrows,par.ncols+1)
    D = zeros(par.nrows, par.ncols)
    V = -I(par.nrows)*par.hex

    Gr = leadGF(Energy, par)
    M[:,:,par.ncols+1] = Gr
        
    # Backward propagation
    Gnn = Gr
    for j in par.ncols:-1:1
        Σ = V*Gnn*V'
        H = Hamiltonian(j,par)
        Gnn = inv((Energy + par.iη)*I(par.nrows) - H - Σ)
        M[:,:,j] = Gnn
    end
   
    # Forward progpagation
    Gnn = Gr
    for j in 1:par.ncols
        # propagate
        Σ = V'*Gnn*V
        H = Hamiltonian(j,par)
        Gnn = inv((Energy + par.iη)*I(par.nrows) - H - Σ)
        
        #merge
        Σ = V*M[:,:,j+1]*V'
        G = (I(par.nrows) - Gnn*Σ) \ Gnn
        
        #DOS
        D[:,j] = -imag(diag(G))
    end
    
    return D
end         

#--------------#
# Transmission #
#--------------#
function Transmission(Energy, par::GreenParameters)
    V = -I(par.nrows)*par.hex
    
    # Begin with left lead
    G_lead = leadGF(Energy, par)
    G_nn = G_lead
    G_Ln = G_lead
    
    # Loop over columns
    for n in 1:par.ncols
        Σ = V'*G_nn*V
        H = Hamiltonian(n,par)
        G_nn = inv((Energy + par.iη)*I(par.nrows) - H - Σ)
        G_Ln = G_Ln*V*G_nn
    end
    
    # Add right lead
    Σ = V'*G_nn*V
    G_nn = (I(par.nrows)-G_lead*Σ) \ G_lead
    G = G_Ln*V*G_nn
    
    # Compute Γ matrix
    Σ = V'*G_lead*V
    Γ = im*(Σ - Σ')
    T = Γ*G*Γ*G'
    t = tr(real(T))
    
    return t
end


#======================================================#

#------------#
# Save Image #
#------------#
function saveImg(fname,data)
    nrows,ncols = size(data)
    #nrows=600
    #ncols=600
    ioff()
    fig = figure(frameon=false,figsize=(nrows+175,ncols+180),dpi=10)
    axis(false)
    imshow(data,aspect="auto",cmap="gray")
    savefig(fname,bbox_inches="tight",pad_inches=0,dpi=1)#,dpi=100)
    close(fig)
    
    return nothing
end

#------------#
# Load Image #
#------------#
function loadImg(fname,fi,fa)
    X = imread(fname)[:,:,1]
        
    mi = minimum(X)
    ma = maximum(X)
        
    return fi .+ (fa-fi)*(X .- mi)./(ma-mi)
end
   

#-----------------------#
# Obtain Instant Reward #
#-----------------------#
function getReward(s,so,Energy)
    L = 675E-9
    m = 0.041
    m = 1
        
    pars = Green(s,L,L,m)
    #Energy = 1E-5 + rand()*2.5E-3
    S = SGM(Energy,pars)
    #S = DOS(Energy,pars)
        
    return cor(S[:],so[:])[1,1]
end

# Auxiliary function
function θ(x)
    γ = 5

    if (x > 0)
        y = 1E-4*exp(-γ*x)
    else
        y = 1E-4
    end
    
    return y
end

#---------------------------#
#       LOSS FUNCTION       #
#---------------------------#
function episode!(so)
    i,j = size(so)      # Size of image
    NPop = 20           # Population size
    Energy = 2.5E-3     # Energy of Green's functions
    α = 0.95            # How fast approach winner
    h = 0               # Episode number
    rmax = 0            # Maximum reward
    kmax = 0            # Position of winner
    
    # Initial rewards are zero
    r = zeros(NPop)

    # Read or create population
    μ = 1E-4*randn(i,j,NPop)
    if (false)
        @load "test.dat" μ
    end
        
    # Begin loop until reach a correlation of 99%
    println("Beginning Simulation...")
    t1 = time_ns()
    fp = open("benchmark.txt","w")
    while(rmax < 0.72)
        h = h + 1
        
        # Get rewards for all agents
        @threads for k in 1:NPop
            r[k] = getReward(μ[:,:,k],so,Energy)
        end
        
        for k in 1:NPop
            println("   k: ",k,", reward: ",r[k])
        end
        
        # Find winner
        kmax = argmax(r)
        rmax = r[kmax]
        
        # Get closer to the winner + exploration
        @threads for k in 1:NPop
            μ[:,:,k] = α*μ[:,:,k] + (1-α)*μ[:,:,kmax] + θ(r[k])*randn(i,j)
        end
        
        # Print and save results
        println("episode ",h,": ",100*rmax)
        #saveImg("test.png",μ[:,:,kmax])
        #@save "test.dat" μ
        print(fp,string(h)*", "*string(100*rmax)*", "*string(((time_ns()-t1)/1E9)/60)*"\n")
    end        
    close(fp)

    pars = Green(μ[:,:,kmax],675E-9,675E-9,1)
    #Energy = 1E-5 + rand()*2.5E-3
    S = SGM(Energy,pars)
    imshow(S)
    show()
    
    return μ[:,:,kmax]
end

#==========================#
#           MAIN           #
#==========================#
X = imread("data/SGM/H1.png")[:,:,1]
Y = episode!(X)

