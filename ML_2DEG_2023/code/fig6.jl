using Images
using PyPlot
using Statistics

function loadImg(filename)
    img = convert(Array{Float64},Gray.(load(filename)))
    Lx,Ly = size(img)

    # Normalize
     ma = maximum(img)
     mi = minimum(img)

     img[img .≠ ma] .= (img[img .≠ ma] .- mi)/(ma - mi)
     img[img .== ma] .= 1.0

    return img
end


#loadImg("../Results_ES/test_"*string(i)*".png") # 215 x 213
#loadImg("../Results_CNN/Pr"*string(i)*".png") # 600 x 600
#loadImg("../Results_Yu/"*string(i-1)*".png") # 600 x 600

X = zeros(215,213,32)
for i in 1:32
    X[:,:,i] = loadImg("../Results_ES/test_"*string(i)*".png") # 600 x 600
end

Z = Array(1.0 ./ var(X,dims=3))

x = 1:215
y = 1:213
X = repeat(x,1,length(y))
Y = repeat(y',length(x),1)

fig = figure()
ax = fig.add_subplot(111,projection="3d")
surface = ax.plot_surface(X,Y,Z[:,:,1],cmap="viridis",linewidth=0,antialiased=true)
ax.view_init(elev=62,azim=-13)
show()


# plt = plot(surface(z=Y,x=x,y=x))
# display(plt)
# readline()

#imshow(Y)
#show()
