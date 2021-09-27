using Statistics
#run(`clear`)

function correlation(x,y)
    num = 0
    den1 = 0
    den2 = 0
    x_ = mean(x)
    y_ = mean(y)
    for (xi,yi) in zip(x,y)
        aux1 = (xi - x_)
        aux2 = (yi - y_)
        num += aux1*aux2
        den1 += aux1^2
        den2 += aux2^2
    end
    den = sqrt(den1) * sqrt(den2)
    c = num / den
    return c
 end

 function tuckey(lambda)
    p = LinRange(0.001,0.999,5000)
    Q = zeros(length(p))
    for (i,pi) in enumerate(p)
        if lambda == 0
            up = pi
            down = 1 - pi
            Q[i] = log(up/down)
        else
            up = pi^lambda - (1 - pi)^lambda
            down = lambda
            Q[i] = up/down
        end
    end
    return Q
end

function data_quantile(data)
    p = LinRange(0.001,0.999,5000)
    Q = zeros(length(p))
    for (i,pi) in enumerate(p)
        Q[i] = quantile(data,pi)
    end
    return Q
end

function find_lambda(var)
    Q_data = data_quantile(var)
    n = 200
    lambdas = [l for l in LinRange(-10,+10,n)]
    corr = zeros(n)
    for (i,lambda) in enumerate(lambdas)
        Q = tuckey(lambda)
        corr[i] = correlation(Q_data,Q)
    end
    if maximum(corr) < 0.90
        optimized_lambda = 5.0
    else
        optimized_lambda = lambdas[argmax(corr)]
    end
    return optimized_lambda
end

function import_data(filename)
    f = open(filename)
    ret = []
    for line in readlines(f)
        data = split(line,"\n")[1]
        append!(ret,parse(Float64,data))
    end
    return ret
end

#each file should have only one time series of logarithmic returns, single column
filenames = ["filename1.txt", "filename2.txt"]
lambda_SP = open("lambdas_portfolio.txt","w")
avg_lambda = zeros(length(filenames))
for (i,filename) in enumerate(filenames)
    r = import_data("DailyPortfolioReturns/"*filename)
    lambda = find_lambda(r)
    write(lambda_SP,filename[begin:end-4]*" "*string(lambda)*"\n")
    avg_lambda[i] = lambda
end
close(lambda_SP)
println(mean(avg_lambda))