using Combinatorics, Graphs
 using StatsBase, Random, Distributions, LsqFit
using DelimitedFiles, GraphIO
using Memoization
using ProgressBars

println(Threads.nthreads())

function loadEdges(file)
    return readdlm(file,',',Int64,'\n')
end

@memoize function splitEdges(n)
    return [loadEdges("./graphs/square/"*string(n)*"x"*string(n)*"/1.el")#Load strain 1 edges
    ,loadEdges("./graphs/square/"*string(n)*"x"*string(n)*"/2.el")]#Load strain 2 edges
end

@memoize function underlyingGraph(n)#Calculate the 'total' graph, where all possible edges are present
    edges=splitEdges(n)
    g = SimpleGraph(UInt32(n^2))
    for i in 1:div(length(edges[1]),2)
        add_edge!(g,edges[1][i,1],edges[1][i,2])
    end
    for i in 1:div(length(edges[2]),2)
        add_edge!(g,edges[2][i,1],edges[2][i,2])
    end
    return g
end

function yN(g,dists,ug)
    comps=connected_components(g)
    m=maximum(length.(comps))
    filter!(x->length(x)!=m,comps)
    if length(comps)==0
        return 0
    end
    v=comps[1][1]
    dists.=spfa_shortest_paths(ug,v)
    lbl=zeros(length(dists))
    ids=Graphs.connected_components!(lbl,g)
    yes=[0 for _=1:maximum(dists)]
    no=[0 for _=1:maximum(dists)]
    for i=eachindex(dists)
        if dists[i]!=0
            if ids[i]==v
                yes[dists[i]]+=1
            else
                no[dists[i]]+=1
            end
        end
    end
    return [yes,no]
end



function findCorrLength(n,β,μ,k,edges,distss,ug,tmpgs)
    xs=1:maximum(distss[1])
    yes=[Threads.Atomic{Int64}(0) for _=xs]
    no=[Threads.Atomic{Int64}(0) for _=xs]
    (dist1,dist2)=(Bernoulli(1-1/(1+exp(β*(μ)))),Bernoulli(1-1/(1+exp(β*(μ-1)))))#The Bernoulli distributions that will be used to see if a given edge should be present in a typical graph
    Threads.@threads for _=1:k
        for i in 1:div(length(edges[1]),2)
            if rand(dist1)
                if !has_edge(tmpgs[Threads.threadid()],edges[1][i,1],edges[1][i,2])
                    add_edge!(tmpgs[Threads.threadid()],edges[1][i,1],edges[1][i,2])#Add strain 1 edge if applicable
                end
            elseif has_edge(tmpgs[Threads.threadid()],edges[1][i,1],edges[1][i,2])
                    rem_edge!(tmpgs[Threads.threadid()],edges[1][i,1],edges[1][i,2])#Remove strain 1 edge if applicable
            end
        end
        for i in 1:div(length(edges[2]),2)
            if rand(dist2)
                if !has_edge(tmpgs[Threads.threadid()],edges[2][i,1],edges[2][i,2])
                    add_edge!(tmpgs[Threads.threadid()],edges[2][i,1],edges[2][i,2])#Add strain 2 edge if applicable
                end
            elseif has_edge(tmpgs[Threads.threadid()],edges[2][i,1],edges[2][i,2])
                    rem_edge!(tmpgs[Threads.threadid()],edges[2][i,1],edges[2][i,2])#Remove strain 2 edge if applicable
            end
        end
        
        yn=yN(tmpgs[Threads.threadid()],distss[Threads.threadid()],ug)
        if yn!=0
            for i=xs
                Threads.atomic_add!(yes[i],yn[1][i])
                Threads.atomic_add!(no[i],yn[2][i])
            end
        end
    end
    yes=vcat([1],[yes[i][] for i=eachindex(yes)])
    no=vcat([0],[no[i][] for i=eachindex(no)])
    ys=yes./(yes.+no)
    map!(x->(if isnan(x) 0. else x end),ys,ys)
    return curve_fit((r, p)->exp.(-r/p[1]), vcat([0],xs), ys, [1.]).param[1]
end


const pointfile=string(ARGS[1])
const n=parse(Int,ARGS[2])
const k=parse(Int,ARGS[3])

const edges=splitEdges(n)
const ug=underlyingGraph(n)
distss = [spfa_shortest_paths(ug,1) for _=1:Threads.nthreads()]
tmpgs=[SimpleGraph(UInt32(n^2)) for _=1:Threads.nthreads()]#Initialise empty graph with the correct amount of vertices

const points=readdlm(pointfile,',',Float64,'\n')

out =zeros(div(length(points),2))



t=@time for i in ProgressBar(1:div(length(points),2), printing_delay=1)
    out[i]=findCorrLength(n,points[i,1],points[i,2],k,edges,distss,ug,tmpgs)
end

open("out_"*string(n)*"_"*string(k)*"_"*pointfile,"w") do io
    writedlm(io,out,',')
end

println(t)


#Summary courtesy of github autopilot:
#This script is for finding the correlation lengths of the square lattice for a given set of points, where the points are of the form [β,μ].
#The script takes 3 arguments, the name of the file containing the points, the size of the lattice, and the number of graphs to use in the fitting.
#The output is a .csv file containing the correlation lengths for each of the points in the input file.
#The output file is named as out_n_k_pointfile.csv, where n is the size of the lattice, k is the number of graphs used in the fitting, and pointfile is the name of the input file.

#The script uses the following packages:
#Combinatorics, Graphs, StatsBase, Random, Distributions, LsqFit, DelimitedFiles, GraphIO, Memoization, ProgressBars

#The script should be run using julia squareCorrLengths.jl pointfile n k.
#The script can be run in parallel using julia -t m squareCorrLengths.jl pointfile n k, where m is the number of threads to use.



