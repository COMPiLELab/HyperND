using Pkg

dependencies = [
    "Distances",
    "LightGraphs",
    "PyCall",
    "ScikitLearn",
    "DataFrames",
    "CSV",
    "MLDatasets",
    "MAT",
    "DataStructures",
    "StatsBase",
    "InvertedIndices",
    "Suppressor",
    "BSON"
]

Pkg.add(dependencies)
Pkg.add(PackageSpec(url="https://github.com/JackDunnNZ/UCIData.jl"))
