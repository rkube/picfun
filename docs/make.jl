using Documenter, picfun
push!(LOAD_PATH, "../src")
makedocs(sitename="picfun",
         pages = [
             "Home" => "index.md",
             "Simulation setup" => "simsetup.md",
             "Code Reference" => "utilities.md"
         ])

deploydocs(; 
    repo="github.com/rkube/picfun")