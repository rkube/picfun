using Documenter, picfun
push!(LOAD_PATH, "../src")
makedocs(sitename="picfun",
         pages = [
             "Home" => "index.md",
             "Simulation setup" => "simsetup.md"
         ],
         html_prettyurls = false)

deploydocs(; 
    repo="github.com/rkube/picfun")