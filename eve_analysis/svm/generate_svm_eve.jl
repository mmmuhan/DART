# prepare data in the format required for SVM input (both ground-truth and DART inferred binarized promoter-state cases)

using Pkg
Pkg.activate(normpath(@__DIR__, "..", ".."))

using JLD2, NPZ, Random

# binarized promoter states from DART
load_copy(path) = ( @load path rn_met_dl; copy(rn_met_dl) )

# --- Base path ---
DART_DIR = joinpath("eve_analysis", "svm")

# --- Load binarized promoter states inferred from deep learning part ---
tel_met_dl   = load_copy(joinpath(DART_DIR, "ntel_dlbnb_compare_seed_1.jld2"))
perm_met_dl  = load_copy(joinpath(DART_DIR, "nperm_dlbnb_compare_seed_1.jld2"))


tel_ms = [];
for d in tel_met_dl
    push!(tel_ms, (even_time = d.binart, label = 1, acc = d.acc))
end

perm_ms = [];
for d in perm_met_dl
    push!(perm_ms, (even_time = d.binart, label = 2, acc = d.acc))
end

## save to file that can be used as the input for SVM

ml_train = shuffle(vcat(tel_ms,perm_ms)) # shuffle dataset ,perm2_ms

ls = [length(vcat(d.even_time[2]...)) for d in ml_train];
ml_train = copy(ml_train[findall(x->x>1000,ls)]);

xoff = [vcat(d.even_time[2]...)[1:1000] for d in ml_train]
labels = [Int(d.label) for d in ml_train];  # labels (model class)
xmat = hcat(xoff...);

#save
npzwrite(joinpath(DART_DIR, "ml_rev23_1000.npz"), Dict("xmat" => xmat, "label" => labels))