# prepare data in the format required for SVM input (both ground-truth and DART inferred binarized promoter-state cases)

using Pkg
Pkg.activate(normpath(@__DIR__, ".."))

using JLD2, NPZ, Random

# ground truth

TRUE_DIR = joinpath("synthetic_data", "svm", "true")
# --- Load ---
@load joinpath(TRUE_DIR, "tel_times.jld2")   tel_times
@load joinpath(TRUE_DIR, "perm_times.jld2")  perm_times
@load joinpath(TRUE_DIR, "perm1_times.jld2") perm1_times
@load joinpath(TRUE_DIR, "perm2_times.jld2") perm2_times

tel_true = [];
for d in tel_times
    push!(tel_true, (even_time = d, label = 1))
end

perm_true = [];
for d in perm_times
    push!(perm_true, (even_time = d, label = 2))
end

perm1_true = [];
for d in perm1_times
    push!(perm1_true, (even_time = d, label = 3))
end

perm2_true = [];
for d in perm1_times
    push!(perm1_true, (even_time = d, label = 4))
end

# save to file that can be used as the input for SVM
ml_train = shuffle(vcat(tel_true,perm_true,perm1_true,perm2_true)) # shuffle dataset

ls = [length(vcat(d.even_time[2]...)) for d in ml_train];
ml_train = copy(ml_train[findall(x->x>1000,ls)]);

xoff = [vcat(d.even_time[2]...)[1:1000] for d in ml_train] # off times
labels = [Int(d.label) for d in ml_train];  # labels (model class)
xmat = hcat(xoff...);

# save
npzwrite(joinpath(TRUE_DIR, "true_rev2345.npz"), Dict("xmat" => xmat, "label" => labels))


# binarized promoter states from DART
load_copy(path) = ( @load path rn_met_dl; copy(rn_met_dl) )

# --- Base path ---
DART_DIR = joinpath("synthetic_data", "svm", "dart")

# --- Load binarized promoter states inferred from deep learning part ---
tel_met_dl   = load_copy(joinpath(DART_DIR, "ntel_dlnb1_compare_seed_1.jld2"))
perm_met_dl  = load_copy(joinpath(DART_DIR, "nperm_dlnb1_compare_seed_1.jld2"))
perm1_met_dl = load_copy(joinpath(DART_DIR, "nperm1_dlnb1_compare_seed_1.jld2"))
perm2_met_dl = load_copy(joinpath(DART_DIR, "nperm2_dlnb1_compare_seed_1.jld2"))


tel_ms = [];
for (i,d) in enumerate(tel_met_dl)
    push!(tel_ms, (even_time = d.binart, label = 1, acc = d.acc, id = i))
end

perm_ms = [];
for (i,d) in enumerate(perm_met_dl)
    push!(perm_ms, (even_time = d.binart, label = 2, acc = d.acc, id = i))
end

perm1_ms = [];
for (i,d) in enumerate(perm1_met_dl)
    push!(perm1_ms, (even_time = d.binart, label = 3, acc = d.acc, id = i))
end

perm2_ms = [];
for (i,d) in enumerate(perm2_met_dl)
    push!(perm2_ms, (even_time = d.binart, label = 4, acc = d.acc, id = i))
end


## save to file that can be used as the input for SVM

ml_train = shuffle(vcat(tel_ms,perm_ms,perm1_ms,perm2_ms)) # shuffle dataset

ls = [length(vcat(d.even_time[2]...)) for d in ml_train];
ml_train = copy(ml_train[findall(x->x>1000,ls)]);

xoff = [vcat(d.even_time[2]...)[1:1000] for d in ml_train] # off times
labels = [Int(d.label) for d in ml_train];  # labels (model class)
xmat = hcat(xoff...);

#save
npzwrite(joinpath(DART_DIR, "ml_rev2345.npz"), Dict("xmat" => xmat, "label" => labels))