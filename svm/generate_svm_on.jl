# prepare data in the format required for SVM input (both ground-truth and DART inferred binarized promoter-state cases)

using Pkg
Pkg.activate(normpath(@__DIR__, ".."))

using JLD2, NPZ, Random

# ground truth

TRUE_DIR = joinpath("synthetic_data", "svmon")
# --- Load ---
@load joinpath(TRUE_DIR, "tel_times.jld2")   tel_times
@load joinpath(TRUE_DIR, "on2_times.jld2")  on2_times
@load joinpath(TRUE_DIR, "on3_times.jld2") on3_times
@load joinpath(TRUE_DIR, "on4_times.jld2") on4_times

tel_true = [];
for d in tel_times
    push!(tel_true, (even_time = d, label = 1))
end

on2_true = [];
for d in on2_times
    push!(on2_true, (even_time = d, label = 2))
end

on3_true = [];
for d in on3_times
    push!(on3_true, (even_time = d, label = 3))
end

on4_true = [];
for d in on4_times
    push!(on4_true, (even_time = d, label = 4))
end

# save to file that can be used as the input for SVM
ml_train = shuffle(vcat(tel_true,on2_true,on3_true,on4_true)) # shuffle dataset

ls = [length(vcat(d.even_time[1]...)) for d in ml_train];
ml_train = copy(ml_train[findall(x->x>1000,ls)]);

xoff = [vcat(d.even_time[1]...)[1:1000] for d in ml_train] # on times
labels = [Int(d.label) for d in ml_train];  # labels (model class)
xmat = hcat(xoff...);

# save
npzwrite(joinpath(TRUE_DIR, "true_revon2345.npz"), Dict("xmat" => xmat, "label" => labels))


# binarized promoter states from DART
load_copy(path) = ( @load path rn_met_dl; copy(rn_met_dl) )

# --- Base path ---
DART_DIR = joinpath("synthetic_data", "svmon")

# --- Load binarized promoter states inferred from deep learning part ---
tel_met_dl   = load_copy(joinpath(DART_DIR, "ntel_dlnb1_compare_seed_1.jld2"))
on2_met_dl  = load_copy(joinpath(DART_DIR, "non2_dlnb1_compare_seed_1.jld2"))
on3_met_dl = load_copy(joinpath(DART_DIR, "non3_dlnb1_compare_seed_1.jld2"))
on4_met_dl = load_copy(joinpath(DART_DIR, "non4_dlnb1_compare_seed_1.jld2"))


tel_ms = [];
for (i,d) in enumerate(tel_met_dl)
    push!(tel_ms, (even_time = d.binart, label = 1, acc = d.acc, id = i))
end

on2_ms = [];
for (i,d) in enumerate(on2_met_dl)
    push!(on2_ms, (even_time = d.binart, label = 2, acc = d.acc, id = i))
end

on3_ms = [];
for (i,d) in enumerate(on3_met_dl)
    push!(on3_ms, (even_time = d.binart, label = 3, acc = d.acc, id = i))
end

on4_ms = [];
for (i,d) in enumerate(on4_met_dl)
    push!(on4_ms, (even_time = d.binart, label = 4, acc = d.acc, id = i))
end


## save to file that can be used as the input for SVM

ml_train = shuffle(vcat(tel_ms,on2_ms,on3_ms,on4_ms)) # shuffle dataset

ls = [length(vcat(d.even_time[1]...)) for d in ml_train];
ml_train = copy(ml_train[findall(x->x>1000,ls)]);

xoff = [vcat(d.even_time[1]...)[1:1000] for d in ml_train] # off times
labels = [Int(d.label) for d in ml_train];  # labels (model class)
xmat = hcat(xoff...);

#save
npzwrite(joinpath(DART_DIR, "ml_revon2345.npz"), Dict("xmat" => xmat, "label" => labels))