using Catalyst

# calculate theoretical mean time

function ss_mean(k,tau)
    n = (k[1]*k[3]*k[5]*(k[10]*k[8]*k[11] + k[10]*k[7]*k[12] + k[7]*k[9]*k[13])*tau)
    d = (k[10]*k[2]*k[4]*k[6]*k[8] + k[1]*k[10]*(k[3]*k[5]*k[7] + k[4]*k[6]*k[8] + k[3]*(k[5] + k[6])*k[8]) +k[1]*k[3]*k[5]*k[7]*k[9])
    return n/d
end

function compute_offt(k, id, rn)
    if rn == "ref"
        return sum(1 ./ k[1:id-1]); 
    elseif rn == "perm"
        return (k[1]+k[2])/(k[1]*k[3])
    elseif rn == "perm1"
        return (k[1]*k[3]+k[1]*k[4]+k[2]*k[4])/(k[1]*k[3]*k[5])
    elseif rn == "gen"
        return (k[2]*k[4]+k[1]*(k[3]+k[4]))/(k[1]*k[3]*k[5])
    end
end

function compute_ont(k, id, rn)
    if rn == "gen"
        return (k[7]*k[9]+k[10]*(k[7]+k[8]))/(k[6]*k[8]*k[10])
    else       
        return 1/k[id]
    end
end


#a1: G1 -> G2
#b1: G2 -> G1
#c1: G2 -> G2 + N, triggers N => τ 0
#d: N -> 0
# 1. G1, 2. G2, 3. N
function construct_prob_delaytel(params)
    
    a1, b1, c1, d, τ, t0, tf = params
    rates = [a1, b1, c1, d]
    
    # Markovian
    react_stoich = [[1=>1],[2=>1],[2=>1],[3=>1]] # reactant index => reactant coefficient
    net_stoich = [[1=>-1,2=>1],[1=>1,2=>-1],[3=>1],[3=>-1]] # reactant index => net change, excluding 0 change
    mass_action_jump = MassActionJump(rates, react_stoich, net_stoich; scale_rates = false) # optimized representation for ConstantRateJumps
    jumpset = JumpSet((),(),nothing,mass_action_jump)
    
    # non-Markovian
    delay_trigger = Dict(3=>[1=>τ]) # indices of reactions that can trigger delay reactions=>[delay channels=>delay time]
    delay_complete = Dict(1=>[3=>-1]) # indices of delay channels =>[species index=>net change]
    delayjumpset = DelayJumpSet(delay_trigger,delay_complete,Dict())
    
    u0 = [1,0,0] # initial condition
    de_chan0 = [[]] # initial delay channel
    tspan = (t0,tf)
    
    dprob = DiscreteProblem(u0, tspan)
    djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset, de_chan0, save_positions = (false,false), save_delay_channel = true)
    return djprob
end

#b1: G1 -> G2
#a1: G2 -> G3
#b2: G3 -> G1
#c1: G3 -> G3 + N, N => τ 0
#d: N -> 0
# 1. G1, 2. G2, 3. G3, 4. N
function construct_prob_delayref(params) # 3 gene states 1 reversible 1 state produces mRNA
    
    b1, a1, b2, c1, d, τ, t0, tf = params
    rates = [b1, a1, b2, c1, d]
    
    # Markovian
    react_stoich = [[1=>1],[2=>1],[3=>1],[3=>1],[4=>1]] # reactant index => reactant coefficient
    net_stoich = [[1=>-1,2=>1],[2=>-1,3=>1],[3=>-1,1=>1],[4=>1],[4=>-1]] # reactant index => net change, excluding 0 change
    mass_action_jump = MassActionJump(rates, react_stoich, net_stoich; scale_rates = false) # optimized representation for ConstantRateJumps
    jumpset = JumpSet((),(),nothing,mass_action_jump)
    
    # non-Markovian
    delay_trigger = Dict(4=>[1=>τ]) # indices of reactions that can trigger delay reactions=>[delay channels=>delay time]
    delay_complete = Dict(1=>[4=>-1]) # indices of delay channels =>[species index=>net change]
    delayjumpset = DelayJumpSet(delay_trigger,delay_complete,Dict())
    
    u0 = [1,0,0,0] # initial condition
    de_chan0 = [[]] # initial delay channel
    tspan = (t0,tf)
    
    dprob = DiscreteProblem(u0, tspan)
    djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset, de_chan0, save_positions = (false,false), save_delay_channel = true)
    return djprob
end

#b1: G1 -> G2
#b2: G2 -> G1
#a1: G2 -> G3
#b3: G3 -> G2
#c1: G3 -> G3 + N, N => τ 0
#d: N -> 0
# 1. G1, 2. G2, 3. G3, 4. N

function construct_prob_delayperm(params) # with true time (not evenly distributed)
    
    b1, b2, a1, b3, c1, d, τ, t0, tf = params
    rates = [b1, b2, a1, b3, c1, d]
    
    # Markovian
    react_stoich = [[1=>1],[2=>1],[2=>1],[3=>1],[3=>1],[4=>1]] # reactant index => reactant coefficient
    net_stoich = [[1=>-1,2=>1],[2=>-1,1=>1],[2=>-1,3=>1],[3=>-1,2=>1],[4=>1],[4=>-1]] # reactant index => net change, excluding 0 change
    mass_action_jump = MassActionJump(rates, react_stoich, net_stoich; scale_rates = false) # optimized representation for ConstantRateJumps
    jumpset = JumpSet((),(),nothing,mass_action_jump)
    
    # non-Markovian
    delay_trigger = Dict(5=>[1=>τ]) # indices of reactions that can trigger delay reactions=>[delay channels=>delay time]
    delay_complete = Dict(1=>[4=>-1]) # indices of delay channels =>[species index=>net change]
    delayjumpset = DelayJumpSet(delay_trigger,delay_complete,Dict())
    
    u0 = [1,0,0,0] # initial condition
    de_chan0 = [[]] # initial delay channel
    tspan = (t0,tf)
    
    dprob = DiscreteProblem(u0, tspan)
    djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset, de_chan0, save_positions = (false,false), save_delay_channel = true)
    return djprob
end

#a1: G1 -> G2
#a2: G2 -> G1
#a3: G2 -> G3
#a4: G3 -> G2
#a5: G3 -> G4
#a6: G4 -> G3
#c1: G4 -> G4 + N, N => τ 0
#d: N -> 0
# 1. G1, 2. G2, 3. G3, 4. G4, 5. N
function construct_prob_delayperm1(params) # with true time (not evenly distributed)
    
    a1, a2, a3, a4, a5, a6, c1, d, τ, t0, tf = params
    rates = [a1, a2, a3, a4, a5, a6, c1, d]
    
    # Markovian
    react_stoich = [[1=>1],[2=>1],[2=>1],[3=>1],[3=>1],[4=>1],[4=>1],[5=>1]] # reactant index => reactant coefficient
    net_stoich = [[1=>-1,2=>1],[2=>-1,1=>1],[2=>-1,3=>1],[3=>-1,2=>1],[3=>-1,4=>1],[4=>-1,3=>1],[5=>1],[5=>-1]] # reactant index => net change, excluding 0 change
    mass_action_jump = MassActionJump(rates, react_stoich, net_stoich; scale_rates = false) # optimized representation for ConstantRateJumps
    jumpset = JumpSet((),(),nothing,mass_action_jump)
    
    # non-Markovian
    delay_trigger = Dict(7=>[1=>τ]) # indices of reactions that can trigger delay reactions=>[delay channels=>delay time]
    delay_complete = Dict(1=>[5=>-1]) # indices of delay channels =>[species index=>net change]
    delayjumpset = DelayJumpSet(delay_trigger,delay_complete,Dict())
    
    u0 = [1,0,0,0,0] # initial condition
    de_chan0 = [[]] # initial delay channel
    tspan = (t0,tf)
    
    dprob = DiscreteProblem(u0, tspan)
    djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset, de_chan0, save_positions = (false,false), save_delay_channel = true)
    return djprob
end

#a1: G1 -> G2
#a2: G2 -> G1
#a3: G2 -> G3
#a4: G3 -> G2
#a5: G3 -> G4
#a6: G4 -> G3
#a7: G4 -> G5
#a8: G5 -> G4
#a9: G5 -> G6
#a10: G6 -> G5
#c1: G4 -> G4 + N, N => τ 0
#c2: G5 -> G5 + N, N => τ 0
#c3: G6 -> G6 + N, N => τ 0
#d: N -> 0
# 1. G1, 2. G2, 3. G3, 4. G4, 5. G5, 6. G6, 7. N
function construct_prob_delaygen(params) # with true time (not evenly distributed)
    
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, c1, c2, c3, d, τ, t0, tf = params
    rates = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, c1, c2, c3, d]
    
    # Markovian
    react_stoich = [[1=>1],[2=>1],[2=>1],[3=>1],[3=>1],[4=>1],[4=>1],[5=>1],[5=>1],[6=>1],[4=>1],[5=>1],[6=>1],[7=>1]] # reactant index => reactant coefficient
    net_stoich = [[1=>-1,2=>1],[2=>-1,1=>1],[2=>-1,3=>1],[3=>-1,2=>1],[3=>-1,4=>1],[4=>-1,3=>1],[4=>-1,5=>1],[5=>-1,4=>1],[5=>-1,6=>1],[6=>-1,5=>1],[7=>1],[7=>1],[7=>1],[7=>-1]] # reactant index => net change, excluding 0 change
    mass_action_jump = MassActionJump(rates, react_stoich, net_stoich; scale_rates = false) # optimized representation for ConstantRateJumps
    jumpset = JumpSet((),(),nothing,mass_action_jump)
    
    # non-Markovian
    delay_trigger = Dict(11=>[1=>τ],12=>[1=>τ],13=>[1=>τ]) # indices of reactions that can trigger delay reactions=>[delay channels=>delay time]
    delay_complete = Dict(1=>[7=>-1]) # indices of delay channels =>[species index=>net change]
    delayjumpset = DelayJumpSet(delay_trigger,delay_complete,Dict())
    
    u0 = [1,0,0,0,0,0,0] # initial condition
    de_chan0 = [[]] # initial delay channel
    tspan = (t0,tf)
    
    dprob = DiscreteProblem(u0, tspan)
    djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset, de_chan0, save_positions = (false,false), save_delay_channel = true)
    return djprob
end

