-module(kdl_alloc).

-export([start/0, start/3, start/2]).

-record(prog,
    {
        varDict,
        aliasSets,
        aliasDict,
        modDict,
        excludeSet
    }).

-define(INIT_POPU_SIZE, 100).

start() ->
    start("test.erl", self()).

start(Fn, Pid) -> start(Fn, filename:rootname(Fn) ++ "-run/", Pid).

start(Fn, Path, RptPid) ->
    ok = filelib:ensure_dir(Path),
    {ok, Terms} = file:consult(Fn),
    {Inputs, Outputs, Prog} = lists:foldl(fun (T, Prog) -> load(T, Prog) end,
                       #prog{varDict = dict:new(),
                             aliasSets = [],
                             modDict = dict:new(),
                             aliasDict = dict:new()},
                       Terms),

    unconsult_prog(Path, {Inputs, Outputs, Prog}),

    Schedules = schedule(Inputs, Outputs, Prog#prog.modDict),

    error_logger:info_report("allocating..."),
    AllPlans = lists:sort(fun (A, B) -> element(1, A) =< element(1, B) end,
                pmap(fun (ASchedule) ->
                    AllocProg = allocater(ASchedule, {Inputs, Outputs, Prog}),
                    {Inf, Sup, FirstGen} = _R = placer(AllocProg),
                    {Inf, Sup, FirstGen, AllocProg, ASchedule}
                    end, Schedules)),
    error_logger:info_msg("Up to 5 out of all ~p possible plans are selected.~n", [length(AllPlans)]),
    SelPlan = lists:sublist(AllPlans, 5),

    F = fun () ->
            {PidDict, _} = lists:foldl(fun ({Inf, Sup, FirstGen, AllocProg, ASchedule}, {DictAcc, Counter}) ->
                VarGroup = analyze_prog(AllocProg),
                unconsult(filename:join([Path, integer_to_list(Counter) ++ ".sch"]), {Inf, Sup, ASchedule, VarGroup}),
                InitPopulation = lists:map(fun (X) -> {Sup, dict:from_list(X)} end, FirstGen),
                Pid = ga(InitPopulation, VarGroup, Inf, Counter, Path),
                {dict:store(Counter, {Pid, ASchedule}, DictAcc), Counter + 1}
            end, {dict:new(), 0}, SelPlan),
            unconsult(filename:join([Path, "plan"]), dict:to_list(PidDict)),
            unconsult(filename:join([Path, "schedule"]),
                dict:fold(fun (DK, {_Pid, ASchedule}, Acc) -> [{DK, ASchedule} | Acc] end, [], PidDict)),
            ga_supervisor(PidDict, RptPid)
    end,
    spawn(F).

ga_supervisor(PidDict, RptPid) ->
    receive
        {done, ID} ->
            dict:fold(fun (_ID0, {Pid, _V}, _Acc) -> Pid ! kill end, 0, PidDict),
            error_logger:info_report("Job done."),
            RptPid ! {done, ?MODULE, ID};
        {quit, ID} ->
            PidDict10 = dict:erase(ID, PidDict),
            case dict:size(PidDict10) of
                0 ->
                    error_logger:info_report("All workers exit."),
                    RptPid ! {fail, ?MODULE};
                _ -> ga_supervisor(PidDict10, RptPid)
            end;
        Msg ->
            error_logger:error_msg("unknown message: ~p~n", [Msg]),
            ga_supervisor(PidDict, RptPid)
    after 2000 ->
            dict:map(fun (_ID, {Pid, _V}) -> Pid ! {i, self()} end, PidDict),
            ga_supervisor(PidDict, RptPid)
    end.

load({func, Name, Inputs, Outputs, Alias},
        #prog{varDict = VarDict, aliasSets = AliasSets, modDict = ModDict} = Prog) ->
    VarDict10 = lists:foldl(fun loadoutput/2,
                            lists:foldl(fun loadinput/2, VarDict, Inputs),
                            Outputs),

    AliasSets10 = lists:foldl(fun loadalias/2, AliasSets, Alias),

    error = dict:find(Name, ModDict),
    Ins = sets:from_list(Inputs),
    Outs = sets:from_list(lists:map(fun ({N, _S}) -> N end, Outputs)),
    ModDict10 = dict:store(Name, {Ins, Outs}, ModDict),
    case sets:is_disjoint(Ins, Outs) of
        false ->
            error_logger:error_msg("Input/Output must be disjoint: ~p~n", [Name]),
            throw(error);
        _ -> ok
    end,

    Prog#prog{varDict = VarDict10, aliasSets = AliasSets10, modDict = ModDict10};

load({program, Inputs, Outputs, Excludes},
        #prog{varDict = VarDict, aliasSets = AliasSets, modDict = ModDict} = Prog) ->

    % this is output from external world
    VarDict5 = lists:foldl(fun loadoutput/2, VarDict, Inputs),

    % check output variables are defined
    lists:map(fun (X) -> {ok, _} = dict:find(X, VarDict5) end, Outputs),

    % alias renaming
    {AliasDict, VarDict10} = lists:foldl(fun (EleSet, {AccDict, AccVarDict}) ->
                [H | T] = sets:to_list(EleSet),
                NH = "alias_" ++ H,
                case dict:find(NH, AccVarDict) of
                    error -> ok;
                    _ -> throw({badname, H})
                end,
                {
                    lists:foldl(fun (X, Acc) -> dict:store(X, NH, Acc) end, AccDict, [H | T]),
                    dict:store(NH, lists:max(lists:map(
                                fun (V) ->
                                        {ok, S} = dict:find(V, AccVarDict),
                                        S
                                end, [H | T])), AccVarDict)
                }
        end,
        {dict:new(), VarDict5},
        AliasSets),

    % update {Ins, Outs} to {Ins, Outs, Depends}, in which Depends = union(Ins, Oups)
    ModDict10 = dict:fold(fun (K, {Ins, Outs}, Dict) ->
                Combine = sets:to_list(sets:union(Ins, Outs)),
                dict:store(K, {Ins, Outs, sets:from_list(alias_map(Combine, AliasDict))}, Dict)
        end, ModDict, ModDict),

    % create a psudo function for output stage
    error = dict:find(output, ModDict10),
    InSet = sets:from_list(alias_map(Outputs, AliasDict)),
    ModDict20 = dict:store(output, {sets:from_list(Outputs), sets:new(), InSet}, ModDict10),

    {lists:map(fun ({N, _S}) -> N end, Inputs), Outputs,
     Prog#prog{modDict = ModDict20, varDict = VarDict10,
               aliasDict = AliasDict,
               excludeSet = sets:from_list(alias_map(Excludes, AliasDict))}}.

alias_map(L, AliasDict) ->
     lists:map(fun (Var) ->
                case dict:find(Var, AliasDict) of
                    error -> Var;
                    {ok, Alias} -> Alias
                end
        end, L).

loadinput(Name, Dict) ->
    case dict:find(Name, Dict) of
        error -> dict:store(Name, 0, Dict);
        _ -> Dict
    end.

loadoutput({Name, Size}, Dict) when Size > 0 ->
    case dict:find(Name, Dict) of
        error -> dict:store(Name, Size, Dict);
        {ok, 0} ->
            dict:store(Name, Size, Dict);
        _ -> throw({redefine, Name})
    end.

loadalias(Alias, Sets) -> loadalias0(sets:from_list(Alias), Sets, []).

loadalias0(Alias, [], Acc) ->
    [Alias | Acc];
loadalias0(Alias, [H | T], Acc) ->
    case sets:is_disjoint(H, Alias) of
        true -> loadalias0(Alias, T, [H | Acc]);
        false -> [sets:union(H, Alias) | T] ++ Acc
    end.

% return [Schedule()]
%       Schedule() = [FunName()]
schedule(Inputs, Outputs, ModDict) ->
    schedule(sets:from_list(Inputs), sets:from_list(Outputs), ModDict, [[]]).

schedule(ReadySet, Needed, ModDict, Acc) ->

    % which modules can be scheduled now?
    CanSchedule = dict:fold(fun (Name, {InSet, _OutSet, _Deps}, AccIn) ->
                                case sets:is_subset(InSet, ReadySet) of
                                    true -> [Name | AccIn];
                                    _ -> AccIn
                                end
                            end, [], ModDict),
    schedule_dispatch(ReadySet, Needed, CanSchedule, ModDict, Acc).

schedule_dispatch(ReadySet, Needed, [], ModDict, Acc) ->
    case sets:is_subset(Needed, ReadySet) of
        false -> error_logger:error_msg("program output not effient: ~p~nNeeded: ~p~n",
                [sets:to_list(ReadySet), sets:to_list(Needed)]);
        _ -> ok
    end,
    case dict:size(ModDict) of
        0 -> ok;
        _ -> error_logger:error_msg("modules padding: ~p~n",
                [dict:fetch_keys(ModDict)])
    end,
    [lists:reverse(X) || X <-Acc];
schedule_dispatch(ReadySet, Needed, CanSchedule, ModDict, Acc) ->
    lists:append(lists:map(fun (Mod) ->
                {_InSet, OutSet, _Deps} = dict:fetch(Mod, ModDict),
                Acc10 = lists:map(fun (Schedule) -> [Mod | Schedule] end, Acc),
                schedule(sets:union(ReadySet, OutSet), Needed, dict:erase(Mod, ModDict), Acc10)
        end, CanSchedule)).

allocater(Schedule,
    {Input, _Output,
     #prog{varDict = VarDict, modDict = ModDict, aliasDict = AliasDict, excludeSet = ExcludeSet} = _Prog}) ->
    Allocated = sets:from_list(alias_map(Input, AliasDict)),
    allocater(Schedule, Allocated, ModDict, VarDict, ExcludeSet,
              gen_cmd(alloc, Allocated, VarDict, ExcludeSet)).

allocater([output], _Allocated, _ModDict, _VarDict, _ExcludeSet, CmdAcc) ->
    CmdAcc;
allocater(Schedule, Allocated, ModDict, VarDict, ExcludeSet, CmdAcc) ->
    CanFree = lists:foldl(fun (F, Acc) ->
                {_InSet, _OutSet, Deps} = dict:fetch(F, ModDict),
                sets:subtract(Acc, Deps)
        end, Allocated, Schedule),

    {_InSet, _OutSet, Deps} = dict:fetch(hd(Schedule), ModDict),
    NeedAlloc = sets:subtract(Deps, Allocated),

    allocater(tl(Schedule),
              sets:union(sets:subtract(Allocated, CanFree), NeedAlloc),
              ModDict,
              VarDict,
              ExcludeSet,
              CmdAcc ++ gen_cmd(free, CanFree, VarDict, ExcludeSet)
              ++ [{call, hd(Schedule)} | gen_cmd(alloc, NeedAlloc, VarDict, ExcludeSet)]).

gen_cmd(Type, Set, VarDict, ExcludeSet) ->
    Needed = sets:subtract(Set, ExcludeSet),
    sets:fold(fun (Ele, Acc) ->
                {ok, Size} = dict:find(Ele, VarDict),
                [{Type, Ele, Size} | Acc] end, [], Needed).

% prog() = [cmd()]
% cmd() = {alloc, name(), size()} | {free, name(), size()}
placer(Prog) ->
    {_Acc, Inf, Sup, AllVar} = lists:foldl(fun
                ({alloc, Name, Size}, {Acc0, Inf0, Sup0, AllVar0}) ->
                    Acc1 = Acc0 + Size,
                    {Acc1, max(Acc1, Inf0), Sup0 + Size, dict:store(Name, Size, AllVar0)};
                ({free, _Name, Size}, {Acc0, Inf0, Sup0, AllVar0}) ->
                    Acc1 = Acc0 - Size,
                    {Acc1, Inf0, Sup0, AllVar0};
                (_Ignore, Acc0) ->
                    Acc0
            end, {0, 0, 0, dict:new()}, Prog),
    FirstGen = lists:map(fun (Vars) ->
                {Sup, R} = lists:foldl(fun ({V, S}, {Addr, Acc}) ->
                                {Addr + S, [{V, {Addr, S}} | Acc]} end, {0, []}, Vars),
                lists:reverse(R)
        end, perms_sample(dict:to_list(AllVar), ?INIT_POPU_SIZE)),
    {Inf, Sup, FirstGen}.

fract(N) ->
    fract(N, 1).

fract(1, Acc) -> Acc;
fract(N, Acc) -> fract(N - 1, N * Acc).

perms_sample(L, N) ->
    Len = length(L),
    P = fract(Len),
    Min = erlang:min(P, N),
    Sets = nestwhile(
        fun (Acc) -> sets:add_element(perms_sample0(L, Len), Acc) end,
        fun (Acc) -> sets:size(Acc) < Min end,
        sets:new()),
    sets:to_list(Sets).

nestwhile(F, T, V) ->
    case T(V) of
        true -> nestwhile(F, T, F(V));
        _ -> V
    end.

nest(_F, 0, Acc) -> Acc;
nest(F, I, Acc) -> nest(F, I - 1, F(Acc)).

perms_sample0(L, Size) ->
    {_LL, 0, R} = nest(fun ({L0, Size0, Acc}) ->
                I = rand:uniform(Size0),
                {H, L1} = pick_item(L0, I),
                {L1, Size0 - 1, [H | Acc]}
        end, Size, {L, Size, []}),
    R.

pick_item(L, I) ->
    pick_item(L, I, []).

pick_item([H | T], 1, Acc) -> {H, lists:append(T, Acc)};
pick_item([H | T], I, Acc) -> pick_item(T, I - 1, [H | Acc]).

% Candi = [{V, {A, S}}] must be sorted!
clean_hole(Candi) ->
    {{Size, _}, R} = lists:foldl(fun
                    ({V, {A, S}}, {{Last, OffAcc}, Acc}) when A - OffAcc > Last ->
                        Off1 = OffAcc + A - Last,
                        {{Last + S, Off1}, [{V, {A - Off1, S}} | Acc]};
                    ({V, {A, S}}, {{Last, OffAcc}, Acc}) ->
                        A1 = A - OffAcc,
                        {{max(Last, A1 + S), OffAcc}, [{V, {A1, S}} | Acc]}
                end, {{0, 0}, []}, Candi),
    {Size, lists:reverse(R)}.

analyze_prog(Prog) ->
    analyze_prog(alloc, Prog, [], []).

analyze_prog(_, [], [], Acc) -> Acc;
analyze_prog(_, [], [_H | _] = CurAcc, Acc) -> [CurAcc | Acc];
analyze_prog(alloc, [{alloc, Name, _} | T], CurAcc, Acc) ->
    analyze_prog(alloc, T, [Name | CurAcc], Acc);
analyze_prog(free, [{free, Name, _} | T], CurAcc, Acc) ->
    analyze_prog(free, T, CurAcc -- [Name], Acc);
analyze_prog(alloc, [{free, _, _} | _T] = In, CurAcc, Acc) ->
    analyze_prog(free, In, CurAcc, [CurAcc | Acc]);
analyze_prog(free, [{alloc, _, _} | _T] = In, CurAcc, Acc) ->
    analyze_prog(alloc, In, CurAcc, Acc);
analyze_prog(Cmd, [_ | T], CurAcc, Acc) ->
    analyze_prog(Cmd, T, CurAcc, Acc).

verify_map([], _Map) ->
    true;
verify_map([H | T] = _VarGroups, Map) ->
    case verify_map0(H, Map) of
        false -> false;
        _ -> verify_map(T, Map)
    end.

verify_map0(VarGroup, Map) ->
    Positions = lists:map(fun (Var) -> dict:fetch(Var, Map) end, VarGroup),
    not segs_has_overlap(Positions).

segs_has_overlap(Segs) ->
    segs_has_overlap(Segs, []).

segs_has_overlap([], _Acc) -> false;
segs_has_overlap([H | T], Acc) ->
    case seg_has_overlap(H, Acc) of
        true -> true;
        _ -> segs_has_overlap(T, [H | Acc])
    end.

seg_has_overlap(_X, []) ->
    false;
seg_has_overlap({A, _S}, [{A0, S0} | _T]) when (A >= A0) and (A < A0 + S0) ->
    true;
seg_has_overlap({A, S}, [{A0, _S0} | _T]) when (A0 >= A) and (A0 < A + S) ->
    true;
seg_has_overlap(X, [_H | T]) ->
    seg_has_overlap(X, T).

-record (ga_state,
         {
             path,
             popu,
             varGroup,
             allGene,
             sizeInf,
             sizeSup,
             supervisor,
             id,

             lastSize = infinity,
             sizeCounter = 0,
             totalGen = 0,

             maxTotal = 100000,
             maxNoChange = 2000,
             popuSize
         }).

ga([{Size, Candi} | _T] = InitPopulation, VarGroup, SizeInf, ID, Path) ->
    State = #ga_state{
        path = Path,
        popu = InitPopulation,
        varGroup = VarGroup,
        allGene = dict:fetch_keys(Candi),
        sizeInf = SizeInf,
        sizeSup = Size,
        supervisor = self(),
        popuSize = length(InitPopulation),
        id = ID},
    spawn(fun () -> ga_loop(State) end).

ga_loop(#ga_state{popu = [{Size, _Candi} | _T] = Population, path = Path} = State) ->
    receive
        {i, _Pid} ->
            error_logger:info_msg("progress of ~p:~nCur -> Inf -> Sup: ~p -> ~p -> ~p~nGen: ~p~n",
                [State#ga_state.id, Size, State#ga_state.sizeInf, State#ga_state.sizeSup, State#ga_state.totalGen]);
        kill ->
            save_population(Path, State#ga_state.id, Population, "kill"),
            exit(normal);
        _ ->
            ok
    after
        0 ->
            ok
    end,

    {Pmutate, SizeFactor} = config_prop(State#ga_state.totalGen),
    NewPopulation = evolve(Population, { Pmutate,
                                         State#ga_state.allGene,
                                         State#ga_state.varGroup,
                                         erlang:max(round(SizeFactor * State#ga_state.popuSize), 1)}),
    case ga_pp(State#ga_state{popu = NewPopulation}) of
        X when is_atom(X) -> State#ga_state.supervisor ! {X, State#ga_state.id};
        NewState -> ga_loop(NewState)
    end.

ga_pp(#ga_state{popu = Population = [{Size, _Candi} | _T], sizeInf = SizeInf, path = Path} = State) when Size < SizeInf ->
    error_logger:error_msg("Worker[~p] error: internal error!", [State#ga_state.id]),
    save_population(Path, State#ga_state.id, Population, "error"),
    done;
ga_pp(#ga_state{popu = Population = [{Size, _Candi} | _T], sizeInf = SizeInf, path = Path} = State) when Size == SizeInf ->
    error_logger:info_msg("Worker[~p] done: Inf reached!", [State#ga_state.id]),
    save_population(Path, State#ga_state.id, Population, "done"),
    done;
ga_pp(#ga_state{popu = [{Size, _Candi} | _T], lastSize = LastSize, totalGen = Total} = State) when Size < LastSize ->
    State#ga_state{lastSize = Size, sizeCounter = 0, totalGen = Total + 1};
ga_pp(#ga_state{popu = [{Size, _Candi} | _T], lastSize = LastSize, totalGen = Total} = State) when Size < LastSize ->
    State#ga_state{lastSize = Size, sizeCounter = 0, totalGen = Total + 1};
ga_pp(#ga_state{popu = Population, totalGen = Total, maxTotal = Max, path = Path} = State) when Total > Max ->
    error_logger:error_msg("Worker[~p] failed: maximum generation reached!~n", [State#ga_state.id]),
    save_population(Path, State#ga_state.id, Population, "failed"),
    quit;
ga_pp(#ga_state{popu = Population, lastSize = LastSize,
                sizeCounter = Counter, maxNoChange = Max, path = Path} = State) when Counter > Max ->
    error_logger:error_msg("Worker[~p] aborted: size couldn't be further minimized: ~p!~n", [State#ga_state.id, LastSize]),
    save_population(Path, State#ga_state.id, Population, "abort"),
    quit;
ga_pp(#ga_state{sizeCounter = Counter, totalGen = Total} = State) ->
    State#ga_state{totalGen = Total + 1, sizeCounter = Counter + 1}.

config_prop(TotalGen) when TotalGen < 50 ->
    {0.8, 1.0};
config_prop(TotalGen) when TotalGen < 200 ->
    {0.6, 1.0};
config_prop(TotalGen) when TotalGen < 500 ->
    {0.5, 0.8};
config_prop(TotalGen) when TotalGen < 1000 ->
    {0.5, 0.5};
config_prop(_TotalGen) ->
    {0.5, 0.1}.

save_population(Path, ID, Population, Type) ->
    Fn = integer_to_list(ID),
    {ok, Fid} = file:open(filename:join([Path, Fn ++ "." ++ Type]), [write]),
    lists:map(fun ({Size, Dict}) ->
                io:format(Fid, "{size, ~p}.~n{map, ~p}.~n%------------------------~n", [Size, dict:to_list(Dict)]) end, Population),
    file:close(Fid).

unconsult_prog(Path, {Inputs, Outputs, #prog{aliasSets = AliasSets, aliasDict = AliasDict} = _Prog}) ->
    {ok, Fid} = file:open(filename:join([Path, "prog"]), [write]),
    io:format(Fid, "%%============== Alias ===========%%~n", []),
    lists:map(fun (Set) ->
        [AItem | _] = Alias = sets:to_list(Set),
        io:format(Fid, "{~p, ~p}.~n", [Alias, dict:fetch(AItem, AliasDict)])
    end, AliasSets),
    io:format(Fid, "%%============== In ===========%%~n~p.~n", [Inputs]),
    io:format(Fid, "%%============== Out ===========%%~n~p.~n", [Outputs]),
    file:close(Fid).

unconsult(FN, Term) ->
    {ok, Fid} = file:open(FN, [write]),
    io:format(Fid, "~p.~n", [Term]),
    file:close(Fid).

% population = [candi()]
% candi() = {size(), map()}
% map() = dict(name(), {address(), size()})
evolve(Population, {Pmutate, AllGene, VarGroups, PopuSize}) ->
    F = case Population of
        [_] -> fun (Candi) ->
                    NewDict = mutate(Population, Candi, AllGene),
                    cdict_cleanup(NewDict)
                end;
        _ -> fun (Candi) ->
                    NewDict = case rand:uniform() =< Pmutate of
                        true -> mutate(Population, Candi, AllGene);
                        _ -> crossover(Population, Candi, AllGene)
                    end,
                    cdict_cleanup(NewDict)
                end
    end,
    NewGen = lists:map(F, Population),
    nature_sel(Population, VarGroups, NewGen, PopuSize).

nature_sel(Population, VarGroups, NewGen, PopulationSize) ->
    Survivals = lists:foldl(fun ({Size, L}, Acc) ->
                Dict = dict:from_list(L),
                case verify_map(VarGroups, Dict) of
                    true -> [{Size, Dict} | Acc];
                    _ -> Acc
                end
                end, [], NewGen),

    % merge with parent generation
    All = lists:umerge(Population, lists:usort(Survivals)),
    lists:sublist(All, PopulationSize).

cdict_cleanup(Dict) ->
    L = dict:to_list(Dict),
    L10 = lists:sort(fun ({_K1, {A1, _S1}}, {_K2, {A2, _S2}}) -> A1 =< A2 end, L),
    clean_hole(L10).

mutate(_Population, {Size, CandiDict}, AllGene) ->
    Gene = choose(AllGene),
    dict:update(Gene,
                fun ({V, S}) ->
                    N = (Size - S),
                    Off = case rand:uniform() >= 0.5 of
                        true -> rnd_walk(N div 2);
                        _ -> -rnd_walk(N div 2)
                    end,
                    {mod(V + Off, N), S}
                end, CandiDict).

mod(X, Y) ->
    case X rem Y of
        V when V < 0 -> V + Y;
        V -> V
    end.

crossover(Population, {_Size, CandiDict} = This, AllGene) ->
    Gene = choose(AllGene),
    T = choose(Population, This),
    {_Size10, Parent} = T,
    dict:store(Gene, dict:fetch(Gene, Parent), CandiDict).

% output: 1..N
rnd_walk(N) ->
    M = N - 1,
    M - round(M * math:sqrt(rand:uniform())) + 1.

choose(Items) ->
    lists:nth(rand:uniform(length(Items)), Items).

choose(Items, Ref) ->
    case choose(Items) of
        Ref -> choose(Items, Ref);
        X -> X
    end.

pmap(F, L) ->
    S = self(),
    Pids = lists:map(fun(I) -> spawn(fun() -> pmap_f(S, F, I) end) end, L),
    pmap_gather(Pids).

pmap_gather([H|T]) ->
    receive
        {H, Ret} -> [Ret|pmap_gather(T)]
    end;
pmap_gather([]) ->
    [].

pmap_f(Parent, F, I) ->
    Parent ! {self(), (catch F(I))}.

-if(defined(TEST)).
%% test
clean_hole_test() ->
    io:format("~p~n", [clean_hole([{"a", {0, 10}}, {"b", {20, 10}}, {"c", {30, 10}}])]),
    io:format("~p~n", [clean_hole([{"a", {0, 10}}, {"b", {5, 10}}, {"c", {30, 10}}])]),
    io:format("~p~n", [clean_hole([{"a", {0, 10}}, {"b", {10, 8}}, {"c", {19, 10}}])]).

analyze_prog_test() ->
    R = analyze_prog(
        [{alloc,"a-i-0",100},
         {call,"a"},
         {alloc,"a-o-0",200},
         {alloc,"a-o-1",200},
         {free,"a-i-0",100},
         {call,"b"},
         {alloc,"b-o-0",200},
         {free,"a-o-0",200},
         {call,"c"},
         {alloc,"b-o-1",200}]),
    io:format("~p~n", [R]).

segs_has_overlap_test() ->
    R = segs_has_overlap(
        [{10, 10},
         {20, 20},
         {40, 20}]),
    io:format("~p~n", [R]).

test_perms_sample() ->
    perms_sample([1, 2, 3, 4], 24).
-endif.