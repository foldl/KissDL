-module(kdl).

-export([compile/2, compile/3, test/0]).

compile(Fn, WorkDir) -> compile(Fn, WorkDir, []).

compile(Fn, WorkDir, Opts) ->
    spawn(fun () -> compile0(Fn, WorkDir, Opts) end).

compile0(Fn, WorkDir, CompileOpts) ->
    AlignOpt = proplists:get_value(align, CompileOpts, auto),
    UseRaw = proplists:get_value(raw, CompileOpts, true),
    ok = filelib:ensure_dir(WorkDir),
    {ok, Ls} = file:consult(Fn),
    io:format("preparing for memory alloc optimization ... "),
    Prog = filename:join([WorkDir, "model.prog"]),
    convert_to_prog(Ls, Prog, AlignOpt),
    io:format("start optimizing~n"),
    kdl_alloc:start(Prog, self()),
    {Schedule, Size, Addresses} = load_alloc_result(filename:join([WorkDir, "model-run"]), wait_for_id()),
    io:format("selected scheduling: ~p~ntotal RAM ~p bytes~ngenerating...~n", [Schedule, Size]),
    {ok, Fid} = file:open(filename:join([WorkDir, "model.c"]), [write]),
    {ok, Content} = file:read_file(priv_fn("kdl.inc")),
    ok = file:write(Fid, Content),
    {ok, ActFile} = file:read_file(priv_fn("activation.c")),
    put(act_templ, ActFile),

    Vars = lists:foldl(fun
                ({tensor, Name, Type, Shape, Opts}, Acc) ->
                    dict:store(Name, [{type, Type}, {shape, Shape} | Opts], Acc);
                (_, Acc) -> Acc
            end, dict:new(), Ls),
    Inited = dict:fold(fun (K, V, Acc) ->
        case proplists:get_value(data, V) of
            undefined -> Acc;
            I when is_integer(I) -> [K | Acc]
        end end, [], Vars),

    io:format(Fid, "// tensors are all statically allocated within `model_arena`~n" ++
                   "static uint8_t model_arena[~p] = {0};~n", [Size]),
    lists:foreach(fun ({N, {A, _S}}) ->
            Var = dict:fetch(N, Vars),
            T = c_type(proplists:get_value(type, Var)),
            Align = align_of_type(proplists:get_value(type, Var), AlignOpt),
            A10 = A + case A rem Align of
                0 -> 0;
                ARem -> Align - ARem
            end,
            io:format(Fid, "// shape of ~ts is ~w~n", [N, proplists:get_value(shape, Var)]),
            io:format(Fid, "~p * const ~ts = (~p *)(model_arena + ~p);~n", [T, to_c_ident(N), T, A10])
        end, Addresses),

    io:format(Fid, "~n// constants~n", []),
    lists:foreach(fun (N) ->
            Var = dict:fetch(N, Vars),
            T = proplists:get_value(type, Var),
            DataId = proplists:get_value(data, Var),
            DataFn = filename:join([filename:dirname(Fn), integer_to_list(DataId) ++ ".dat"]),
            OFn = filename:join([WorkDir, integer_to_list(DataId) ++ ".dat"]),
            case UseRaw of
                true ->
                    write_init(uint8, DataFn, OFn),
                    io:format(Fid, "// shape of ~ts is ~w~n", [N, proplists:get_value(shape, Var)]),
                    io:format(Fid, "static const int8_t ~ts_container[] = {~n    #include \"~p.dat\"~n};~n",
                                [to_c_ident(N), proplists:get_value(data, Var)]),
                    io:format(Fid, "const ~p *~ts = (const ~p *)~ts_container;~n",
                                [c_type(T), to_c_ident(N), c_type(T), to_c_ident(N)]);
                _ ->
                    write_init(T, DataFn, OFn),
                    io:format(Fid, "const ~p ~ts[] = {~n    #include \"~p.dat\"~n};~n",
                         [c_type(T), to_c_ident(N), proplists:get_value(data, Var)])
            end
        end, Inited),

    io:format(Fid, "~n", []),

    KerDict = lists:foldl(fun
            ({op, OpName, Input, Output, _Opts} = Op, Acc) ->
                case gen_kernel(Op, Vars) of
                    {ok, ScheduleName, Code} ->
                        file:write(Fid, Code),
                        dict:store(ScheduleName, {OpName, ScheduleName, Input, Output}, Acc);
                    {stored, ScheduleName, StoredName} ->
                        dict:store(ScheduleName, {OpName, StoredName, Input, Output}, Acc)
                end;
            (_, Acc) -> Acc
        end, dict:new(), Ls),

    io:format(Fid, "~nvoid model_run(void)~n{~n", []),
    lists:foreach(fun
            (output) -> ok;
            (ScheduleName) ->
                {OpName, StoredName, Input, Output} = dict:fetch(ScheduleName, KerDict),
                io:format(Fid, "    ~ts(~ts);~n", [StoredName, inject_param(OpName, Input, Output)])
        end, Schedule),
    io:format(Fid, "}~n", []),

    [{Inputs, Outputs}] = lists:filtermap(fun
            (L) when is_list(L) ->
                {true, {proplists:get_value(inputs, L), proplists:get_value(outputs, L)}};
            (_) -> false
        end, Ls),

    gen_main(Inputs, Outputs, Vars, WorkDir),

    file:close(Fid),
    io:format("done.~n").

inject_param(conv_2d, [A, B], Output) ->
    inject_param(conv_2d, [A, B, "NULL"], Output);
inject_param(depthwise_conv_2d, [A, B], Output) ->
    inject_param(depthwise_conv_2d, [A, B, "NULL"], Output);
inject_param(dense, [A, B], Output) ->
    inject_param(dense, [A, B, "NULL"], Output);
inject_param(_OpName, Input, Output) ->
    lists:join(", ", [to_c_ident(X) || X <- Input ++ Output]).

gen_main(Inputs, Outputs, Vars, WorkDir) ->
    MakeExtDec = fun (L) ->
        lists:map(fun (N) ->
            Var = dict:fetch(N, Vars),
            T = proplists:get_value(type, Var),
            io_lib:format("// shape of ~ts is ~w~n", [N, proplists:get_value(shape, Var)]) ++
                io_lib:format("extern ~p * const ~ts;~n", [c_type(T), to_c_ident(N)])
        end, L
    ) end,

    AOutput = hd(Outputs),
    Var = dict:fetch(AOutput, Vars),
    ExtDecl = ["// input tensors\n"] ++ MakeExtDec(Inputs) ++ ["\n// output tensors\n"] ++ MakeExtDec(Outputs),
    Cfg = #{extern_vars => ExtDecl, output_size => prod(proplists:get_value(shape, Var)),
      output => to_c_ident(AOutput),
      output_format => c_type(proplists:get_value(type, Var)),
      output_format_char => fmt_char(c_type(proplists:get_value(type, Var)))},

    {ok, Template} = file:read_file(priv_fn("main.c")),
    file:write_file(filename:join([WorkDir, "main_demo.c"]), kdl_template:run(Template, Cfg)).

priv_fn(Fn) ->
    PrivDir = filename:join([filename:dirname(code:which(kdl)), "..", "priv"]),
    filename:join([PrivDir, Fn]).

load_init(float32, Bin, Acc) ->
    case Bin of
        <<F:32/little-float, Rem/binary>> -> load_init(float32, Rem, [F | Acc]);
        <<>> -> Acc
    end;
load_init(float16, Bin, Acc) -> load_init(int16, Bin, Acc);
load_init(int32, Bin, Acc) ->
    case Bin of
        <<V:32/signed-little-integer, Rem/binary>> -> load_init(int32, Rem, [V | Acc]);
        <<>> -> Acc
    end;
load_init(int16, Bin, Acc) ->
    case Bin of
        <<V:16/signed-little-integer, Rem/binary>> -> load_init(int16, Rem, [V | Acc]);
        <<>> -> Acc
    end;
load_init(int8, Bin, Acc) ->
    case Bin of
        <<V:8/signed-little-integer, Rem/binary>> -> load_init(int8, Rem, [V | Acc]);
        <<>> -> Acc
    end;
load_init(uint8, Bin, Acc) ->
    case Bin of
        <<V:8/little-integer, Rem/binary>> -> load_init(int8, Rem, [V | Acc]);
        <<>> -> Acc
    end;
load_init(complex64, Bin, Acc) ->
    case Bin of
        <<Re:32/little-float, Im:32/little-float, Rem/binary>> -> load_init(complex64, Rem, [{Re, Im} | Acc]);
        <<>> -> Acc
    end.

read_template(X, Opts)->
    {ok, Content} = case maps:get(template, Opts, X) of
        A when is_atom(A) ->
            file:read_file(priv_fn(atom_to_list(X) ++ ".c"));
        L when is_list(L) ->
            file:read_file(priv_fn(L ++ ".c"))
    end,
    Content.

gen_kernel(Op, Vars) ->
    Opts0 = prepare(Op, Vars),
    Opts = maps:put(c_type, c_type(maps:get(type, Opts0)), Opts0),
    Ker = element(2, Op),
    FAcc = case get(ker_names) of undefined -> sets:new(); Acc -> Acc end,
    {ScheduleName, Acc10} = gen_key(atom_to_list(Ker), FAcc),
    put(ker_names, Acc10),
    case get({Ker, Opts}) of
        undefined ->
            Act = maps:get(activation, Opts, undefined),
            Call = fun
                (["act", Value]) ->
                    if  Act /= undefined ->
                            Opts20 = #{activation => Act, value => Value, type => maps:get(type, Opts)},
                            kdl_template:run(get(act_templ), Opts20);
                        true -> Value
                    end;
                (_) -> "???"
            end,
            put({Ker, Opts}, ScheduleName),
            Content = read_template(Ker, Opts),
            Opts10 = maps:put(kernel_fun, ScheduleName, Opts),
            Result = kdl_template:run(Content, Opts10, Call),
            {ok, ScheduleName, Result};
        FStoredName -> {stored, ScheduleName, FStoredName}
    end.

padding_with_offset(Stride, DilationRate, InSize, FilterSize, OutSize) ->
    EffectFilterSize = (FilterSize - 1) * DilationRate + 1,
    TotalPadding = max(0, (OutSize - 1) * Stride + EffectFilterSize - InSize),
    {TotalPadding div 2, TotalPadding rem 2}.

prepare({op, average_pool_2d, [Input], [Output], Opts}, Vars) ->
    prepare_pooling(average, [Input], [Output], Opts, Vars);
prepare({op, max_pool_2d, [Input], [Output], Opts}, Vars) ->
    prepare_pooling(max, [Input], [Output], Opts, Vars);
prepare({op, l2_pool_2d, [Input], [Output], Opts}, Vars) ->
    prepare_pooling(l2, [Input], [Output], Opts, Vars);
prepare({op, conv_2d, [Input, Filter | InputT], [Output], Opts}, Vars) ->
    [B, IH, IW, ID] = proplists:get_value(shape, dict:fetch(Input, Vars)),
    [OD, FH, FW, ID] = proplists:get_value(shape, dict:fetch(Filter, Vars)),
    [B, OH, OW, OD] = proplists:get_value(shape, dict:fetch(Output, Vars)),
    {StrideW, StrideH} = proplists:get_value(stride, Opts, {1, 1}),
    {DilationW, DilationH} = proplists:get_value(dilation_factor, Opts, {1, 1}),
    {PH, _PHO} = padding_with_offset(StrideH, DilationH, IH, FH, OH),
    {PW, _PWO} = padding_with_offset(StrideW, DilationW, IW, FW, OW),
    #{'BATCH_SIZE' => B, 'INPUT_HEIGHT' => IH, 'INPUT_WIDTH' => IW, 'INPUT_DEPTH' => ID,
        'FILTER_WIDTH' => FW, 'FILTER_HEIGHT' => FH, 'FILTER_DEPTH' => ID,
        'OUTPUT_HEIGHT' => OH, 'OUTPUT_WIDTH' => OW, 'OUTPUT_DEPTH' => OD,
        type => proplists:get_value(type, dict:fetch(Input, Vars)),
        dilation_width_factor => DilationW, dilation_height_factor => DilationH,
        padding => proplists:get_value(padding, Opts), padding_height => PH, padding_width => PW,
        stride_width => StrideW, stride_height => StrideH,
        has_bias => InputT =/= [],
        activation => proplists:get_value(activation, Opts)
    };
prepare({op, depthwise_conv_2d, [Input, Filter | InputT], [Output], Opts}, Vars) ->
    [B, IH, IW, ID] = proplists:get_value(shape, dict:fetch(Input, Vars)),
    [1, FH, FW, OD] = proplists:get_value(shape, dict:fetch(Filter, Vars)),
    [B, OH, OW, OD] = proplists:get_value(shape, dict:fetch(Output, Vars)),
    {StrideW, StrideH} = proplists:get_value(stride, Opts, {1, 1}),
    {DilationW, DilationH} = proplists:get_value(dilation_factor, Opts, {1, 1}),
    true = (OD == ID * proplists:get_value(depth_multiplier, Opts, 1)),
    Multi = proplists:get_value(depth_multiplier, Opts, 1),
    true = ID * Multi == OD,
    {PH, _PHO} = padding_with_offset(StrideH, DilationH, IH, FH, OH),
    {PW, _PWO} = padding_with_offset(StrideW, DilationW, IW, FW, OW),
    #{'BATCH_SIZE' => B, 'INPUT_HEIGHT' => IH, 'INPUT_WIDTH' => IW, 'INPUT_DEPTH' => ID,
        'FILTER_WIDTH' => FW, 'FILTER_HEIGHT' => FH,
        'OUTPUT_HEIGHT' => OH, 'OUTPUT_WIDTH' => OW, 'OUTPUT_DEPTH' => OD,
        type => proplists:get_value(type, dict:fetch(Input, Vars)),
        dilation_width_factor => DilationW, dilation_height_factor => DilationH,
        depth_multiplier => proplists:get_value(depth_multiplier, Opts),
        padding => proplists:get_value(padding, Opts), padding_height => PH, padding_width => PW,
        stride_width => StrideW, stride_height => StrideH,
        has_bias => InputT =/= [],
        activation => proplists:get_value(activation, Opts)
    };
prepare({op, dense, [Input, Filter | InputT], [Output], Opts}, Vars) ->
    [B | IS] = proplists:get_value(shape, dict:fetch(Input, Vars)),
    [FH, FW] = proplists:get_value(shape, dict:fetch(Filter, Vars)),
    [B | OS] = proplists:get_value(shape, dict:fetch(Output, Vars)),
    true = prod(IS) == FW,
    true = prod(OS) == FH,
    #{'BATCH_SIZE' => B, 'INPUT_SIZE' => FW, 'OUTPUT_SIZE' => FH,
        type => proplists:get_value(type, dict:fetch(Input, Vars)),
        has_bias => InputT =/= [],
        activation => proplists:get_value(activation, Opts)
    };
prepare({op, softmax, [Input], [Output], Opts}, Vars) ->
    [B | IS] = proplists:get_value(shape, dict:fetch(Input, Vars)),
    [B | OS] = proplists:get_value(shape, dict:fetch(Output, Vars)),
    true = prod(IS) == prod(OS),
    #{'BATCH_SIZE' => B, 'INPUT_SIZE' => prod(IS),
        beta => proplists:get_value(beta, Opts),
        type => proplists:get_value(type, dict:fetch(Input, Vars))
    };
prepare({op, ceil, _, _, _Opts} = Op, Vars) ->
    prepare_simple(Op, Vars);
prepare({op, floor, _, _, _Opts} = Op, Vars) ->
    prepare_simple(Op, Vars);
prepare({op, logistic, _, _, _Opts} = Op, Vars) ->
    prepare_simple(Op, Vars);
prepare({op, neg, _, _, _Opts} = Op, Vars) ->
    prepare_simple(Op, Vars);
prepare({op, round, _, _, _Opts} = Op, Vars) ->
    prepare_simple(Op, Vars).

prepare_pooling(Algo, [Input], [Output], Opts, Vars) ->
    [B, IH, IW, ID] = proplists:get_value(shape, dict:fetch(Input, Vars)),
    [B, OH, OW, OD] = proplists:get_value(shape, dict:fetch(Output, Vars)),
    {StrideW, StrideH} = proplists:get_value(stride, Opts, {1, 1}),
    {FW, FH} = proplists:get_value(filter, Opts, {1, 1}),
    {PH, _PHO} = padding_with_offset(StrideH, 1, IH, FH, OH),
    {PW, _PWO} = padding_with_offset(StrideW, 1, IW, FW, OW),
    #{'BATCH_SIZE' => B, 'INPUT_HEIGHT' => IH, 'INPUT_WIDTH' => IW, 'INPUT_DEPTH' => ID,
        'FILTER_WIDTH' => FW, 'FILTER_HEIGHT' => FH,
        'OUTPUT_HEIGHT' => OH, 'OUTPUT_WIDTH' => OW, 'OUTPUT_DEPTH' => OD,
        type => proplists:get_value(type, dict:fetch(Input, Vars)),
        padding => proplists:get_value(padding, Opts), padding_height => PH, padding_width => PW,
        stride_width => StrideW, stride_height => StrideH,
        activation => proplists:get_value(activation, Opts),
        algo => Algo,
        template => "pooling"
    }.

prepare_simple({op, _opName, [Input], [Output], _Opts}, Vars) ->
    Shape1 = proplists:get_value(shape, dict:fetch(Input, Vars)),
    Shape2 = proplists:get_value(shape, dict:fetch(Output, Vars)),
    true = prod(Shape1) == prod(Shape2),
    #{'FLAT_SIZE' => prod(Shape1),
        type => proplists:get_value(type, dict:fetch(Input, Vars))
    }.

write_init(Type, Fn, OFn) when is_list(Fn) ->
    {ok, [Bin]} = file:consult(Fn),
    Values = lists:reverse(load_init(Type, Bin, [])),
    {ok, Fid} = file:open(OFn, [write]),
    [T] = io_lib:format("~p", [Values]),
    T10 = lists:sublist(T, 2, length(T) - 2),
    file:write(Fid, T10),
    file:close(Fid).

to_c_ident(L) ->
    R = [
            case C of
                C when $a =< C, C =< $z -> C;
                C when $A =< C, C =< $Z -> C;
                C when $0 =< C, C =< $9 -> C;
                _ -> $_
            end || C <- L
        ],
    case hd(R) of
        C when $0 =< C, C =< $9 -> [$_ | R];
        _ -> R
    end.

wait_for_id() ->
    receive
        {fail, kdl_alloc} ->
            io:format("try to run again.", []),
            throw(fail);
        {done, kdl_alloc, ID} ->
            ID
    end.

align_of_type(_Type, 1) -> 1;
align_of_type(Type, auto) -> sizeof(Type).

convert_to_prog(Ls, Fn, AlignOpt) ->
    {ok, Fid} = file:open(Fn, [write]),
    [{inputs, Ins}, {outputs, Outs}] = hd(Ls),
    Vars = lists:foldl(fun
                ({tensor, Name, Type, _Shape, Opts} = Tensor, Acc) ->
                    TensorSize = sizeof(Tensor) + align_of_type(Type, AlignOpt) - 1,
                    Info = [{size, TensorSize}, {init, proplists:get_value(data, Opts) =/= undefined}],
                    dict:store(Name, Info, Acc);
                (_, Acc) -> Acc
            end, dict:new(), Ls),
    Inited = dict:fold(fun (K, V, Acc) ->
        case proplists:get_value(init, V) of
            true -> [K | Acc];
            _ -> Acc
        end end, [], Vars),
    Ins10 = Ins ++ Inited,

    lists:foldl(fun
                ({op, Name, OpIns, OpOuts, _Opts} = Op, FAcc) ->
                    Temps = get_intermidiates(Op, Vars),
                    Aliases = get_aliases(Op, Vars),
                    OpOuts10 = to_var_size_list(Vars, OpOuts) ++ Temps,
                    {FName10, FAcc10} = gen_key(atom_to_list(Name), FAcc),
                    io:format(Fid, "{func, ~p, ~p, ~p, ~p}.~n", [
                                           FName10,
                                           OpIns,
                                           OpOuts10, Aliases]),
                    FAcc10;
                (_, FAcc) -> FAcc
        end, sets:new(), Ls),

    io:format(Fid, "{program, ~p, ~p, ~p}.~n", [to_var_size_list(Vars, Ins10),
                                           Outs, Inited]),

    file:close(Fid).

to_var_size_list(Vars, L) -> [{X, proplists:get_value(size, dict:fetch(X, Vars))} || X <- L].

c_type(float32) -> float;
c_type(float16) -> int16_t;
c_type(int32) -> int32_t;
c_type(int8) -> int8_t;
c_type(int64) -> int64_t;
c_type(bool) -> bool;
c_type(int16) -> int16_t.

fmt_char(int8_t) -> d;
fmt_char(int16_t) -> d;
fmt_char(int32_t) -> d;
fmt_char(int64_t) -> lld;
fmt_char(bool) -> d;
fmt_char(float) -> f.

sizeof(float32) -> 4;
sizeof(float16) -> 2;
sizeof(int32) -> 4;
sizeof(int8) -> 1;
sizeof(int64) -> 8;
sizeof(bool) -> 1;
sizeof(int16) -> 2;
sizeof(complex64) -> 8;
sizeof({tensor, _Name, Type, Shape, _Opts}) -> sizeof(Type) * prod(Shape).

prod(Ls) -> lists:foldl(fun (A, Prod) -> A * Prod end, 1, Ls).

get_intermidiates(_Op, _Vars) ->
    [].

get_aliases({op, reshape, OpIns, OpOuts, _Opts} = _Op, _Vars) ->
    OpIns ++ OpOuts;
get_aliases(_Op, _Vars) ->
    [].

gen_key(K, S) ->
    case sets:is_element(K, S) of
        true -> gen_key(K ++ "!", S, 1);
        _ -> {K, S}
    end.

gen_key(K, S, N) ->
    K2 = K ++ integer_to_list(N),
    case sets:is_element(K2, S) of
        true -> gen_key(K, S, N + 1);
        _ -> {K2, sets:add_element(K2, S)}
    end.

load_alloc_result(Path, Id) ->
    {ok, [Schedules]} = file:consult(filename:join([Path, "schedule"])),
    Schedule = dict:fetch(Id, dict:from_list(Schedules)),
    {ok, [{size, Size}, {map, Addresses} | _]} = file:consult(filename:join([Path, integer_to_list(Id) ++ ".done"])),
    {Schedule, Size, Addresses}.

test() ->
    compile("./examples/conv_graph.emodel", "/tmp/dl_proj").