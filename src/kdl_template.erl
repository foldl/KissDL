-module(kdl_template).

-export([run/2, run/3]).

-export([test/0]).

-define(tag_open, "{{").
-define(tag_close, "}}").

-record(item_text, {data}).
-record(item_if, {test, then = [], else = []}).
-record(item_call, {param}).
-record(item_value, {var}).

run(Bin, Dict) -> run(Bin, Dict, none).

run(Bin, Dict, Call) when is_binary(Bin) -> run(binary_to_list(Bin), Dict, Call);
run(Str, Dict, Call) when is_list(Str), is_map(Dict) ->
    {Tree, []} = parse(Str, []),
    exec(Tree, Dict, Call, []).

get_tmpl_value(K, Dict) ->
    V = case maps:get(K, Dict, undefined) of
        undefined when is_list(K) -> maps:get(list_to_atom(K), Dict);
        X when X =/= undefined -> X
    end,
    to_list(V).

get_tmpl_value(K, Dict, Def) ->
    V = case maps:get(K, Dict, undefined) of
        undefined when is_list(K) -> maps:get(list_to_atom(K), Dict, Def);
        undefined -> Def;
        X -> X
    end,
    to_list(V).

to_list(V) when not is_list(V) -> lists:flatten(io_lib:format("~p", [V]));
to_list(V) -> V.

exec([], _Dict, _Call, Acc) -> lists:flatten(Acc);
exec([#item_text{data = Data} | T], Dict, Call, Acc) -> exec(T, Dict, Call, [Data | Acc]);
exec([#item_call{param = Param} | T], Dict, Call, Acc) -> exec(T, Dict, Call, [Call(string:split(Param, " ", all)) | Acc]);
exec([#item_value{var = Var} | T], Dict, Call, Acc) -> exec(T, Dict, Call, [get_tmpl_value(Var, Dict) | Acc]);
exec([#item_if{test = Test, then = Then, else = Else} | T], Dict, Call, Acc) ->
    TestResult = case string:lexemes(string:trim(Test), " ") of
        [Word] -> get_tmpl_value(Word, Dict, false) == "true";
        [Word, "==", V] -> V == get_tmpl_value(Word, Dict);
        [Word, "!=", V] -> V /= get_tmpl_value(Word, Dict);
        [Word, "in" | Ws] -> lists:member(get_tmpl_value(Word, Dict), Ws);
        [Word, "not-in" | Ws] -> not lists:member(get_tmpl_value(Word, Dict), Ws)
    end,

    case TestResult of
        true -> exec(T, Dict, Call, exec(Then, Dict, Call, []) ++ Acc);
        _ ->    exec(T, Dict, Call, exec(Else, Dict, Call, []) ++ Acc)
    end.

parse([], Acc) -> {Acc, []};
parse(Str, Acc) ->
    case string:split(Str, ?tag_open) of
        [Str] -> {[#item_text{data = Str} | Acc], []};
        [Pre, L] ->
            Acc2 = case Pre of
                "" -> Acc;
                _ -> [#item_text{data = Pre} | Acc]
            end,
            TagL = ?tag_open ++ L,
            {Tag, Remain} = start_tag(TagL),
            case Tag of
                {"if", Test} ->
                    {If, Remain20} = parse_if(Test, Remain),
                    parse(Remain20, [If | Acc2]);
                "else" -> {Acc2, TagL};
                "endif" -> {Acc2, TagL};
                {"call", Param} -> parse(Remain, [#item_call{param = Param} | Acc2]);
                ValueTag -> parse(Remain, [#item_value{var = ValueTag} | Acc2])
            end
    end.

parse_if(Test, Body) ->
    {Then, Remain10} = parse(Body, []),
    {Tag, Remain20} = start_tag(Remain10),
    case Tag of
        "else" ->
            {Else, Remain30} = parse(Remain20, []),
            {"endif", Remain40} = start_tag(Remain30),
            {#item_if{test = Test, then = Then, else = Else}, Remain40};
        "endif" ->
            {#item_if{test = Test, then = Then}, Remain20}
    end.

start_tag(Str) ->
    true = lists:prefix(?tag_open, Str),
    case string:split(string:slice(Str, string:len(?tag_open)), ?tag_close) of
        [Tag, Remain] ->
            case string:lexemes(string:trim(Tag), [$ , $\r, $\n, "\r\n"]) of
                [X] -> {X, Remain};
                [A | T] -> {{A, lists:join(" ", T)}, Remain}
            end;
        [X] -> {X, ""}
    end.

test() -> run("abc{{if  v in a b c }}xxx{{b}} {{if a}}1 a_then 234 {{else}}a_else{{endif}}{{endif}}", #{v => "b", a => true, b => "lkjkjlj"}).