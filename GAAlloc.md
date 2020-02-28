# Using Genetic Algorithm for Optimal Static Memory Allocation

In some systems, variables can be allocated in an offline or static manner, i.e. allocation
time, size and deallocation time of each variable are all known. So it's possible to
figure out a optimal static allocation that minimizes the peak memory usage. This is
essential for lots of embedded systems.

This is an Erlang module that uses genetic algorithm to achieve optimal static memory
allocation for a whole program.

## Data Driven Model

A program is composed by a list of functions. A program and a function have a list of
input and output variables.

This module uses a data driven model. We define two level of variables.

### Logic Variables

Logic variables have a one-to-one correspondence to user data. Or rather,
a logic variable can only be bound for one time, just the same as in Erlang.

For example, this is a function `transpose` that transposes its input matrix `M`:

```C
transpose(&M);
```

In our model, we will define a new logic variable `Mt` to identify the transposed
matrix of `M`:

```C
Mt = transpose(M);
```

But, indeed, `transpose` stores the result in the original variable `M`. So, We
introduce the physical level of variables to address this.

### Physical Variables

Physical variables maps memory locations. In our model, we
declare `M` and `Mt` are aliases to each other, which means that they map to the
same memory location and have the same life cycle, i.e., they are physically the same.

## Model Definition

Below is the formal definition for a program.

```Erlang
% program() must be defined after function()s
-type model() :: function() ... program()

-type program() :: {program, prog_inputs(), prog_output(), excludes()}.
-type prog_inputs() :: var_size_list().
-type prog_outputs() :: var_list().
-type exclues() :: var_list().

-type function() :: {func, func_name(), fun_inputs(), fun_outputs(), aliases()}.
-type fun_name() :: string().
-type fun_inputs() :: var_list().
-type fun_outputs() :: var_size_list().
-type aliases() :: [alias_list()].
-type alias_list() :: var_list().

-type var_list() :: [var_name()].
-type var_size_list() :: [{var_name(), size()}].

-type var_name() :: string().
```

### Examples

```Erlang
% f1 takes v1 as input, and output v2 and v5
{func, "f1", ["v1"], [{"v2", 10}, {"v5", 100}], []}.

% f2 takes v2 and v5 as inputs, and output v2' and v3.
% v2 and v2' are physically the same thing.
{func, "f2", ["v2", "v5"], [{"v2'", 10}, {"v3", 20}], [["v2", "v2'"]]}.

% f2 takes v2 and v3 as inputs, and output v4.
{func, "f3", ["v2'", "v3"], [{"v4", 100}], []}.

% our program takes v1 as input, and result is stored in v4.
% variable v1 is allocated elsewhere and excluded from this GA allocation procedure.
{program,
    [{"v1", 100}],
    ["v4"],
    ["v1"]}.
```

Save above program to a file (say "1.erl"). To optimize the allocation for this examples, simply call:

```Erlang
kdl_alloc:start("1.erl").
```

## Algorithm

This module loads the model, builds all possible function scheduling schemes, and
calculate a lower bound of memory requirement for each scheme. This lower bound might
be the infimum. Then it select the 5 scheduling schemes that have the least lower
bounds, and use genetic algorithm on each scheme to search for the optimal allocation,
i.e. memory requirement equals to the lower bound.

One of the key elements in GA is how to do the mutation and crossover. The mutation
and crossover used here are very simple.

* Mutation: the address is changed by an offset which following normal distribution.
* Crossover: the addresses of different variables are inherited from different parents.
