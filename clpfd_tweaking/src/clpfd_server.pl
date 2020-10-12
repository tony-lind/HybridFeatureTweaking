:- module(json_server, [start_clpfd_server/0]).

:- use_module([clpfd_tweaking, library(lists)]).
:- use_module(library(json), [json_write/3,json_read/2]).
%:- use_module(library(codesio), [write_term_to_codes/3,open_codes_stream/2]).

:- dynamic rf/1.
:- volatile rf/1.

%max memory usage -> 8 GB then restart
max_memory_usage(8000000000).

% Setup input and output
start_clpfd_server:-       
        current_input(In),
        current_output(Out),
        loop_commands(In,Out). 
       
% Usa input to recive commands
% shutdown server
loop_commands(In, Out):-
        (json_read(In, Command),
         %write(user_error, Command),
         Command=json([command=DoWhat, Data]), 
         write(user_error, DoWhat),
         write(user_error, '\n')) ->
                (
                        (handle(DoWhat, Data, Out, quit),
                         write(user_error, 'quitting server'),
                         write(user_error, '\n'),
                         !)
                        ;
                        (handle(DoWhat, Data, Out, continue),
                         %write(user_error, 'handeled command, continue looping'),
                         %write(user_error, '\n'),           
                         loop_commands(In, Out)
                        )
                ).

handle(quit, _, _, quit).
     %json_write(Out, json([result=quit]), [compact(true)]),
     %nl(Out),
     %flush_output(Out), halt.

handle(need_to_restart, _, Out, continue):-
     statistics(memory, [CurrM,_]),
     max_memory_usage(MaxM),
     %write(user_error, 'CurrM: '), write(user_error, CurrM), write(user_error, '\n'), 
     %write(user_error, 'MaxM: '), write(user_error, MaxM), write(user_error, '\n'), 
     (CurrM < MaxM ->    
        json_write(Out, json([result=ok]), [compact(true)])
     ;
        json_write(Out, json([result=restart]), [compact(true)])
     ),
     nl(Out),
     flush_output(Out).

handle(setup_problem, Data, Out, continue):-
     write(user_error, Data),
     %Extract rf description   
     arg(2, Data, RF),
     %write(user_error, RF),
     retractall(rf(_)), 
     assert(rf(RF)),
     json_write(Out, json([result=problem_setup_finished]), [compact(true)]),    
     nl(Out),
     flush_output(Out).

handle(tweak_example, Data, Out, continue):-
     %write(user_error, Data),   
     arg(2, Data, JsonParametersList),
     %write(user_error, JsonParametersList),
     JsonParametersList = [json(ParamsList)],
     %write(user_error, ParamsList),    
     rf(RF),
     %call_tweaking(RF, ParamsList, TweakedEx),
     call_tweaking(RF, ParamsList, Outcome, Cost, Trees, TweakedEx), 
     %write(user_error, TweakedEx), 
     
     ((Outcome = time_out; Outcome = error) ->
        flush_output(Out),         
        json_write(Out, json([outcome=Outcome, cost=Outcome, trees=Outcome, ex=Outcome]), [compact(true)])
     ; 
        flush_output(Out),         
        json_write(Out, json([outcome=Outcome, cost=Cost, trees=Trees, ex=TweakedEx]), [compact(true)])
     ),
     nl(Out),
     flush_output(Out).

handle(clean_problem, _, Out, continue):-
     json_write(Out, json([result=problem_cleaned]), [compact(true)]),
     retractall(rf(_)),
     nl(Out),
     flush_output(Out).

handle(Unknown_command, _, Out, continue):-
     json_write(Out, json([result=Unknown_command]), [compact(true)]),
     nl(Out),
     flush_output(Out).
     