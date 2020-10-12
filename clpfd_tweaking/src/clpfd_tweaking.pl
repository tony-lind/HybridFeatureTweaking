/* -*- Mode:Prolog; coding:iso-8859-1; indent-tabs-mode:nil; prolog-indent-width:8; prolog-paren-indent:4; tab-width:8; -*- */
%
% CLPFD tweaking 
%
% Author: anonymous

:- module(clpfd_tweaking, [tweak_example/12, call_tweaking/6]). %, tweak_example/11, rf/1, call_tweaking/6]). 
:- use_module([library(clpfd), library(fdbg), library(lists), library(random)]). %, library(timeout) 
%:- attribute index/1.

%verify_attributes(_,_,[]).

call_tweaking(Forest, ParamsList, Outcome, Cost, Trees, CorrectXVs):-
        get_params(ParamsList, Xs, Class, TimeBudget, MinMaxVals, DirectionList, ProbThreshold, CostList),!,
        catch(tweak_example(Forest, Xs, Class, TimeBudget, MinMaxVals, DirectionList, ProbThreshold, CostList, Outcome1, Cost, Trees, XVs), error(resource_error(ResourceType), _), exception_handler),
        %write(user_error, Outcome1),nl(user_error),
        (nonvar(ResourceType) ->
          CorrectXVs = [error],
          Outcome = error
        ;         
          (Outcome1 = time_out -> 
            CorrectXVs = [time_out],
            Outcome = time_out
          ; 
           %get_correct_values(XVs, Decimals, CorrectXVs), 
            CorrectXVs = XVs,
            Outcome1 = Outcome
          )
        ).

exception_handler:-
        write(user_error, 'we run out of memory '), nl(user_error).
      
get_params(ParamsList, Xs, Class, TimeBudget, MinMaxVals, DirectionList, ProbThreshold, CostList):-
        member(ex=Xs, ParamsList),
        member(whish_c=Class, ParamsList),
        member(time=TimeBudget, ParamsList),
        member(min_max=MinMaxVals, ParamsList),
        %member(decimals=Decimals, ParamsList), 
        member(direction=DirectionList, ParamsList),
        member(probability=ProbThreshold, ParamsList),
        member(cost_f=CostList, ParamsList).
        
 
% this is just a test case, should come from python
% balance_scale with 3 trees 
%rf([[[[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,>,2],[3,>,2],[2,>,3],[0,>,3],[1,>,2],[class,=,0]],[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,>,2],[3,>,2],[2,<=,3],[3,>,3],[1,>,3],[0,<=,3],[class,=,0]],[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,>,2],[3,<=,2],[0,>,3],[1,<=,2],[class,=,0]],[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,<=,2],[1,<=,2],[2,>,1],[0,<=,3],[class,=,0]],[[1,>,1],[0,>,0],[3,>,1],[0,<=,2],[2,>,1],[0,>,1],[3,>,2],[2,<=,2],[1,>,2],[3,<=,3],[class,=,0]],[[1,>,1],[0,>,0],[3,>,1],[0,<=,2],[2,<=,1],[0,<=,1],[1,<=,2],[2,>,0],[3,<=,2],[class,=,0]],[[1,>,1],[0,>,0],[3,<=,1],[2,>,2],[1,>,2],[0,<=,1],[3,>,0],[2,<=,3],[class,=,0]],[[1,>,1],[0,<=,0],[3,>,0],[3,>,1],[2,<=,0],[1,>,3],[3,>,3],[class,=,0]],[[1,>,1],[0,<=,0],[3,>,0],[3,>,1],[2,<=,0],[1,<=,3],[3,<=,2],[class,=,0]],[[1,>,1],[0,<=,0],[3,<=,0],[2,>,3],[1,>,3],[class,=,0]],[[1,<=,1],[3,>,1],[0,>,1],[0,<=,3],[1,>,0],[3,<=,3],[2,>,0],[2,<=,1],[3,<=,2],[class,=,0]],[[1,<=,1],[3,>,1],[0,>,1],[0,<=,3],[1,<=,0],[3,<=,3],[0,>,2],[2,<=,1],[class,=,0]],[[1,<=,1],[3,>,1],[0,>,1],[0,<=,3],[1,<=,0],[3,<=,3],[0,<=,2],[3,<=,2],[2,<=,1],[class,=,0]],[[1,<=,1],[3,<=,1],[0,>,0],[3,>,0],[0,>,1],[2,>,2],[1,>,0],[0,>,3],[class,=,0]],[[1,<=,1],[3,<=,1],[0,<=,0],[3,>,0],[2,<=,0],[class,=,0]],[[1,<=,1],[3,<=,1],[0,<=,0],[3,<=,0],[1,<=,0],[2,<=,0],[class,=,0]]],[[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,>,2],[3,>,2],[2,>,3],[0,>,3],[1,<=,2],[class,=,1]],[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,>,2],[3,>,2],[2,>,3],[0,<=,3],[class,=,1]],[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,>,2],[3,>,2],[2,<=,3],[3,>,3],[1,<=,3],[class,=,1]],[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,>,2],[3,>,2],[2,<=,3],[3,<=,3],[0,>,3],[1,<=,2],[class,=,1]],[[1,>,1],[0,>,0],[3,>,1],[0,<=,2],[2,>,1],[0,>,1],[3,>,2],[2,>,2],[class,=,1]],[[1,>,1],[0,>,0],[3,>,1],[0,<=,2],[2,>,1],[0,>,1],[3,>,2],[2,<=,2],[1,>,2],[3,>,3],[class,=,1]],[[1,>,1],[0,>,0],[3,>,1],[0,<=,2],[2,>,1],[0,>,1],[3,>,2],[2,<=,2],[1,<=,2],[class,=,1]],[[1,>,1],[0,>,0],[3,>,1],[0,<=,2],[2,>,1],[0,>,1],[3,<=,2],[1,<=,2],[class,=,1]],[[1,>,1],[0,>,0],[3,>,1],[0,<=,2],[2,>,1],[0,<=,1],[class,=,1]],[[1,>,1],[0,>,0],[3,>,1],[0,<=,2],[2,<=,1],[0,<=,1],[1,<=,2],[2,>,0],[3,>,2],[class,=,1]],[[1,>,1],[0,>,0],[3,<=,1],[2,>,2],[1,>,2],[0,<=,1],[3,>,0],[2,>,3],[class,=,1]],[[1,>,1],[0,>,0],[3,<=,1],[2,>,2],[1,<=,2],[3,>,0],[2,>,3],[0,<=,2],[class,=,1]],[[1,>,1],[0,>,0],[3,<=,1],[2,>,2],[1,<=,2],[3,>,0],[2,<=,3],[0,<=,2],[class,=,1]],[[1,>,1],[0,<=,0],[3,>,0],[3,>,1],[2,>,0],[class,=,1]],[[1,>,1],[0,<=,0],[3,>,0],[3,>,1],[2,<=,0],[1,<=,3],[3,>,2],[class,=,1]],[[1,>,1],[0,<=,0],[3,>,0],[3,<=,1],[class,=,1]],[[1,>,1],[0,<=,0],[3,<=,0],[2,>,3],[1,<=,3],[class,=,1]],[[1,>,1],[0,<=,0],[3,<=,0],[2,<=,3],[1,<=,2],[class,=,1]],[[1,<=,1],[3,>,1],[0,>,1],[0,>,3],[3,>,3],[class,=,1]],[[1,<=,1],[3,>,1],[0,>,1],[0,>,3],[3,<=,3],[3,>,2],[1,>,0],[2,>,2],[class,=,1]],[[1,<=,1],[3,>,1],[0,>,1],[0,>,3],[3,<=,3],[3,>,2],[1,<=,0],[2,>,0],[class,=,1]],[[1,<=,1],[3,>,1],[0,>,1],[0,>,3],[3,<=,3],[3,<=,2],[1,>,0],[2,>,2],[class,=,1]],[[1,<=,1],[3,>,1],[0,>,1],[0,>,3],[3,<=,3],[3,<=,2],[1,<=,0],[class,=,1]],[[1,<=,1],[3,>,1],[0,>,1],[0,<=,3],[1,>,0],[3,>,3],[class,=,1]],[[1,<=,1],[3,>,1],[0,>,1],[0,<=,3],[1,>,0],[3,<=,3],[2,>,0],[2,>,1],[class,=,1]],[[1,<=,1],[3,>,1],[0,>,1],[0,<=,3],[1,>,0],[3,<=,3],[2,>,0],[2,<=,1],[3,>,2],[class,=,1]],[[1,<=,1],[3,>,1],[0,>,1],[0,<=,3],[1,<=,0],[3,>,3],[class,=,1]],[[1,<=,1],[3,>,1],[0,>,1],[0,<=,3],[1,<=,0],[3,<=,3],[0,>,2],[2,>,1],[class,=,1]],[[1,<=,1],[3,>,1],[0,>,1],[0,<=,3],[1,<=,0],[3,<=,3],[0,<=,2],[3,>,2],[class,=,1]],[[1,<=,1],[3,>,1],[0,>,1],[0,<=,3],[1,<=,0],[3,<=,3],[0,<=,2],[3,<=,2],[2,>,1],[class,=,1]],[[1,<=,1],[3,>,1],[0,<=,1],[class,=,1]],[[1,<=,1],[3,<=,1],[0,>,0],[3,>,0],[0,>,1],[2,>,2],[1,>,0],[0,<=,3],[class,=,1]],[[1,<=,1],[3,<=,1],[0,>,0],[3,>,0],[0,>,1],[2,>,2],[1,<=,0],[class,=,1]],[[1,<=,1],[3,<=,1],[0,>,0],[3,>,0],[0,>,1],[2,<=,2],[2,>,1],[1,<=,0],[class,=,1]],[[1,<=,1],[3,<=,1],[0,>,0],[3,>,0],[0,<=,1],[2,>,1],[class,=,1]],[[1,<=,1],[3,<=,1],[0,>,0],[3,<=,0],[2,>,3],[1,>,0],[0,<=,1],[class,=,1]],[[1,<=,1],[3,<=,1],[0,>,0],[3,<=,0],[2,>,3],[1,<=,0],[class,=,1]],[[1,<=,1],[3,<=,1],[0,<=,0],[3,>,0],[2,>,0],[class,=,1]],[[1,<=,1],[3,<=,1],[0,<=,0],[3,<=,0],[1,>,0],[2,>,1],[class,=,1]],[[1,<=,1],[3,<=,1],[0,<=,0],[3,<=,0],[1,<=,0],[2,>,0],[class,=,1]]],[[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,>,2],[3,>,2],[2,<=,3],[3,>,3],[1,>,3],[0,>,3],[class,=,2]],[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,>,2],[3,>,2],[2,<=,3],[3,<=,3],[0,>,3],[1,>,2],[class,=,2]],[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,>,2],[3,>,2],[2,<=,3],[3,<=,3],[0,<=,3],[class,=,2]],[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,>,2],[3,<=,2],[0,>,3],[1,>,2],[class,=,2]],[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,>,2],[3,<=,2],[0,<=,3],[class,=,2]],[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,<=,2],[1,>,2],[class,=,2]],[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,<=,2],[1,<=,2],[2,>,1],[0,>,3],[class,=,2]],[[1,>,1],[0,>,0],[3,>,1],[0,>,2],[2,<=,2],[1,<=,2],[2,<=,1],[class,=,2]],[[1,>,1],[0,>,0],[3,>,1],[0,<=,2],[2,>,1],[0,>,1],[3,<=,2],[1,>,2],[class,=,2]],[[1,>,1],[0,>,0],[3,>,1],[0,<=,2],[2,<=,1],[0,>,1],[class,=,2]],[[1,>,1],[0,>,0],[3,>,1],[0,<=,2],[2,<=,1],[0,<=,1],[1,>,2],[class,=,2]],[[1,>,1],[0,>,0],[3,>,1],[0,<=,2],[2,<=,1],[0,<=,1],[1,<=,2],[2,<=,0],[class,=,2]],[[1,>,1],[0,>,0],[3,<=,1],[2,>,2],[1,>,2],[0,>,1],[class,=,2]],[[1,>,1],[0,>,0],[3,<=,1],[2,>,2],[1,>,2],[0,<=,1],[3,<=,0],[class,=,2]],[[1,>,1],[0,>,0],[3,<=,1],[2,>,2],[1,<=,2],[3,>,0],[2,>,3],[0,>,2],[class,=,2]],[[1,>,1],[0,>,0],[3,<=,1],[2,>,2],[1,<=,2],[3,>,0],[2,<=,3],[0,>,2],[class,=,2]],[[1,>,1],[0,>,0],[3,<=,1],[2,>,2],[1,<=,2],[3,<=,0],[class,=,2]],[[1,>,1],[0,>,0],[3,<=,1],[2,<=,2],[class,=,2]],[[1,>,1],[0,<=,0],[3,>,0],[3,>,1],[2,<=,0],[1,>,3],[3,<=,3],[class,=,2]],[[1,>,1],[0,<=,0],[3,<=,0],[2,<=,3],[1,>,2],[class,=,2]],[[1,<=,1],[3,>,1],[0,>,1],[0,>,3],[3,<=,3],[3,>,2],[1,>,0],[2,<=,2],[class,=,2]],[[1,<=,1],[3,>,1],[0,>,1],[0,>,3],[3,<=,3],[3,>,2],[1,<=,0],[2,<=,0],[class,=,2]],[[1,<=,1],[3,>,1],[0,>,1],[0,>,3],[3,<=,3],[3,<=,2],[1,>,0],[2,<=,2],[class,=,2]],[[1,<=,1],[3,>,1],[0,>,1],[0,<=,3],[1,>,0],[3,<=,3],[2,<=,0],[class,=,2]],[[1,<=,1],[3,<=,1],[0,>,0],[3,>,0],[0,>,1],[2,<=,2],[2,>,1],[1,>,0],[class,=,2]],[[1,<=,1],[3,<=,1],[0,>,0],[3,>,0],[0,>,1],[2,<=,2],[2,<=,1],[class,=,2]],[[1,<=,1],[3,<=,1],[0,>,0],[3,>,0],[0,<=,1],[2,<=,1],[class,=,2]],[[1,<=,1],[3,<=,1],[0,>,0],[3,<=,0],[2,>,3],[1,>,0],[0,>,1],[class,=,2]],[[1,<=,1],[3,<=,1],[0,>,0],[3,<=,0],[2,<=,3],[class,=,2]],[[1,<=,1],[3,<=,1],[0,<=,0],[3,<=,0],[1,>,0],[2,<=,1],[class,=,2]]]],[[[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,>,3],[0,<=,3],[1,>,3],[2,<=,3],[class,=,0]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,<=,3],[0,>,3],[1,<=,3],[1,>,2],[class,=,0]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,<=,3],[0,<=,3],[1,>,2],[0,>,2],[2,>,2],[1,>,3],[class,=,0]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,<=,3],[0,<=,3],[1,>,2],[0,>,2],[2,>,2],[1,<=,3],[2,<=,3],[class,=,0]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,<=,3],[0,<=,3],[1,>,2],[0,<=,2],[2,<=,2],[class,=,0]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,<=,3],[0,<=,3],[1,<=,2],[0,>,2],[2,<=,3],[class,=,0]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,<=,2],[2,>,3],[1,>,3],[0,<=,2],[class,=,0]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,<=,2],[2,>,3],[1,<=,3],[0,>,3],[class,=,0]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,<=,2],[2,<=,3],[1,>,2],[0,<=,2],[1,<=,3],[2,>,2],[class,=,0]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,<=,2],[2,<=,3],[1,<=,2],[0,>,2],[class,=,0]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,<=,2],[2,<=,3],[1,<=,2],[0,<=,2],[2,<=,2],[class,=,0]],[[1,>,1],[3,>,1],[2,<=,1],[1,>,3],[0,<=,1],[2,>,0],[3,>,3],[class,=,0]],[[1,>,1],[3,>,1],[2,<=,1],[1,>,3],[0,<=,1],[2,<=,0],[0,<=,0],[3,>,3],[class,=,0]],[[1,>,1],[3,>,1],[2,<=,1],[1,<=,3],[0,<=,1],[0,>,0],[2,>,0],[3,<=,2],[class,=,0]],[[1,>,1],[3,>,1],[2,<=,1],[1,<=,3],[0,<=,1],[0,<=,0],[3,<=,2],[2,<=,0],[class,=,0]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,>,0],[1,>,3],[2,>,3],[0,>,0],[class,=,0]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,>,0],[1,<=,3],[1,>,2],[0,>,0],[class,=,0]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,>,0],[1,<=,3],[1,<=,2],[0,>,0],[2,<=,2],[class,=,0]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,<=,0],[0,<=,0],[1,>,3],[2,>,3],[class,=,0]],[[1,<=,1],[3,>,0],[3,>,2],[2,<=,1],[1,>,0],[0,>,2],[3,<=,3],[2,>,0],[class,=,0]],[[1,<=,1],[3,>,0],[3,>,2],[2,<=,1],[1,>,0],[0,<=,2],[0,<=,1],[0,>,0],[2,<=,0],[3,<=,3],[class,=,0]],[[1,<=,1],[3,>,0],[3,>,2],[2,<=,1],[1,<=,0],[2,<=,0],[3,<=,3],[0,>,2],[class,=,0]],[[1,<=,1],[3,>,0],[3,<=,2],[0,>,2],[2,>,2],[3,<=,1],[0,>,3],[2,>,3],[class,=,0]],[[1,<=,1],[3,>,0],[3,<=,2],[0,>,2],[2,>,2],[3,<=,1],[0,<=,3],[1,>,0],[class,=,0]],[[1,<=,1],[3,>,0],[3,<=,2],[0,<=,2],[1,>,0],[2,<=,1],[2,>,0],[0,>,0],[class,=,0]],[[1,<=,1],[3,<=,0],[1,>,0],[0,<=,1],[0,>,0],[2,>,1],[2,<=,3],[class,=,0]],[[1,<=,1],[3,<=,0],[1,<=,0],[0,>,0],[2,>,2],[class,=,0]],[[1,<=,1],[3,<=,0],[1,<=,0],[0,>,0],[2,<=,2],[2,<=,1],[2,>,0],[0,<=,2],[class,=,0]],[[1,<=,1],[3,<=,0],[1,<=,0],[0,<=,0],[2,<=,0],[class,=,0]]],[[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,>,3],[0,<=,3],[1,>,3],[2,>,3],[class,=,1]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,>,3],[0,<=,3],[1,<=,3],[class,=,1]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,<=,3],[0,>,3],[1,<=,3],[1,<=,2],[2,>,2],[class,=,1]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,<=,3],[0,<=,3],[1,>,2],[0,>,2],[2,>,2],[1,<=,3],[2,>,3],[class,=,1]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,<=,3],[0,<=,3],[1,>,2],[0,<=,2],[2,>,2],[class,=,1]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,<=,3],[0,<=,3],[1,<=,2],[0,>,2],[2,>,3],[class,=,1]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,<=,3],[0,<=,3],[1,<=,2],[0,<=,2],[class,=,1]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,<=,2],[2,>,3],[1,<=,3],[0,<=,3],[class,=,1]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,<=,2],[2,<=,3],[1,<=,2],[0,<=,2],[2,>,2],[class,=,1]],[[1,>,1],[3,>,1],[2,>,1],[0,<=,1],[3,>,2],[class,=,1]],[[1,>,1],[3,>,1],[2,>,1],[0,<=,1],[3,<=,2],[1,>,3],[0,>,0],[2,>,3],[class,=,1]],[[1,>,1],[3,>,1],[2,>,1],[0,<=,1],[3,<=,2],[1,>,3],[0,<=,0],[class,=,1]],[[1,>,1],[3,>,1],[2,>,1],[0,<=,1],[3,<=,2],[1,<=,3],[class,=,1]],[[1,>,1],[3,>,1],[2,<=,1],[1,<=,3],[0,<=,1],[0,>,0],[2,>,0],[3,>,2],[class,=,1]],[[1,>,1],[3,>,1],[2,<=,1],[1,<=,3],[0,<=,1],[0,<=,0],[3,>,2],[class,=,1]],[[1,>,1],[3,>,1],[2,<=,1],[1,<=,3],[0,<=,1],[0,<=,0],[3,<=,2],[2,>,0],[class,=,1]],[[1,>,1],[3,<=,1],[0,>,1],[2,>,3],[1,<=,2],[3,>,0],[class,=,1]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,>,0],[1,>,3],[2,>,3],[0,<=,0],[class,=,1]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,>,0],[1,>,3],[2,<=,3],[0,<=,0],[class,=,1]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,>,0],[1,<=,3],[1,>,2],[0,<=,0],[class,=,1]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,>,0],[1,<=,3],[1,<=,2],[0,>,0],[2,>,2],[class,=,1]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,>,0],[1,<=,3],[1,<=,2],[0,<=,0],[class,=,1]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,<=,0],[0,<=,0],[1,<=,3],[2,>,3],[class,=,1]],[[1,<=,1],[3,>,0],[3,>,2],[2,>,1],[class,=,1]],[[1,<=,1],[3,>,0],[3,>,2],[2,<=,1],[1,>,0],[0,<=,2],[0,>,1],[2,>,0],[class,=,1]],[[1,<=,1],[3,>,0],[3,>,2],[2,<=,1],[1,>,0],[0,<=,2],[0,<=,1],[0,>,0],[2,>,0],[class,=,1]],[[1,<=,1],[3,>,0],[3,>,2],[2,<=,1],[1,>,0],[0,<=,2],[0,<=,1],[0,>,0],[2,<=,0],[3,>,3],[class,=,1]],[[1,<=,1],[3,>,0],[3,>,2],[2,<=,1],[1,>,0],[0,<=,2],[0,<=,1],[0,<=,0],[class,=,1]],[[1,<=,1],[3,>,0],[3,>,2],[2,<=,1],[1,<=,0],[2,>,0],[class,=,1]],[[1,<=,1],[3,>,0],[3,>,2],[2,<=,1],[1,<=,0],[2,<=,0],[3,>,3],[class,=,1]],[[1,<=,1],[3,>,0],[3,>,2],[2,<=,1],[1,<=,0],[2,<=,0],[3,<=,3],[0,<=,2],[class,=,1]],[[1,<=,1],[3,>,0],[3,<=,2],[0,>,2],[2,>,2],[3,>,1],[class,=,1]],[[1,<=,1],[3,>,0],[3,<=,2],[0,>,2],[2,>,2],[3,<=,1],[0,>,3],[2,<=,3],[class,=,1]],[[1,<=,1],[3,>,0],[3,<=,2],[0,>,2],[2,>,2],[3,<=,1],[0,<=,3],[1,<=,0],[class,=,1]],[[1,<=,1],[3,>,0],[3,<=,2],[0,>,2],[2,<=,2],[2,>,0],[0,>,3],[1,<=,0],[2,>,1],[class,=,1]],[[1,<=,1],[3,>,0],[3,<=,2],[0,>,2],[2,<=,2],[2,>,0],[0,>,3],[1,<=,0],[2,<=,1],[3,>,1],[class,=,1]],[[1,<=,1],[3,>,0],[3,<=,2],[0,<=,2],[1,>,0],[2,>,1],[class,=,1]],[[1,<=,1],[3,>,0],[3,<=,2],[0,<=,2],[1,>,0],[2,<=,1],[2,>,0],[0,<=,0],[class,=,1]],[[1,<=,1],[3,>,0],[3,<=,2],[0,<=,2],[1,>,0],[2,<=,1],[2,<=,0],[3,>,1],[0,<=,0],[class,=,1]],[[1,<=,1],[3,>,0],[3,<=,2],[0,<=,2],[1,<=,0],[class,=,1]],[[1,<=,1],[3,<=,0],[1,>,0],[0,<=,1],[0,>,0],[2,>,1],[2,>,3],[class,=,1]],[[1,<=,1],[3,<=,0],[1,>,0],[0,<=,1],[0,<=,0],[class,=,1]],[[1,<=,1],[3,<=,0],[1,<=,0],[0,<=,0],[2,>,0],[class,=,1]]],[[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,>,3],[0,>,3],[class,=,2]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,<=,3],[0,>,3],[1,>,3],[class,=,2]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,<=,3],[0,>,3],[1,<=,3],[1,<=,2],[2,<=,2],[class,=,2]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,>,2],[3,<=,3],[0,<=,3],[1,>,2],[0,>,2],[2,<=,2],[class,=,2]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,<=,2],[2,>,3],[1,>,3],[0,>,2],[class,=,2]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,<=,2],[2,<=,3],[1,>,2],[0,>,2],[class,=,2]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,<=,2],[2,<=,3],[1,>,2],[0,<=,2],[1,>,3],[class,=,2]],[[1,>,1],[3,>,1],[2,>,1],[0,>,1],[3,<=,2],[2,<=,3],[1,>,2],[0,<=,2],[1,<=,3],[2,<=,2],[class,=,2]],[[1,>,1],[3,>,1],[2,>,1],[0,<=,1],[3,<=,2],[1,>,3],[0,>,0],[2,<=,3],[class,=,2]],[[1,>,1],[3,>,1],[2,<=,1],[1,>,3],[0,>,1],[class,=,2]],[[1,>,1],[3,>,1],[2,<=,1],[1,>,3],[0,<=,1],[2,>,0],[3,<=,3],[class,=,2]],[[1,>,1],[3,>,1],[2,<=,1],[1,>,3],[0,<=,1],[2,<=,0],[0,>,0],[class,=,2]],[[1,>,1],[3,>,1],[2,<=,1],[1,>,3],[0,<=,1],[2,<=,0],[0,<=,0],[3,<=,3],[class,=,2]],[[1,>,1],[3,>,1],[2,<=,1],[1,<=,3],[0,>,1],[class,=,2]],[[1,>,1],[3,>,1],[2,<=,1],[1,<=,3],[0,<=,1],[0,>,0],[2,<=,0],[class,=,2]],[[1,>,1],[3,<=,1],[0,>,1],[2,>,3],[1,>,2],[class,=,2]],[[1,>,1],[3,<=,1],[0,>,1],[2,>,3],[1,<=,2],[3,<=,0],[class,=,2]],[[1,>,1],[3,<=,1],[0,>,1],[2,<=,3],[class,=,2]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,>,0],[1,>,3],[2,<=,3],[0,>,0],[class,=,2]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,<=,0],[0,>,0],[class,=,2]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,<=,0],[0,<=,0],[1,>,3],[2,<=,3],[class,=,2]],[[1,>,1],[3,<=,1],[0,<=,1],[2,>,1],[3,<=,0],[0,<=,0],[1,<=,3],[2,<=,3],[class,=,2]],[[1,>,1],[3,<=,1],[0,<=,1],[2,<=,1],[class,=,2]],[[1,<=,1],[3,>,0],[3,>,2],[2,<=,1],[1,>,0],[0,>,2],[3,>,3],[class,=,2]],[[1,<=,1],[3,>,0],[3,>,2],[2,<=,1],[1,>,0],[0,>,2],[3,<=,3],[2,<=,0],[class,=,2]],[[1,<=,1],[3,>,0],[3,>,2],[2,<=,1],[1,>,0],[0,<=,2],[0,>,1],[2,<=,0],[class,=,2]],[[1,<=,1],[3,>,0],[3,<=,2],[0,>,2],[2,<=,2],[2,>,0],[0,>,3],[1,>,0],[class,=,2]],[[1,<=,1],[3,>,0],[3,<=,2],[0,>,2],[2,<=,2],[2,>,0],[0,>,3],[1,<=,0],[2,<=,1],[3,<=,1],[class,=,2]],[[1,<=,1],[3,>,0],[3,<=,2],[0,>,2],[2,<=,2],[2,>,0],[0,<=,3],[class,=,2]],[[1,<=,1],[3,>,0],[3,<=,2],[0,>,2],[2,<=,2],[2,<=,0],[class,=,2]],[[1,<=,1],[3,>,0],[3,<=,2],[0,<=,2],[1,>,0],[2,<=,1],[2,<=,0],[3,>,1],[0,>,0],[class,=,2]],[[1,<=,1],[3,>,0],[3,<=,2],[0,<=,2],[1,>,0],[2,<=,1],[2,<=,0],[3,<=,1],[class,=,2]],[[1,<=,1],[3,<=,0],[1,>,0],[0,>,1],[class,=,2]],[[1,<=,1],[3,<=,0],[1,>,0],[0,<=,1],[0,>,0],[2,<=,1],[class,=,2]],[[1,<=,1],[3,<=,0],[1,<=,0],[0,>,0],[2,<=,2],[2,>,1],[class,=,2]],[[1,<=,1],[3,<=,0],[1,<=,0],[0,>,0],[2,<=,2],[2,<=,1],[2,>,0],[0,>,2],[class,=,2]],[[1,<=,1],[3,<=,0],[1,<=,0],[0,>,0],[2,<=,2],[2,<=,1],[2,<=,0],[class,=,2]]]],[[[[1,>,1],[3,>,2],[2,>,2],[0,>,2],[1,>,3],[3,>,3],[0,<=,3],[2,<=,3],[class,=,0]],[[1,>,1],[3,>,2],[2,>,2],[0,>,2],[1,<=,3],[1,>,2],[2,>,3],[0,>,3],[class,=,0]],[[1,>,1],[3,>,2],[2,>,2],[0,>,2],[1,<=,3],[1,>,2],[2,<=,3],[3,<=,3],[class,=,0]],[[1,>,1],[3,>,2],[2,<=,2],[1,>,3],[3,>,3],[2,<=,1],[2,>,0],[0,<=,1],[class,=,0]],[[1,>,1],[3,>,2],[2,<=,2],[1,>,3],[3,>,3],[2,<=,1],[2,<=,0],[0,<=,1],[class,=,0]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,>,1],[3,<=,3],[2,>,1],[0,<=,3],[1,>,2],[class,=,0]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,>,1],[1,>,2],[0,>,1],[0,<=,2],[2,>,3],[class,=,0]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,>,1],[1,<=,2],[0,>,2],[0,>,3],[2,>,3],[class,=,0]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,>,1],[1,<=,2],[0,>,2],[0,<=,3],[class,=,0]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,<=,1],[1,>,2],[1,>,3],[2,>,3],[0,<=,1],[class,=,0]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,<=,1],[1,>,2],[1,<=,3],[0,<=,1],[class,=,0]],[[1,>,1],[3,<=,2],[3,>,0],[0,<=,0],[1,<=,3],[3,>,1],[2,<=,0],[class,=,0]],[[1,<=,1],[0,>,1],[1,>,0],[3,>,0],[3,<=,2],[2,>,2],[3,<=,1],[2,>,3],[class,=,0]],[[1,<=,1],[0,>,1],[1,>,0],[3,>,0],[3,<=,2],[2,<=,2],[2,<=,1],[2,>,0],[0,<=,2],[class,=,0]],[[1,<=,1],[0,>,1],[1,<=,0],[3,>,0],[0,>,2],[2,<=,0],[0,<=,3],[3,<=,3],[3,>,2],[class,=,0]],[[1,<=,1],[0,>,1],[1,<=,0],[3,>,0],[0,<=,2],[2,<=,1],[3,<=,2],[class,=,0]],[[1,<=,1],[0,>,1],[1,<=,0],[3,<=,0],[2,>,3],[0,>,3],[class,=,0]],[[1,<=,1],[0,<=,1],[0,>,0],[1,>,0],[2,>,0],[3,<=,1],[2,>,1],[3,<=,0],[2,<=,3],[class,=,0]],[[1,<=,1],[0,<=,1],[0,>,0],[1,>,0],[2,>,0],[3,<=,1],[2,<=,1],[class,=,0]],[[1,<=,1],[0,<=,1],[0,>,0],[1,>,0],[2,<=,0],[3,>,1],[3,<=,3],[class,=,0]],[[1,<=,1],[0,<=,1],[0,>,0],[1,<=,0],[3,<=,1],[2,<=,2],[3,>,0],[class,=,0]],[[1,<=,1],[0,<=,1],[0,>,0],[1,<=,0],[3,<=,1],[2,<=,2],[3,<=,0],[2,>,0],[class,=,0]],[[1,<=,1],[0,<=,1],[0,<=,0],[1,>,0],[2,<=,0],[3,<=,2],[class,=,0]]],[[[1,>,1],[3,>,2],[2,>,2],[0,>,2],[1,>,3],[3,>,3],[0,<=,3],[2,>,3],[class,=,1]],[[1,>,1],[3,>,2],[2,>,2],[0,>,2],[1,<=,3],[1,>,2],[2,>,3],[0,<=,3],[class,=,1]],[[1,>,1],[3,>,2],[2,>,2],[0,>,2],[1,<=,3],[1,>,2],[2,<=,3],[3,>,3],[class,=,1]],[[1,>,1],[3,>,2],[2,>,2],[0,>,2],[1,<=,3],[1,<=,2],[class,=,1]],[[1,>,1],[3,>,2],[2,>,2],[0,<=,2],[class,=,1]],[[1,>,1],[3,>,2],[2,<=,2],[1,>,3],[3,<=,3],[2,>,0],[0,<=,1],[class,=,1]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,>,1],[3,>,3],[1,>,2],[2,>,1],[0,<=,2],[class,=,1]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,>,1],[3,>,3],[1,<=,2],[0,>,2],[2,>,1],[class,=,1]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,>,1],[3,>,3],[1,<=,2],[0,<=,2],[2,>,0],[class,=,1]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,>,1],[3,<=,3],[2,>,1],[0,<=,3],[1,<=,2],[class,=,1]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,<=,1],[2,>,0],[class,=,1]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,<=,1],[2,<=,0],[1,>,2],[class,=,1]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,<=,1],[2,<=,0],[1,<=,2],[3,>,3],[class,=,1]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,<=,1],[2,<=,0],[1,<=,2],[3,<=,3],[0,<=,0],[class,=,1]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,>,1],[1,>,2],[0,<=,1],[class,=,1]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,>,1],[1,<=,2],[0,<=,2],[class,=,1]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,<=,2],[1,<=,2],[3,>,1],[0,<=,1],[2,>,1],[class,=,1]],[[1,>,1],[3,<=,2],[3,>,0],[0,<=,0],[1,>,3],[2,>,1],[class,=,1]],[[1,>,1],[3,<=,2],[3,>,0],[0,<=,0],[1,<=,3],[3,>,1],[2,>,0],[class,=,1]],[[1,>,1],[3,<=,2],[3,>,0],[0,<=,0],[1,<=,3],[3,<=,1],[2,>,1],[class,=,1]],[[1,<=,1],[0,>,1],[1,>,0],[3,>,0],[3,>,2],[2,>,0],[class,=,1]],[[1,<=,1],[0,>,1],[1,>,0],[3,>,0],[3,<=,2],[2,>,2],[3,>,1],[class,=,1]],[[1,<=,1],[0,>,1],[1,>,0],[3,>,0],[3,<=,2],[2,>,2],[3,<=,1],[2,<=,3],[0,<=,3],[class,=,1]],[[1,<=,1],[0,>,1],[1,<=,0],[3,>,0],[0,>,2],[2,>,0],[class,=,1]],[[1,<=,1],[0,>,1],[1,<=,0],[3,>,0],[0,>,2],[2,<=,0],[0,<=,3],[3,>,3],[class,=,1]],[[1,<=,1],[0,>,1],[1,<=,0],[3,>,0],[0,<=,2],[2,>,1],[class,=,1]],[[1,<=,1],[0,>,1],[1,<=,0],[3,>,0],[0,<=,2],[2,<=,1],[3,>,2],[class,=,1]],[[1,<=,1],[0,>,1],[1,<=,0],[3,<=,0],[2,>,3],[0,<=,3],[class,=,1]],[[1,<=,1],[0,<=,1],[0,>,0],[1,>,0],[2,>,0],[3,>,1],[class,=,1]],[[1,<=,1],[0,<=,1],[0,>,0],[1,>,0],[2,>,0],[3,<=,1],[2,>,1],[3,>,0],[class,=,1]],[[1,<=,1],[0,<=,1],[0,>,0],[1,>,0],[2,>,0],[3,<=,1],[2,>,1],[3,<=,0],[2,>,3],[class,=,1]],[[1,<=,1],[0,<=,1],[0,>,0],[1,>,0],[2,<=,0],[3,>,1],[3,>,3],[class,=,1]],[[1,<=,1],[0,<=,1],[0,>,0],[1,<=,0],[3,>,1],[class,=,1]],[[1,<=,1],[0,<=,1],[0,>,0],[1,<=,0],[3,<=,1],[2,>,2],[class,=,1]],[[1,<=,1],[0,<=,1],[0,<=,0],[1,>,0],[2,>,0],[class,=,1]],[[1,<=,1],[0,<=,1],[0,<=,0],[1,>,0],[2,<=,0],[3,>,2],[class,=,1]],[[1,<=,1],[0,<=,1],[0,<=,0],[1,<=,0],[class,=,1]]],[[[1,>,1],[3,>,2],[2,>,2],[0,>,2],[1,>,3],[3,>,3],[0,>,3],[class,=,2]],[[1,>,1],[3,>,2],[2,>,2],[0,>,2],[1,>,3],[3,<=,3],[class,=,2]],[[1,>,1],[3,>,2],[2,<=,2],[1,>,3],[3,>,3],[2,>,1],[class,=,2]],[[1,>,1],[3,>,2],[2,<=,2],[1,>,3],[3,>,3],[2,<=,1],[2,>,0],[0,>,1],[class,=,2]],[[1,>,1],[3,>,2],[2,<=,2],[1,>,3],[3,>,3],[2,<=,1],[2,<=,0],[0,>,1],[class,=,2]],[[1,>,1],[3,>,2],[2,<=,2],[1,>,3],[3,<=,3],[2,>,0],[0,>,1],[class,=,2]],[[1,>,1],[3,>,2],[2,<=,2],[1,>,3],[3,<=,3],[2,<=,0],[class,=,2]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,>,1],[3,>,3],[1,>,2],[2,>,1],[0,>,2],[class,=,2]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,>,1],[3,>,3],[1,>,2],[2,<=,1],[class,=,2]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,>,1],[3,>,3],[1,<=,2],[0,>,2],[2,<=,1],[class,=,2]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,>,1],[3,>,3],[1,<=,2],[0,<=,2],[2,<=,0],[class,=,2]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,>,1],[3,<=,3],[2,>,1],[0,>,3],[class,=,2]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,>,1],[3,<=,3],[2,<=,1],[class,=,2]],[[1,>,1],[3,>,2],[2,<=,2],[1,<=,3],[0,<=,1],[2,<=,0],[1,<=,2],[3,<=,3],[0,>,0],[class,=,2]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,>,1],[1,>,2],[0,>,1],[0,>,2],[class,=,2]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,>,1],[1,>,2],[0,>,1],[0,<=,2],[2,<=,3],[class,=,2]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,>,1],[1,<=,2],[0,>,2],[0,>,3],[2,<=,3],[class,=,2]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,<=,1],[1,>,2],[1,>,3],[2,>,3],[0,>,1],[class,=,2]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,<=,1],[1,>,2],[1,>,3],[2,<=,3],[class,=,2]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,<=,1],[1,>,2],[1,<=,3],[0,>,1],[class,=,2]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,>,2],[3,<=,1],[1,<=,2],[class,=,2]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,<=,2],[1,>,2],[class,=,2]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,<=,2],[1,<=,2],[3,>,1],[0,>,1],[class,=,2]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,<=,2],[1,<=,2],[3,>,1],[0,<=,1],[2,<=,1],[class,=,2]],[[1,>,1],[3,<=,2],[3,>,0],[0,>,0],[2,<=,2],[1,<=,2],[3,<=,1],[class,=,2]],[[1,>,1],[3,<=,2],[3,>,0],[0,<=,0],[1,>,3],[2,<=,1],[class,=,2]],[[1,>,1],[3,<=,2],[3,>,0],[0,<=,0],[1,<=,3],[3,<=,1],[2,<=,1],[class,=,2]],[[1,>,1],[3,<=,2],[3,<=,0],[class,=,2]],[[1,<=,1],[0,>,1],[1,>,0],[3,>,0],[3,>,2],[2,<=,0],[class,=,2]],[[1,<=,1],[0,>,1],[1,>,0],[3,>,0],[3,<=,2],[2,>,2],[3,<=,1],[2,<=,3],[0,>,3],[class,=,2]],[[1,<=,1],[0,>,1],[1,>,0],[3,>,0],[3,<=,2],[2,<=,2],[2,>,1],[class,=,2]],[[1,<=,1],[0,>,1],[1,>,0],[3,>,0],[3,<=,2],[2,<=,2],[2,<=,1],[2,>,0],[0,>,2],[class,=,2]],[[1,<=,1],[0,>,1],[1,>,0],[3,>,0],[3,<=,2],[2,<=,2],[2,<=,1],[2,<=,0],[class,=,2]],[[1,<=,1],[0,>,1],[1,>,0],[3,<=,0],[class,=,2]],[[1,<=,1],[0,>,1],[1,<=,0],[3,>,0],[0,>,2],[2,<=,0],[0,>,3],[class,=,2]],[[1,<=,1],[0,>,1],[1,<=,0],[3,>,0],[0,>,2],[2,<=,0],[0,<=,3],[3,<=,3],[3,<=,2],[class,=,2]],[[1,<=,1],[0,>,1],[1,<=,0],[3,<=,0],[2,<=,3],[class,=,2]],[[1,<=,1],[0,<=,1],[0,>,0],[1,>,0],[2,<=,0],[3,<=,1],[class,=,2]],[[1,<=,1],[0,<=,1],[0,>,0],[1,<=,0],[3,<=,1],[2,<=,2],[3,<=,0],[2,<=,0],[class,=,2]]]]]).


% tweak_example([[1, 0, 0, 1], [2, 0, 1, 1]], 0, 15000, [[0, 4], [0, 4], [0, 4], [0, 4]], [up, up, up, up], 51, [1, 1, 1, 1], Outcome, Cost, Trees, XVs).
%tweak_example(Xs, Class, TimeBudget, MinMaxVals, DirList, ProbThreshold, CostList, Outcome, Cost, Trees, XVs):-
%        rf(Forest),  
%        tweak_example(Forest, Xs, Class, TimeBudget, MinMaxVals, DirList, ProbThreshold, CostList, Outcome, Cost, Trees, XVs).

% No example found, start looking for tweking from example it self 
% needs to be updated
tweak_example(Forest, IntXs, Class, TimeBudget, MinMaxVals, [], ProbThreshold, CostList, Outcome, Cost, Trees, XVs):-
        length(IntXs, NoVars),
        length(Forest, NoTrees),
        %write(user_error, 'No Trees: '),write(user_error, NoTrees),nl(user_error), 
        TreePercent is 100 / NoTrees,
        %write(user_error, 'TreePercent: '),write(user_error, TreePercent),nl(user_error),
        find_no_trees(0, 0, TreePercent, ProbThreshold, MinTrees),
        %write(user_error, 'MinTrees: '),write(user_error, MinTrees),nl(user_error),
        %transform_ex(Xs, IntXs),       
        %write(user_error, 'Setup of problem started'),
        %nl(user_error),
        setup_variables(IntXs, MinMaxVals, XVs, 1, CostList, 0, MaxCost),                 
        %write_vars(XVs),      
        setup_trees_vars(NoTrees,Trees),
        setup_forest(Forest, XVs, Class, Trees),   
        %If sum times out, this indicates that that the problem is overconstrained, i.e. impossible to solve -> no tweaking can be done 
        (sum(Trees, #>=, MinTrees) ->
                get_scalar_p_const(NoVars, ConstList), 
                NewConstList = [1|ConstList],
                Cost in 0..MaxCost, 
                get_abs_diff_vars(XVs, IntXs, CostList, AbsDiffList),
                CostVarsList = [Cost|AbsDiffList],
                scalar_product(NewConstList,CostVarsList,#=,0),
                %write(user_error, 'Setup of problem finished'),
                %nl(user_error),
                %write(user_error, 'Starting labeling'),
                %nl(user_error),
                labeling([minimize(Cost), time_out(TimeBudget, Outcome), ffc, middle], XVs)
                %write(user_error, XVs),
                %nl(user_error)
        ;
                write(user_error, 'Fail to tweak example'), nl(user_error),  
                Outcome = time_out  
         ).
                   
% We have found a tweked example that we will start as basis for search in the direction towards the original example 
tweak_example(Forest, [StartIntXs, GoalIntXs], Class, TimeBudget, MinMaxVals, DirectionList, ProbThreshold, CostList, Outcome, Cost, Trees, XVs):-
        length(StartIntXs, NoVars),
        length(Forest, NoTrees),
        %write(user_error, 'No Trees: '),write(user_error, NoTrees),nl(user_error), 
        TreePercent is 100 / NoTrees,
        %write(user_error, 'TreePercent: '),write(user_error, TreePercent),nl(user_error),
        find_no_trees(0, 0, TreePercent, ProbThreshold, MinTrees),
        %write(user_error, 'MinTrees: '),write(user_error, MinTrees),nl(user_error),
        %transform_ex(StartXs, Decimals, StartIntXs),  
        %transform_ex(GoalXs, Decimals, GoalIntXs),      
        %write(user_error, 'Setup of problem started'),
        %nl(user_error),
        %write(user_error, 'StartIntXs: '),write(user_error, StartIntXs),nl(user_error),
        %write(user_error, 'GoalIntXs: '),write(user_error, GoalIntXs),nl(user_error),
        setup_start_goal_variables(StartIntXs, GoalIntXs, MinMaxVals, XVs, 1, CostList, 0, MaxCost),                 
        %write_vars(XVs),      
        %write(user_error, 'before setup trees_vars'),
        setup_trees_vars(NoTrees, Trees),
        %write(user_error, 'before setup forest'),
        setup_forest(Forest, XVs, Class, Trees),  
        %write(user_error, ' sum '), 
        %write(user_error, 'Trees: '), write(user_error, Trees),
        %write(user_error, ' MinTrees: '), write(user_error, MinTrees), 
        %If sum times out, this indicates that that the problem is overconstrained, i.e. impossible to solve -> no tweaking can be done         
        (sum(Trees, #>=, MinTrees) ->
                %write(user_error, 'get_scalar_p_const'),
                get_scalar_p_const(NoVars, ConstList), 
                NewConstList = [1|ConstList],
                Cost in 0..MaxCost, 
                %write(user_error, 'before get_abs_diff_vars'),
                get_abs_diff_vars(XVs, GoalIntXs, CostList, AbsDiffList),
                CostVarsList = [Cost|AbsDiffList],
                %write(user_error, 'before scalar_product'),
                scalar_product(NewConstList,CostVarsList,#=,0),
                %write(user_error, 'Setup of problem finished'),
                %nl(user_error),
                %write(user_error, 'Starting solving'),
                %nl(user_error),
                getDirection(DirectionList, XVs, UpXVs, DownXVs),
                %write(user_error, 'DirectionList: '),write(user_error, DirectionList),nl(user_error),
                solve([minimize(Cost),time_out(TimeBudget, Outcome)], [labeling([ffc, up], UpXVs), labeling([ffc, down], DownXVs)])
        ; 
                write(user_error, 'Fail to tweak example'), nl(user_error),  
                Outcome = time_out   
        ).
             
     
getDirection([], [], [], []).
getDirection([up|DLs], [X|XVs], [X|UpXVs], DownXVs):-
        getDirection(DLs, XVs, UpXVs, DownXVs).
getDirection([down|DLs], [X|XVs], UpXVs, [X|DownXVs]):-
        getDirection(DLs, XVs, UpXVs, DownXVs).

get_scalar_p_const(0, []).
get_scalar_p_const(NoVars, [-1|Rest]):-
        NewNoVars is NoVars - 1,
        get_scalar_p_const(NewNoVars, Rest).          

setup_trees_vars(0,[]).
setup_trees_vars(NoTrees,[Tree|Trees]):-
        Tree in 0..1,
        NewNoTrees is NoTrees - 1,
        setup_trees_vars(NewNoTrees,Trees).
            
get_abs_diff_vars([], [], [], []).
get_abs_diff_vars([Var|XVs], [Value|IntXs], [Cost|Cs], [AbsDiff|AbsDiffL]):-
        AbsDiff #= abs(Var - Value) * Cost,
        get_abs_diff_vars(XVs, IntXs, Cs, AbsDiffL).
        
find_no_trees(Percent, Trees, _, ProbThreshold, MinTrees):-
        Percent >= ProbThreshold,!, 
        MinTrees = Trees.
find_no_trees(TempPercent, TempTrees, TreePercent, ProbThreshold, MinTrees):-
        NewTempPercent is TempPercent + TreePercent,
        NewTempTrees is TempTrees + 1,!,
        find_no_trees(NewTempPercent, NewTempTrees, TreePercent, ProbThreshold, MinTrees).

setup_forest([], _, _, []).
setup_forest([Tree|Trees], XVs, Class, [TreeV|TreeVs]):-
        nth0(Class, Tree, ClassPaths),
        %NoPaths = length(ClassPaths),
        collect_paths(ClassPaths, XVs, PathsLiteralsList),
        bool_or(PathsLiteralsList, TreeV),
        setup_forest(Trees, XVs, Class, TreeVs).

collect_paths([], _, []).
collect_paths([Path|Paths], XVs, [PathLiterals|PathLs]):-
        collect_conditions(Path, XVs, ConditionLiterals),
        bool_and(ConditionLiterals, PathLiterals),
        collect_paths(Paths, XVs, PathLs).

collect_conditions([[class,=,_]], _, []).
collect_conditions([[VarI,>,Threshold]|Conds], XVs, [CondL|CLs]):-
        nth0(VarI, XVs, XV),
        XV #> Threshold #<=> CondL,
        collect_conditions(Conds, XVs, CLs).
collect_conditions([[VarI,<=,Threshold]|Conds], XVs, [CondL|CLs]):-
        nth0(VarI, XVs, XV),
        XV #=< Threshold #<=> CondL,
        collect_conditions(Conds, XVs, CLs).

setup_start_goal_variables([], [], [], [], _, [], Cost, Cost).
setup_start_goal_variables([StartIntVal|StartIntXs], [GoalIntVal|GoalIntXs], [[Min,Max]|MinMaxVs], [X|XVs], Index, [Cost|Cs], TCost, FCost):-
        Diff is abs(StartIntVal - GoalIntVal),
        (StartIntVal > GoalIntVal ->
                %write(user_error, 'Set up domain 1'),
                X in Min..StartIntVal,
                %put_atts(X, index(Index)),
                NewTCost is TCost + (Diff * Cost),
                NewIndex is Index + 1, 
                setup_start_goal_variables(StartIntXs, GoalIntXs, MinMaxVs, XVs, NewIndex, Cs, NewTCost, FCost)                   
        ;                
                (Diff = 0 ->
                        %write(user_error, 'Set up domain 2'),
                        X in StartIntVal..GoalIntVal,
                        NewIndex is Index + 1, 
                        setup_start_goal_variables(StartIntXs, GoalIntXs, MinMaxVs, XVs, NewIndex, Cs, TCost, FCost)
                ;      
                        %write(user_error, 'Set up domain 3'),
                        X in StartIntVal..Max,
                        %put_atts(X, index(Index)),
                        NewTCost is TCost + (Diff * Cost),
                        NewIndex is Index + 1, 
                        setup_start_goal_variables(StartIntXs, GoalIntXs, MinMaxVs, XVs, NewIndex, Cs, NewTCost, FCost)
       
                )
        ).

%fix below should work fine, evethoug a smaller domain could be setup at the possible expense of completness 
setup_variables([], [], [], _, [], Cost, Cost).        
setup_variables([Val|Xs], [[Min,Max]|MinMaxVs], [X|XVs], Index, [Cost|Cs],  TCost, FCost):-       
        LeftRange is Val - Min,
        RightRange is Max - Val,
        Diff is abs(LeftRange - RightRange),
        (LeftRange > RightRange -> 
                NMax is round(Max + Diff),
                X in Min..NMax
        ;                     
                NMin is round(Min - Diff),
                X in NMin..Max
        ),
        %put_atts(X, index(Index)),
        NewTCost is TCost + (Max * Cost),
        NewIndex is Index + 1, 
        setup_variables(Xs, MinMaxVs, XVs, NewIndex, Cs, NewTCost, FCost).
   