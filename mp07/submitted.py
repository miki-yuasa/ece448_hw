"""
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
"""

import copy, queue
from typing import TypeAlias, TypedDict

Proposition: TypeAlias = list[str | bool]


class RuleDict(TypedDict):
    antecedents: list[Proposition]
    consequent: Proposition
    text: str


def standardize_variables(
    nonstandard_rules: dict[str, RuleDict]
) -> tuple[dict[str, RuleDict], list[str]]:
    """
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).

    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    """

    standardized_rules: dict[str, RuleDict] = {}

    var_counter: int = 0
    variables: list[str] = []

    for (rule, items) in nonstandard_rules.items():
        new_antecedents: list[Proposition] = []
        new_consequent: Proposition = []

        new_var: str = "x{:0=4}".format(var_counter)
        is_new_var_used: bool = False

        if items["antecedents"]:

            for antecedent in items["antecedents"]:
                new_antecedent: Proposition = [
                    new_var if item == "something" else item for item in antecedent
                ]

                if new_var in new_antecedent:
                    is_new_var_used = True
                else:
                    pass

                new_antecedents.append(new_antecedent)

        else:
            pass

        if items["consequent"]:
            new_consequent: Proposition = [
                new_var if item == "something" else item for item in items["consequent"]
            ]

            if new_var in new_consequent:
                is_new_var_used = True
            else:
                pass
        else:
            pass

        standardized_rules[rule] = {
            "antecedents": new_antecedents,
            "consequent": new_consequent,
            "text": items["text"],
        }

        if is_new_var_used:
            variables.append(new_var)
            var_counter += 1
        else:
            pass

    return standardized_rules, variables


def unify(query, datum, variables):
    """
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.

    Unification succeeds if (1) every variable x in the unified query is replaced by a
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is
      detected, the query is changed to ['bobcat','eats','bobcat',True], which
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the
      rest of the contents of the query or datum.
    """
    raise RuntimeError("You need to write this part!")
    return unification, subs


def apply(rule, goals, variables):
    """
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.

    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).
       If every one of the goals can be unified with the rule consequent, then
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with
       applications[i]['consequent'] has been removed, and replaced by
       the members of applications[i]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True],
        ['bald eagle','eats','squirrel',False]
        ['bobcat','is','nice',True],
        ['bobcat','is','hungry',False]
      ],[
        ['bobcat','eats','squirrel',False]
        ['bobcat','visits','squirrel',True],
        ['bald eagle','is','nice',True],
        ['bald eagle','is','hungry',False]
      ]
    """
    raise RuntimeError("You need to write this part!")
    return applications, goalsets


def backward_chain(query, rules, variables):
    """
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    """
    raise RuntimeError("You need to write this part!")
    return proof
