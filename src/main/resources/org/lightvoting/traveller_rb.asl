voted(0).
state(0, undefined).

!start.

+!start
    : >>state(0, S)
        <-
        -mychair(_);
        -mygroup(_);
        -state(0, S);
        +state(1, start);
        generic/print("start" );
        !group/joined;
        !nextcycle
    .

+!nextcycle
    <-
        update/cycle();
        !nextcycle
    .

+!group/joined
    : >>mygroup(Group) && >>state(1, S)
    <-  -state(1, S);
        +state(2, group/joined);
        generic/print( "group/joined");
        !vote/submitted
    : >>state(1, _)
    <-
        !start
    .

+!vote/submitted
    : >>(voted(N), N==0) && >>mychair(Chair) && >>state(2, S)
    <-
        generic/print( "submit vote ");
        -voted(0);
        submit/vote(Chair);
        -state(2, S);
        +state(3, vote/submitted);
        generic/print( "vote/submitted");
        !diss/submitted
    : >>state(2, _)
    <-
        !vote/submitted
    .

+!diss/submitted
    : >>result(Chair, Result) && >>state(3, S)
    <-
        generic/print(MyName, "submit diss for result", Result);
        submit/diss(Chair,Result);
        -state(3, S);
        +state(4, diss/submitted);
        generic/print(MyName, "diss/submitted")
    : >>state(3, _)
    <-
        !diss/submitted
    .

// TODO refine the following
// +state(0, undefined)
// -my/group(Group)
// !start

+!leftgroup
    : >>mygroup(G) && >>mychair(C)
    //: >>leavegroup(Broker)
    <-
        generic/print(MyName, " I needed to leave my group, need a new one");
        -mygroup(G);
        -mychair(C);
        +state(0, undefined);
        !start
    .

+!done()
    : >>stay(Broker)
    <-
        generic/print("I'm staying in the group")
    .
