/**
 * @cond LICENSE
 * ######################################################################################
 * # LGPL License                                                                       #
 * #                                                                                    #
 * # This file is part of LightVoting by Sophie Dennisen.                               #
 * # Copyright (c) 2017, Sophie Dennisen (sophie.dennisen@tu-clausthal.de)              #
 * # This program is free software: you can redistribute it and/or modify               #
 * # it under the terms of the GNU Lesser General Public License as                     #
 * # published by the Free Software Foundation, either version 3 of the                 #
 * # License, or (at your option) any later version.                                    #
 * #                                                                                    #
 * # This program is distributed in the hope that it will be useful,                    #
 * # but WITHOUT ANY WARRANTY; without even the implied warranty of                     #
 * # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                      #
 * # GNU Lesser General Public License for more details.                                #
 * #                                                                                    #
 * # You should have received a copy of the GNU Lesser General Public License           #
 * # along with this program. If not, see http://www.gnu.org/licenses/                  #
 * ######################################################################################
 * @endcond
 */

package org.lightvoting.simulation.agent.random_iterative;


import cern.colt.matrix.tbit.BitVector;
import org.lightjason.agentspeak.action.binding.IAgentAction;
import org.lightjason.agentspeak.action.binding.IAgentActionFilter;
import org.lightjason.agentspeak.action.binding.IAgentActionName;
import org.lightjason.agentspeak.agent.IBaseAgent;
import org.lightjason.agentspeak.common.CCommon;
import org.lightjason.agentspeak.configuration.IAgentConfiguration;
import org.lightjason.agentspeak.generator.IBaseAgentGenerator;
import org.lightjason.agentspeak.language.CLiteral;
import org.lightjason.agentspeak.language.CRawTerm;
import org.lightjason.agentspeak.language.instantiable.plan.trigger.CTrigger;
import org.lightjason.agentspeak.language.instantiable.plan.trigger.ITrigger;
import org.lightvoting.simulation.environment.random_iterative.CEnvironmentRI;
import org.lightvoting.simulation.environment.random_iterative.CGroupRI;
import org.lightvoting.simulation.rule.CMinisumApproval;
import org.lightvoting.simulation.rule.CMinisumRanksum;
import org.lightvoting.simulation.statistics.EDataDB;

import java.io.InputStream;
import java.sql.SQLException;
import java.text.MessageFormat;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.stream.Stream;

// import org.lightvoting.simulation.statistics.CDataWriter;

// import org.lightjason.agentspeak.language.score.IAggregation;


/**
 * Created by sophie on 21.02.17.
 */

// annotation to mark the class that actions are inside
@IAgentAction
public final class CChairAgentRI extends IBaseAgent<CChairAgentRI>
{
    private static final long serialVersionUID = -2704070311180663530L;

    /**
     * name of chair
     */
    private String m_name;

    /**
     * environment
     */

    private CEnvironmentRI m_environment;

    /**
     * grouping algorithm: "RANDOM" or "COORDINATED"
     */

    private String m_grouping;

    private List<BitVector> m_bitVotes = Collections.synchronizedList( new LinkedList<>() );
    private List<List<Long>> m_cLinearOrders = Collections.synchronizedList( new LinkedList<>() );
    private List<CVotingAgentRI> m_voters = Collections.synchronizedList( new LinkedList<>() );
    private List<Double> m_dissList = Collections.synchronizedList( new LinkedList<>() );
    private List<CVotingAgentRI> m_dissVoters = Collections.synchronizedList( new LinkedList<>() );
    private int m_iteration;
   // private List<CVotingAgentRI> m_agents = Collections.synchronizedList( new LinkedList<>() );
    private boolean m_iterative;
    private String m_protocol;
    private double m_dissThreshold;
    private String m_fileName;
    private int m_run;
    private String m_conf;
    // private boolean m_dissStored;

    // counter for intermediate elections in coordinated grouping
    private int m_coorNum;

    private List<String> m_paths = new ArrayList();
    private List<Object> m_data = new ArrayList();
    private HashMap<String, Object> m_map = new HashMap<>();
    private int m_comsize;
    private int m_altnum;
    private int m_groupNum;
    private final double m_voteTimeout;
    private CGroupRI m_group;
    private ConcurrentHashMap<CVotingAgentRI, Double> m_dissMap = new ConcurrentHashMap<>();
    private CBrokerAgentRI m_broker;
    private ConcurrentHashMap<CVotingAgentRI, Double> m_newdissMap;
    private boolean m_removedGoalAdded;
    // private long m_dissCounter;
    private boolean m_waitingForDiss;

    private final String m_rule;
    private int m_sim;
    private HashMap<String,Float> m_dissMapStr = new HashMap<>();
    private List<String> m_votersStr = Collections.synchronizedList( new LinkedList<>() );;
    private BitVector m_comResultBV;


    // TODO merge ctors

    /**
     * constructor of the agent
     * @param p_configuration agent configuration of the agent generator
     * @param p_fileName h5 file
     * @param p_run run number
     * @param p_dissthr dissatisfaction threshold
     * @param p_comsize size of committee to be elected
     * @param p_sim
     */


    public CChairAgentRI(final String p_name, final IAgentConfiguration<CChairAgentRI> p_configuration, final CEnvironmentRI p_environment,
                         final String p_fileName,
                         final int p_run,
                         final double p_dissthr,
                         final int p_comsize,
                         final int p_altnum,
                         final String p_rule,
                         int p_sim)
    {
        super( p_configuration );
        m_name = p_name;
        m_environment = p_environment;
        m_fileName = p_fileName;
        m_run = p_run;
        m_dissThreshold = p_dissthr;
        m_comsize = p_comsize;
        m_altnum = p_altnum;
        // TODO via parameters
        m_voteTimeout = 20;
        m_rule = p_rule;
        m_sim = p_sim;

        System.out.println( "sim in chair " + m_sim);
    }

    /**
     * ctor
     * @param p_name chair name
     * @param p_configuration configuration
     * @param p_environment environment
     * @param p_altnum number of alternatives
     * @param p_comsize committee size
     * @param p_dissthr dissatisfaction threshold
     * @param p_broker broker agent
     * @param p_run run number
     * @param p_sim sim number
     */
    public CChairAgentRI(final String p_name, final IAgentConfiguration<CChairAgentRI> p_configuration, final CEnvironmentRI p_environment, final int p_altnum,
                         final int p_comsize,
                         final double p_dissthr,
                         final CBrokerAgentRI p_broker,
                         final String p_rule,
                         int p_run,
                         int p_sim)
    {
        super( p_configuration );
        m_name = p_name;
        m_altnum = p_altnum;
        m_comsize = p_comsize;
        m_dissThreshold = p_dissthr;
        // TODO via parameters
        m_voteTimeout = 20;
        m_broker = p_broker;
        m_rule = p_rule;
        m_run = p_run;
        m_sim = p_sim;

        System.out.println( "sim in chair " + m_sim);
    }


    // overload agent-cycle
    @Override
    public final CChairAgentRI call() throws Exception
    {
        // run default cycle
     //   this.checkConditions();
        return super.call();
    }

    // public methods

    public String name()
    {
        return m_name;
    }

    /**
     * set configuration
     * @param p_conf config id
     * @param p_grouping grouping method
     * @param p_protocol protocol
     */

    public void setConf( final String p_conf, final String p_grouping, final String p_protocol )
    {
        m_conf = p_conf;
        m_grouping = p_grouping;
        m_protocol = p_protocol;
    }

    /**
     * return voters
     * @return voters
     */

    public List<CVotingAgentRI> voters()
    {
        return m_voters;
    }

    /**
     * reset chair agent for next simulation run
     */

    public void reset()
    {

        m_bitVotes = Collections.synchronizedList( new LinkedList<>() );
        m_cLinearOrders = Collections.synchronizedList( new LinkedList<>() );
        m_dissList = Collections.synchronizedList( new LinkedList<>() );
        m_dissVoters = Collections.synchronizedList( new LinkedList<>() );
        m_dissMap = new ConcurrentHashMap<>();
    //    m_agents = Collections.synchronizedList( new LinkedList<>() );
        m_iteration = 0;
        m_iterative = false;
        // m_dissStored = false;
        m_groupNum = 0;

        this.trigger( CTrigger.from(
            ITrigger.EType.ADDGOAL,
            CLiteral.from(
                "main" )
                      )
        );
    }

    /**
     * return path list
     * @return m_paths
     */
    public List<String> pathList()
    {
        return m_paths;
    }


    /**
     * return data list
     * @return m_data
     */
    public List<Object> dataList()
    {
        return m_data;
    }


    // agent actions

//    /**
//     * perceive group
//     */
//    @IAgentActionFilter
//    @IAgentActionName( name = "perceive/group" )
//    /**
//     * add literal for group of chair agent if it exists
//     */
//    private void perceiveGroup()
//    {
//        if ( !( m_environment.detectGroup( this ).emptyValues() ) )
//            this.beliefbase().add( m_environment.detectGroup( this ) );
//    }

    public CGroupRI group()
    {
        return m_group;
    }

    public HashMap<String, Object> map()
    {
        return m_map;
    }

//    /**
//     * return whether chair reached timeout
//     * @return boolean value
//     */
//    public boolean timedout()
//    {
//        System.out.println( this.name() + " " + this.cycle() + " Timeout: " + m_voteTimeout );
//        return this.cycle() >= m_voteTimeout;
//    }

    public List<CVotingAgentRI> dissvoters()
    {
        return new ArrayList<>( m_dissMap.keySet() );
    }



    // private methods

//    private void checkConditions()0 Timeout: 0.0
//    {
//      //  System.out.println( this.name() + " checking conditions " );
//        final CGroupRI l_group = this.determineGroup();
//
//        if ( l_group != null )
//
//        {
//            // if conditions for election are fulfilled, trigger goal start/criterion/fulfilled
//
//            final ITrigger l_trigger;
//
//            // if m_iterative is true, we have the case of iterative voting, i.e. we already have the votes
//            // we only need to repeat the computation of the result
//
//            if ( m_iterative && ( l_group.readyForElection() && !( l_group.electionInProgress() ) ) )
//            {
//                m_iteration++;
//                this.computeResult();
//                return;
//            }
//
//
//            if ( l_group.readyForElection() && ( !( l_group.electionInProgress() ) ) )
//            {
//                l_group.startProgress();
//
//                l_trigger = CTrigger.from(
//                    ITrigger.EType.ADDGOAL,
//                    CLiteral.from( "start/criterion/fulfilled" )
//                );
//
//                this.trigger( l_trigger );
//            }
//        }
//    }

    private CGroupRI determineGroup()
    {
        final AtomicReference<CGroupRI> l_groupAtomic = new AtomicReference<>();
        this.beliefbase().beliefbase().literal( "group" ).stream().forEach(
            i -> l_groupAtomic.set( i .values().findFirst().get().raw() ) );
        return l_groupAtomic.get();
    }




//    /**
//     * start election
//     */
//    @IAgentActionFilter
//    @IAgentActionName( name = "start/election" )
//
//    public void startElection()
//    {
//        final CGroupRI l_group = this.determineGroup();
//        l_group.triggerAgents( this );
//    }

    /**
     * store vote
     *
     * @param p_vote vote
     */
    @IAgentActionFilter
    @IAgentActionName( name = "store/vote" )
    public synchronized void storeVote( final String p_votingAgent, final BitVector p_vote )
    {
    //    final CGroupRI l_group = this.determineGroup();

    //    m_agents.add( l_group.determineAgent( p_agentName ) );

        if ( !this.group().timedout() && !m_votersStr.contains( p_votingAgent ) )
        {

            // for MS-AV and MM-AV, the votes are 01-vectors
            if ( m_rule.equals( "MINISUM_APPROVAL") || m_rule.equals( "MINIMAX_APPROVAL" ) )
                this.storeAV( p_votingAgent, p_vote );

            else
                if ( m_rule.equals( "MINISUM_RANKSUM") )
            // for MS-RS, the votes are complete linear orders
            {
                System.out.println( "store complete linear order" );
                this.storeCLO( p_votingAgent, p_vote );
            }
        }

        else if ( this.group().timedout() )
        {
            System.out.println( this.name() + " timeout reached, not accepting vote of agent " + p_votingAgent );

        }

        else if ( m_votersStr.contains( p_votingAgent ) )

            System.out.println( this.name() + " already containing " + p_votingAgent );


//        if ( m_bitVotes.size() != l_group.size() )
//            return;
//
//        final ITrigger l_trigger = CTrigger.from(
//            ITrigger.EType.ADDGOAL,
//            CLiteral.from(
//                "all/votes/received" )
//
//        );
//
//        System.out.println( " xxxxxxxxxxxxxxxxxxxxxxxxxxxxx " + this.name() + " all votes received " );
//
//        this.trigger( l_trigger );
    }

    private void storeAV( final String p_votingAgent, final Object p_vote )
    {
        m_bitVotes.add( (BitVector) p_vote );

        m_voters.add( this.group().getAgent( p_votingAgent ) );
        m_votersStr.add ( p_votingAgent );

        System.out.println( " --------------------- " + this.name() + " received vote from " + p_votingAgent );
    }

    private void storeCLO( final String p_votingAgent, final Object p_vote )
    {
        ArrayList<Long> l_vote = (ArrayList<Long>) p_vote;

        m_cLinearOrders.add( l_vote );
        m_voters.add( this.group().getAgent( p_votingAgent ) );
        m_votersStr.add( p_votingAgent );

        System.out.println( " --------------------- " + this.name() + " received vote from " + p_votingAgent );

    }

//    /**
//     * close group
//     */
//    @IAgentActionFilter
//    @IAgentActionName( name = "close/group" )
//
//    public void closeGroup( )
//    {
//        System.out.println("close group");
//        m_group.close();
//    }


    /**
     * set group to votes submitted
     */
    @IAgentActionFilter
    @IAgentActionName( name = "set/group/submitted" )
    public void setGroupSubmitted( )
    {
        m_group.setSubmitted();

    }

        /**
         * compute result of election
         */

    @IAgentActionFilter
    @IAgentActionName( name = "compute/result" )

    public synchronized void computeResult( final Number p_iteration ) throws SQLException {

        m_iteration = p_iteration.intValue();

        // TODO check
        if ( m_voters.size() == 0 )
        {
            System.out.println( this.name() + " I have received no voters, cannot conduct election" );
            System.out.println( Arrays.toString( this.group().getVoters().toArray() ) );
            // reset timeout for diss vals

            this.group().setWaitingForDiss();

            this.group().setDissCounter( new AtomicLong( 50 ) );


            return;
        }

        try
        {

            System.out.println( "Iteration:" + p_iteration );

            if ( m_rule.equals( "MINISUM_APPROVAL" ) )
                m_comResultBV = this.computeMSAV();

            else // if ( m_rule.equals( "MINISUM_RANKSUM" ) )

                m_comResultBV = this.computeMSRS();

            System.out.println( " ------------------------ " + this.name() + ", Iteration " + p_iteration + ", Result of election as BV: " + m_comResultBV );



            //        m_voters.stream().forEach( i ->
            //            i.trigger(
            //                CTrigger.from(
            //                    ITrigger.EType.ADDGOAL,
            //                    CLiteral.from(
            //                        "submit/diss",
            //                        CRawTerm.from( this ),
            //                        CRawTerm.from( l_comResultBV )
            //                    )
            //                )
            //            )
            //        );

            m_voters.stream().forEach( i ->
                {
                    i.beliefbase().add(
                        CLiteral.from(
                            "result",
                            CRawTerm.from( this ),
                            CRawTerm.from( m_comResultBV ),
                            CRawTerm.from( p_iteration )
                        )
                    );
                    System.out.println( "addbelief result to agent " + i.name() );
                    System.out.println( "result " + i.toString() );
                }
            );

            // store intermediate election results
            m_map.put( this.name() + "/" + p_iteration + "/election result", m_comResultBV );
            // store contributing agents
            m_map.put( this.name() + "/" + p_iteration + "/agents", this.asString( m_voters ) );

            // store election result in map
            m_map.put( this.name() + "/election result", m_comResultBV );
            // store group size in map
            m_map.put( this.name() + "/group size", m_voters.size() );
            // store names of agents
            //     for ( int i = 0; i < m_voters.size(); i++ )
            m_map.put( this.name() + "/agents", this.asString( m_voters ) );

            // store iteration number

            m_map.put( this.name() + "/itNum", p_iteration.intValue() );

            // store group ID
            m_map.put( this.name() + "/groupID", this.group().id() );

            // m_dissStored = false;

            // reset timeout for diss vals

            this.group().setWaitingForDiss();

            this.group().setDissCounter( new AtomicLong( 50 ) );
         //   m_dissCounter = this.cycle() + 50;

        }
        catch ( final ConcurrentModificationException l_ex )
        {
            System.out.println( "ConcurrentModificationException in computeResult" );
            System.exit( 1 );
        }

    }

    private BitVector computeMSAV()
    {
        final CMinisumApproval l_minisumApproval = new CMinisumApproval();

        final List<String> l_alternatives = new LinkedList<>();

        System.out.println( "number of alternatives: " + m_altnum );

        for ( int i = 0; i < m_altnum; i++ )
            l_alternatives.add( "POI" + i );

        System.out.println( " Alternatives: " + l_alternatives );

        System.out.println( " Votes: " + m_bitVotes );

        return l_minisumApproval.applyRuleBV( l_alternatives, m_bitVotes, m_comsize );
    }

    private BitVector computeMSRS()
    {
        final CMinisumRanksum l_minisumRanksum = new CMinisumRanksum();

        final List<String> l_alternatives = new LinkedList<>();

        System.out.println( "number of alternatives: " + m_altnum );

        for ( int i = 0; i < m_altnum; i++ )
            l_alternatives.add( "POI" + i );

        System.out.println( " Alternatives: " + l_alternatives );

        System.out.println( " Votes: " + m_cLinearOrders );

        return l_minisumRanksum.applyRuleBV( l_alternatives, m_cLinearOrders, m_comsize );

    }

    private String asString( final List<CVotingAgentRI> p_voters )
    {
        String l_string = "{";
        for ( int i = 0; i < p_voters.size() - 1; i++ )
            l_string = l_string.concat( p_voters.get( i ).name() + ", " );
        l_string = l_string.concat( p_voters.get( p_voters.size() - 1 ).name() + "}" );

        return l_string;
    }

    /**
     * store dissatisfaction value
     *
     * @param p_diss dissatisfaction value
     */
    @IAgentActionFilter
    @IAgentActionName( name = "store/diss" )

    public synchronized void storeDiss( final String p_votingAgent, final Number p_diss, final Number p_fill, final Number p_iteration ) throws SQLException {

        if ( this.group().dissTimedOut() )
        {
            System.out.println( "diss timeout reached, not accepting diss of agent " + p_votingAgent );
            return;
        }
//        m_dissList.add( p_diss.doubleValue() );
//        m_dissVoters.add( this.getAgent( p_votingAgent ) );

        m_dissMap.put( this.getAgent( p_votingAgent ), p_diss.doubleValue() );
        m_dissMapStr.put( p_votingAgent, p_diss.floatValue() );

//        System.out.println( this.name() + " storing diss " + p_diss + " from agent " + p_votingAgent + " for iteration " + p_iteration
//                            + " dissMap " + m_dissMap.size() + " fill " + p_fill );

        System.out.println( this.name() + " storing diss " + p_diss + " from agent " + p_votingAgent + " for iteration " + p_iteration
                            + " dissMap " + m_dissMap.size() + " voters " + m_voters.size() );

        // store diss for each iteration

        m_map.put( this.name() + "/" + p_iteration + "/" + p_votingAgent, p_diss.doubleValue() );

        m_map.put( this.name() + "/" + p_votingAgent, p_diss.doubleValue() );

        //  final String l_path = m_run + l_slash + m_conf + l_slash + "group " + this.getGroupID() + l_slash + p_iteration + l_slash + "dissVals";

        //   m_map.put( l_path, l_dissVals );

        // TODO write data to list instead
        //    EDataWriter.INSTANCE.writeDataVector( m_run, m_conf, this, p_iteration, l_dissVals );
        //    new CDataWriter().writeDataVector( m_fileName, m_run, m_conf, this, p_iteration, l_dissVals );

        // TODO refactor

      //  if ( m_dissMap.size() >= p_fill.intValue() )

        if ( ( m_dissMap.size() == m_voters.size() ) && !m_removedGoalAdded )
        {
            System.out.println( "Number of voters and diss vals: " + m_voters.size() );

            // for iterative election, set lastElection to false per default, is set to true later
            // TODO use p_dbGroup instead?

            EDataDB.INSTANCE.addResult(this.group().getDB(),
                    m_comResultBV.toString(),
                    "ITERATIVE",
                    false,
                    m_iteration,
                    -1,
                    m_dissMapStr,
                    m_run,
                    m_sim  );

            this.group().setDissSubmitted();

            this.trigger(
                CTrigger.from(
                    ITrigger.EType.ADDGOAL,
                    CLiteral.from(
                        "removed/voter"
                    )
                )
            );

            System.out.println( "fill: " +  m_dissMap.size() + " add goal !removed/voter" );
            if ( !( m_dissMap.isEmpty() ) )
            {
                m_newdissMap = new ConcurrentHashMap<>( m_dissMap );
                m_dissMap.clear();
            }
            m_removedGoalAdded = true;

        }

    }

    /**
     * return true if waiting time for diss vals is timedout
     * @return boolean value
     */

//    public boolean dissTimedOut()
//    {
//        System.out.println( this.name() + " Diss timeout: " + m_dissCounter );
//
//        if ( m_dissCounter == 0 )
//            return false;
//        else  return this.cycle() >= m_dissCounter;
//    }

    private CVotingAgentRI getAgent( final String p_votingAgent )
    {
        for ( int i = 0; i < m_voters.size(); i++ )
            if ( m_voters.get( i ).name().equals( p_votingAgent ) )
                return m_voters.get( i );
        return null;
    }


    //    /**
//     * compute result of election
//     */
//
//    @IAgentActionFilter
//    @IAgentActionName( name = "compute/result" )
//
//    public void computeResult()
//    {
//        final CGroupRI l_group = this.determineGroup();
//
//        final CMinisumApproval l_minisumApproval = new CMinisumApproval();
//
//        final List<String> l_alternatives = new LinkedList<>();
//
//     //   for ( char l_char : "ABCD".toCharArray() )
//
//        for ( int i = 0; i < m_altnum; i++ )
//            l_alternatives.add( "POI" + i );
//
//         //   l_alternatives.add( String.valueOf( l_char ) );
//
//        System.out.println( " Alternatives: " + l_alternatives );
//
//        System.out.println( " Votes: " + m_bitVotes );
//
//        final BitVector l_comResultBV = l_minisumApproval.applyRuleBV( l_alternatives, m_bitVotes, m_comsize );
//
//        System.out.println( " ------------------------ " + this.name() + " Result of election as BV: " + l_comResultBV );
//
//        // set inProgress and readyForElection to false in group
//        l_group.reset();
//
//        // write resulting committee for coordinated grouping
//        if ( "COORDINATED".equals( m_grouping ) )
//        {
//            final String l_slash = "/";
//
//            final String l_path = m_run + l_slash  + m_conf + l_slash + "group " + this.getGroupID() + l_slash + "im_" + m_coorNum + l_slash + "committee";
//
//            m_map.put( l_path, l_comResultBV );
//
//            // old code
//            // EDataWriter.INSTANCE.writeCommitteeCoordinated( m_run, m_conf, this, l_comResultBV, m_coorNum );
//            // new CDataWriter().writeCommitteeCoordinated( m_fileName, m_run, m_conf, this, l_comResultBV, m_coorNum );
//            m_coorNum++;
//        }
//        if ( "BASIC".equals( m_protocol ) )
//        {
//            this.beliefbase().add( l_group.updateBasic( this, l_comResultBV ) );
//
//            if ( "RANDOM".equals( m_grouping ) )
//            {
//                // ask all agents in group to submit their dissatisfaction value
//                System.out.println( "Ask agents to submit final diss " );
//                this.beliefbase().add( l_group.submitDiss( this, l_comResultBV, m_iteration ) );
//            }
//
//            if ( "COORDINATED".equals( m_grouping ) && l_group.finale() )
//            {
//                System.out.println( "Ask agents to submit final diss " );
//                this.beliefbase().add( l_group.submitDiss( this, l_comResultBV, m_iteration ) );
//            }
//        }
//
//        // if grouping is coordinated, reopen group for further voters
//        if ( "COORDINATED".equals( m_grouping ) && !l_group.finale() && !m_iterative )
//        {
//            System.out.println( " reopening group " );
//            m_environment.reopen( l_group );
//        }
//
//        // for the iterative case, you need to differentiate between the final election and intermediate elections.
//        if ( "ITERATIVE".equals( m_protocol ) && ( l_group.finale() ) || m_iterative )
//        {
//            System.out.println( " Update iterative " );
//
//            this.beliefbase().add( l_group.updateIterative( this,  l_comResultBV, m_iteration ) );
//            return;
//        }
//
//        if ( "ITERATIVE".equals( m_protocol ) && !l_group.finale() )
//        {
//            System.out.println( " Update basic " );
//            this.beliefbase().add( l_group.updateBasic( this,  l_comResultBV ) );
//        }
//
//        m_dissStored = false;
//
//        // TODO test all cases
//    }

//    /**
//     * store dissatisfaction value
//     *
//     * @param p_diss dissatisfaction value
//     * @param p_iteration iteration number
//     */
//    @IAgentActionFilter
//    @IAgentActionName( name = "store/diss" )
//
//    public void storeDiss( final String p_name, final Double p_diss, final Integer p_iteration )
//    {
//        final CGroupRI l_group = this.determineGroup();
//
//        m_dissList.add( p_diss );
//        final CVotingAgentRI l_dissAg = l_group.determineAgent( p_name );
//        m_dissVoters.add( l_dissAg );
//
//
//        System.out.println( "Storing diss " + p_diss );
//
//        if ( m_dissList.size() == l_group.size() )
//            // && ( !m_dissStored ) )
//        {
//            //     m_dissStored = true;
//            System.out.println( this.name() + " Size of group " + m_dissVoters.size() );
//
//            final ITrigger l_trigger = CTrigger.from(
//                ITrigger.EType.ADDGOAL,
//                CLiteral.from(
//                    "all/dissValues/received",
//                    CRawTerm.from( p_iteration )
//                )
//
//            );
//
//            this.trigger( l_trigger );
//
//            System.out.println( p_iteration + " All " + m_dissList.size() + " voters submitted their dissatisfaction value" );
//            System.out.println( Arrays.toString( m_dissList.toArray() ) );
//
//            final AtomicDoubleArray l_dissVals = new AtomicDoubleArray( new double[m_dissList.size()] );
//            for ( int i = 0; i < m_dissList.size(); i++ )
//                l_dissVals.set( i, m_dissList.get( i ) );
//
//            // final String l_config = "RANDOM_BASIC";
//
//            // TODO refactor string building
//            final String l_slash = "/";
//
//            final String l_path = m_run + l_slash + m_conf + l_slash + "group " + this.getGroupID() + l_slash + p_iteration + l_slash + "dissVals";
//
//            m_map.put( l_path, l_dissVals );
//
//            // TODO write data to list instead
//        //    EDataWriter.INSTANCE.writeDataVector( m_run, m_conf, this, p_iteration, l_dissVals );
//        //    new CDataWriter().writeDataVector( m_fileName, m_run, m_conf, this, p_iteration, l_dissVals );
//        }
//    }

//    /**
//     * store final dissatisfaction value
//     *
//     * @param p_diss dissatisfaction value
//     */
//    @IAgentActionFilter
//    @IAgentActionName( name = "store/final/diss" )
//
//    public void storeFinalDiss( final String p_name, final Double p_diss, final Integer p_iteration )
//    {
//        final CGroupRI l_group = this.determineGroup();
//
//        m_dissList.add( p_diss );
//        final CVotingAgentRI l_dissAg = l_group.determineAgent( p_name );
//        m_dissVoters.add( l_dissAg );
//
//        System.out.println( "Storing diss " + p_diss );
//
//        if ( ( m_dissList.size() == l_group.size() ) && ( !m_dissStored ) )
//        {
//            m_dissStored = true;
//            System.out.println( p_iteration + " All " + m_dissList.size() + " voters submitted their dissatisfaction value" );
//            System.out.println( Arrays.toString( m_dissList.toArray() ) );
//
//            final AtomicDoubleArray l_dissVals = new AtomicDoubleArray( new double[m_dissList.size()] );
//            for ( int i = 0; i < m_dissList.size(); i++ )
//                l_dissVals.set( i, m_dissList.get( i ) );
//
//            final String l_slash = "/";
//
//            final String l_groupStr = "group ";
//
//            final String l_path = m_run + l_slash + m_conf + l_slash + l_groupStr + this.getGroupID() + l_slash + p_iteration + l_slash + "dissVals";
//
//            m_map.put( l_path, l_dissVals );
//
//            final String l_pathIt = m_run + l_slash + m_conf + l_slash + l_groupStr + this.getGroupID() +  l_slash + "lastIt";
//
//            m_map.put( l_pathIt, p_iteration );
//
//            m_environment.incrementGroupCount( m_run, m_conf );
//
//            // TODO write data to list instead
//        //    EDataWriter.INSTANCE.writeDataVector( m_run, m_conf, this, p_iteration, l_dissVals );
//        //    EDataWriter.INSTANCE.writeLastIteration( m_run, m_conf, this, p_iteration );
//        //    new CDataWriter().writeDataVector( m_fileName, m_run, m_conf, this, p_iteration, l_dissVals );
//        //    new CDataWriter().writeLastIteration( m_fileName, m_run, m_conf, this, p_iteration );
//        }
//
//    }

    /**
     * remove most dissatisfied voter
     */
    @IAgentActionFilter
    @IAgentActionName( name = "remove/voter" )
    public synchronized void removeVoter( ) throws SQLException {
        m_removedGoalAdded = false;

        System.out.println( "removing voter " );
        final CGroupRI l_group = this.group();

        final double l_max = this.getMaxDiss( m_newdissMap );

//        final int l_maxIndex = this.getMaxIndex( m_dissList );
//        final double l_max = m_dissList.get( l_maxIndex );
        System.out.println( " max diss is " + l_max );
        System.out.println( "dissThr is " + m_dissThreshold );

        if ( l_max <= m_dissThreshold )
        {
            System.out.println( this.name() + ": no dissatisfied voter left, we are done " );
            System.out.println( Arrays.toString( l_group.getVoters().toArray() ) );
            EDataDB.INSTANCE.setLastElection( l_group.getDB(), true );
            return;
        }

        // TODO: revise, try alternative approaches
        // TODO: this should be consistent with l_group.size()
        else if ( m_newdissMap.size() == 1 )
        {
            System.out.println( this.name() + ": only one voter left, we are done " );
            System.out.println( Arrays.toString( l_group.getVoters().toArray() ) );
            EDataDB.INSTANCE.setLastElection( l_group.getDB(), true );
            return;
        }

        System.out.println( " Determining most dissatisfied voter " );


//        final CVotingAgentRI l_maxDissAg = m_dissVoters.get( l_maxIndex );
//        System.out.println( " Most dissatisfied voter is " + l_maxDissAg.name() );

        final CVotingAgentRI l_maxDissAg = this.getMaxAg( m_newdissMap );
        System.out.println( " Most dissatisfied voter is " + l_maxDissAg.name() );
        // remove vote of most dissatisfied voter from list
        m_bitVotes.remove( l_maxDissAg.getBitVote() );
        l_group.remove( l_maxDissAg );

        m_voters.remove( l_maxDissAg );
        m_votersStr.remove( l_maxDissAg.name() );

        // add belief in broker
        m_broker.removeAndAddAg( l_maxDissAg );

        System.out.println( this.name() + ": Removing " + l_maxDissAg.name() );
        // System.out.println( this.name() + ":Size of List after removing " + m_dissVoters.size() );
        System.out.println( this.name() + ": Size of Group after removing " + l_group.size() );

        // if a voter needed to be removed, the election has to be repeated

        this.trigger( CTrigger.from(
            ITrigger.EType.ADDGOAL,
            CLiteral.from(
                "reelected" )
                      )
        );

        // remove diss Values for next iteration
//        m_dissVoters.clear();
//        m_dissList.clear();

 //       m_dissMap.clear();

        m_dissMapStr.clear();
        m_newdissMap.clear();


        // add altered group to database
        l_group.setDB( EDataDB.INSTANCE.newGroup(l_group.chair().name(), l_group.getDB(), l_group.getVoters(), m_run, m_sim ) );


        // update map
   //     m_map.remove( this.name() + "/" + l_maxDissAg.name() );
   //     m_map.put( this.name() + "/agents", this.asString( m_voters ) );

//        m_iterative = true;
//        l_group.makeReady();
    }

    private double getMaxDiss( final ConcurrentHashMap<CVotingAgentRI, Double> p_dissMap )
    {
        double l_max = 0;

        for ( final CVotingAgentRI l_key : p_dissMap.keySet() )
        {
            if ( p_dissMap.get( l_key ) > l_max )
                l_max = p_dissMap.get( l_key );
        }
        return l_max;
    }

    private CVotingAgentRI getMaxAg( final ConcurrentHashMap<CVotingAgentRI, Double> p_dissMap )
    {
        double l_max = 0;
        CVotingAgentRI l_maxAg = null;

        for ( final CVotingAgentRI l_key : p_dissMap.keySet() )
        {
            if ( p_dissMap.get( l_key ) > l_max )
            {
                l_maxAg = l_key;
                l_max = p_dissMap.get( l_key );
            }
        }
        return l_maxAg;
    }

    /**
     * set group of chair
     * @param p_group group of chair
     */

    public void setGroup( final CGroupRI p_group )
    {
        m_group = p_group;

    }


    //   public int getGroupID()
//    {
//        return this.determineGroup().getID();
//    }


    /**
     * Class CChairAgentGenerator
     */

    public static final class CChairAgentGenerator extends IBaseAgentGenerator<CChairAgentRI>
    {

        /**
         * environment
         */
        private final CEnvironmentRI m_environment;

        /**
         * Current free agent id, needs to be thread-safe, therefore using AtomicLong.
         */
        private final AtomicLong m_agentcounter = new AtomicLong();

        private final String m_fileName;
        private int m_sim;
        private int m_run;
        private double m_dissthr;
        private int m_comsize;
        private int m_altnum;
        private CBrokerAgentRI m_broker;
        private String m_rule;

        /**
         * constructor of the generator
         * @param p_stream ASL code as any stream e.g. FileInputStream
         * @param p_fileName h5 file
         * @param p_run run number
         * @param p_dissthr dissatisfaction threshold
         * @param p_comsize size of committee to be elected
         * @param p_broker broker agent
         * @param p_sim
         * @throws Exception Thrown if something goes wrong while generating agents.
         */
        public CChairAgentGenerator(final InputStream p_stream, final CEnvironmentRI p_environment,
                                    final String p_fileName,
                                    final int p_run,
                                    final double p_dissthr, final int p_comsize, final int p_altnum,
                                    final CBrokerAgentRI p_broker, final String p_rule,
                                    int p_sim) throws Exception
        {
            super(
                // input ASL stream
                p_stream,

                // a set with all possible actions for the agent
                Stream.concat(
                    // we use all build-in actions of LightJason
                    CCommon.actionsFromPackage(),
                    Stream.concat(
                        // use the actions which are defined inside the agent class
                        CCommon.actionsFromAgentClass( CChairAgentRI.class ),
                        // add VotingAgent related external actions
                        Stream.of(

                        )
                    )
                    // build the set with a collector
                ).collect( Collectors.toSet() )
                //,

                // aggregation function for the optimisation function, here
                // we use an empty function
         //       IAggregation.EMPTY
            );
            m_environment = p_environment;
            m_fileName = p_fileName;
            m_run = p_run;
            m_dissthr = p_dissthr;
            m_comsize = p_comsize;
            m_altnum = p_altnum;
            m_broker = p_broker;
            m_rule = p_rule;
            m_sim = p_sim;
        }

        /**
         * constructor of the generator
         * @param p_chairstream ASL code as any stream e.g. FileInputStream
         * @param p_environment environment
         * @param p_name file name
         * @param p_altnum number of alternatives
         * @param p_dissthr dissatisfaction threshold
         * @param p_broker broker agent
         * @param p_run
         * @param p_sim
         * @throws Exception Thrown if something goes wrong while generating agents.
         */

        public CChairAgentGenerator(final InputStream p_chairstream, final CEnvironmentRI p_environment, final String p_name, final int p_altnum,
                                    final int p_comsize, final double p_dissthr,
                                    final CBrokerAgentRI p_broker, final String p_rule,
                                    int p_run, int p_sim)
        throws Exception
        {
            super(
                // input ASL stream
                p_chairstream,

                // a set with all possible actions for the agent
                Stream.concat(
                    // we use all build-in actions of LightJason
                    CCommon.actionsFromPackage(),
                    Stream.concat(
                        // use the actions which are defined inside the agent class
                        CCommon.actionsFromAgentClass( CChairAgentRI.class ),
                        // add VotingAgent related external actions
                        Stream.of(

                        )
                    )
                    // build the set with a collector
                ).collect( Collectors.toSet() )
                //,

                // aggregation function for the optimisation function, here
                // we use an empty function
                //       IAggregation.EMPTY
            );

            m_environment = p_environment;
            m_fileName = p_name;
            m_altnum = p_altnum;
            m_comsize = p_comsize;
            m_dissthr = p_dissthr;
            m_broker = p_broker;
            m_rule = p_rule;
            m_run = p_run;
            m_sim = p_sim;

        }

        /**
         * generator method of the agent
         * @param p_data any data which can be put from outside to the generator method
         * @return returns an agent
         */

        @Override
        public final CChairAgentRI generatesingle( final Object... p_data )
        {

            final CChairAgentRI l_chairAgent = new CChairAgentRI(
                // create a string with the agent name "chair <number>"
                // get the value of the counter first and increment, build the agent
                // name with message format (see Java documentation)
                MessageFormat.format( "chair {0}", m_agentcounter.getAndIncrement() ), m_configuration, m_environment, m_fileName, m_run, m_dissthr, m_comsize, m_altnum, m_rule, m_sim);
            l_chairAgent.sleep( Integer.MAX_VALUE );
            System.out.println( "Creating chair " + l_chairAgent.name() );
            return l_chairAgent;
        }

        /**
         * generator method of the agent
         * @return returns an agent
         */

        public CChairAgentRI generatesinglenew()
        {
            final CChairAgentRI l_chairAgent = new CChairAgentRI(
                // create a string with the agent name "chair <number>"
                // get the value of the counter first and increment, build the agent
                // name with message format (see Java documentation)
                MessageFormat.format( "chair {0}", m_agentcounter.getAndIncrement() ), m_configuration, m_environment, m_altnum, m_comsize, m_dissthr,
                m_broker, m_rule, m_run, m_sim
            );
            return l_chairAgent;
        }

    }


//    /**
//     * remove most dissatisfied voter
//     */
//    @IAgentActionFilter
//    @IAgentActionName( name = "remove/voter" )
//    public void removeVoter( final Integer p_iteration )
//    {
//        System.out.println( "removing voter " );
//        final CGroupRI l_group = this.determineGroup();
//
//        final int l_maxIndex = this.getMaxIndex( m_dissList );
//        final double l_max = m_dissList.get( l_maxIndex );
//        System.out.println( " max diss is " + l_max );
//
//        if ( l_max <= m_dissThreshold )
//        {
//            System.out.println( " No dissatisfied voter left, we are done " );
//
//            final String l_slash = "/";
//
//            final String l_groupStr = "group ";
//
//            //            final String l_path = m_run + l_slash + m_conf + l_slash + l_groupStr + this.getGroupID() +  l_slash + "lastIt";
//            //
//            //            m_map.put( l_path, p_iteration );
//
//            //            m_environment.incrementGroupCount( m_run, m_conf );
//
//
//            // EDataWriter.INSTANCE.writeLastIteration( m_run, m_conf, this, p_iteration );
//            //    new CDataWriter().writeLastIteration( m_fileName, m_run, m_conf, this, p_iteration );
//            return;
//        }
//        System.out.println( " Determining most dissatisfied voter " );
//        final CVotingAgentRI l_maxDissAg = m_dissVoters.get( l_maxIndex );
//        System.out.println( " Most dissatisfied voter is " + l_maxDissAg.name() );
//        // remove vote of most dissatisfied voter from list
//        m_bitVotes.remove( l_maxDissAg.getBitVote() );
//        l_group.remove( l_maxDissAg );
//
//        System.out.println( "Removing " + l_maxDissAg.name() );
//        // System.out.println( this.name() + ":Size of List after removing " + m_dissVoters.size() );
//        System.out.println( this.name() + ":Size of Group after removing " + l_group.size() );
//
//        if ( l_group.size() == 0 )
//        {
//            System.out.println( " Voter list is empty, we are done " );
//            // TODO write data to list instead
//
//            final String l_slash = "/";
//
//            final String l_groupStr = "group ";
//
//            //            final String l_path = m_run + l_slash + m_conf + l_slash + l_groupStr + this.getGroupID() +  l_slash + "lastIt";
//            //
//            //            m_map.put( l_path, p_iteration );
//            //
//            //            m_environment.incrementGroupCount( m_run, m_conf );
//        }
//
//        // remove diss Values for next iteration
//        m_dissVoters.clear();
//        m_dissList.clear();
//
//        //        m_iterative = true;
//        //        l_group.makeReady();
//    }
}
