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

package org.lightvoting.simulation.agent.coordinated_iterative;

import cern.colt.matrix.tbit.BitVector;
import com.google.common.util.concurrent.AtomicDoubleArray;
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
import org.lightvoting.simulation.action.message.coordinated_iterative.CSendCI;
import org.lightvoting.simulation.agent.coordinated_iterative.CVotingAgentCI.CVotingAgentGenerator;
import org.lightvoting.simulation.environment.coordinated_iterative.CEnvironmentCI;
import org.lightvoting.simulation.environment.coordinated_iterative.CGroupCI;
import org.lightvoting.simulation.statistics.EDataDB;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.FileInputStream;
import java.io.InputStream;
import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.Stream;


/**
 * Created by sophie on 18.07.17.
 */
@IAgentAction
public class CBrokerAgentCI extends IBaseAgent<CBrokerAgentCI>
{
    private List<CVotingAgentCI> m_voters = new ArrayList<>();
    private HashSet<CChairAgentCI> m_chairs = new HashSet<>();
    private HashSet<CGroupCI> m_groups = new HashSet<>();
    private final String m_name;
    private final int m_agNum;
    private int m_count;
    private IAgentConfiguration<CVotingAgentCI> m_configuration;
    private final InputStream m_stream;
    private int m_altnum;
    private double m_joinThr;
    private List<AtomicDoubleArray> m_prefList;
    private final String m_broker;
    private CVotingAgentGenerator m_votingagentgenerator;
    private int m_groupNum;
    // TODO read via yaml
    private int m_capacity;
    // TODO read via yaml
    final private long m_timeout;
    private Object m_fileName;
    private CChairAgentCI.CChairAgentGenerator m_chairagentgenerator;
    private final InputStream m_chairstream;
    private final int m_comsize;
    // TODO lining limit for allowing agents to drive alone
    // HashMap for storing how often an agent had to leave a group
    private final HashMap<String, Long> m_lineHashMap = new HashMap<String, Long>();
    // TODO via config
    private final int m_maxLiningCount = 2;
    private final CEnvironmentCI m_environmentCI;
    private final double m_dissthr;
    private HashMap<String, Object> m_map = new HashMap<>();
    private String m_rule;
    private int m_run;
    private int m_sim;

    /**
     * ctor
     * @param p_broker broker name
     * @param p_configuration agent configuration
     * @param p_agNum number of voters
     * @param p_stream agent stream
     * @param p_chairstream chair stream
     * @param p_environment enviroment
     * @param p_altnum number of alternatives
     * @param p_name name
     * @param p_joinThr join threshold
     * @param p_prefList preference list
     * @param p_comsize committee size
     * @param p_dissthr
     * @param p_run
     * @param p_sim
     * @throws Exception exception
     */
    public CBrokerAgentCI(final String p_broker,
                          @Nonnull final IAgentConfiguration p_configuration,
                          final int p_agNum,
                          final InputStream p_stream,
                          final InputStream p_chairstream,
                          final CEnvironmentCI p_environment,
                          final int p_altnum,
                          final String p_name,
                          final double p_joinThr,
                          final List<AtomicDoubleArray> p_prefList,
                          final int p_comsize,
                          final double p_dissthr,
                          final String p_rule,
                          int p_run, int p_sim) throws Exception
    {
        super( p_configuration );
        m_broker = p_broker;
        m_agNum = p_agNum;
        m_stream = p_stream;
        m_chairstream = p_chairstream;
        m_environmentCI = p_environment;
        m_altnum = p_altnum;
        m_name = p_name;
        m_joinThr = p_joinThr;
        m_prefList = p_prefList;
        // TODO set via yaml
        m_capacity = 10;
        m_timeout = 20;
        m_comsize = p_comsize;
        m_dissthr = p_dissthr;
        m_rule = p_rule;
        m_run = p_run;
        m_sim = p_sim;

        System.out.println( "dissthr in broker: " + m_dissthr );

        m_votingagentgenerator = new CVotingAgentGenerator( new CSendCI(), m_stream, m_environmentCI, m_altnum, m_name,
                                                                           m_joinThr, m_prefList, m_rule, m_run, m_sim);
        m_chairagentgenerator = new CChairAgentCI.CChairAgentGenerator( m_chairstream, m_environmentCI, m_name, m_altnum, m_comsize, m_capacity, this,
                                                                        m_dissthr, m_rule, m_run, m_sim );
        this.trigger( CTrigger.from(
            ITrigger.EType.ADDBELIEF,
            CLiteral.from(
                "agnum",
                CRawTerm.from( m_agNum ) )
                      )
        );

//        this.beliefbase().beliefbase().add( CLiteral.from(
//
//            "agnum",
//            CRawTerm.from( m_agNum )  ) );
    }

    /**
     * return stream over agents
     * @return agent stream
     */
    public Stream<IBaseAgent> agentstream()
    {
        return Stream.concat(
            m_voters.stream(),
            m_chairs.stream()
        );
    }

    public HashMap<String,Object> map()
    {
        return m_map;
    }

    @IAgentActionFilter
    @IAgentActionName( name = "decrement/counters" )
    private synchronized void decrementCounters() throws Exception
    {

        for ( final CGroupCI l_group : m_groups )
        {
            l_group.decrementCounter();

            if ( l_group.waitingforDiss() )
                l_group.decrementDissCounter();
        }

    }

    @IAgentActionFilter
    @IAgentActionName( name = "create/ag" )
    private CVotingAgentCI createAgent( final Number p_createdNum ) throws Exception
    {

        System.out.println( "voters generated so far: " + p_createdNum );

        final CVotingAgentCI l_testvoter = m_votingagentgenerator.generatesinglenew();

        System.out.println( "new voter:" + l_testvoter.name() );

        m_voters.add( l_testvoter );

        String l_name = l_testvoter.name();

        // add voter instance to database

        EDataDB.INSTANCE.addVoter(l_name, m_run, m_sim);

        return l_testvoter;

    }

    @IAgentActionFilter
    @IAgentActionName( name = "assign/group" )
    private synchronized void assignGroup( final CVotingAgentCI p_votingAgent ) throws Exception
    {
        CGroupCI l_determinedGroup = null;
        int l_dist = Integer.MAX_VALUE;

        System.out.println( "Assigning group to " + p_votingAgent.name() );


        System.out.println( "join threshold: " + m_joinThr );
        for ( final CGroupCI l_group : m_groups )
        {
            System.out.println( "group " + l_group.id() + " open: " + l_group.open() );
            // you can only add an agent to group if it is still open and the result is not null
            if ( l_group.open() && !( l_group.result() == null ) && p_votingAgent.unknownGroup( l_group ) )
            {
                System.out.println( "Result:" + l_group.result() );
                // use new distance if it is lower than the joint threshold and than the old distance
                final int l_newDist;

                if ( m_rule.equals( "MINISUM_APPROVAL"))
                    l_newDist = this.hammingDistance( p_votingAgent.getBitVote(), l_group.result() );
                else
                    l_newDist = this.ranksum( p_votingAgent.getCLOVote(), l_group.result() );
                System.out.println( "Distance: " + l_newDist );
                if ( l_newDist < m_joinThr && l_newDist < l_dist )
                {
                    l_dist = l_newDist;
                    l_determinedGroup = l_group;
                }
            }
        }


        // if there is no available group, create a new group
        if ( l_determinedGroup != null )
        {
            l_determinedGroup.add( p_votingAgent );
            System.out.println( "Adding agent " + p_votingAgent.name() + " to existing group" + ", ID " + l_determinedGroup.id() );
            p_votingAgent.beliefbase().add( CLiteral.from( "mygroup", CRawTerm.from( l_determinedGroup ) ) );
            System.out.println( p_votingAgent.name() + " gets belief regarding group with id " + l_determinedGroup.id() );

            p_votingAgent.beliefbase().add( CLiteral.from( "mychair", CRawTerm.from( l_determinedGroup.chair() ) ) );
            System.out.println( p_votingAgent.name() + " gets belief regarding chair with chair name " + l_determinedGroup.chair().name() );

            p_votingAgent.addGroupID( l_determinedGroup );
            p_votingAgent.setChair( l_determinedGroup.chair() );

            // add new group entity to database

            l_determinedGroup.setDB( EDataDB.INSTANCE.newGroup(l_determinedGroup.chair().name(), l_determinedGroup.getDB(), l_determinedGroup.getVoters(), m_run, m_sim ) );

            //    m_chairs.add( l_determinedGroup.chair() );
            return;
        }

        final CChairAgentCI l_chairAgent = m_chairagentgenerator.generatesinglenew();

        // if there was no available group, create a new group

        final CGroupCI l_group = new CGroupCI( p_votingAgent, l_chairAgent, m_groupNum++, m_capacity, new AtomicLong( m_timeout ) );
        m_groups.add( l_group );
        System.out.println( "Creating new group with agent " + p_votingAgent.name() + ", ID " + l_group.id() );

        p_votingAgent.beliefbase().add( CLiteral.from( "mygroup", CRawTerm.from( l_group ) ) );
        System.out.println( p_votingAgent.name() + " gets belief regarding group with id " + l_group.id() );

        p_votingAgent.beliefbase().add( CLiteral.from( "mychair", CRawTerm.from( l_chairAgent ) ) );
        System.out.println( p_votingAgent.name() + " gets belief regarding chair with chair name " + l_chairAgent.name() );

        p_votingAgent.addGroupID( l_group );
        p_votingAgent.setChair( l_chairAgent );
        System.out.println( "created chair " + l_chairAgent.name() );

        m_chairs.add( l_chairAgent );
        m_map.put( "chairNum", m_chairs.size() );

        // create group in database
        l_group.setDB( EDataDB.INSTANCE.addGroup( l_chairAgent.name(), p_votingAgent.name(), m_run, m_sim ) );

    }

    private int hammingDistance( final BitVector p_bitVote, final BitVector p_result )
    {
        System.out.println( "vote: " + p_bitVote + " committee: " + p_result );
        final BitVector l_diff = p_result.copy();
        l_diff.xor( p_bitVote );
        System.out.println( "diff: " + l_diff + " cardinality: " + l_diff.cardinality() );
        return l_diff.cardinality();
    }


    private int ranksum( final List<Long> p_voteCLO, final BitVector p_result )
    {
        System.out.println( "Vote: " + p_voteCLO );
        System.out.println( "Result: " + p_result );

        int l_sum = 0;

        for ( int i=0; i < m_altnum; i++ )
        {
            if ( p_result.get( i ) )
            {
                l_sum = l_sum + this.getPos( i, p_voteCLO );
            }
        }

        // normalisation: subtract (k*(k+1))/2
        l_sum = l_sum -( m_comsize * ( m_comsize + 1 ) )/2;

        return l_sum;
    }

    private int getPos( final int p_i, final List<Long> p_voteCLO )
    {
        for ( int j = 0; j < p_voteCLO.size(); j++ )
            if ( p_voteCLO.get( j ) == p_i )
                return ( j+1 );
        return 0;
    }

    // TODO refactor method

    @IAgentActionFilter
    @IAgentActionName( name = "update/groups" )
    private synchronized void updateGroups() throws Exception
    {
        try {
            boolean l_allReady = true;

            final CopyOnWriteArrayList<CGroupCI> l_cleanGroups = new CopyOnWriteArrayList<>();

            for (final CGroupCI l_group : m_groups) {
                // System.out.println( l_group.id() + "result: " + l_group.result() );

                // if chair is timed out or group is full, update info on current election

                if (l_group.timedout() || l_group.chair().full())

                    l_group.chair().updateElection();

                // TODO adapt in database accordingly

                if (l_group.result() == null)
                    l_allReady = false;

                if (l_group.areVotesSubmitted() || l_group.timedout()) {

                    // remove voters from group who didn't vote/whose votes didn't reach the chair
                    final CopyOnWriteArrayList<String> l_toRemoveList = new CopyOnWriteArrayList();
                    final CopyOnWriteArrayList<CVotingAgentCI> l_toRemoveAgents = new CopyOnWriteArrayList();
                    l_group.agents().filter(i -> !l_group.chair().voters().contains(i))
                            .forEach(j ->
                            {
                                l_toRemoveList.add(j.name());
                                l_toRemoveAgents.add(j);
                                m_lineHashMap.put(j.name(), j.liningCounter());
                            });
                    System.out.println(l_group.chair().name() + " toRemoveList:" + l_toRemoveList);

                    l_group.removeAll(l_toRemoveList);

                    // "re-queue" removed voters

                    l_toRemoveAgents.parallelStream().forEach(i ->
                            this.removeAndAddAg(i));

                    l_cleanGroups.add(l_group);

                    // if agents have been removed, add new group entry to database
                    if ( l_toRemoveList.size() > 0 )
                        l_group.setDB( EDataDB.INSTANCE.newGroup(l_group.chair().name(), l_group.getDB(), l_group.getVoters(), m_run, m_sim ) );

                }


                if (l_group.areDissValsSubmitted()) {
                    System.out.println("All diss vals are submitted");
                } else if (l_group.waitingforDiss())
                    if (l_group.dissTimedOut()) {
                        // if there are agents whose diss vals were not stored by the chair, remove them
                        final CopyOnWriteArrayList<String> l_toRemoveList = new CopyOnWriteArrayList();
                        final CopyOnWriteArrayList<CVotingAgentCI> l_toRemoveAgents = new CopyOnWriteArrayList();
                        l_group.agents().filter(i -> !l_group.chair().dissvoters().contains(i))
                                .forEach(
                                        j ->
                                        {
                                            l_toRemoveList.add(j.name());
                                            l_toRemoveAgents.add(j);
                                            //     m_lineHashMap.put( j.name(), j.liningCounter() );
                                        });
                        System.out.println("toRemoveList:" + l_toRemoveList);

                        l_group.removeAll(l_toRemoveList);

                        // "re-queue" removed voters

                        l_toRemoveAgents.parallelStream().forEach(
                                i -> this.removeAndAddAg(i)
                        );

                        l_group.endWaitForDiss();

                        // if agents have been removed, add new group entry to database
                        if ( l_toRemoveList.size() > 0 )
                            l_group.setDB( EDataDB.INSTANCE.newGroup(l_group.chair().name(), l_group.getDB(), l_group.getVoters(), m_run, m_sim ) );


                    }

                // check if chair is timedout, if yes, the chair needs to send the result to the agents to be sure that all agents received the final result

                if (l_group.timedout() && l_group.result() != null) {
                    l_group.chair().resendResult();
                }

            }


            if (l_allReady) {
                System.out.println("all groups ready");

                this.beliefbase().add(
                        CLiteral.from(
                                "allgroupsready",
                                CRawTerm.from(1)
                        )
                );
            }
        }
        catch ( IndexOutOfBoundsException l_ex )
        {
            System.out.println( "IndexOutOfBoundsException" );
            l_ex.printStackTrace();
        }
    }


    // old method

  /*   @IAgentActionFilter
    @IAgentActionName( name = "update/groups" )
    private synchronized void updateGroups() throws Exception
    {

        // TODO refactor method

        boolean l_allReady = true;

        final CopyOnWriteArrayList<CGroupCI> l_cleanGroups = new CopyOnWriteArrayList<>();

        for ( final CGroupCI l_group : m_groups )
        {
            System.out.println( l_group.id() + "result: " + l_group.result() );


            // if chair is timed out or group is full, update info on current election

            if ( l_group.timedout() || l_group.chair().full() )

                l_group.chair().updateElection();

            if ( l_group.result() == null )
                l_allReady = false;
            // if all voters have submitted their votes, there is nothing to check, group is clean

            //            if ( l_group.areVotesSubmitted() )
            //                l_group.chair().beliefbase().add(
            //                    CLiteral.from(
            //                        "cleangroup",
            //                        CRawTerm.from( 1 )
            //                    )
            //                );
            //
            //            else  if ( l_group.chair().timedout() )
            //
            //                // remove voters from group who didn't vote/whose votes didn't reach the chair
            //            {
            //
            //                final CopyOnWriteArrayList<String> l_toRemoveList = new CopyOnWriteArrayList();
            //                final CopyOnWriteArrayList<CVotingAgentCI> l_toRemoveAgents = new CopyOnWriteArrayList();
            //                l_group.agents().filter( i -> !l_group.chair().voters().contains( i ) )
            //                       .forEach( j ->
            //                       {
            //                           l_toRemoveList.add( j.name() );
            //                           l_toRemoveAgents.add( j );
            //                           m_lineHashMap.put( j, j.liningCounter() );
            //                       } );
            //                System.out.println( "XXXXXXX" + l_toRemoveList );
            //
            //                l_group.removeAll( l_toRemoveList );
            //
            //                // "re-queue" removed voters
            //
            //                l_toRemoveAgents.parallelStream().forEach( i -> this.beliefbase().add(
            //                    CLiteral.from(
            //                        "newag",
            //                        CRawTerm.from( i ),
            //                        CRawTerm.from( m_lineHashMap.get( i ) )
            //                    )
            //                                                 )
            //                );
            //
            //                // set belief in chair that group was "cleaned up"
            //
            //                l_group.chair().beliefbase().add(
            //                    CLiteral.from(
            //                        "cleangroup",
            //                        CRawTerm.from( 1 )
            //                    )
            //                );
            //            }


            if ( l_group.areVotesSubmitted() || l_group.timedout() )

            {
                // TODO test
                // remove voters from group who didn't vote/whose votes didn't reach the chair
                final CopyOnWriteArrayList<String> l_toRemoveList = new CopyOnWriteArrayList();
                final CopyOnWriteArrayList<CVotingAgentCI> l_toRemoveAgents = new CopyOnWriteArrayList();
                l_group.agents().filter( i -> !l_group.chair().voters().contains( i ) )
                       .forEach( j ->
                                 {
                                     l_toRemoveList.add( j.name() );
                                     l_toRemoveAgents.add( j );
                                     m_lineHashMap.put( j, j.liningCounter() );
                                 } );
                System.out.println( l_group.chair().name() + " toRemoveList:" + l_toRemoveList );

                l_group.removeAll( l_toRemoveList );

                // "re-queue" removed voters

                l_toRemoveAgents.parallelStream().forEach( i ->
                                                               this.removeAndAddAg( i ) );

                //this.beliefbase().add(

                //                        CLiteral.from(
                //                            "newag",
                //                            CRawTerm.from( i ),
                //                            CRawTerm.from( m_lineHashMap.get( i ) )
                //                        )
                //    )

                l_cleanGroups.add( l_group );
            }


            if ( l_group.areDissValsSubmitted() )
            {
                System.out.println( "All diss vals are submitted" );
            }

            else if ( l_group.waitingforDiss() )
                if ( l_group.dissTimedOut() )
            {
                //                // TODO refactor
                //                l_group.chair().determineDissVals();


                // TODO reinsert?
                // if there are agents whose diss vals were not stored by the chair, remove them
                final CopyOnWriteArrayList<String> l_toRemoveList = new CopyOnWriteArrayList();
                final CopyOnWriteArrayList<CVotingAgentCI> l_toRemoveAgents = new CopyOnWriteArrayList();
                l_group.agents().filter( i -> !l_group.chair().dissvoters().contains( i ) )
                       .forEach(
                           j ->
                           {
                               l_toRemoveList.add( j.name() );
                               l_toRemoveAgents.add( j );
                               m_lineHashMap.put( j, j.liningCounter() );
                           } );
                System.out.println( "toRemoveList:" + l_toRemoveList );

                l_group.removeAll( l_toRemoveList );

                // "re-queue" removed voters

                l_toRemoveAgents.parallelStream().forEach(
                    i -> this.removeAndAddAg( i )
                );

                l_group.endWaitForDiss();

            }

            // TODO refactor
            // check if chair is timedout, if yes, the chair needs to send the result to the agents to be sure that all agents received the final result

            if ( l_group.timedout() && l_group.result() != null )
            {
                l_group.chair().resendResult();
            }

        }


        if ( l_allReady )
        {
            System.out.println( "all groups ready" );

            this.beliefbase().add(
                CLiteral.from(
                    "allgroupsready",
                    CRawTerm.from( 1 )
                )
            );
        }
    }*/



    /**
     * add Ag
     * @param p_Ag agent
     */

    // TODO refactor


    public void removeAndAddAg( final CVotingAgentCI p_Ag )
    {
        p_Ag.trigger( CTrigger.from(
            ITrigger.EType.ADDGOAL,
            CLiteral.from(
                "leftgroup" )
                      )
        );

    //  p_Ag.beliefbase().remove( CLiteral.from( "mychair", CRawTerm.from( p_Ag.getChair() ) ) );

        System.out.println( "adding Agent " + p_Ag.name() );
        // increase lining counter of ag
        m_lineHashMap.put( p_Ag.name(), p_Ag.liningCounter() );

        this.beliefbase().add(
            CLiteral.from(
                "newag",
                CRawTerm.from( p_Ag ),
                CRawTerm.from( m_lineHashMap.get( p_Ag.name() ) )
            )
        );
    }

    /**
     * Class CBrokerAgentGenerator
     */
    public static class CBrokerAgentGenerator extends IBaseAgentGenerator<CBrokerAgentCI>
    {

        /**
         * Store reference to send action to registered agents upon creation.
         */
        private final CSendCI m_send;
        private final int m_agNum;
        private int m_count;
        private final InputStream m_stream;
        private final CEnvironmentCI m_environment;
        private final String m_name;
        private final double m_joinThr;
        private final List<AtomicDoubleArray> m_prefList;
        private final InputStream m_chairstream;
        private int m_altnum;
        private int m_comsize;
        private double m_dissthr;
        private String m_rule;
        private int m_run;
        private int m_sim;

        /**
         * constructor of CBrokerAgentGenerator
         * @param p_send external actions
         * @param p_brokerStream broker stream
         * @param p_agNum number of voters
         * @param p_stream input stream
         * @param p_chairStream chair stream
         * @param p_environment environment
         * @param p_altnum number of alternatives
         * @param p_name name
         * @param p_joinThr join threshold
         * @param p_prefList preference list
         * @param p_comsize committee size
         * @param p_dissthr
         * @param p_run
         * @param p_sim
         * @throws Exception exception
         */
        public CBrokerAgentGenerator(final CSendCI p_send,
                                     final FileInputStream p_brokerStream,
                                     final int p_agNum,
                                     final InputStream p_stream,
                                     final InputStream p_chairStream,
                                     final CEnvironmentCI p_environment,
                                     final int p_altnum,
                                     final String p_name,
                                     final double p_joinThr,
                                     final List<AtomicDoubleArray> p_prefList,
                                     final int p_comsize,
                                     final double p_dissthr,
                                     final String p_rule,
                                     int p_run, int p_sim) throws Exception
        {
            super(
                    // input ASL stream
                    p_brokerStream,

                    // a set with all possible actions for the agent
                    Stream.concat(
                        // we use all build-in actions of LightJason
                        CCommon.actionsFromPackage(),
                        Stream.concat(
                            // use the actions which are defined inside the agent class
                            CCommon.actionsFromAgentClass( CBrokerAgentCI.class ),
                            // add VotingAgent related external actions
                            Stream.of(
                                p_send
                            )
                        )
                        // build the set with a collector
                    ).collect( Collectors.toSet() ) );

            System.out.println( "actions defined in broker class: " + CCommon.actionsFromAgentClass( CBrokerAgentCI.class ).collect( Collectors.toSet() ) );


            // aggregation function for the optimization function, here
            // we use an empty function
            //         IAggregation.EMPTY,
            m_send = p_send;
            m_agNum = p_agNum;
            m_stream = p_stream;
            m_chairstream = p_chairStream;
            m_environment = p_environment;
            m_altnum = p_altnum;
            m_name = p_name;
            m_joinThr = p_joinThr;
            m_prefList = p_prefList;
            m_comsize = p_comsize;
            m_dissthr = p_dissthr;
            m_rule = p_rule;
            m_run = p_run;
            m_sim = p_sim;
        }

        @Nullable
        @Override
        public CBrokerAgentCI generatesingle( @Nullable final Object... p_data )
        {
            CBrokerAgentCI l_broker = null;
            try
            {
                l_broker = new CBrokerAgentCI(

                    // create a string with the agent name "agent <number>"
                    // get the value of the counter first and increment, build the agent
                    // name with message format (see Java documentation)
                    MessageFormat.format( "broker", 0 ),

                    // add the agent configuration
                    m_configuration,
                    m_agNum,
                    m_stream,
                    m_chairstream,
                    m_environment,
                    m_altnum,
                    m_name,
                    m_joinThr,
                    m_prefList,
                    m_comsize,
                    m_dissthr,
                    m_rule,
                    m_run,
                    m_sim);
            }
            catch ( final Exception l_ex )
            {
                l_ex.printStackTrace();
            }

            return l_broker;
        }


    }

}
