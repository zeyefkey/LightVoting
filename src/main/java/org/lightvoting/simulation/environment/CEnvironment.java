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

package org.lightvoting.simulation.environment;

import cern.colt.bitvector.BitVector;
import org.lightjason.agentspeak.agent.IBaseAgent;
import org.lightjason.agentspeak.language.CLiteral;
import org.lightjason.agentspeak.language.CRawTerm;
import org.lightjason.agentspeak.language.instantiable.plan.trigger.CTrigger;
import org.lightjason.agentspeak.language.instantiable.plan.trigger.ITrigger;
import org.lightvoting.simulation.agent.CChairAgent;
import org.lightvoting.simulation.agent.CVotingAgent;
import org.lightvoting.simulation.rule.CMinisumApproval;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicIntegerArray;


/**
 * Created by sophie on 22.02.17.
 * Environment class
 */
public final class CEnvironment
{

    // TODO set variables via config file or commando line
    // TODO use m_protocol?

 // private String m_protocol = "BASIC";

    private String m_protocol = "ITERATIVE";

    private String m_grouping = "RANDOM";

//  private String m_grouping = "COORDINATED";

    // TODO join threshold via config

    private final int m_joinThreshold = 0;

    private final Map<CChairAgent, int[]> m_groupResults;
    /**
     * group capacity
     */

    private final int m_capacity = 3;

    // private final AtomicReferenceArray<CVotingAgent> m_group;

    /**
     * Set of voting agents
     */

    private final Set<CVotingAgent> m_agents;

    /**
     * Map for storing agent groups
     */

    private final Map<CChairAgent, List<CVotingAgent>> m_chairgroup;

    private final Map<CChairAgent, List> m_voteSets;

    /**
     * List for storing current active chairs, i.e. chairs who accept more agents
     */

    private final List<CChairAgent> m_activechairs;

    /**
     * maximum size
     */
    private final int m_size;

    /**
     * indicates if new cycle can be triggered
     */
    private boolean m_ready = true;

    /**
     * number of cycles
     */

    private int m_cycles;
    private boolean m_resultComputed;
    private Map<CChairAgent, List<Double>> m_dissSets;


    /**
     * constructor
     *
     * @param p_size number of agents
     */
    public CEnvironment( final int p_size )
    {
        m_size = p_size;

        //  m_group = new AtomicReferenceArray<CVotingAgent>( new CVotingAgent[m_size] );
        m_agents = new HashSet<>();
        m_chairgroup = new HashMap<>();
        m_activechairs = new LinkedList<>();
        m_voteSets = new HashMap<CChairAgent, List>();
        m_groupResults = new HashMap<CChairAgent, int[]>();
        m_resultComputed = false;
        m_dissSets = new HashMap<>();

    }

    /**
     * initialize groups
     *
     * @param p_votingAgent agent
     *
     */
    public final void initialset( final CVotingAgent p_votingAgent )
    {
        m_agents.add( p_votingAgent );
    }

    /**
     * open a new group
     *
     * @param p_votingAgent voting opening the group
     */

    public final void openNewGroup( final CVotingAgent p_votingAgent )
    {
        final ITrigger l_triggerChair = CTrigger.from(
            ITrigger.EType.ADDGOAL,
            CLiteral.from(
                "myGroup",
                CLiteral.from( p_votingAgent.name() )
            )
        );

        p_votingAgent.getChair().trigger( l_triggerChair );


        final List l_list = new LinkedList<CVotingAgent>();
        l_list.add( p_votingAgent );

        m_chairgroup.put( p_votingAgent.getChair(), l_list );
        m_activechairs.add( p_votingAgent.getChair() );

        final List l_initalList = new LinkedList<AtomicIntegerArray>();
        m_voteSets.put( p_votingAgent.getChair(), l_initalList );

        final List l_initialDissList = new LinkedList<Double>();
        m_dissSets.put( p_votingAgent.getChair(), l_initialDissList );

        System.out.println( p_votingAgent.name() + " opened group with chair " + p_votingAgent.getChair() );


/*        final ITrigger l_trigger = CTrigger.from(

        final List l_initalList = new LinkedList<AtomicIntegerArray>();
        m_voteSets.put( p_votingAgent.getChair(), l_initalList );


        final ITrigger l_trigger = CTrigger.from(

            ITrigger.EType.ADDGOAL,
            CLiteral.from(
                "new/group/opened",
                CLiteral.from( p_votingAgent.name() ),
                CLiteral.from( ( p_votingAgent.getChair() ).toString() )
            )
        );


        // trigger all agents and tell them that the group was opened

        m_agents
            .parallelStream()

            .forEach( i -> i.trigger( l_trigger ) );*/

       //     .forEach( i -> i.trigger( l_trigger ) );

  /*      // TODO this is only for testing, here each agent looks for a group as soon as agent 0 opened its group. Needs to be rewritten for general case
        if ( m_chairgroup.size() == 1 )
        {
            final ITrigger l_triggerJoin = CTrigger.from(
                ITrigger.EType.ADDGOAL,
                CLiteral.from(
                    "lookforgroup" )

            );


            // trigger all agents and tell them to choose one of the available groups

            m_agents
                .parallelStream()
                .forEach( i -> i.trigger( l_triggerJoin ) );
        }*/


    }

    /**
     * join a group
     *
     * @param p_votingAgent voting agent joining a group
     * @return the chair of the group
     */

    public final CChairAgent joinGroup( final CVotingAgent p_votingAgent )
    {

        if ( "RANDOM".equals( m_grouping ) )
            return this.joinRandomGroup( p_votingAgent );

        if ( "COORDINATED".equals( m_grouping ) )
            return this.joinGroupCoordinated( p_votingAgent );
        return null;
    }

    public boolean getReady()
    {
        return m_ready;
    }

    /**
     * submit dissatisfaction regarding latest result to chair
     * @param p_votingAgent voting agent
     * @param p_chairAgent chair agent
     */
    public void submitDiss( final CVotingAgent p_votingAgent, final IBaseAgent<CChairAgent> p_chairAgent, final int p_iteration )
    {

        System.out.println( "submit diss for iteration " + p_iteration );
        final double l_diss = p_votingAgent.computeDiss( m_groupResults.get( p_chairAgent ) );

        if ( this.isChair( p_votingAgent, p_chairAgent ) )

        {
            final ITrigger l_trigger = CTrigger.from(
                ITrigger.EType.ADDGOAL,
                CLiteral.from(
                    "diss/received",
                    CLiteral.from( p_votingAgent.name() ),
                    //   CLiteral.from( p_chairAgent.toString() ),
                    CRawTerm.from( l_diss ),
                    CRawTerm.from( p_iteration )
                )
            );
            p_chairAgent.trigger( l_trigger );
        }

    }

    /**
     * store dissatisfaction value
     * @param p_chairAgent chair agent
     * @param p_diss dissatisfaction value
     */

    public void storeDiss( final CChairAgent p_chairAgent, final Double p_diss, final int p_iteration )
    {

        m_dissSets.get( p_chairAgent ).add( p_diss );

        System.out.println( "After iteration " + p_iteration + " number of agents " + m_dissSets.get( p_chairAgent ).size() );
        if ( m_dissSets.get( p_chairAgent ).size() == m_chairgroup.get( p_chairAgent ).size() )
        //m_capacity )
        {
            System.out.println( p_iteration + " All voters submitted their dissatisfaction value" );
            final ITrigger l_trigger = CTrigger.from(
                ITrigger.EType.ADDGOAL,
                CLiteral.from(
                    "all/dissatisfaction/received" )

            );

            p_chairAgent.trigger( l_trigger );

            System.out.println( " trying to recompute, old iteration  " + p_iteration );

            this.recomputeResult( p_chairAgent, p_iteration );
        }

    }

    private boolean isChair( final CVotingAgent p_votingAgent, final IBaseAgent<CChairAgent> p_chairAgent )
    {
        if ( m_chairgroup.get( p_chairAgent ).contains( p_votingAgent ) )
            return true;
        return false;
    }

    private final CChairAgent joinRandomGroup( final CVotingAgent p_votingAgent )
    {
        if ( m_activechairs.size() == 0 )
        {
            this.openNewGroup( p_votingAgent );
            return p_votingAgent.getChair();

        }

        // choose random group to join

        final Random l_rand = new Random();
        final CChairAgent l_randomChair = m_activechairs.get( l_rand.nextInt( m_activechairs.size() ) );

        if ( this.containsnot( l_randomChair, p_votingAgent ) )
        {

            m_chairgroup.get( l_randomChair ).add( p_votingAgent );
            System.out.println( p_votingAgent.name() + " joins group with chair " + l_randomChair );

            if ( m_chairgroup.get( l_randomChair ).size() == m_capacity )
            {

                m_activechairs.remove( l_randomChair );

                for ( int i = 0; i < m_capacity; i++ )
                    System.out.println( m_chairgroup.get( l_randomChair ).get( i ).name() + " with chair " + l_randomChair );

                System.out.println( "trigger election " );

                final ITrigger l_triggerStart = CTrigger.from(
                    ITrigger.EType.ADDGOAL,
                    CLiteral.from(
                        "start/criterion/fulfilled" )

                );

                l_randomChair.trigger( l_triggerStart );

            }
            return l_randomChair;
        }

        // if the agent is already in the group, just return the chair agent
        else
        {
            return l_randomChair;
        }

            //            final ITrigger l_trigger = CTrigger.from(
            //                ITrigger.EType.ADDGOAL,
            //                CLiteral.from(
            //                    "joined/group",
            //                    CLiteral.from( p_votingAgent.name() ),
            //                    CLiteral.from( l_randomChair.toString() )
            //                )
            //            );
            //
            //            // trigger all agents and tell them that the agent joined a group
            //            m_agents
            //                .parallelStream()
            //                .forEach( i -> i.trigger( l_trigger ) );


            // if it was not possible to join a group, open a new group

//        this.openNewGroup( p_votingAgent );
//        System.out.println( p_votingAgent.name() + " opened group with chair " + p_votingAgent.getChair() );
//        return p_votingAgent.getChair();

    }

    private final CChairAgent joinGroupCoordinated( final CVotingAgent p_votingAgent )
    {
        if ( m_activechairs.size() == 0 )
        {
            this.openNewGroup( p_votingAgent );

            final ITrigger l_triggerStart = CTrigger.from(
                ITrigger.EType.ADDGOAL,
                CLiteral.from(
                    "start/criterion/fulfilled" )

            );

            p_votingAgent.getChair().trigger( l_triggerStart );


            return p_votingAgent.getChair();
        }

        if ( m_groupResults.size() == 0 )
        {
            this.openNewGroup( p_votingAgent );
            System.out.println( p_votingAgent.name() + " opened group with chair " + p_votingAgent.getChair() );

            final ITrigger l_triggerStart = CTrigger.from(
                ITrigger.EType.ADDGOAL,
                CLiteral.from(
                    "start/criterion/fulfilled" )

            );

            p_votingAgent.getChair().trigger( l_triggerStart );

            return p_votingAgent.getChair();
        }

        // choose group to join
        final Map<CChairAgent, Integer> l_groupDistances = new HashMap<>();
        final AtomicIntegerArray l_vote = p_votingAgent.getVote();

        System.out.println( "Vote: " + l_vote );

        for ( int i = 0; i < m_activechairs.size(); i++ )
        {
            final AtomicIntegerArray l_com = new AtomicIntegerArray( m_groupResults.get( m_activechairs.get( i ) ) );

            System.out.println( "Committee: " + l_com );

            // TODO own class for Hamming distance computation

            final BitVector l_bitVote = new BitVector( l_vote.length() );
            final BitVector l_bitCom = new BitVector( l_vote.length() );

            for ( int j = 0; j < l_vote.length(); j++ )
                l_bitVote.put( j, l_vote.get( j ) == 1 );

            System.out.println( "Vote: " + l_bitVote );

            for ( int j = 0; j < l_vote.length(); j++ )
                l_bitCom.put( j, l_com.get( j ) == 1 );

            System.out.println( "Committee: " + l_bitCom );

            // final BitVector l_curBitCom = l_bitCom.copy();

            l_bitCom.xor( l_bitVote );

            final int l_HD = l_bitCom.cardinality();

            System.out.println( "Hamming distance: " + l_HD );

            l_groupDistances.put( m_activechairs.get( i ), l_HD );

        }

        final Map l_sortedDistances = this.sortMapDESC( l_groupDistances );

        final Map.Entry<CChairAgent, Integer> l_entry = (Map.Entry<CChairAgent, Integer>) l_sortedDistances.entrySet().iterator().next();

        final CChairAgent l_chair = l_entry.getKey();

        // if Hamming distance is above the threshold, do not join the chair but create a new group
        if ( l_entry.getValue() > m_joinThreshold )
        {
            this.openNewGroup( p_votingAgent );

            final ITrigger l_triggerStart = CTrigger.from(
                ITrigger.EType.ADDGOAL,
                CLiteral.from(
                    "start/criterion/fulfilled" )

            );

            p_votingAgent.getChair().trigger( l_triggerStart );

            return p_votingAgent.getChair();
        }

        System.out.println( " Chosen chair " + l_chair );

        if ( this.containsnot( l_chair, p_votingAgent ) )
        {

            m_chairgroup.get( l_chair ).add( p_votingAgent );
       //     System.out.println( p_votingAgent.name() + " joins group with chair " + l_chair );

            if ( m_chairgroup.get( l_chair ).size() == m_capacity )
            {
                System.out.println( " XXXXXXXXXXXXXXXX Capacity reached " );

                m_activechairs.remove( l_chair );

            //    for ( int i = 0; i < m_capacity; i++ )
              //      System.out.println( m_chairgroup.get( l_chair ).get( i ).name() + " with chair " + l_chair );

          //      System.out.println( "trigger election " );

                final ITrigger l_triggerStart = CTrigger.from(
                    ITrigger.EType.ADDGOAL,
                    CLiteral.from(
                        "start/criterion/fulfilled" )

                );

                l_chair.trigger( l_triggerStart );

            }
            return l_chair;






      /*      final BitVector l_bitVote = new BitVector( p_altNum );

        for ( int j = 0; j < p_altNum; j++ )
        {
            l_bitVote.put( j, l_booleanVote[j] );
        }

        final BitVector l_curBitCom = l_bitCom.copy();

        l_curBitCom.xor( l_bitVote );

        final int l_curHD = l_curBitCom.cardinality();*/

            //        final CChairAgent l_randomChair = m_activechairs.get( l_rand.nextInt( m_activechairs.size() ) );
            //
            //
            //
            //        if ( this.containsnot( l_randomChair, p_votingAgent ) )
            //        {
            //
            //            m_chairgroup.get( l_randomChair ).add( p_votingAgent );
            //            System.out.println( p_votingAgent.name() + " joins group with chair " + l_randomChair );
            //
            //            if ( m_chairgroup.get( l_randomChair ).size() == m_capacity )
            //            {
            //
            //                m_activechairs.remove( l_randomChair );
            //
            //                for ( int i = 0; i < m_capacity; i++ )
            //                    System.out.println( m_chairgroup.get( l_randomChair ).get( i ).name() + " with chair " + l_randomChair );
            //
            //                System.out.println( "trigger election " );
            //
            //                final ITrigger l_triggerStart = CTrigger.from(
            //                    ITrigger.EType.ADDGOAL,
            //                    CLiteral.from(
            //                        "start/criterion/fulfilled" )
            //
            //                );
            //
            //                l_randomChair.trigger( l_triggerStart );
            //
            //            }
            //            return l_randomChair;
            //        }
        }
        // if the agent is already in the group, just return the chair of the agent
        else
        {
            return p_votingAgent.getChair();
        }

//        this.openNewGroup( p_votingAgent );
//        System.out.println( p_votingAgent.name() + " opened group with chair " + p_votingAgent.getChair() );
//        return p_votingAgent.getChair();

    }

    private Map sortMapDESC( final Map<CChairAgent, Integer> p_valuesMap )

    {

        final List<Map.Entry<CChairAgent, Integer>> l_list = new LinkedList<>( p_valuesMap.entrySet() );

        /* Sorting the list based on values in descending order */
        Collections.sort( l_list, ( p_first, p_second ) ->
            p_second.getValue().compareTo( p_first.getValue() ) );

        /* Maintaining insertion order with the help of LinkedList */
        final Map<CChairAgent, Integer> l_sortedMap = new LinkedHashMap<>();
        for ( final Map.Entry<CChairAgent, Integer> l_entry : l_list )
        {
            l_sortedMap.put( l_entry.getKey(), l_entry.getValue() );
        }

        return l_sortedMap;
    }


    private boolean containsnot( final CChairAgent p_randomChair, final CVotingAgent p_votingAgent )
    {
        for ( int i = 0; i < m_chairgroup.get( p_randomChair ).size(); i++ )
            if ( m_chairgroup.get( p_randomChair ).get( i ).name().equals( p_votingAgent.name() ) )
            {
                return false;
            }

        return true;
    }


    public final int size()
    {
        return m_size;
    }

    /**
     * start the election, i.e. ask the travellers
     * @param p_chairAgent the chair starting the election
     */
    public void startElection( final CChairAgent p_chairAgent )
    {
        final ITrigger l_trigger = CTrigger.from(
            ITrigger.EType.ADDGOAL,
            CLiteral.from(
                "submit/your/vote", CRawTerm.from( p_chairAgent  ) )
        );

        final List<CVotingAgent> l_agents = m_chairgroup.get( p_chairAgent );

        l_agents.forEach( i -> i.trigger( l_trigger ) );

    }

    // TODO migrate to CVote ?

    /**
     * submit vote to chair
     * @param p_votingAgent voting agent
     * @param p_chairAgent chair agent
     */
    public void submitVote( final CVotingAgent p_votingAgent, final IBaseAgent<CChairAgent> p_chairAgent )
    {
        final AtomicIntegerArray l_vote = new AtomicIntegerArray( new int[] {1, 1, 1, 0, 0, 0} );
        System.out.println( "Agent " + p_votingAgent.name() + " sends vote " + l_vote + " to " + p_chairAgent.toString() );

        final ITrigger l_trigger = CTrigger.from(
            ITrigger.EType.ADDGOAL,
            CLiteral.from(
                "vote/received",
                CLiteral.from( p_votingAgent.name() ),
                //   CLiteral.from( p_chairAgent.toString() ),
                CRawTerm.from( l_vote )
            )
        );

        p_chairAgent.trigger( l_trigger );

    }

    /**
     * store vote submitted to chair
     * @param p_chairAgent chair agent to whom vote is submitted
     * @param p_votingAgent voting agent submitting the vote
     * @param p_vote submitted vote
     */

    public void storeVote( final CChairAgent p_chairAgent, final Object p_votingAgent, final AtomicIntegerArray p_vote )
    {

        m_voteSets.get( p_chairAgent ).add( p_vote );
        if ( ( m_voteSets.get( p_chairAgent ) ).size() == m_chairgroup.get( p_chairAgent ).size() )
            //m_capacity )
        {
            System.out.println( " All voters submitted their votes" );
            final ITrigger l_trigger = CTrigger.from(
                ITrigger.EType.ADDGOAL,
                CLiteral.from(
                    "all/votes/received" )

            );

            p_chairAgent.trigger( l_trigger );
        }
    }

    // TODO move computeResult() to chair class or chair action

    /**
     * compute result of election
     * @param p_chairAgent responsible chair
     */

    public void computeResult( final CChairAgent p_chairAgent )
    {
        System.out.println( "Computing result " );
        final CMinisumApproval l_minisumApproval = new CMinisumApproval();

        final List<String> l_alternatives = new LinkedList<>();

        // TODO remove ugly hack

        for ( char l_char: "ABCDEF".toCharArray() )

            l_alternatives.add( String.valueOf( l_char ) );

        // TODO specify comsize via config file

        System.out.println( " Alternatives: " + l_alternatives );

        System.out.println( " Votes: " + m_voteSets.get( p_chairAgent ) );

        final int[] l_comResult = l_minisumApproval.applyRule( l_alternatives, m_voteSets.get( p_chairAgent ), 3 );

        m_groupResults.put( p_chairAgent, l_comResult );

        m_resultComputed = true;

        // set Ready to true

        m_ready = true;

        System.out.println( " Result of election: " + Arrays.toString( l_comResult ) );

        // broadcast result

        if ( "BASIC".equals( m_protocol ) )
        {
            final ITrigger l_trigger = CTrigger.from(
                ITrigger.EType.ADDGOAL,
                CLiteral.from(
                    "election/result",
                    CRawTerm.from( p_chairAgent ),
                    CRawTerm.from( Arrays.toString( l_comResult ) )
                )
            );

            m_agents.stream().forEach( i -> i.trigger( l_trigger ) );
        }

        else if ( "ITERATIVE".equals( m_protocol ) )
        {

            final ITrigger l_iterativeTrigger = CTrigger.from(
                ITrigger.EType.ADDGOAL,
                CLiteral.from(
                    "election/result",
                    CRawTerm.from( p_chairAgent ),
                    CRawTerm.from( Arrays.toString( l_comResult ) ),
                    CRawTerm.from( 0 )
                )
            );

            m_agents.stream().forEach( i -> i.trigger( l_iterativeTrigger ) );
          //  p_chairAgent.trigger( l_chairTrigger );
        }
    }

    /**
     * recompute result of election regarding to criteria of iterative election
     * @param p_chairAgent chair agent conducting the election
     * @param p_iteration number of current iteration
     */

    public void recomputeResult( final CChairAgent p_chairAgent, final int p_iteration )
    {
        final int l_newIteration = p_iteration + 1;
        System.out.println( " --------------- Recomputing result " + " iteration " + l_newIteration  + "  --------------- " +  p_chairAgent );

        System.out.println( "Computing result " );
        final CMinisumApproval l_minisumApproval = new CMinisumApproval();

        final List<String> l_alternatives = new LinkedList<>();

        for ( char l_char: "ABCDEF".toCharArray() )

            l_alternatives.add( String.valueOf( l_char ) );

        System.out.println( " Alternatives: " + l_alternatives );

        this.removeVoter( p_chairAgent );

        if ( m_voteSets.get( p_chairAgent ).isEmpty() )
        {
            System.out.println( " Voter list is empty " );
            return;
        }

        System.out.println( " Votes: " +   m_voteSets.get( p_chairAgent ) );

        final int[] l_comResult = l_minisumApproval.applyRule( l_alternatives,  m_voteSets.get( p_chairAgent ), 3 );

        m_groupResults.put( p_chairAgent, l_comResult );

        System.out.println( " Result of iteration " + l_newIteration + ": " + Arrays.toString( l_comResult ) );

        final ITrigger l_trigger = CTrigger.from(
            ITrigger.EType.ADDGOAL,
            CLiteral.from(
                "election/result",
                CLiteral.from( p_chairAgent.toString() ),
                CRawTerm.from( Arrays.toString( l_comResult ) ),
                CRawTerm.from( l_newIteration )
            )
        );

        m_agents.stream().forEach( i -> i.trigger( l_trigger ) );
        // p_chairAgent.trigger( l_chairTrigger );

    }

    private void removeVoter( final CChairAgent p_chairAgent )
    {
        m_voteSets.get( p_chairAgent ).remove( 0 );
    }

    public void setReady( final boolean p_ready )
    {
        m_ready = p_ready;
    }

    public void setCycles( final int p_cycles )
    {
        m_cycles = p_cycles;
    }

    public boolean getResultComputed()
    {
        return m_resultComputed;
    }

    public void setResultComputed( boolean p_resultComputed )
    {
        m_resultComputed = p_resultComputed;
    }
}
