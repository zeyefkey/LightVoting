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

package org.lightvoting.simulation.agent;

import org.lightjason.agentspeak.action.binding.IAgentAction;
import org.lightjason.agentspeak.action.binding.IAgentActionFilter;
import org.lightjason.agentspeak.action.binding.IAgentActionName;
import org.lightjason.agentspeak.agent.IBaseAgent;
import org.lightjason.agentspeak.configuration.IAgentConfiguration;
import org.lightjason.agentspeak.language.CLiteral;
import org.lightjason.agentspeak.language.CRawTerm;
import org.lightjason.agentspeak.language.ILiteral;
import org.lightjason.agentspeak.language.instantiable.plan.trigger.CTrigger;
import org.lightjason.agentspeak.language.instantiable.plan.trigger.ITrigger;
import org.lightvoting.simulation.environment.CEnvironment;
import org.lightvoting.simulation.environment.CGroup;
import org.lightvoting.simulation.rule.CMinisumApproval;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;


/**
 * Created by sophie on 21.02.17.
 */

// annotation to mark the class that actions are inside
@IAgentAction
public final class CChairAgent extends IBaseAgent<CChairAgent>
{

    /**
     * name of chair
     */
    private final String m_name;

    /**
     * environment
     */

    private final CEnvironment m_environment;
    private List<AtomicIntegerArray> m_votes;
    private List<CVotingAgent> m_agents;

    // TODO define via config file
    /**
     * grouping algorithm: "RANDOM" or "COORDINATED"
     */
    private String m_grouping;

    private int m_iteration;
    private String m_protocol;
    private List<Double> m_dissList;
    private List<CVotingAgent> m_dissVoters;
    // TODO via config file
    private double m_dissThreshold = 1.1;
    private boolean m_iterative;

    /**
     * constructor of the agent
     *
     * @param p_configuration agent configuration of the agent generator
     * @param p_grouping grouping algorithm
     * @param p_protocol voting protocol
     */


    public CChairAgent( final String p_name, final IAgentConfiguration<CChairAgent> p_configuration, final CEnvironment p_environment, final String p_grouping,
                        final String p_protocol
    )
    {
        super( p_configuration );
        m_name = p_name;
        m_environment = p_environment;
        m_votes = Collections.synchronizedList( new LinkedList<>() );
        m_dissList = Collections.synchronizedList( new LinkedList<>() );
        m_dissVoters = Collections.synchronizedList( new LinkedList<>() );
        m_grouping = p_grouping;
        m_protocol = p_protocol;
        m_iteration = 0;
        m_agents = Collections.synchronizedList( new LinkedList<>() );
        m_iterative = false;
    }

    // overload agent-cycle
    @Override
    public final CChairAgent call() throws Exception
    {
        // run default cycle
        return super.call();
    }

    public String name()
    {
        return m_name;
    }


    /**
     * perceive group
     */
    @IAgentActionFilter
    @IAgentActionName( name = "perceive/group" )
    /**
     * add literal for group of chair agent if it exists
     */
    public void perceiveGroup()
    {
        if ( !( m_environment.detectGroup( this ) == null ) )
            this.beliefbase().add( m_environment.detectGroup( this ) );
    }

    /**
     * check conditions
     */
    @IAgentActionFilter
    @IAgentActionName( name = "check/conditions" )
    /**
     * add literal for group of chair agent if it exists
     */
    public void checkConditions()
    {
        final CGroup l_group = this.determineGroup();

        // if conditions for election are fulfilled, trigger goal start/criterion/fulfilled

        final ITrigger l_trigger;

        // if m_iterative is true, we have the case of iterative voting, i.e. we already have the votes
        // we only need to repeat the computation of the result

        if  ( m_iterative && ( l_group.readyForElection() && !( l_group.electionInProgress() ) ) )
        {
            m_iteration++;
            this.computeResult();
            return;
        }


        if ( l_group.readyForElection() && ( !( l_group.electionInProgress() ) ) )
        {
            l_group.startProgress();

            l_trigger = CTrigger.from(
                ITrigger.EType.ADDGOAL,
                CLiteral.from( "start/criterion/fulfilled" )
            );

            this.trigger( l_trigger );
        }
    }

    private CGroup determineGroup()
    {
        final AtomicReference<CGroup> l_groupAtomic = new AtomicReference<>();
        final Collection l_groups = this.beliefbase().beliefbase().literal( "group" );
        l_groups.stream().forEach( i -> l_groupAtomic.set( ( (ILiteral) i ).values().findFirst().get().raw() ) );
        return l_groupAtomic.get();
    }


    /**
     * start election
     */
    @IAgentActionFilter
    @IAgentActionName( name = "start/election" )

    public void startElection()
    {
        final CGroup l_group = this.determineGroup();
        l_group.triggerAgents( this );
    }

    /**
     * store vote
     *
     * @param p_vote vote
     */
    @IAgentActionFilter
    @IAgentActionName( name = "store/vote" )
    public void storeVote( final String p_agentName, final AtomicIntegerArray p_vote )
    {
        final CGroup l_group = this.determineGroup();

        m_agents.add( l_group.determineAgent( p_agentName ) );
        m_votes.add( p_vote );

        if ( m_votes.size() == l_group.size() )
        {

            final ITrigger l_trigger = CTrigger.from(
                ITrigger.EType.ADDGOAL,
                CLiteral.from(
                    "all/votes/received" )

            );

            System.out.println( " CChairAgent.java: all votes received " );

            this.trigger( l_trigger );
        }
    }

    /**
     * store dissatisfaction value
     *
     * @param p_diss dissatisfaction value
     * @param p_iteration iteration number
     */
    @IAgentActionFilter
    @IAgentActionName( name = "store/diss" )

    public void storeDiss( final String p_name, final Double p_diss, final Integer p_iteration )
    {
        final CGroup l_group = this.determineGroup();

        m_dissList.add( p_diss );
        final CVotingAgent l_dissAg = l_group.determineAgent( p_name );
        m_dissVoters.add( l_dissAg );

        System.out.println( "Storing diss " + p_diss );

        if ( m_dissList.size() == l_group.size() )
        {
            final ITrigger l_trigger = CTrigger.from(
                ITrigger.EType.ADDGOAL,
                CLiteral.from(
                    "all/dissValues/received",
                    CRawTerm.from( p_iteration )
                )

            );

            this.trigger( l_trigger );

            System.out.println( p_iteration + " All voters submitted their dissatisfaction value" );
        }
    }


    /**
     * compute result of election
     */

    // TODO Minisum via parameter
    // TODO Alternatives via parameter/environment
    // TODO specify comsize via parameter
    // TODO see HDF5 structure in old code in CEnvironment
    @IAgentActionFilter
    @IAgentActionName( name = "compute/result" )

    public void computeResult()
    {
        final CGroup l_group = this.determineGroup();

        final CMinisumApproval l_minisumApproval = new CMinisumApproval();

        final List<String> l_alternatives = new LinkedList<>();

        for ( char l_char : "ABCDEF".toCharArray() )

            l_alternatives.add( String.valueOf( l_char ) );

        System.out.println( " Alternatives: " + l_alternatives );

        System.out.println( " Votes: " + m_votes );

        final int[] l_comResult = l_minisumApproval.applyRule( l_alternatives, m_votes, 3 );

        System.out.println( " Result of election: " + Arrays.toString( l_comResult ) );

        // set inProgress and readyForElection to false in group
        l_group.reset();

        if ( "BASIC".equals( m_protocol ) )
        {
            this.beliefbase().add( l_group.updateBasic( this, l_comResult ) );
        }


        // for the iterative case, you need to differentiate between the final election and intermediate elections.

        if ( "ITERATIVE".equals( m_protocol ) && ( !l_group.finale() && !m_iterative ) )
        {

            this.beliefbase().add( l_group.updateBasic( this, l_comResult ) );
        }

        if ( "ITERATIVE".equals( m_protocol ) && ( l_group.finale() ) || m_iterative )
        {

            this.beliefbase().add( l_group.updateIterative( this, l_comResult, m_iteration ) );
    //        m_iteration++;
        }

        // if grouping is coordinated, reopen group for further voters
        if ( "COORDINATED".equals( m_grouping ) && !l_group.finale() )
            m_environment.reopen( l_group );

        // TODO watch out with case coordinated grouping and iterative voting

    }

    /**
     * remove most dissatisfied voter
     */
    @IAgentActionFilter
    @IAgentActionName( name = "remove/voter" )
    public void removeVoter()
    {
        final CGroup l_group = this.determineGroup();

        final int l_maxIndex = this.getMaxIndex( m_dissList );
        final double l_max = m_dissList.get( l_maxIndex );
        System.out.println( " max diss is " + l_max );

        if ( l_max > m_dissThreshold )
        {
            System.out.println( " Determining most dissatisfied voter " );
            final CVotingAgent l_maxDissAg = m_dissVoters.get( l_maxIndex );
            System.out.println( " Most dissatisfied voter is " + l_maxDissAg.name() );
            // remove vote of most dissatisfied voter from list
            m_votes.remove( l_maxDissAg.getVote() );
            m_dissVoters.remove( l_maxDissAg );
            l_group.remove( l_maxDissAg );

            System.out.println( "Removing " + l_maxDissAg.name() );

            // remove diss Values for next iteration
            m_dissList.clear();

            m_iterative = true;
            l_group.makeReady();

            if ( l_group.size() == 0 )
                System.out.println( " Voter list is empty, we are done " );

            return;
        }

        System.out.println( " No dissatisfied voter left, we are done " );
    }

    private int getMaxIndex( final List<Double> p_dissValues )
    {
        int l_maxIndex = 0;
        for ( int i = 0; i < p_dissValues.size(); i++ )
        {
            if ( p_dissValues.get( i ) > p_dissValues.get( l_maxIndex ) )
            {
                System.out.println( " changed max index to " + i + " diss: " + p_dissValues.get( i ) );
                l_maxIndex = i;
            }
        }
        return l_maxIndex;
    }
}

// XXXXXXXXXXXXXXXXXXXXXXXX Old code XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
// TODO if necessary, reinsert into checkConditions()

/*            System.out.println( ".................. print group  " + i );
            System.out.println( "Contents of group " + ( (ILiteral) i ).values().findFirst().get().raw() );
            System.out.println( "Class " + ( (ILiteral) i ).values().findFirst().get().raw().getClass() );*/


/*    @IAgentActionFilter
    @IAgentActionName( name = "start/election" )
    public void startElection( )
    {
        m_environment.startElection( this );
    }*/

  /*  @IAgentActionFilter
    @IAgentActionName( name = "store/vote" )
    private void storeVote( final Object p_votingAgent, final AtomicIntegerArray p_vote )
    {

        System.out.println( " trying to add vote from agent " + p_votingAgent + ": " + p_vote );
        m_environment.storeVote( this, p_votingAgent, p_vote );
        System.out.println( " added vote from agent " + p_votingAgent );

    }*/

/*    @IAgentActionFilter
    @IAgentActionName( name = "compute/result" )
    private void computeResult( )
    {

        System.out.println( " compute result " );
        m_environment.computeResult( this );
//        System.out.println( " computed result " );

    }*/

  /*  @IAgentActionFilter
    @IAgentActionName( name = "store/diss" )
    private void storeDiss( final Object p_votingAgent, final Double p_diss, final int p_iteration )
    {

        System.out.println( " trying to add diss from agent " + p_votingAgent + ": " + p_diss + " next iteration " + p_iteration );
        m_environment.storeDiss( this, p_diss, p_iteration );
        System.out.println( " added diss from agent " + p_votingAgent );

    }*/

/*    @IAgentActionFilter
    @IAgentActionName( name = "recompute/result" )
    private void recomputeResult( final int p_iteration )
    {

        System.out.println( " recompute result " );
        m_environment.recomputeResult( this, p_iteration );
        //        System.out.println( " computed result " );

    }*/

//    final ITrigger l_ack = CTrigger.from(
//        ITrigger.EType.ADDGOAL,
//        CLiteral.from( "ack"
//
//        )
//
//    );
//
//        p_votingAgent.trigger( l_ack );

