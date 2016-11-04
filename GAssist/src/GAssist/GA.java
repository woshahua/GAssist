/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. S逍｣chez (luciano@uniovi.es)
    J. Alcal�ｽFdez (jalcala@decsai.ugr.es)
    S. Garc蜒�(sglopez@ujaen.es)
    A. Fern逍｣dez (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see http://www.gnu.org/licenses/
  
**********************************************************************/

/**
 * <p>
 * @author Written by Jaume Bacardit (La Salle, Ram詹｢ Llull University - Barcelona) 28/03/2004
 * @author Modified by Xavi Sol�ｽ(La Salle, Ram詹｢ Llull University - Barcelona) 23/12/2008
 * @version 1.1
 * @since JDK1.2
 * </p>
 */


package GAssist;

import java.util.*;
import java.lang.*;
import keel.Algorithms.Genetic_Rule_Learning.Globals.*;
import java.lang.Math;

public class GA {

  Classifier[] population;
  Classifier[] bestIndiv;
  int numVersions;

  /** Creates a new instance of GA */
  public GA() {
  }

  /**
   *  Prepares GA for a new run.
   */
  public void initGA() {
    //Init population
    population = new Classifier[Parameters.popSize];
    PopulationWrapper.initInstancesEvaluation();

    numVersions = PopulationWrapper.numVersions();
    bestIndiv = new Classifier[numVersions];

    Factory.initialize();
    initPopulation(population);
    Statistics.initStatistics();
  }

  /**
   *  Inits a new population.
   *  @param _population Population to init.
   */
  private void initPopulation(Classifier[] _population) {
    for (int i = 0; i < Parameters.popSize; i++) {
      _population[i] = Factory.newClassifier();
      _population[i].initRandomClassifier();
    }
  }

  public void checkBestIndividual() {
    Classifier best = PopulationWrapper.getBest(population);
    int currVer = PopulationWrapper.getCurrentVersion();

    if (bestIndiv[currVer] == null) {
      bestIndiv[currVer] = best.copy();
    }
    else {
      if (best.compareToIndividual(bestIndiv[currVer])) {
        bestIndiv[currVer] = best.copy();
      }
    }
  }

  /**
   *  Executes a number of iterations of GA.
   */
  public void run() {
    Classifier[] offsprings;
    Classifier[] midsprings = new Classifier[Parameters.popSize];//NSGA2
    Classifier[] tempsprings= new Classifier[Parameters.popSize];//NSGA2
    
    
    //add to sum the rule number of every iteration
    int[][] SumRuleNum=new int[Parameters.numIterations][];
    for(int i=0;i<Parameters.numIterations;i++){
      SumRuleNum[i]=new int[20];
    }
    
    
    PopulationWrapper.doEvaluation(population);

    int numIterations = Parameters.numIterations;

    for (int iteration = 0; iteration < numIterations; iteration++) {
      boolean lastIteration = (iteration == numIterations - 1);
      Parameters.percentageOfLearning = (double) iteration
          / (double) numIterations;
      boolean res1 = PopulationWrapper.initIteration();
      boolean res2 = Timers.runTimers(iteration, population);
      if (res1 || res2) {
        PopulationWrapper.setModified(population);
      }
      
      
      // GA cycle
      //-------------------NSGA2---------------------------
      for(int i=0;i<Parameters.popSize;i++){
        midsprings[i]=population[i].copy();
        tempsprings[i]=population[i].copy();
      }
      //--------------------NSGA2--------------------------
      population = doTournamentSelection(population);
      offsprings = doCrossover(population);
      doMutation(offsprings);
      PopulationWrapper.doEvaluation(offsprings);
      PopulationWrapper.doEvaluation(midsprings);
      checkBestIndividual();
      population = replacementPolicy(offsprings,midsprings,tempsprings, lastIteration);
     
      
      
      for(int i=0;i<Parameters.popSize;i++){
        double rulenum=population[i].getNumAliveRules();
        if(SumRuleNum[iteration][(int)(rulenum-1)]>=0){
          SumRuleNum[iteration][(int)(rulenum-1)]+=1;
        }
        else{
          SumRuleNum[iteration][(int)(rulenum-1)]=1;       
        }
      }
      
      Statistics.computeStatistics(population);
      Timers.runOutputTimers(iteration, population);
    }

    Statistics.statisticsToFile();
    Classifier best = PopulationWrapper.getBest(population);

    
    for(int i=0;i<Parameters.numIterations;i++){
      LogManager.println("-----------iteration"+i+"------------");
      for(int j=0;j<20;j++){
        LogManager.println(j+":"+(double)(SumRuleNum[i][j])/Parameters.popSize+"\n");
      }
    }
    
    double [] fitness1=new double [Parameters.popSize];
    double [] fitness2=new double [Parameters.popSize];
    double [] testfit1=new double [Parameters.popSize];
    double [] testfit2=new double [Parameters.popSize];
    
    LogManager.println("\nPhenotype: ");
    best.printClassifier();
    //-----------------------NSGA2-PART----------------------------------------
    
    for(int i=0;i<Parameters.popSize;i++){
    PopulationWrapper.testClassifier(i,fitness1,fitness2,population[i],"training",Parameters.train2InputFile,Parameters.trainOutputFile);
    PopulationWrapper.testClassifier(i,testfit1,testfit2,population[i],"test",Parameters.testInputFile,Parameters.testOutputFile);
    }
    
    NSGA2 mt=new NSGA2(Parameters.popSize);
    mt.rank(fitness1, fitness2);
    
  LogManager.println("final result");
    for(int i=0;i<Parameters.popSize;i++){
      if(mt.population_rank[i]==0){
        LogManager.println(fitness2[i]+" "+fitness1[i]+" "+testfit1[i]);
      }
    }
  }

  Classifier[] doCrossover(Classifier[] _population) {
    Chronometer.startChronCrossover();

    int i, j, k, countCross = 0;
    int numNiches = _population[0].getNumNiches();
    ArrayList[] parents = new ArrayList[numNiches];
    Classifier parent1, parent2;
    Classifier[] offsprings = new Classifier[2];
    Classifier[] offspringPopulation = new Classifier[Parameters.popSize];

    for (i = 0; i < numNiches; i++) {
      parents[i] = new ArrayList();
      parents[i].ensureCapacity(Parameters.popSize);
    }

    for (i = 0; i < Parameters.popSize; i++) {
      int niche = _population[i].getNiche();
      parents[niche].add(new Integer(i));
    }

    for (i = 0; i < numNiches; i++) {
      int size = parents[i].size();
      Sampling samp = new Sampling(size);
      int p1 = -1;
      for (j = 0; j < size; j++) {
        if (Rand.getReal() < Parameters.probCrossover) {
          if (p1 == -1) {
            p1 = samp.getSample();
          }
          else {
            int p2 = samp.getSample();
            int pos1 = ( (Integer) parents[i].get(p1)).intValue();
            int pos2 = ( (Integer) parents[i].get(p2)).intValue();
            parent1 = _population[pos1];
            parent2 = _population[pos2];

            offsprings = parent1.crossoverClassifiers(parent2);
            offspringPopulation[countCross++] = offsprings[0];
            offspringPopulation[countCross++] = offsprings[1];
            p1 = -1;
          }
        }
        else {
          int pos = ( (Integer) parents[i].get(samp.getSample())).intValue();
          offspringPopulation[countCross++] = _population[pos].copy();
        }
      }
      if (p1 != -1) {
        int pos = ( (Integer) parents[i].get(p1)).intValue();
        offspringPopulation[countCross++] = _population[pos].copy();
      }
    }

    Chronometer.stopChronCrossover();
    return offspringPopulation;
  }

  private int selectNicheWOR(int[] quotas) {
    int num = quotas.length;
    if (num == 1) {
      return 0;
    }

    int total = 0, i;
    for (i = 0; i < num; i++) {
      total += quotas[i];
    }
    if (total == 0) {
      return Rand.getInteger(0, num - 1);
    }
    int pos = Rand.getInteger(0, total - 1);
    total = 0;
    for (i = 0; i < num; i++) {
      total += quotas[i];
      if (pos < total) {
        quotas[i]--;
        return i;
      }
    }

    LogManager.printErr("We should not be here");
    System.exit(1);
    return -1;
  }

  private void initPool(ArrayList pool, int whichNiche,
                        Classifier[] _population) {
    if (Globals_DefaultC.nichingEnabled) {
      for (int i = 0; i < Parameters.popSize; i++) {
        if (_population[i].getNiche() == whichNiche) {
          pool.add(new Integer(i));
        }
      }
    }
    else {
      for (int i = 0; i < Parameters.popSize; i++) {
        pool.add(new Integer(i));
      }
    }
  }

  private int selectCandidateWOR(ArrayList pool, int whichNiche,
                                 Classifier[] _population) {
    if (pool.size() == 0) {
      initPool(pool, whichNiche, population);
      if (pool.size() == 0) {
        return Rand.getInteger(0, Parameters.popSize - 1);
      }
    }

    int pos = Rand.getInteger(0, pool.size() - 1);
    int elem = ( (Integer) pool.get(pos)).intValue();
    pool.remove(pos);
    return elem;
  }

  /**
   *  Does Tournament Selection without replacement.
   */
  public Classifier[] doTournamentSelection(Classifier[] _population) {
    Chronometer.startChronSelection();

    Classifier[] selectedPopulation;
    selectedPopulation = new Classifier[Parameters.popSize];
    int i, j, winner, candidate;
    int numNiches;
    if (Globals_DefaultC.nichingEnabled) {
      numNiches = _population[0].getNumNiches();
    }
    else {
      numNiches = 1;
    }

    ArrayList[] pools = new ArrayList[numNiches];
    for (i = 0; i < numNiches; i++) {
      pools[i] = new ArrayList();
    }
    int[] nicheCounters = new int[numNiches];
    int nicheQuota = Parameters.popSize / numNiches;
    for (i = 0; i < numNiches; i++) {
      nicheCounters[i] = nicheQuota;
    }
    
    //--------------------EMO-nsga2-PART-----------------------------------
    double [] fitness1=new double[Parameters.popSize];
    double [] fitness2=new double[Parameters.popSize];
    
    for(i=0;i<Parameters.popSize;i++){
      fitness1[i]=100-100*_population[i].getAccuracy();
      fitness2[i]=_population[i].getNumAliveRules();
      if(fitness2[i]<=1){
        fitness2[i]=10000;
      }
//      fitness2[i]=_population[i].getNumRules();
//
//      else if(fitness2[i]<Parameters.sizePenaltyMinRules){
//        penalty = (1 - 0.25 * (Parameters.sizePenaltyMinRules- fitness2[i]));
//        if (penalty <= 0) {
//          penalty = 0.1;
//        }
//        penalty *= penalty;
//      }
////    fitness1[i]=fitness1[i]/penalty;
//    fitness2[i]=fitness2[i]/penalty;
    }
    
    NSGA2 nt=new NSGA2(Parameters.popSize);
    nt.rank(fitness1, fitness2);
    
    double max_rank=0;
    for(i=0;i<Parameters.popSize;i++){
      if(nt.population_rank[i]>max_rank){
        max_rank=nt.population_rank[i];
      }
    }
    
    double rk=0;
    
    while(rk<=max_rank){
      for(i=0;i<Parameters.popSize;i++){
        if(nt.population_rank[i]==rk){
          LogManager.println(nt.population_rank[i]+" "+nt.crowding_dist[i]+" "+fitness1[i]+" "+fitness2[i]);
        }
      }
      rk++;
    }
    
    for (i = 0; i < Parameters.popSize; i++) {
      // There can be only one
      int niche = selectNicheWOR(nicheCounters);
      winner = selectCandidateWOR(pools[niche], niche
                                  , _population);
      for (j = 1; j < Parameters.tournamentSize; j++) {
        candidate = selectCandidateWOR(pools[niche]
                                       , niche, _population);
//        if (_population[candidate].compareToIndividual(_population[winner])) {
        //----------------------------------NSGA2-------------------------------------------
//        if(fitness1[candidate]==fitness1[winner]&&fitness2[candidate]==fitness2[winner]){
//          if(_population[candidate].getTheoryLength()<_population[winner].getTheoryLength()){
//            winner=candidate;
//          }
//        }
        if(nt.population_rank[candidate]<nt.population_rank[winner]){
          winner = candidate;
        }
        else if(nt.population_rank[candidate]==nt.population_rank[winner]&&_population[candidate].getFitness()<_population[winner].getFitness()){
          winner=candidate;
        }
        else if(nt.population_rank[candidate]==nt.population_rank[winner]&&nt.crowding_dist[candidate]==nt.crowding_dist[winner]){
          if(Rand.getInteger(0, 9)>4){
          winner=candidate;
          }
        }
//        should set random select here later
        //------------------------------------NSGA2--------------------------------------------
      }
      selectedPopulation[i] = _population[winner].copy();
    }
    Chronometer.stopChronSelection();
    return selectedPopulation;
  }

  public void doMutation(Classifier[] _population) {
    Chronometer.startChronMutation();
    int popSize = Parameters.popSize;
    double probMut = Parameters.probMutationInd;

    for (int i = 0; i < Parameters.popSize; i++) {
      if (Rand.getReal() < probMut) {
        _population[i].doMutation();
      }
    }

    doSpecialStages(_population);

    Chronometer.stopChronMutation();
  }

  void sortedInsert(ArrayList set, Classifier cl) {
    for (int i = 0, max = set.size(); i < max; i++) {
      if (cl.compareToIndividual( (Classifier) set.get(i))) {
        set.add(i, cl);
        return;
      }
    }
    set.add(cl);
  }

  public Classifier[] replacementPolicy(Classifier[] offspring, Classifier[] midsprings, Classifier[] tempsprings
                                        , boolean lastIteration) {
    int i;

    Chronometer.startChronReplacement();
    
    
    if(lastIteration){
      for(i=0;i<Parameters.popSize;i++){
        PopulationWrapper.evaluateClassifier(midsprings[i]);
      }
    }
//    if (lastIteration) {
//      for (i = 0; i < numVersions; i++) {
//        if (bestIndiv[i] != null) {
//          PopulationWrapper.evaluateClassifier(
//              bestIndiv[i]);
//        }
//      }
//      ArrayList set = new ArrayList();
//      for (i = 0; i < Parameters.popSize; i++) {
//        sortedInsert(set, offspring[i]);
//      }
//      for (i = 0; i < numVersions; i++) {
//        if (bestIndiv[i] != null) {
//          sortedInsert(set, bestIndiv[i]);
//        }
//      }
//
//      for (i = 0; i < Parameters.popSize; i++) {
//        offspring[i] = (Classifier) set.get(i);
//      }
//    }
//    else {
//      boolean previousVerUsed = false;
//      int currVer = PopulationWrapper.getCurrentVersion();
//      if (bestIndiv[currVer] == null && currVer > 0) {
//        previousVerUsed = true;
//        currVer--;
//      }

//      if (bestIndiv[currVer] != null) {
//        PopulationWrapper.evaluateClassifier(bestIndiv[currVer]);
//        int worst = PopulationWrapper.getWorst(offspring);
//        offspring[worst] = bestIndiv[currVer].copy();
//      }
//      if (!previousVerUsed) {
//        int prevVer;
//        if (currVer == 0) {
//          prevVer = numVersions - 1;
//        }
//        else {
//          prevVer = currVer - 1;
//        }
//        if (bestIndiv[prevVer] != null) {
//          PopulationWrapper.evaluateClassifier(bestIndiv[prevVer]);
//          int worst = PopulationWrapper.getWorst(offspring);
//          offspring[worst] = bestIndiv[prevVer].copy();
//        }
//      }
//    }
    
    //----------------------------------NSGA2-PART-----------------------------------
    double [] fitness1=new double [Parameters.popSize*2];
    double [] fitness2=new double [Parameters.popSize*2];
    
    for(i=0;i<Parameters.popSize*2;i++){
      if(i<Parameters.popSize){
        fitness1[i]=100-100*midsprings[i].getAccuracy();
        fitness2[i]=midsprings[i].getNumAliveRules();
        if(fitness2[i]<=1){
          fitness2[i]=10000;
        }
      }
      else{
        fitness1[i]=100-100*offspring[i-Parameters.popSize].getAccuracy();
        fitness2[i]=offspring[i-Parameters.popSize].getNumAliveRules();
        if(fitness2[i]<=1){
          fitness2[i]=10000;
        }
      }
    }
    
    NSGA2 nr=new NSGA2(Parameters.popSize*2);
    
    nr.rank(fitness1, fitness2);
    
    
    double max_rank=0;
    for(i=0;i<Parameters.popSize;i++){
      if(nr.population_rank[i]>max_rank){
        max_rank=nr.population_rank[i];
      }
    }
    
    int rank=0;
    i=0;
    while(i<Parameters.popSize){
      int size=nr.population_fronts.get(rank).size();
      if((i+size)>Parameters.popSize){
        if(size==2){
          int a1=nr.population_fronts.get(rank).get(0);
          if(a1>=Parameters.popSize){
            tempsprings[i]=offspring[a1-Parameters.popSize].copy();
            i++;
          }
          else{
            tempsprings[i]=midsprings[a1].copy();
            i++;
          }
        }
        else{
//          LogManager.println(i+"aaaaaaaaaaaaa");
          for(int j=0;j<size-1;j++){
//            LogManager.println(j+"+++++++++");
            for(int k=j+1;k<size;k++){
              int a1=nr.population_fronts.get(rank).get(j);
              int a2=nr.population_fronts.get(rank).get(k);
              if(nr.crowding_dist[a1]<nr.crowding_dist[a2]){
                nr.population_fronts.get(rank).set(k,a1);
                nr.population_fronts.get(rank).set(j,a2);
              }
            }
          }
          for(int l=0;l<size;l++){
//            LogManager.println(i+"-------");
            int a1=nr.population_fronts.get(rank).get(l);
            if(a1>=Parameters.popSize){
              tempsprings[i]=offspring[a1-Parameters.popSize].copy();
              i++;
            }
            else{
              tempsprings[i]=midsprings[a1].copy();
              i++;
            }
            if(i==Parameters.popSize){
              break;
            }
          }
        }
      }
      else{
        for(int j=0;j<Parameters.popSize*2;j++){
           if(nr.population_rank[j]==rank){
             if(j<Parameters.popSize){
             tempsprings[i]=midsprings[j].copy();
             i++;
           }
           else{
             tempsprings[i]=offspring[j-Parameters.popSize].copy();
             i++;
           }
           }
          if(i==Parameters.popSize){
            break;
          }
        }
      }
      rank++;
    }
//    int rank=0;
//    i=0;
//    while(i<Parameters.popSize){
//      for(int j=0;j<Parameters.popSize*2;j++){
//      if(nr.population_rank[j]==rank){
//        if(j<Parameters.popSize){
//          tempsprings[i]=midsprings[j].copy();
//          i++;
//        }
//        else{
//          tempsprings[i]=offspring[j-Parameters.popSize].copy();
//          i++;
//        }
//      }
//      if(i==Parameters.popSize){
//        break;
//      }
//      }
//      rank++;
//    }
    
    for(i=0;i<Parameters.popSize;i++){
      offspring[i]=tempsprings[i].copy();
    }
    
    Chronometer.stopChronReplacement();
    
    return offspring;
//    return offspring;
  }

  public void doSpecialStages(Classifier[] population) {
    int numStages = population[0].numSpecialStages();

    for (int i = 0; i < numStages; i++) {
      for (int j = 0; j < population.length; j++) {
        population[j].doSpecialStage(i);
      }
    }
  }
  
  public class NSGA2{

    public static final double INF=100000000;
    public int popSize=0;
    public int[] population_rank;
    public int[] population_count;
    public double[] crowding_dist;
    public ArrayList <ArrayList<Integer>> dominate_solutions=new ArrayList< ArrayList <Integer> > ();
    public ArrayList <ArrayList<Integer>> population_fronts=new ArrayList< ArrayList <Integer> > ();
    
      
    NSGA2(int pop_num){
      popSize=pop_num;
      population_rank=new int[popSize];
      population_count=new int[popSize];
      crowding_dist=new double[popSize];
      for(int i=0;i<popSize;i++){
        population_count[i]=0;
        population_rank[i]=0;
        crowding_dist[i]=0;
      }
    }
    
    public void rank(double [] f1,double [] f2){
      
      //--------------set dominate cnt and fist level front--------------
      ArrayList <Integer> front_1=new ArrayList <Integer> ();
      for(int i=0;i<popSize;i++){
        ArrayList <Integer> temp=new ArrayList <Integer> ();
        for(int j=0;j<popSize;j++){
          if(i!=j){
            if((f1[j]<f1[i]&&f2[j]<=f2[i])||(f1[j]<=f1[i]&&f2[j]<f2[i])){
              population_count[i]+=1;
            }
            else if((f1[i]<f1[j]&&f2[i]<=f2[j])||(f1[i]<=f1[j]&&f2[i]<f2[j])){
              temp.add(j);//the solutions dominate by "i"
            }
          }
        }
        dominate_solutions.add(temp);
        if(population_count[i]==0){
          front_1.add(i);
        }
      }
      population_fronts.add(front_1);
//      System.out.println(population_fronts.get(0));
      
      //--------------set all population rank------------------------------
      int i=0;
      while (population_fronts.get(i).size()>0){
        ArrayList <Integer> temp=new ArrayList <Integer> ();
        
        //-------make a change to nsga2 to fit the pitts LCS
        
        for(int n=0;n<population_fronts.get(i).size()-1;n++){
          for(int m=n+1;m<population_fronts.get(i).size();m++){
            if(f1[population_fronts.get(i).get(n)]==f1[population_fronts.get(i).get(m)]&&
                f2[population_fronts.get(i).get(n)]==f2[population_fronts.get(i).get(m)]){
              temp.add(population_fronts.get(i).get(m));
              population_fronts.get(i).remove(m);
              m--;
            }
          }
        }
       //-------------------------------------------------- 
          
        for(int j:population_fronts.get(i)){
          for(int k: dominate_solutions.get(j)){
            population_count[k]--;
            if(population_count[k]==0){
              temp.add(k);
            }
          }
        }
        for(int j:temp){
          population_rank[j]=i+1;
        }
        i++;
        population_fronts.add(temp);
      }

      
      //-----------------crowding-distance------------------
      i=0;
      while (population_fronts.get(i).size()>0){
        int front_size=population_fronts.get(i).size();
        if(population_fronts.get(i).size()==2){
          for(int j:population_fronts.get(i)){
            crowding_dist[j]=INF;
          }
        }//front only have two member
        else if(population_fronts.get(i).size()==1){
          crowding_dist[population_fronts.get(i).get(0)]=INF;
        }//front only have one member
        else{
          //----------------OBJECT NO.1----------------------
          for(int j=0;j<front_size-1;j++){
            for(int k=j+1;k<front_size;k++){
              int a1=population_fronts.get(i).get(j);
              int a2=population_fronts.get(i).get(k);
              if(f1[a1]>f1[a2]){
                population_fronts.get(i).set(k,a1);
                population_fronts.get(i).set(j,a2);
              }
            }
          }           
         
//          System.out.println(population_fronts.get(i));
          double max_value,min_value;
          max_value=f1[population_fronts.get(i).get(front_size-1)];
          min_value=f1[population_fronts.get(i).get(0)];
//          System.out.println(max_value+" "+min_value);
          
          crowding_dist[population_fronts.get(i).get(front_size-1)]=INF;
          crowding_dist[population_fronts.get(i).get(0)]=INF;
          
          for(int l=1;l<front_size-1;l++){
            if(max_value==min_value){
              crowding_dist[population_fronts.get(i).get(l)]=0;
            }
            else{
            crowding_dist[population_fronts.get(i).get(l)]=(f1[population_fronts.get(i).get(l+1)]
                -f1[population_fronts.get(i).get(l-1)])/(max_value-min_value);
            }
          }
          //-----------------OBJECT NO.2-----------------
          for(int j=0;j<front_size-1;j++){
            for(int k=j+1;k<front_size;k++){
              int a1=population_fronts.get(i).get(j);
              int a2=population_fronts.get(i).get(k);
              if(f2[a1]>f2[a2]){
                population_fronts.get(i).set(k,a1);
                population_fronts.get(i).set(j,a2);
              }
            }
          }
          max_value=f2[population_fronts.get(i).get(front_size-1)];
          min_value=f2[population_fronts.get(i).get(0)];
          
          crowding_dist[population_fronts.get(i).get(front_size-1)]=INF;
          crowding_dist[population_fronts.get(i).get(0)]=INF;
          
          int crowd_count=0;
          for(int l=1;l<front_size-1;l++){
            if(max_value==min_value){
              crowding_dist[population_fronts.get(i).get(l)]+=0;
            }
            else{
            crowding_dist[population_fronts.get(i).get(l)]+=(f2[population_fronts.get(i).get(l+1)]
                -f2[population_fronts.get(i).get(l-1)])/(max_value-min_value);
            }
//            if(crowding_dist[population_fronts.get(i).get(l)]==0){
//              crowd_count+=1;
//              population_rank[population_fronts.get(i).get(l)]+=crowd_count;
//            }
          }
        }
        i++;
      }
    }   
  }
  
}

