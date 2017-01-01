
///////////////////////////////////////////////////////////////////////////////
//  
// Main Class File:  DecisionTreeBuilder.java
// File:             DecisionTree.java
// Semester:         CS540 Artificial Intelligence Summer 2016
// Author:           David Liang dliang23@wisc.edu
//
//////////////////////////////////////////////////////////////////////////////

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Random;
import java.util.Stack;

public class DecisionTree
{
   private DecisionTreeNode root;
   // ordered list of class labels
   private List<String> labels;
   // ordered list of attributes
   private List<String> attributes;
   // map to ordered discrete values taken by attributes
   private Map<String, List<String>> attributeValues;

   /**
    * Answers static questions about decision trees.
    */
   DecisionTree()
   {
      // empty
   }

   /**
    * Build a decision tree given only a training set.
    * 
    * @param trainSet:
    *           the training set
    */
   DecisionTree(DataSet trainSet)
   {
      this.labels = trainSet.labels;
      this.attributes = trainSet.attributes;
      this.attributeValues = trainSet.attributeValues;
      this.root = buildDecisionTree(trainSet, plurality(trainSet), null, null, 0);
   }

   private DecisionTreeNode buildDecisionTree(DataSet trainSet, String defaultClassification, String parentAttribute, String parentAttributeValue, int level)
   {
      if (trainSet.instances.size() == 0) // no more examples
      {
         return new DecisionTreeNode(defaultClassification, null, parentAttributeValue, true);
      }
      else if (sameClassification(trainSet))
      {
         return new DecisionTreeNode(trainSet.instances.get(0).label, null, parentAttributeValue, true);
      }
      else if (emptyAttributes(trainSet.attributes)) // no more attributes
      {
         return new DecisionTreeNode(plurality(trainSet), null, parentAttributeValue, true);
      }

      String bestAttr = bestAttribute(trainSet, false);
      int bestAttrInd = getAttributeIndex(bestAttr);
      defaultClassification = plurality(trainSet);

      // new attributes list with best attribute removed
      ArrayList<String> newAttributes = new ArrayList<String>(trainSet.attributes);

      for (int i = 0; i < newAttributes.size(); i++)
      {
         if (newAttributes.get(i) != null && newAttributes.get(i).equals(bestAttr))
         {
            newAttributes.remove(i);
            newAttributes.add(i, null);
            i += newAttributes.size();
         }
      }

      DecisionTreeNode newNode = new DecisionTreeNode(defaultClassification, bestAttr, null, false);

      if (parentAttribute != null && parentAttributeValue != null)
      {
         newNode.parentAttributeValue = parentAttributeValue;
      }

      // build subtrees
      for (int i = 0; i < trainSet.attributeValues.get(bestAttr).size(); i++)
      {
         DataSet newTrainSet = new DataSet(); // prepare new DataSet

         newTrainSet.instances = new ArrayList<Instance>();
         newTrainSet.labels = trainSet.labels;
         newTrainSet.attributeValues = trainSet.attributeValues;
         newTrainSet.attributes = newAttributes;

         String currVal = trainSet.attributeValues.get(bestAttr).get(i);

         for (int j = 0; j < trainSet.instances.size(); j++)
         { 
            Instance currInstance = trainSet.instances.get(j);
            if (currInstance.attributes.get(bestAttrInd).equals(currVal))
            {
               newTrainSet.instances.add(currInstance); // add into new trainSet
            }
         }

         newNode.addChild(buildDecisionTree(newTrainSet, defaultClassification, bestAttr, currVal, level + 1));
      }

      return newNode;
   }

   /**
    * Calculate the entropy of the class label.
    * 
    * @param trainSet
    * @return
    */
   private double classEntropy(DataSet trainSet)
   {
      double p0, p1;
      double[] classCounts = new double[] { 1.0, 0.0 };
      String firstClassEncountered = trainSet.instances.get(0).label;

      for (int i = 1; i < trainSet.instances.size(); i++)
      {
         if (trainSet.instances.get(i).label.equals(firstClassEncountered))
         {
            classCounts[0]++;
         }
         else
         {
            classCounts[1]++;
         }
      }

      p0 = classCounts[0] / trainSet.instances.size();
      p1 = classCounts[1] / trainSet.instances.size();

      return (-p0 * Math.log(p0) - p1 * Math.log(p1)) / Math.log(2);
   }

   /**
    * Determine plurality of a given dataSet.
    * 
    * @param parent
    * @return
    */
   private String plurality(DataSet parent)
   {
      int[] counts = new int[parent.labels.size()];

      for (int i = 0; i < parent.instances.size(); i++)
      {
         String curr = parent.instances.get(i).label;
         for (int j = 0; j < parent.labels.size(); j++)
         {
            if (curr.equals(parent.labels.get(j)))
            {
               counts[j]++;
            }
         }
      }

      int maxCount = counts[0]; // determine the plurality
      String maxCountLabel = parent.labels.get(0);

      for (int i = 1; i < counts.length; i++)
      {
         if (counts[i] > maxCount)
         {
            maxCount = counts[i];
            maxCountLabel = parent.labels.get(i);
         }
      }

      return maxCountLabel;
   }

   /**
    * Determine if all examples in the training set have the same classification (label).
    * 
    * @param trainSet
    * @return
    */
   private boolean sameClassification(DataSet trainSet)
   {
      String label = trainSet.instances.get(0).label;
      for (int i = 1; i < trainSet.instances.size(); i++)
      {
         if (!trainSet.instances.get(i).label.equals(label))
         {
            return false;
         }
      }

      return true;
   }

   /**
    * Determine if there are no remaining attributes to consider.
    * 
    * @param attributez
    * @return
    */
   private boolean emptyAttributes(List<String> attributez)
   {
      for (int i = 0; i < attributez.size(); i++)
      {
         if (attributez.get(i) != null)
         {
            return false;
         }
      }

      return true;
   }

   public String classify(Instance instance)
   {
      return classify(instance, root);
   }

   /**
    * Classify given an instance and the root to a decision tree.
    * 
    * @param instance
    * @param newRoot
    * @return
    */
   private String classify(Instance instance, DecisionTreeNode newRoot)
   {
      DecisionTreeNode curr = newRoot;
      while (!curr.terminal)
      {
         String currAttrVal = instance.attributes.get(getAttributeIndex(curr.attribute));
         boolean found = false;
         // for each attribute (equiv. to number of levels of tree)
         int size = curr.children.size();
         for (int i = 0; i < size; i++)
         {
            if (currAttrVal.equals(curr.children.get(i).parentAttributeValue))
            {
               curr = curr.children.get(i);
               if (curr.attribute == null)
               {
                  break;
               }

               currAttrVal = instance.attributes.get(getAttributeIndex(curr.attribute));
               i += 10000;
               found = true;
            }
         }
         if (!found)
         {
            return curr.label;
         }
      }

      return curr.label;
   }

   public void rootInfoGain(DataSet trainSet)
   {
      bestAttribute(trainSet, true);
   }

   /**
    * Select the best attribute for a training set and print if needed.
    * 
    * @param trainSet
    * @param print
    * @return
    */
   private String bestAttribute(DataSet trainSet, boolean print)
   {
      double classEntropy = classEntropy(trainSet), maxTotalInfoGain = -1000000000;
      String bestAttr = null;

      for (int i = 0; i < trainSet.attributes.size(); i++) // for each attribute
      {
         if (trainSet.attributes.get(i) == null)
         {
            continue;
         }

         double totalInfoGain = classEntropy; // reset totalInfoGain
         String attribute = trainSet.attributes.get(i); // get the attribute
         double[][] attributeCounts = new double[trainSet.attributeValues.get(attribute).size()][2]; // [attributeValue][label]

         for (int j = 0; j < trainSet.instances.size(); j++)
         {
            String currValue = trainSet.instances.get(j).attributes.get(i);
            for (int k = 0; k < trainSet.attributeValues.get(attribute).size(); k++) // for
                                                                                     // each
                                                                                     // attribute
                                                                                     // value
            {
               if (currValue.equals(trainSet.attributeValues.get(attribute).get(k))) // if
               {
                  if (trainSet.instances.get(j).label.equals(trainSet.labels.get(0)))
                  {
                     attributeCounts[k][0]++;
                  }
                  else if (trainSet.instances.get(j).label.equals(trainSet.labels.get(1)))
                  {
                     attributeCounts[k][1]++;
                  }

                  k += trainSet.attributeValues.get(attribute).size();
               }
            }
         }

         for (int j = 0; j < attributeCounts.length; j++)
         { // for each
           // attribute value
            double sum = attributeCounts[j][0] + attributeCounts[j][1];
            double p0 = attributeCounts[j][0] / sum;
            double p1 = attributeCounts[j][1] / sum;

            if (p0 > 0)
            {
               totalInfoGain += sum / trainSet.instances.size() * (p0 * Math.log(p0)) / Math.log(2);
            }
            if (p1 > 0)
            {
               totalInfoGain += sum / trainSet.instances.size() * (p1 * Math.log(p1)) / Math.log(2);
            }
         }

         if (print)
         {
            System.out.print(trainSet.attributes.get(i) + " ");
            System.out.format("%.5f\n", totalInfoGain);
         }

         if (totalInfoGain > maxTotalInfoGain) 
         {
            maxTotalInfoGain = totalInfoGain;
            bestAttr = trainSet.attributes.get(i);
         }
      }
      return bestAttr;
   }

   public void printAccuracy(DataSet test)
   {
      System.out.format("%.5f", accuracy(test, root));
   }

   /**
    * Calculate accuracy of all instances given in a test set based on a given tree.
    * 
    * @param test
    * @param newRoot
    * @return
    */
   private double accuracy(DataSet test, DecisionTreeNode newRoot)
   {
      if (newRoot == null)
      {
         return 0;
      }

      double count = 0.0;
      for (Instance instance : test.instances)
      {
         if (instance.label.equals(classify(instance, newRoot)))
         {
            count++;
         }
      }

      return count / test.instances.size();
   }

   /**
    * Build a decision tree given a training set then prune it using a tuning set. ONLY for extra credits
    * 
    * @param trainSet:
    *           the training set
    * @param tune:
    *           the tuning set
    */
   DecisionTree(DataSet trainSet, DataSet tuneSet)
   {
      this.labels = trainSet.labels;
      this.attributes = trainSet.attributes;
      this.attributeValues = trainSet.attributeValues;
      root = buildDecisionTree(trainSet, plurality(trainSet), null, null, 0);

      // use pruning algorithms; this sequence produced good results
      for (int i = 0; i < 3; i++)
      {
         randomPrune(trainSet, tuneSet, buildDecisionTree(trainSet, plurality(trainSet), null, null, 0));
         randomPrune2(trainSet, tuneSet, buildDecisionTree(trainSet, plurality(trainSet), null, null, 0));
      }

      splitPrune(trainSet, tuneSet, buildDecisionTree(trainSet, plurality(trainSet), null, null, 0));
      splitPrune2(trainSet, tuneSet, buildDecisionTree(trainSet, plurality(trainSet), null, null, 0));
      splitPrune3(trainSet, tuneSet, buildDecisionTree(trainSet, plurality(trainSet), null, null, 0));
      splitPrune4(trainSet, tuneSet, buildDecisionTree(trainSet, plurality(trainSet), null, null, 0));
      DFSprune(trainSet, tuneSet, buildDecisionTree(trainSet, plurality(trainSet), null, null, 0));
      DFSprune2(trainSet, tuneSet, buildDecisionTree(trainSet, plurality(trainSet), null, null, 0));
      BFSprune(trainSet, tuneSet, buildDecisionTree(trainSet, plurality(trainSet), null, null, 0));
      BFSprune2(trainSet, tuneSet, buildDecisionTree(trainSet, plurality(trainSet), null, null, 0));
      twinPrune(trainSet, tuneSet, buildDecisionTree(trainSet, plurality(trainSet), null, null, 0));
      twinPrune2(trainSet, tuneSet, buildDecisionTree(trainSet, plurality(trainSet), null, null, 0));

      for (int j = 0; j < 2; j++)
      {
         splitPrune(trainSet, tuneSet, root);
         splitPrune2(trainSet, tuneSet, root);
         splitPrune3(trainSet, tuneSet, root);
         splitPrune4(trainSet, tuneSet, root);

         BFSprune(trainSet, tuneSet, root);
         DFSprune(trainSet, tuneSet, root);
         DFSprune2(trainSet, tuneSet, root);
         BFSprune2(trainSet, tuneSet, root);
         twinPrune(trainSet, tuneSet, root);
         twinPrune2(trainSet, tuneSet, root);
      }
   }

   /**
    * Prune the tree based on a DFS consideration of the nodes.
    * 
    * @param trainSet
    * @param tuneSet
    * @param pruneRoot
    */
   private void DFSprune(DataSet trainSet, DataSet tuneSet, DecisionTreeNode pruneRoot)
   {
      Stack<DecisionTreeNode> stack = new Stack<DecisionTreeNode>();

      for (int i = 0; i < pruneRoot.children.size(); i++)
      {
         stack.push(pruneRoot.children.get(i));
      }

      double preAccuracy = accuracy(tuneSet, pruneRoot);

      while (!stack.isEmpty())
      {
         DecisionTreeNode curr = stack.pop();

         if (curr.children == null || curr.terminal)
         {
            continue;
         }

         for (int i = 0; i < curr.children.size(); i++)
         {
            if (!curr.children.get(i).terminal)
            {
               stack.push(curr.children.get(i));
            }
         }

         curr.terminal = true;

         double postAccuracy = accuracy(tuneSet, pruneRoot);

         if (preAccuracy >= postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            postAccuracy = preAccuracy;
         }
      }

      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }
   }

   private void DFSprune2(DataSet trainSet, DataSet tuneSet, DecisionTreeNode pruneRoot)
   {
      Stack<DecisionTreeNode> stack = new Stack<DecisionTreeNode>();

      for (int i = 0; i < pruneRoot.children.size(); i++)
      {
         stack.push(pruneRoot.children.get(i));
      }

      double preAccuracy = accuracy(tuneSet, pruneRoot);

      while (!stack.isEmpty())
      {
         DecisionTreeNode curr = stack.pop();

         if (curr.children == null || curr.terminal)
         {
            continue;
         }

         for (int i = 0; i < curr.children.size(); i++)
         {
            if (!curr.children.get(i).terminal)
            {
               stack.push(curr.children.get(i));
            }
         }

         curr.terminal = true;

         double postAccuracy = accuracy(tuneSet, pruneRoot);

         if (preAccuracy > postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            postAccuracy = preAccuracy;
         }

      }

      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }
   }

   /**
    * Prune the tree in consideration of a BFS order.
    * 
    * @param trainSet
    * @param tuneSet
    * @param pruneRoot
    */
   private void BFSprune(DataSet trainSet, DataSet tuneSet, DecisionTreeNode pruneRoot)
   {
      Queue<DecisionTreeNode> queue = new LinkedList<DecisionTreeNode>();

      for (int i = 0; i < pruneRoot.children.size(); i++)
      {
         queue.add(pruneRoot.children.get(i));
      }

      double preAccuracy = accuracy(tuneSet, pruneRoot);

      while (!queue.isEmpty())
      {
         DecisionTreeNode curr = queue.poll();

         if (curr.children == null || curr.terminal)
         {
            continue;
         }

         for (int i = 0; i < curr.children.size(); i++)
         {
            if (!curr.children.get(i).terminal)
            {
               queue.add(curr.children.get(i));
            }
         }

         curr.terminal = true;

         double postAccuracy = accuracy(tuneSet, pruneRoot);

         if (preAccuracy >= postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            postAccuracy = preAccuracy;
         }
      }

      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }
   }

   private void BFSprune2(DataSet trainSet, DataSet tuneSet, DecisionTreeNode pruneRoot)
   {
      Queue<DecisionTreeNode> queue = new LinkedList<DecisionTreeNode>();

      for (int i = 0; i < pruneRoot.children.size(); i++)
      {
         queue.add(pruneRoot.children.get(i));
      }

      double preAccuracy = accuracy(tuneSet, pruneRoot);

      while (!queue.isEmpty())
      {
         DecisionTreeNode curr = queue.poll();

         if (curr.children == null || curr.terminal)
         {
            continue;
         }

         for (int i = 0; i < curr.children.size(); i++)
         {
            if (!curr.children.get(i).terminal)
            {
               queue.add(curr.children.get(i));
            }
         }

         curr.terminal = true;

         double postAccuracy = accuracy(tuneSet, pruneRoot);

         if (preAccuracy > postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            postAccuracy = preAccuracy;
         }
      }

      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }
   }

   /**
    * Generate a list of nodes in a BFS order and then try pruning the middle node.
    * 
    * @param trainSet
    * @param tuneSet
    * @param pruneRoot
    */
   private void splitPrune(DataSet trainSet, DataSet tuneSet, DecisionTreeNode pruneRoot)
   {
      ArrayList<DecisionTreeNode> nodes = new ArrayList<DecisionTreeNode>();
      nodes.add(pruneRoot);
      for (int j = 0; j < nodes.size(); j++) // add nodes of decision tree into
                                             // nodes
      {
         if (nodes.get(j).terminal)
         {
            continue;
         }
         for (int k = 0; k < nodes.get(j).children.size(); k++)
         {
            if (!nodes.get(j).children.get(k).terminal)
            {
               nodes.add(nodes.get(j).children.get(k));
            }
         }
      }

      double preAccuracy = accuracy(tuneSet, pruneRoot);

      while (!nodes.isEmpty())
      {
         DecisionTreeNode curr = nodes.remove((int) (nodes.size() / 2.0));
         curr.terminal = true;
         double postAccuracy = accuracy(tuneSet, pruneRoot);

         if (preAccuracy >= postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            preAccuracy = postAccuracy;
         }
      }

      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }

      nodes = new ArrayList<DecisionTreeNode>();
      nodes.add(pruneRoot);

      for (int j = 0; j < nodes.size(); j++) // add nodes of decision tree into
                                             // nodes
      {
         if (nodes.get(j).terminal)
         {
            continue;
         }
         for (int k = 0; k < nodes.get(j).children.size(); k++)
         {
            if (!nodes.get(j).children.get(k).terminal)
            {
               nodes.add(nodes.get(j).children.get(k));
            }
         }
      }

      preAccuracy = accuracy(tuneSet, pruneRoot);

      while (!nodes.isEmpty())
      {
         DecisionTreeNode curr = nodes.remove(nodes.size() - 1);
         curr.terminal = true;
         double postAccuracy = accuracy(tuneSet, pruneRoot);

         if (preAccuracy >= postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            preAccuracy = postAccuracy;
         }
      }
      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }
   }

   private void splitPrune2(DataSet trainSet, DataSet tuneSet, DecisionTreeNode pruneRoot)
   {
      ArrayList<DecisionTreeNode> nodes = new ArrayList<DecisionTreeNode>();
      nodes.add(pruneRoot);

      for (int j = 0; j < nodes.size(); j++) 
      {
         if (nodes.get(j).terminal)
         {
            continue;
         }
         for (int k = 0; k < nodes.get(j).children.size(); k++)
         {
            if (!nodes.get(j).children.get(k).terminal)
            {
               nodes.add(nodes.get(j).children.get(k));
            }
         }
      }

      double preAccuracy = accuracy(tuneSet, pruneRoot);

      while (!nodes.isEmpty())
      {
         DecisionTreeNode curr = nodes.remove((int) (nodes.size() / 2.0));
         curr.terminal = true;
         double postAccuracy = accuracy(tuneSet, pruneRoot);

         if (preAccuracy > postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            preAccuracy = postAccuracy;
         }
      }

      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }

      nodes = new ArrayList<DecisionTreeNode>();
      nodes.add(pruneRoot);

      for (int j = 0; j < nodes.size(); j++) 
      {
         if (nodes.get(j).terminal)
         {
            continue;
         }
         for (int k = 0; k < nodes.get(j).children.size(); k++)
         {
            if (!nodes.get(j).children.get(k).terminal)
            {
               nodes.add(nodes.get(j).children.get(k));
            }
         }
      }

      preAccuracy = accuracy(tuneSet, pruneRoot);

      while (!nodes.isEmpty())
      {
         DecisionTreeNode curr = nodes.remove(nodes.size() - 1);
         curr.terminal = true;
         double postAccuracy = accuracy(tuneSet, pruneRoot);

         if (preAccuracy > postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            preAccuracy = postAccuracy;
         }
      }

      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }
   }

   /**
    * Generate a list of nodes by DFS and select the middle node to prune.
    * 
    * @param trainSet
    * @param tuneSet
    * @param pruneRoot
    */
   private void splitPrune3(DataSet trainSet, DataSet tuneSet, DecisionTreeNode pruneRoot)
   {
      ArrayList<DecisionTreeNode> nodes = new ArrayList<DecisionTreeNode>();
      ArrayList<DecisionTreeNode> visited = new ArrayList<DecisionTreeNode>();
      nodes.add(pruneRoot);

      for (int j = 0; j < nodes.size(); j++) 
      {
         visited.add(nodes.get(j));

         if (nodes.get(j).terminal)
         {
            continue;
         }

         for (int k = 0; k < nodes.get(j).children.size(); k++)
         {
            if (!nodes.get(j).children.get(k).terminal && !visited.contains(nodes.get(j).children.get(k)))
            {
               nodes.add(0, nodes.get(j).children.get(k));
               j = 0;
            }
         }
      }

      double preAccuracy = accuracy(tuneSet, pruneRoot);

      while (!nodes.isEmpty())
      {
         DecisionTreeNode curr = nodes.remove((int) (nodes.size() / 2.0));
         curr.terminal = true;
         double postAccuracy = accuracy(tuneSet, pruneRoot);

         if (preAccuracy >= postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            preAccuracy = postAccuracy;
         }
      }

      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }

      nodes = new ArrayList<DecisionTreeNode>();
      nodes.add(pruneRoot);

      for (int j = 0; j < nodes.size(); j++) 
      {
         if (nodes.get(j).terminal)
         {
            continue;
         }
         for (int k = 0; k < nodes.get(j).children.size(); k++)
         {
            if (!nodes.get(j).children.get(k).terminal)
            {
               nodes.add(nodes.get(j).children.get(k));
            }
         }
      }

      preAccuracy = accuracy(tuneSet, pruneRoot);

      while (!nodes.isEmpty())
      {
         DecisionTreeNode curr = nodes.remove(nodes.size() - 1);
         curr.terminal = true;
         double postAccuracy = accuracy(tuneSet, pruneRoot);
         if (preAccuracy >= postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            preAccuracy = postAccuracy;
         }
      }

      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }
   }

   private void splitPrune4(DataSet trainSet, DataSet tuneSet, DecisionTreeNode pruneRoot)
   {
      ArrayList<DecisionTreeNode> nodes = new ArrayList<DecisionTreeNode>();
      ArrayList<DecisionTreeNode> visited = new ArrayList<DecisionTreeNode>();
      nodes.add(pruneRoot);

      for (int j = 0; j < nodes.size(); j++) 
      {
         visited.add(nodes.get(j));

         if (nodes.get(j).terminal)
         {
            continue;
         }
         for (int k = 0; k < nodes.get(j).children.size(); k++)
         {
            if (!nodes.get(j).children.get(k).terminal && !visited.contains(nodes.get(j).children.get(k)))
            {
               nodes.add(0, nodes.get(j).children.get(k));
               j = 0;
            }
         }
      }

      double preAccuracy = accuracy(tuneSet, pruneRoot);

      while (!nodes.isEmpty())
      {
         DecisionTreeNode curr = nodes.remove((int) (nodes.size() / 2.0));
         curr.terminal = true;
         double postAccuracy = accuracy(tuneSet, pruneRoot);

         if (preAccuracy > postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            preAccuracy = postAccuracy;
         }
      }

      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }

      nodes = new ArrayList<DecisionTreeNode>();
      visited = new ArrayList<DecisionTreeNode>();
      nodes.add(pruneRoot);

      for (int j = 0; j < nodes.size(); j++) 
      {
         visited.add(nodes.get(j));
         if (nodes.get(j).terminal)
         {
            continue;
         }

         for (int k = 0; k < nodes.get(j).children.size(); k++)
         {
            if (!nodes.get(j).children.get(k).terminal && !visited.contains(nodes.get(j).children.get(k)))
            {
               nodes.add(0, nodes.get(j).children.get(k));
               j = 0;
            }
         }
      }

      nodes.add(pruneRoot);

      for (int j = 0; j < nodes.size(); j++) 
      {
         if (nodes.get(j).terminal)
         {
            continue;
         }
         for (int k = 0; k < nodes.get(j).children.size(); k++)
         {
            if (!nodes.get(j).children.get(k).terminal)
            {
               nodes.add(nodes.get(j).children.get(k));
            }
         }
      }

      preAccuracy = accuracy(tuneSet, pruneRoot);

      while (!nodes.isEmpty())
      {
         DecisionTreeNode curr = nodes.remove(nodes.size() - 1);
         curr.terminal = true;
         double postAccuracy = accuracy(tuneSet, pruneRoot);

         if (preAccuracy > postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            preAccuracy = postAccuracy;
         }
      }

      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }
   }

   /**
    * For all pairs of nodes, try pruning those. Only pair instead of triple or quadrule in consideration of time.
    * 
    * @param trainSet
    * @param tuneSet
    * @param pruneRoot
    */
   private void twinPrune(DataSet trainSet, DataSet tuneSet, DecisionTreeNode pruneRoot)
   {
      ArrayList<DecisionTreeNode> nodes = new ArrayList<DecisionTreeNode>();
      nodes.add(pruneRoot);

      for (int j = 0; j < nodes.size(); j++)
      {
         if (nodes.get(j).terminal || nodes.get(j).children.isEmpty())
         {
            continue;
         }

         for (int i = 0; i < nodes.get(j).children.size(); i++)
         {
            if (!nodes.get(j).children.get(i).terminal && !nodes.get(j).children.get(i).children.isEmpty())
            {
               nodes.add(nodes.get(j).children.get(i));
            }
         }
      }

      for (int y = 0; y < 3; y++)
      {
         double preAccuracy = accuracy(tuneSet, pruneRoot);

         ArrayList<Integer> nodeSelection = new ArrayList<Integer>();

         for (int i = 0; i < 3; i++)
         {
            nodeSelection.add(0);
         }

         for (int i = -1; i < nodes.size(); i++)
         {
            nodeSelection.remove(0);
            nodeSelection.add(0, i);

            for (int j = -1; j < nodes.size(); j++)
            {
               nodeSelection.remove(1);
               nodeSelection.add(1, j);

               for (int z = 0; z < nodeSelection.size(); z++)
               {
                  if (nodeSelection.get(z) == -1)
                  {
                     continue;
                  }

                  nodes.get(nodeSelection.get(z)).terminal = true;
               }

               double postAccuracy = accuracy(tuneSet, pruneRoot);

               if (preAccuracy >= postAccuracy)
               {
                  for (int z = 0; z < nodeSelection.size(); z++)
                  {
                     if (nodeSelection.get(z) == -1)
                     {
                        continue;
                     }

                     nodes.get(nodeSelection.get(z)).terminal = false;
                  }
               }
               else
               {
                  preAccuracy = postAccuracy;
               }
            }
         }

         if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
         {
            this.root = pruneRoot;
         }
      }
   }

   private void twinPrune2(DataSet trainSet, DataSet tuneSet, DecisionTreeNode pruneRoot)
   {
      ArrayList<DecisionTreeNode> nodes = new ArrayList<DecisionTreeNode>();
      nodes.add(pruneRoot);

      for (int j = 0; j < nodes.size(); j++)
      {
         if (nodes.get(j).terminal || nodes.get(j).children.isEmpty())
         {
            continue;
         }
         for (int i = 0; i < nodes.get(j).children.size(); i++)
         {
            if (!nodes.get(j).children.get(i).terminal && !nodes.get(j).children.get(i).children.isEmpty())
            {
               nodes.add(nodes.get(j).children.get(i));
            }
         }
      }

      for (int y = 0; y < 3; y++)
      {
         double preAccuracy = accuracy(tuneSet, pruneRoot);

         ArrayList<Integer> nodeSelection = new ArrayList<Integer>();
         for (int i = 0; i < 3; i++)
         {
            nodeSelection.add(0);
         }

         for (int i = -1; i < nodes.size(); i++)
         {
            nodeSelection.remove(0);
            nodeSelection.add(0, i);

            for (int j = -1; j < nodes.size(); j++)
            {
               nodeSelection.remove(1);
               nodeSelection.add(1, j);

               for (int z = 0; z < nodeSelection.size(); z++)
               {
                  if (nodeSelection.get(z) == -1)
                  {
                     continue;
                  }

                  nodes.get(nodeSelection.get(z)).terminal = true;
               }

               double postAccuracy = accuracy(tuneSet, pruneRoot);

               if (preAccuracy > postAccuracy)
               {
                  for (int z = 0; z < nodeSelection.size(); z++)
                  {
                     if (nodeSelection.get(z) == -1)
                     {
                        continue;
                     }
                     nodes.get(nodeSelection.get(z)).terminal = false;
                  }
               }
               else
               {
                  preAccuracy = postAccuracy;
               }
            }
         }

         if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
         {
            this.root = pruneRoot;
         }
      }
   }

   /**
    * Randomly select a node from the BFS traversal to tune.
    * 
    * @param trainSet
    * @param tuneSet
    * @param pruneRoot
    */
   private void randomPrune(DataSet trainSet, DataSet tuneSet, DecisionTreeNode pruneRoot)
   {
      ArrayList<DecisionTreeNode> nodes = new ArrayList<DecisionTreeNode>();
      nodes.add(pruneRoot);

      for (int j = 0; j < nodes.size(); j++)
      {
         if (nodes.get(j).terminal || nodes.get(j).children.isEmpty())
         {
            continue;
         }
         for (int i = 0; i < nodes.get(j).children.size(); i++)
         {
            if (!nodes.get(j).children.get(i).terminal && !nodes.get(j).children.get(i).children.isEmpty())
            {
               nodes.add(nodes.get(j).children.get(i));
            }
         }
      }

      Random RNG = new Random();

      double preAccuracy = accuracy(trainSet, pruneRoot);

      while (!nodes.isEmpty())
      {
         DecisionTreeNode curr = nodes.remove(RNG.nextInt(nodes.size()));
         curr.terminal = true;
         double postAccuracy = accuracy(trainSet, pruneRoot);

         if (preAccuracy >= postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            preAccuracy = postAccuracy;
         }
      }

      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }
   }

   private void randomPrune2(DataSet trainSet, DataSet tuneSet, DecisionTreeNode pruneRoot)
   {
      ArrayList<DecisionTreeNode> nodes = new ArrayList<DecisionTreeNode>();
      nodes.add(pruneRoot);

      for (int j = 0; j < nodes.size(); j++)
      {
         if (nodes.get(j).terminal || nodes.get(j).children.isEmpty())
         {
            continue;
         }

         for (int i = 0; i < nodes.get(j).children.size(); i++)
         {
            if (!nodes.get(j).children.get(i).terminal && !nodes.get(j).children.get(i).children.isEmpty())
            {
               nodes.add(nodes.get(j).children.get(i));
            }
         }
      }

      Random RNG = new Random();

      double preAccuracy = accuracy(trainSet, pruneRoot);

      while (!nodes.isEmpty())
      {
         DecisionTreeNode curr = nodes.remove(RNG.nextInt(nodes.size()));
         curr.terminal = true;
         double postAccuracy = accuracy(trainSet, pruneRoot);

         if (preAccuracy > postAccuracy)
         {
            curr.terminal = false;
         }
         else
         {
            preAccuracy = postAccuracy;
         }

      }

      if (accuracy(tuneSet, this.root) <= accuracy(tuneSet, pruneRoot))
      {
         this.root = pruneRoot;
      }
   }

   /**
    * Print the decision tree in the specified format
    */
   public void print()
   {
      printTreeNode(root, null, 0);
   }

   /**
    * Prints the subtree of the node with each line prefixed by 4 * k spaces.
    */
   public void printTreeNode(DecisionTreeNode p, DecisionTreeNode parent, int k)
   {
      StringBuilder sb = new StringBuilder();

      for (int i = 0; i < k; i++)
      {
         sb.append("    ");
      }

      String value;

      if (parent == null)
      {
         value = "ROOT";
      }
      else
      {
         int attributeValueIndex = this.getAttributeValueIndex(parent.attribute, p.parentAttributeValue);
         value = attributeValues.get(parent.attribute).get(attributeValueIndex);
      }

      sb.append(value);

      if (p.terminal)
      {
         sb.append(" (" + p.label + ")");
         System.out.println(sb.toString());
      }
      else
      {
         sb.append(" {" + p.attribute + "?}");
         System.out.println(sb.toString());
         for (DecisionTreeNode child : p.children)
         {
            printTreeNode(child, p, k + 1);
         }
      }
   }

   /**
    * Helper function to get the index of the label in labels list
    */
   private int getLabelIndex(String label)
   {
      for (int i = 0; i < this.labels.size(); i++)
      {
         if (label.equals(this.labels.get(i)))
         {
            return i;
         }
      }

      return -1;
   }

   /**
    * Helper function to get the index of the attribute in attributes list
    */
   private int getAttributeIndex(String attr)
   {
      for (int i = 0; i < this.attributes.size(); i++)
      {
         if (attr.equals(this.attributes.get(i)))
         {
            return i;
         }
      }

      return -1;
   }

   /**
    * Helper function to get the index of the attributeValue in the list for the attribute key in the attributeValues map
    */
   private int getAttributeValueIndex(String attr, String value)
   {
      for (int i = 0; i < attributeValues.get(attr).size(); i++)
      {
         if (value.equals(attributeValues.get(attr).get(i)))
         {
            return i;
         }
      }

      return -1;
   }
}
