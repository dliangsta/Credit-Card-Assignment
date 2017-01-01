
///////////////////////////////////////////////////////////////////////////////
//  
// Main Class File:  DecisionTreeBuilder.java
// File:             DecisionTreeNode.java
// Semester:         CS540 Artificial Intelligence Summer 2016
// Author:           David Liang dliang23@wisc.edu
//
//////////////////////////////////////////////////////////////////////////////

import java.util.ArrayList;
import java.util.List;

/**
 * Possible class for internal organization of a decision tree. Included to show
 * standardized output method, print().
 * 
 */
public class DecisionTreeNode {
   String label; // for
   String attribute;
   String parentAttributeValue; // if is the root, set to null
   boolean terminal;
   List<DecisionTreeNode> children;

   DecisionTreeNode(String _label, String _attribute, String _parentAttributeValue, boolean _terminal) {
      label = _label;
      attribute = _attribute;
      parentAttributeValue = _parentAttributeValue;
      terminal = _terminal;

      if (_terminal) {
         children = null;
      } else {
         children = new ArrayList<DecisionTreeNode>();
      }
   }

   /**
    * Add child to the node.
    * 
    * For printing to be consistent, children should be added in order of the
    * attribute values as specified in the dataset.
    */
   public void addChild(DecisionTreeNode child) {
      if (children != null) {
         children.add(child);
      }
   }
}
