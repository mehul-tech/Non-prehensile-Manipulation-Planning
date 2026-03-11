import numpy as np
import time
import samplers
import utils


class Tree(object):
    """
    The tree class for Kinodynamic RRT.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.dim_state = self.pdef.get_state_dimension()
        self.nodes = []
        self.stateVecs = np.empty(shape=(0, self.dim_state))

    def add(self, node):
        """
        Add a new node into the tree.
        """
        self.nodes.append(node)
        self.stateVecs = np.vstack((self.stateVecs, node.state['stateVec']))
        assert len(self.nodes) == self.stateVecs.shape[0]

    def nearest(self, rstateVec):
        """
        Find the node in the tree whose state vector is nearest to "rstateVec".
        """
        dists = self.pdef.distance_func(rstateVec, self.stateVecs)
        nnode_id = np.argmin(dists)
        return self.nodes[nnode_id]

    def size(self):
        """
        Query the size (number of nodes) of the tree. 
        """
        return len(self.nodes)
    

class Node(object):
    """
    The node class for Tree.
    """

    def __init__(self, state):
        """
        args: state: The state associated with this node.
                     Type: dict, {"stateID": int, 
                                  "stateVec": numpy.ndarray}
        """
        self.state = state
        self.control = None # the control asscoiated with this node
        self.parent = None # the parent node of this node

    def get_control(self):
        return self.control

    def get_parent(self):
        return self.parent

    def set_control(self, control):
        self.control = control

    def set_parent(self, pnode):
        self.parent = pnode
    

class KinodynamicRRT(object):
    """
    The Kinodynamic Rapidly-exploring Random Tree (RRT) motion planner.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance for the problem to be solved.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.tree = Tree(pdef)
        self.state_sampler = samplers.StateSampler(self.pdef) # state sampler
        self.control_sampler = samplers.ControlSampler(self.pdef) # control sampler

    def solve(self, time_budget):
        """
        The main algorithm of Kinodynamic RRT.
        args:  time_budget: The planning time budget (in seconds).
        returns: is_solved: True or False.
                      plan: The motion plan found by the planner,
                            represented by a sequence of tree nodes.
                            Type: a list of rrt.Node
        """
        ########## TODO ##########
        start_time = time.time()

        solved = False
        plan = None        

        # 1. Initialize the tree with the start state [cite: 117]
        start_state = self.pdef.get_start_state()
        start_node = Node(start_state)
        start_node.set_parent(None)
        self.tree.add(start_node) 
        
        # 2. Loop until a solution is found or the time budget is exceeded [cite: 140]
        while time.time() - start_time < time_budget:
            rand_state_vec = self.state_sampler.sample()
            
            # 3. Find the closest node we already have
            nearest_node = self.tree.nearest(rand_state_vec)
            
            # 4. Try k different moves from that node
            # Increasing k to 30 or 50 helps the robot "find" the cube faster
            best_control, outcome_state = self.control_sampler.sample_to(
                nearest_node, rand_state_vec, k=1
            )
            
            # 5. Only add if the move worked and is safe
            if outcome_state is not None:
                if self.pdef.is_state_valid(outcome_state):
                    new_node = Node(outcome_state)
                    new_node.set_parent(nearest_node)
                    new_node.set_control(best_control)
                    self.tree.add(new_node)
                    
                    # 6. Check if we reached the goal
                    if self.pdef.goal.is_satisfied(new_node.state):
                        plan = []
                        curr = new_node
                        while curr is not None:
                            plan.append(curr)
                            curr = curr.get_parent()
                        return True, plan[::-1]
        
        ##########################

        return solved, plan
