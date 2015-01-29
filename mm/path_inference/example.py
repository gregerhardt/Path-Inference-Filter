#Copyright 2011, 2012 Timothy Hunter <tjhunter@eecs.berkeley.edu>
#
#This library is free software; you can redistribute it and/or
#modify it under the terms of the GNU Lesser General Public
#License as published by the Free Software Foundation; either
#version 2.1 of the License, or (at your option) version 3.
#
#This library is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#Lesser General Public License for more details.
#
#You should have received a copy of the GNU Lesser General Public 
#License along with this library.  If not, see <http://www.gnu.org/licenses/>.

# pylint: disable=W0105
'''
Tutorial to the Path inference filter code.

Created on Jan 26, 2012

@author: tjhunter

This code defines a simple maze as a road network, simulates the crossing of
 this maze by a vehicle and runs the filter to reconstruct the trajectory.
'''
import random
import math
from mm.path_inference.example_utils import \
  create_trajectory_for_lattice_network, LatticePathBuilder

""" The road network is a an NxN lattice. Each of the nodes of the lattice is
defined by its (x,y) coordinate ((0,0) at the bottom left). Each link of the 
lattice  is directed. There only a single link between each node of the 
lattice: the lattice only contains links that go up or right.
 
A link is defined as a pair of nodes (from, to).
"""
lattice_size = 5
lattice_link_length = 100 # meters

""" Since all the links have the same length, we use the travel time on each
link as a feature. Each link has a random travel time that follows a Gaussian 
distribution. The standard deviation of all the distributions is the same.
the mean is randomly chosen for each link.
"""
""" Standard deviation.
"""
sigma = 10 # meters

""" The observation noise is simulated using a normal distribution.
"""
obs_sigma = 0 # meters

""" Mean travel times on every link.
It is a random value on each link to ensure that the travel times will be
different by taking different paths.
"""
means = {}  # nodeid -> seconds
for x in range(lattice_size):
  for y in range(lattice_size):
    n = (x, y)
    n_right = (x+1, y)
    n_up = (x, y+1)
    means[(n, n_right)] = random.random() * 10 + 40
    means[(n, n_up)] = random.random() * 10 + 40

""" The number of observations, minus 2.
A trajectory goes from the (0,0) node up to the (N-1, N-1) node. 
We observe a number of points on this trajectory: one observation on 
the first link, one observation on the last link, and some additional 
observations picked at random among the links that define thr trajectory.
"""
num_observations = 1

""" Generation of some synthetic observations on this road network, and some
travel times between the observations.
"""
# NOTES (gde): 
#
# There will be one observation for each GPS point on a trip.  Here they are 
# being synthetically generated, but normally we would get them by using
# the PointProjector, which needs to be implemented.  The observation includes
# the GPS position, and a set of candidate states.  
#
# The observed_travel_times are the times between recorded GPS points.
# They are based on the number of points - 1. 
(observations, observed_travel_times) = \
  create_trajectory_for_lattice_network(lattice_size, lattice_link_length, 
                                        sigma, obs_sigma, num_observations, 
                                        means)

""" FILTERING CODE.

This is an example of running the path inference filter on the observed data.

Each point and each path is represented as a feature vector, with one feature
for paths and one feature for observations.
The gps noise model is a Gaussian model. The path model is a Gaussian model
on the travel time of the vehicle. This model is different than the one used
in the paper (based on traveled distances). The model in the paper would
make no sense since all links have the same length.
"""

from mm.path_inference.learning_traj import LearningTrajectory
from mm.path_inference.learning_traj_viterbi import TrajectoryViterbi1
from mm.path_inference.learning_traj_smoother import TrajectorySmoother1
import numpy as np

""" Building the feature vectors and the transitions.
"""

path_builder = LatticePathBuilder()


def distance(gps1, gps2):
  """ Distance between two gps points, defined as the Cartesian measure of the
  coordinates.
  """
  return math.sqrt((gps1.lat-gps2.lat)**2 + (gps1.lng-gps2.lng)**2)

# NOTES (gde): 
#
# This is used as a scoring function, where each possible state is given
# a score based on the distance from that state to the recorded GPS
# position.  It is a maximization problem, so the score must be negative. 
# 
# It returns an array with two elements, the first element being the
# pathscore and the second being the pointscore.  Since this is for points, 
# the pathscore is always zero.  
def point_feature_vector(sc):
  """ The feature vector of a point.
  """
  return [[0, -0.5 * distance(sc.gps_pos, s.gps_pos)] for s in sc.states]

# NOTES (gde): 
#
# This is used as a scoring function, where the path is given a score
# based on the square of the difference in travel time calculated from 
# the links versus between the GPS recordings. It is a maximization problem, 
# so the score must be negative. 
# 
# It returns an array with two elements, the first element being the
# pathscore and the second being the pointscore.  Since this is for paths, 
# the pointscore is always zero.  
# 
# Why should we prefer paths that are close to the GPS travel time, rather
# paths with the lowest travel time?  This requires reliable link travel times.
# If the link travel times are free-flow, for example, it may result in 
# more circuitous paths being rewarded in order to come close to the GPS
# travel time.
#
# Either way, this function is specific to the dummy network, so 
# needs to be implemented for the actual network being used.  
def path_feature_vector(path, tt):
  """ The feature vector of a path.
  """
  (s1, p, s2) = path
  # We suppose we make this travel time observation for this vehicle:
  assert p[0] == s1.link_id
  assert p[-1] == s2.link_id
  # This is some "observed" travel time
  # Mean of some idealized travel time distribution.
  m = (1 - s1.offset / lattice_link_length) * means[s1.link_id] \
      + sum([means[link] for link in p[1:-1]]) \
      + s2.offset / lattice_link_length * means[s2.link_id]
  var = (sigma * (1 - s1.offset / lattice_link_length)) ** 2
  var += (sigma ** 2) * len(path[1:-1])
  var += (sigma * s2.offset / lattice_link_length) ** 2
  return [-0.5 * ((m - tt) ** 2) / var, 0]

""" The final list of lists of feature vectors."""
# NOTES (gde): 
#
# point_feature_vector contains [pathscore, pointscore] for each 
# candidate state.  
# 
# path_feature_vector contains [pathscore, pointscore] for each 
# candidate path. 
# 
# In total, the features are an alternating sequence of points and paths, 
# starting and ending with points. 
features = []

""" The transitions between the elements. """
# NOTES (gde): 
#
# The transitions go between each element (point or path) in the trajectory, 
# so there are len(features) - 1 transitions.  
# For points, a transition is (index of candidate state, index of candidate path)
# For paths, a transition is (index of candidate path, index of candidate state)
transitions = []

features.append(point_feature_vector(observations[0]))
# NOTES (gde): for every "space" between two observations
for (sc1, sc2, tt_) in zip(observations[:-1], observations[1:], 
                          observed_travel_times):
                          
  # NOTES (gde): look at every possible pair of states, and get a set
  # of candidate paths between each, along with the transitions.                         
  (trans1, ps, trans2) = path_builder.getPathsBetweenCollections(sc1, sc2)
  
  # NOTES (gde): fill up the data structures used by the algoritms
  transitions.append(trans1)
  paths_features = []
  for path_ in ps:
    paths_features.append(path_feature_vector(path_, tt_))
  features.append(paths_features)
  transitions.append(trans2)
  features.append(point_feature_vector(sc2))

""" We can build a trajectory.
"""
# NOTES (gde): 
#
# a LearningTrajectory is the basic data structure that stores the 
# features (scores candidate states candidate paths), and transitions 
# (indices to look up those states or paths).  
traj = LearningTrajectory(features, transitions)

""" *** Running filters ***
"""

""" Weight vector of the model.

Note: these weights have not be tuned in any way and are here for
demonstration only.
"""
# NOTES (gde): 
#
# theta determines the relative weights of the pathscores versus 
# the pointscores.  What are recommended values?
# Also beware of units!
theta = np.array([1, 1])

""" Viterbi filter (most likely elements) """
# NOTES (gde): 
#
# The viterbi is a specific algorithm that calculates the most likely
# states and most likley paths.  The key output are the indices noted 
# below, which can be used to look up the specific candidate states
# and candidate paths (although those must be stored externally.  
# There is one index for each feature. 
viterbi = TrajectoryViterbi1(traj, theta)
viterbi.computeAssignments()
# The indexes of the most likely elements of the trajectory
# Point indexes and path indexes are interleaved.
most_likely_indexes = viterbi.assignments

""" Alpha-beta smoother (distributions) """
# NOTES (gde): 
#
# The smoother is a different algorithm that gives probabilities instead
# of the most likley.  
smoother = TrajectorySmoother1(traj, theta)
smoother.computeProbabilities()
probabilities = smoother.probabilities
