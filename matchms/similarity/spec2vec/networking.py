#
# Spec2Vec
#
# Copyright 2019 Netherlands eScience Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Import libraries
import numpy as np
import networkx as nx
import community
from networkx.algorithms.connectivity import minimum_st_edge_cut  # , minimum_st_node_cut
from networkx.algorithms.flow import shortest_augmenting_path
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
