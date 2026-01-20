# Databricks notebook source
!pip install graphframes-py networkx pyvis
dbutils.library.restartPython()

# COMMAND ----------

# File location and type
file_location = "/Volumes/workspace/graph/graph/users_age.txt"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_users = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df_users)

# COMMAND ----------

# File location and type
file_location = "/Volumes/workspace/graph/graph/followers2.txt"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = " "

# The applied options are for CSV files. For other file types, these will be ignored.
df_fol = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

df_fol = df_fol.withColumnRenamed('_c0', 'src')
df_fol = df_fol.withColumnRenamed('_c1', 'dst')

display(df_fol)

# COMMAND ----------

from graphframes import GraphFrame

g = GraphFrame(df_users, df_fol)
display(g)
display(g.edges)
display(g.vertices)
#
display(g.inDegrees)

# COMMAND ----------

import networkx as nx

graph_pandas = g.edges.toPandas()
nxg = nx.from_pandas_edgelist(graph_pandas, 'src', 'dst')
nx.draw(nxg, with_labels=True)

# COMMAND ----------

my_edges = [ (r.src, r.dst) for r in df_fol.collect()]

# COMMAND ----------

print(my_edges)

# COMMAND ----------

nxdg = nx.MultiDiGraph()
nxdg.add_edges_from(my_edges)
nx.draw(nxdg, with_labels=True, connectionstyle='arc3, rad=0.1')

# COMMAND ----------



#from graphframes.examples import Graphs
#g = Graphs(spark).friends()  # Get example graph

results = g.shortestPaths(landmarks=[3, 1])
#results = g.shortestPaths(landmarks=['Barack Obama', 'Justin Bieber'])
#results
results.show()
#results.select("id", "distances").show()

# COMMAND ----------

display(g.inDegrees)
display(g.outDegrees)

# COMMAND ----------

from pyvis.network import Network
from IPython.display import HTML

nt = Network(notebook=True, directed=True)
nt.from_nx(nxdg)
nt.prep_notebook()
display(HTML(nt.generate_html()))


# COMMAND ----------

#from graphframes import StorageLevel  

from graphframes.classic.graphframe import StorageLevel
display(g.triangleCount(storage_level=StorageLevel.MEMORY_ONLY))

# COMMAND ----------

#sc.setCheckpointDir('/tmp')
display(g.connectedComponents())

# COMMAND ----------

pr_res = g.pageRank(resetProbability=0.15, tol=0.02)
display(pr_res)

# COMMAND ----------

pr_res.vertices.sort("pagerank", ascending=False).show()

# COMMAND ----------

display(pr_res.vertices.select('pagerank').groupBy().sum())

# COMMAND ----------

from graphframes.examples import Graphs
#g = Graphs(spark).friends()  # Get example graph
#results = g.triangleCount()
#results.select("id", "count").show()
